"""
server.py — FastAPI Web 服务器
- GET  /                  → 监控大屏 HTML
- GET  /api/status        → 摄像头状态 JSON
- GET  /api/alerts        → 最近告警列表 JSON
- GET  /stream/{cam_id}   → MJPEG 实时视频流
- WS   /ws                → WebSocket 实时告警推送
"""

import asyncio
import base64
import json
import queue
import logging
import os
import secrets
import threading
import time
import csv
import io
import socket
import urllib.request
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import uvicorn

logger = logging.getLogger("server")

app = FastAPI(title="超算中心监控预警系统")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

screenshots_dir = Path(__file__).parent / "outputs" / "screenshots"
screenshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=screenshots_dir), name="screenshots")


# 运行时注入（main.py 启动前赋值）
_frame_hub = None
_topology = None
_broadcaster = None
_identity_store = None
_calibrator = None
_pipelines = None
_shutdown_callback = None
_mjpeg_sleep = 0.033  # 默认 ~30fps（GPU），CPU 模式由 init_server 覆盖为 0.067（15fps）

# ROI 校验常量与文件锁
_ROI_MAX_VERTICES = 64
_ROI_COORD_MIN = 0.0
_ROI_COORD_MAX = 1.0
_roi_file_lock = asyncio.Lock()


def _read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_roi_file(
    path: Path,
    camera_id: str,
    polygon: list,
    name: str,
) -> dict:
    current = {}
    if path.exists():
        try:
            current = _read_json_file(path)
        except (OSError, json.JSONDecodeError):
            current = {}
    if polygon:
        current[camera_id] = [{"name": name, "polygon": polygon}]
    else:
        current.pop(camera_id, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temp.open("w", encoding="utf-8") as output:
            json.dump(current, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return current


def _configured_basic_auth() -> tuple[str, str] | None:
    username = os.getenv("LAB_MONITOR_USERNAME", "")
    password = os.getenv("LAB_MONITOR_PASSWORD", "")
    if not username and not password:
        return None
    if not username or not password:
        raise RuntimeError(
            "LAB_MONITOR_USERNAME 和 LAB_MONITOR_PASSWORD 必须同时配置"
        )
    return username, password


def _authorization_valid(authorization: str | None) -> bool:
    credentials = _configured_basic_auth()
    if credentials is None:
        return True
    if not authorization or not authorization.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(
            authorization.removeprefix("Basic "), validate=True
        ).decode("utf-8")
        supplied_username, supplied_password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return False
    expected_username, expected_password = credentials
    return (
        secrets.compare_digest(supplied_username, expected_username)
        and secrets.compare_digest(supplied_password, expected_password)
    )


def _is_loopback_host(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def _ensure_secure_bind(host: str) -> None:
    if not _is_loopback_host(host) and _configured_basic_auth() is None:
        raise RuntimeError(
            "绑定非本机地址前必须配置 LAB_MONITOR_USERNAME 和 LAB_MONITOR_PASSWORD"
        )


@app.middleware("http")
async def require_basic_auth(request: Request, call_next):
    try:
        authorized = _authorization_valid(request.headers.get("authorization"))
    except RuntimeError as error:
        logger.error("认证配置无效: %s", error)
        return JSONResponse({"error": str(error)}, status_code=500)
    if authorized:
        return await call_next(request)
    return Response(
        content="Authentication required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Lab-Monitor", charset="UTF-8"'},
    )


def init_server(frame_hub, broadcaster, identity_store, calibrator=None, pipelines=None, topology=None, mjpeg_fps: float = 30.0, shutdown_callback=None):
    global _frame_hub, _broadcaster, _identity_store, _calibrator, _pipelines, _topology, _mjpeg_sleep, _shutdown_callback
    _frame_hub = frame_hub
    _broadcaster = broadcaster
    _identity_store = identity_store
    _calibrator = calibrator
    _pipelines = pipelines
    _topology = topology
    _mjpeg_sleep = 1.0 / max(1.0, mjpeg_fps)
    _shutdown_callback = shutdown_callback



# ------------------------------------------------------------------ #
# 主页：监控大屏                                                         #
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    html = await run_in_threadpool(html_path.read_text, encoding="utf-8")
    return HTMLResponse(html)


# ------------------------------------------------------------------ #
# REST API                                                              #
# ------------------------------------------------------------------ #

@app.get("/api/roi")
async def get_roi():
    roi_file = Path(__file__).parent / "config" / "roi.json"
    if not roi_file.exists():
        return JSONResponse({})
    try:
        data = await run_in_threadpool(_read_json_file, roi_file)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/roi")
async def save_roi(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(
                {"status": "error", "error": "请求体必须是 JSON 对象"},
                status_code=400,
            )
        camera_id = body.get("camera_id")
        if not camera_id:
            return JSONResponse({"status": "error", "error": "缺少 camera_id 参数"}, status_code=400)

        # P0-2a: camera_id 白名单校验 —— 只允许已配置的摄像头
        valid_cam_ids: set[str] = set()
        if _pipelines:
            valid_cam_ids = {p.camera_id for p in _pipelines if hasattr(p, "camera_id")}
        if valid_cam_ids and camera_id not in valid_cam_ids:
            return JSONResponse(
                {"status": "error", "error": f"未知摄像头 ID: {camera_id}"},
                status_code=400,
            )

        polygon = body.get("polygon", [])
        if not isinstance(polygon, list):
            return JSONResponse(
                {"status": "error", "error": "polygon 必须是点数组"},
                status_code=400,
            )
        name = str(body.get("name", "自定义电子围栏"))[:64]  # 名称最长 64 字符

        # P0-2b: polygon 顶点数上限 + 坐标范围 [0, 1] 校验
        if polygon:
            if len(polygon) < 3:
                return JSONResponse(
                    {"status": "error", "error": "多边形至少需要 3 个顶点"},
                    status_code=400,
                )
            if len(polygon) > _ROI_MAX_VERTICES:
                return JSONResponse(
                    {"status": "error", "error": f"多边形顶点数超限（最多 {_ROI_MAX_VERTICES} 个）"},
                    status_code=400,
                )
            for pt in polygon:
                try:
                    valid_point = (
                        isinstance(pt, (list, tuple))
                        and len(pt) == 2
                        and _ROI_COORD_MIN <= float(pt[0]) <= _ROI_COORD_MAX
                        and _ROI_COORD_MIN <= float(pt[1]) <= _ROI_COORD_MAX
                    )
                except (TypeError, ValueError):
                    valid_point = False
                if not valid_point:
                    return JSONResponse(
                        {"status": "error", "error": "polygon 坐标必须在 [0, 1] 范围内"},
                        status_code=400,
                    )

        # P0-2c: 读-改-写加锁，防止并发 POST 竞态覆盖（_roi_file_lock 在 startup() 初始化）
        roi_file = Path(__file__).parent / "config" / "roi.json"
        lock = _roi_file_lock or asyncio.Lock()  # startup 未完成时降级为临时锁
        async with lock:
            current = await run_in_threadpool(
                _update_roi_file, roi_file, camera_id, polygon, name
            )

        logger.info("保存 ROI 配置成功: camera_id=%s, polygon_points=%d", camera_id, len(polygon))

        # 动态通知运行中的 CameraPipelines
        if _pipelines:
            for p in _pipelines:
                if hasattr(p, "reload_rois"):
                    p.reload_rois()

        return JSONResponse({"status": "success", "config": current})
    except Exception as e:
        logger.error("保存 ROI 失败: %s", e, exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/api/topology")
async def get_topology():
    if _topology:
        return JSONResponse(_topology.to_dict())
    topology_file = Path(__file__).parent / "config" / "topology.json"
    if not topology_file.exists():
        return JSONResponse({})
    try:
        data = await run_in_threadpool(_read_json_file, topology_file)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/topology")
async def save_topology(request: Request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            {"status": "error", "error": "请求体必须是有效的 JSON"},
            status_code=400,
        )

    if _topology is None:
        return JSONResponse(
            {"status": "error", "error": "拓扑服务尚未初始化"},
            status_code=503,
        )

    try:
        normalized = await run_in_threadpool(_topology.update_config, body)
        logger.info("在线更新拓扑配置成功: %d 个节点通道", len(body))
        return JSONResponse({"status": "success", "topology": normalized})
    except ValueError as e:
        logger.warning("拒绝非法拓扑配置: %s", e)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=400)
    except Exception as e:
        logger.error("保存拓扑配置失败: %s", e, exc_info=True)
        return JSONResponse(
            {"status": "error", "error": "拓扑配置保存失败"},
            status_code=500,
        )


@app.get("/api/status")
async def get_status():
    if _frame_hub is None:
        return JSONResponse({"cameras": []})
    cameras = _frame_hub.get_status()
    # 附加摄像头 id 列表（用于前端渲染格子）
    return JSONResponse({"cameras": cameras})


@app.get("/api/metrics/reid")
async def get_reid_metrics():
    if _identity_store is None:
        return JSONResponse({
            "gallery_size": 0,
            "total_searches": 0,
            "successful_matches": 0,
            "ratio_blocked_count": 0,
            "match_rate": 0.0,
            "avg_top1_similarity": 0.0,
            "avg_ratio_margin": 0.0,
            "avg_latency_ms": 0.0,
            "avg_feature_quality": 0.0,
        })
    return JSONResponse(_identity_store.get_metrics())



@app.get("/api/alerts")
async def get_alerts():
    if _broadcaster is None:
        return JSONResponse({"alerts": []})
    return JSONResponse({"alerts": _broadcaster.recent()})


@app.get("/healthz")
async def health_check():
    """方向四：运维健康检查端点"""
    cams_online = 0
    if _frame_hub:
        cams_online = len([c for c in _frame_hub.get_status() if c.get("is_online")])
    return JSONResponse({
        "status": "ok",
        "service": "Lab-Monitor",
        "cameras_online": cams_online,
        "timestamp": time.time(),
    })


@app.post("/api/admin/shutdown")
async def shutdown_service(request: Request):
    client_host = request.client.host if request.client else ""
    if not _is_loopback_host(client_host):
        return JSONResponse(
            {"error": "安全停止接口仅允许本机调用"}, status_code=403
        )
    if _shutdown_callback is None:
        return JSONResponse({"error": "安全停止尚未初始化"}, status_code=503)
    _shutdown_callback()
    return JSONResponse({"status": "stopping"}, status_code=202)


@app.get("/api/system/metrics")
async def get_system_metrics():
    """方向四：工程可观测性与运维指标接口"""
    import os, sys
    from src.db import db

    db_stats = db.get_stats()
    cams = _frame_hub.get_status() if _frame_hub else []

    # 尝试获取 Python 进程内存
    mem_mb = 0.0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = round(process.memory_info().rss / 1024 / 1024, 2)
    except Exception:
        pass

    return JSONResponse({
        "cameras_total": len(cams),
        "cameras_online": len([c for c in cams if c.get("is_online")]),
        "process_memory_mb": mem_mb,
        "reid_metrics": _identity_store.get_metrics() if _identity_store else {},
        "database": db_stats,
    })


@app.get("/api/alerts/history")
async def get_alert_history(
    limit: int = Query(default=100, ge=1, le=500),  # P1: 上限 500 防止内存爆炸
    offset: int = Query(default=0, ge=0),
    risk_level: str | None = None,
    camera_id: str | None = None,
    global_id: str | None = None,
):
    """从 SQLite 唯一权威数据源分页查询历史告警。"""
    from src.db import db
    try:
        total, alerts = await run_in_threadpool(
            db.query_alert_page,
            limit,
            offset,
            camera_id,
            global_id,
            risk_level,
        )
        summary = await run_in_threadpool(db.get_alert_summary)
        return JSONResponse({
            "total": total,
            "limit": limit,
            "offset": offset,
            "summary": summary,
            "alerts": alerts,
        })
    except Exception:
        logger.exception("查询历史告警失败")
        return JSONResponse({"error": "历史告警查询失败"}, status_code=500)


@app.get("/api/alerts/export")
def export_alerts_csv():
    import time as pytime
    from src.db import db

    # P1: CSV 字段公式注入防护 —— Excel 会把 =/@/+/- 开头的值当作公式执行
    def _safe_csv(v) -> str:
        s = str(v) if v is not None else ""
        if s and s[0] in ('=', '+', '-', '@', '\t', '\r'):
            s = "'" + s   # 在 Excel 中强制视为文本
        return s

    total, alerts = db.query_alert_page(limit=2_147_483_647)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow([
        "Alert ID", "Timestamp", "Stage", "Type", "Risk Level",
        "Global ID", "Camera", "Elapsed Seconds",
    ])
    for item in alerts:
        ts_str = pytime.strftime(
            "%Y-%m-%d %H:%M:%S",
            pytime.localtime(item.get("timestamp", 0)),
        )
        writer.writerow([
            _safe_csv(item.get("alert_id")),
            ts_str,
            _safe_csv(item.get("stage")),
            _safe_csv(item.get("alert_type", "TIMEOUT")),
            _safe_csv(item.get("risk_level")),
            _safe_csv(item.get("global_id")),
            _safe_csv(item.get("last_camera")),
            _safe_csv(item.get("elapsed_seconds")),
        ])
    csv_content = output.getvalue()
    return Response(
        content=csv_content.encode("utf-8-sig"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="lab_alerts_history.csv"'}
    )


@app.get("/api/identities")
async def get_identities():
    if _identity_store is None:
        return JSONResponse({"count": 0, "ids": []})
    ids = _identity_store.all_ids()
    return JSONResponse({"count": len(ids), "ids": ids})


@app.get("/api/identities/{global_id}")
async def get_identity_detail(
    global_id: str,
    limit: int = Query(default=1000, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
):
    import time as pytime  # P2: 移出循环体，避免重复导入
    if _identity_store is None:
        return JSONResponse({"error": "Identity store unavailable"}, status_code=503)

    rec = _identity_store.get(global_id)
    if rec is None:
        return JSONResponse({"error": "Identity not found"}, status_code=404)

    # 抽取并格式化出现轨迹（按照摄像头变化压缩关键节点）
    raw_apps = list(rec.appearances)
    total_appearances = rec.total_appearances
    try:
        from src.db import db
        db_total, persisted_apps = await run_in_threadpool(
            db.query_identity_appearances,
            global_id,
            limit,
            offset,
        )
        if db_total:
            raw_apps = persisted_apps
            total_appearances = db_total
    except Exception:
        logger.exception("读取身份完整轨迹失败: %s", global_id)
    trajectory = []
    last_cam = None

    for app_item in raw_apps:
        cam = app_item.get("camera")
        ts = app_item.get("time", 0.0)
        time_str = pytime.strftime("%H:%M:%S", pytime.localtime(ts)) if ts else "未知"
        
        # 仅在跨相机切换或首条记录时保留主要轨迹点
        if cam != last_cam or not trajectory:
            trajectory.append({
                "camera": cam,
                "timestamp": ts,
                "time_str": time_str,
                "bbox": app_item.get("bbox")
            })
            last_cam = cam
        else:
            # 更新同一相机的最后活跃时间
            trajectory[-1]["end_timestamp"] = ts
            trajectory[-1]["end_time_str"] = time_str

    return JSONResponse({
        "global_id": rec.global_id,
        "last_camera": rec.last_camera,
        "last_seen": rec.last_seen,
        "total_appearances": total_appearances,
        "limit": limit,
        "offset": offset,
        "trajectory": trajectory
    })


@app.get("/api/stats")
async def get_stats():
    """Phase 4: 返回通行时间校准统计数据"""
    if _calibrator is None:
        return JSONResponse({"calibration": {}})
    return JSONResponse({"calibration": _calibrator.stats()})


# ------------------------------------------------------------------ #
# MJPEG 视频流                                                          #
# ------------------------------------------------------------------ #

async def _mjpeg_generator(cam_id: str):
    """异步生成器：持续推送 MJPEG 帧"""
    boundary = b"--frame\r\n"
    offline_frame = _make_offline_frame(cam_id)

    while True:
        jpeg = _frame_hub.get_jpeg(cam_id) if _frame_hub else None
        data = jpeg if jpeg else offline_frame
        yield (
            boundary
            + b"Content-Type: image/jpeg\r\n\r\n"
            + data
            + b"\r\n"
        )
        await asyncio.sleep(_mjpeg_sleep)


@app.get("/stream/{cam_id}")
async def video_stream(cam_id: str):
    return StreamingResponse(
        _mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _make_offline_frame(cam_id: str) -> bytes:
    """生成一张"摄像头离线"占位图"""
    import numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, f"{cam_id}  OFFLINE", (160, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes()


# ------------------------------------------------------------------ #
# WebSocket：实时告警推送                                                #
# ------------------------------------------------------------------ #

_ws_clients: set[WebSocket] = set()
# 延迟初始化：在事件循环启动后的 startup() 中创建，避免模块级 asyncio.Lock() 绑定错误循环
_ws_lock: asyncio.Lock | None = None
_WS_SEND_TIMEOUT_SECONDS = 2.0


async def _send_ws_message(ws: WebSocket, message: str) -> bool:
    try:
        await asyncio.wait_for(
            ws.send_text(message), timeout=_WS_SEND_TIMEOUT_SECONDS
        )
        return True
    except Exception:
        try:
            await ws.close(code=1011, reason="WebSocket client too slow")
        except Exception:
            pass
        return False


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    try:
        authorized = _authorization_valid(ws.headers.get("authorization"))
    except RuntimeError:
        authorized = False
    if not authorized:
        await ws.close(code=1008, reason="Authentication required")
        return
    await ws.accept()
    async with _ws_lock:
        _ws_clients.add(ws)
    # 连接后立即推送最近告警
    if _broadcaster:
        since = ws.query_params.get("since")
        for alert in _broadcaster.recent_after(since):
            if not await _send_ws_message(
                ws, json.dumps(alert, ensure_ascii=False)
            ):
                async with _ws_lock:
                    _ws_clients.discard(ws)
                return
    try:
        while True:
            await ws.receive_text()   # 保持连接活跃（client ping）
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            _ws_clients.discard(ws)


async def _broadcast_loop():
    """后台协程：从告警 Queue 读取并广播给所有 WebSocket 客户端
    GPU 服务器优化：优先使用 asyncio.Queue（零延迟），回退到 threading.queue（兼容）
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            # 优先路径：asyncio.Queue.get() — 真正的 async 等待，告警到达即触发，零额外延迟
            if _broadcaster and _broadcaster._async_queue is not None:
                alert = await _broadcaster._async_queue.get()
            else:
                # 回退路径：事件循环注入前的兼容模式（启动瞬间短暂使用）
                alert = await loop.run_in_executor(
                    None,
                    lambda: (_broadcaster.queue.get(timeout=1) if _broadcaster else None),
                )
            if alert is None:
                continue
            msg = json.dumps(alert, ensure_ascii=False)
            async with _ws_lock:
                clients = list(_ws_clients)
            results = await asyncio.gather(
                *(_send_ws_message(ws, msg) for ws in clients),
                return_exceptions=False,
            )
            dead = {
                ws for ws, delivered in zip(clients, results) if not delivered
            }
            if dead:
                async with _ws_lock:
                    _ws_clients.difference_update(dead)
        except Exception:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup():
    global _ws_lock, _roi_file_lock
    _ws_lock = asyncio.Lock()
    _roi_file_lock = asyncio.Lock()
    # GPU 服务器优化：注入 event loop 到 broadcaster，激活 asyncio.Queue 零延迟模式
    if _broadcaster:
        _broadcaster.set_event_loop(asyncio.get_running_loop())
    asyncio.create_task(_broadcast_loop())
    logger.info("WebSocket 广播任务已启动（asyncio.Queue 零延迟模式）")


# ------------------------------------------------------------------ #
# 启动函数（在独立线程中调用）                                            #
# ------------------------------------------------------------------ #

def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    _ensure_secure_bind(host)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _ensure_port_available(host: str, port: int) -> None:
    bind_host = host if host not in {"localhost"} else "127.0.0.1"
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((bind_host, port))


def _wait_for_health(
    thread: threading.Thread,
    host: str,
    port: int,
    timeout: float,
) -> None:
    connect_host = host if _is_loopback_host(host) else "127.0.0.1"
    if connect_host == "::1":
        url = f"http://[::1]:{port}/healthz"
    else:
        url = f"http://{connect_host}:{port}/healthz"
    headers = {}
    credentials = _configured_basic_auth()
    if credentials:
        token = base64.b64encode(
            f"{credentials[0]}:{credentials[1]}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("Web 服务线程在健康检查前退出")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("service") == "Lab-Monitor":
                return
        except Exception as error:
            last_error = error
        time.sleep(0.1)
    raise RuntimeError(f"Web 服务健康检查超时: {last_error}")


def start_server_thread(host: str = "127.0.0.1", port: int = 8000, startup_timeout: float = 15.0) -> threading.Thread:
    _ensure_secure_bind(host)
    _ensure_port_available(host, port)
    t = threading.Thread(
        target=run_server,
        args=(host, port),
        daemon=True,
        name="web-server",
    )
    t.start()
    _wait_for_health(t, host, port, startup_timeout)
    logger.info("Web 监控大屏已启动：http://%s:%d", host, port)
    return t
