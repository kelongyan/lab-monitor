"""
server.py — FastAPI Web 服务器
- GET  /                  → 监控大屏 HTML
- GET  /api/status        → 摄像头状态 JSON
- GET  /api/alerts        → 最近告警列表 JSON
- GET  /stream/{cam_id}   → MJPEG 实时视频流
- WS   /ws                → WebSocket 实时告警推送
- GET  /api/floorplan     → 平面图底图 + 摄像头点位（路线可视化）
- GET  /api/identities/{gid}/trajectory → 时序轨迹（路线回放）
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
import urllib.parse
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

# 平面图底图（/floorplan/floorplan.jpg）
# 底图是客户设施图纸渲染产物，与 docs/*.xlsx 同级敏感，已 gitignore——
# 新克隆的仓库里 assets/floorplan/ 不存在，此 mount 会在目录缺失时抛错，
# 所以用 try 包住：无底图 = 前端只显示点位与连线，不阻塞服务。
_floorplan_assets = Path(__file__).parent / "assets" / "floorplan"
try:
    _floorplan_assets.mkdir(parents=True, exist_ok=True)
    app.mount("/floorplan", StaticFiles(directory=_floorplan_assets), name="floorplan")
except Exception:
    logger.warning("平面图资源目录不可用: %s", _floorplan_assets)


# 运行时注入（main.py 启动前赋值）
_frame_hub = None
_topology = None
_broadcaster = None
_identity_store = None
_calibrator = None
_pipelines = None
_shutdown_callback = None
_floorplan = None
_mjpeg_sleep = 0.033  # 兜底值；实际由 main.py 经 init_server(mjpeg_fps=...) 覆盖

# F9a：MJPEG 增量推送 —— 帧序号未变化时不重复发字节，但最长 5 秒必须心跳重发一次，
# 否则中间代理/浏览器可能把长时间静默的连接判定为假死。
_MJPEG_HEARTBEAT_SECONDS = 5.0

# F5：写接口跨站保护
# - Host 白名单由实际绑定的 host/port 生成（见 _configure_allowed_hosts），空集合表示未配置（不拦截）
# - 写方法必须带自定义头，浏览器发跨站自定义头需先过预检，没有 CORS 允许头就根本发不出来
_allowed_hosts: set[str] = set()
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_REQUEST_HEADER_NAME = "x-lab-monitor-request"
_REQUEST_HEADER_VALUE = "1"
_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "*", ""})

# ROI 校验常量与文件锁
_ROI_MAX_VERTICES = 64
_ROI_COORD_MIN = 0.0
_ROI_COORD_MAX = 1.0
_roi_file_lock = asyncio.Lock()

# CSV 导出行数上限（P2-13：避免全表读进内存拼字符串导致 OOM）
_ALERT_EXPORT_MAX_ROWS = 50000


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
    try:
        return (
            secrets.compare_digest(supplied_username, expected_username)
            and secrets.compare_digest(supplied_password, expected_password)
        )
    except TypeError:
        # compare_digest 对含非 ASCII 字符的 str 抛 TypeError → 按鉴权失败处理，不要 500
        return False


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


# ------------------------------------------------------------------ #
# F5：写接口最小安全边界（Host 白名单 + 跨站写保护）                        #
# 注意：Starlette 的 add_middleware 是 insert(0)，最后注册的中间件最外层，     #
#      因此本守卫必须定义在 require_basic_auth 之后，才能先于鉴权执行。        #
# ------------------------------------------------------------------ #

def _build_allowed_hosts(host: str, port: int) -> set[str]:
    """由实际绑定参数生成 Host 白名单（防 DNS rebinding），不硬编码端口。"""
    allowed = {
        "127.0.0.1", f"127.0.0.1:{port}",
        "localhost", f"localhost:{port}",
        "[::1]", f"[::1]:{port}",
    }
    name = (host or "").strip().lower()
    if name and name not in _WILDCARD_BIND_HOSTS:
        if ":" in name and not name.startswith("["):
            name = f"[{name}]"        # 裸 IPv6 在 Host 头里必须带方括号
        allowed.add(name)
        allowed.add(f"{name}:{port}")
    for item in os.getenv("LAB_MONITOR_ALLOWED_HOSTS", "").split(","):
        item = item.strip().lower()
        if item:
            allowed.add(item)
    return allowed


def _configure_allowed_hosts(host: str, port: int) -> None:
    """记录允许的 Host 集合；绑定通配地址且未显式配置时关闭该校验（避免误杀局域网访问）。"""
    global _allowed_hosts
    if (host or "").strip().lower() in _WILDCARD_BIND_HOSTS and not os.getenv(
        "LAB_MONITOR_ALLOWED_HOSTS"
    ):
        _allowed_hosts = set()
        logger.warning(
            "绑定通配地址且未配置 LAB_MONITOR_ALLOWED_HOSTS，Host 白名单校验已关闭"
        )
        return
    _allowed_hosts = _build_allowed_hosts(host, port)


def _host_allowed(host_header: str | None) -> bool:
    if not _allowed_hosts:
        return True          # 未配置（单测/直接调用 app）时不拦截
    if not host_header:
        return True          # Host 缺失：部分内部探活不带 Host，放行
    return host_header.strip().lower() in _allowed_hosts


def _origin_allowed(origin: str, host_header: str | None) -> bool:
    """写方法的 Origin 必须与本服务同源。"""
    candidate = origin.strip().lower()
    if not candidate or candidate == "null":
        return False
    parsed = urllib.parse.urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    origins = {parsed.netloc}
    if ":" not in parsed.netloc or parsed.netloc.endswith("]"):
        # Origin 省略了默认端口，补全后再与带端口的 Host 比对
        origins.add(f"{parsed.netloc}:{'80' if parsed.scheme == 'http' else '443'}")
    if host_header and host_header.strip().lower() in origins:
        return True
    return bool(_allowed_hosts and (origins & _allowed_hosts))


@app.middleware("http")
async def guard_request(request: Request, call_next):
    host_header = request.headers.get("host")
    if not _host_allowed(host_header):
        logger.warning("拒绝非白名单 Host: %s", host_header)
        return JSONResponse(
            {"error": "Host 头不在允许列表内"}, status_code=421
        )
    if request.method.upper() in _WRITE_METHODS:
        origin = request.headers.get("origin")
        if origin and not _origin_allowed(origin, host_header):
            logger.warning("拒绝跨站写请求: origin=%s path=%s", origin, request.url.path)
            return JSONResponse(
                {"error": "跨站来源被拒绝（Origin 与本服务不同源）"}, status_code=403
            )
        if request.headers.get(_REQUEST_HEADER_NAME) != _REQUEST_HEADER_VALUE:
            return JSONResponse(
                {
                    "error": "写操作必须携带请求头 X-Lab-Monitor-Request: 1"
                             "（防止跨站简单请求直接触发写操作）"
                },
                status_code=403,
            )
    return await call_next(request)


def init_server(frame_hub, broadcaster, identity_store, calibrator=None, pipelines=None, topology=None, mjpeg_fps: float = 30.0, shutdown_callback=None, host: str | None = None, port: int | None = None, floorplan=None):
    global _frame_hub, _broadcaster, _identity_store, _calibrator, _pipelines, _topology, _mjpeg_sleep, _shutdown_callback, _floorplan
    _frame_hub = frame_hub
    _broadcaster = broadcaster
    _identity_store = identity_store
    _calibrator = calibrator
    _pipelines = pipelines
    _topology = topology
    _mjpeg_sleep = 1.0 / max(1.0, mjpeg_fps)
    _shutdown_callback = shutdown_callback
    # 平面图未显式注入时惰性构造（见 _get_floorplan），单元测试直接 TestClient(app) 也能用
    _floorplan = floorplan
    # host/port 已知时立即生成 Host 白名单；未传则由 run_server/start_server_thread 生成
    if host is not None and port is not None:
        _configure_allowed_hosts(host, port)



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
        # P1-4: 校验的同时构造清洗后的坐标（统一转 float），避免字符串坐标落盘
        #       —— pipeline 侧 int(x * width) 遇到字符串会抛 ValueError 打死摄像头线程
        sanitized: list[list[float]] = []
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
                x = y = 0.0
                try:
                    if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                        raise ValueError("坐标点必须是 [x, y] 形式")
                    x = float(pt[0])
                    y = float(pt[1])
                    valid_point = (
                        _ROI_COORD_MIN <= x <= _ROI_COORD_MAX
                        and _ROI_COORD_MIN <= y <= _ROI_COORD_MAX
                    )
                except (TypeError, ValueError):
                    valid_point = False
                if not valid_point:
                    return JSONResponse(
                        {"status": "error", "error": "polygon 坐标必须在 [0, 1] 范围内"},
                        status_code=400,
                    )
                sanitized.append([x, y])

        # P0-2c: 读-改-写加锁，防止并发 POST 竞态覆盖（_roi_file_lock 在 startup() 初始化）
        roi_file = Path(__file__).parent / "config" / "roi.json"
        lock = _roi_file_lock or asyncio.Lock()  # startup 未完成时降级为临时锁
        async with lock:
            current = await run_in_threadpool(
                _update_roi_file, roi_file, camera_id, sanitized, name
            )

        logger.info("保存 ROI 配置成功: camera_id=%s, polygon_points=%d", camera_id, len(sanitized))

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

    total, alerts = db.query_alert_page(limit=_ALERT_EXPORT_MAX_ROWS)
    truncated = total > len(alerts)
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
    headers = {
        "Content-Disposition": 'attachment; filename="lab_alerts_history.csv"',
        # P2-13：超出上限时截断，并通过响应头告知客户端真实总数
        "X-Export-Truncated": "true" if truncated else "false",
        "X-Export-Rows": str(len(alerts)),
        "X-Export-Total": str(total),
    }
    if truncated:
        logger.warning(
            "告警导出被截断：共 %d 条，仅导出最近 %d 条", total, len(alerts)
        )
    return Response(
        content=csv_content.encode("utf-8-sig"),
        media_type="text/csv",
        headers=headers,
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


# ------------------------------------------------------------------ #
# 平面图与轨迹回放（客户诉求：按拍摄时刻在地图上还原人员路线）                #
# ------------------------------------------------------------------ #
# 坐标系分两套，别混：
#   - config/roi.json   → 相机「画面内」的归一化多边形，用于 INTRUSION 判定
#   - config/camera_map.json → 相机在「楼层平面图」上的归一化点位，用于画路线
# 轨迹点位来自 identity_appearances 表（global_id + camera_id + timestamp），
# 地图坐标不是算出来的，是由 camera_map.json 查表注入的，未标注的点 map_xy 为 null，
# 前端应跳过而不是画到原点。

_FLOORPLAN_LOAD_FAILED = object()  # 已尝试加载但失败，避免每次请求重复 import


def _get_floorplan():
    """惰性构造平面图映射。未注入时直接读 config/camera_map.json。"""
    global _floorplan
    if _floorplan is None:
        try:
            from src.floorplan import build_floorplan
            _floorplan = build_floorplan()
        except Exception:
            logger.exception("平面图加载失败，轨迹接口退化为无坐标模式")
            _floorplan = _FLOORPLAN_LOAD_FAILED
    return None if _floorplan is _FLOORPLAN_LOAD_FAILED else _floorplan


def _empty_floorplan_payload() -> dict:
    return {
        "image": None,
        "image_size": [0, 0],
        "total": 0,
        "mapped": 0,
        "complete": False,
        "cameras": {},
    }


@app.get("/api/floorplan")
async def get_floorplan():
    """平面图底图 + 全部摄像头点位（含未标注的，供标注工具/前端灰度显示）。"""
    fp = _get_floorplan()
    if fp is None:
        return JSONResponse(_empty_floorplan_payload())
    return JSONResponse(fp.payload(camera_ids=_known_camera_ids() or None))


_APPROARANCE_QUERY_LIMIT = 20000


def _collapse_loop_period(seq: list[str]) -> int | None:
    """检测相机序列的最小重复周期（严格周期，快速路径）。

    背景：本地 MP4 素材会自动循环播放（pipeline._run_file），而部分素材只有
    40 秒。一个在镜头里走一趟的人，会被持续记录成上千段「A→B→A→B...」，
    直接画路线会得到「这个人在两点之间来回跑了 545 次」的假象。

   判据用差分而非相位对齐：`seq[i] == seq[i - period]`。
    朴素写法 `seq[i] == seq[i % period]` 看着等价，实际对相位漂移极其敏感——
    真实数据里 A→B→A→B 中途翻成 B→A→B→A 时，后半段会 100% 判不匹配
    （实测 9340 段严格交替的序列因此只有 5% 匹配率）。差分判据只受翻转点
    那一两处影响，剩余部分照常成立，配合 90% 阈值即可吸收抖动。

    周期上限压到 200：一轮真实轨迹不会经过上百个停留段，同时避免
    O(n × period) 在 n 接近 2 万时把请求打爆。
    """
    n = len(seq)
    if n < 4:
        return None
    max_period = min(n // 2, 200)
    for period in range(1, max_period + 1):
        checked = n - period
        if checked <= 0:
            break
        match = sum(1 for i in range(period, n) if seq[i] == seq[i - period])
        if match / checked >= 0.9:
            return period
    return None


def _collapse_by_fingerprint(segments: list[dict]) -> list[int]:
    """按 (camera, 首帧 bbox 中心 20px 量化) 去重，保留首次出现的段。

    兜底路径：真实数据往往不是严格周期（人在素材里出现的时机有抖动、
    偶尔漏检会让 A→B→A→B 变成 A→B→A→A→B），周期检测会失败。
    但循环播放的本质是「同一段像素被反复播放」，所以段的起始画面位置
    几乎完全一致——用位置指纹去重比序列周期更鲁棒。

    20px 量化是权衡：太细（<5px）会因检测抖动漏折叠，太粗（>50px）
    会把真实的不同停留点误折叠。
    """
    seen: set = set()
    keep: list[int] = []
    for i, seg in enumerate(segments):
        key = seg["camera"]
        bb = seg.get("bbox_start") or []
        if len(bb) >= 4:
            try:
                cx = (float(bb[0]) + float(bb[2])) / 2.0
                cy = (float(bb[1]) + float(bb[3])) / 2.0
                key = (seg["camera"], round(cx / 20), round(cy / 20))
            except (TypeError, ValueError):
                pass  # bbox 异常时退化为只按相机去重
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


@app.get("/api/identities/{global_id}/trajectory")
async def get_identity_trajectory(
    global_id: str,
    start: float | None = Query(default=None, description="起始 Unix 秒，缺省不限"),
    end: float | None = Query(default=None, description="结束 Unix 秒，缺省不限"),
    camera: str | None = Query(default=None, description="只看某一路相机"),
    max_points: int = Query(default=2000, ge=1, le=20000, description="返回路径点上限"),
    min_gap_s: float = Query(default=0.0, ge=0.0, le=3600.0,
                             description="相邻路径点最小时间间隔，用于抽稀；0 表示不抽稀"),
    split_gap_s: float = Query(default=30.0, ge=0.0, le=7200.0,
                               description="同相机内时间间隔超过该值则拆成新停留段"),
    collapse_loops: bool = Query(default=True,
                                 description="折叠循环播放产生的重复轨迹（素材仅数十秒时必开）"),
):
    """按时间窗返回某身份的时序轨迹，用于平面图路线回放。

    返回两类数据：
    - segments：按「连续停留」聚合的段，每段一个相机，用于画停留气泡与统计驻留时长
    - path：降采样后的时序点序列，用于播放动画（含每个点的地图坐标）

    单次最多返回 max_points 个路径点；原始点数超过 20000 时 truncated=True，
    前端应提示缩小时间窗（rnd_08 这类长期停留相机单个身份可达 11 万行）。
    """
    from src.db import db  # 延迟导入：模块级会连生产库

    if start is not None and end is not None and start > end:
        return JSONResponse({"error": "start must not be greater than end"}, status_code=400)

    fp = _get_floorplan()

    try:
        total, rows = await run_in_threadpool(
            db.query_trajectory, global_id, start, end, camera, _APPROARANCE_QUERY_LIMIT
        )
    except Exception:
        logger.exception("查询轨迹失败: %s", global_id)
        return JSONResponse({"error": "Trajectory query failed"}, status_code=500)

    # 抽稀：相邻点间隔小于 min_gap_s 的丢弃（保留第一个）
    if min_gap_s > 0 and rows:
        kept = [rows[0]]
        last_t = rows[0]["time"]
        for row in rows[1:]:
            if row["time"] - last_t >= min_gap_s:
                kept.append(row)
                last_t = row["time"]
        rows = kept

    # 分段：连续同相机合并；间隔超过 split_gap_s 视为重新进入视野，拆开
    segments: list[dict] = []
    for row in rows:
        cam, ts = row["camera"], row["time"]
        if (segments and segments[-1]["camera"] == cam
                and ts - segments[-1]["exit"] <= split_gap_s):
            seg = segments[-1]
            seg["exit"] = ts
            seg["frames"] += 1
            seg["bbox_end"] = row["bbox"]
        else:
            segments.append({
                "camera": cam,
                "enter": ts,
                "exit": ts,
                "frames": 1,
                "bbox_start": row["bbox"],
                "bbox_end": row["bbox"],
            })
    for seg in segments:
        seg["duration_s"] = round(seg["exit"] - seg["enter"], 2)
        xy = fp.point(seg["camera"]) if fp else None
        seg["map_xy"] = xy
        meta = fp.meta(seg["camera"]) if fp else None
        seg["desc"] = (meta or {}).get("desc") or ""

    # 折叠循环播放：把循环素材反复播出的同一条轨迹压回一轮。
    # 素材只有几十秒时，不折叠会得到「这个人在两点之间来回跑了 545 次」的假象。
    loop_info = {"detected": False, "method": None, "period_segments": 0,
                 "loops": 1, "raw_segments": len(segments)}
    if collapse_loops and len(segments) >= 4:
        raw = len(segments)
        period = _collapse_loop_period([s["camera"] for s in segments])

        # 覆盖校验：硬截断到第一轮会丢掉「只在后续轮次出现的相机」。
        # 实测 9cd02946 的序列是 rnd_08→rnd_08→reg_08→…，检测到周期 2 后
        # 截断成 [rnd_08, rnd_08]，把只在第 3 段出现的 reg_08 整个丢了。
        # 宁可少折叠也不能丢轨迹，覆盖不全就降级到指纹法。
        if period and raw // period >= 2:
            cams_first = {s["camera"] for s in segments[:period]}
            cams_all = {s["camera"] for s in segments}
            if not cams_all <= cams_first:
                period = None

        if period and raw // period >= 2:
            # 严格周期且覆盖完整：截断到第一轮，时间线连续，播放动画不会瞬移
            loop_info = {"detected": True, "method": "period",
                         "period_segments": period, "loops": raw // period,
                         "raw_segments": raw}
            cut_at = segments[period]["enter"]
            segments = segments[:period]
            rows = [r for r in rows if r["time"] < cut_at]
        else:
            keep = _collapse_by_fingerprint(segments)
            if len(keep) < raw:
                # 非严格周期（或覆盖不全）：按画面位置指纹去重，只保留独特停留段的点
                windows = [(segments[i]["enter"], segments[i]["exit"]) for i in keep]
                segments = [segments[i] for i in keep]
                rows = [r for r in rows
                        if any(a <= r["time"] <= b for a, b in windows)]
                loop_info = {"detected": True, "method": "fingerprint",
                             "period_segments": len(segments),
                             "loops": round(raw / max(1, len(segments)), 1),
                             "raw_segments": raw}

    # 路径点降采样（等距抽样，保证首尾都在）
    returned = len(rows)
    truncated = False
    if rows and len(rows) > max_points:
        step = len(rows) / max_points
        rows = [rows[min(len(rows) - 1, int(i * step))] for i in range(max_points)]
        returned = len(rows)
        truncated = total > len(rows)

    path = []
    for row in rows:
        xy = fp.point(row["camera"]) if fp else None
        path.append({
            "t": row["time"],
            "camera": row["camera"],
            "map_xy": xy,
            "bbox": row["bbox"],
        })

    payload = {
        "global_id": global_id,
        "start": start,
        "end": end,
        "camera_filter": camera,
        "total": total,
        "returned": returned,
        "truncated": truncated,
        "segment_count": len(segments),
        "segments": segments,
        "path": path,
        "loop": loop_info,
        "floorplan": fp.payload(camera_ids=_known_camera_ids() or None) if fp else _empty_floorplan_payload(),
    }
    return JSONResponse(payload)


@app.get("/api/stats")
async def get_stats():
    """Phase 4: 返回通行时间校准统计数据"""
    if _calibrator is None:
        return JSONResponse({"calibration": {}})
    return JSONResponse({"calibration": _calibrator.stats()})


# ------------------------------------------------------------------ #
# WebSocket 状态请求（HTTP 轮询改走 WS 通道，腾出浏览器连接槽位）           #
# ------------------------------------------------------------------ #
# 浏览器同域 HTTP/1.1 并发连接上限 6，MJPEG 长连接占用过多会把 REST 轮询
# 饿死（见 stream_manager.js 注释）。将 /api/status、/api/identities、
# /api/metrics/reid、/api/stats 四类轮询并入 WS 请求-响应后，连接预算变为
# 5 MJPEG + 1 WS = 6，恰好占满且互不饥饿。
# 协议约定：前端发送 {"type": "status"|"identities"|"reid"|"calib"}，
# 服务端应答 {"type": <同名>, "data": {...}}；告警推送保持原格式（无 type 字段）。


def _status_payload() -> dict:
    if _frame_hub is None:
        return {"cameras": []}
    return {"cameras": _frame_hub.get_status()}


def _identities_payload() -> dict:
    if _identity_store is None:
        return {"count": 0, "ids": []}
    ids = _identity_store.all_ids()
    return {"count": len(ids), "ids": ids}


def _reid_metrics_payload() -> dict:
    if _identity_store is None:
        return {
            "gallery_size": 0,
            "total_searches": 0,
            "successful_matches": 0,
            "ratio_blocked_count": 0,
            "match_rate": 0.0,
            "avg_top1_similarity": 0.0,
            "avg_ratio_margin": 0.0,
            "avg_latency_ms": 0.0,
            "avg_feature_quality": 0.0,
        }
    return _identity_store.get_metrics()


def _calib_payload() -> dict:
    if _calibrator is None:
        return {"calibration": {}}
    return {"calibration": _calibrator.stats()}


async def _handle_ws_request(raw: str) -> str | None:
    """处理前端经 WS 发来的数据请求；非请求消息（保持活跃的 ping 等）返回 None"""
    try:
        msg = json.loads(raw)
        req_type = msg.get("type")
    except (json.JSONDecodeError, AttributeError):
        return None
    if req_type == "status":
        payload = _status_payload()
    elif req_type == "identities":
        payload = _identities_payload()
    elif req_type == "reid":
        payload = _reid_metrics_payload()
    elif req_type == "calib":
        payload = _calib_payload()
    else:
        return None
    return json.dumps({"type": req_type, "data": payload}, ensure_ascii=False)


# ------------------------------------------------------------------ #
# MJPEG 视频流                                                          #
# ------------------------------------------------------------------ #

async def _mjpeg_generator(cam_id: str):
    """异步生成器：持续推送 MJPEG 帧

    F9a：JPEG 编码（copy + resize + imencode，实测 ~8.7ms/次）通过 run_in_threadpool
    卸出 uvicorn 唯一事件循环；并按 FrameHub.generation 做增量推送 —— 帧未更新时
    只 await sleep 让出控制权，不重复发送同样的字节（最长 5 秒仍会心跳重发一次）。
    """
    boundary = b"--frame\r\n"
    offline_frame = await run_in_threadpool(_make_offline_frame, cam_id)
    last_generation = None
    last_sent = 0.0

    try:
        while True:
            generation = _frame_hub.get_generation(cam_id) if _frame_hub else 0
            now = time.monotonic()
            if generation != last_generation or (now - last_sent) >= _MJPEG_HEARTBEAT_SECONDS:
                jpeg = None
                if _frame_hub is not None:
                    jpeg, generation = await run_in_threadpool(
                        _frame_hub.get_jpeg_with_generation, cam_id
                    )
                data = jpeg if jpeg else offline_frame
                last_generation = generation
                last_sent = time.monotonic()
                yield (
                    boundary
                    + b"Content-Type: image/jpeg\r\n\r\n"
                    + data
                    + b"\r\n"
                )
            await asyncio.sleep(_mjpeg_sleep)
    except (asyncio.CancelledError, GeneratorExit):
        # 客户端断开：停止再向线程池提交编码任务，交由上层正常收尾
        logger.debug("MJPEG 客户端断开: %s", cam_id)
        raise


def _known_camera_ids() -> set[str]:
    """已配置的摄像头集合（pipelines ∪ FrameHub 注册表），用于流接口白名单。"""
    ids: set[str] = set()
    if _pipelines:
        ids.update(p.camera_id for p in _pipelines if hasattr(p, "camera_id"))
    if _frame_hub is not None:
        try:
            ids.update(_frame_hub.camera_ids())
        except Exception:
            logger.debug("读取 FrameHub 摄像头列表失败", exc_info=True)
    return ids


@app.get("/stream/{cam_id}")
async def video_stream(cam_id: str):
    # 未知 id 直接 404：白名单为空时同样拒绝（fail-closed），避免刷出无限路 OFFLINE 流
    if cam_id not in _known_camera_ids():
        return JSONResponse(
            {"error": f"未知摄像头 ID: {cam_id}"}, status_code=404
        )
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
            # 保持连接活跃（client ping）的同时支持状态数据请求-响应
            raw = await ws.receive_text()
            reply = await _handle_ws_request(raw)
            if reply is not None:
                if not await _send_ws_message(ws, reply):
                    break
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
    _configure_allowed_hosts(host, port)
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
    _configure_allowed_hosts(host, port)   # 线程启动前先就位，避免健康检查撞上未配置窗口
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
