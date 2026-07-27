"""
server.py — FastAPI Web 服务器
- GET  /                  → 监控大屏 HTML
- GET  /api/status        → 摄像头状态 JSON
- GET  /api/alerts        → 最近告警列表 JSON
- GET  /stream/{cam_id}   → MJPEG 实时视频流
- WS   /ws                → WebSocket 实时告警推送
"""

import asyncio
import json
import queue
import logging
import threading
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger("server")

app = FastAPI(title="超算中心监控预警系统")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

screenshots_dir = Path(__file__).parent / "outputs" / "screenshots"
screenshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=screenshots_dir), name="screenshots")


# 运行时注入（main.py 启动前赋值）
_frame_hub = None
_broadcaster = None
_identity_store = None
_calibrator = None
_pipelines = None

# ROI 文件读-改-写互斥锁，防止并发 POST 请求竞态覆盖配置（startup() 中初始化）
_roi_file_lock: asyncio.Lock | None = None

# ROI 配置约束
_ROI_MAX_VERTICES = 64     # 每个多边形最多顶点数
_ROI_COORD_MIN = 0.0       # 归一化坐标下限
_ROI_COORD_MAX = 1.0       # 归一化坐标上限


def init_server(frame_hub, broadcaster, identity_store, calibrator=None, pipelines=None):
    global _frame_hub, _broadcaster, _identity_store, _calibrator, _pipelines
    _frame_hub = frame_hub
    _broadcaster = broadcaster
    _identity_store = identity_store
    _calibrator = calibrator
    _pipelines = pipelines


# ------------------------------------------------------------------ #
# 主页：监控大屏                                                         #
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ #
# REST API                                                              #
# ------------------------------------------------------------------ #

@app.get("/api/roi")
async def get_roi():
    roi_file = Path(__file__).parent / "config" / "roi.json"
    if not roi_file.exists():
        return JSONResponse({})
    try:
        data = json.loads(roi_file.read_text(encoding="utf-8"))
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


from fastapi import Request

@app.post("/api/roi")
async def save_roi(request: Request):
    try:
        body = await request.json()
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
        name = str(body.get("name", "自定义电子围栏"))[:64]  # 名称最长 64 字符

        # P0-2b: polygon 顶点数上限 + 坐标范围 [0, 1] 校验
        if polygon:
            if len(polygon) > _ROI_MAX_VERTICES:
                return JSONResponse(
                    {"status": "error", "error": f"多边形顶点数超限（最多 {_ROI_MAX_VERTICES} 个）"},
                    status_code=400,
                )
            for pt in polygon:
                if (not isinstance(pt, (list, tuple)) or len(pt) != 2
                        or not (_ROI_COORD_MIN <= float(pt[0]) <= _ROI_COORD_MAX)
                        or not (_ROI_COORD_MIN <= float(pt[1]) <= _ROI_COORD_MAX)):
                    return JSONResponse(
                        {"status": "error", "error": "polygon 坐标必须在 [0, 1] 范围内"},
                        status_code=400,
                    )

        # P0-2c: 读-改-写加锁，防止并发 POST 竞态覆盖（_roi_file_lock 在 startup() 初始化）
        roi_file = Path(__file__).parent / "config" / "roi.json"
        lock = _roi_file_lock or asyncio.Lock()  # startup 未完成时降级为临时锁
        async with lock:
            current = {}
            if roi_file.exists():
                try:
                    current = json.loads(roi_file.read_text(encoding="utf-8"))
                except Exception:
                    current = {}

            if polygon:
                current[camera_id] = [{"name": name, "polygon": polygon}]
            else:
                current.pop(camera_id, None)

            roi_file.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")

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


from fastapi import Query

@app.get("/api/alerts/history")
async def get_alert_history(
    limit: int = Query(default=100, ge=1, le=500),  # P1: 上限 500 防止内存爆炸
    risk_level: str | None = None,
    camera_id: str | None = None,
    alert_type: str | None = None,
):
    log_file = Path(__file__).parent / "outputs" / "alerts.jsonl"
    if not log_file.exists():
        return JSONResponse({"total": 0, "alerts": []})
    
    alerts = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if risk_level and item.get("risk_level") != risk_level:
                        continue
                    if camera_id and item.get("last_camera") != camera_id:
                        continue
                    if alert_type and item.get("alert_type") != alert_type:
                        continue
                    alerts.append(item)
                except Exception:
                    continue
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
    alerts.reverse()  # 逆序，最新在前
    return JSONResponse({
        "total": len(alerts),
        "alerts": alerts[:limit]
    })


@app.get("/api/alerts/export")
async def export_alerts_csv():
    import time as pytime
    from fastapi.responses import Response

    # P1: CSV 字段公式注入防护 —— Excel 会把 =/@/+/- 开头的值当作公式执行
    def _safe_csv(v) -> str:
        s = str(v) if v is not None else ""
        if s and s[0] in ('=', '+', '-', '@', '\t', '\r'):
            s = "'" + s   # 在 Excel 中强制视为文本
        return s

    log_file = Path(__file__).parent / "outputs" / "alerts.jsonl"
    csv_rows = ["Alert ID,Timestamp,Stage,Type,Risk Level,Global ID,Camera,Elapsed Seconds"]
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        ts_str = pytime.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            pytime.localtime(item.get("timestamp", 0))
                        )
                        csv_rows.append(
                            f'"{_safe_csv(item.get("alert_id"))}","{ts_str}",'
                            f'"{_safe_csv(item.get("stage"))}","{_safe_csv(item.get("alert_type","TIMEOUT"))}",'
                            f'"{_safe_csv(item.get("risk_level"))}","{_safe_csv(item.get("global_id"))}",'
                            f'"{_safe_csv(item.get("last_camera"))}","{_safe_csv(item.get("elapsed_seconds"))}"'
                        )
                    except Exception:
                        continue
        except Exception:
            pass

    csv_content = "\n".join(csv_rows)
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
async def get_identity_detail(global_id: str):
    import time as pytime  # P2: 移出循环体，避免重复导入
    if _identity_store is None:
        return JSONResponse({"error": "Identity store unavailable"}, status_code=503)

    rec = _identity_store.get(global_id)
    if rec is None:
        return JSONResponse({"error": "Identity not found"}, status_code=404)

    # 抽取并格式化出现轨迹（按照摄像头变化压缩关键节点）
    raw_apps = list(rec.appearances)
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
        "total_appearances": len(raw_apps),
        "trajectory": trajectory
    })


@app.get("/api/topology")
async def get_topology():
    topo_file = Path(__file__).parent / "config" / "topology.json"
    if not topo_file.exists():
        return JSONResponse({"edges": []})
    try:
        data = json.loads(topo_file.read_text(encoding="utf-8"))
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
        await asyncio.sleep(0.08)   # ~12 fps，CPU 友好


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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with _ws_lock:
        _ws_clients.add(ws)
    # 连接后立即推送最近告警
    if _broadcaster:
        for alert in _broadcaster.recent():
            await ws.send_text(json.dumps(alert, ensure_ascii=False))
    try:
        while True:
            await ws.receive_text()   # 保持连接活跃（client ping）
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            _ws_clients.discard(ws)


async def _broadcast_loop():
    """后台协程：从告警 Queue 读取并广播给所有 WebSocket 客户端"""
    loop = asyncio.get_running_loop()   # Python 3.10+ 推荐，替代已弃用的 get_event_loop()
    while True:
        try:
            alert = await loop.run_in_executor(
                None,
                lambda: (_broadcaster.queue.get(timeout=1) if _broadcaster else None),
            )
            if alert is None:
                continue
            msg = json.dumps(alert, ensure_ascii=False)
            dead = set()
            async with _ws_lock:
                clients = set(_ws_clients)
            for ws in clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            if dead:
                async with _ws_lock:
                    _ws_clients.difference_update(dead)
        except Exception:
            await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup():
    global _ws_lock, _roi_file_lock
    _ws_lock = asyncio.Lock()       # 在事件循环内创建，确保绑定到正确的 loop
    _roi_file_lock = asyncio.Lock() # ROI 文件写入锁，防止并发竞态
    asyncio.create_task(_broadcast_loop())
    logger.info("WebSocket 广播任务已启动")


# ------------------------------------------------------------------ #
# 启动函数（在独立线程中调用）                                            #
# ------------------------------------------------------------------ #

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


def start_server_thread(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    t = threading.Thread(
        target=run_server,
        args=(host, port),
        daemon=True,
        name="web-server",
    )
    t.start()
    logger.info("Web 监控大屏已启动：http://%s:%d", host if host != "0.0.0.0" else "localhost", port)
    return t
