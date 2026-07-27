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


# 运行时注入（main.py 启动前赋值）
_frame_hub = None
_broadcaster = None
_identity_store = None
_calibrator = None


def init_server(frame_hub, broadcaster, identity_store, calibrator=None):
    global _frame_hub, _broadcaster, _identity_store, _calibrator
    _frame_hub = frame_hub
    _broadcaster = broadcaster
    _identity_store = identity_store
    _calibrator = calibrator


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

@app.get("/api/status")
async def get_status():
    if _frame_hub is None:
        return JSONResponse({"cameras": []})
    cameras = _frame_hub.get_status()
    # 附加摄像头 id 列表（用于前端渲染格子）
    return JSONResponse({"cameras": cameras})


@app.get("/api/alerts")
async def get_alerts():
    if _broadcaster is None:
        return JSONResponse({"alerts": []})
    return JSONResponse({"alerts": _broadcaster.recent()})


@app.get("/api/identities")
async def get_identities():
    if _identity_store is None:
        return JSONResponse({"count": 0, "ids": []})
    ids = _identity_store.all_ids()
    return JSONResponse({"count": len(ids), "ids": ids})


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
_ws_lock = asyncio.Lock()


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
    loop = asyncio.get_event_loop()
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
