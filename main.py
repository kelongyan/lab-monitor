"""
main.py — 入口：加载配置，多线程启动各路视频流水线 + Web 服务器
Phase 4 更新：集成 TransitCalibrator
"""

import json
import time
import logging
import threading
from pathlib import Path

from src.detector import PersonDetector
from src.reid import build_reid_extractor
from src.identity_store import IdentityStore
from src.topology import CameraTopology
from src.alerter import AlertManager, AlertBroadcaster
from src.frame_hub import FrameHub
from src.notifier import build_notifier
from src.calibrator import TransitCalibrator
from src.pipeline import CameraPipeline
import server as web_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

CONFIG_DIR     = Path("config")
SOURCES_CFG    = CONFIG_DIR / "sources.json"
TOPO_CFG       = CONFIG_DIR / "topology.json"
NOTIFY_CFG     = CONFIG_DIR / "notify.json"
OUTPUT_DIR     = Path("outputs")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
ALERT_LOG      = OUTPUT_DIR / "alerts.jsonl"


def load_sources() -> dict[str, str]:
    with open(SOURCES_CFG, encoding="utf-8") as f:
        return json.load(f)


def alert_ticker(alert_manager: AlertManager, interval: float = 2.0) -> None:
    """后台线程：每隔 interval 秒检查一次超时告警"""
    while True:
        alert_manager.tick()
        time.sleep(interval)


def main(display: bool = False, web: bool = True, web_port: int = 8000) -> None:
    sources = load_sources()

    # 检查视频文件是否存在（仅本地文件，RTSP 跳过检查）
    missing = [
        cam for cam, path in sources.items()
        if not path.startswith("rtsp://") and not Path(path).exists()
    ]
    if missing:
        logger.warning(
            "以下摄像头视频文件不存在，已跳过：%s\n"
            "请将视频文件放入 videos/ 目录，或修改 config/sources.json",
            missing,
        )
        sources = {k: v for k, v in sources.items() if k not in missing}

    if not sources:
        logger.error("没有可用的视频源，请先添加视频文件到 videos/ 目录")
        return

    # ---- 初始化共享组件 ----
    logger.info("加载模型中（首次运行会自动下载权重）...")
    detector       = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4)
    reid_extractor = build_reid_extractor(device="cpu")
    identity_store = IdentityStore()
    topology       = CameraTopology(TOPO_CFG)
    frame_hub      = FrameHub()
    broadcaster    = AlertBroadcaster()
    notifier       = build_notifier(NOTIFY_CFG)
    calibrator     = TransitCalibrator(OUTPUT_DIR / "transit_stats.json")
    alert_manager  = AlertManager(
        alert_log=ALERT_LOG,
        notifier=notifier,
        broadcaster=broadcaster,
    )

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 启动 Web 服务器 ----
    # ---- 启动各路流水线 ----
    pipelines: list[CameraPipeline] = []
    for cam_id, source in sources.items():
        p = CameraPipeline(
            camera_id=cam_id,
            source=source,
            detector=detector,
            reid_extractor=reid_extractor,
            identity_store=identity_store,
            topology=topology,
            alert_manager=alert_manager,
            screenshot_dir=SCREENSHOT_DIR,
            frame_hub=frame_hub,
            calibrator=calibrator,
            display=display,
        )
        pipelines.append(p)
        p.start()
        logger.info("启动摄像头 %s → %s", cam_id, source)

    # ---- 启动 Web 服务器 ----
    if web:
        web_server.init_server(frame_hub, broadcaster, identity_store, calibrator, pipelines=pipelines, topology=topology)
        web_server.start_server_thread(port=web_port)
        time.sleep(0.5)

    # ---- 启动预警后台检查线程 ----
    ticker = threading.Thread(
        target=alert_ticker,
        args=(alert_manager,),
        daemon=True,
        name="alert-ticker",
    )
    ticker.start()

    logger.info("所有流水线已启动（%d 路）", len(pipelines))
    if web:
        logger.info("监控大屏：http://localhost:%d", web_port)

    try:
        for p in pipelines:
            p.join()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
        for p in pipelines:
            p.stop()

    logger.info("系统已停止，告警日志：%s", ALERT_LOG)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="超算中心监控预警系统")
    parser.add_argument("--display",   action="store_true", help="显示本地画面窗口")
    parser.add_argument("--no-web",    action="store_true", help="不启动 Web 服务器")
    parser.add_argument("--port",      type=int, default=8000, help="Web 端口（默认 8000）")
    args = parser.parse_args()
    main(display=args.display, web=not args.no_web, web_port=args.port)
