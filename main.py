"""
main.py — 入口：加载配置，多线程启动各路视频流水线 + Web 服务器
Phase 4 更新：集成 TransitCalibrator
"""

import json
import time
import logging
import threading
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 必须在导入 cv2（由 src.* 间接导入）之前设置：屏蔽 FFmpeg C 层 stderr
# （损坏视频源会持续刷 "PPS out of range" / "no start code" 等噪声日志）
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2  # 必须在上面的 OPENCV_FFMPEG_LOGLEVEL 之后导入

from src.detector import PersonDetector
from src.reid import build_reid_extractor
from src.model_pool import PooledDetector, PooledReIDExtractor, resolve_pool_size
from src.identity_store import IdentityStore
from src.topology import CameraTopology, TopologyValidationError
from src.alerter import AlertManager, AlertBroadcaster
from src.frame_hub import FrameHub
from src.notifier import build_notifier
from src.calibrator import TransitCalibrator
from src.pipeline import CameraPipeline, redact_source
from src.db import db
import server as web_server

CONFIG_DIR     = Path("config")
SOURCES_CFG    = CONFIG_DIR / "sources.json"
TOPO_CFG       = CONFIG_DIR / "topology.json"
NOTIFY_CFG     = CONFIG_DIR / "notify.json"
OUTPUT_DIR     = Path("outputs")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
ALERT_LOG      = OUTPUT_DIR / "alerts.jsonl"
SERVER_LOG     = OUTPUT_DIR / "server.log"
PID_FILE       = OUTPUT_DIR / "server.pid"

# 日志：控制台（前台运行时可见）+ 轮转文件（单个 64MB，保留 5 份），避免 server.log 无限膨胀
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)
_file_handler = RotatingFileHandler(
    SERVER_LOG,
    maxBytes=64 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger("main")


def load_sources() -> dict[str, str]:
    with open(SOURCES_CFG, encoding="utf-8") as f:
        return json.load(f)


def alert_ticker(
    alert_manager: AlertManager,
    stop_event: threading.Event,
    interval: float = 0.5,
) -> None:
    """后台线程：每隔 interval 秒检查一次超时告警（GPU 服务器模式：0.5s）

    它是唯一的超时检测线程，任何异常只记录不退出循环，
    否则 MISSING_PERSON / SCENE_EXIT 会静默失效。
    """
    failures = 0
    while not stop_event.is_set():
        try:
            alert_manager.tick()
        except Exception:
            failures += 1
            logger.exception("告警轮询异常（累计 %d 次），线程继续运行", failures)
        stop_event.wait(interval)


def clear_pid_file() -> None:
    """退出时清理 PID 文件；仅当内容确为本进程 PID，避免误删新实例写入的值"""
    try:
        if PID_FILE.exists() and PID_FILE.read_text(encoding="ascii").strip() == str(os.getpid()):
            PID_FILE.unlink()
    except (OSError, ValueError):
        pass


def flush_identity_features(identity_store) -> None:
    """
    关停前补写被节流的 ReID 主特征与 feature_bank。

    IdentityStore 为消除写放大，对特征列做了「累计 N 次更新或间隔 M 秒」的落盘节流，
    不显式 flush 就会在每次停服时静默丢掉最近一段的滑动平均结果（轨迹不受影响，
    它每次都写增量行）。异常一律吞掉：关停路径不能因为落盘失败而抛出。
    """
    try:
        flushed = identity_store.flush()
    except Exception:
        logger.exception("关停前补写 ReID 特征失败")
        return
    if flushed:
        logger.info("关停前补写 %d 个身份的 ReID 特征", flushed)


def apply_output_retention(retention_days: int) -> dict[str, int]:
    cutoff = time.time() - max(1, retention_days) * 86400
    removed_screenshots = 0
    if SCREENSHOT_DIR.exists():
        for screenshot in SCREENSHOT_DIR.glob("*.jpg"):
            try:
                if screenshot.stat().st_mtime < cutoff:
                    screenshot.unlink()
                    removed_screenshots += 1
            except OSError:
                logger.warning("无法清理过期快照: %s", screenshot)

    removed_log_rows = 0
    if ALERT_LOG.exists():
        retained = []
        with ALERT_LOG.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    alert = json.loads(line)
                except json.JSONDecodeError:
                    removed_log_rows += 1
                    continue
                if float(alert.get("timestamp", 0)) < cutoff:
                    removed_log_rows += 1
                else:
                    retained.append(json.dumps(alert, ensure_ascii=False) + "\n")
        temp = ALERT_LOG.with_name(f".{ALERT_LOG.name}.retention.tmp")
        try:
            with temp.open("w", encoding="utf-8") as output:
                output.writelines(retained)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, ALERT_LOG)
        finally:
            if temp.exists():
                temp.unlink()
    return {
        "screenshots": removed_screenshots,
        "jsonl_rows": removed_log_rows,
    }


def main(
    display: bool = False,
    web: bool = True,
    web_port: int = 8000,
    web_host: str = "127.0.0.1",
) -> None:
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

    # ---- 自动检测 GPU 算力 (RTX 3090 / CUDA) ----
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        device = "cuda"
        logger.info("🚀 成功检测到 GPU 硬件加速: %s (%sGB 显存)，已自动激活 CUDA + FP16 加速！", gpu_name, gpu_mem)
    else:
        device = "cpu"
        logger.info("ℹ️ 未检测到 CUDA 显卡，当前运行于 CPU 模式")

    # ---- 性能配置：根据设备自动切换 CPU / GPU 参数 ----
    if device == "cuda":
        perf = dict(
            detect_every_n  = 1,     # GPU：每帧检测
            reid_every_n    = 5,     # GPU：每5帧 ReID
            frame_rate_cap  = 30.0,  # GPU：上限30fps
            mjpeg_fps       = 30.0,  # GPU：MJPEG 30fps
            jpeg_quality    = 85,    # GPU：高画质
        )
        logger.info("⚙️ 性能模式: GPU 高帧率 (30fps / YOLO每帧 / ReID每5帧)")
    else:
        perf = dict(
            detect_every_n  = 3,     # CPU：每3帧检测，中间帧 Kalman 预测
            reid_every_n    = 15,    # CPU：每15帧 ReID，降低 OSNet 推理压力
            frame_rate_cap  = 15.0,  # CPU：上限15fps，避免 CPU 打满
            mjpeg_fps       = 15.0,  # CPU：MJPEG 15fps
            jpeg_quality    = 65,    # CPU：降低编码成本
        )
        logger.info("⚙️ 性能模式: CPU 节能 (15fps / YOLO每3帧 / ReID每15帧)")

    # ---- 线程钳制：22 路解码线程 + 池化并发推理会把 CPU 打满，
    # OpenCV 与 torch 各自的内部线程池再抢核只会加剧上下文切换（吞吐反而下降）。
    # 解码每路固定 1 线程（并发度由 pipeline 线程数提供），单次推理最多 2 线程。
    cv2.setNumThreads(1)
    torch.set_num_threads(2)

    # ---- 初始化共享组件 ----
    # 池化模型实例：打破 detector/reid 的单实例锁，让 N 路 pipeline 真正并发推理
    pool_size = resolve_pool_size(device, len(sources))
    logger.info(
        "加载模型中（模型池大小=%d, device=%s, 视频源=%d 路；首次运行会自动下载权重，"
        "加载耗时 ≈ 池大小 × 单实例耗时）...",
        pool_size, device, len(sources),
    )
    detector = PooledDetector(
        lambda: PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4, device=device),
        size=pool_size,
    )
    reid_extractor = PooledReIDExtractor(
        lambda: build_reid_extractor(device=device),
        size=pool_size,
    )
    logger.info(
        "模型池就绪: detector×%d / reid×%d (device=%s, 特征空间=%s)",
        detector.pool_size, reid_extractor.pool_size, device, reid_extractor.feature_space,
    )
    imported_alerts = db.import_alert_log(ALERT_LOG)
    if imported_alerts:
        logger.info("已将 %d 条旧 JSONL 告警合并进 SQLite", imported_alerts)
    retention_days = int(os.getenv("LAB_MONITOR_RETENTION_DAYS", "30"))
    retention_db = db.apply_retention(retention_days)
    retention_files = apply_output_retention(retention_days)
    if any(retention_db.values()) or any(retention_files.values()):
        logger.info(
            "保留策略已清理: SQLite=%s, files=%s",
            retention_db,
            retention_files,
        )
    identity_store = IdentityStore(
        database=db,
        feature_space=reid_extractor.feature_space,
        max_records=int(os.getenv("LAB_MONITOR_MAX_IDENTITIES", "10000")),
    )
    # 白名单用过滤后的 sources（磁盘不存在的源已被剔除），避免误报未知摄像头 ID
    try:
        topology = CameraTopology(TOPO_CFG, allowed_camera_ids=set(sources))
    except TopologyValidationError as exc:
        logger.error(
            "拓扑配置 %s 校验失败，已降级为空拓扑继续启动："
            "MISSING_PERSON（跨相机超时）告警将不可用，修好配置后需重启服务。原因：%s",
            TOPO_CFG,
            exc,
        )
        # 指向不存在的路径以获得空拓扑，再还原配置路径，便于后续通过 Web 接口修正并落盘
        topology = CameraTopology(
            TOPO_CFG.with_name(f".{TOPO_CFG.name}.disabled"),
            allowed_camera_ids=set(sources),
        )
        topology._config_path = TOPO_CFG
    frame_hub      = FrameHub(jpeg_quality=perf["jpeg_quality"])
    frame_hub.register_cameras(sources)
    broadcaster    = AlertBroadcaster(delivery_enabled=web)
    notifier       = build_notifier(NOTIFY_CFG)
    calibrator     = TransitCalibrator(
        OUTPUT_DIR / "transit_stats.json",
        valid_edges=topology.edges(),
    )
    alert_manager  = AlertManager(
        alert_log=ALERT_LOG,
        notifier=notifier,
        broadcaster=broadcaster,
        identity_store=identity_store,  # 场景消失检测
        scene_exit_seconds=300.0,       # 5 分钟无出现则触发 SCENE_EXIT
        screenshot_dir=SCREENSHOT_DIR,
        frame_provider=frame_hub.get_frame,
        database=db,
    )

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    shutdown_event = threading.Event()

    # 写入真实服务进程 PID（start.ps1 拿到的是 venv launcher 的 PID，并不可靠）
    try:
        PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    except OSError as exc:
        logger.warning("无法写入 PID 文件 %s: %s", PID_FILE, exc)

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
            detect_every_n = perf["detect_every_n"],
            reid_every_n   = perf["reid_every_n"],
            frame_rate_cap = perf["frame_rate_cap"],
        )
        pipelines.append(p)
        p.start()
        logger.info("启动摄像头 %s → %s", cam_id, redact_source(source))

    def request_shutdown() -> None:
        if shutdown_event.is_set():
            return
        logger.info("收到安全停止请求，正在停止流水线并落盘...")
        shutdown_event.set()
        for pipeline in pipelines:
            pipeline.stop()

    # ---- 启动 Web 服务器 ----
    if web:
        web_server.init_server(
            frame_hub, broadcaster, identity_store, calibrator,
            pipelines=pipelines, topology=topology,
            mjpeg_fps=perf["mjpeg_fps"],
            shutdown_callback=request_shutdown,
        )
        try:
            web_server.start_server_thread(host=web_host, port=web_port)
        except Exception:
            request_shutdown()
            for pipeline in pipelines:
                pipeline.join(timeout=5)
            alert_manager.close()
            flush_identity_features(identity_store)
            calibrator.flush()
            db.close()
            clear_pid_file()
            raise

    # ---- 启动预警后台检查线程 ----
    ticker = threading.Thread(
        target=alert_ticker,
        args=(alert_manager, shutdown_event),
        daemon=True,
        name="alert-ticker",
    )
    ticker.start()

    logger.info("所有流水线已启动（%d 路）", len(pipelines))
    if web:
        logger.info("监控大屏：http://%s:%d", web_host, web_port)

    try:
        # 只等停止信号：Web 大屏在流水线全部退出后仍需继续提供历史查询
        while not shutdown_event.wait(0.5):
            # 无 Web 模式没有查询需求，流水线全部退出即可结束进程
            if not web and not any(p.is_alive() for p in pipelines):
                break
    except KeyboardInterrupt:
        request_shutdown()
    finally:
        request_shutdown()
        for pipeline in pipelines:
            pipeline.join(timeout=5)
        ticker.join(timeout=2)
        flush_identity_features(identity_store)
        calibrator.flush()
        alert_manager.close()
        db.close()
        clear_pid_file()

    logger.info("系统已停止，告警日志：%s", ALERT_LOG)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="超算中心监控预警系统")
    parser.add_argument("--display",   action="store_true", help="显示本地画面窗口")
    parser.add_argument("--no-web",    action="store_true", help="不启动 Web 服务器")
    parser.add_argument("--host",      default="127.0.0.1", help="Web 监听地址（默认仅本机）")
    parser.add_argument("--port",      type=int, default=8000, help="Web 端口（默认 8000）")
    args = parser.parse_args()
    main(
        display=args.display,
        web=not args.no_web,
        web_port=args.port,
        web_host=args.host,
    )
