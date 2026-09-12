"""
pipeline.py — 单路视频处理流水线（独立线程运行）
Phase 4 更新：集成 ReIDValidator（多帧确认）+ TransitCalibrator（通行时间记录）
"""

import time
import logging
import threading
import os
import cv2
import numpy as np
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .detector import PersonDetector
from .tracker import PersonTracker
from .reid import ReIDExtractor
from .reid_validator import ReIDValidator
from .identity_store import IdentityStore
from .topology import CameraTopology
from .alerter import AlertManager
from .frame_hub import FrameHub
from .calibrator import TransitCalibrator

logger = logging.getLogger("pipeline")

# RTSP 重连配置：指数退避策略（无最大重试次数限制）
_RTSP_INITIAL_DELAY = 3.0
_RTSP_MAX_DELAY = 60.0
_RTSP_BACKOFF_FACTOR = 1.5
_RTSP_OPEN_TIMEOUT_MS = int(os.getenv("LAB_MONITOR_RTSP_OPEN_TIMEOUT_MS", "10000"))
_RTSP_READ_TIMEOUT_MS = int(os.getenv("LAB_MONITOR_RTSP_READ_TIMEOUT_MS", "10000"))

# 本地文件源重试配置：正常循环播放保持 0.5s 间隔，坏源指数退避至 60s（F3）
_FILE_INITIAL_DELAY = 0.5
_FILE_MAX_DELAY = 60.0

# 离场宽限（F2b）：track_id 从 tracker 输出里消失多少个「处理帧」才算真的离场。
#
# ultralytics 8.4 的 BYTETracker._format_output() 只返回 is_activated 的轨迹，
# 一次 IoU 关联失败该 track 当帧就从输出里消失（tracker 内部仍按 track_buffer=30
# 帧保留、后续可复活）。若当帧即判离场，代价是三重的：
#   1. watch() 起 MISSING_PERSON 倒计时 → 人根本没走，纯误报；
#   2. _track_to_global 弹出 → 同一个人复活后要重新走一遍身份识别；
#   3. _validator.clear() 把攒了一半的 ReID 缓冲清空 → 8 帧缓冲永远填不满、
#      register_if_new() 没有机会执行。
#
# 注意宽限期只在帧率足够时才有收益：实测 12.5fps 下伪离场事件 18 → 10（清零），
# 但 1.5~3fps 下 11 → 11（毫无变化）—— 低帧率时 tracker 压根关联不上、直接分配
# 新 track_id，旧 id 不会在宽限期内回来。所以这条改动必须和「把帧率提到 10fps
# 以上」一起才成立，别指望它单独救低帧率下的 MISSING_PERSON 误报。
# 宽限期必须小于 track_buffer（30，见 src/tracker.py:_default_args），
# 超过之后 tracker 已彻底丢弃该轨迹，人再出现也会拿到新的 track_id —— 故上限钳到 30。
#
# 默认 12 帧是在 7 路真实素材上扫出来的（12.5fps 工作点，10 条真实轨迹）：
#   grace  1 → 18 次离场事件（8 次是误报）      grace  5 → 12 次（2 误报）
#   grace  8 → 11 次（1 误报）                  grace 12 → 10 次（0 误报）
# 代价是真离场要晚约 grace/fps 秒才上报（12 帧 @12.5fps ≈ 0.96s）。相对
# MISSING_PERSON 的 5~180s 时间窗可忽略，但会给 TransitCalibrator 的通行时间
# 带来同量级的正偏差（共位反向相机对那种 5s/10s 的短边上约占 20%，仍在容忍内）。
_LEAVE_GRACE_FRAMES = min(30, max(1, int(os.getenv("LAB_MONITOR_LEAVE_GRACE_FRAMES", "12"))))


def _is_rtsp(source: str) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


def redact_source(source: str) -> str:
    if not _is_rtsp(source):
        return source
    try:
        parsed = urlsplit(source)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        netloc = f"***:***@{host}" if parsed.username is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        return "rtsp://<redacted>"


class CameraPipeline(threading.Thread):
    # 默认值保持向后兼容（GPU 模式），CPU 模式由 main.py 传参覆盖
    REID_EVERY_N_FRAMES = 5

    def __init__(
        self,
        camera_id: str,
        source: str,
        detector: PersonDetector,
        reid_extractor: ReIDExtractor,
        identity_store: IdentityStore,
        topology: CameraTopology,
        alert_manager: AlertManager,
        screenshot_dir: Path,
        frame_hub: FrameHub | None = None,
        calibrator: TransitCalibrator | None = None,
        display: bool = False,
        detect_every_n: int = 1,      # YOLO 跳帧：每 N 帧推理一次（CPU=3，GPU=1）
        reid_every_n: int = 5,        # ReID 跳帧：每 N 帧提取一次特征（CPU=15，GPU=5）
        frame_rate_cap: float = 30.0, # 帧率上限（这个默认值只是兜底，实际由 main.py 传入）
        process_max_width: int = 960, # 运行时缩放上限（worklist 2.4：RTSP 在线流无法离线转码，
                                      # 只能靠解码后缩放；超过则等比缩小）
        personnel=None,               # PersonnelGallery（worklist 3.3）：底库 1:N 自动命名
    ):
        super().__init__(name=f"pipeline-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.detector = detector
        self.reid = reid_extractor
        self.store = identity_store
        self.topology = topology
        self.alerter = alert_manager
        self.screenshot_dir = screenshot_dir
        self.frame_hub = frame_hub
        self.calibrator = calibrator
        self.display = display

        self._stop_event = threading.Event()
        self._track_to_global: dict[int, str] = {}
        # 在场的 track_id（含仍处于离场宽限期内的），不是「上一帧 tracker 输出」
        self._prev_track_ids: set[int] = set()
        # track_id → 连续从 tracker 输出中缺席的处理帧数（F2b 离场宽限）
        self._absent_streak: dict[int, int] = {}
        self._reid_frame_counter: dict[int, int] = {}
        self._tracker = PersonTracker(fps=25)
        self._frame_idx = 0

        # 性能参数（CPU/GPU 自动切换，由 main.py 传入）
        self._detect_every_n = max(1, detect_every_n)
        self._reid_every_n   = max(1, reid_every_n)
        self._frame_rate_cap = max(1.0, frame_rate_cap)
        # 跳帧时复用上次 tracker 输出：保持 track_id 稳定，ReID 能正常积累
        # 不能传空列表给 ByteTracker（会立刻清空所有 track，导致 ReID 永远重置）
        self._last_tracks: list = []

        # Phase 4: 多帧 ReID 确认器（每路摄像头独立）
        self._validator = ReIDValidator(
            buffer_size=8,
            confirm_frames=3,
            # 匹配阈值刻意不在调用点写死：默认值来自 src/reid_config.py
            # （可用 LAB_MONITOR_REID_THRESHOLD / _RATIO 覆盖），保证只有一处真相。
            # 历史上这里与另外 5 处各自硬编码 0.75，导致"调阈值"在工程上不可执行。
        )

        self._reconnect_count = 0
        self._rois = self._load_rois()

        # ---- 视频内坐标（worklist 2.3）----
        # _frame_idx 是**进程生命周期**累计值，不随素材循环复位，不能当"源视频第几帧"用。
        # 必须单独维护一个"当前文件内帧号"，在每次重开流（文件循环 / RTSP 重连）时归零。
        self._file_frame_idx = 0
        self._asset_id: int | None = None
        self._video_fps = 25.0          # 把帧号换算成"视频内秒数"用的帧率
        self._process_max_width = max(0, int(process_max_width))
        self.personnel = personnel
        self._register_asset()

    def _register_asset(self) -> None:
        """
        登记本路视频资产并取回 asset_id（worklist 2.1/2.3）。

        元数据（实测帧数/时长）由 scripts/seed_video_assets.py 以 ffprobe 实测写入；
        这里只保证 (camera_id, rel_path) 有行可查 —— 若 seeder 还没跑过，
        INSERT OR IGNORE 会先放一个占位行，之后 seeder 再补齐元数据。
        """
        if self.store is None or _is_rtsp(self.source):
            return
        try:
            path = Path(self.source)
            size = path.stat().st_size if path.exists() else None
            self._asset_id = self.store.register_video_asset_stub(
                self.camera_id, self.source, path.name, size
            )
            asset = self.store.video_asset_for(self.camera_id)
            if asset and asset.get("fps_declared"):
                fps = float(asset["fps_declared"])
                # 元数据帧率可能损坏（rnd_05 报 351.56），只接受合理区间
                if 10.0 <= fps <= 60.0:
                    self._video_fps = fps
        except Exception:
            logger.warning("[%s] 视频资产登记失败，视频内坐标将缺失", self.camera_id,
                           exc_info=True)

    def _load_rois(self) -> list[dict]:
        import json
        roi_file = Path(__file__).parent.parent / "config" / "roi.json"
        if not roi_file.exists():
            return []
        try:
            data = json.loads(roi_file.read_text(encoding="utf-8"))
            raw_rois = data.get(self.camera_id, [])
        except Exception as error:
            # 静默失败等于关闭全域电子围栏，必须留下痕迹（F10）
            logger.error(
                "[%s] roi.json 读取失败（%s: %s），本路电子围栏已关闭",
                self.camera_id, type(error).__name__, error,
            )
            return []
        return self._sanitize_rois(raw_rois)

    def _sanitize_rois(self, raw_rois) -> list[dict]:
        """
        清洗 ROI 配置（F10）：坐标强制转 float 并裁剪到 [0,1]，顶点数不足或
        含非法坐标的 ROI 直接跳过并记日志。
        原因：写入接口可能把坐标原样落盘为字符串，_process_frame 里的
        int(p[0] * w) 会抛 ValueError，异常穿出 _read_loop 后该路线程永久退出。
        """
        if not isinstance(raw_rois, list):
            logger.error("[%s] roi.json 中本相机的配置不是数组，已忽略", self.camera_id)
            return []

        cleaned: list[dict] = []
        for idx, roi in enumerate(raw_rois):
            if not isinstance(roi, dict):
                logger.error("[%s] ROI #%d 不是对象，已跳过", self.camera_id, idx)
                continue
            roi_name = roi.get("name", "未命名")
            polygon = roi.get("polygon")
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                logger.error(
                    "[%s] ROI #%d(%s) 缺少 polygon 或顶点数不足 3，已跳过",
                    self.camera_id, idx, roi_name,
                )
                continue

            points: list[list[float]] = []
            for point in polygon:
                try:
                    x = min(1.0, max(0.0, float(point[0])))
                    y = min(1.0, max(0.0, float(point[1])))
                except (TypeError, ValueError, IndexError, KeyError):
                    points = []
                    break
                points.append([x, y])
            if len(points) < 3:
                logger.error(
                    "[%s] ROI #%d(%s) 含非法坐标，已跳过",
                    self.camera_id, idx, roi_name,
                )
                continue

            item = dict(roi)
            item["polygon"] = points
            cleaned.append(item)
        return cleaned

    def _roi_points(self, roi: dict, w_img: int, h_img: int) -> np.ndarray:
        """归一化多边形 → 像素坐标（坐标已由 _load_rois 清洗为 [0,1] 的 float）"""
        return np.array(
            [[int(p[0] * w_img), int(p[1] * h_img)] for p in roi["polygon"]], np.int32
        )

    def reload_rois(self) -> None:
        self._rois = self._load_rois()

    def stop(self) -> None:
        self._stop_event.set()

    def _reset_stream_state(self) -> None:
        """
        重开流（本地文件循环播放 / RTSP 重连）前重置跟踪与 ReID 状态（F2）。

        不重置时新片头的检测框与旧轨迹关联不上，旧 track_id 会集体从 tracker
        输出中消失，被 _process_frame 误判为人员离场 → watch() → MISSING_PERSON 误报。
        清空 _prev_track_ids 与 _absent_streak 即可消除这批伪离场事件。

        注意：**不要**在这里重建 PersonTracker。BYTETracker.__init__ 会调用
        reset_id()，而 BaseTrack._count 是**类级共享**计数器，重建会把全部
        摄像头线程的 track_id 一起归零，导致其它路正在活跃的 track_id 与新分配
        的 id 撞号（_track_to_global 会把新来的人认成旧身份）。旧轨迹留在 tracker
        里由 track_buffer 自然淘汰即可，不影响正确性。
        _track_to_global 与 _validator 必须清空：track_id 可能被新片段复用，
        否则新进入的人会直接继承上一轮的 global_id。
        """
        self._validator = ReIDValidator(
            buffer_size=8,
            confirm_frames=3,
            # 匹配阈值刻意不在调用点写死：默认值来自 src/reid_config.py
            # （可用 LAB_MONITOR_REID_THRESHOLD / _RATIO 覆盖），保证只有一处真相。
            # 历史上这里与另外 5 处各自硬编码 0.75，导致"调阈值"在工程上不可执行。
        )
        self._prev_track_ids = set()
        self._absent_streak = {}
        self._track_to_global = {}
        self._reid_frame_counter = {}
        self._last_tracks = []
        # 新一轮播放从第 0 帧开始 —— 视频内坐标必须跟着归零
        self._file_frame_idx = 0

    def _capture_ready(self, cap) -> bool:
        """
        判定 capture 是否真的可用（F3）。
        损坏文件（H.264 缺 PPS 等）isOpened() 仍返回 True，但元数据全废、
        画面宽度为 0，首帧必然解码失败，这里提前识别避免无意义重开。
        """
        if cap is None or not cap.isOpened():
            return False
        try:
            width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        except (TypeError, ValueError):
            return True  # 取不到宽度时不做预判，交由实际读帧结果决定
        return width > 0

    # ------------------------------------------------------------------ #
    # 主循环                                                                #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            if _is_rtsp(self.source):
                self._run_rtsp()
            else:
                self._run_file()
        except Exception:
            logger.exception("[%s] 流水线发生未处理异常", self.camera_id)
            if self.frame_hub:
                self.frame_hub.mark_offline(
                    self.camera_id, status_text="PIPELINE_ERROR"
                )

    def _run_file(self) -> None:
        # 坏源熔断（F3）：本轮解码 0 帧则计入失败并指数退避，只在状态翻转时打一条日志
        fail_streak = 0
        delay = _FILE_INITIAL_DELAY
        while not self._stop_event.is_set():
            cap = None
            frames = 0
            try:
                cap = cv2.VideoCapture(self.source)
                if self._capture_ready(cap):
                    logger.debug("[%s] 开始处理文件: %s", self.camera_id, self.source)
                    frames = self._read_loop(cap)
            except Exception:
                logger.exception("[%s] 本地视频处理异常", self.camera_id)
                if self.frame_hub:
                    self.frame_hub.mark_offline(
                        self.camera_id, status_text="PIPELINE_ERROR"
                    )
                return
            finally:
                if cap is not None:
                    cap.release()

            if frames:
                # 正常循环播放：不给可用源引入退避
                fail_streak = 0
                delay = _FILE_INITIAL_DELAY
            else:
                fail_streak += 1
                if fail_streak == 1:
                    logger.error(
                        "[%s] 视频源不可用（无法打开或首帧解码失败）: %s，转入退避重试",
                        self.camera_id, self.source,
                    )
                if self.frame_hub:
                    self.frame_hub.mark_offline(
                        self.camera_id,
                        status_text="FILE_ERROR",
                        reconnect_count=fail_streak,
                    )
                delay = min(delay * 2, _FILE_MAX_DELAY)

            # 重开前重置跟踪状态，EOF 不等于人员离场（F2）
            self._reset_stream_state()
            # 用 wait 而非 sleep，stop() 可立刻中断退避
            if self._stop_event.wait(delay):
                break
        if self.frame_hub:
            self.frame_hub.mark_offline(self.camera_id, status_text="STOPPED")
        logger.info("[%s] 文件流水线结束", self.camera_id)

    def _run_rtsp(self) -> None:
        """RTSP 流处理：无限重连+指数退避，消除网络抖动导致的永久下线"""
        retry_delay = _RTSP_INITIAL_DELAY
        while not self._stop_event.is_set():
            logger.info(
                "[%s] 连接 RTSP: %s (重连次数: %d)",
                self.camera_id,
                redact_source(self.source),
                self._reconnect_count,
            )
            cap = None
            connected = False
            try:
                params = []
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _RTSP_OPEN_TIMEOUT_MS])
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, _RTSP_READ_TIMEOUT_MS])
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
                if not cap.isOpened():
                    raise ConnectionError("RTSP 连接未打开")
                connected = True
                logger.info("[%s] RTSP 连接成功", self.camera_id)
                retry_delay = _RTSP_INITIAL_DELAY
                self._read_loop(cap)
            except Exception as error:
                self._reconnect_count += 1
                logger.error(
                    "[%s] RTSP %s异常 (%s，已重试 %d 次)，%.1fs 后重试",
                    self.camera_id,
                    "读取" if connected else "连接",
                    type(error).__name__,
                    self._reconnect_count,
                    retry_delay,
                )
                if self.frame_hub:
                    self.frame_hub.mark_offline(
                        self.camera_id,
                        status_text="RECONNECTING",
                        reconnect_count=self._reconnect_count,
                    )
                connected = False
            finally:
                if cap is not None:
                    cap.release()

            if self._stop_event.is_set():
                break
            if connected:
                self._reconnect_count += 1
                logger.warning("[%s] RTSP 流中断，%.1fs 后重连", self.camera_id, retry_delay)
                if self.frame_hub:
                    self.frame_hub.mark_offline(
                        self.camera_id,
                        status_text="RECONNECTING",
                        reconnect_count=self._reconnect_count,
                    )
            # 重连前重置跟踪状态：新连接的 track_id 会重新分配，旧轨迹并非人员离场（F2）
            self._reset_stream_state()
            self._stop_event.wait(retry_delay)
            retry_delay = min(
                retry_delay * _RTSP_BACKOFF_FACTOR, _RTSP_MAX_DELAY
            )

        if self.frame_hub:
            self.frame_hub.mark_offline(self.camera_id, status_text="STOPPED", reconnect_count=self._reconnect_count)
        logger.info("[%s] RTSP 流水线结束", self.camera_id)

    def _read_loop(self, cap: cv2.VideoCapture) -> int:
        """读帧主循环；返回本轮成功解码的帧数（供 _run_file 判定坏源熔断，F3）"""
        # GPU 服务器：读帧速度远快于源视频帧率，需要限速避免空跑浪费 GPU 算力
        # 上限由 main.py 传入：CPU=15fps，GPU=30fps
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_fps = min(source_fps, self._frame_rate_cap)
        frame_interval = 1.0 / target_fps
        decoded = 0

        while not self._stop_event.is_set():
            t_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                break
            self._frame_idx += 1
            self._file_frame_idx += 1
            decoded += 1
            # 运行时缩放兜底（worklist 2.4）：必须在 _process_frame 与 push_frame 之前，
            # 否则 YOLO / ReID / MJPEG / 告警截图拿到的还是原始大帧。
            # 本地素材已离线转码时这一步不会触发（宽 already <= 上限），零开销。
            if self._process_max_width and frame.shape[1] > self._process_max_width:
                scale = self._process_max_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (self._process_max_width, max(1, int(round(frame.shape[0] * scale)))),
                )
            self._process_frame(frame)
            if self.frame_hub:
                self.frame_hub.push_frame(self.camera_id, frame, status_text="ONLINE")
            if self.display:
                cv2.imshow(f"Camera {self.camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop_event.set()
                    break

            # 帧率限速：sleep 剩余时间，对齐到目标帧率
            elapsed = time.monotonic() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.display:
            cv2.destroyWindow(f"Camera {self.camera_id}")
        return decoded

    # ------------------------------------------------------------------ #
    # 核心处理逻辑（Phase 4：多帧验证 + 校准记录）                             #
    # ------------------------------------------------------------------ #

    def _process_frame(self, frame: np.ndarray) -> None:
        h_img, w_img = frame.shape[:2]

        # 绘制 ROI 危险区域边框
        for roi in self._rois:
            pts = self._roi_points(roi, w_img, h_img)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            roi_label = f"[ROI] {roi.get('name', '危险区')}"
            cv2.putText(frame, roi_label, (pts[0][0][0], max(20, pts[0][0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # YOLO 跳帧检测：CPU 模式每 N 帧推理一次
        # 非检测帧直接复用上次 tracker 输出（track_id 稳定，ReID 持续积累）
        # 不能传空列表给 ByteTracker：空列表会导致所有 track 瞬间丢失
        if self._frame_idx % self._detect_every_n == 0:
            detections = self.detector.detect(frame)
            self._last_tracks = self._tracker.update(detections, frame.shape)
        tracks = self._last_tracks
        current_track_ids = {t["track_id"] for t in tracks}
        self.alerter.trigger_crowd_warning(
            self.camera_id,
            len(current_track_ids),
            frame=frame,
        )

        # 离场判定带宽限期（F2b，原因见 _LEAVE_GRACE_FRAMES 注释）：
        # 重新出现就把缺席计数清零，连续缺席够 _LEAVE_GRACE_FRAMES 帧才按离场处理。
        for tid in current_track_ids:
            self._absent_streak.pop(tid, None)
        for tid in self._prev_track_ids - current_track_ids:
            streak = self._absent_streak.get(tid, 0) + 1
            if streak >= _LEAVE_GRACE_FRAMES:
                self._absent_streak.pop(tid, None)
                self._on_person_leave(tid, snapshot_frame=frame)
            else:
                self._absent_streak[tid] = streak

        for track in tracks:
            tid = track["track_id"]
            bbox = track["bbox"]
            conf = track["conf"]

            # ReID 跳帧：CPU=15帧，GPU=5帧（由 main.py 传入）
            last_reid = self._reid_frame_counter.get(tid, -999)
            if self._frame_idx - last_reid < self._reid_every_n:
                continue
            self._reid_frame_counter[tid] = self._frame_idx

            feat = self.reid.extract(frame, bbox)
            if feat is None:
                continue

            # 积累到多帧缓冲区
            self._validator.add_feature(tid, feat)

            gid = self._track_to_global.get(tid)
            if gid is None:
                # 用多帧平均特征做确认匹配
                # 匹配上下文把「已中心化的 gallery」与「所用中心」原子打包的原因见 build_match_context 的文档。
                context = self.store.build_match_context()

                # 底库 1:N 检索（worklist 3.3，能力一路径 B）：**实名优先于匿名编号**。
                # 命中且该人名下已有绑定的 global_id → 直接复用那个身份（这就是
                # "认出熟人"）；命中但未绑定 → 先走正常注册，注册成功后自动命名。
                # 放在全局身份库检索之前的原因见 src/personnel.py 模块注释。
                personnel_hit = None
                known_gid = None
                if self.personnel is not None:
                    avg_feat = self._validator.get_avg_feature(tid)
                    if avg_feat is not None:
                        personnel_hit = self.personnel.match(avg_feat)
                        if personnel_hit is not None:
                            known_gid = self.personnel.bound_gid(
                                personnel_hit["person_id"], self.store)
                            if known_gid:
                                gid = known_gid
                                self._track_to_global[tid] = gid
                                self._record_arrival(gid)
                                self.alerter.resolve(gid, self.camera_id)
                                logger.info(
                                    "[%s] ✓ 底库命中已知人员: %s (%s, 相似度 %.3f)",
                                    self.camera_id, personnel_hit["name"],
                                    personnel_hit["person_id"], personnel_hit["score"])

                confirmed_gid = self._validator.get_confirmed_match(
                    tid, context.gallery, metrics=self.store.metrics,
                    prepare=context.prepare,
                )

                if confirmed_gid:
                    gid = confirmed_gid
                    # 记录通行时间（校准用）
                    self._record_arrival(gid)
                    # 解除预警监听
                    self.alerter.resolve(gid, self.camera_id)
                    logger.info("[%s] ✓ 身份确认（多帧）: %s", self.camera_id, gid)
                else:
                    # 尚未确认，但缓冲帧够了且完全无匹配 → 注册（或归并）身份
                    # P2: 使用公开接口 buffer_len()，替代直接访问私有 _buffers
                    if self._validator.buffer_len(tid) >= self._validator.buffer_size:
                        avg_feat = self._validator.get_avg_feature(tid)
                        if avg_feat is not None:
                            resolution = self.store.register_if_new(avg_feat)
                            gid = resolution.global_id
                            if gid is not None:
                                self._validator.confirm(tid, gid)
                                if resolution.is_new:
                                    logger.info("[%s] 注册新身份（多帧平均）: %s", self.camera_id, gid)
                                    # 底库已命中但名下没有身份 → 自动命名（路径 B 的闭环）
                                    if personnel_hit is not None and not known_gid:
                                        self.store.bind_person(
                                            gid, personnel_hit["person_id"],
                                            personnel_hit["score"])
                                        logger.info(
                                            "[%s] 自动命名: %s -> %s (相似度 %.3f)",
                                            self.camera_id, gid,
                                            personnel_hit["person_id"],
                                            personnel_hit["score"])
                                else:
                                    self._record_arrival(gid)
                                    self.alerter.resolve(gid, self.camera_id)
                                    logger.info("[%s] 跨摄归并身份: %s", self.camera_id, gid)
                            elif resolution.status == "ambiguous":
                                logger.debug(
                                    "[%s] Track %s ReID 结果有歧义，继续积累特征",
                                    self.camera_id,
                                    tid,
                                )

                if gid:
                    self._track_to_global[tid] = gid

            if gid:
                # 计算帧质量分（bbox 面积 × 检测置信度），指导特征滑动平均权重（P1-3）
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area_score = min(1.0, (w * h) / (128.0 * 256.0))
                quality = area_score * min(1.0, conf)
                self.store.update_appearance(
                    gid, self.camera_id, feat, bbox,
                    quality_score=quality,
                    asset_id=self._asset_id,
                    video_frame=self._file_frame_idx,
                    video_ts=self._file_frame_idx / self._video_fps,
                )
                self.alerter.mark_seen(gid)

        # 绘制定位框和身份/追踪 ID 标签 + ROI 校验
        for track in tracks:
            tid = track["track_id"]
            x1, y1, x2, y2 = map(int, track["bbox"])
            conf = track["conf"]
            gid = self._track_to_global.get(tid)

            # 围栏入侵检测：检查 bbox 底部 3 个关键点 + 中心点
            # 任一点落入 ROI 多边形即视为侵入，避免纯足点漏报
            check_points = [
                (float((x1 + x2) / 2), float(y2)),            # 底部中心（脚点）
                (float(x1),            float(y2)),            # 底部左角
                (float(x2),            float(y2)),            # 底部右角
                (float((x1 + x2) / 2), float((y1 + y2) / 2)), # 身体中心
            ]
            is_intrusion = False
            for roi in self._rois:
                pts = self._roi_points(roi, w_img, h_img)
                if any(cv2.pointPolygonTest(pts, pt, measureDist=False) >= 0 for pt in check_points):
                    is_intrusion = True
                    self.alerter.trigger_intrusion(
                        self.camera_id,
                        gid or f"Trk_{tid}",
                        roi.get("name", "危险区域"),
                        [x1, y1, x2, y2],
                        frame=frame,
                    )
                    break

            # 计算质量分（bbox 面积 × 检测置信度）
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area_score = min(1.0, (w * h) / (128.0 * 256.0))
            quality = area_score * min(1.0, conf)

            # 如果侵入 ROI 用亮红色高亮框，否则已识别天蓝/未识别绿
            if is_intrusion:
                color = (0, 0, 239)        # BGR 亮红
                text_color = (255, 255, 255)
                label_text = f"🚨 INTRUSION ID: {gid or f'Trk_{tid}'}"
            elif gid:
                color = (248, 189, 56)     # BGR 天蓝/金色
                text_color = (255, 255, 255)
                # 已绑实名的身份显示"姓名 (person_id)"，否则显示匿名编号
                person_label = self.store.person_label(gid) if self.store else None
                label_text = (f"{person_label} | #{gid}" if person_label
                              else f"ID: #{gid}")
            else:
                color = (129, 185, 16)     # BGR 翡翠绿
                text_color = (255, 255, 255)
                buf_len = self._validator.buffer_len(tid)
                label_text = f"Trk: #{tid} [{buf_len}/3]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_intrusion else 2)

            # 超清晰高对比度大字号 ID 标签绘制（黑底 + 亮色描边）
            font_scale = 0.6
            font_thick = 2
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            text_y = max(th + 10, y1)

            # 1. 绘制带有高亮边框的纯黑底框矩形
            cv2.rectangle(frame, (x1 - 1, text_y - th - 8), (x1 + tw + 12, text_y + 6), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1 - 1, text_y - th - 8), (x1 + tw + 12, text_y + 6), color, 1)

            # 2. 绘制黑色外加粗描边 + 内部纯白/彩色高清文字
            cv2.putText(frame, label_text, (x1 + 5, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 2, cv2.LINE_AA)
            cv2.putText(frame, label_text, (x1 + 5, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)

        # 宽限期内的 track_id 仍算「在场」，否则它下一帧就从 _prev_track_ids 里消失，
        # 缺席计数再也累加不到阈值 —— 宽限期会静默失效，离场事件永远不触发。
        self._prev_track_ids = current_track_ids | set(self._absent_streak)

    def _record_arrival(self, gid: str) -> None:
        """人员到达本摄像头时，计算并记录通行时间"""
        if self.calibrator is not None:
            self.calibrator.record_arrival(gid, self.camera_id)

    def _on_person_leave(self, track_id: int, snapshot_frame=None) -> None:
        gid = self._track_to_global.pop(track_id, None)
        self._reid_frame_counter.pop(track_id, None)
        self._validator.clear(track_id)

        if gid is None:
            return

        next_hops = self.topology.next_hops(self.camera_id)
        if not next_hops:
            return

        # Phase 4：用校准后的时间窗口（样本不足时回退到配置值）
        calibrated_hops = []
        for hop in next_hops:
            if self.calibrator:
                exp, tol = self.calibrator.calibrated_window(
                    self.camera_id, hop.camera_id,
                    hop.expected_seconds, hop.tolerance_seconds,
                )
            else:
                exp, tol = hop.expected_seconds, hop.tolerance_seconds
            calibrated_hops.append((hop.camera_id, exp + tol))

        max_deadline = max(d for _, d in calibrated_hops)
        expected_cams = [c for c, _ in calibrated_hops]

        # 离开状态由共享校准器保存，供其他摄像头原子消费
        if self.calibrator:
            self.calibrator.record_departure(
                gid,
                self.camera_id,
                expected_cams,
            )

        self.alerter.watch(
            global_id=gid,
            last_camera=self.camera_id,
            expected_cameras=expected_cams,
            deadline_offset=max_deadline,
            # 使用锁内安全接口读取 bbox，替代锁外裸访问 rec.appearances
            last_bbox=self.store.get_last_bbox(gid),
            snapshot_frame=snapshot_frame,
        )
        logger.debug(
            "[%s] Person %s 离开，预期在 %s 出现（校准后 %.0fs 内）",
            self.camera_id, gid, expected_cams, max_deadline,
        )
