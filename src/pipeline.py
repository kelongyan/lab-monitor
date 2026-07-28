"""
pipeline.py — 单路视频处理流水线（独立线程运行）
Phase 4 更新：集成 ReIDValidator（多帧确认）+ TransitCalibrator（通行时间记录）
"""

import time
import logging
import threading
import cv2
import numpy as np
from pathlib import Path

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

# _leave_times 条目的过期时间：超过该时间未到达下一个摄像头视为已离开场景
_LEAVE_EXPIRY_SECONDS = 3600
# RTSP 重连配置：指数退避策略（无最大重试次数限制）
_RTSP_INITIAL_DELAY = 3.0
_RTSP_MAX_DELAY = 60.0
_RTSP_BACKOFF_FACTOR = 1.5


def _is_rtsp(source: str) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


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
        frame_rate_cap: float = 30.0, # 帧率上限（CPU=15，GPU=30）
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
        self._prev_track_ids: set[int] = set()
        self._reid_frame_counter: dict[int, int] = {}
        self._tracker = PersonTracker(fps=25)
        self._frame_idx = 0

        # 性能参数（CPU/GPU 自动切换，由 main.py 传入）
        self._detect_every_n = max(1, detect_every_n)
        self._reid_every_n   = max(1, reid_every_n)
        self._frame_rate_cap = max(1.0, frame_rate_cap)
        self._last_detections: list = []  # YOLO 跳帧时复用上一次检测结果

        # Phase 4: 多帧 ReID 确认器（每路摄像头独立）
        self._validator = ReIDValidator(
            buffer_size=8,
            confirm_frames=3,
            threshold=0.75,
        )

        # Phase 4: 记录人员离开时间戳，供校准器计算通行时间
        # global_id → (leave_time, leave_camera)
        self._leave_times: dict[str, tuple[float, str]] = {}
        self._reconnect_count = 0
        self._rois = self._load_rois()

    def _load_rois(self) -> list[dict]:
        import json
        roi_file = Path(__file__).parent.parent / "config" / "roi.json"
        if not roi_file.exists():
            return []
        try:
            data = json.loads(roi_file.read_text(encoding="utf-8"))
            return data.get(self.camera_id, [])
        except Exception:
            return []

    def reload_rois(self) -> None:
        self._rois = self._load_rois()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # 主循环                                                                #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        if _is_rtsp(self.source):
            self._run_rtsp()
        else:
            self._run_file()

    def _run_file(self) -> None:
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.error("[%s] 无法打开视频文件: %s", self.camera_id, self.source)
                if self.frame_hub:
                    self.frame_hub.mark_offline(self.camera_id, status_text="FILE_ERROR")
                return
            logger.info("[%s] 开始处理文件: %s", self.camera_id, self.source)
            self._read_loop(cap)
            cap.release()
            time.sleep(0.5)
        if self.frame_hub:
            self.frame_hub.mark_offline(self.camera_id, status_text="STOPPED")
        logger.info("[%s] 文件流水线结束", self.camera_id)

    def _run_rtsp(self) -> None:
        """RTSP 流处理：无限重连+指数退避，消除网络抖动导致的永久下线"""
        retry_delay = _RTSP_INITIAL_DELAY
        while not self._stop_event.is_set():
            logger.info("[%s] 连接 RTSP: %s (重连次数: %d)", self.camera_id, self.source, self._reconnect_count)
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self._reconnect_count += 1
                logger.warning("[%s] RTSP 连接失败 (已重试 %d 次)，%.1fs 后重试", self.camera_id, self._reconnect_count, retry_delay)
                if self.frame_hub:
                    self.frame_hub.mark_offline(self.camera_id, status_text="RECONNECTING", reconnect_count=self._reconnect_count)
                self._stop_event.wait(retry_delay)
                # 指数退避：每次失败延迟增长 1.5 倍，上限 60 秒
                retry_delay = min(retry_delay * _RTSP_BACKOFF_FACTOR, _RTSP_MAX_DELAY)
                continue

            logger.info("[%s] RTSP 连接成功", self.camera_id)
            retry_delay = _RTSP_INITIAL_DELAY  # 连接成功后重置退避
            self._read_loop(cap)
            cap.release()

            if self._stop_event.is_set():
                break
            self._reconnect_count += 1
            logger.warning("[%s] RTSP 流中断，%.1fs 后重连", self.camera_id, retry_delay)
            if self.frame_hub:
                self.frame_hub.mark_offline(self.camera_id, status_text="RECONNECTING", reconnect_count=self._reconnect_count)
            self._stop_event.wait(retry_delay)

        if self.frame_hub:
            self.frame_hub.mark_offline(self.camera_id, status_text="STOPPED", reconnect_count=self._reconnect_count)
        logger.info("[%s] RTSP 流水线结束", self.camera_id)

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        # GPU 服务器：读帧速度远快于源视频帧率，需要限速避免空跑浪费 GPU 算力
        # 上限由 main.py 传入：CPU=15fps，GPU=30fps
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_fps = min(source_fps, self._frame_rate_cap)
        frame_interval = 1.0 / target_fps

        while not self._stop_event.is_set():
            t_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                break
            self._frame_idx += 1
            self._process_frame(frame)
            # 每500帧清理一次过期的离开记录，防止长时间运行内存泄漏
            if self._frame_idx % 500 == 0:
                self._cleanup_stale_leave_times()
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

    # ------------------------------------------------------------------ #
    # 核心处理逻辑（Phase 4：多帧验证 + 校准记录）                             #
    # ------------------------------------------------------------------ #

    def _process_frame(self, frame: np.ndarray) -> None:
        h_img, w_img = frame.shape[:2]

        # 绘制 ROI 危险区域边框
        for roi in self._rois:
            pts = np.array([[int(p[0] * w_img), int(p[1] * h_img)] for p in roi["polygon"]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            roi_label = f"[ROI] {roi.get('name', '危险区')}"
            cv2.putText(frame, roi_label, (pts[0][0][0], max(20, pts[0][0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # YOLO 跳帧检测：CPU 模式每 N 帧推理一次
        # 跳帧时传空列表让 Kalman 滤波器自主预测，避免旧检测误导 tracker
        if self._frame_idx % self._detect_every_n == 0:
            self._last_detections = self.detector.detect(frame)
            detections = self._last_detections
        else:
            detections = []  # 让 Kalman 预测，不喂旧坐标
        tracks = self._tracker.update(detections, frame.shape)
        current_track_ids = {t["track_id"] for t in tracks}

        left_ids = self._prev_track_ids - current_track_ids
        for tid in left_ids:
            self._on_person_leave(tid)

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
                gallery = self.store.get_gallery()
                confirmed_gid = self._validator.get_confirmed_match(tid, gallery, metrics=self.store.metrics)

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
                    if self._validator.buffer_len(tid) >= self._validator._buffer_size:
                        avg_feat = self._validator.get_avg_feature(tid)
                        if avg_feat is not None:
                            # 原子性查重+注册（P1-5）：防止多路并发重复注册同一人
                            gid, is_new = self.store.register_if_new(avg_feat)
                            # 通过公开接口确认身份，同时清理候选脏状态（P1-5）
                            self._validator.confirm(tid, gid)
                            if is_new:
                                logger.info("[%s] 注册新身份（多帧平均）: %s", self.camera_id, gid)
                            else:
                                logger.info("[%s] 跨摄归并身份: %s", self.camera_id, gid)

                if gid:
                    self._track_to_global[tid] = gid

            if gid:
                # 计算帧质量分（bbox 面积 × 检测置信度），指导特征滑动平均权重（P1-3）
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area_score = min(1.0, (w * h) / (128.0 * 256.0))
                quality = area_score * min(1.0, conf)
                self.store.update_appearance(gid, self.camera_id, feat, bbox, quality_score=quality)

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
                pts = np.array([[int(p[0] * w_img), int(p[1] * h_img)] for p in roi["polygon"]], np.int32)
                if any(cv2.pointPolygonTest(pts, pt, measureDist=False) >= 0 for pt in check_points):
                    is_intrusion = True
                    self.alerter.trigger_intrusion(self.camera_id, gid or f"Trk_{tid}", roi.get("name", "危险区域"), [x1, y1, x2, y2])
                    break

            # 计算质量分（bbox 面积 × 检测置信度）
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area_score = min(1.0, (w * h) / (128.0 * 256.0))
            quality = area_score * min(1.0, conf)

            # 如果侵入 ROI 用亮红色高亮框，否则已识别天蓝/未识别绿
            if is_intrusion:
                color = (0, 0, 239)
                label_text = f"🚨INTRUSION ID:{gid or f'Trk_{tid}'}"
            elif gid:
                color = (248, 189, 56)  # BGR 天蓝/亮黄
                label_text = f"ID:{gid} | Q:{quality:.2f}"
            else:
                color = (129, 185, 16)  # BGR 翡翠绿
                # P2: 用公开方法 buffer_len() 替代访问私有 _buffers
                buf_len = self._validator.buffer_len(tid)
                label_text = f"Trk:{tid} [{buf_len}/3] | Q:{quality:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_intrusion else 2)

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_y = max(th + 4, y1)
            cv2.rectangle(frame, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 2), color, -1)
            cv2.putText(frame, label_text, (x1 + 3, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        self._prev_track_ids = current_track_ids

    def _record_arrival(self, gid: str) -> None:
        """人员到达本摄像头时，计算并记录通行时间"""
        if self.calibrator is None:
            return
        leave_info = self._leave_times.pop(gid, None)
        if leave_info is None:
            return
        leave_time, leave_cam = leave_info
        actual_seconds = time.time() - leave_time
        self.calibrator.record_transit(leave_cam, self.camera_id, actual_seconds)

    def _on_person_leave(self, track_id: int) -> None:
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

        # 记录离开时间供到达时计算通行时间
        if self.calibrator:
            self._leave_times[gid] = (time.time(), self.camera_id)

        self.alerter.watch(
            global_id=gid,
            last_camera=self.camera_id,
            expected_cameras=expected_cams,
            deadline_offset=max_deadline,
            # 使用锁内安全接口读取 bbox，替代锁外裸访问 rec.appearances
            last_bbox=self.store.get_last_bbox(gid),
        )
        logger.debug(
            "[%s] Person %s 离开，预期在 %s 出现（校准后 %.0fs 内）",
            self.camera_id, gid, expected_cams, max_deadline,
        )

    def _cleanup_stale_leave_times(self) -> None:
        """清理超过 _LEAVE_EXPIRY_SECONDS 的离开记录，防止内存泄漏和校准数据污染"""
        now = time.time()
        stale = [
            gid for gid, (ts, _) in self._leave_times.items()
            if now - ts > _LEAVE_EXPIRY_SECONDS
        ]
        for gid in stale:
            self._leave_times.pop(gid, None)
        if stale:
            logger.debug("[%s] 清理 %d 条过期离开记录", self.camera_id, len(stale))
