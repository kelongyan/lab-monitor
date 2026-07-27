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

        # Phase 4: 多帧 ReID 确认器（每路摄像头独立）
        self._validator = ReIDValidator(
            buffer_size=8,
            confirm_frames=3,
            threshold=0.75,
        )

        # Phase 4: 记录人员离开时间戳，供校准器计算通行时间
        # global_id → (leave_time, leave_camera)
        self._leave_times: dict[str, tuple[float, str]] = {}

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
                    self.frame_hub.mark_offline(self.camera_id)
                return
            logger.info("[%s] 开始处理文件: %s", self.camera_id, self.source)
            self._read_loop(cap)
            cap.release()
            time.sleep(0.5)
        if self.frame_hub:
            self.frame_hub.mark_offline(self.camera_id)
        logger.info("[%s] 文件流水线结束", self.camera_id)

    def _run_rtsp(self) -> None:
        """RTSP 流处理：无限重连+指数退避，消除网络抖动导致的永久下线"""
        retry_delay = _RTSP_INITIAL_DELAY
        while not self._stop_event.is_set():
            logger.info("[%s] 连接 RTSP: %s", self.camera_id, self.source)
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.warning("[%s] RTSP 连接失败，%.1fs 后重试", self.camera_id, retry_delay)
                if self.frame_hub:
                    self.frame_hub.mark_offline(self.camera_id)
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
            logger.warning("[%s] RTSP 流中断，%.1fs 后重连", self.camera_id, retry_delay)
            if self.frame_hub:
                self.frame_hub.mark_offline(self.camera_id)
            self._stop_event.wait(retry_delay)

        if self.frame_hub:
            self.frame_hub.mark_offline(self.camera_id)
        logger.info("[%s] RTSP 流水线结束", self.camera_id)

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            self._frame_idx += 1
            self._process_frame(frame)
            # 每500帧清理一次过期的离开记录，防止长时间运行内存泄漏
            if self._frame_idx % 500 == 0:
                self._cleanup_stale_leave_times()
            if self.frame_hub:
                self.frame_hub.push_frame(self.camera_id, frame)
            if self.display:
                cv2.imshow(f"Camera {self.camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop_event.set()
                    break
        if self.display:
            cv2.destroyWindow(f"Camera {self.camera_id}")

    # ------------------------------------------------------------------ #
    # 核心处理逻辑（Phase 4：多帧验证 + 校准记录）                             #
    # ------------------------------------------------------------------ #

    def _process_frame(self, frame: np.ndarray) -> None:
        detections = self.detector.detect(frame)
        tracks = self._tracker.update(detections, frame.shape)
        current_track_ids = {t["track_id"] for t in tracks}

        left_ids = self._prev_track_ids - current_track_ids
        for tid in left_ids:
            self._on_person_leave(tid)

        for track in tracks:
            tid = track["track_id"]
            bbox = track["bbox"]
            conf = track["conf"]

            # 限频提取特征
            last_reid = self._reid_frame_counter.get(tid, -999)
            if self._frame_idx - last_reid < self.REID_EVERY_N_FRAMES:
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
                confirmed_gid = self._validator.get_confirmed_match(tid, gallery)

                if confirmed_gid:
                    gid = confirmed_gid
                    # 记录通行时间（校准用）
                    self._record_arrival(gid)
                    # 解除预警监听
                    self.alerter.resolve(gid, self.camera_id)
                    logger.info("[%s] ✓ 身份确认（多帧）: %s", self.camera_id, gid)
                else:
                    # 尚未确认，但缓冲帧够了且完全无匹配 → 注册（或归并）身份
                    if len(self._validator._buffers.get(tid, [])) >= self._validator._buffer_size:
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

        # 绘制定位框和身份/追踪 ID 标签
        for track in tracks:
            tid = track["track_id"]
            x1, y1, x2, y2 = map(int, track["bbox"])
            conf = track["conf"]
            gid = self._track_to_global.get(tid)

            # 已识别出身份的用天蓝色，未识别确认的用绿色
            color = (248, 189, 56) if gid else (129, 185, 16)  # BGR 格式
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"ID:{gid}" if gid else f"Trk:{tid} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(th + 4, y1)
            cv2.rectangle(frame, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 2), color, -1)
            cv2.putText(frame, label_text, (x1 + 3, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

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

        rec = self.store.get(gid)
        if rec is None:
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
            # P2-4：先拍快照再访问，规避并发修改时的 IndexError
            last_bbox=list(rec.appearances)[-1]["bbox"] if rec.appearances else [],
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
