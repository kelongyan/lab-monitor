"""
frame_hub.py — 线程安全的多路摄像头帧缓存
pipeline 推入最新帧，web server 拉取用于 MJPEG 流和大屏展示

GPU 服务器优化：
- 每路摄像头独立锁（per-camera lock），消除多路并发串行瓶颈
- JPEG 编码质量提升至 85（服务器带宽充裕，清晰度优先）
"""

import threading
import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class CameraState:
    camera_id: str
    latest_frame: np.ndarray | None = None
    latest_jpeg: bytes | None = None
    frame_time: float = 0.0
    is_online: bool = False
    status_text: str = "OFFLINE"
    reconnect_count: int = 0
    fps: float = 0.0
    generation: int = 0
    _frame_times: deque = field(default_factory=lambda: deque(maxlen=30))


class FrameHub:
    """
    所有摄像头的帧缓存中心
    - pipeline 调用 push_frame() 推入最新帧
    - server 调用 get_jpeg() 获取 JPEG 字节用于 MJPEG 流
    """

    # Web 展示用的缩放尺寸
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 360
    # GPU 服务器：提高 JPEG 编码质量（带宽充裕，画面更清晰）
    JPEG_QUALITY = 85

    def __init__(self, jpeg_quality: int = None, display_width: int = 640, display_height: int = 360):
        # jpeg_quality 参数优先，否则用类默认值（CPU=65，GPU=85）
        if jpeg_quality is not None:
            self.JPEG_QUALITY = jpeg_quality
        self.DISPLAY_WIDTH  = display_width
        self.DISPLAY_HEIGHT = display_height
        # _global_lock 仅保护 _states/_cam_locks 字典结构（新增/删除 key）
        # 读写具体摄像头数据时使用各自的 per-camera lock
        self._global_lock = threading.Lock()
        self._cam_locks: dict[str, threading.Lock] = {}
        self._states: dict[str, CameraState] = {}

    def _ensure_camera(self, camera_id: str) -> tuple[CameraState, threading.Lock]:
        """获取或初始化摄像头状态与独立锁（double-check 避免重复创建）"""
        lock = self._cam_locks.get(camera_id)
        if lock is None:
            with self._global_lock:
                if camera_id not in self._cam_locks:
                    self._cam_locks[camera_id] = threading.Lock()
                    self._states[camera_id] = CameraState(camera_id=camera_id)
                lock = self._cam_locks[camera_id]
        return self._states[camera_id], lock

    def register_camera(self, camera_id: str) -> None:
        self._ensure_camera(camera_id)

    def register_cameras(self, camera_ids) -> None:
        for camera_id in camera_ids:
            self.register_camera(camera_id)

    # ------------------------------------------------------------------ #
    # Pipeline → Hub                                                        #
    # ------------------------------------------------------------------ #

    def push_frame(self, camera_id: str, frame: np.ndarray, status_text: str = "ONLINE") -> None:
        now = time.time()
        s, cam_lock = self._ensure_camera(camera_id)
        with cam_lock:
            s.latest_frame = frame
            s.latest_jpeg = None  # 标记新帧到来，清除旧 JPEG 编码缓存
            s.generation += 1
            s.frame_time = now
            s.is_online = True
            s.status_text = status_text
            s._frame_times.append(now)
            # 计算实时 FPS（最近30帧）
            if len(s._frame_times) >= 2:
                elapsed = s._frame_times[-1] - s._frame_times[0]
                s.fps = round((len(s._frame_times) - 1) / max(elapsed, 1e-6), 1)

    def mark_offline(self, camera_id: str, status_text: str = "OFFLINE", reconnect_count: int = 0) -> None:
        s, cam_lock = self._ensure_camera(camera_id)
        with cam_lock:
            s.is_online = False
            s.status_text = status_text
            s.reconnect_count = reconnect_count
            s.latest_frame = None
            s.latest_jpeg = None
            s.generation += 1
            s.fps = 0.0
            s._frame_times.clear()

    # ------------------------------------------------------------------ #
    # Hub → Server                                                          #
    # ------------------------------------------------------------------ #

    def get_jpeg(self, camera_id: str, quality: int = None) -> bytes | None:
        """返回最新帧的 JPEG 字节（带编码缓存），用于 MJPEG 流；摄像头离线返回 None"""
        if quality is None:
            quality = self.JPEG_QUALITY

        lock = self._cam_locks.get(camera_id)
        if lock is None:
            return None

        # 编码期间若有新帧到达，最多重试两次，绝不把旧 generation 写回缓存。
        for _ in range(3):
            with lock:
                s = self._states.get(camera_id)
                if s is None or not s.is_online or s.latest_frame is None:
                    return None
                if s.latest_jpeg is not None:
                    return s.latest_jpeg
                frame = s.latest_frame.copy()
                generation = s.generation

            h, w = frame.shape[:2]
            if w != self.DISPLAY_WIDTH or h != self.DISPLAY_HEIGHT:
                frame = cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))

            success, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if not success:
                return None
            jpeg_bytes = buf.tobytes()

            with lock:
                s = self._states.get(camera_id)
                if s is None or not s.is_online:
                    return None
                if s.generation == generation:
                    s.latest_jpeg = jpeg_bytes
                    return jpeg_bytes
        return None

    def get_frame(self, camera_id: str) -> np.ndarray | None:
        """返回在线 camera 最新帧的副本，供告警快照使用。"""
        lock = self._cam_locks.get(camera_id)
        if lock is None:
            return None
        with lock:
            state = self._states.get(camera_id)
            if state is None or not state.is_online or state.latest_frame is None:
                return None
            return state.latest_frame.copy()

    def get_status(self) -> list[dict]:
        """返回所有摄像头状态，供 API 接口用"""
        with self._global_lock:
            cam_ids = list(self._states.keys())

        result = []
        now = time.time()
        for cam_id in cam_ids:
            lock = self._cam_locks.get(cam_id)
            if lock is None:
                continue
            with lock:
                s = self._states.get(cam_id)
                if s is None:
                    continue
                result.append({
                    "camera_id": s.camera_id,
                    "is_online": s.is_online,
                    "status_text": s.status_text,
                    "reconnect_count": s.reconnect_count,
                    "fps": s.fps,
                    "last_frame_time": s.frame_time,
                    "frame_age_seconds": (
                        round(max(0.0, now - s.frame_time), 3)
                        if s.frame_time > 0
                        else None
                    ),
                    "display_width": self.DISPLAY_WIDTH,
                    "display_height": self.DISPLAY_HEIGHT,
                    "jpeg_quality": self.JPEG_QUALITY,
                })
        return result

    def camera_ids(self) -> list[str]:
        with self._global_lock:
            return list(self._states.keys())
