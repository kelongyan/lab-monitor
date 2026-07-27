"""
frame_hub.py — 线程安全的多路摄像头帧缓存
pipeline 推入最新帧，web server 拉取用于 MJPEG 流和大屏展示
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
    frame_time: float = 0.0
    is_online: bool = False
    fps: float = 0.0
    _frame_times: deque = field(default_factory=lambda: deque(maxlen=30))


class FrameHub:
    """
    所有摄像头的帧缓存中心
    - pipeline 调用 push_frame() 推入最新帧
    - server 调用 get_jpeg() 获取 JPEG 字节用于 MJPEG 流
    """

    # Web 展示用的缩放尺寸（降低带宽 + CPU）
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 360

    def __init__(self):
        self._lock = threading.Lock()
        self._states: dict[str, CameraState] = {}

    # ------------------------------------------------------------------ #
    # Pipeline → Hub                                                        #
    # ------------------------------------------------------------------ #

    def push_frame(self, camera_id: str, frame: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            if camera_id not in self._states:
                self._states[camera_id] = CameraState(camera_id=camera_id)
            s = self._states[camera_id]
            s.latest_frame = frame
            s.frame_time = now
            s.is_online = True
            s._frame_times.append(now)
            # 计算实时 FPS（最近30帧）
            if len(s._frame_times) >= 2:
                elapsed = s._frame_times[-1] - s._frame_times[0]
                s.fps = round((len(s._frame_times) - 1) / max(elapsed, 1e-6), 1)

    def mark_offline(self, camera_id: str) -> None:
        with self._lock:
            if camera_id in self._states:
                self._states[camera_id].is_online = False

    # ------------------------------------------------------------------ #
    # Hub → Server                                                          #
    # ------------------------------------------------------------------ #

    def get_jpeg(self, camera_id: str, quality: int = 70) -> bytes | None:
        """返回最新帧的 JPEG 字节，用于 MJPEG 流；摄像头离线返回 None"""
        with self._lock:
            s = self._states.get(camera_id)
            if s is None or s.latest_frame is None:
                return None
            frame = s.latest_frame.copy()

        # 缩放到展示分辨率
        h, w = frame.shape[:2]
        if w != self.DISPLAY_WIDTH:
            frame = cv2.resize(frame, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def get_status(self) -> list[dict]:
        """返回所有摄像头状态，供 API 接口用"""
        with self._lock:
            result = []
            for s in self._states.values():
                result.append({
                    "camera_id": s.camera_id,
                    "is_online": s.is_online,
                    "fps": s.fps,
                    "last_frame_time": s.frame_time,
                })
        return result

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._states.keys())
