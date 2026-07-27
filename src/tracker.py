"""
tracker.py — ByteTrack 单路跟踪（基于 ultralytics 内置实现）
输入：检测框列表；输出：带 track_id 的轨迹列表
"""

import numpy as np
from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


class Detections:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4) if len(xyxy) > 0 else np.empty((0, 4), dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        if len(self.xyxy) > 0:
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            xc = self.xyxy[:, 0] + w / 2.0
            yc = self.xyxy[:, 1] + h / 2.0
            self.xywh = np.stack([xc, yc, w, h], axis=-1)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)

    def __getitem__(self, idx):
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])

    def __len__(self):
        return len(self.xyxy)


def _default_args(fps: int = 25) -> IterableSimpleNamespace:
    return IterableSimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        frame_rate=fps,
    )


class PersonTracker:
    def __init__(self, fps: int = 25):
        self.tracker = BYTETracker(_default_args(fps))

    def update(self, detections: list[list[float]], frame_shape: tuple) -> list[dict]:
        """
        detections: [[x1,y1,x2,y2,conf], ...]
        frame_shape: (H, W, C)
        返回: [{"track_id": int, "bbox": [x1,y1,x2,y2], "conf": float}, ...]
        """
        if not detections:
            det_objs = Detections([], [], [])
        else:
            det_arr = np.array(detections, dtype=float)
            xyxy = det_arr[:, :4]
            conf = det_arr[:, 4]
            cls = np.zeros(len(det_arr), dtype=float)
            det_objs = Detections(xyxy, conf, cls)

        tracks = self.tracker.update(det_objs)

        results = []
        if len(tracks) > 0:
            for t in tracks:
                if len(t) >= 6:
                    x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                    track_id = int(t[4])
                    score = float(t[5])
                    results.append({
                        "track_id": track_id,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": score,
                    })
        return results
