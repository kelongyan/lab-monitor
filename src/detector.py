"""
detector.py — YOLOv8n 人员检测
只检测 person 类（class 0），返回 [x1, y1, x2, y2, conf] 列表
"""

import threading
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.4, device: str = "cpu"):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.device = device
        self._lock = threading.Lock()

        # 预热并提前触发模型 fuse，避免多线程并发 predict 时产生竞态条件
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """
        输入：BGR 图像帧
        返回：[[x1, y1, x2, y2, conf], ...] — 仅 person 类
        """
        with self._lock:
            results = self.model.predict(
                frame,
                classes=[0],          # person only
                conf=self.conf_thresh,
                device=self.device,
                verbose=False,
            )
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
        return detections
