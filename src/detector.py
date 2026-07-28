"""
detector.py — YOLOv8n 人员检测
只检测 person 类（class 0），返回 [x1, y1, x2, y2, conf] 列表
"""

import threading
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.4, device: str = None):
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self._lock = threading.Lock()

        # 如果运行在 CUDA 显卡（如 RTX 3090）上，使用 FP16 自动半精度加速
        self.use_fp16 = (self.device == "cuda" or (isinstance(self.device, str) and "cuda" in self.device))

        # 预热并提前触发模型 fuse，避免多线程并发 predict 时产生竞态条件
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False, half=self.use_fp16)

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """
        输入：BGR 图像帧
        返回：[[x1, y1, x2, y2, conf], ...] — 仅 person 类
        """
        kwargs = {
            "classes": [0],
            "conf": self.conf_thresh,
            "device": self.device,
            "verbose": False,
        }
        if self.use_fp16:
            kwargs["half"] = True

        with self._lock:
            results = self.model.predict(frame, **kwargs)
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
