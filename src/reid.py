"""
reid.py — ResNet50 ReID 特征提取 + 跨摄像头余弦相似度匹配
"""

import threading
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
from scipy.spatial.distance import cosine


# 图像预处理（与 ImageNet 训练一致）
_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ReIDExtractor:
    """用 ResNet50 提取人员外观特征向量（2048维）"""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 去掉最后的分类头，只保留特征提取部分
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        """
        从帧中裁剪人员区域，提取特征向量
        bbox: [x1, y1, x2, y2]
        返回: np.ndarray shape (2048,)，裁剪区域太小时返回 None
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        # OpenCV BGR → RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(crop_rgb).unsqueeze(0).to(self.device)
        with self._lock:
            feat = self.model(tensor).squeeze().cpu().numpy()  # (2048,)
        # L2 归一化，便于余弦计算
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return None
        return feat / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """返回余弦相似度 [0, 1]，值越高越相似"""
    return float(1.0 - cosine(a, b))


def match_feature(
    query: np.ndarray,
    gallery: list[tuple[str, np.ndarray]],  # [(global_id, feat), ...]
    threshold: float = 0.75,
) -> str | None:
    """
    在 gallery 中找与 query 最相似的身份
    返回 global_id 或 None（无匹配）
    """
    if not gallery:
        return None
    sims = [(gid, cosine_similarity(query, feat)) for gid, feat in gallery]
    best_id, best_sim = max(sims, key=lambda x: x[1])
    return best_id if best_sim >= threshold else None
