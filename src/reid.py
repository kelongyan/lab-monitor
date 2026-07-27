"""
reid.py — ReID 特征提取 + 跨摄像头余弦相似度匹配

提取器优先级：
  1. ReIDExtractorOSNet  — OSNet-x0.25（专用 ReID 训练，512 维，需 torchreid）
  2. ReIDExtractor       — ResNet50 ImageNet（通用回退，2048 维）

外部调用统一使用 build_reid_extractor() 工厂函数自动选择。
"""

import logging
import threading
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2

logger = logging.getLogger("reid")


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
        try:
            with self._lock:
                feat = self.model(tensor).squeeze().cpu().numpy()  # (2048,)
        except RuntimeError as e:
            import logging
            logging.getLogger("reid").warning("ReID 推理失败（可能 OOM）: %s", e)
            return None
        # L2 归一化，便于余弦计算
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return None
        return feat / norm


class ReIDExtractorOSNet:
    """
    用 OSNet-x0.25 提取人员 ReID 特征向量（512 维）。
    OSNet 专为 Re-Identification 设计，在 Market-1501 上 Rank-1 约 78%，
    显著优于 ImageNet 预训练的 ResNet50（约 45%）。
    依赖：pip install torchreid tensorboard gdown
    """

    def __init__(self, device: str = "cpu"):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        import torchreid

        self.device = torch.device(device)
        # eval 模式下 OSNet.forward() 直接返回 512 维特征向量（已内置 BN + GAP）
        self.model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1000,
            loss="softmax",
            pretrained=True,   # 下载 ImageNet 预训练权重（首次运行需联网）
        )
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()
        logger.info("ReIDExtractorOSNet 初始化完成（device=%s, feat_dim=512）", device)

    @torch.no_grad()
    def extract(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        """
        从帧中裁剪人员区域，提取 512 维 OSNet 特征向量。
        bbox: [x1, y1, x2, y2]，返回 L2 归一化后的 np.ndarray(512,) 或 None
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(crop_rgb).unsqueeze(0).to(self.device)
        try:
            with self._lock:
                # eval 模式下 OSNet 直接返回特征向量，不经过分类头
                feat = self.model(tensor).squeeze().cpu().numpy()  # (512,)
        except RuntimeError as e:
            logger.warning("OSNet 推理失败（可能 OOM）: %s", e)
            return None
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return None
        return feat / norm


def build_reid_extractor(device: str = None):
    """
    ReID 提取器工厂函数：优先使用 OSNet（精度高），若依赖不可用则回退 ResNet50。
    自动检测 GPU / CUDA 设备（如 RTX 3090）。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        extractor = ReIDExtractorOSNet(device=device)
        logger.info("已选用 ReIDExtractorOSNet（OSNet-x0.25, 512维, device=%s）", device)
        return extractor
    except Exception as e:
        logger.warning("OSNet 初始化失败（%s），回退到 ResNet50", e)
        extractor = ReIDExtractor(device=device)
        logger.info("已选用 ReIDExtractor（ResNet50, 2048维, device=%s）", device)
        return extractor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """返回余弦相似度 [0, 1]；两向量已 L2 归一化时等价于点积，无需 scipy"""
    return float(np.dot(a, b))


from dataclasses import dataclass


@dataclass
class MatchDetail:
    matched_id: str | None
    best_sim: float = 0.0
    second_sim: float = 0.0
    is_ratio_blocked: bool = False


def match_feature_detailed(
    query: np.ndarray,
    gallery: list[tuple[str, np.ndarray]],  # [(global_id, feat), ...]
    threshold: float = 0.75,
    ratio: float = 0.85,   # Ratio Test 阈值：second/best > ratio 时视为歧义拒绝
) -> MatchDetail:
    """
    在 gallery 中找与 query 最相似的身份，返回详细匹配元数据 MatchDetail。
    """
    if not gallery:
        return MatchDetail(matched_id=None)

    ids = [gid for gid, _ in gallery]
    feats = np.stack([f for _, f in gallery])  # (N, D)
    sims = feats @ query                        # (N,)

    # 单身份时无需 Ratio Test
    if len(gallery) == 1:
        best_sim = float(sims[0])
        if best_sim >= threshold:
            return MatchDetail(matched_id=ids[0], best_sim=best_sim)
        return MatchDetail(matched_id=None, best_sim=best_sim)

    idx = np.argsort(sims)[::-1]
    best_sim = float(sims[idx[0]])
    second_sim = float(sims[idx[1]])

    if best_sim < threshold:
        return MatchDetail(matched_id=None, best_sim=best_sim, second_sim=second_sim)

    # Ratio Test：若第二名相似度与最佳相似度过于接近，说明有歧义，拒绝匹配
    if best_sim > 0 and second_sim / best_sim > ratio:
        return MatchDetail(matched_id=None, best_sim=best_sim, second_sim=second_sim, is_ratio_blocked=True)

    return MatchDetail(matched_id=ids[idx[0]], best_sim=best_sim, second_sim=second_sim)


def match_feature(
    query: np.ndarray,
    gallery: list[tuple[str, np.ndarray]],  # [(global_id, feat), ...]
    threshold: float = 0.75,
    ratio: float = 0.85,   # Ratio Test 阈值：second/best > ratio 时视为歧义拒绝
) -> str | None:
    """
    在 gallery 中找与 query 最相似的身份，并通过 Ratio Test 过滤歧义匹配。
    返回 global_id 或 None（无匹配 / 歧义）
    """
    detail = match_feature_detailed(query, gallery, threshold=threshold, ratio=ratio)
    return detail.matched_id

