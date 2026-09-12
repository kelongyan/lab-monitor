"""
reid.py — ReID 特征提取 + 跨摄像头余弦相似度匹配

提取器优先级：
  1. ReIDExtractorOSNet  — OSNet-x0.25 + **ReID 度量学习权重**（512 维，需 torchreid）
  2. ReIDExtractor       — ResNet50 ImageNet（通用回退，2048 维）

外部调用统一使用 build_reid_extractor() 工厂函数自动选择。

⚠️ 关于权重的重要说明
--------------------
torchreid 的 `build_model(pretrained=True)` **只加载 ImageNet 分类权重**
（`torchreid/reid/models/osnet.py:11-22` 的 `pretrained_urls` 全指向 imagenet），
分类嵌入空间不保证"同人近、异人远"。本项目此前一直是这个状态，实测后果是
990 对异人特征余弦中位 0.984、99.8% 越过匹配阈值 —— 身份识别实际不工作。

现在改为从 `src/reid_config.py` 的权重注册表加载 ReID 数据集权重
（默认 msmt17，可切 market1501），并用分类头维度校验数据集一致性。
阈值与权重选择也统一收拢在该模块，本文件不再出现写死的魔数。
"""

import logging
import threading
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2

from .reid_config import (
    ALLOW_IMAGENET_FALLBACK,
    IMAGENET_WEIGHT,
    REID_MATCH_THRESHOLD,
    REID_RATIO_TEST,
    ReIDWeight,
    ReIDWeightUnavailable,
    describe as describe_reid_config,
    get_reid_weight,
    load_reid_state_dict,
    resolve_reid_weight_path,
)

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
        self.feature_space = "resnet50-imagenet1k-v1:2048"
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

    加载哪份权重由 `src/reid_config.py` 的注册表决定：
      - `is_metric_learning=True` 的权重（market1501 / msmt17 / dukemtmc）从本地 .pth
        加载，并用分类头维度校验数据集一致性；
      - `IMAGENET_WEIGHT` 只能作为**显式降级**, 它不是 ReID 权重，
        构造时会打 ERROR —— 这条路径存在只是为了让监控链路不至于整机停摆。

    依赖：pip install torchreid tensorboard gdown
    """

    def __init__(self, device: str = "cpu", weight: ReIDWeight | None = None,
                 weight_key: str | None = None):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        import torchreid

        self.device = torch.device(device)
        if weight is None:
            weight, _ = resolve_reid_weight_path(weight_key)
        self.weight = weight
        self.feature_space = weight.feature_space

        if weight.is_metric_learning:
            # build_model 的 num_classes 必须与权重一致，否则分类头形状不符；
            # 虽然 eval 模式下分类头不参与前向，但形状对不上会让 load_state_dict
            # 报一堆无关的 missing，掩盖真正的问题。
            self.model = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=weight.num_classes,
                loss="softmax",
                pretrained=False,   # 关键：不走 torchreid 的 imagenet 下载
            )
            state = load_reid_state_dict(weight.path, weight)
            incompatible = self.model.load_state_dict(state, strict=False)
            backbone_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith("classifier")
            ]
            if backbone_missing:
                raise ReIDWeightUnavailable(
                    f"权重 {weight.file_name} 缺少骨架层: {backbone_missing[:5]}"
                    f"（共 {len(backbone_missing)} 个）—— 文件可能损坏或不是 OSNet-x0.25"
                )
            if incompatible.unexpected_keys:
                logger.warning(
                    "权重 %s 含 %d 个未使用张量: %s",
                    weight.file_name, len(incompatible.unexpected_keys),
                    incompatible.unexpected_keys[:5],
                )
            logger.info(
                "ReIDExtractorOSNet 就绪：weight=%s（%s）device=%s "
                "训练类别=%d 特征空间=%s",
                weight.key, weight.benchmark, device,
                weight.num_classes, self.feature_space,
            )
        else:
            # 降级路径：ImageNet 分类权重。用 torchreid 自带的下载/缓存机制。
            self.model = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=IMAGENET_WEIGHT.num_classes,
                loss="softmax",
                pretrained=True,
            )
            logger.error(
                "⚠️ ReID 降级为 ImageNet 分类权重（feature_space=%s）。"
                "这不是度量学习权重，跨相机身份比对不可信 —— "
                "检测/跟踪/围栏/告警不受影响，但『身份识别』与『以人搜视频』的结果"
                "不应采信。修复："
                "./.venv/Scripts/python.exe scripts/fetch_reid_weights.py",
                self.feature_space,
            )

        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

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


def build_reid_extractor(device: str = None, weight_key: str | None = None):
    """
    ReID 提取器工厂函数。自动检测 GPU / CUDA 设备（如 RTX 3090）。

    降级链（越往下越差，每一步都打日志，不静默）：
      1. OSNet-x0.25 + ReID 度量学习权重（目标路径）
      2. OSNet-x0.25 + ImageNet 分类权重（权重文件缺失且允许降级时）——
         **特征不可用于身份比对**，但检测/跟踪/告警链路不受影响
      3. ResNet50 ImageNet（torchreid 不可用等硬失败）—— 同上，且维度变 2048

    选择"降级而不是抛异常"是刻意的：本系统的主价值是实时监控与越界/失踪告警，
    这些不依赖 ReID。让整机因为身份特征降级而停摆是更坏的权衡
    （与 topology 校验失败降级为空拓扑同一口径）。
    需要严格模式时设 LAB_MONITOR_ALLOW_IMAGENET_FALLBACK=0。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("ReID 配置: %s", describe_reid_config())

    try:
        extractor = ReIDExtractorOSNet(device=device, weight_key=weight_key)
        logger.info(
            "已选用 ReIDExtractorOSNet（OSNet-x0.25 + %s 权重, 512维, device=%s）",
            extractor.weight.key, device,
        )
        return extractor
    except ReIDWeightUnavailable as error:
        logger.error("ReID 权重不可用：%s", error)
    except Exception as error:
        logger.warning("OSNet 初始化失败（%s），尝试降级", error)

    if ALLOW_IMAGENET_FALLBACK:
        try:
            extractor = ReIDExtractorOSNet(device=device, weight=IMAGENET_WEIGHT)
            logger.error(
                "已降级为 OSNet + ImageNet 分类权重（%s）。"
                "此模式下列用身份的检索匹配结果不可采信。",
                extractor.feature_space,
            )
            return extractor
        except Exception as error:
            logger.warning("ImageNet OSNet 降级也失败（%s），继续降级到 ResNet50", error)

    extractor = ReIDExtractor(device=device)
    logger.error(
        "已降级为 ResNet50 ImageNet（%s）。同上：身份比对不可采信。", extractor.feature_space
    )
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
    gallery: list[tuple[str, np.ndarray]],  # [(global_id, feat), ...]，同一 gid 可多行
    threshold: float = REID_MATCH_THRESHOLD,
    ratio: float = REID_RATIO_TEST,   # Ratio Test：second/best > ratio 时视为歧义拒绝
) -> MatchDetail:
    """
    在 gallery 中找与 query 最相似的身份，返回详细匹配元数据 MatchDetail。

    默认阈值来自 `src/reid_config.py`（可用 LAB_MONITOR_REID_THRESHOLD / _RATIO 覆盖），
    不在本文件写死数字 —— 此前 0.75 散落 6 处，改一处不生效。

    **按身份去重**（重要）
    --------------------
    同一个 global_id 允许多行：主特征 + `feature_bank` 里的若干个姿态特征。
    先按身份取相似度最大值，再做阈值与 Ratio 判定。这是必需的，原因有两条：

    1. 不去重时同一身份会同时占据 best 与 second 两个位次，
       Ratio Test（second/best > ratio）必然把真实命中判成"歧义"，
       调用方于是新建一个身份 —— 这正是身份表不断膨胀的直接机制。
    2. 主特征是 EMA 滑动平均，会指数抹平身份特异残差
       （见 `scripts/diagnose_ema_collapse.py`：同一人反复观测 20 次后，
       跨身份余弦 p50 从 0.50 升到 0.73；库内更是达到 0.98）。
       `feature_bank` 保存的是**原始**特征且带多样性约束（<0.92 才入池），
       取 max 可以绕开 EMA 塌缩。主匹配路径必须把 bank 用起来。

    单身份时跳过 Ratio Test 的判定按**去重后的身份数**计算，不是按行数。
    """
    if not gallery:
        return MatchDetail(matched_id=None)

    ids = [gid for gid, _ in gallery]
    feats = np.stack([f for _, f in gallery])  # (N, D)
    sims = feats @ query                        # (N,)

    best_per_id: dict[str, float] = {}
    for gid, sim in zip(ids, sims):
        value = float(sim)
        if value > best_per_id.get(gid, -2.0):
            best_per_id[gid] = value

    # 单身份时无需 Ratio Test
    if len(best_per_id) == 1:
        only_id, best_sim = next(iter(best_per_id.items()))
        if best_sim >= threshold:
            return MatchDetail(matched_id=only_id, best_sim=best_sim)
        return MatchDetail(matched_id=None, best_sim=best_sim)

    ranked = sorted(best_per_id.items(), key=lambda item: item[1], reverse=True)
    best_id, best_sim = ranked[0]
    second_sim = ranked[1][1]

    if best_sim < threshold:
        return MatchDetail(matched_id=None, best_sim=best_sim, second_sim=second_sim)

    # Ratio Test：若第二名相似度与最佳相似度过于接近，说明有歧义，拒绝匹配
    if best_sim > 0 and second_sim / best_sim > ratio:
        return MatchDetail(matched_id=None, best_sim=best_sim, second_sim=second_sim, is_ratio_blocked=True)

    return MatchDetail(matched_id=best_id, best_sim=best_sim, second_sim=second_sim)


def match_feature(
    query: np.ndarray,
    gallery: list[tuple[str, np.ndarray]],  # [(global_id, feat), ...]
    threshold: float = REID_MATCH_THRESHOLD,
    ratio: float = REID_RATIO_TEST,   # Ratio Test：second/best > ratio 时视为歧义拒绝
) -> str | None:
    """
    在 gallery 中找与 query 最相似的身份，并通过 Ratio Test 过滤歧义匹配。
    返回 global_id 或 None（无匹配 / 歧义）
    """
    detail = match_feature_detailed(query, gallery, threshold=threshold, ratio=ratio)
    return detail.matched_id
