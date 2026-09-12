"""
reid_config.py — ReID 匹配参数与权重选择的唯一来源。

为什么要这个模块
----------------
`0.75` 这个匹配阈值此前散落在 6 个调用点：`reid.py:171`、`reid.py:208`、
`reid_validator.py:31`、`identity_store.py:371`、`pipeline.py:138`、`pipeline.py:238`
（`CLAUDE.md` 只记了前 5 处且行号有误）。改一处不生效，导致"调阈值"这件事
在工程上不可执行。

更根本的问题是它从未被标定过。实测（`scripts/probe_reid_separability.py`）：
45 个身份、990 对**异人**特征余弦 mean 0.950 / p50 0.984，**988 对（99.8%）越过 0.75**；
全体均值向量范数 0.9754、去均值残差范数仅 0.166。根因是 `reid.py` 用
`pretrained=True` 加载的其实是 **ImageNet 分类权重**（torchreid 的
`pretrained_urls` 全指向 imagenet，见 `torchreid/reid/models/osnet.py:11-22`），
分类嵌入空间不保证"同人近、异人远"。

本模块提供
----------
1. `REID_MATCH_THRESHOLD` / `REID_RATIO_TEST`：单一常量，支持环境变量覆盖
2. 权重注册表：数据集 → 文件名 → `feature_space` 字符串
3. `resolve_reid_weight_path()`：定位权重，缺失时给出可直接执行的修复指引

依赖约束
--------
**只依赖标准库**。它被 `reid_validator` / `identity_store` 在热路径上导入，
不能把 torch / cv2 这类重依赖拖进导入链。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("reid_config")

# --------------------------------------------------------------------------- #
# 匹配阈值                                                                      #
# --------------------------------------------------------------------------- #

# 未标定前的保守默认值。**这不是一个"已验证"的值**，只是保持与历史行为一致，
# 避免换权重的同时又改阈值、两个变量纠缠在一起无法归因。
# 真实工作点由 scripts/tune_reid_threshold.py 在标注集上扫出来后回填。
#
# 2026-09-12 已完成首次标定（scripts/build_label_set.py 自动标注 + tune_reid_threshold.py）：
#   正样本 = 同相机相邻采样帧(0.4s) IoU>=0.25 的检测框（运动学约束，无需人工）；
#   负样本 = 同帧不重叠框 + 重建库 22 身份的跨身份对（226 对，生产语义）。
#   在负样本误报率 <= 5% 的约束下 F1 最大 → **阈值 0.68**（msmt17 与 market1501 相同）。
#   注意方向与直觉相反：是**下调**而非上调 —— 中心化把异人分布的 p95 压到 0.673，
#   0.75 会漏掉大量同人匹配（同人对 p50 只有 0.69~0.75）。
#   指标：market1501 在 0.68 处 P=0.913 / R=0.639 / F1=0.752；msmt17 P=0.899 / R=0.544 / F1=0.678。
_DEFAULT_THRESHOLD = 0.68
_DEFAULT_RATIO = 0.85


def _env_unit_float(name: str, default: float) -> float:
    """
    读取一个 (0, 1) 开区间内的浮点环境变量。

    取值非法只回落到默认值并打 warning，绝不抛异常 —— 与 `main.py:_env_positive`
    同一口径：调参写错不该让服务起不来。开区间是刻意的，余弦相似度阈值取 0 或 1
    没有任何物理意义（0 等于全放行，1 等于全拒绝）。
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r 不是合法数字，改用默认值 %s", name, raw, default)
        return default
    if not 0.0 < value < 1.0:
        logger.warning("%s=%s 必须在 (0, 1) 开区间内，改用默认值 %s", name, value, default)
        return default
    return value


REID_MATCH_THRESHOLD: float = _env_unit_float(
    "LAB_MONITOR_REID_THRESHOLD", _DEFAULT_THRESHOLD
)
REID_RATIO_TEST: float = _env_unit_float("LAB_MONITOR_REID_RATIO", _DEFAULT_RATIO)


# --------------------------------------------------------------------------- #
# 权重注册表                                                                    #
# --------------------------------------------------------------------------- #

CHECKPOINT_DIR = Path.home() / ".cache" / "torch" / "checkpoints"
FEATURE_DIM = 512


@dataclass(frozen=True)
class ReIDWeight:
    key: str
    file_name: str
    dataset: str
    num_classes: int
    #: 该数据集上的公开指标，用于在日志里说明"这份权重是什么水平"
    benchmark: str
    #: 是否是 ReID 度量学习权重（ImageNet 分类权重为 False —— 它不适合做身份比对）
    is_metric_learning: bool = True
    #: torchreid 官方 MODEL_ZOO 的 Google Drive 文件 id；空串表示不适用
    drive_id: str = ""

    @property
    def feature_space(self) -> str:
        """
        特征空间标识。会被 `IdentityStore._restore()` 做**全等比较**来决定
        历史身份是否可用，所以它必须准确反映"产生这批特征的权重"。

        换权重后这个字符串变化 → 旧身份自动被跳过，不需要手工清库，
        但启动日志必须显式说明跳过了多少个（否则会被误判为数据丢失）。
        """
        return f"osnet-x0.25-{self.dataset}:{FEATURE_DIM}"

    @property
    def path(self) -> Path:
        return CHECKPOINT_DIR / self.file_name


#: 可选的 ReID 度量学习权重。公开指标取自 torchreid `docs/MODEL_ZOO.md` 的
#: Same-domain ReID 表（osnet_x0_25 行，rank-1 / mAP）。
#:
#: **2026-09-12 已实测选定 market1501**。在自动标注集上
#: （`scripts/build_label_set.py`：同相机相邻采样帧 IoU≥0.25 = 同一人；
#:   重建库 22 身份的跨身份对 = 异人），负样本误报率 ≤ 5% 约束下：
#:     market1501  阈值 0.68  P=0.913  R=0.639  F1=0.752
#:     msmt17      阈值 0.68  P=0.899  R=0.544  F1=0.678
#: 相同阈值与误报率下 market1501 召回明显更高。公开指标里"msmt17 域内数字低
#: 是因为数据集更难"的说法在本项目上不成立 —— 本项目是固定室内走廊，
#: 与 Market1501（校园街道式取景）成像风格更接近。
#: 明细见 `outputs/reports/reid_threshold_sweep.json`。
REID_WEIGHTS: dict[str, ReIDWeight] = {
    "market1501": ReIDWeight(
        key="market1501",
        file_name="osnet_x0_25_market1501.pth",
        dataset="market1501",
        num_classes=751,
        benchmark="Market1501 Rank-1 91.2 / mAP 75.0（本项目实测 F1=0.752，优于 msmt17）",
        drive_id="1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj",
    ),
    "msmt17": ReIDWeight(
        key="msmt17",
        file_name="osnet_x0_25_msmt17.pth",
        dataset="msmt17",
        num_classes=1041,
        benchmark="MSMT17 Rank-1 61.4 / mAP 29.5（域内；本项目实测 F1=0.678）",
        drive_id="1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    ),
    "dukemtmc": ReIDWeight(
        key="dukemtmc",
        file_name="osnet_x0_25_dukemtmcreid.pth",
        dataset="dukemtmcreid",
        num_classes=702,
        benchmark="DukeMTMC Rank-1 82.0 / mAP 61.4",
        drive_id="1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l",
    ),
}

#: ImageNet 分类权重。**不是** ReID 权重，只作为权重缺失时的显式降级目标。
#: 保留它是因为"监控可用性 > 身份识别准确性"—— 检测/跟踪/围栏/告警不依赖 ReID，
#: 不该因为身份特征降级就整机停摆（与 topology 校验失败降级为空拓扑同一口径）。
IMAGENET_WEIGHT = ReIDWeight(
    key="imagenet",
    file_name="osnet_x0_25_imagenet.pth",
    dataset="imagenet",
    num_classes=1000,
    benchmark="ImageNet 分类，非 ReID 权重",
    is_metric_learning=False,
    drive_id="1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
)

#: 分类头维度 → 数据集。这是判定"权重训在哪个数据集上"唯一可靠的证据：
#: 它等于该数据集的身份数，参数量与张量数（567 个 / 0.59~0.74M）三项高度相似，
#: 区分不出 market1501 与 msmt17。
DATASET_BY_CLASSES: dict[int, str] = {
    weight.num_classes: weight.dataset
    for weight in (*REID_WEIGHTS.values(), IMAGENET_WEIGHT)
}

DEFAULT_REID_WEIGHTS: str = os.getenv("LAB_MONITOR_REID_WEIGHTS", "market1501").strip().lower()
ALLOW_IMAGENET_FALLBACK: bool = os.getenv(
    "LAB_MONITOR_ALLOW_IMAGENET_FALLBACK", "1"
).strip().lower() not in {"0", "false", "no"}


class ReIDWeightUnavailable(RuntimeError):
    """配置的 ReID 权重不可用（文件缺失 / 内容不是 ReID 权重）。"""


def get_reid_weight(key: str | None = None) -> ReIDWeight:
    """按 key 取权重定义；未知 key 回落默认并打 warning。"""
    name = (key or DEFAULT_REID_WEIGHTS or "msmt17").strip().lower()
    weight = REID_WEIGHTS.get(name)
    if weight is None:
        logger.warning(
            "未知的 ReID 权重 %r（可选：%s），改用 %s",
            name, "/".join(REID_WEIGHTS), DEFAULT_REID_WEIGHTS,
        )
        weight = REID_WEIGHTS.get(DEFAULT_REID_WEIGHTS) or REID_WEIGHTS["market1501"]
    return weight


def resolve_reid_weight_path(key: str | None = None) -> tuple[ReIDWeight, Path]:
    """
    定位权重文件。缺失或内容异常时抛 `ReIDWeightUnavailable`，
    异常消息里带**可直接执行的修复命令**，避免排查时再去翻文档。
    """
    weight = get_reid_weight(key)
    path = weight.path
    if not path.exists():
        raise ReIDWeightUnavailable(
            f"ReID 权重文件缺失: {path}\n"
            f"  修复: ./.venv/Scripts/python.exe scripts/fetch_reid_weights.py --only {weight.key}\n"
            f"  （下载需要代理，脚本默认走 http://127.0.0.1:7897）"
        )
    if path.stat().st_size < 1024 * 1024:
        raise ReIDWeightUnavailable(
            f"ReID 权重文件异常（{path.stat().st_size} 字节，疑似下载不完整）: {path}\n"
            f"  修复: 删除该文件后重跑 scripts/fetch_reid_weights.py --only {weight.key}"
        )
    return weight, path


def load_reid_state_dict(path: Path, expected: ReIDWeight) -> dict:
    """
    读取权重并用**分类头维度**校验数据集一致性，防止"文件名叫 msmt17、
    内容其实是 imagenet"这类事故 —— 项目此前正是因为没人校验这一点，
    带着 ImageNet 分类权重跑了很久，还在 UI 上标着 "OSNet Enterprise"。
    """
    import torch  # 延迟导入：本模块的阈值常量会被热路径导入，不能带重依赖

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ReIDWeightUnavailable(f"{path} 内容不是 state_dict，无法作为 OSNet 权重加载")

    classifier = next((k for k in state if k.endswith("classifier.weight")), None)
    if classifier is None:
        raise ReIDWeightUnavailable(f"{path} 缺少分类头，无法判定训练数据集")

    classes, dim = (int(v) for v in state[classifier].shape)
    if dim != FEATURE_DIM:
        raise ReIDWeightUnavailable(
            f"{path} 特征维度 {dim} != {FEATURE_DIM}，不是 OSNet-x0.25"
        )
    actual_dataset = DATASET_BY_CLASSES.get(classes)
    if actual_dataset is None:
        raise ReIDWeightUnavailable(
            f"{path} 分类头 {classes} 类，不属于已知数据集 {sorted(DATASET_BY_CLASSES)}"
        )
    if actual_dataset != expected.dataset:
        raise ReIDWeightUnavailable(
            f"权重数据集不匹配：期望 {expected.dataset}（{expected.num_classes} 类），"
            f"实际 {actual_dataset}（{classes} 类）。文件可能放错了。"
        )
    return state


def describe() -> str:
    """启动日志用的一行摘要。"""
    return (
        f"权重={DEFAULT_REID_WEIGHTS} 匹配阈值={REID_MATCH_THRESHOLD} "
        f"Ratio={REID_RATIO_TEST} 允许ImageNet降级={'是' if ALLOW_IMAGENET_FALLBACK else '否'}"
    )
