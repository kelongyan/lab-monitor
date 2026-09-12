"""diagnose_ema_collapse.py — 验证「特征塌缩是否由滑动平均（EMA）本身造成」。

背景（这是一个被实测逼出来的怀疑）
--------------------------------
`scripts/probe_reid_separability.py` 读**数据库里存的特征**，得到：
    45 个身份、990 对异人余弦 mean 0.950 / p50 0.984，均值向量范数 0.9754。

但 `scripts/compare_reid_weights.py` 读**视频里现提的原始特征**，用同一份
ImageNet 权重，得到：均值向量范数 0.7958 / 余弦 p50 0.6293。

同一份模型，库里的特征严重塌缩，现提的原始特征却不塌缩 —— 说明塌缩是**流程引入的**，
不能全归因于权重。

机制假设
--------
`src/identity_store.py:update_appearance` 做的是：

    rec.feature = alpha * rec.feature + (1 - alpha) * feat   # 然后重新 L2 归一化

再把结果归一化。这是个带重新注入的递归低通滤波：
- 所有样本共享一个"公共方向"（原始特征均值范数 0.80 说明它确实存在）
- 每次迭代公共分量被按 alpha 保留并持续被新样本重新注入
- 而**身份特异的残差**每轮只被保留 alpha 倍，且没有来源补充

于是残差按 alpha^k 指数衰减。alpha=0.85 时 0.85^50 ≈ 3e-4 ——
**一个身份只要累积几十次更新，它的特征就会收敛到"公共方向"，
与其他身份无法区分**。这正好解释了：
  - 库里 45 个身份两两余弦 ~0.98（都收敛到同一个方向）
  - 出现次数越多塌缩越彻底（现有库里有 13.7 万次出现的"垃圾桶身份"）
  - 现提的原始特征却没问题（原始特征没有经过这个递归）

本脚本用真实 crop 特征做对照实验：
  1. 采一批真实人体 crop，提原始特征
  2. 按"模拟身份"分组，对每组跑 N 次真实的 EMA 更新
  3. 比较「原始特征的两两余弦」vs「EMA 若干次之后的两两余弦」
若后者显著升高，则塌缩机制成立。

只读，不写项目文件（结果打印到 stdout）。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import PersonDetector  # noqa: E402
from src.reid import ReIDExtractorOSNet  # noqa: E402
from src.reid_config import (  # noqa: E402
    IMAGENET_WEIGHT,
    REID_MATCH_THRESHOLD,
    REID_WEIGHTS,
)

#: 与数据库里那批特征同一特征空间，才能做对照
CAMERAS = ["rnd_08", "rnd_19", "rnd_04", "rnd_16"]
SAMPLE_STRIDE = 10
MIN_BOX_HEIGHT = 60
BASE_ALPHA = 0.85          # src/identity_store.py 原来的 base_alpha（当前实现）
UPDATE_COUNTS = [0, 5, 20, 50, 200]
WINDOW_SIZES = [5, 20, 50]      # 候选方案 ① 的窗口大小
RECOVERY_STEPS = [10, 30, 60]   # 污染后观测多少次，看能否自愈


def collect(cameras: list[str], per_cam: int) -> dict[str, list]:
    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4)
    out: dict[str, list] = {}
    for cam in cameras:
        path = ROOT / sources.get(cam, "")
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        crops = []
        decoded = 0
        while len(crops) < per_cam:
            if not cap.grab():
                break
            decoded += 1
            if decoded % SAMPLE_STRIDE:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            for x1, y1, x2, y2, _c in detector.detect(frame):
                if (y2 - y1) < MIN_BOX_HEIGHT:
                    continue
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size:
                    crops.append(crop.copy())
        cap.release()
        out[cam] = crops
        print(f"  {cam}: {len(crops)} 个 crop")
    return out


def pairwise(mat: np.ndarray) -> np.ndarray:
    sims = mat @ mat.T
    return np.array([sims[i, j] for i, j in combinations(range(len(mat)), 2)])


def ema_update(feature: np.ndarray, sample: np.ndarray, alpha: float) -> np.ndarray:
    """复刻 src/identity_store.py 原更新式（EMA + 重新归一化）。"""
    updated = alpha * feature + (1.0 - alpha) * sample
    norm = np.linalg.norm(updated)
    return updated / norm if norm > 1e-8 else updated


class WindowMeanAggregator:
    """候选方案 ①：有界窗口均值（最近 N 个原始特征的均值 + 重新归一化）。

    与 EMA 的本质差别：EMA 每轮把已有向量乘 alpha（身份特异残差按 alpha^k 衰减，
    且无来源补充）；有界窗口只是丢弃最旧的样本，窗口内每个样本权重相同，
    **不存在对已有估计的指数衰减**，而且被污染的样本会在 N 次观测后自动淘汰。
    """

    def __init__(self, size: int):
        self._size = size
        self._samples: list[np.ndarray] = []

    def update(self, sample: np.ndarray) -> np.ndarray:
        self._samples.append(sample)
        if len(self._samples) > self._size:
            self._samples.pop(0)
        mean = np.mean(np.stack(self._samples), axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-8 else mean


def aggregate(groups: list[np.ndarray], mat: np.ndarray, updates: int,
              aggregator_factory) -> np.ndarray:
    """
    对每个身份跑 updates 次观测，返回各身份的最终特征向量。

    aggregator_factory 返回 None 表示用 EMA（当前实现），
    返回 WindowMeanAggregator 表示用候选方案 ①。
    两种策略的更新式都在本文件里独立实现，不改动 src/ 的代码。
    """
    centers = []
    for g in groups:
        aggregator = aggregator_factory()
        feature = None
        start = 0
        if aggregator is None:                 # EMA 需要初始向量
            feature = mat[g[0]].copy()
            start = 1
        for step in range(start, updates + 1):
            sample = mat[g[step % len(g)]]
            feature = (ema_update(feature, sample, BASE_ALPHA)
                       if aggregator is None else aggregator.update(sample))
        centers.append(feature)
    return np.stack(centers)


def summarize(strategy: str, updates: int, centers: np.ndarray) -> dict:
    pairs = pairwise(centers) if len(centers) > 1 else np.array([1.0])
    return {
        "strategy": strategy,
        "updates": updates,
        "mean_vec_norm": float(np.linalg.norm(centers.mean(axis=0))),
        "cross_p50": float(np.percentile(pairs, 50)),
        "cross_p95": float(np.percentile(pairs, 95)),
        "over_threshold": float((pairs >= REID_MATCH_THRESHOLD).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 EMA 导致的特征塌缩")
    parser.add_argument("--per-cam", type=int, default=30)
    parser.add_argument("--identities", type=int, default=8, help="模拟多少个身份")
    parser.add_argument("--chunk", type=int, default=12,
                        help="同相机连续多少帧算『同一个人』")
    parser.add_argument("--weights", default="imagenet",
                        choices=["imagenet", *REID_WEIGHTS],
                        help="用哪份权重提特征；换权重后本脚本的结论可能完全不同")
    parser.add_argument("--center", action="store_true",
                        help="去掉公共分量：先减去全体特征均值再重新归一化")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = (IMAGENET_WEIGHT if args.weights == "imagenet"
              else REID_WEIGHTS[args.weights])
    print(f"设备 {device}；权重 {weight.key}（{weight.benchmark}）")
    print(f"基 alpha = {BASE_ALPHA}（src/identity_store.py 原来的默认值）\n")

    print("第 1 步：采集真实人体 crop")
    crops_by_cam = collect(CAMERAS, args.per_cam)
    flat = [(cam, c) for cam, cs in crops_by_cam.items() for c in cs]
    if len(flat) < args.identities * 2:
        print("crop 太少，无法实验")
        return 1
    print(f"  合计 {len(flat)} 个 crop")

    print("\n第 2 步：提取原始特征")
    model = ReIDExtractorOSNet(device=device, weight=weight)
    feats = []
    for cam, crop in flat:
        feat = model.extract(crop, [0, 0, crop.shape[1], crop.shape[0]])
        if feat is not None:
            feats.append(feat)
    mat = np.stack(feats)
    print(f"  得到 {len(mat)} 个 {mat.shape[1]} 维特征")

    raw_pairs = pairwise(mat)
    print(f"  原始特征：均值向量范数 {np.linalg.norm(mat.mean(axis=0)):.4f}"
          f" | 两两余弦 mean {raw_pairs.mean():.4f} / p50 {np.percentile(raw_pairs, 50):.4f}"
          f" / p95 {np.percentile(raw_pairs, 95):.4f}")

    if args.center:
        # 去掉公共分量：所有特征都和一个全局方向高度共线（均值范数 ~0.80），
        # 这个分量对区分身份毫无贡献，却会在任何"求平均"的操作里相干叠加、
        # 把身份特异的余量挤掉。减均值后重新归一化再做全部统计。
        # 生产侧的可行做法：维护一个全体已观测特征的滑动均值作为中心向量。
        mean_vec = mat.mean(axis=0)
        centered = mat - mean_vec
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        mat = np.divide(centered, np.maximum(norms, 1e-8))
        centered_pairs = pairwise(mat)
        print(f"  去公共分量后：均值向量范数 {np.linalg.norm(mat.mean(axis=0)):.4f}"
              f" | 两两余弦 mean {centered_pairs.mean():.4f} / "
              f"p50 {np.percentile(centered_pairs, 50):.4f} / "
              f"p95 {np.percentile(centered_pairs, 95):.4f}")

    # 第 3 步：构造"身份"。这里的关键是**必须用同一人的样本**，否则实验会自证、
    # 毫无信息量：如果拿随机 crop 凑一个身份（等于把不同人混在一起），
    # 身份特异残差在样本间不相关，EMA 必然把它平均掉 —— 那只是证明了
    # "混人 + 平均 = 塌缩"，是个同义反复。
    #
    # 真实可用的近似：**同一台相机、时间上连续的一小段**（采样间隔 0.4s）
    # 极大概率为同一人。于是把每台相机的 crop 序列按时间切块，
    # 每块 = 一个模拟身份。
    print(f"\n第 3 步：按『同相机 + 时间连续』切块构造身份，跑真实的 EMA 更新")
    groups: list[np.ndarray] = []
    chunk = args.chunk
    cursor = 0
    for cam, crops in crops_by_cam.items():
        count = len(crops)
        for start in range(0, count, chunk):
            block = list(range(cursor + start, cursor + min(start + chunk, count)))
            if len(block) >= 3:
                groups.append(np.array(block))
        cursor += count
    groups = groups[:args.identities]
    if len(groups) < 2:
        print("  可用的连续时间段不足，无法实验")
        return 1
    print(f"  身份数 {len(groups)}，各身份样本数 {[len(g) for g in groups]}"
          f"（同相机连续采样，大概率同一人）")

    raw_centers = np.stack([mat[g[0]] for g in groups])
    raw_cross = pairwise(raw_centers)
    print(f"  原始特征（各身份取首个样本）：均值范数 "
          f"{np.linalg.norm(raw_centers.mean(axis=0)):.4f} | "
          f"跨身份余弦 p50 {np.percentile(raw_cross, 50):.4f}")

    print(f"\n第 4 步：聚合方式对比（每个身份反复观测自己的样本）")
    header = (f"{'策略':>18s} {'观测次数':>8s} {'均值范数':>10s} "
              f"{'跨身份p50':>11s} {'跨身份p95':>11s} {'≥阈值比例':>10s}")
    print(header)
    print("-" * len(header))
    rows = []
    for updates in UPDATE_COUNTS:
        centers = aggregate(groups, mat, updates, lambda: None)
        rows.append(summarize("EMA(a=0.85)", updates, centers))
        if args.center:
            # 顺序对照：聚合之后再减去各身份中心的公共方向。
            # 这对应"最小改动"的实现方式（不动 update_appearance，只在匹配时减中心），
            # 所以必须验证它是否同样有效 —— 聚合会把身份余量压小，
            # 有可能压到"减完中心什么都不剩"。
            recentered = centers - centers.mean(axis=0)
            recentered /= np.maximum(np.linalg.norm(recentered, axis=1, keepdims=True), 1e-8)
            rows.append(summarize("EMA→去中心", updates, recentered))
    for size in WINDOW_SIZES:
        for updates in UPDATE_COUNTS:
            if updates == 0:
                continue
            centers = aggregate(groups, mat, updates,
                                lambda s=size: WindowMeanAggregator(s))
            rows.append(summarize(f"窗口均值(N={size})", updates, centers))
    for row in rows:
        print(f"{row['strategy']:>18s} {row['updates']:>8d} "
              f"{row['mean_vec_norm']:10.4f} {row['cross_p50']:11.4f} "
              f"{row['cross_p95']:11.4f} {row['over_threshold'] * 100:9.1f}%")

    # 第 5 步：污染恢复能力 —— 身份误吞了另一个人的样本之后，多久能恢复。
    # 这直接对应"匹配偶发失败后的自愈能力"：EMA 的污染几乎永久保留，
    # 有界窗口会在 N 次观测后把污染样本淘汰掉。
    print(f"\n第 5 步：污染恢复（先混入 5 个『别人的』样本，之后只观测自己的）")
    print(header)
    print("-" * len(header))
    strategies = [("EMA(a=0.85)", None)]
    strategies += [(f"窗口均值(N={size})", size) for size in WINDOW_SIZES]
    for name, size in strategies:
        for recovery in RECOVERY_STEPS:
            centers = []
            for index, g in enumerate(groups):
                stranger = groups[(index + 1) % len(groups)]   # 下一个身份的样本当"别人"
                aggregator = None if size is None else WindowMeanAggregator(size)
                feature = mat[g[0]].copy()
                for step in range(5):                          # 污染
                    sample = mat[stranger[step % len(stranger)]]
                    feature = (ema_update(feature, sample, BASE_ALPHA)
                               if aggregator is None else aggregator.update(sample))
                for step in range(recovery):                   # 只观测自己
                    sample = mat[g[step % len(g)]]
                    feature = (ema_update(feature, sample, BASE_ALPHA)
                               if aggregator is None else aggregator.update(sample))
                centers.append(feature)
            stats = summarize("", recovery, np.stack(centers))
            print(f"{name:>18s} {recovery:>8d} {stats['mean_vec_norm']:10.4f} "
                  f"{stats['cross_p50']:11.4f} {stats['cross_p95']:11.4f} "
                  f"{stats['over_threshold'] * 100:9.1f}%")

    print("\n判读：")
    print("  · 跨身份 p50 随观测次数上升 = 该聚合方式在制造塌缩；")
    print("    p50 稳定不升 = 该方式对同一人收敛但不会互相靠拢。")
    print("  · 第 5 步看的是自愈能力：误吞别人的样本后，跨身份 p50 能否回落。")
    print("    EMA 的污染基本不可逆；有界窗口在 N 次观测后淘汰污染样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
