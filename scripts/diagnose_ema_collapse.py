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
from src.reid_config import IMAGENET_WEIGHT, REID_MATCH_THRESHOLD  # noqa: E402

#: 与数据库里那批特征同一特征空间，才能做对照
CAMERAS = ["rnd_08", "rnd_19", "rnd_04", "rnd_16"]
SAMPLE_STRIDE = 10
MIN_BOX_HEIGHT = 60
BASE_ALPHA = 0.85          # src/identity_store.py 的默认 base_alpha
UPDATE_COUNTS = [0, 5, 20, 50, 200]


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
    """复刻 src/identity_store.py:update_appearance 的更新式（含重新归一化）。"""
    updated = alpha * feature + (1.0 - alpha) * sample
    norm = np.linalg.norm(updated)
    return updated / norm if norm > 1e-8 else updated


def main() -> int:
    parser = argparse.ArgumentParser(description="验证 EMA 导致的特征塌缩")
    parser.add_argument("--per-cam", type=int, default=30)
    parser.add_argument("--identities", type=int, default=8, help="模拟多少个身份")
    parser.add_argument("--chunk", type=int, default=12,
                        help="同相机连续多少帧算『同一个人』")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 {device}；使用 ImageNet 权重（与库内特征同空间，便于对照）")
    print(f"基 alpha = {BASE_ALPHA}（src/identity_store.py 默认值）\n")

    print("第 1 步：采集真实人体 crop")
    crops_by_cam = collect(CAMERAS, args.per_cam)
    flat = [(cam, c) for cam, cs in crops_by_cam.items() for c in cs]
    if len(flat) < args.identities * 2:
        print("crop 太少，无法实验")
        return 1
    print(f"  合计 {len(flat)} 个 crop")

    print("\n第 2 步：提取原始特征")
    model = ReIDExtractorOSNet(device=device, weight=IMAGENET_WEIGHT)
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

    print(f"\n{'更新次数':>8s} {'均值范数':>10s} {'跨身份mean':>11s} {'跨身份p50':>11s} "
          f"{'跨身份max':>11s} {'≥阈值比例':>10s}")
    print("-" * 70)
    for updates in UPDATE_COUNTS:
        centers = []
        for g in groups:
            feature = mat[g[0]].copy()
            for step in range(1, updates + 1):
                # 循环吸收**本身份自己的**样本 —— 模拟同一人反复被观测
                feature = ema_update(feature, mat[g[step % len(g)]], BASE_ALPHA)
            centers.append(feature)
        cmat = np.stack(centers)
        cpairs = pairwise(cmat) if len(cmat) > 1 else np.array([1.0])
        print(f"{updates:>8d} {np.linalg.norm(cmat.mean(axis=0)):10.4f} "
              f"{cpairs.mean():11.4f} {np.percentile(cpairs, 50):11.4f} "
              f"{cpairs.max():11.4f} "
              f"{(cpairs >= REID_MATCH_THRESHOLD).mean() * 100:9.1f}%")

    print("\n判读：")
    print("  · 若「跨身份 p50」随更新次数显著上升 → EMA 即使在同一人样本上也会塌缩，")
    print("    特征聚合方式本身是根因，仅换权重不够。")
    print("  · 若「跨身份 p50」基本不升 → EMA 在同一人样本上是安全的，")
    print("    那么库内塌缩的真凶是「匹配失败 → 身份混入多人 → 平均抹平残差」这个")
    print("    反馈环，修复重点应放在防止误归并（阈值标定 + 重复写入缺陷）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
