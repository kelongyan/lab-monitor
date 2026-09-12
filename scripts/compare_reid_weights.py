"""compare_reid_weights.py — 对比不同 ReID 权重的嵌入质量（label-free 判据）。

为什么需要它
------------
换权重是整个修复的核心动作，但"换完到底有没有用"必须能证伪。
本脚本用**不需要人工标注**的统计量给出判据：

  1. **全体均值向量范数** —— 单位向量集合里，这个值越接近 1，说明所有人共享
     同一个主导分量、身份信息被吞掉。ImageNet 权重实测 0.9754（几乎共线）。
  2. **去均值后每个样本的残差范数** —— ImageNet 权重实测均值仅 0.166。
  3. **两两余弦分布** —— 素材里绝大多数 crop 对属于**不同人**，所以再好的模型下
     余弦也不该普遍接近 1。ImageNet 权重实测 p50 = 0.984（等于完全无判别力）。

判据：均值向量范数显著下降（目标 < 0.85）、组间余弦 p50 显著下降（目标 < 0.6）
即视为"嵌入空间不再塌缩"。真正的身份准确率由 1.3 的标注集给出。

附带作用：输出每路视频检测到的**有效人体框数**，可回答"某些相机为何零身份数据"。

用法：
    ./.venv/Scripts/python.exe scripts/compare_reid_weights.py
    ./.venv/Scripts/python.exe scripts/compare_reid_weights.py --frames 12 --crops 200
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
    describe,
)

#: 优先选"库里确实有身份数据"的相机 —— 它们才可能提供足够多的人体框。
#: 实测（2026-09-12）：reg_01 全程 239 秒、按 1Hz 采样只有 12 个检测样本，
#: 走廊基本是空的；rnd_08 / rnd_19 才有稳定人流。这也解释了部分相机零身份数据。
DEFAULT_CAMERAS = ["rnd_08", "rnd_19", "rnd_04", "rnd_07", "rnd_16", "reg_06"]
MIN_BOX_HEIGHT = 60   # 人体框太小时 OSNet 输入等于噪声，纳入会污染统计
SAMPLE_STRIDE = 10    # 每 N 帧解一帧做检测（25fps 素材下约 0.4s 一个采样点）
MAX_CROPS_PER_CAM = 80
MAX_CROPS_PER_WEIGHT = 300


def collect_crops(cameras: list[str], max_crops_per_cam: int) -> tuple[dict, dict]:
    """
    扫描整段视频采集人体 crop。

    用 grab()/retrieve() 而不是 read()：grab 只推进、不解码，扫过不采样的帧时
    开销极小，这样才能覆盖**整段**视频而不是只看开头几秒
    （第一版只读了前 4 秒，5 路全部 0 个框，纯属采样窗口错误）。

    不用 CAP_PROP_POS_FRAMES 定位：该路素材的容器元数据已被证实不可信
    （ffprobe 与 OpenCV 一致报出 11948~71617 秒的荒谬时长），seek 结果同样不可信。
    """
    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4)
    crops_by_cam: dict[str, list] = {}
    counts: dict[str, dict] = {}

    for cam in cameras:
        rel = sources.get(cam)
        if not rel:
            print(f"  [跳过] {cam} 不在 sources.json")
            continue
        path = ROOT / rel
        if not path.exists():
            print(f"  [跳过] {cam} 文件不存在")
            continue

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  [跳过] {cam} 无法打开")
            continue

        crops: list = []
        decoded = 0
        sampled = 0
        boxes = 0
        while len(crops) < max_crops_per_cam:
            if not cap.grab():
                break
            decoded += 1
            if decoded % SAMPLE_STRIDE:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            sampled += 1
            for x1, y1, x2, y2, _conf in detector.detect(frame):
                boxes += 1
                if (y2 - y1) < MIN_BOX_HEIGHT:
                    continue
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size:
                    crops.append(crop.copy())
        cap.release()

        crops_by_cam[cam] = crops
        counts[cam] = {
            "decoded_frames": decoded,
            "sampled_frames": sampled,
            "person_boxes": boxes,
            "kept_crops": len(crops),
        }
        density = boxes / sampled if sampled else 0.0
        print(f"  [读出] {cam}: 解码 {decoded} 帧 / 采样 {sampled} 帧 / "
              f"人体框 {boxes} 个（{density:.2f} 框/采样帧）/ 保留 {len(crops)}")

    return crops_by_cam, counts


def describe_embedding(model, crops: list) -> dict:
    feats = []
    for crop in crops:
        feat = model.extract(crop, [0, 0, crop.shape[1], crop.shape[0]])
        if feat is not None:
            feats.append(feat)
    if len(feats) < 2:
        return {"n": len(feats)}

    mat = np.stack(feats)
    mean_vec = mat.mean(axis=0)
    centered = mat - mean_vec
    residual = np.linalg.norm(centered, axis=1)
    sims = mat @ mat.T
    off = np.array([sims[i, j] for i, j in combinations(range(len(mat)), 2)])

    return {
        "n": len(mat),
        "dim": int(mat.shape[1]),
        "mean_vec_norm": float(np.linalg.norm(mean_vec)),
        "residual_norm_mean": float(residual.mean()),
        "residual_norm_p5": float(np.percentile(residual, 5)),
        "sim_mean": float(off.mean()),
        "sim_p50": float(np.percentile(off, 50)),
        "sim_p95": float(np.percentile(off, 95)),
        "sim_max": float(off.max()),
        "frac_over_threshold": float((off >= REID_MATCH_THRESHOLD).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="对比 ReID 权重的嵌入质量")
    parser.add_argument("--cameras", nargs="*", default=DEFAULT_CAMERAS)
    parser.add_argument("--per-cam", type=int, default=MAX_CROPS_PER_CAM,
                        help="每路最多保留多少个 crop")
    parser.add_argument("--crops", type=int, default=MAX_CROPS_PER_WEIGHT,
                        help="参与对比的最大 crop 数（跨权重必须用同一批）")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    print(f"ReID 配置: {describe()}")
    print(f"匹配阈值 {REID_MATCH_THRESHOLD}（来自 src/reid_config.py）\n")

    # ReIDExtractorOSNet 需要一个具体设备；工厂函数才有 None→自动检测的逻辑
    device = args.device
    if not device:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"推理设备: {device}\n")

    print("第 1 步：扫描整段视频采集人体 crop")
    crops_by_cam, counts = collect_crops(args.cameras, args.per_cam)
    all_crops = [c for cs in crops_by_cam.values() for c in cs]
    print(f"  合计 {len(all_crops)} 个 crop")

    if counts:
        print("\n  各路检测密度（可回答『某些相机为何零身份数据』）：")
        for cam, info in counts.items():
            density = info["person_boxes"] / max(1, info["sampled_frames"])
            flag = "  ← 几乎没有人体框" if density < 0.05 else ""
            print(f"    {cam:9s} 解码 {info['decoded_frames']:5d} 帧 / "
                  f"采样 {info['sampled_frames']:4d} / 人体框 {info['person_boxes']:4d} / "
                  f"密度 {density:.2f}{flag}")

    if len(all_crops) < 4:
        print("\n可用 crop 太少（< 4），无法做嵌入质量对比。")
        return 1

    rng = np.random.default_rng(0)   # 固定种子：三个权重必须用同一批 crop 才可比
    if len(all_crops) > args.crops:
        idx = rng.choice(len(all_crops), args.crops, replace=False)
        sample = [all_crops[i] for i in idx]
    else:
        sample = all_crops
    print(f"  抽样 {len(sample)} 个 crop 参与对比（三个权重共用同一批）")

    print("\n第 2 步：逐个权重计算嵌入质量")
    results = {}
    targets = [("imagenet", IMAGENET_WEIGHT)] + sorted(REID_WEIGHTS.items())
    for key, weight in targets:
        try:
            model = ReIDExtractorOSNet(device=device, weight=weight)
        except Exception as error:
            print(f"  [{key}] 构建失败: {type(error).__name__}: {error}")
            continue
        stats = describe_embedding(model, sample)
        results[key] = stats
        print(f"  [{key}] {weight.benchmark}")
        if stats.get("n", 0) < 2:
            print("        特征太少，跳过")
            continue
        print(f"        均值向量范数 {stats['mean_vec_norm']:.4f}（越接近 1 越差，"
              f"说明所有人共线）")
        print(f"        去均值残差范数 mean {stats['residual_norm_mean']:.4f} / "
              f"p5 {stats['residual_norm_p5']:.4f}")
        print(f"        两两余弦 mean {stats['sim_mean']:.4f} / p50 {stats['sim_p50']:.4f} / "
              f"p95 {stats['sim_p95']:.4f} / max {stats['sim_max']:.4f}")
        print(f"        余弦 ≥ {REID_MATCH_THRESHOLD} 的比例 "
              f"{stats['frac_over_threshold'] * 100:.2f}%（越低越好）")
        del model

    print("\n" + "=" * 78)
    print(f"{'权重':12s} {'均值范数':>10s} {'残差均值':>10s} {'余弦p50':>10s} {'超阈比例':>10s}")
    print("-" * 78)
    for key, stats in results.items():
        if stats.get("n", 0) < 2:
            continue
        print(f"{key:12s} {stats['mean_vec_norm']:10.4f} {stats['residual_norm_mean']:10.4f} "
              f"{stats['sim_p50']:10.4f} {stats['frac_over_threshold'] * 100:9.2f}%")

    out = ROOT / "outputs" / "reports" / "reid_weight_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"threshold": REID_MATCH_THRESHOLD, "counts": counts, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
