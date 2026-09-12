"""tune_reid_threshold.py — 用自动标注集标定 ReID 匹配阈值，并对比候选权重。

标签来源（全部自动，无需人工）
------------------------------
**正样本（同人）**：`scripts/build_label_set.py` 生成 —— 同相机、相邻采样帧（0.4s）、
IoU ≥ 0.25 的检测框。0.4 秒内一个人最多移动 ~0.5m，两框仍重叠 1/4 以上，
几乎不可能是两个人。规则来自运动学约束，误标率低于人工标注的疲劳失误。

**负样本（异人）分两路**：
  A. 自动负样本：同相机同帧 IoU ≤ 0.05 的两个框（同帧两个人）。
     本项目走廊素材里两人同框极少，这类只有个位数。
  B. **代理负样本**：重建后身份库里 22 个身份的跨身份对。
     其中少数对其实可能是同一人（见 worklist 1.8 欠归并），会把负样本分布的
     高位略微抬高 → 选出的阈值偏保守（更高），可接受且已在报告中注明。

产出
----
  outputs/reports/reid_threshold_sweep.csv   每个权重 × 每个阈值的 P/R/F1
  outputs/reports/reid_threshold_sweep.json  汇总与推荐工作点

判读：推荐阈值取「在负样本误报率 ≤ 5% 约束下使 F1 最大的阈值」；
同时报告同人召回，供人工判断是否接受。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reid import ReIDExtractorOSNet  # noqa: E402
from src.reid_config import IMAGENET_WEIGHT, REID_WEIGHTS  # noqa: E402

CROPS_NPZ = ROOT / "outputs" / "labeled" / "crops.npz"
PAIRS_JSON = ROOT / "config" / "labeled" / "pairs.json"
DB = ROOT / "outputs" / "lab_monitor.db"


def extract_features(weight_key: str, crops: np.ndarray) -> np.ndarray:
    weight = (IMAGENET_WEIGHT if weight_key == "imagenet" else REID_WEIGHTS[weight_key])
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ReIDExtractorOSNet(device=device, weight=weight)
    feats = []
    for crop in crops:
        feat = model.extract(crop, [0, 0, crop.shape[1], crop.shape[0]])
        feats.append(feat if feat is not None else np.zeros(512, dtype=np.float32))
    del model
    return np.stack(feats)


def load_db_cross_sims() -> tuple[np.ndarray, list[tuple[float, str, str]]]:
    """重建库 22 个身份的跨身份对（生产语义：中心化 + 重新归一化 + max-over-bank）。"""
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT global_id, last_camera, feature_dim, feature_blob, "
        "feature_bank_count, feature_bank_blob FROM identities"
    ).fetchall()
    conn.close()
    recs = {}
    for gid, cam, dim, blob, bank_count, bank_blob in rows:
        main = np.frombuffer(blob, dtype=np.float32).copy()
        main = main / np.linalg.norm(main)
        bank = [main]
        if bank_count and bank_blob:
            matrix = np.frombuffer(bank_blob, dtype=np.float32).reshape(int(bank_count), int(dim))
            for row in matrix:
                row = row.copy()
                norm = float(np.linalg.norm(row))
                if norm > 1e-8:
                    bank.append(row / norm)
        recs[gid] = (cam, bank)

    mains = np.stack([recs[g][1][0] for g in recs])
    center = mains.mean(axis=0)

    def prep(vector: np.ndarray) -> np.ndarray:
        diff = vector - center
        norm = float(np.linalg.norm(diff))
        return diff / norm if norm > 1e-8 else vector / max(float(np.linalg.norm(vector)), 1e-8)

    sims = []
    for gi, gj in combinations(recs, 2):
        if recs[gi][0] == recs[gj][0]:
            continue                     # 同相机对不作为跨相机负样本
        sim = max(float(prep(p) @ prep(q)) for p in recs[gi][1] for q in recs[gj][1])
        sims.append((sim, gi, gj))
    return np.array([s for s, _a, _b in sims]), sims


def sweep(pos: np.ndarray, neg: np.ndarray, thresholds) -> list[dict]:
    rows = []
    for threshold in thresholds:
        tp = int((pos >= threshold).sum())
        fn = int((pos < threshold).sum())
        fp = int((neg >= threshold).sum())
        tn = int((neg < threshold).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "threshold": round(float(threshold), 3),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="标定 ReID 匹配阈值")
    parser.add_argument("--weights", nargs="*", default=["msmt17", "market1501"])
    parser.add_argument("--fpr-cap", type=float, default=0.05,
                        help="允许的负样本误报率上限（默认 5%%）")
    args = parser.parse_args()

    if not CROPS_NPZ.exists() or not PAIRS_JSON.exists():
        print("缺少标注集，请先运行 scripts/build_label_set.py")
        return 1

    crops = np.load(CROPS_NPZ)["crops"]
    pairs = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    positives = np.array(pairs["positives"], dtype=int)
    auto_negatives = np.array(pairs["negatives"], dtype=int)
    print(f"标注集：{crops.shape[0]} 个 crop / 正样本对 {len(positives)} / "
          f"自动负样本对 {len(auto_negatives)}")

    db_neg, db_detail = load_db_cross_sims()
    print(f"代理负样本（重建库跨身份对）：{len(db_neg)} 个 "
          f"p50={np.percentile(db_neg, 50):.3f} p95={np.percentile(db_neg, 95):.3f} "
          f"max={db_neg.max():.3f}")
    print("  注：其中少数对可能是同一人（worklist 1.8 欠归并），会把负样本高位略抬高，"
          "选出的阈值因此偏保守。\n")

    thresholds = np.arange(0.20, 0.92, 0.02)
    report = {"fpr_cap": args.fpr_cap, "weights": {}}
    csv_rows: list[dict] = []

    for weight_key in args.weights:
        print(f"=== 权重 {weight_key} ===")
        features = extract_features(weight_key, crops)
        center = features.mean(axis=0)

        def prepared(index: int) -> np.ndarray:
            diff = features[index] - center
            norm = float(np.linalg.norm(diff))
            return diff / norm if norm > 1e-8 else features[index] / max(
                float(np.linalg.norm(features[index])), 1e-8)

        pos = np.array([float(prepared(a) @ prepared(b)) for a, b in positives])
        neg_a = np.array([float(prepared(a) @ prepared(b)) for a, b in auto_negatives]) \
            if len(auto_negatives) else np.array([])
        # 代理负样本与自动负样本合并（自动负样本权重更高，但数量太少）
        neg = np.concatenate([db_neg, neg_a]) if len(neg_a) else db_neg

        print(f"  同人相似度    p50={np.percentile(pos, 50):.3f} "
              f"p5={np.percentile(pos, 5):.3f} min={pos.min():.3f}")
        print(f"  异人相似度    p50={np.percentile(neg, 50):.3f} "
              f"p95={np.percentile(neg, 95):.3f} max={neg.max():.3f}")

        rows = sweep(pos, neg, thresholds)
        feasible = [r for r in rows if r["false_positive_rate"] <= args.fpr_cap]
        best = max(feasible or rows, key=lambda r: r["f1"])
        recall_side = [r for r in rows if r["recall"] >= 0.95]
        recall_best = min(recall_side, key=lambda r: r["threshold"]) if recall_side else None

        print(f"  推荐（误报率≤{args.fpr_cap:.0%} 内 F1 最大）：阈值 {best['threshold']} "
              f"P={best['precision']:.3f} R={best['recall']:.3f} F1={best['f1']:.3f} "
              f"FPR={best['false_positive_rate']:.3f}")
        if recall_best:
            print(f"  参照（同人召回 ≥95% 的最低阈值）：{recall_best['threshold']} "
                  f"P={recall_best['precision']:.3f} FPR={recall_best['false_positive_rate']:.3f}")
        print()

        for row in rows:
            csv_rows.append({"weight": weight_key, **row})
        report["weights"][weight_key] = {
            "positive_stats": {
                "p50": float(np.percentile(pos, 50)),
                "p5": float(np.percentile(pos, 5)),
                "min": float(pos.min()),
                "count": int(pos.size),
            },
            "negative_stats": {
                "p50": float(np.percentile(neg, 50)),
                "p95": float(np.percentile(neg, 95)),
                "max": float(neg.max()),
                "count": int(neg.size),
            },
            "recommended_threshold": best["threshold"],
            "recommended": best,
            "min_threshold_for_95pct_recall": (recall_best or {}).get("threshold"),
            "rows": rows,
        }

    out_csv = ROOT / "outputs" / "reports" / "reid_threshold_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["weight", "threshold", "tp", "fn",
                                                    "fp", "tn", "precision", "recall",
                                                    "f1", "false_positive_rate"])
        writer.writeheader()
        writer.writerows(csv_rows)

    out_json = ROOT / "outputs" / "reports" / "reid_threshold_sweep.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细: {out_csv}\n汇总: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
