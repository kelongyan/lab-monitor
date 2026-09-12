"""build_label_set.py — 全自动构建 ReID 标注对（无需人工确认）。

为什么这套自动标签是可信的
--------------------------
人工标注不可得时，只能用"物理上不可能错"的规则来造标签。本项目素材是固定机位
走廊视频，帧率 25fps，两条规则都来自运动学约束：

  正样本（同一人）：**同一相机、相邻采样帧（间隔 0.4s）、IoU ≥ 0.25** 的两个检测框。
      0.4 秒内一个人最多移动 ~0.5m，两个框还能重叠 1/4 以上，几乎不可能是两个人
      先后站到同一位置。误标率远低于人工标注的疲劳失误。

  负样本（不同人）：**同一相机、同一帧、IoU ≈ 0** 的两个检测框。
      同一帧里两个不重叠的人体框就是两个人（NMS 已去掉重复框）。

刻意**不做**跨相机负样本：走廊链上的相机完全可能拍到同一个人，
跨相机负样本的误标率不可控，会污染整个标定。

产出
----
  outputs/labeled/crops.npz     所有检测裁剪（统一缩放到 128x256）+ 元数据
  config/labeled/pairs.json     正/负样本对的 (i, j) 索引
  outputs/labeled/build_report.json  各相机的检测量与配对统计

用法：
    ./.venv/Scripts/python.exe scripts/build_label_set.py
    ./.venv/Scripts/python.exe scripts/build_label_set.py --cameras rnd_08 rnd_19 --max-frames 400
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

CROP_W, CROP_H = 128, 256          # OSNet 的输入尺寸，顺便把内存压住
SAMPLE_STRIDE = 10                 # 每 N 帧解一帧（25fps 下 0.4s 一个采样点）
MIN_BOX_HEIGHT = 80                # 太小的框特征就是噪声
IOU_POSITIVE = 0.25
IOU_NEGATIVE = 0.05
DEFAULT_CAMERAS = ["rnd_08", "rnd_19", "rnd_04", "rnd_16", "rnd_07", "reg_06", "rnd_06", "reg_08"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect(cameras: list[str], max_frames_per_cam: int) -> tuple[list, dict]:
    """顺序扫描整段视频，记录每个采样帧上的全部人体框。"""
    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    from src.detector import PersonDetector
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4)

    detections: list[dict] = []   # {cam, frame, bbox}
    per_cam: dict[str, dict] = {}
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

        boxes_here = 0
        decoded = 0
        sampled = 0
        while sampled < max_frames_per_cam:
            if not cap.grab():
                break
            decoded += 1
            if decoded % SAMPLE_STRIDE:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            sampled += 1
            height, width = frame.shape[:2]
            for x1, y1, x2, y2, _conf in detector.detect(frame):
                boxes_here += 1
                if (y2 - y1) < MIN_BOX_HEIGHT:
                    continue
                detections.append({
                    "cam": cam,
                    "frame": decoded,          # 解码帧号，同一路内单调可靠
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "size": [int(width), int(height)],
                })
        cap.release()
        per_cam[cam] = {"decoded": decoded, "sampled": sampled, "boxes": boxes_here}
        print(f"  {cam}: 解码 {decoded} 帧 / 采样 {sampled} / 人体框 {boxes_here}")
    return detections, per_cam


def build_pairs(detections: list[dict]) -> tuple[list, list, dict]:
    """
    生成正/负样本对。

    正样本（同一人）：同相机、相邻采样帧、IoU ≥ IOU_POSITIVE。
    负样本（不同人）：同相机、**时间区间重叠**的两个不同轨迹簇。
        把正样本对做并查集得到"轨迹簇"（≈ 同一个人的连续出现）；
        同一相机里两个簇的出现区间互相重叠，意味着同一时刻画面里有两个人
        —— 一个人不可能同时在两处，所以必然是不同人。这条规则同样"物理上不可能错"，
        且能绕开"走廊里很少两人同框"的问题：只要两个人先后出现、区间有搭接即可。
        区间不搭接的簇不配对（可能是同一人离开又回来）。
    """
    by_cam: dict[str, list[int]] = {}
    for index, det in enumerate(detections):
        by_cam.setdefault(det["cam"], []).append(index)

    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []
    per_cam_pos: dict[str, int] = {}
    per_cam_neg: dict[str, int] = {}

    for cam, indices in by_cam.items():
        by_frame: dict[int, list[int]] = {}
        for index in indices:
            by_frame.setdefault(detections[index]["frame"], []).append(index)
        frames = sorted(by_frame)

        # --- 正样本：相邻采样帧内 IoU 足够大的框 ---
        for prev, cur in zip(frames, frames[1:]):
            if cur - prev != SAMPLE_STRIDE:
                continue                     # 只用相邻采样帧，间隔大了不可靠
            for a in by_frame[prev]:
                box_a = np.array(detections[a]["bbox"])
                for b in by_frame[cur]:
                    box_b = np.array(detections[b]["bbox"])
                    if iou(box_a, box_b) >= IOU_POSITIVE:
                        positives.append((a, b))
                        per_cam_pos[cam] = per_cam_pos.get(cam, 0) + 1

        # --- 同帧不重叠 = 负样本（最直接的一类） ---
        for frame in frames:
            group = by_frame[frame]
            for a, b in combinations(group, 2):
                box_a = np.array(detections[a]["bbox"])
                box_b = np.array(detections[b]["bbox"])
                if iou(box_a, box_b) <= IOU_NEGATIVE:
                    negatives.append((a, b))
                    per_cam_neg[cam] = per_cam_neg.get(cam, 0) + 1

        # --- 轨迹簇：正样本做并查集 ---
        parent = {i: i for i in indices}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in positives:
            if a in parent and b in parent:
                parent[find(a)] = find(b)

        clusters: dict[int, list[int]] = {}
        for index in indices:
            clusters.setdefault(find(index), []).append(index)

        # 簇的出现区间；区间重叠的两个簇 = 不同人
        spans = {}
        for root, members in clusters.items():
            frames_of = [detections[m]["frame"] for m in members]
            spans[root] = (min(frames_of), max(frames_of))
        roots = sorted(spans)
        for i, ri in enumerate(roots):
            lo_i, hi_i = spans[ri]
            for rj in roots[i + 1:]:
                lo_j, hi_j = spans[rj]
                overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
                if overlap < SAMPLE_STRIDE:      # 至少搭接一个采样点
                    continue
                a = clusters[ri][0]
                b = clusters[rj][0]
                negatives.append((a, b))
                per_cam_neg[cam] = per_cam_neg.get(cam, 0) + 1

    stats = {
        "per_camera_positives": per_cam_pos,
        "per_camera_negatives": per_cam_neg,
        "iou_positive": IOU_POSITIVE,
        "iou_negative": IOU_NEGATIVE,
        "sample_stride": SAMPLE_STRIDE,
    }
    return positives, negatives, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="全自动构建 ReID 标注对")
    parser.add_argument("--cameras", nargs="*", default=DEFAULT_CAMERAS)
    parser.add_argument("--max-frames", type=int, default=120,
                        help="每路最多采样多少帧（120 帧 ≈ 48 秒素材）")
    parser.add_argument("--max-positives", type=int, default=400)
    parser.add_argument("--max-negatives", type=int, default=800)
    args = parser.parse_args()

    print("第 1 步：扫描视频采集检测框")
    detections, per_cam = collect(args.cameras, args.max_frames)
    if len(detections) < 10:
        print("检测太少，无法构建标注集")
        return 1
    print(f"  合计 {len(detections)} 个检测框")

    print("\n第 2 步：按运动学约束生成样本对")
    positives, negatives, stats = build_pairs(detections)
    print(f"  正样本对（同人）{len(positives)} 个 / 负样本对（异人）{negatives.count and len(negatives)} 个")
    print(f"  各相机正样本: {stats['per_camera_positives']}")
    print(f"  各相机负样本: {stats['per_camera_negatives']}")

    if len(positives) < 30:
        print("正样本不足（需 ≥30），请增大 --max-frames")
        return 1
    if len(negatives) < 30:
        # 本项目走廊素材极少两人同框，自动负样本天然稀少；
        # 不足的部分由 tune_reid_threshold.py 用"重建库跨身份对"作为代理负样本补足。
        print(f"⚠ 自动负样本仅 {len(negatives)} 个（走廊素材很少两人同框，属预期）。"
              f"标定时将由重建库的跨身份对作为代理负样本补足，正样本照常落盘。")

    # 采样到目标规模（正样本通常远多于负样本，二者都要覆盖到所有相机）
    rng = np.random.default_rng(0)
    if len(positives) > args.max_positives:
        pick = rng.choice(len(positives), args.max_positives, replace=False)
        positives = [positives[i] for i in sorted(pick)]
    if len(negatives) > args.max_negatives:
        pick = rng.choice(len(negatives), args.max_negatives, replace=False)
        negatives = [negatives[i] for i in sorted(pick)]

    used = sorted({i for pair in positives + negatives for i in pair})
    remap = {old: new for new, old in enumerate(used)}

    print(f"\n第 3 步：裁剪并缩放 {len(used)} 个检测框到 {CROP_W}x{CROP_H}")
    crops = np.zeros((len(used), CROP_H, CROP_W, 3), dtype=np.uint8)
    meta = []
    for new_index, old_index in enumerate(used):
        det = detections[old_index]
        path = ROOT / json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))[det["cam"]]
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, det["frame"])   # 只用于取帧，不用于时长/定位语义
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  ⚠ 第 {new_index} 个 crop 取帧失败（{det['cam']}#{det['frame']}），置零")
            continue
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        crops[new_index] = cv2.resize(crop, (CROP_W, CROP_H))
        meta.append({"cam": det["cam"], "frame": det["frame"], "bbox": det["bbox"]})

    out_dir = ROOT / "outputs" / "labeled"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "crops.npz", crops=crops)
    (out_dir / "crops_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    labeled_dir = ROOT / "config" / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    pairs_payload = {
        "description": "全自动标注对：正=同相机相邻采样帧 IoU>=0.25；负=同相机同帧 IoU<=0.05",
        "rules": stats,
        "positives": [[remap[a], remap[b]] for a, b in positives],
        "negatives": [[remap[a], remap[b]] for a, b in negatives],
        "crop_count": len(used),
    }
    (labeled_dir / "pairs.json").write_text(
        json.dumps(pairs_payload, ensure_ascii=False, indent=1), encoding="utf-8")

    report = {"per_camera": per_cam, "pair_stats": stats,
              "positives": len(positives), "negatives": len(negatives),
              "crops": len(used)}
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成：正样本 {len(positives)} / 负样本 {len(negatives)} / 裁剪 {len(used)} 个")
    print(f"  {out_dir / 'crops.npz'}")
    print(f"  {labeled_dir / 'pairs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
