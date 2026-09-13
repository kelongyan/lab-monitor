"""ab_resolution.py — 能力三 A/B：原片 vs 低清转码片，到底能不能切源（worklist 2.5 的收尾）。

回答两个问题
------------
1. **帧级**：把"生产口径"（`process_max_width=960` 的运行时缩放）换成低清转码片，
   检测结果会变差多少？计划给的判据是**人数召回下降 ≤5%**。
2. **端到端**：两臂各跑同样时长，每路实际 fps 与身份注册/确认差多少？计划要求
   **每路 fps 提升 ≥10%**。

为什么这个实验必须做，以及它和计划当初的假设有何不同
--------------------------------------------------
`main.py:252` 的 `process_max_width` 默认 **960**，即**生产链路本来就把帧缩到 960 宽**
才喂给 YOLO / OSNet（`pipeline.py:474`）。所以"原片 → 低清片"在**模型输入尺寸**上几乎
没有差别，真正的变量只剩两个：

  ① 转码 CRF28 一次压缩带来的画质损失；
  ② 解码成本（1920×1080 / 2560×1440 → 960×540）。

计划里担心的"远距离小目标在 540p 掉出有效感受野"在**当前配置下已经不成立**
（处理侧本来就是 960 宽），需要用实测把这个结论钉死，而不是继续悬着。

方法
----
**帧级对照**（全自动，无需人工真值）。顺序解码取同一帧号的三个版本：

    HI    = 原片第 N 帧（全分辨率）
    HI960 = HI 缩放到 960 宽（= 生产口径）
    LO    = 低清片第 N 帧（本来就是 960 宽）

用生产同款检测器（yolov8n / conf=0.4 / FP16）各检一遍，两两比对（IoU≥0.5 贪心匹配）：

    HI   → HI960   运行时缩放的代价（这部分**已经在生产里**，作为参照基线）
    HI960 → LO     **切源真正改变的东西** ← 判据看这一行
    HI   → LO      总代价（含全分辨率本身的优势）

为什么必须**顺序**解码：原片容器时间戳损坏（实测 reg_06 / reg_08 按时间 seek 会落到
完全无关的画面），而顺序读在两片上严格同帧号对齐（实测相关系数 ≥0.99）。

**端到端两臂**：同参数（detect_every_n=1 / reid_every_n=1 / frame_rate_cap=10 /
process_max_width=960），只改 source，各跑 `--seconds` 秒，记录每路处理帧数、fps、
轨迹数、特征数、注册/归并/歧义、身份数、匹配率。

产物
----
    outputs/reports/resolution_ab.csv    逐相机逐对照的检测一致性（追加写）
    outputs/reports/resolution_ab.json   每次运行一条 run 记录（追加写，不覆盖）

全程只写临时目录（`src.db` 惰性单例未被触碰），生产库零写入。

⚠️ 结论与教训（2026-09-13 实测，务必先读再决定切源）
--------------------------------------------------
本脚本的两条判据**都通过了**：3 路帧级检测召回下降 4.47%（≤5%）、22 路每路 fps
1.71 → 1.94（+13.5%，生产线程钳制口径）。但按此结论切源后，**在线身份匹配明显变差**：

| 指标（同一起点 gallery=30，各约 12 分钟） | 原片 | 低清片 |
|---|---|---|
| match_rate | 0.636 | 0.27 |
| Ratio 判歧义占比 | 24% | 62% |
| 新建身份数 | +1 | **+5** |
| avg_top1_similarity | 0.911 | 0.812 |
| avg_ratio_margin | 0.382 | 0.327 |

再对库内 35 个身份做离线中心化两两余弦：新建的 5 个里有 **4 个**与已有身份相似度
0.70~0.92（越阈 = 线上判为同一人）→ 是重复身份，不是新出现的人。

**教训：检测"人数召回"与身份的"可分辨性"是两件事。** CRF28 压缩抹掉的是衣物纹理这类
细粒度判别信息，而人数召回对这类损失不敏感 —— 只比召回会得出"可以切源"的错误结论。
因此本脚本现在同时记录 `avg_top1_similarity` / `avg_ratio_margin` /
`avg_feature_quality` / `ratio_blocked_count` / 新建身份速度；下次评估切源必须让这些
指标同向，否则以原片为准。低清语料的正确定位是**回放与归档**（检索结果已归一到低清片，
见 `src/search.py:_asset_index`），不是处理输入。

用法
----
    ./.venv/Scripts/python.exe scripts/ab_resolution.py
    ./.venv/Scripts/python.exe scripts/ab_resolution.py --cameras reg_01 rnd_21 rnd_16
    ./.venv/Scripts/python.exe scripts/ab_resolution.py --skip-pipeline   # 只跑帧级对照
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

# 必须在导入 cv2 之前设置（与 main.py 同源）
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

#: 计划指定的三路：1080p 常规 / 1440p 随机 / 含唯一 ROI 围栏
DEFAULT_CAMERAS = ["reg_01", "rnd_21", "rnd_16"]
#: 生产口径（main.py 的 process_max_width 默认值）
PRODUCTION_MAX_WIDTH = 960
#: 判定"小目标"的框高阈值（计划里的担心点）
SMALL_BOX_HEIGHT = 80
#: 帧级对照的最小采样步长（帧）。短素材若只按 want 均分，会退化成逐帧取样，
#: 相邻样本高度相关、统计功效虚高 —— 按 12 帧（25fps 下 0.48s）兜底更诚实。
MIN_STRIDE = 12


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 帧级检测一致性                                                               #
# --------------------------------------------------------------------------- #

def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _rescale(boxes: list[list[float]], factor: float) -> list[list[float]]:
    """把候选框缩放到参考帧的坐标系（否则 1920 尺度的框和 960 尺度的框没法比 IoU）。"""
    if factor == 1.0:
        return boxes
    return [[b[0] * factor, b[1] * factor, b[2] * factor, b[3] * factor, b[4]]
            for b in boxes]


def _greedy_match(ref: list[list[float]], cand: list[list[float]],
                  iou_thr: float = 0.5) -> tuple[set[int], set[int]]:
    """
    贪心 IoU 匹配（按 IoU 从大到小，一对一）。
    返回 (被匹配上的参考框下标集合, 被匹配上的候选框下标集合) ——
    召回、漏检、多检、小目标召回都从这两个集合派生，避免重复算一遍匹配。
    """
    pairs = []
    for i, r in enumerate(ref):
        for j, c in enumerate(cand):
            score = _iou(r, c)
            if score >= iou_thr:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_ref, used_cand = set(), set()
    for _score, i, j in pairs:
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
    return used_ref, used_cand


def _sample_frames(hi_path: Path, lo_path: Path, want: int):
    """
    顺序解码两片，按同一帧号成对返回 [(idx, hi, lo), ...]。

    为什么不用 seek：原片容器时间戳损坏，按时间 seek 会落到无关画面；
    顺序读保证两片严格同帧号对齐。取帧上限用较短的一路。
    """
    cap_hi, cap_lo = cv2.VideoCapture(str(hi_path)), cv2.VideoCapture(str(lo_path))
    if not (cap_hi.isOpened() and cap_lo.isOpened()):
        cap_hi.release()
        cap_lo.release()
        return [], 0, 0
    total_hi = int(cap_hi.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_lo = int(cap_lo.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    usable = min(total_hi, total_lo) if (total_hi and total_lo) else max(total_hi, total_lo)
    stride = max(MIN_STRIDE, usable // max(1, want))

    out = []
    idx = 0
    while True:
        ok_hi, frame_hi = cap_hi.read()
        ok_lo, frame_lo = cap_lo.read()
        if not (ok_hi and ok_lo):
            break
        if idx % stride == 0:
            out.append((idx, frame_hi, frame_lo))
        idx += 1
    cap_hi.release()
    cap_lo.release()
    return out, total_hi, total_lo


def frame_level_compare(camera: str, hi_path: Path, lo_path: Path,
                        detector, want_frames: int) -> dict:
    pairs, total_hi, total_lo = _sample_frames(hi_path, lo_path, want_frames)
    if not pairs:
        return {"camera_id": camera, "error": "无法解码（检查文件是否可读）"}

    comparisons = {
        "HI->HI960": {"ref_boxes": 0, "matched": 0, "ref_only": 0, "cand_only": 0,
                      "small_ref": 0, "small_matched": 0, "ref_heights": []},
        "HI960->LO": {"ref_boxes": 0, "matched": 0, "ref_only": 0, "cand_only": 0,
                      "small_ref": 0, "small_matched": 0, "ref_heights": []},
        "HI->LO": {"ref_boxes": 0, "matched": 0, "ref_only": 0, "cand_only": 0,
                   "small_ref": 0, "small_matched": 0, "ref_heights": []},
    }

    def accumulate(key: str, ref: list[list[float]], cand: list[list[float]],
                   ref_width: int, cand_width: int) -> None:
        # 参考臂与候选臂的分辨率不同时必须先把候选缩放到参考帧尺度，
        # 否则 1920 尺度的框与 960 尺度的框算 IoU 恒为 0（第一版就踩了这个坑）
        factor = (ref_width / cand_width) if cand_width else 1.0
        cand_scaled = _rescale(cand, factor)
        used_ref, used_cand = _greedy_match(ref, cand_scaled)
        slot = comparisons[key]
        slot["ref_boxes"] += len(ref)
        slot["matched"] += len(used_ref)
        slot["ref_only"] += len(ref) - len(used_ref)
        slot["cand_only"] += len(cand_scaled) - len(used_cand)
        # 小目标召回：参考臂里框高 < 80px 的那些有多少被匹配上。
        # 阈值按**各自参考帧**的尺度（HI 参考下 80px 是 1080p 的 80 行，
        # HI960 参考下是 960×540 的 80 行 = 画面高的 14.8%），跨行不可直接比。
        for i, box in enumerate(ref):
            height = box[3] - box[1]
            slot["ref_heights"].append(height)
            if height < SMALL_BOX_HEIGHT:
                slot["small_ref"] += 1
                if i in used_ref:
                    slot["small_matched"] += 1

    for idx, frame_hi, frame_lo in pairs:
        height, width = frame_hi.shape[:2]
        if width > PRODUCTION_MAX_WIDTH:
            scale = PRODUCTION_MAX_WIDTH / width
            frame_hi960 = cv2.resize(
                frame_hi, (PRODUCTION_MAX_WIDTH, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA)
        else:
            frame_hi960 = frame_hi
        # 参考臂的框都在各自帧的坐标系里，比较前统一到参考帧尺度
        boxes_hi = detector.detect(frame_hi)
        boxes_hi960 = detector.detect(frame_hi960)
        boxes_lo = detector.detect(frame_lo)

        accumulate("HI->HI960", boxes_hi, boxes_hi960,
                   frame_hi.shape[1], frame_hi960.shape[1])
        accumulate("HI960->LO", boxes_hi960, boxes_lo,
                   frame_hi960.shape[1], frame_lo.shape[1])
        accumulate("HI->LO", boxes_hi, boxes_lo,
                   frame_hi.shape[1], frame_lo.shape[1])

    rows = []
    for key, slot in comparisons.items():
        ref_boxes = slot["ref_boxes"]
        rows.append({
            "camera_id": camera,
            "comparison": key,
            "sampled_frames": len(pairs),
            "ref_boxes": ref_boxes,
            "matched": slot["matched"],
            "recall": round(slot["matched"] / ref_boxes, 4) if ref_boxes else None,
            "missing": slot["ref_only"],
            "extra": slot["cand_only"],
            "small_ref": slot["small_ref"],
            "small_matched": slot["small_matched"],
            "small_recall": (round(slot["small_matched"] / slot["small_ref"], 4)
                             if slot["small_ref"] else None),
            "median_ref_height": (round(float(statistics.median(slot["ref_heights"])), 1)
                                  if slot["ref_heights"] else None),
        })
    return {
        "camera_id": camera,
        "frames_total_hi": total_hi,
        "frames_total_lo": total_lo,
        "sampled_frames": len(pairs),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# 端到端两臂                                                                   #
# --------------------------------------------------------------------------- #

class Counter:
    """极简记账：轨迹与特征事件（锁保护，跨 pipeline 线程）。"""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.seen: dict[tuple, list[float]] = {}
        #: 与 IdentityResolution.status 的取值一致（identity_store.py:694-771）：
        #: created=新建身份 / matched=匹配到已有身份 / ambiguous=Ratio 判歧义 / invalid=特征非法
        self.registrations = {"created": 0, "matched": 0, "ambiguous": 0, "invalid": 0}

    def see(self, cam: str, tids: list[int], now: float) -> None:
        with self.lock:
            for tid in tids:
                self.seen.setdefault((cam, tid), []).append(now)

    def feature(self, cam: str, tid: int, now: float) -> None:
        with self.lock:
            self.features.setdefault((cam, tid), []).append(now)


def instrument(pipeline, camera: str, counter: "Counter") -> None:
    """
    给 pipeline 挂探针：只统计"见过多少条轨迹"。

    刻意**不**挂 `_validator`：文件循环会在 `_reset_stream_state()` 里换一个新的
    validator，探针会中途失效、统计口径前后不一致（tracker 不重建，所以这个探针是稳的）。
    """
    orig_update = pipeline._tracker.update

    def spy_update(detections, shape):
        tracks = orig_update(detections, shape)
        counter.see(camera, [t["track_id"] for t in tracks], time.monotonic())
        return tracks

    pipeline._tracker.update = spy_update


def run_arm(arm: str, sources: dict[str, str], seconds: float,
            detector, reid_extractor, device: str) -> dict:
    """跑一臂（同参数、只差 source），返回指标。"""
    from src.alerter import AlertManager, AlertBroadcaster
    from src.calibrator import TransitCalibrator
    from src.db import Database
    from src.frame_hub import FrameHub
    from src.identity_store import IdentityStore
    from src.pipeline import CameraPipeline
    from src.topology import CameraTopology, TopologyValidationError

    with tempfile.TemporaryDirectory(prefix=f"ab_{arm}_") as temp:
        temp_dir = Path(temp)
        database = Database(temp_dir / "ab.db")
        try:
            store = IdentityStore(database=database,
                                  feature_space=reid_extractor.feature_space)
            try:
                topology = CameraTopology(ROOT / "config" / "topology.json",
                                          allowed_camera_ids=set(sources))
            except TopologyValidationError:
                # 子集相机触发拓扑校验失败 → 空拓扑（本实验不测告警）
                topology = CameraTopology(
                    (ROOT / "config" / "topology.json").with_name(".ab-disabled"),
                    allowed_camera_ids=set(sources))
            frame_hub = FrameHub(jpeg_quality=50)
            frame_hub.register_cameras(sources)
            alert_manager = AlertManager(
                alert_log=temp_dir / "alerts.jsonl",
                notifier=None,                      # 绝不允许实验触发真实外部通知
                broadcaster=AlertBroadcaster(delivery_enabled=False),
                identity_store=store,
                screenshot_dir=temp_dir / "screenshots",
                database=database,
            )
            calibrator = TransitCalibrator(temp_dir / "transit_stats.json",
                                           valid_edges=topology.edges())

            orig_register = store.register_if_new
            counter = Counter()

            def spy_register(feat):
                resolution = orig_register(feat)
                status = resolution.status if resolution.status in counter.registrations else None
                if status:
                    counter.registrations[status] += 1
                return resolution

            store.register_if_new = spy_register

            pipelines = []
            for cam, source in sources.items():
                pipeline = CameraPipeline(
                    camera_id=cam, source=source,
                    detector=detector, reid_extractor=reid_extractor,
                    identity_store=store, topology=topology,
                    alert_manager=alert_manager,
                    screenshot_dir=temp_dir / "screenshots",
                    frame_hub=frame_hub, calibrator=calibrator,
                    display=False,
                    detect_every_n=1 if device == "cuda" else 3,
                    reid_every_n=1,                     # 生产 GPU 档已定为 R=1
                    frame_rate_cap=10.0,
                    process_max_width=PRODUCTION_MAX_WIDTH,   # 与生产一致
                    personnel=None,
                )
                instrument(pipeline, cam, counter)
                pipelines.append(pipeline)

            t0 = time.monotonic()
            for pipeline in pipelines:
                pipeline.start()
            time.sleep(seconds)
            for pipeline in pipelines:
                pipeline.stop()
            for pipeline in pipelines:
                pipeline.join(timeout=15)
            elapsed = time.monotonic() - t0

            try:
                store.flush()
            except Exception:
                pass
            alert_manager.close()          # 不关的话 Windows 上临时目录清理会 WinError 32
            calibrator.flush()

            with counter.lock:
                seen = {k: list(v) for k, v in counter.seen.items()}
                registrations = dict(counter.registrations)

            metrics = store.get_metrics()
            per_camera = {}
            for pipeline in pipelines:
                per_camera[pipeline.camera_id] = {
                    "processed_frames": pipeline._frame_idx,
                    "fps": round(pipeline._frame_idx / elapsed, 2),
                }
            return {
                "arm": arm,
                "elapsed_s": round(elapsed, 1),
                "per_camera": per_camera,
                "mean_fps": round(statistics.mean(
                    c["fps"] for c in per_camera.values()), 2),
                "tracks_total": len(seen),
                "registrations": registrations,
                "identities_in_store": len(store.all_ids()),
                "match_rate": metrics.get("match_rate"),
                "total_searches": metrics.get("total_searches"),
                "successful_matches": metrics.get("successful_matches"),
                "collapse_warnings": metrics.get("collapse_warnings"),
                # ⚠️ 这三个是**身份可分辨性**指标，帧级召回测不到它们：
                # 2026-09-13 实测低清语料上 avg_top1 0.91→0.81、ratio_margin 0.38→0.33、
                # 在线 match_rate 0.64→0.27，同时新建身份速度是原片的 5 倍。
                # 只比"检测人数召回"会得出"可以切源"的错误结论。
                "ratio_blocked_count": metrics.get("ratio_blocked_count"),
                "avg_top1_similarity": metrics.get("avg_top1_similarity"),
                "avg_ratio_margin": metrics.get("avg_ratio_margin"),
                "avg_feature_quality": metrics.get("avg_feature_quality"),
            }
        finally:
            database.close()


# --------------------------------------------------------------------------- #
# 主流程                                                                       #
# --------------------------------------------------------------------------- #

def load_lowres_paths() -> dict[str, str]:
    """从生产库**只读**取 videos_low/<cam>.mp4 路径（不写、不建连接池）。"""
    db_path = ROOT / "outputs" / "lab_monitor.db"
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT camera_id, rel_path FROM video_assets "
            "WHERE rel_path LIKE 'videos_low/%'").fetchall()
    finally:
        conn.close()
    return {cam: rel for cam, rel in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="能力三 A/B：原片 vs 低清转码片")
    parser.add_argument("--cameras", nargs="*", default=DEFAULT_CAMERAS)
    parser.add_argument("--frames", type=int, default=240,
                        help="帧级对照每相机采样帧数")
    parser.add_argument("--seconds", type=float, default=120.0,
                        help="端到端每臂运行秒数")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="只跑帧级对照（不跑端到端两臂）")
    parser.add_argument("--skip-frame", action="store_true",
                        help="只跑端到端两臂")
    args = parser.parse_args()

    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    lowres = load_lowres_paths()
    cameras = [c for c in args.cameras if sources.get(c) and lowres.get(c)]
    if not cameras:
        log("没有可用的相机（需要同时存在于 sources.json 与 video_assets 的低清行）")
        return 1

    import torch

    # 与 main.py:263-264 同款线程钳制。**不设的话帧率不可与生产比较**：
    # 22 路解码 + 推理时，OpenCV/torch 各自的内部线程池会和 22 个 pipeline 线程抢核，
    # 实测（16:30 未钳制）每路 1.51 fps，而钳制后同一批相机在 90s 受控实验里是 2.48 fps
    # （outputs/reports/reid_sampling_sweep.json）。两个数字不是同一套线程制度下的产物。
    import cv2 as _cv2
    _cv2.setNumThreads(1)
    torch.set_num_threads(2)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"设备 {device} | 相机 {cameras} | 帧级采样 {args.frames} 帧/路 "
        f"| 端到端每臂 {args.seconds:.0f}s | 处理宽度上限 {PRODUCTION_MAX_WIDTH}")

    from src.detector import PersonDetector
    from src.reid import build_reid_extractor
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4, device=device)

    report: dict = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "cameras": cameras,
        "production_max_width": PRODUCTION_MAX_WIDTH,
        "small_box_height": SMALL_BOX_HEIGHT,
        "frame_level": [],
        "pipeline_arms": [],
    }
    csv_rows: list[dict] = []

    if not args.skip_frame:
        log("\n=== 帧级检测一致性 ===")
        for cam in cameras:
            hi_path = ROOT / sources[cam]
            lo_path = ROOT / lowres[cam]
            result = frame_level_compare(cam, hi_path, lo_path, detector, args.frames)
            report["frame_level"].append(result)
            if result.get("error"):
                log(f"  {cam}: {result['error']}")
                continue
            log(f"  {cam}: 原片 {result['frames_total_hi']} 帧 / 低清 "
                f"{result['frames_total_lo']} 帧，采样 {result['sampled_frames']} 帧")
            for row in result["rows"]:
                csv_rows.append(row)
                recall = "—" if row["recall"] is None else f"{row['recall'] * 100:5.1f}%"
                small = ("—" if row["small_recall"] is None
                         else f"{row['small_recall'] * 100:5.1f}%")
                log(f"    {row['comparison']:<12} 召回 {recall} "
                    f"(参考框 {row['ref_boxes']:5d} / 漏 {row['missing']:3d} / 多 {row['extra']:3d})"
                    f"  小目标(<{SMALL_BOX_HEIGHT}px) {small} ({row['small_matched']}/{row['small_ref']})"
                    f"  框高中位 {row['median_ref_height']}px")

    if not args.skip_pipeline:
        log("\n=== 端到端两臂 ===")
        reid_extractor = build_reid_extractor(device=device)
        arms = {
            "HI": {cam: str(ROOT / sources[cam]) for cam in cameras},
            "LO": {cam: str(ROOT / lowres[cam]) for cam in cameras},
        }
        for arm, arm_sources in arms.items():
            log(f"  跑 {arm} 臂（{args.seconds:.0f}s）…")
            result = run_arm(arm, arm_sources, args.seconds, detector,
                             reid_extractor, device)
            report["pipeline_arms"].append(result)
            per_cam = " · ".join(
                f"{cam} {info['fps']:.2f}fps/{info['processed_frames']}帧"
                for cam, info in result["per_camera"].items())
            log(f"    {arm}: 平均 {result['mean_fps']} fps | {per_cam}")
            log(f"       轨迹 {result['tracks_total']} 条 | 注册 {result['registrations']} "
                f"| 身份 {result['identities_in_store']} | 匹配率 {result['match_rate']}"
                f"（{result['successful_matches']}/{result['total_searches']}）")
            log(f"       身份可分辨性：top1 {result['avg_top1_similarity']} / "
                f"ratio_margin {result['avg_ratio_margin']} / "
                f"特征质量 {result['avg_feature_quality']} / "
                f"Ratio 判歧义 {result['ratio_blocked_count']} / "
                f"塌缩告警 {result['collapse_warnings']}")

    # ---- 判据 ----
    verdicts = {}
    key_rows = [r for r in csv_rows if r["comparison"] == "HI960->LO" and r["recall"] is not None]
    if key_rows:
        total_ref = sum(r["ref_boxes"] for r in key_rows)
        total_matched = sum(r["matched"] for r in key_rows)
        overall = total_matched / total_ref if total_ref else None
        verdicts["HI960_to_LO_recall"] = round(overall, 4) if overall is not None else None
        verdicts["HI960_to_LO_drop"] = (round(1 - overall, 4) if overall is not None else None)
        verdicts["frame_level_pass"] = bool(overall is not None and (1 - overall) <= 0.05)
    if len(report["pipeline_arms"]) == 2:
        hi, lo = report["pipeline_arms"]
        verdicts["mean_fps_hi"] = hi["mean_fps"]
        verdicts["mean_fps_lo"] = lo["mean_fps"]
        if hi["mean_fps"]:
            gain = (lo["mean_fps"] - hi["mean_fps"]) / hi["mean_fps"]
            verdicts["fps_gain"] = round(gain, 4)
            verdicts["fps_pass"] = bool(gain >= 0.10)
    report["verdicts"] = verdicts

    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "resolution_ab.json"
    # 追加而不是覆盖：帧级对照用 3 路（快），吞吐对照必须 22 路（才压得出 GIL 瓶颈），
    # 两次运行各有各的结论，覆盖式写法会把上一次的数据静默丢掉。
    history = []
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                history = existing["runs"]
            elif isinstance(existing, list):
                history = existing
        except json.JSONDecodeError:
            history = []
    history.append(report)
    json_path.write_text(json.dumps({"runs": history}, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    csv_path = out_dir / "resolution_ab.csv"
    if csv_rows:
        for row in csv_rows:
            row["ts"] = report["ts"]
            row["cameras"] = len(cameras)
        fieldnames = list(csv_rows[0].keys())
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(csv_rows)

    log("\n" + "=" * 66)
    log("判据")
    log("-" * 66)
    if "HI960_to_LO_drop" in verdicts:
        log(f"  帧级：生产口径(HI960) → 低清片(LO) 人数召回下降 "
            f"{verdicts['HI960_to_LO_drop'] * 100:.2f}%  "
            f"{'✓ ≤5%' if verdicts.get('frame_level_pass') else '✗ 超过 5%'}")
    if "fps_gain" in verdicts:
        log(f"  端到端：每路 fps {verdicts['mean_fps_hi']} → {verdicts['mean_fps_lo']} "
            f"（{verdicts['fps_gain'] * 100:+.1f}%）  "
            f"{'✓ ≥10%' if verdicts.get('fps_pass') else '✗ <10%'}")
    log(f"  明细：{csv_path.name} / {json_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
