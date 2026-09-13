"""ab_feature_window.py — 主特征聚合策略的对照实验（worklist 1.5 ②，2026-09-13）。

回答一个问题：把主特征从**质量加权 EMA** 换成**有界窗口内质量加权平均**，
能不能止住"同一个人被反复注册成新身份"？

背景（实测，见 identity_store.FEATURE_WINDOW_SIZE 的注释）
------------------------------------------------------
EMA 的新观测权重是 (1-α)，α = base + (1-base)(1-q) —— 权重随质量**塌陷**：

    q=1.00 → 0.150   q=0.50 → 0.075   q=0.20 → 0.030   q→0 → 0

本语料实测 `avg_feature_quality` 只有 0.12~0.28（走廊素材里人是小目标），
单次观测权重仅 3%~6% → 主特征被最早的观测主导（30 次观测后最初的仍占 25%~40%）。
同一个人重新入镜时，新鲜特征与陈旧主特征差到阈值之外 → register_if_new 认不出
→ 又注册一个重复身份。实测：归并到 26 个身份后重启 5 分钟又涨到 34，
新建的 8 个里 4 个与已有身份中心化余弦 0.77~0.92（越阈＝线上判为同一人）。

方法
----
同一份 22 路语料、同一套参数（detect_every_n=1 / reid_every_n=1 / frame_rate_cap=10 /
process_max_width=960，与 main.py 同款线程钳制），只改聚合策略，各跑 `--seconds` 秒。
两臂用**独立的临时库**，起点都是空库 → 可比。

度量
----
- **新建身份数**（主判据）+ 每次新建时的**最近邻相似度**（诊断"为什么新建"）
- `created_near_miss`：新建时最近邻相似度 ≥0.5 的次数（差一点就能认出）
- match_rate / Ratio 判歧义占比 / avg_ratio_margin（越高越能区分）
- **越阈重复对**：用**线上同一套语义**（IdentityStore._feature_center_locked +
  _prepare_for_match + 重新归一化）算两两中心化余弦，越阈 = 线上判为同一人

判据：要"修好了"，必须 新建身份数明显下降、created_near_miss 下降、
且越阈重复对不上升（不能靠放松阈值的假象换来的）。

用法
----
    ./.venv/Scripts/python.exe scripts/ab_feature_window.py --policy ema    --seconds 300
    ./.venv/Scripts/python.exe scripts/ab_feature_window.py --policy window --seconds 300
    ./.venv/Scripts/python.exe scripts/ab_feature_window.py --policy window --window 8
两次都跑完会打印对照表并追加写入 outputs/reports/feature_window_ab.json。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- 必须在 import src.* 之前确定聚合策略：FEATURE_WINDOW_SIZE 是 import 期常量 ----
_parser = argparse.ArgumentParser(description="主特征聚合策略对照实验")
_parser.add_argument("--policy", choices=("ema", "window"), required=True,
                     help="ema=改造前的质量加权 EMA；window=有界窗口均值")
_parser.add_argument("--window", type=int, default=20, help="窗口长度（window 策略）")
_parser.add_argument("--seconds", type=float, default=300.0, help="每臂运行秒数")
_parser.add_argument("--cameras", nargs="*", default=None, help="默认 sources.json 全部")
_parser.add_argument("--seed-db", default="production",
                     help="两臂的起始身份库：production（默认，直接用生产库副本，"
                          "这样起点就是线上那个已被污染的 gallery）/ none / 自定义路径")
_parser.add_argument("--device", default=None)
_args = _parser.parse_args()

os.environ["LAB_MONITOR_FEATURE_WINDOW"] = (
    "0" if _args.policy == "ema" else str(max(1, _args.window))
)

import numpy as np  # noqa: E402

from src.identity_store import (  # noqa: E402
    FEATURE_WINDOW_SIZE,
    IdentityStore,
    _aggregate_window_feature,   # noqa: E402  (用于自检：与线上同一实现)
)
from src.reid_config import REID_MATCH_THRESHOLD  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def duplicate_stats(store: IdentityStore) -> dict:
    """
    **线上同一套语义**下的两两中心化余弦（不要另写一套 —— 项目已经在"离线分析口径
    与线上不一致"上踩过坑：不重新归一化的点积可以超过 1）。
    """
    with store._lock:
        records = list(store._records.values())
        center = store._feature_center_locked()
    if len(records) < 2:
        return {"identities": len(records), "pairs": 0, "over_threshold": 0,
                "over_threshold_pct": 0.0, "p50": 0.0, "p95": 0.0, "center_norm": 0.0}
    mains = [IdentityStore._prepare_for_match(rec.feature, center) for rec in records]
    matrix = np.stack(mains).astype(np.float64)
    normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    sims = normalized @ normalized.T
    values = np.array([sims[i, j] for i, j in combinations(range(len(records)), 2)])
    over = int((values >= REID_MATCH_THRESHOLD).sum())
    return {
        "identities": len(records),
        "pairs": int(values.size),
        "over_threshold": over,
        "over_threshold_pct": round(100.0 * over / max(1, values.size), 2),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "center_norm": round(float(np.linalg.norm(center)), 4) if center is not None else 0.0,
    }


def run_arm(sources: dict[str, str], seconds: float, detector, reid_extractor,
            device: str, seed_db: Path | None) -> dict:
    from src.alerter import AlertManager, AlertBroadcaster
    from src.calibrator import TransitCalibrator
    from src.db import Database
    from src.frame_hub import FrameHub
    from src.pipeline import CameraPipeline
    from src.topology import CameraTopology, TopologyValidationError

    with tempfile.TemporaryDirectory(prefix="feature_window_") as temp:
        temp_dir = Path(temp)
        arm_db = temp_dir / "arm.db"
        if seed_db is not None and seed_db.exists():
            # 从生产库副本起步：起点就是线上那个已被污染、且内容循环播放的 gallery。
            # 空库起步的对照几乎只测"注册"（matched 只有 1 次），复刻不出线上的复现问题。
            import shutil as _shutil
            _shutil.copy2(seed_db, arm_db)
        database = Database(arm_db)
        try:
            store = IdentityStore(database=database,
                                  feature_space=reid_extractor.feature_space)
            try:
                topology = CameraTopology(ROOT / "config" / "topology.json",
                                          allowed_camera_ids=set(sources))
            except TopologyValidationError:
                topology = CameraTopology(
                    (ROOT / "config" / "topology.json").with_name(".ab-disabled"),
                    allowed_camera_ids=set(sources))
            frame_hub = FrameHub(jpeg_quality=50)
            frame_hub.register_cameras(sources)
            alert_manager = AlertManager(
                alert_log=temp_dir / "alerts.jsonl",
                notifier=None,
                broadcaster=AlertBroadcaster(delivery_enabled=False),
                identity_store=store,
                screenshot_dir=temp_dir / "screenshots",
                database=database,
            )
            calibrator = TransitCalibrator(temp_dir / "transit_stats.json",
                                           valid_edges=topology.edges())

            created_sims: list[float] = []
            statuses = {"matched": 0, "created": 0, "ambiguous": 0, "invalid": 0}
            orig_register = store.register_if_new

            def spy_register(feat):
                resolution = orig_register(feat)
                statuses[resolution.status] = statuses.get(resolution.status, 0) + 1
                if resolution.status == "created":
                    created_sims.append(float(resolution.best_similarity))
                return resolution

            store.register_if_new = spy_register
            identities_at_start = len(store.all_ids())

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
                    reid_every_n=1,
                    frame_rate_cap=10.0,
                    process_max_width=960,
                    personnel=None,
                )
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
            alert_manager.close()
            calibrator.flush()

            metrics = store.get_metrics()
            frames = sum(p._frame_idx for p in pipelines)
            result = {
                "policy": _args.policy,
                "feature_window_size": FEATURE_WINDOW_SIZE,
                "seed_db": (str(seed_db) if seed_db is not None else None),
                "identities_at_start": identities_at_start,
                "seconds": seconds,
                "elapsed_s": round(elapsed, 1),
                "cameras": len(sources),
                "processed_frames": frames,
                "mean_fps": round(frames / elapsed / max(1, len(pipelines)), 2),
                "statuses": statuses,
                "created_best_sims": [round(s, 4) for s in created_sims],
                "metrics": {k: metrics.get(k) for k in (
                    "gallery_size", "total_searches", "successful_matches",
                    "match_rate", "ratio_blocked_count", "avg_top1_similarity",
                    "avg_ratio_margin", "avg_feature_quality",
                    "created_near_miss", "avg_created_best_similarity",
                    "collapse_warnings", "center_enabled", "center_norm")},
            }
            result["duplicates"] = duplicate_stats(store)
            return result
        finally:
            database.close()


def print_comparison(runs: list[dict]) -> None:
    """
    只比较**同条件**的运行（相同起始库 / 时长 / 相机数）—— 否则拿"空库 20 秒"和
    "生产库 300 秒"并列，数字好看但没有意义。
    """
    if len(runs) < 2:
        return
    latest = runs[-1]
    comparable = [r for r in runs
                  if r.get("seed_db") == latest.get("seed_db")
                  and r.get("seconds") == latest.get("seconds")
                  and r.get("cameras") == latest.get("cameras")]
    if len(comparable) < 2:
        log(f"\n（当前只有 {len(comparable)} 个可比运行：起始库={latest.get('seed_db')} "
            f"{latest.get('seconds')}s {latest.get('cameras')} 路；"
            "跑完另一臂才会打印对照）")
        return
    runs = comparable
    log("\n" + "=" * 78)
    log(f"对照（起始库={latest.get('seed_db')} | {latest.get('seconds')}s | "
        f"{latest.get('cameras')} 路，只改主特征聚合策略）")
    log("-" * 78)
    header = f"{'指标':<28}" + "".join(
        f"{r['policy']}(N={r['feature_window_size']:>2})".rjust(16) for r in runs)
    log(header)
    rows = [
        ("起始身份数", lambda r: r.get("identities_at_start")),
        ("结束身份数", lambda r: r["duplicates"]["identities"]),
        ("★ 新增身份数", lambda r: r["duplicates"]["identities"]
         - (r.get("identities_at_start") or 0)),
        ("新建身份数 created", lambda r: r["statuses"]["created"]),
        ("created_near_miss(≥0.5)", lambda r: r["metrics"]["created_near_miss"]),
        ("新建时最近邻相似度均值", lambda r: r["metrics"]["avg_created_best_similarity"]),
        ("匹配到已有身份 matched", lambda r: r["statuses"]["matched"]),
        ("Ratio 判歧义 ambiguous", lambda r: r["statuses"]["ambiguous"]),
        ("match_rate", lambda r: r["metrics"]["match_rate"]),
        ("Ratio 阻断占比", lambda r: round(r["metrics"]["ratio_blocked_count"]
                                          / max(1, r["metrics"]["total_searches"]), 4)),
        ("avg_ratio_margin", lambda r: r["metrics"]["avg_ratio_margin"]),
        ("avg_top1_similarity", lambda r: r["metrics"]["avg_top1_similarity"]),
        ("avg_feature_quality", lambda r: r["metrics"]["avg_feature_quality"]),
        ("越阈重复对 %", lambda r: r["duplicates"]["over_threshold_pct"]),
        ("越阈重复对（个数）", lambda r: r["duplicates"]["over_threshold"]),
        ("异人对余弦 p50", lambda r: r["duplicates"]["p50"]),
        ("每路 fps", lambda r: r["mean_fps"]),
    ]
    for label, getter in rows:
        log(f"{label:<28}" + "".join(str(getter(r)).rjust(16) for r in runs))


def main() -> int:
    import torch

    # 与 main.py:263-264 同款线程钳制，否则帧率不可与生产比较
    import cv2 as _cv2
    _cv2.setNumThreads(1)
    torch.set_num_threads(2)

    device = _args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sources_all = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    cameras = _args.cameras or list(sources_all)
    sources = {c: str(ROOT / sources_all[c]) for c in cameras if sources_all.get(c)}
    if not sources:
        log("没有可用的相机")
        return 1

    log(f"设备 {device} | {len(sources)} 路 | {_args.seconds:.0f}s | "
        f"策略 {_args.policy} | FEATURE_WINDOW_SIZE={FEATURE_WINDOW_SIZE}")
    if _args.policy == "window":
        probe = _aggregate_window_feature([(np.ones(4, dtype=np.float32), 1.0),
                                           (np.ones(4, dtype=np.float32), 3.0)])
        assert probe is not None and abs(float(np.linalg.norm(probe)) - 1.0) < 1e-5

    from src.detector import PersonDetector
    from src.reid import build_reid_extractor
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4, device=device)
    reid_extractor = build_reid_extractor(device=device)
    log(f"特征空间 {reid_extractor.feature_space}")

    seed_db: Path | None
    if _args.seed_db == "none":
        seed_db = None
    elif _args.seed_db == "production":
        seed_db = ROOT / "outputs" / "lab_monitor.db"
    else:
        seed_db = Path(_args.seed_db)
    log(f"起始身份库: {seed_db if seed_db else '空库'}")

    result = run_arm(sources, _args.seconds, detector, reid_extractor, device, seed_db)
    result["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"  {result['identities_at_start']} → {result['duplicates']['identities']} 个身份 "
        f"(+{result['duplicates']['identities'] - result['identities_at_start']}) | "
        f"created={result['statuses']['created']} "
        f"matched={result['statuses']['matched']} "
        f"ambiguous={result['statuses']['ambiguous']} | "
        f"near_miss={result['metrics']['created_near_miss']} | "
        f"越阈重复对={result['duplicates']['over_threshold_pct']}% | "
        f"每路 {result['mean_fps']} fps")
    for sim in result["created_best_sims"]:
        log(f"    新建时最近邻相似度 {sim}")

    out = ROOT / "outputs" / "reports" / "feature_window_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            history = data.get("runs", []) if isinstance(data, dict) else data
            if not isinstance(history, list):
                history = []
        except json.JSONDecodeError:
            history = []
    history = [r for r in history
               if not (r.get("policy") == result["policy"]
                       and r.get("feature_window_size") == result["feature_window_size"]
                       and r.get("seed_db") == result["seed_db"]
                       and r.get("seconds") == result["seconds"]
                       and r.get("cameras") == result["cameras"])]
    history.append(result)
    out.write_text(json.dumps({"runs": history}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log(f"结果已写入 {out}")
    print_comparison(history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
