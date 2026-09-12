"""eval_identity_pipeline.py — 有界端到端身份识别验证（走真实链路，非模拟）。

为什么需要它
------------
`main.py` 是无限循环（本地素材自动循环播放），跑一轮约 30 分钟且不会自己停；
而"修复到底有没有效"需要的是**可重复、有界、走真实代码路径**的度量。
本脚本直接装配生产用的组件（PersonDetector / build_reid_extractor /
IdentityStore / CameraPipeline），跑指定秒数后停机并给出验收指标。

度量与判据
----------
1. `identities` / `unique_blobs`：唯一 feature_blob 数应等于身份数。
   库内历史数据是 45 行只有 25 个唯一值 —— 那是塌缩的指纹，重跑后不应复现。
2. `center_enabled` / `center_norm`：公共分量中心化是否生效。
3. `cross_p50` / `over_threshold`：各身份主特征两两余弦。中心化生效后 p50 应显著
   为负或接近 0，越阈比例应是**个位数百分比**（改造前实测 99.8%）。
4. `collapse_warnings`：塌缩护栏计数，应为 0。
5. `match_rate`：`successful_matches / total_searches`，越高说明越能复用已有身份
   而不是不断注册新身份。

用法
----
    ./.venv/Scripts/python.exe scripts/eval_identity_pipeline.py --seconds 180
    ./.venv/Scripts/python.exe scripts/eval_identity_pipeline.py --cameras rnd_08 rnd_19 --seconds 120
    ./.venv/Scripts/python.exe scripts/eval_identity_pipeline.py --weights market1501
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alerter import AlertManager  # noqa: E402
from src.db import Database  # noqa: E402
from src.detector import PersonDetector  # noqa: E402
from src.frame_hub import FrameHub  # noqa: E402
from src.identity_store import IdentityStore  # noqa: E402
from src.pipeline import CameraPipeline  # noqa: E402
from src.reid import build_reid_extractor  # noqa: E402
from src.topology import CameraTopology  # noqa: E402

#: 默认选"库里确实有身份数据"的相机，保证能在有限时间内采到足够样本
DEFAULT_CAMERAS = ["rnd_08", "rnd_19", "rnd_04", "rnd_16", "rnd_07"]


def build_components(device: str, weight_key: str | None, db_path: Path, work_dir: Path):
    """按 main.py 的口径装配真实组件（单实例，不用模型池 —— 评测只看正确性）。"""
    detector = PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4, device=device)
    reid = build_reid_extractor(device=device, weight_key=weight_key)
    database = Database(db_path)
    store = IdentityStore(database=database, feature_space=reid.feature_space)
    topology = CameraTopology(work_dir / "absent_topology.json", allowed_camera_ids=set())
    alerter = AlertManager(
        alert_log=work_dir / "alerts.jsonl",
        identity_store=store,
        screenshot_dir=work_dir / "screenshots",
        database=database,
    )
    frame_hub = FrameHub(jpeg_quality=50)
    return detector, reid, database, store, topology, alerter, frame_hub


def _pairwise_stats(vectors: list[np.ndarray]) -> dict:
    if len(vectors) < 2:
        return {}
    mat = np.stack(vectors)
    sims = mat @ mat.T
    off = np.array([sims[i, j] for i, j in combinations(range(len(mat)), 2)])
    return {
        "mean": round(float(off.mean()), 4),
        "p50": round(float(np.percentile(off, 50)), 4),
        "p95": round(float(np.percentile(off, 95)), 4),
        "over_threshold": round(float((off >= 0.75).mean()), 4),
        "pairs": int(off.size),
    }


def collect_results(db_path: Path) -> dict:
    """
    从落盘结果里取指标（读库而非内存态，避免把未落盘的状态算进来）。

    同时给出**原始**与**中心化后**两组余弦统计：
    - 原始：落库形态，直接算会显示虚高（公共分量未减）
    - 中心化后：与线上匹配时看到的一致，这才是验收依据
    """
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT global_id, feature_dim, feature_blob, feature_bank_count, feature_bank_blob "
        "FROM identities WHERE feature_dim > 0 AND feature_blob IS NOT NULL"
    ).fetchall()
    conn.close()

    mains, banks, digests = [], [], set()
    for _gid, dim, blob, bank_count, bank_blob in rows:
        if len(blob) != int(dim) * 4:
            continue
        dim = int(dim)
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        norm = float(np.linalg.norm(vector))
        if norm > 1e-8:
            mains.append(vector / norm)
        if bank_count and bank_blob:
            matrix = np.frombuffer(bank_blob, dtype=np.float32).reshape(int(bank_count), dim)
            for row in matrix:
                row_norm = float(np.linalg.norm(row))
                if row_norm > 1e-8:
                    banks.append(row / row_norm)
        digests.add(hash(blob))

    result = {"identities": len(rows), "unique_blobs": len(digests)}

    raw = _pairwise_stats(mains)
    if raw:
        result["raw"] = raw

    # 复刻 IdentityStore._feature_center_locked + _prepare_for_match 的规则
    all_vectors = mains + banks
    if len(all_vectors) >= 8:
        center = np.mean(np.stack(all_vectors), axis=0)
        if float(np.linalg.norm(center)) >= 0.35:
            prepared = []
            for vector in mains:
                diff = vector - center
                norm = float(np.linalg.norm(diff))
                prepared.append(diff / norm if norm > 1e-8 else vector)
            centered = _pairwise_stats(prepared)
            if centered:
                centered["center_norm"] = round(float(np.linalg.norm(center)), 4)
                result["centered"] = centered
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="有界端到端身份识别验证")
    parser.add_argument("--cameras", nargs="*", default=DEFAULT_CAMERAS)
    parser.add_argument("--seconds", type=float, default=180.0, help="运行时长（秒）")
    parser.add_argument("--weights", default=None, help="ReID 权重 key（默认取配置）")
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-db", default=None, help="把评测库复制到该路径以便复查")
    parser.add_argument("--verbose", action="store_true", help="打印各组件 INFO 日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    cameras = {
        cam: sources[cam] for cam in args.cameras if sources.get(cam)
    }
    if not cameras:
        print("没有可用的相机（检查 --cameras 是否在 sources.json 里）")
        return 1

    print(f"设备 {device} | 相机 {list(cameras)} | 运行 {args.seconds:.0f}s "
          f"| 权重 {args.weights or '默认'}")

    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        db_path = work_dir / "eval.db"
        (work_dir / "screenshots").mkdir(exist_ok=True)

        detector, reid, database, store, topology, alerter, frame_hub = build_components(
            device, args.weights, db_path, work_dir
        )
        print(f"特征空间: {reid.feature_space}")

        frame_hub.register_cameras(cameras)
        pipelines = []
        for cam, rel in cameras.items():
            pipeline = CameraPipeline(
                camera_id=cam,
                source=str(ROOT / rel),
                detector=detector,
                reid_extractor=reid,
                identity_store=store,
                topology=topology,
                alert_manager=alerter,
                screenshot_dir=work_dir / "screenshots",
                frame_hub=frame_hub,
                calibrator=None,
                detect_every_n=1,
                reid_every_n=10,
                frame_rate_cap=10.0,
            )
            pipelines.append(pipeline)
            pipeline.start()
        print(f"已启动 {len(pipelines)} 路流水线，开始计时…")

        deadline = time.monotonic() + args.seconds
        last_report = 0.0
        try:
            while time.monotonic() < deadline:
                time.sleep(1.0)
                remaining = max(0.0, deadline - time.monotonic())
                now = time.monotonic()
                if now - last_report >= 15.0:
                    last_report = now
                    alive = sum(1 for p in pipelines if p.is_alive())
                    print(f"  [{remaining:5.0f}s 剩余] 存活 {alive}/{len(pipelines)} | "
                          f"已注册身份 {len(store.all_ids())}", flush=True)
        except KeyboardInterrupt:
            print("用户中断，提前收尾")

        for pipeline in pipelines:
            pipeline.stop()
        for pipeline in pipelines:
            pipeline.join(timeout=10)

        # 关停路径要与 main.py 一致：先把节流中的特征补写落盘，再关连接
        store.flush()
        alerter.close()

        metrics = store.get_metrics()
        results = collect_results(db_path)
        results.update({
            "feature_space": reid.feature_space,
            "match_rate": metrics.get("match_rate"),
            "total_searches": metrics.get("total_searches"),
            "successful_matches": metrics.get("successful_matches"),
            "ratio_blocked_count": metrics.get("ratio_blocked_count"),
            "collapse_warnings": metrics.get("collapse_warnings"),
            "center_enabled": metrics.get("center_enabled"),
            "center_norm": metrics.get("center_norm"),
            "seconds": args.seconds,
            "cameras": list(cameras),
        })

        if args.keep_db:
            target = Path(args.keep_db)
            target.write_bytes(db_path.read_bytes())
            print(f"评测库已另存: {target}")

        database.close()

    print("\n" + "=" * 66)
    print("验收指标")
    print("-" * 66)
    verdict = "✓ 一致" if results["identities"] == results["unique_blobs"] else "✗ 存在重复 blob"
    print(f"  身份数 / 唯一 feature_blob 数   {results['identities']} / "
          f"{results['unique_blobs']}   {verdict}")
    print(f"  公共分量中心化                 "
          f"{'已启用' if results['center_enabled'] else '未启用（身份太少）'}"
          f"   中心范数 {results['center_norm']}")
    raw = results.get("raw")
    if raw:
        print(f"  原始特征余弦（落库形态）        mean {raw['mean']} / p50 {raw['p50']} / "
              f"p95 {raw['p95']}   越阈 {raw['over_threshold'] * 100:.1f}%   n={raw['pairs']}")
    centered = results.get("centered")
    if centered:
        print(f"  中心化后余弦（匹配所见）        mean {centered['mean']} / "
              f"p50 {centered['p50']} / p95 {centered['p95']}   "
              f"越阈 {centered['over_threshold'] * 100:.1f}%   n={centered['pairs']}")
    else:
        print("  中心化后余弦                   （样本不足，无法统计）")
    print(f"  匹配率 success/total           {results['successful_matches']} / "
          f"{results['total_searches']} = {results['match_rate']}")
    print(f"  Ratio 判歧义次数               {results['ratio_blocked_count']}")
    print(f"  塌缩护栏告警次数               {results['collapse_warnings']}"
          f"   {'✓' if not results['collapse_warnings'] else '✗ 特征仍在塌缩'}")
    if results["identities"] < 8:
        print("\n  ⚠ 身份数不足 8，两组余弦统计的样本量太小，结论仅供参考；"
              "请加大 --seconds 或增多 --cameras。")

    out = ROOT / "outputs" / "reports" / "identity_pipeline_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if out.exists():
        try:
            history = json.loads(out.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = [history]
        except json.JSONDecodeError:
            history = []
    history.append(results)
    out.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已追加到 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
