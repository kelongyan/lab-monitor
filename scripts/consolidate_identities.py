"""consolidate_identities.py — 对正式身份库执行身份归并（worklist 1.8）。

背景
----
中心化的冷启动护栏（有效特征 < 8 时不启用）是必要的安全设计，但带来一个副作用：
服务刚启动、gallery 还很小时，早期身份是在**未中心化**的空间里注册的；
之后中心化生效，却没有任何回溯归并机制 —— 同一个人被拆成的多个身份永远不会合并。

实测（2026-09-12 重建后 22 个身份）：a093d0a6(rnd_01) 与 rnd_02/rnd_05/rnd_11/rnd_17/rnd_22
的 5 个身份中心化相似度 = 1.0000，全部在同一条走廊链上，极大概率是同一人。

本脚本以**生产语义**（中心化 + 重新归一化 + max-over-bank）找出候选并归并。
归并把 identity_appearances 的 global_id 一并改挂到主身份，避免轨迹断链。

用法：
    ./.venv/Scripts/python.exe scripts/consolidate_identities.py --dry-run
    ./.venv/Scripts/python.exe scripts/consolidate_identities.py            # 真正归并
    ./.venv/Scripts/python.exe scripts/consolidate_identities.py --threshold 0.85

⚠️ 会修改正式库 outputs/lab_monitor.db，执行前自动备份到
   outputs/lab_monitor.db.bak-consolidate-<时间戳>（--no-backup 可跳过）。
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import Database  # noqa: E402
from src.identity_store import IdentityStore  # noqa: E402
from src.reid_config import DEFAULT_REID_WEIGHTS, REID_MATCH_THRESHOLD, get_reid_weight  # noqa: E402

DB = ROOT / "outputs" / "lab_monitor.db"


def main() -> int:
    parser = argparse.ArgumentParser(description="对正式身份库执行身份归并")
    parser.add_argument("--threshold", type=float, default=REID_MATCH_THRESHOLD,
                        help=f"归并阈值（默认与线上匹配阈值一致 {REID_MATCH_THRESHOLD}）")
    parser.add_argument("--dry-run", action="store_true", help="只列出候选，不修改")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    if not DB.exists():
        print(f"身份库不存在: {DB}")
        return 1

    if not args.dry_run and not args.no_backup:
        target = DB.with_name(f"{DB.name}.bak-consolidate-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(DB, target)
        print(f"已备份: {target}")

    database = Database(DB)
    weight = get_reid_weight()
    store = IdentityStore(database=database, feature_space=weight.feature_space)
    print(f"特征空间 {weight.feature_space}（权重 {weight.key}）| 归并阈值 {args.threshold}")

    before = len(store.all_ids())
    result = store.consolidate(threshold=args.threshold, dry_run=args.dry_run)
    store.flush()
    database.close()

    print(f"\n归并前身份 {before} 个")
    if result["dry_run"]:
        print(f"候选 {len(result['candidates'])} 组（未做任何修改）：")
        for item in result["candidates"]:
            print(f"   {item['keep']} <- {item['merge']}  sim={item['similarity']}")
        print("\n确认无误后去掉 --dry-run 再执行。")
        return 0

    print(f"实际归并 {result['merged_count']} 组，归并后身份 {len(store.all_ids())} 个：")
    for item in result["merged"]:
        print(f"   {item['keep']} <- {item['merge']}  sim={item['similarity']}  "
              f"合并后 appearances={item['kept_appearances']}")
    remaining = result["candidates"]
    print(f"\n归并后仍高于阈值的对 {len(remaining)} 组：")
    for item in remaining[:8]:
        print(f"   {item['keep']} <-> {item['merge']}  sim={item['similarity']}")

    out = ROOT / "outputs" / "reports" / "consolidation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"threshold": args.threshold, "before": before,
         "after": len(store.all_ids()), **{k: v for k, v in result.items()}},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
