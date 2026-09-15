"""render_cross_camera_strip.py — 生成指定身份的「跨相机抓拍序列」对比图（CLI）。

裁图与拼图的**唯一实现**在 `src/snapshots.py`，本脚本只负责命令行包装。
服务端同款能力见 `GET /api/identities/{gid}/snapshots`。

为什么不用平面图轨迹
--------------------
`config/camera_map.json` 的 `map_xy` 全为 null，平面图轨迹需要先标点位；
且素材循环播放使相机序列呈 A↔B 高频交替（用 --report 可看），直接连折线会得到
锯齿而非行走路线。抓拍序列按相机分组取代表帧，不受该问题影响。

用法
----
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --list
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --report --gid 5a7991c7
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --gid 5a7991c7 --gid d1742abb
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --gid 5a7991c7 --force

输出：outputs/identity_snapshots/<gid>_strip.jpg（拼图）与 <gid>_<cam>.jpg（单张裁图）。

注意
----
- `bbox_json` 坐标系是**处理帧**（960×540），不是源片分辨率，详见 `src/snapshots.py`
- **生成后必须人工目视核对**：当前身份库存在欠归并（部分身份可能是不同的人），
  演示前务必确认各路相机中确为同一人
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import Database  # noqa: E402
from src.snapshots import build_identity_snapshots  # noqa: E402

# 一律用绝对路径：相对路径会让 build_identity_snapshots 落盘的文件名带上 cwd 前缀，
# 服务端的 URL 拼接与 relative_to() 都会因此出问题。
DB_PATH = ROOT / "outputs" / "lab_monitor.db"
LOW_DIR = ROOT / "videos_low"
OUT_DIR = ROOT / "outputs" / "identity_snapshots"

def list_candidates(conn, min_cams: int = 3, limit: int = 20) -> None:
    """列出跨相机数较多的身份，供挑选演示对象。"""
    rows = conn.execute(
        """
        SELECT global_id, COUNT(DISTINCT camera_id) nc, COUNT(*) n
        FROM identity_appearances GROUP BY global_id
        HAVING nc >= ? ORDER BY nc DESC, n DESC LIMIT ?
        """,
        (min_cams, limit),
    ).fetchall()
    print(f"{'gid':<12}{'相机数':>6}{'轨迹数':>10}  相机链（按首次出现排序）")
    for gid, nc, n in rows:
        cams = [
            r[0] for r in conn.execute(
                "SELECT camera_id FROM identity_appearances WHERE global_id=? "
                "GROUP BY camera_id ORDER BY MIN(timestamp)",
                (gid,),
            )
        ]
        print(f"{gid:<12}{nc:>6}{n:>10}  {' → '.join(cams)}")


def report_sequence(conn, gids: list[str]) -> None:
    """诊断相机序列质量：切换次数远多于相机数说明存在抖动，不宜直接连折线。"""
    for gid in gids:
        rows = conn.execute(
            "SELECT camera_id FROM identity_appearances WHERE global_id=? ORDER BY timestamp",
            (gid,),
        ).fetchall()
        if not rows:
            print(f"{gid}: 无轨迹")
            continue
        seq: list[str] = []
        for (cam,) in rows:
            if not seq or seq[-1] != cam:
                seq.append(cam)
        uniq = len(set(seq))
        verdict = "序列干净" if len(seq) <= uniq * 2 else "存在抖动（由 merge_flicker_segments 压成视图组）"
        print(f"{gid}: 轨迹 {len(rows)} 行 / 相机 {uniq} 路 / 切换 {len(seq)} 次 → {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gid", action="append", default=[], help="身份 global_id，可重复")
    ap.add_argument("--list", action="store_true", help="列出跨相机候选身份")
    ap.add_argument("--report", action="store_true", help="诊断相机序列质量")
    ap.add_argument("--force", action="store_true", help="忽略缓存重新生成")
    ap.add_argument("--max-cameras", type=int, default=8)
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"身份库不存在: {DB_PATH}")
        return 1

    if args.list or args.report:
        import sqlite3
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            if args.list:
                list_candidates(conn)
            if args.report:
                report_sequence(conn, args.gid)
        finally:
            conn.close()
        return 0

    if not args.gid:
        ap.print_help()
        return 1

    desc_path = ROOT / "config" / "camera_map.json"
    desc_of = {}
    if desc_path.exists():
        try:
            raw = json.loads(desc_path.read_text(encoding="utf-8"))
            desc_of = {k: (v.get("desc") or "") for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            pass

    database = Database(DB_PATH)
    try:
        for gid in args.gid:
            print(f"=== {gid} ===")
            result = build_identity_snapshots(
                database, gid, OUT_DIR, LOW_DIR, desc_of,
                max_cameras=args.max_cameras, force=args.force,
            )
            print(f"  {result['camera_count']} 路相机（cached={result['cached']}）")
            for item in result["cameras"]:
                print(f"    {item['camera']:<9}{item['desc'][:22]:<24}"
                      f"video_ts={item['video_ts']:<8.1f}候选帧={item['frames']}")
            if result["skipped"]:
                for cam, reason in result["skipped"].items():
                    print(f"    跳过 {cam}: {reason}")
            if result["strip_file"]:
                print(f"  -> {Path(result['strip_file']).relative_to(ROOT)}")
    finally:
        database.close()

    print("\n提醒：请目视核对每张对比图，确认各路相机中确为同一人后再用于演示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
