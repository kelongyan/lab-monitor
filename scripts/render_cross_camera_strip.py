"""render_cross_camera_strip.py — 生成指定身份的「跨相机抓拍序列」对比图。

用途
----
把某个全局身份在各路相机上的真实抓拍拼成一张带标注的对比图，用于演示
"同一个人出现在多路相机"。图像来源是 `videos_low/` 的低清素材 + 轨迹表里的
`video_ts` 与 `bbox_json`，**全部来自真实帧**，不做任何合成。

为什么不用平面图轨迹
--------------------
`config/camera_map.json` 的 `map_xy` 全为 null，平面图轨迹需要先标点位；
且素材循环播放使相机序列呈 A↔B 高频交替（见 --report 输出），直接连折线会得到
锯齿而非行走路线。抓拍序列按相机分组取代表帧，不受该问题影响。

用法
----
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --gid d1742abb
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --gid 23c29863 --gid 5a7991c7
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --list      # 候选身份排行
    ./.venv/Scripts/python.exe scripts/render_cross_camera_strip.py --report    # 序列质量诊断

输出：outputs/dev/demo_strips/<gid>_strip.jpg 与 <gid>_tiles/<cam>.jpg

注意
----
- `bbox_json` 的坐标系是**处理帧**（960×540，即 `process_max_width=960`），
  不是源片分辨率（1920×1080 / 2560×1440）。低清素材恰好也是 960×540，因此坐标可直接用。
- 选取代表帧时优先「框面积大且不贴画面边缘」的样本，避免取到残缺的半身框。
- **生成后必须人工目视核对**：当前身份库存在欠归并，个别身份可能混入不同的人。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "outputs" / "lab_monitor.db"
LOW_DIR = ROOT / "videos_low"
OUT_DIR = ROOT / "outputs" / "dev" / "demo_strips"

TILE_H = 360
PAD = 14
LABEL_H = 30
BG = (245, 245, 245)
FG = (30, 30, 30)
ACCENT = (150, 80, 10)


def _db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"找不到生产库: {DB_PATH}")
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def list_candidates(conn: sqlite3.Connection, min_cams: int = 3, limit: int = 20) -> None:
    """列出跨相机数较多的身份，供挑选演示对象。"""
    rows = conn.execute(
        """
        SELECT global_id, COUNT(DISTINCT camera_id) nc, COUNT(*) n
        FROM identity_appearances GROUP BY global_id
        HAVING nc >= ? ORDER BY nc DESC, n DESC LIMIT ?
        """,
        (min_cams, limit),
    ).fetchall()
    print(f"{'gid':<12}{'相机数':>6}{'轨迹数':>10}  相机链")
    for gid, nc, n in rows:
        cams = [
            r[0]
            for r in conn.execute(
                "SELECT camera_id FROM identity_appearances WHERE global_id=? "
                "GROUP BY camera_id ORDER BY MIN(timestamp)",
                (gid,),
            )
        ]
        print(f"{gid:<12}{nc:>6}{n:>10}  {' → '.join(cams)}")


def report_sequence(conn: sqlite3.Connection, gids: list[str]) -> None:
    """诊断相机序列质量：切换次数过多说明是循环素材造成的 A↔B 交替，不适合画路线。"""
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
        print(
            f"{gid}: 轨迹 {len(rows)} 行 / 相机 {uniq} 路 / 切换 {len(seq)} 次 → "
            + ("序列干净" if len(seq) <= uniq * 2 else "存在高频交替，不适合直接连折线")
        )


def _pick_representative(conn: sqlite3.Connection, gid: str, cam: str) -> tuple[float, list[float]] | None:
    """选该相机下最适合作图的样本：框面积最大且不贴画面边缘偏好居中。"""
    rows = conn.execute(
        "SELECT video_ts, bbox_json FROM identity_appearances "
        "WHERE global_id=? AND camera_id=? AND video_ts IS NOT NULL AND bbox_json IS NOT NULL",
        (gid, cam),
    ).fetchall()
    best = None
    best_score = -1.0
    for vt, bbox_json in rows:
        try:
            x1, y1, x2, y2 = json.loads(bbox_json)[:4]
        except Exception:
            continue
        w, h = x2 - x1, y2 - y1
        if w < 24 or h < 48:
            continue
        area = w * h
        margin = min(x1, y1, 960 - x2, 540 - y2)
        edge_penalty = 1.0 if margin < 2 else 0.0
        # 竖长比接近人体的样本优先（排除横长条误检）
        ratio = h / max(w, 1e-6)
        ratio_bonus = 1.0 if ratio >= 1.2 else 0.0
        score = area * (1.0 - 0.6 * edge_penalty) * (1.0 + 0.5 * ratio_bonus)
        if score > best_score:
            best_score = score
            best = (float(vt), [float(x1), float(y1), float(x2), float(y2)])
    return best


def _grab(cam: str, video_ts: float, bbox: list[float]) -> np.ndarray | None:
    path = LOW_DIR / f"{cam}.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ts = max(0.0, min(video_ts, (n_frames - 2) / fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    # bbox 在处理帧（960×540）坐标系下；若素材尺寸不同则等比换算
    sx, sy = w / 960.0, h / 540.0
    x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
    # 外扩 12% 留出上下文，便于目视认人
    pad_x = int((x2 - x1) * 0.12)
    pad_y = int((y2 - y1) * 0.12)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def _fit(img: np.ndarray, box_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = box_h / h
    return cv2.resize(img, (max(1, int(w * scale)), box_h), interpolation=cv2.INTER_AREA)


def _put_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.5, color=FG) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def build_strip(gid: str, conn: sqlite3.Connection, desc_of: dict) -> Path | None:
    cams = [
        r[0]
        for r in conn.execute(
            "SELECT camera_id FROM identity_appearances WHERE global_id=? "
            "GROUP BY camera_id ORDER BY MIN(timestamp)",
            (gid,),
        )
    ]
    if not cams:
        print(f"{gid}: 无轨迹，跳过")
        return None

    tiles = []
    for cam in cams:
        picked = _pick_representative(conn, gid, cam)
        if picked is None:
            print(f"  {cam}: 无可用样本，跳过")
            continue
        video_ts, bbox = picked
        img = _grab(cam, video_ts, bbox)
        if img is None:
            print(f"  {cam}: 解码失败，跳过")
            continue
        tile = _fit(img, TILE_H)
        tiles.append((cam, tile, video_ts))
        print(f"  {cam}: video_ts={video_ts:.1f}s 裁图 {img.shape[1]}x{img.shape[0]}")

    if not tiles:
        return None

    total_w = PAD + sum(t.shape[1] + PAD for _, t, _ in tiles)
    canvas = np.full((TILE_H + LABEL_H + PAD * 2, total_w, 3), BG, dtype=np.uint8)
    x = PAD
    for cam, tile, video_ts in tiles:
        canvas[PAD : PAD + TILE_H, x : x + tile.shape[1]] = tile
        cv2.rectangle(canvas, (x, PAD), (x + tile.shape[1], PAD + TILE_H), (200, 200, 200), 1)
        _put_text(canvas, cam, x, PAD + TILE_H + 20, 0.55, ACCENT)
        desc = (desc_of.get(cam) or "")[:14]
        if desc:
            _put_text(canvas, desc, x + 62, PAD + TILE_H + 20, 0.45, FG)
        x += tile.shape[1] + PAD

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tile_dir = OUT_DIR / f"{gid}_tiles"
    tile_dir.mkdir(exist_ok=True)
    for cam, tile, _ in tiles:
        cv2.imwrite(str(tile_dir / f"{cam}.jpg"), tile)

    out = OUT_DIR / f"{gid}_strip.jpg"
    cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"  -> {out.relative_to(ROOT)}")
    return out


def _desc_map() -> dict:
    path = ROOT / "config" / "camera_map.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gid", action="append", default=[], help="身份 global_id，可重复")
    ap.add_argument("--list", action="store_true", help="列出跨相机候选身份")
    ap.add_argument("--report", action="store_true", help="诊断相机序列质量")
    args = ap.parse_args()

    conn = _db()
    if args.list:
        list_candidates(conn)
        return 0
    if args.report:
        report_sequence(conn, args.gid or [])
        return 0
    if not args.gid:
        ap.print_help()
        return 1

    desc_raw = _desc_map()
    desc_of = {k: (v.get("desc") if isinstance(v, dict) else "") for k, v in desc_raw.items()}
    for gid in args.gid:
        print(f"=== {gid} ===")
        build_strip(gid, conn, desc_of)
    print("\n提醒：请目视核对每张对比图，确认各路相机中确为同一人后再用于演示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
