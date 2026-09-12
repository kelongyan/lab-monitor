"""seed_video_assets.py — 用 ffprobe 实测写入视频资产索引（worklist 2.1 / 2.6）。

为什么必须有这一步
------------------
这批素材的**容器元数据不可信**：ffprobe 与 OpenCV 一致报出 11948~71617 秒的荒谬时长、
`nb_frames` 缺失（合计报 599,232 秒，实解只有约 45 分钟，虚高 222 倍）。
而"人员视频检索"要能回答"出现在哪个文件、第几秒"，就必须有可信的帧数与时长。

所以本脚本用 `ffprobe -count_frames` **真实解码**每一路来数帧（22 路约 10 分钟），
结果落进 `video_assets` 表。结果一旦写入就会复用（`--refresh` 才重新实测）。

fps 的选择规则（很重要）
------------------------
`r_frame_rate` 可能损坏（rnd_05 报 351.56）。选择顺序：
  1. `avg_frame_rate` 落在 [10, 60] → 用它
  2. 否则 `r_frame_rate` 落在 [10, 60] → 用它
  3. 否则回落 25.0（固定监控相机的标准帧率，本项目 21/22 路都是 25）
`duration_real = frames_real / 所选 fps`。rnd_05 用此规则得到 18.4 秒，
而非元数据欺骗出的 1.3 秒 —— 之前的文档数字据此修正。

low_value 判定：duration_real < 5 秒 → 内容过短，统计与检索默认排除
（不删除：采集点不能随意摘除）。

用法：
    ./.venv/Scripts/python.exe scripts/seed_video_assets.py            # 只补缺失的
    ./.venv/Scripts/python.exe scripts/seed_video_assets.py --refresh  # 全部重新实测
    ./.venv/Scripts/python.exe scripts/seed_video_assets.py --only rnd_05
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import Database  # noqa: E402

SANE_FPS = (10.0, 60.0)
FALLBACK_FPS = 25.0
LOW_VALUE_SECONDS = 5.0


def pick_fps(avg_rate: str, r_rate: str) -> float:
    def parse(value: str) -> float:
        try:
            num, _, den = value.partition("/")
            den_v = float(den) if den else 1.0
            return float(num) / den_v if den_v else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    for candidate in (parse(avg_rate), parse(r_rate)):
        if SANE_FPS[0] <= candidate <= SANE_FPS[1]:
            return candidate
    return FALLBACK_FPS


def probe(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    raw = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-count_frames",
         "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,avg_frame_rate",
         "-show_entries", "format=size",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=900,
    )
    data = json.loads(raw.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {
        "codec": stream.get("codec_name", ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "avg_frame_rate": stream.get("avg_frame_rate", "0/0"),
        "r_frame_rate": stream.get("r_frame_rate", "0/0"),
        "size_bytes": int((data.get("format") or {}).get("size") or 0),
    }


def count_frames(path: Path) -> int:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    raw = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=900,
    )
    text = (raw.stdout or "").strip().splitlines()
    try:
        return int(text[-1])
    except (ValueError, IndexError):
        return -1


def short_hash(path: Path) -> str:
    """sha1 前 8 位。全量哈希 983MB 约几秒，换全量以保证幂等判据可靠。"""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:8]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="实测并写入视频资产索引")
    parser.add_argument("--refresh", action="store_true",
                        help="已有实测值的也重新实测（默认只补缺失的）")
    parser.add_argument("--only", nargs="*", help="只处理指定相机")
    args = parser.parse_args()

    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    database = Database(ROOT / "outputs" / "lab_monitor.db")

    cameras = {k: v for k, v in sources.items() if not args.only or k in args.only}
    print(f"共 {len(cameras)} 路（-count_frames 实解码，22 路约需 10 分钟）\n")

    failures = []
    for index, (camera_id, rel_path) in enumerate(cameras.items(), start=1):
        existing = database.get_video_asset(camera_id)
        if (existing and existing.get("frames_real") and not args.refresh
                and (not args.only or existing.get("rel_path") == rel_path)):
            print(f"[{index:>2}/{len(cameras)}] {camera_id}: 已有实测值，跳过")
            continue

        path = ROOT / rel_path
        if not path.exists():
            print(f"[{index:>2}/{len(cameras)}] {camera_id}: 文件缺失 {rel_path}")
            failures.append(camera_id)
            continue

        started = time.monotonic()
        info = probe(path)
        frames_real = count_frames(path)
        fps = pick_fps(info["avg_frame_rate"], info["r_frame_rate"])
        duration = frames_real / fps if frames_real > 0 and fps > 0 else 0.0
        low_value = 1 if 0 < duration < LOW_VALUE_SECONDS else 0

        asset_id = database.seed_video_asset(
            camera_id=camera_id,
            rel_path=rel_path,
            file_name=path.name,
            sha1_8=short_hash(path),
            size_bytes=info["size_bytes"] or path.stat().st_size,
            width=info["width"],
            height=info["height"],
            codec=info["codec"],
            fps_declared=fps,
            frames_real=frames_real,
            duration_real=round(duration, 3),
            low_value=low_value,
        )
        flag = "  ← low_value" if low_value else ""
        print(f"[{index:>2}/{len(cameras)}] {camera_id}: {info['width']}x{info['height']} "
              f"{info['codec']} {frames_real} 帧 / {duration:.1f}s "
              f"(fps={fps:.2f}, {time.monotonic() - started:.0f}s){flag}")
        if frames_real < 0:
            failures.append(camera_id)

    assets = database.list_video_assets()
    database.close()

    total_frames = sum(a.get("frames_real") or 0 for a in assets)
    total_seconds = sum(a.get("duration_real") or 0.0 for a in assets)
    print("\n" + "=" * 62)
    print(f"资产索引 {len(assets)} 行 | 实测合计 {total_frames} 帧 / {total_seconds / 60:.1f} 分钟")
    if failures:
        print(f"⚠ 实测失败: {failures}")
        return 1
    print("全部就绪")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
