"""transcode_lowres.py — 离线把源视频转码到低分辨率（worklist 2.5）。

定位要说清楚（2026-09-12 实测后的结论）
--------------------------------------
吞吐瓶颈是**单进程 GIL**，不是解码（22 路实测 38 帧/s，GPU 只用 35~40%），
所以降分辨率对吞吐的提升有限（预期 +10~20%）。真正的价值是：
  1. 体积 983MB → 29MB（实测，压缩 34 倍），语料可随项目归档/分发（videos_low/ 已 gitignore）；
  2. **修复损坏的 fps 元数据**：rnd_05 的 r_frame_rate 报 351.56，
     转码时统一 -r 25 可一并修正 —— 这对"视频内时间"换算（worklist 2.3）是实打实的收益；
  3. 为将来接入 RTSP 在线流铺路（在线流无法离线转码，只能靠运行时 process_max_width）。

目标分辨率 960x540 的依据：YOLOv8n 内部 resize 到 640、OSNet 输入 256x128，
模型侧的有效分辨率上限由它们决定，往下游喂 1080p 纯属浪费解码与带宽。
960/2560*1440 = 540，960/1920*1080 = 540 —— 两种源分辨率都**恰好**落在 540p，无黑边。

产物是 ffmpeg 刚写出的标准 mp4，元数据可信，会自动登记进 video_assets
（rel_path 指向 videos_low/，与源文件是不同的行，靠 UNIQUE(camera_id, rel_path) 区分）。

用法：
    ./.venv/Scripts/python.exe scripts/transcode_lowres.py --dry-run
    ./.venv/Scripts/python.exe scripts/transcode_lowres.py --only reg_01 rnd_02
    ./.venv/Scripts/python.exe scripts/transcode_lowres.py --seed-only   # 只补资产索引
    ./.venv/Scripts/python.exe scripts/transcode_lowres.py --scale 1280:720 --crf 26
    ./.venv/Scripts/python.exe scripts/transcode_lowres.py --switch-sources
"""

from __future__ import annotations

import argparse
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

LOW_DIR = ROOT / "videos_low"
FPS_TARGET = 25.0
LOW_VALUE_SECONDS = 5.0


def parse_scale(scale: str) -> tuple[int, int]:
    width_text, _, height_text = scale.partition(":")
    width = int(width_text)
    height = int(height_text) if height_text and height_text != "-2" else 0
    return width, height


def transcode(src: Path, dst: Path, scale: str, crf: str) -> bool:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale={scale}",
        "-c:v", "libx264", "-crf", crf, "-preset", "medium",
        "-r", "25",                 # 统一帧率：顺带修复 rnd_05 损坏的 351.56 元数据
        "-an",                      # 监控素材无音轨需求，省码率
        "-movflags", "+faststart",  # 便于网络播放/拖动
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"      ✗ ffmpeg 失败: {(result.stderr or '').strip()[:200]}")
        return False
    return True


def probe_quick(path: Path) -> tuple[int, float]:
    """转码产物是 ffmpeg 刚写出来的标准 mp4，元数据可信，可直接用 ffprobe 读时长。"""
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    raw = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames",
         "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    data = json.loads(raw.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    try:
        duration = float((data.get("format") or {}).get("duration") or 0)
    except ValueError:
        duration = 0.0
    return int(stream.get("nb_frames") or 0), duration


def seed_asset(database: Database, camera_id: str, rel_path: str,
               width: int, height: int) -> None:
    dst = ROOT / rel_path
    if not dst.exists():
        print(f"      ⚠ {dst} 不存在，无法登记")
        return
    frames, duration = probe_quick(dst)
    database.seed_video_asset(
        camera_id=camera_id,
        rel_path=rel_path,
        file_name=dst.name,
        codec="h264",
        width=width,
        height=height or None,
        fps_declared=FPS_TARGET,
        frames_real=frames,
        duration_real=round(duration, 3),
        low_value=1 if 0 < duration < LOW_VALUE_SECONDS else 0,
    )
    print(f"      + 资产索引已登记（{frames} 帧 / {duration:.1f}s）")


def main() -> int:
    parser = argparse.ArgumentParser(description="离线转码到低分辨率")
    parser.add_argument("--scale", default="960:-2", help="目标分辨率（默认 960:-2）")
    parser.add_argument("--crf", default="28", help="质量（默认 28）")
    parser.add_argument("--only", nargs="*", help="只转指定相机")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="已存在的也重新转码")
    parser.add_argument("--seed-only", action="store_true",
                        help="不转码，只为已存在的 videos_low/*.mp4 登记资产索引")
    parser.add_argument("--switch-sources", action="store_true",
                        help="全部转完后把 sources.json 指向 videos_low/（原文件另存 sources_high.json）")
    args = parser.parse_args()

    sources_path = ROOT / "config" / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    cameras = {k: v for k, v in sources.items() if not args.only or k in args.only}
    LOW_DIR.mkdir(exist_ok=True)
    database = Database(ROOT / "outputs" / "lab_monitor.db")
    width, height = parse_scale(args.scale)

    if args.seed_only:
        seeded = 0
        for camera_id in cameras:
            rel_path = f"videos_low/{camera_id}.mp4"
            existing = database.get_video_asset(camera_id)
            already = (existing and existing.get("rel_path") == rel_path
                       and existing.get("frames_real"))
            if already:
                print(f"{camera_id}: 已登记，跳过")
                continue
            before = len(database.list_video_assets())
            seed_asset(database, camera_id, rel_path, width, height)
            if len(database.list_video_assets()) > before:
                seeded += 1
        database.close()
        print(f"\n共登记 {seeded} 行")
        return 0

    print(f"目标 scale={args.scale} crf={args.crf}，共 {len(cameras)} 路\n")
    done, failed, skipped = 0, [], 0
    started_all = time.monotonic()

    for index, (camera_id, rel_path) in enumerate(cameras.items(), start=1):
        src = ROOT / rel_path
        dst = LOW_DIR / f"{camera_id}.mp4"
        if dst.exists() and not args.refresh:
            print(f"[{index:>2}/{len(cameras)}] {camera_id}: 已存在，跳过（--refresh 重转）")
            if not args.dry_run:
                seed_asset(database, camera_id,
                           str(dst.relative_to(ROOT)).replace("\\", "/"), width, height)
            skipped += 1
            continue
        if not src.exists():
            print(f"[{index:>2}/{len(cameras)}] {camera_id}: 源文件缺失")
            failed.append(camera_id)
            continue

        src_mb = src.stat().st_size / 1048576
        print(f"[{index:>2}/{len(cameras)}] {camera_id}: {src_mb:.1f}MB → {dst.name}")
        if args.dry_run:
            continue
        started = time.monotonic()
        if not transcode(src, dst, args.scale, args.crf):
            failed.append(camera_id)
            continue
        dst_mb = dst.stat().st_size / 1048576
        frames, duration = probe_quick(dst)
        print(f"      ✓ {src_mb:.1f}MB → {dst_mb:.1f}MB（压缩 {src_mb / max(dst_mb, 0.01):.1f}x）"
              f"{frames} 帧 / {duration:.1f}s，耗时 {time.monotonic() - started:.0f}s")
        seed_asset(database, camera_id,
                   str(dst.relative_to(ROOT)).replace("\\", "/"), width, height)
        done += 1

    database.close()
    print("\n" + "=" * 62)
    print(f"转码 {done} 路 / 跳过 {skipped} 路 / 失败 {len(failed)} 路，"
          f"总耗时 {time.monotonic() - started_all:.0f}s")
    if failed:
        print(f"⚠ 失败: {failed}")
        return 1

    if args.switch_sources:
        backup = sources_path.with_name("sources_high.json")
        if not backup.exists():
            backup.write_text(json.dumps(sources, ensure_ascii=False, indent=2),
                              encoding="utf-8")
            print(f"原配置已备份: {backup}")
        switched = {cam: f"videos_low/{cam}.mp4" for cam in sources}
        sources_path.write_text(json.dumps(switched, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print("sources.json 已切换到 videos_low/（重启服务生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
