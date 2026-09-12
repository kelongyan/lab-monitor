"""实解码统计各路视频的真实帧数与时长（容器元数据不可信，只能数帧）。

判据来源：src/../docs 与实测均显示这些素材的 container duration / nb_frames 缺失或
严重失真（12000~71617 秒，实际素材只有几十秒到几分钟）。本脚本用
`ffprobe -count_frames` 真实解码每一路，得到可用于方案设计的时长基线。
只读，不修改任何文件。
"""

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def count_frames(path: Path, fps: float) -> tuple[int, float]:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(path),
    ]
    raw = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    text = (raw.stdout or "").strip().splitlines()
    try:
        frames = int(text[-1])
    except (ValueError, IndexError):
        return -1, -1.0
    return frames, (frames / fps if fps > 0 else 0.0)


def main() -> None:
    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    print(f"{'cam':8s} {'res':11s} {'fps':7s} {'real_frames':12s} {'real_dur':10s} {'size'}")
    print("-" * 62)
    total_frames = 0
    total_dur = 0.0
    for cam_id, rel in sources.items():
        path = ROOT / rel
        if not path.exists():
            print(f"{cam_id:8s} MISSING")
            continue
        meta = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        stream = ((json.loads(meta.stdout or "{}").get("streams") or [{}])[0])
        width, height = stream.get("width", 0), stream.get("height", 0)
        num, _, den = (stream.get("r_frame_rate") or "0/0").partition("/")
        try:
            fps = float(num) / float(den) if float(den or 0) else 0.0
        except ValueError:
            fps = 0.0
        frames, duration = count_frames(path, fps)
        if frames < 0:
            print(f"{cam_id:8s} {width}x{height:<6d} {fps:<7.2f} COUNT_FAILED")
            continue
        total_frames += frames
        total_dur += duration
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"{cam_id:8s} {str(width) + 'x' + str(height):11s} {fps:<7.2f} "
              f"{frames:<12d} {duration:<10.1f} {size_mb:.1f}MB")
    print("-" * 62)
    print(f"合计 {total_frames} 帧 / 真实总时长 {total_dur:.0f}s = {total_dur / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
