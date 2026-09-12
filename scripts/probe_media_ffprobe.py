"""用 ffprobe 取各路视频的「真实」元数据，验证并替代 OpenCV 的不可靠读数。

背景：OpenCV 的 CAP_PROP_FPS / CAP_PROP_FRAME_COUNT 在本项目素材上大面积失真
（多路报出 12000~71000 秒的荒谬时长，rnd_05 甚至报 fps=351.56）。
本脚本以 ffprobe 的容器/流元数据为准，输出可供方案设计引用的对照表。
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


def probe(path: Path) -> dict:
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,bit_rate",
        "-show_entries", "format=duration,bit_rate,size",
        "-of", "json", str(path),
    ]
    raw = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if raw.returncode != 0:
        return {"error": (raw.stderr or "").strip()[:80]}
    data = json.loads(raw.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    num, _, den = (stream.get("r_frame_rate") or "0/0").partition("/")
    try:
        fps = float(num) / float(den) if float(den or 0) else 0.0
    except ValueError:
        fps = 0.0
    try:
        duration = float(fmt.get("duration") or 0)
    except ValueError:
        duration = 0.0
    return {
        "codec": stream.get("codec_name", "?"),
        "width": stream.get("width", 0),
        "height": stream.get("height", 0),
        "fps": fps,
        "frames": stream.get("nb_frames") or "?",
        "duration": duration,
        "size_mb": round(int(fmt.get("size") or 0) / 1024 / 1024, 1),
        "bitrate_mbps": round(int(fmt.get("bit_rate") or 0) / 1e6, 2),
    }


def main() -> None:
    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    rows = []
    for cam_id, rel in sources.items():
        path = ROOT / rel
        if not path.exists():
            rows.append((cam_id, {"error": "MISSING"}))
            continue
        rows.append((cam_id, probe(path)))

    header = f"{'cam':8s} {'codec':6s} {'res':11s} {'fps':7s} {'frames':9s} {'dur':9s} {'size':9s} {'Mbps'}"
    print(header)
    print("-" * len(header))
    total_size = 0.0
    total_dur = 0.0
    for cam_id, info in rows:
        if "error" in info:
            print(f"{cam_id:8s} ERROR: {info['error']}")
            continue
        total_size += info["size_mb"]
        total_dur += info["duration"]
        print(
            f"{cam_id:8s} {info['codec']:6s} "
            f"{str(info['width']) + 'x' + str(info['height']):11s} "
            f"{info['fps']:<7.2f} {str(info['frames']):9s} "
            f"{info['duration']:<9.1f} {str(info['size_mb']) + 'MB':9s} "
            f"{info['bitrate_mbps']}"
        )
    print("-" * len(header))
    print(f"合计 {len(rows)} 路 / {total_size:.0f} MB / 真实总时长 {total_dur:.0f}s "
          f"({total_dur / 60:.1f} min)")

    resolutions = {}
    codecs = {}
    for _, info in rows:
        if "error" in info:
            continue
        key = f"{info['width']}x{info['height']}"
        resolutions[key] = resolutions.get(key, 0) + 1
        codecs[info["codec"]] = codecs.get(info["codec"], 0) + 1
    print(f"分辨率分布: {resolutions}")
    print(f"编码分布: {codecs}")


if __name__ == "__main__":
    main()
