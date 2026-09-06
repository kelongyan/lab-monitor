"""
scripts/render_floorplan.py — 从客户平面图 PDF 渲染轨迹回放底图

用法（在项目根目录）：
    ./.venv/Scripts/python.exe scripts/render_floorplan.py "C:/path/to/视频监控系统平面图-屋面（IP分配）.pdf"

产出：
    assets/floorplan/floorplan.jpg   (2420px 宽 JPEG，约 0.6MB)

为什么有这脚本：
    底图是客户设施图纸渲染产物，与 docs/*.xlsx 同级敏感（含机位/IP 分配信息），
    已 gitignore 不入公开仓库。新环境 clone 后跑一次本脚本即可恢复底图，
    跑不了就只显示点位与连线（接口已做无底图降级）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"PDF 不存在: {pdf_path}", file=sys.stderr)
        return 1

    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    if doc.page_count < 1:
        print("PDF 没有页面", file=sys.stderr)
        return 1
    page = doc[0]

    # 渲染 300dpi 后缩到 2420px 宽（保留标注可读性、控制体积 ~0.6MB）
    pix = page.get_pixmap(dpi=300)
    tmp = ROOT / "assets" / "floorplan" / "_render_tmp.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(tmp))

    from PIL import Image

    im = Image.open(tmp).convert("RGB")
    w, h = im.size
    nw = 2420
    im2 = im.resize((nw, int(h * nw / w)), Image.LANCZOS)

    out = tmp.parent / "floorplan.jpg"
    im2.save(str(out), quality=88, optimize=True)
    tmp.unlink(missing_ok=True)
    print(f"底图已生成: {out}  ({im2.size[0]}x{im2.size[1]}, {out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
