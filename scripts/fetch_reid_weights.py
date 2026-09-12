"""fetch_reid_weights.py — 下载 torchreid 官方 ReID 度量学习权重。

为什么需要它
------------
`src/reid.py` 原先用 `build_model(pretrained=True)`，而 torchreid 这个开关**只会
加载 ImageNet 分类权重**（`torchreid/reid/models/osnet.py:11-22` 的 `pretrained_urls`
全指向 imagenet）。分类嵌入空间不保证"同人近、异人远"，实测后果是 990 对异人
特征余弦中位 0.984、99.8% 越过匹配阈值 —— 身份识别实际不工作。

修复办法是显式加载 ReID 数据集权重，而这类权重不在 torchreid 的自动下载路径上，
需要单独获取。本脚本负责这一步。

清单与校验逻辑都来自 `src/reid_config.py`（单一来源），本脚本只做下载，
**不再维护第二份权重清单** —— 两份清单迟早会不一致。

用法（在项目根目录）：
    ./.venv/Scripts/python.exe scripts/fetch_reid_weights.py --list
    ./.venv/Scripts/python.exe scripts/fetch_reid_weights.py                  # 默认两份
    ./.venv/Scripts/python.exe scripts/fetch_reid_weights.py --only market1501
    ./.venv/Scripts/python.exe scripts/fetch_reid_weights.py --proxy ""       # 直连

权重落在 ~/.cache/torch/checkpoints/，被 .gitignore 的 *.pth 覆盖，不进仓库。
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reid_config import (  # noqa: E402
    CHECKPOINT_DIR,
    REID_WEIGHTS,
    ReIDWeightUnavailable,
    load_reid_state_dict,
)

#: 默认下载哪几份。选这两份是因为"哪个更适合本项目"必须实测：
#: msmt17 域内指标低（61.4）是因为数据集本身更难（15 台相机、跨 4 天、室内外混合），
#: 而 market1501（校园街道式取景）成像风格更接近本项目的固定室内走廊。
#: 结论由 scripts/tune_reid_threshold.py 在同一标注集上对比后定稿。
DEFAULT_TARGETS = ("market1501", "msmt17")
DEFAULT_PROXY = "http://127.0.0.1:7897"


def _apply_proxy(proxy: str | None) -> None:
    """gdown 走 requests，认 HTTP(S)_PROXY 环境变量。直连不通的机器必须设。"""
    if not proxy:
        print("不使用代理（直连）")
        return
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    print(f"使用代理: {proxy}")


def _verify(path: Path, weight) -> bool:
    """复用 reid_config 的校验：分类头维度是判定训练数据集的唯一可靠证据。"""
    try:
        state = load_reid_state_dict(path, weight)
    except ReIDWeightUnavailable as error:
        print(f"  ✗ 校验失败: {error}")
        return False
    classifier = next(k for k in state if k.endswith("classifier.weight"))
    params = sum(int(v.numel()) for v in state.values() if hasattr(v, "numel"))
    print(
        f"  ✓ 分类头 {tuple(state[classifier].shape)} → 数据集 {weight.dataset}"
        f" | {len(state)} 张量 / {params / 1e6:.2f}M 参数"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 torchreid ReID 度量学习权重")
    parser.add_argument("--list", action="store_true", help="只列出候选，不下载")
    parser.add_argument("--only", choices=sorted(REID_WEIGHTS), help="只处理指定权重")
    parser.add_argument("--proxy", default=DEFAULT_PROXY,
                        help=f"HTTP 代理（默认 {DEFAULT_PROXY}，传空字符串则直连）")
    parser.add_argument("--dir", default=str(CHECKPOINT_DIR), help="下载目录")
    args = parser.parse_args()

    if args.list:
        print(f"{'key':12s} {'文件名':36s} {'drive_id':36s} 指标")
        print("-" * 118)
        for key, weight in REID_WEIGHTS.items():
            print(f"{key:12s} {weight.file_name:36s} {weight.drive_id:36s} {weight.benchmark}")
        print(f"\n默认下载: {', '.join(DEFAULT_TARGETS)}")
        print(f"目标目录: {CHECKPOINT_DIR}")
        return 0

    targets = (args.only,) if args.only else DEFAULT_TARGETS
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("缺少 gdown，请先安装：pip install gdown")
        return 2

    _apply_proxy(args.proxy or None)
    failures = []
    for key in targets:
        weight = REID_WEIGHTS[key]
        output = output_dir / weight.file_name
        print(f"\n[{key}] {weight.benchmark}")

        if output.exists() and output.stat().st_size > 1024 * 1024:
            print(f"  已存在，校验后决定是否跳过：{output}")
            if _verify(output, weight):
                continue
            print("  校验未通过，重新下载")

        if not weight.drive_id:
            print("  ✗ 该权重没有登记 drive_id，无法自动下载")
            failures.append(key)
            continue

        print(f"  下载 → {output}")
        try:
            result = gdown.download(id=weight.drive_id, output=str(output), quiet=False)
        except Exception as error:
            print(f"  ✗ 下载异常: {type(error).__name__}: {error}")
            failures.append(key)
            continue
        if not result or not output.exists():
            print("  ✗ 下载失败（Google Drive 可能返回了配额提示页）")
            failures.append(key)
            continue
        if not _verify(output, weight):
            failures.append(key)

    print("\n" + "=" * 60)
    if failures:
        print(f"失败: {', '.join(failures)}")
        print(f"排查：1) 确认代理可用  2) 手动下载后放入 {output_dir}")
        return 1
    print(f"全部就绪，权重目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
