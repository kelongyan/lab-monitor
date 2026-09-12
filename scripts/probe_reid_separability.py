"""实测当前 ReID 特征的判别力（决定「身份识别 / 视频检索」能否建立在现有特征上）。

方法：读 outputs/lab_monitor.db 里全部身份的 feature_blob / feature_bank_blob，
     - 组内相似度：同一身份 feature_bank 内部两两余弦相似度（应高）
     - 组间相似度：不同身份主特征两两余弦相似度（应低）
     - 阈值代价：统计有多少「异人」对的相似度会越过当前阈值 0.75
输出为一组可直接写进方案文档的量化结论。只读，不写任何文件。
"""

import io
import json
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 判据阈值直接读线上配置，不写死 —— 否则诊断结论会与线上行为脱节
from src.reid_config import REID_MATCH_THRESHOLD as THRESHOLD  # noqa: E402
from src.reid_config import REID_RATIO_TEST as RATIO  # noqa: E402


def load() -> list[dict]:
    conn = sqlite3.connect(str(ROOT / "outputs" / "lab_monitor.db"))
    rows = conn.execute(
        "SELECT global_id, feature_dim, feature_blob, feature_bank_count, feature_bank_blob "
        "FROM identities WHERE feature_dim > 0 AND feature_blob IS NOT NULL"
    ).fetchall()
    conn.close()
    out = []
    for gid, dim, blob, bank_n, bank_blob in rows:
        main = np.frombuffer(blob, dtype=np.float32).copy()
        n = int(main.size)
        if n != int(dim):
            continue
        bank = []
        if bank_n and bank_blob:
            bank = [r.copy() for r in np.frombuffer(bank_blob, dtype=np.float32).reshape(int(bank_n), n)]
        main = main / max(np.linalg.norm(main), 1e-8)
        out.append({"gid": gid, "main": main, "bank": [b / max(np.linalg.norm(b), 1e-8) for b in bank]})
    return out


def main() -> None:
    records = load()
    print(f"载入身份 {len(records)} 个，特征维度 {records[0]['main'].size if records else 0}")
    if len(records) < 2:
        print("样本不足，无法统计组间分布")
        return

    intra = []
    bank_sizes = []
    for rec in records:
        bank_sizes.append(len(rec["bank"]))
        for a, b in combinations(rec["bank"], 2):
            intra.append(float(a @ b))

    mains = np.stack([r["main"] for r in records])
    sim_all = mains @ mains.T
    inter = [float(sim_all[i, j]) for i in range(len(records)) for j in range(i + 1, len(records))]

    inter = np.asarray(inter)
    intra = np.asarray(intra) if intra else np.asarray([0.0])

    print(f"feature_bank 大小分布: min={min(bank_sizes)} max={max(bank_sizes)} "
          f"avg={np.mean(bank_sizes):.1f}（设计上限 5）")
    print(f"组内相似度：n={intra.size}  mean={intra.mean():.3f}  "
          f"p5={np.percentile(intra, 5):.3f}  min={intra.min():.3f}")
    print(f"组间相似度：n={inter.size}  mean={inter.mean():.3f}  "
          f"p50={np.percentile(inter, 50):.3f}  p95={np.percentile(inter, 95):.3f}  "
          f"max={inter.max():.3f}")

    over = int((inter >= THRESHOLD).sum())
    print(f"\n判据阈值 {THRESHOLD}：组间相似度 ≥ 阈值的「异人」对 = {over} / {inter.size} "
          f"({over / inter.size * 100:.2f}%)")

    warns = [(records[i]["gid"], records[j]["gid"], float(sim_all[i, j]))
             for i in range(len(records)) for j in range(i + 1, len(records))
             if sim_all[i, j] >= THRESHOLD]
    warns.sort(key=lambda x: -x[2])
    vs_ratio = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            row = np.delete(sim_all[i], [i, j])
            if row.size and sim_all[i, j] >= THRESHOLD:
                second = float(row.max())
                if sim_all[i, j] > 0 and second / sim_all[i, j] > RATIO:
                    vs_ratio += 1
    print(f"其中会被 Ratio Test({RATIO}) 拦下的歧义对 = {vs_ratio}")
    print("\n最相似的 10 对身份（异人对，相似度越高越容易误归并）：")
    for a, b, s in warns[:10]:
        print(f"  {a} <-> {b}  sim={s:.4f}")


if __name__ == "__main__":
    main()
