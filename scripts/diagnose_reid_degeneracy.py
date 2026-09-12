"""诊断：异人相似度接近 1.0 是「特征退化」还是「持久化重复」？

上一轮 probe_reid_separability.py 报出组间相似度 p50=0.984、多对精确 1.0000。
本脚本做三件事区分病因：
  1. feature_blob 的 md5 去重计数（若是 1 份被复制，则属持久化/注册缺陷）
  2. 特征向量的均值范数与「去均值后」的诊断（若所有向量共享一个大常量分量，
     余弦会被该分量支配 → 属特征退化）
  3. 零向量 / 单位化失效检查
只读，不写任何文件。
"""

import hashlib
import io
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    conn = sqlite3.connect(str(ROOT / "outputs" / "lab_monitor.db"))
    rows = conn.execute(
        "SELECT global_id, feature_dim, feature_blob, feature_bank_count, feature_bank_blob, "
        "total_appearances, last_camera FROM identities "
        "WHERE feature_dim > 0 AND feature_blob IS NOT NULL"
    ).fetchall()
    conn.close()

    digests = [hashlib.md5(r[2]).hexdigest() for r in rows]
    dup = {d: c for d, c in Counter(digests).items() if c > 1}
    print(f"身份数 {len(rows)}，feature_blob 唯一 md5 数 {len(set(digests))}")
    print(f"存在重复 copy 的组数 {len(dup)}，涉及身份 {sum(dup.values())} 个")
    if dup:
        gid_of = {}
        for r, d in zip(rows, digests):
            gid_of.setdefault(d, []).append(r[0])
        for d, c in list(dup.items())[:5]:
            print(f"  {c} 个身份共享同一 feature_blob: {gid_of[d][:6]}")

    mains = []
    for gid, dim, blob, bank_n, bank_blob, total, cam in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size != int(dim):
            continue
        mains.append((gid, v.copy(), v.size))
    print(f"\n可解析特征 {len(mains)} 个，维度 {mains[0][2] if mains else 0}")

    mat = np.stack([m for _, m, _ in mains])
    norms = np.linalg.norm(mat, axis=1)
    print(f"原始特征范数: min={norms.min():.4f} mean={norms.mean():.4f} max={norms.max():.4f}")

    # 是否已 L2 归一化
    unit = np.allclose(norms, 1.0, atol=1e-3)
    print(f"是否已 L2 归一化: {unit}")

    mean_vec = mat.mean(axis=0)
    print(f"\n全体均值向量范数 = {np.linalg.norm(mean_vec):.4f}（单位向量下该值越接近 1，"
          f"说明所有人共享同一个主导分量）")

    centered = mat - mean_vec
    cn = np.linalg.norm(centered, axis=1)
    print(f"去均值后每个身份的残差范数: mean={cn.mean():.4f} min={cn.min():.4f} "
          f"max={cn.max():.4f}（越小说明身份间差异越小）")

    if np.all(cn > 1e-6):
        cu = centered / cn[:, None]
        cs = cu @ cu.T
        off = [float(cs[i, j]) for i in range(len(cu)) for j in range(i + 1, len(cu))]
        print(f"去均值后组间余弦: mean={np.mean(off):.3f} p50={np.percentile(off, 50):.3f} "
              f"p95={np.percentile(off, 95):.3f} max={np.max(off):.3f}")

    sims = mat @ mat.T
    offs = [float(sims[i, j]) for i in range(len(mat)) for j in range(i + 1, len(mat))]
    print(f"\n原始组间余弦: mean={np.mean(offs):.3f} p50={np.percentile(offs, 50):.3f} "
          f"max={np.max(offs):.3f}")
    print(f"组间余弦 ≥ 0.999 的对数 = {sum(1 for s in offs if s >= 0.999)} / {len(offs)}")

    print("\n每身份出现数（total_appearances）分布：")
    totals = sorted((r[5] for r in rows), reverse=True)
    print(f"  {totals[:12]} ... 最小 {totals[-1] if totals else 0}")


if __name__ == "__main__":
    main()
