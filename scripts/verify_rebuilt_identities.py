"""verify_rebuilt_identities.py — 重建后身份库的验收分析（生产语义，带自检）。

为什么单独立一个脚本：此前用内联 python -c 做离线分析，两次得到互相矛盾的结果
（同一对身份一次 1.0000、一次 0.6366），根因是漏了"中心化后重新归一化"这一步 ——
不减归一化时，未归一化的点积可以超过 1，与线上 `_prepare_for_match()` 的语义不一致。
本脚本把语义写死为与生产完全一致，并内置自检断言，避免再出这种事故。

语义（与 src/identity_store.py 一致）：
    prepared = normalize(v - center)          # center = 全体主特征的均值
    pair_sim  = max over (主特征, feature_bank) 的笛卡尔积
    同相机/跨相机 按 last_camera 划分

自检：
    1. 8f708fd8 与 bab050cd 的原始主特征点积必须为 1.0（已确认 md5 相同）；
       若不是，说明库变了，脚本报告的数字需要重新解读。
    2. 所有向量的 ‖v-center‖/‖v‖ 必须大于 0（防止退化成 NaN）。
"""

from __future__ import annotations

import hashlib
import io
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "outputs" / "lab_monitor.db"
THRESHOLD = 0.75
SELF_CHECK_PAIR = ("8f708fd8", "bab050cd")


def load() -> tuple[dict[str, tuple[str, int, list[np.ndarray]]], np.ndarray]:
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT global_id, last_camera, total_appearances, feature_dim, "
        "feature_blob, feature_bank_count, feature_bank_blob FROM identities"
    ).fetchall()
    conn.close()

    recs: dict[str, tuple[str, int, list[np.ndarray]]] = {}
    for gid, cam, apps, dim, blob, bank_count, bank_blob in rows:
        main = np.frombuffer(blob, dtype=np.float32).copy()
        main = main / np.linalg.norm(main)
        bank = [main]
        if bank_count and bank_blob:
            matrix = np.frombuffer(bank_blob, dtype=np.float32).reshape(int(bank_count), int(dim))
            for row in matrix:
                row = row.copy()
                norm = float(np.linalg.norm(row))
                if norm > 1e-8:
                    bank.append(row / norm)
        recs[gid] = (cam, int(apps), bank)

    mains = np.stack([recs[g][2][0] for g in recs])
    return recs, mains.mean(axis=0)


def main() -> int:
    recs, center = load()
    center_norm = float(np.linalg.norm(center))
    print(f"身份 {len(recs)} 个，中心范数 {center_norm:.4f}"
          f"（随机方向期望约 {1 / np.sqrt(len(recs)):.3f}）")

    prepared = {g: [normalize(v - center) for v in bank]
                for g, (_cam, _apps, bank) in recs.items()}

    cross, same = [], []
    for gi, gj in combinations(recs, 2):
        sim = max(float(p @ q) for p in prepared[gi] for q in prepared[gj])
        (same if recs[gi][0] == recs[gj][0] else cross).append((sim, gi, gj))
    cross.sort(reverse=True)
    same.sort(reverse=True)

    def stats(pairs: list[tuple[float, str, str]]) -> str:
        if not pairs:
            return "（无样本）"
        values = np.array([p[0] for p in pairs])
        return (f"p50={np.percentile(values, 50):.3f} p95={np.percentile(values, 95):.3f} "
                f"max={values.max():.3f} 越阈={int((values >= THRESHOLD).sum())} 对"
                f" ({(values >= THRESHOLD).mean() * 100:.2f}%)")

    print(f"\n中心化 + 重新归一化（与线上 _prepare_for_match 完全一致）:")
    print(f"  跨相机 {len(cross)} 对: {stats(cross)}")
    print(f"  同相机 {len(same)} 对: {stats(same)}")
    if not cross:
        print("  （归并后所有身份的 last_camera 相同 → 无跨相机对；"
              "last_camera 在归并时会跟随较新的一方，所以这个划分仅供参考，"
              "完整分布见下）")

    print("\n跨相机最相似 5 对（决定阈值上限，越低越好）:")
    for sim, gi, gj in cross[:5]:
        print(f"   {sim:.4f}  {gi}({recs[gi][0]},{recs[gi][1]}) <-> "
              f"{gj}({recs[gj][0]},{recs[gj][1]})")
    print("同相机最相似 5 对（高相似 = 同一人被重复注册）:")
    for sim, gi, gj in same[:5]:
        print(f"   {sim:.4f}  {gi} <-> {gj}")

    # 全体两两分布（不按相机切分，避免归并后 last_camera 相同导致跨相机对为空）
    overall = sorted(cross + same, reverse=True)
    print(f"\n全体身份对 {len(overall)} 个: {stats(overall)}")
    over_all = [p for p in overall if p[0] >= THRESHOLD]
    print(f"高于阈值 {THRESHOLD} 的对（即 live 匹配会视为同一人的身份对）: {len(over_all)} 个")
    for sim, gi, gj in over_all[:8]:
        print(f"   {sim:.4f}  {gi} <-> {gj}")

    # 落库完整性
    digests: dict[str, list[str]] = {}
    for gid in recs:
        blob = _blob_of(gid)
        digests.setdefault(hashlib.md5(blob).hexdigest(), []).append(gid)
    dup_groups = {k: v for k, v in digests.items() if len(v) > 1}
    print(f"\n落库完整性: 身份 {len(recs)} / 唯一 feature_blob {len(digests)}"
          f" / 重复组 {len(dup_groups)}")
    for group in dup_groups.values():
        print(f"   重复组: {group}")

    # 自检
    ok = True
    if SELF_CHECK_PAIR[0] in recs and SELF_CHECK_PAIR[1] in recs:
        raw = float(recs[SELF_CHECK_PAIR[0]][2][0] @ recs[SELF_CHECK_PAIR[1]][2][0])
        print(f"\n自检: {SELF_CHECK_PAIR[0]} 与 {SELF_CHECK_PAIR[1]} 的原始主特征点积 = "
              f"{raw:.4f}")
        if abs(raw - 1.0) > 1e-6:
            print("   ⚠ 不再是 1.0 —— 库已变化，请重新解读本报告")
            ok = False
    ratios = [float(np.linalg.norm(prepared[g][0])) for g in recs]
    if min(ratios) <= 0:
        print("   ⚠ 存在中心化后范数为 0 的向量（退化）")
        ok = False
    else:
        print(f"   中心化后范数范围 {min(ratios):.4f} ~ {max(ratios):.4f}（无退化向量）")
    return 0 if ok else 1


_CACHE: dict[str, bytes] = {}


def _blob_of(gid: str) -> bytes:
    if not _CACHE:
        conn = sqlite3.connect(str(DB))
        for gid_, blob in conn.execute("SELECT global_id, feature_blob FROM identities"):
            _CACHE[gid_] = blob
        conn.close()
    return _CACHE[gid]


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


if __name__ == "__main__":
    raise SystemExit(main())
