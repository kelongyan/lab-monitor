"""
test_personnel_gallery_space.py — 底库 1:N 检索必须与实时匹配在同一坐标系（2026-09-13 新增）

缺陷（实测，本机 29 个真实身份）：
`PersonnelGallery.match()` 原先把**原始**特征（注册照特征未减公共分量）拿去比对一个在
**中心化空间**标定出来的阈值 `REID_MATCH_THRESHOLD=0.68`。两个空间的异人分布差距：

    原始空间   异人对余弦 p50 0.6893  p95 0.9038  越过 0.68 的比例 **54.9%**（223/406）
    中心化空间 异人对余弦 p50 -0.0642 p95 0.5202  越过 0.68 的比例 **1.7%**（7/406）

差 30 倍。更糟的是底库只有 1 个人时 `match_feature_detailed` 会跳过 Ratio Test，
于是**陌生人几乎必然被判成该人**，pipeline 的路径 B 闭环
（`personnel.match()` 命中 → `bind_person()` 自动命名）会把陌生人的身份挂到该人名下。

修复：`match(feature, context=...)` 要求传入 IdentityStore 的 `MatchContext`，
query 与 gallery 用**同一个中心**一起变换 —— 这正是 MatchContext 存在的理由
（见 `src/identity_store.py:MatchContext`）。本文件锁定该契约。

所有用例都在 tempfile 临时库上跑，不碰生产库。
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.db import Database
from src.identity_store import IdentityStore
from src.personnel import PersonnelGallery
from src.reid_config import REID_MATCH_THRESHOLD, REID_PERSONNEL_THRESHOLD

#: 与真实链路同为 512 维没有额外价值，这里用小维度让"哪个分量在做判别"一目了然
DIM = 16
#: 公共分量在特征里的权重（真实 OSNet 实测全体均值范数 0.80~0.83）
COMMON_WEIGHT = 0.84
#: 身份特异残差的权重
RESIDUAL_WEIGHT = 0.10


def _axis(index: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _feature(residual_axis: int) -> np.ndarray:
    """公共分量 + 某一条轴上的身份残差，再归一化（模拟 OSNet 输出的共线现象）。"""
    vector = COMMON_WEIGHT * _axis(0) + RESIDUAL_WEIGHT * _axis(residual_axis)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


class PersonnelGalleryCoordinateTests(unittest.TestCase):
    """
    用真实 IdentityStore 产生 MatchContext（而不是桩件），这样"中心从哪来、
    怎么变换"两件事都被覆盖到。
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.database = Database(Path(self._temp.name) / "gallery.db")
        self.store = IdentityStore(database=self.database, feature_space=f"test:{DIM}")
        # 8 个身份（= _CENTER_MIN_FEATURES）都带同一条公共分量，
        # 用 register() 绕过匹配强行塞入 —— 本用例只关心坐标系，不关心注册策略。
        for axis in range(1, 9):
            self.store.register(_feature(axis))

        self.gallery = PersonnelGallery(database=self.database)
        self.person_id = self.gallery.create("张三")
        # 注册照：残差落在第 12 条轴上（这 8 个身份都没用过，中心里不含该方向）
        self.gallery.add_photo(self.person_id, _feature(12), quality=1.0)
        # 查询：另一个人的残差落在第 15 条轴上 → 中心化后两者近乎正交
        self.stranger = _feature(15)

    def tearDown(self):
        self.database.close()
        self._temp.cleanup()

    def test_center_is_active_so_the_scenario_is_meaningful(self):
        """前置断言：中心化确实生效（否则下面两条断言退化为同一个空间，测不出东西）。"""
        context = self.store.build_match_context()
        self.assertTrue(context.centering_enabled, "8 个特征 / 中心范数应已越过冷启动护栏")
        self.assertGreater(float(np.linalg.norm(context.center)), 0.35)

    def test_centered_space_rejects_stranger_that_raw_space_would_accept(self):
        """
        核心用例：同一个人（原始空间）在中心化空间里必须是"不命中"。

        原始空间两者余弦 ≈ 0.986（都压倒性地指向公共分量）→ 远超 0.68；
        减掉公共分量后残差正交 → 余弦 ≈ 0。所以：
          - 传 context：不命中（修复后的行为）
          - 不传 context：命中（改造前行为，即那个把陌生人自动命名的缺陷）
        """
        context = self.store.build_match_context()

        raw_similarity = cosine(self.stranger, _feature(12))
        self.assertGreater(raw_similarity, REID_MATCH_THRESHOLD,
                           "原始空间应当虚高到越阈 —— 这是缺陷成立的前提")

        prepared_query = context.prepare(self.stranger)
        prepared_photo = context.prepare(_feature(12))
        self.assertLess(cosine(prepared_query, prepared_photo), REID_MATCH_THRESHOLD,
                        "中心化后应当判为不命中")

        self.assertIsNone(
            self.gallery.match(self.stranger, context=context),
            "传了 context 就不该把陌生人认成张三")

    def test_without_context_it_keeps_the_pre_fix_behaviour(self):
        """
        不传 context 时退化为改造前行为 —— 本条**锁住这个退化**，
        避免以后有人误以为 context 是可选参数：它一旦被省掉，缺陷立即复现。
        """
        hit = self.gallery.match(self.stranger)
        self.assertIsNotNone(hit, "不传 context 时原始空间的虚高相似度会导致命中")
        self.assertEqual(hit["person_id"], self.person_id)

    def test_genuine_owner_still_matches_with_context(self):
        """反向护栏：修复不能把真主人也一起挡掉（用同一个残差方向查询）。"""
        context = self.store.build_match_context()
        hit = self.gallery.match(_feature(12), context=context)
        self.assertIsNotNone(hit, "同一个人（同残差方向）必须仍然命中")
        self.assertEqual(hit["person_id"], self.person_id)
        self.assertEqual(hit["name"], "张三")

    def test_multiple_photos_of_one_person_are_deduped_by_person(self):
        """
        同一个人多张注册照只占一个候选位次：否则 Ratio Test（second/best）
        会把"自己和自己的第二张照片"判成歧义，真实命中被拒。
        """
        self.gallery.add_photo(self.person_id, _feature(13), quality=1.0)
        context = self.store.build_match_context()
        hit = self.gallery.match(_feature(12), context=context)
        self.assertIsNotNone(hit, "同人多照不应触发 Ratio 判歧义")
        self.assertEqual(hit["person_id"], self.person_id)


class PersonnelThresholdTests(unittest.TestCase):
    """底库阈值必须来自 reid_config 的单一来源，且默认与实时阈值一致。"""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.database = Database(Path(self._temp.name) / "threshold.db")

    def tearDown(self):
        self.database.close()
        self._temp.cleanup()

    def test_default_threshold_is_reid_config_value(self):
        gallery = PersonnelGallery(database=self.database)
        self.assertEqual(REID_PERSONNEL_THRESHOLD, gallery.threshold())
        self.assertEqual(REID_PERSONNEL_THRESHOLD, REID_MATCH_THRESHOLD,
                         "尚未单独标定前，底库阈值应与实时阈值同值")

    def test_explicit_threshold_overrides_default(self):
        gallery = PersonnelGallery(database=self.database, threshold=0.9)
        self.assertEqual(0.9, gallery.threshold())


if __name__ == "__main__":
    unittest.main()
