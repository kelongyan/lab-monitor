"""
test_reid_identity_integrity.py — ReID 身份完整性与匹配语义回归测试（2026-09-12 新增）

背景：库内 45 个身份实测只有 25 个唯一 feature_blob，其中 27 个分属 7 组、
每组特征**字节完全相同**，且每组的 total_appearances 都是 230、last_camera 都是 rnd_19。
逐层追下去确认了塌缩机制（`scripts/diagnose_ema_collapse.py` 实测：同一人反复观测
20 次后，跨身份余弦 p50 从 0.50 升到 0.73；混入不同人后升到 0.96）：

    主特征 EMA 滑动平均 → 抹平身份特异残差 → 所有身份互相 ~0.98 相似
      → Ratio Test（second/best > 0.85）把每一次真实命中都判成"歧义"
      → register_if_new 返回 ambiguous、不归并

注意"不归并"不等于"新建身份"：生产路径里 `register()` 从不被调用
（只有 demo.py 与测试用），所以歧义的实际后果是**永远认不出**——
track 拿不到 global_id，攒满的 8 帧缓冲被丢弃。库里那 27 个重复身份是在
查询与已有身份相似度低于阈值时各自注册的，之后各自的 EMA 又收敛到同一不动点。
精确的创建时点需要当次运行日志才能定论，本文件只锁定可复现的部分。

因此本文件覆盖四件事：
  1. 并发注册不得产生重复 feature_blob（写路径完整性）
  2. match_feature_detailed 必须**按身份去重**后再做阈值/Ratio 判定，
     否则同一身份会同时占据 best 与 second，Ratio Test 必然误判
  3. feature_bank（存原始特征）必须能救回主特征（EMA 后）匹配不到的查询
  4. 特征塌缩护栏必须触发，并可在 /api/metrics/reid 观测到

所有用例都在 tempfile 临时库上跑，不碰生产库。
"""

import hashlib
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.db import Database
from src.identity_store import IdentityStore
from src.reid import match_feature_detailed
from src.reid_config import REID_MATCH_THRESHOLD


@contextmanager
def temporary_store(feature_space: str = "test-model:16"):
    with tempfile.TemporaryDirectory() as temp_dir:
        database = Database(Path(temp_dir) / "identity_integrity.db")
        try:
            yield IdentityStore(database=database, feature_space=feature_space)
        finally:
            database.close()


def unit(*values) -> np.ndarray:
    feature = np.array(values, dtype=np.float32)
    return feature / np.linalg.norm(feature)


def orthogonal_basis(count: int, dim: int) -> list[np.ndarray]:
    """构造 count 个两两正交的单位向量（cosine 恰为 0，不会触发任何阈值）。"""
    rng = np.random.default_rng(1234)
    matrix = rng.normal(size=(dim, count))
    q, _ = np.linalg.qr(matrix)
    return [q[:, i].astype(np.float32) for i in range(count)]


class FeatureBlobUniquenessTests(unittest.TestCase):
    """1. 写路径完整性：并发注册互异特征，不得出现重复 feature_blob。"""

    def test_concurrent_registration_keeps_feature_blobs_unique(self):
        with temporary_store() as store:
            features = orthogonal_basis(8, 16)
            errors: list[BaseException] = []

            def worker(feature: np.ndarray) -> None:
                try:
                    store.register_if_new(feature)
                except BaseException as exc:      # noqa: BLE001 - 线程内异常需带回主线程
                    errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=(feature,))
                for feature in features
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [], f"并发注册抛出异常: {errors}")
            ids = store.all_ids()
            self.assertEqual(len(ids), len(features), "互异特征应各自成为独立身份")
            digests = {
                hashlib.md5(store.get(gid).feature.tobytes()).hexdigest()
                for gid in ids
            }
            self.assertEqual(
                len(digests), len(ids),
                "存在特征字节完全相同的身份 —— 写路径出现重复 blob",
            )

    def test_registration_persists_one_unique_blob_per_identity(self):
        """落盘后重新从库里恢复，blob 仍须互不相同（防止只在内存里唯一）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "persist.db")
            try:
                store = IdentityStore(database=database, feature_space="test-model:16")
                for feature in orthogonal_basis(5, 16):
                    store.register_if_new(feature)
                store.flush()

                restored = IdentityStore(database=database, feature_space="test-model:16")
                ids = restored.all_ids()
                self.assertEqual(len(ids), 5, "重启后应恢复 5 个身份")
                digests = {
                    hashlib.md5(restored.get(gid).feature.tobytes()).hexdigest()
                    for gid in ids
                }
                self.assertEqual(
                    len(digests), len(ids),
                    "落盘的特征 blob 存在重复 —— 写路径缺陷",
                )
            finally:
                database.close()


class GalleryDedupTests(unittest.TestCase):
    """2. 按身份去重：同一身份多行时不得被 Ratio Test 误判为歧义。"""

    def test_single_identity_with_multiple_bank_rows_matches(self):
        query = unit(1, 0, 0, 0)
        # 一个身份、5 行（主特征 + 4 个姿态），其中一行与 query 完全一致
        gallery = [
            ("gid-a", unit(0.9, 0.1, 0, 0)),
            ("gid-a", unit(0.8, 0.2, 0, 0)),
            ("gid-a", query),
            ("gid-a", unit(0.7, 0.3, 0, 0)),
            ("gid-a", unit(0.95, 0.05, 0, 0)),
        ]
        detail = match_feature_detailed(query, gallery)
        self.assertEqual(
            detail.matched_id, "gid-a",
            "同一身份的多行被当成两个候选 → Ratio Test 误判歧义",
        )
        self.assertFalse(detail.is_ratio_blocked)
        self.assertAlmostEqual(detail.best_sim, 1.0, places=5)

    def test_two_identities_with_similar_features_still_blocked(self):
        """去重不能削弱既有语义：真的是两个身份且过于相似时仍须拒绝归并。"""
        query = unit(1, 0, 0, 0)
        gallery = [
            ("gid-a", unit(1, 0.02, 0, 0)),
            ("gid-b", unit(1, 0.03, 0, 0)),
        ]
        detail = match_feature_detailed(query, gallery)
        self.assertIsNone(detail.matched_id)
        self.assertTrue(detail.is_ratio_blocked)

    def test_below_threshold_identity_is_not_matched(self):
        query = unit(1, 0, 0, 0)
        gallery = [("gid-a", unit(0, 1, 0, 0)), ("gid-b", unit(0, 0, 1, 0))]
        detail = match_feature_detailed(query, gallery)
        self.assertIsNone(detail.matched_id)
        self.assertFalse(detail.is_ratio_blocked)
        self.assertLess(detail.best_sim, REID_MATCH_THRESHOLD)


class FeatureBankRescueTests(unittest.TestCase):
    """3. feature_bank（原始特征）必须能救回主特征（EMA 后）匹配不到的查询。"""

    def test_bank_row_matches_when_main_feature_has_drifted(self):
        main = unit(1, 0, 0, 0)
        raw = unit(0, 1, 0, 0)
        with temporary_store() as store:
            gid = store.register(main)
            # 一次高质量观测：raw 与主特征相似度 0 → 满足入池条件（<0.92）
            store.update_appearance(gid, "cam_a", raw, [0, 0, 10, 20], quality_score=1.0)

            record = store.get(gid)
            self.assertGreaterEqual(
                len(record.feature_bank), 2,
                "差异明显的原始特征应进入 feature_bank",
            )
            main_sim = float(record.feature @ raw)
            self.assertLess(
                main_sim, REID_MATCH_THRESHOLD,
                "本用例前提：主特征（EMA 后）已漂移，与查询不相似",
            )

            plain = match_feature_detailed(raw, store.get_gallery())
            self.assertIsNone(
                plain.matched_id,
                "只用主特征时应当匹配失败 —— 这正是塌缩后的典型症状",
            )

            rescued = match_feature_detailed(raw, store.get_match_gallery())
            self.assertEqual(
                rescued.matched_id, gid,
                "展开 feature_bank 后应当命中 —— 主匹配路径必须用它",
            )

    def test_match_gallery_expands_bank_rows(self):
        with temporary_store() as store:
            gid = store.register(unit(1, 0, 0, 0))
            store.update_appearance(gid, "cam_a", unit(0, 1, 0, 0), [0, 0, 8, 16],
                                    quality_score=1.0)
            bank_size = len(store.get(gid).feature_bank)

            self.assertEqual(len(store.get_gallery()), 1, "get_gallery 保持一身份一行")
            self.assertEqual(
                len(store.get_match_gallery()), 1 + bank_size,
                "get_match_gallery 应展开主特征 + 全部 bank 行",
            )


class CollapseGuardrailTests(unittest.TestCase):
    """4. 特征塌缩护栏：必须告警并可观测，而不是静默新建身份。"""

    def _force_collapsed_identities(self, store: IdentityStore, count: int = 3) -> None:
        """用 register() 绕过匹配强行塞入多个近乎相同的身份，复现塌缩态。"""
        base = unit(1, 0, 0, 0)
        for index in range(count):
            near_identical = base + np.float32(index) * np.float32(1e-4) * unit(0, 1, 0, 0)
            store.register(near_identical / np.linalg.norm(near_identical))

    def test_guardrail_counts_and_exposes_collapse_warning(self):
        with temporary_store() as store:
            self._force_collapsed_identities(store, count=3)
            before = len(store.all_ids())

            # 与所有已存在身份几乎完全一致 → Ratio Test 会判歧义
            resolution = store.register_if_new(unit(1, 0, 0, 0))

            self.assertEqual(resolution.status, "ambiguous")
            self.assertGreaterEqual(
                store.get_metrics()["collapse_warnings"], 1,
                "塌缩护栏必须计数并出现在 get_metrics() 中",
            )
            # 歧义分支直接返回、不注册。所以塌缩的实际后果不是"错认"也不是
            # "身份膨胀"，而是**永远认不出**：本 track 拿不到 global_id，
            # 调用方只打一行 debug，攒满的 8 帧 ReID 缓冲被丢弃。
            self.assertEqual(
                len(store.all_ids()), before,
                "歧义分支不应创建身份（现状如此；改判策略需先把特征修好）",
            )

    def test_health_metrics_shape_matches_empty_payload(self):
        """护栏字段必须同时存在于 store 的返回与 server 的空指标里，避免前端缺字段。"""
        import server

        with temporary_store() as store:
            store_keys = set(store.get_metrics())
        self.assertEqual(
            store_keys, set(server._EMPTY_REID_METRICS),
            "IdentityStore.get_metrics() 与 server._EMPTY_REID_METRICS 字段必须同构",
        )


class CommonComponentCenteringTests(unittest.TestCase):
    """
    5. 公共分量中心化 —— ReID 识别失效的真正修复。

    实测（scripts/diagnose_ema_collapse.py --center）：OSNet 输出的单位特征都和一个
    全局方向高度共线（全体特征均值范数 0.80~0.83，随机方向的期望只有 ~0.1）。
    跨身份余弦因此虚高：imagenet 原始 p50 0.4961、越阈 21.4%；
    减去公共分量后 p50 -0.1021、越阈 0.0%。

    本组用例构造"共享大公共分量 + 小身份余量"的特征，验证：
      · 身份太少时**不启用**中心化（冷启动安全，行为与改造前一致）
      · 身份够多时启用，且原先必然判歧义的查询能被正确归并
      · 新增身份会让中心缓存失效（否则 query 与 gallery 不在同一坐标系）
    """

    DIM = 16

    def _shared_component_features(self, count: int,
                                   common: float = 0.9) -> list[np.ndarray]:
        """第 i 个身份 = normalize(common * e0 + (1-common) * e_{i+1})。

        所有身份共享 e0 这个主导方向 → 彼此余弦约 0.81，远超阈值 0.75；
        身份特异的余量藏在 e_{i+1} 上。减掉公共分量后它们才互相正交。
        """
        basis = orthogonal_basis(count + 1, self.DIM)
        common_dir = basis[0]
        features = []
        for index in range(count):
            vec = common * common_dir + (1.0 - common) * basis[index + 1]
            features.append((vec / np.linalg.norm(vec)).astype(np.float32))
        return features

    def test_centering_disabled_on_cold_start(self):
        with temporary_store() as store:
            for feature in self._shared_component_features(3):
                store.register(feature)
            metrics = store.get_metrics()
            self.assertFalse(
                metrics["center_enabled"],
                "身份不足下限时应关闭中心化 —— 冷启动必须与改造前行为一致",
            )
            self.assertEqual(metrics["center_norm"], 0.0)

    def test_centering_separates_identities_sharing_a_common_component(self):
        with temporary_store() as store:
            features = self._shared_component_features(8)
            ids = [store.register(feature) for feature in features]
            target_index = 3
            query = features[target_index]

            metrics = store.get_metrics()
            self.assertTrue(metrics["center_enabled"], "身份足够时应启用中心化")
            self.assertGreater(metrics["center_norm"], 0.35)

            # 对照：不中心化时所有身份余弦都 ~0.99，Ratio Test 必然判歧义
            raw_detail = match_feature_detailed(query, store.get_gallery())
            self.assertIsNone(raw_detail.matched_id)
            self.assertTrue(
                raw_detail.is_ratio_blocked,
                "未中心化时应当因歧义而被拒绝 —— 这正是线上'永远认不出'的成因",
            )

            # 中心化后应当命中正确身份。注意 query 必须用 context.prepare 变换，
            # 与 gallery 用同一个中心 —— 这是 MatchContext 存在的意义。
            context = store.build_match_context()
            self.assertTrue(context.centering_enabled)
            centered = match_feature_detailed(
                context.prepare(query), context.gallery
            )
            self.assertEqual(
                centered.matched_id, ids[target_index],
                "减去公共分量后应能区分共享同一主导方向的各个身份",
            )

    def test_new_identity_invalidates_center_cache(self):
        with temporary_store() as store:
            for feature in self._shared_component_features(8):
                store.register(feature)
            before = store._feature_center_locked()
            self.assertIsNotNone(before)

            # 再注册一个方向差别很大的身份 → 中心必须重算
            store.register(unit(0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            after = store._feature_center_locked()
            self.assertIsNotNone(after)
            self.assertFalse(
                np.allclose(before, after),
                "新增身份后中心缓存必须失效，否则 query 与 gallery 不在同一坐标系",
            )

    def test_centering_survives_zero_norm_vectors(self):
        """向量几乎就是公共分量本身时，减完会趋零 —— 不得产生 NaN。"""
        with temporary_store() as store:
            features = self._shared_component_features(8)
            for feature in features:
                store.register(feature)
            center = store._feature_center_locked()
            prepared = store._prepare_for_match(center.copy(), center)
            self.assertFalse(np.isnan(prepared).any())
            self.assertEqual(prepared.shape, center.shape)


if __name__ == "__main__":
    unittest.main()
