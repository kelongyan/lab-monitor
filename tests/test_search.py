"""
test_search.py — 人员视频检索（能力二）回归测试（worklist 4.2 / 4.3 / 4.4）

三条来自实测教训的语义都在这里锁定：
  1. 双时间轴：timestamp 用于排序，video_ts 用于回放定位，检索结果必须带
     position_known 让前端降级（老数据三列为 NULL）。
  2. 循环折叠复用 src/trajectory.py（4.1 抽出），检索结果不得出现"来回 545 次"。
  3. 已下线相机必须过滤 —— 库里有 rnd_03 的 6447 条孤儿轨迹（历史配置残留），
     不过滤的话检索会返回一个已下线的相机，前端点进去就是 404。
  4. 以图搜人返回 Top-K 排序（离线检索阈值语义与实时匹配独立）。

所有用例都在 tempfile 临时库上跑，不碰生产库。
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.db import Database
from src.search import (
    aggregate_identity_assets,
    best_crop,
    rank_identities_by_feature,
)


@contextmanager
def temporary_database(filename: str = "search.db"):
    with tempfile.TemporaryDirectory() as temp_dir:
        database = Database(Path(temp_dir) / filename)
        try:
            yield database
        finally:
            database.close()


class SearchFixture(unittest.TestCase):
    """构造一个可控的检索语料：2 个资产、1 个身份、跨相机轨迹。"""

    KNOWN_CAMERAS = {"rnd_08", "rnd_19", "rnd_01"}

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.database = Database(Path(self._temp.name) / "search.db")
        self.database.seed_video_asset(
            camera_id="rnd_08", rel_path="videos_low/rnd_08.mp4",
            codec="h264", width=960, height=540,
            fps_declared=25.0, frames_real=957, duration_real=38.28)
        self.database.seed_video_asset(
            camera_id="rnd_19", rel_path="videos_low/rnd_19.mp4",
            codec="h264", width=960, height=540,
            fps_declared=25.0, frames_real=973, duration_real=38.92)
        self.asset_rnd08 = self.database.get_video_asset("rnd_08")["asset_id"]
        self.asset_rnd19 = self.database.get_video_asset("rnd_19")["asset_id"]

        self.gid = "person-a"
        self.database.save_identity(
            global_id=self.gid, feature_dim=4,
            feature_blob=b"\x00" * 16, feature_bank_count=0, feature_bank_blob=b"",
            total_appearances=0, last_camera="rnd_08", last_seen=1000.0,
            feature_space="test:4", first_seen=1000.0,
        )
        # 交替相机 A,B,A,B,...（与真实数据 rnd_03<->rnd_04 的循环模式一致），
        # 25 段 → 周期 2 → 折叠回 1 轮（2 段）
        for k in range(25):
            cam = "rnd_08" if k % 2 == 0 else "rnd_19"
            asset_id = self.asset_rnd08 if k % 2 == 0 else self.asset_rnd19
            frame = k * 10
            self.database.record_appearance(
                self.gid, cam, 1000.0 + k, [0, 0, 50, 100], k + 1,
                asset_id=asset_id, video_frame=frame, video_ts=frame / 25.0)
        # 已下线相机 rnd_03 的孤儿轨迹（worklist 4.3）
        self.database.record_appearance(
            self.gid, "rnd_03", 3000.0, [0, 0, 70, 140], 200,
            video_frame=3, video_ts=0.12)

    def tearDown(self):
        self.database.close()
        self._temp.cleanup()


class AggregateTests(SearchFixture):
    def test_assets_are_grouped_by_file(self):
        result = aggregate_identity_assets(
            self.database, self.gid,
            known_cameras=self.KNOWN_CAMERAS)
        self.assertEqual(result["asset_count"], 2, "应聚合到 2 个资产")
        files = {asset["file"] for asset in result["assets"]}
        self.assertEqual(files, {"videos_low/rnd_08.mp4", "videos_low/rnd_19.mp4"})

    def test_loop_collapse_prevents_infinite_repeats(self):
        result = aggregate_identity_assets(
            self.database, self.gid, known_cameras=self.KNOWN_CAMERAS)
        self.assertTrue(result["loop"]["detected"])
        rnd08 = next(a for a in result["assets"] if a["camera_id"] == "rnd_08")
        self.assertEqual(len(rnd08["segments"]), 1,
                         "25 个连续同相机停留点应折叠成 1 段，否则就是'来回跑 25 次'")

    def test_video_coordinates_are_reported(self):
        result = aggregate_identity_assets(
            self.database, self.gid, known_cameras=self.KNOWN_CAMERAS)
        self.assertTrue(result["position_known_rows"] > 0)
        rnd19 = next(a for a in result["assets"] if a["camera_id"] == "rnd_19")
        self.assertTrue(rnd19["position_known"])
        # 循环折叠把轨迹截断到第一轮：rnd_19 只剩第一轮里的那一行（k=1 → 帧 10）
        self.assertAlmostEqual(rnd19["video_first_ts"], 0.4, places=2)
        self.assertAlmostEqual(rnd19["video_last_ts"], 0.4, places=2)

    def test_offline_camera_is_filtered(self):
        """rnd_03 已下线：孤儿轨迹必须被过滤，否则前端点进去 404。"""
        result = aggregate_identity_assets(
            self.database, self.gid, known_cameras=self.KNOWN_CAMERAS)
        cameras = {asset["camera_id"] for asset in result["assets"]}
        self.assertNotIn("rnd_03", cameras)
        self.assertEqual(result["dropped_offline_rows"], 1)

    def test_legacy_rows_degrade_position_known(self):
        """没有视频内坐标的行要标 position_known=False，不能编造坐标。"""
        self.database.record_appearance(
            self.gid, "rnd_01", 4000.0, [0, 0, 80, 160], 300)
        result = aggregate_identity_assets(
            self.database, self.gid, known_cameras=self.KNOWN_CAMERAS)
        self.assertFalse(result["position_known_rows"] == result["returned_rows"])
        legacy = next(a for a in result["assets"] if a["camera_id"] == "rnd_01")
        self.assertFalse(legacy["position_known"])

    def test_time_window_filters(self):
        # 交替夹具里 k=0 是 rnd_08（墙钟 1000.0）、k=1 是 rnd_19（1001.0）
        result = aggregate_identity_assets(
            self.database, self.gid, start=0, end=1000.5,
            known_cameras=self.KNOWN_CAMERAS)
        cameras = {asset["camera_id"] for asset in result["assets"]}
        self.assertEqual(cameras, {"rnd_08"}, "时间窗应过滤掉 rnd_19")

    def test_unknown_identity_is_empty(self):
        result = aggregate_identity_assets(
            self.database, "no-such-id", known_cameras=self.KNOWN_CAMERAS)
        self.assertEqual(result["asset_count"], 0)
        self.assertEqual(result["total_appearances"], 0)


class RankByFeatureTests(unittest.TestCase):
    """以图搜人：Top-K 排序语义。

    刻意不继承 SearchFixture —— 那个夹具会存一个零向量身份（模拟老库的塌缩行），
    它中心化后会得到一个反向的伪方向，干扰 Top-1 判定。
    这里用两个彼此接近正交的身份，让 Top-1 可判定。
    """

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.database = Database(Path(self._temp.name) / "rank.db")

    def tearDown(self):
        self.database.close()
        self._temp.cleanup()

    def _feature_for(self, direction: float) -> np.ndarray:
        base = np.zeros(4, dtype=np.float32)
        base[0] = 1.0
        base[1] = direction
        return base / np.linalg.norm(base)

    def _make_store(self):
        from src.identity_store import IdentityStore
        store = IdentityStore(database=self.database, feature_space="test:4")
        # 两个彼此接近正交的身份，Top-1 语义可判定
        self.gid_a = store.register(self._feature_for(0.05))
        other = np.array([0, 0, 1, 0.05], dtype=np.float32)
        self.gid_b = store.register(other / np.linalg.norm(other))
        return store

    def test_top1_matches_query_identity(self):
        store = self._make_store()
        query = self._feature_for(0.05)
        result = rank_identities_by_feature(store, query, top_k=3)
        self.assertEqual(result["matches"][0]["global_id"], self.gid_a)
        self.assertTrue(result["matches"][0]["matched"])
        self.assertGreater(
            result["matches"][0]["score"], result["matches"][1]["score"])

    def test_ranking_is_sorted_descending(self):
        store = self._make_store()
        result = rank_identities_by_feature(store, self._feature_for(0.0), top_k=5)
        scores = [m["score"] for m in result["matches"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unknown_direction_still_ranks_without_crash(self):
        store = self._make_store()
        # 与两个身份都正交的方向：分数低，但不得崩溃或产生 NaN
        query = np.array([0, 0, 0, 1], dtype=np.float32)
        result = rank_identities_by_feature(store, query, top_k=5)
        self.assertTrue(all(np.isfinite(m["score"]) for m in result["matches"]))

    def test_best_crop_picks_largest_area(self):
        detections = [[0, 0, 40, 80], [100, 100, 300, 500], [500, 0, 520, 60]]
        best = best_crop(detections)
        self.assertEqual(best, [100, 100, 300, 500])
        self.assertIsNone(best_crop([]))


if __name__ == "__main__":
    unittest.main()
