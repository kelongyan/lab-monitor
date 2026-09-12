"""
test_video_assets.py — 视频资产索引与"视频内坐标"回归测试（worklist 2.1 / 2.2 / 2.3）

背景：这批素材的容器元数据不可信（ffprobe 与 OpenCV 一致报出 11948~71617 秒的荒谬
时长、nb_frames 缺失），而"人员视频检索"要能回答"出现在哪个文件、第几秒"，
必须自建资产索引 + 在轨迹行上落"视频内坐标"。

三列（asset_id / video_frame / video_ts）都可能为 NULL：
老数据没有这些值，检索侧必须降级为"仅相机 + 墙钟时间"，绝不做无根据的回填猜测。

所有用例都在 tempfile 临时库上跑，不碰生产库。
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from src.db import Database


@contextmanager
def temporary_database(filename: str = "video_assets.db"):
    with tempfile.TemporaryDirectory() as temp_dir:
        database = Database(Path(temp_dir) / filename)
        try:
            yield database
        finally:
            database.close()


class VideoAssetIndexTests(unittest.TestCase):
    def test_stub_insert_is_idempotent_and_returns_same_asset_id(self):
        with temporary_database() as database:
            first = database.upsert_video_asset_stub("rnd_08", "videos_low/rnd_08.mp4")
            second = database.upsert_video_asset_stub("rnd_08", "videos_low/rnd_08.mp4")
            self.assertEqual(first, second, "同一 (camera, path) 应复用同一行")

    def test_stub_does_not_clobber_seeded_metadata(self):
        """占位登记用 INSERT OR IGNORE，绝不覆盖 seeder 的实测元数据。"""
        with temporary_database() as database:
            database.seed_video_asset(
                camera_id="rnd_08", rel_path="videos_low/rnd_08.mp4",
                file_name="rnd_08.mp4", codec="h264", width=960, height=540,
                fps_declared=25.0, frames_real=957, duration_real=38.28,
                low_value=0,
            )
            database.upsert_video_asset_stub("rnd_08", "videos_low/rnd_08.mp4",
                                             file_name="other.mp4")
            asset = database.get_video_asset("rnd_08")
            self.assertEqual(asset["codec"], "h264")
            self.assertEqual(asset["frames_real"], 957)
            self.assertEqual(asset["duration_real"], 38.28)

    def test_seed_upserts_and_reports_low_value(self):
        """内容过短（< 5 秒）必须标 low_value —— rnd_05 的真实内容只有十几秒。"""
        with temporary_database() as database:
            database.seed_video_asset(
                camera_id="rnd_05", rel_path="videos_low/rnd_05.mp4",
                fps_declared=25.0, frames_real=460, duration_real=18.4, low_value=0,
            )
            database.seed_video_asset(
                camera_id="rnd_05", rel_path="videos_low/rnd_05.mp4",
                fps_declared=25.0, frames_real=460, duration_real=18.4, low_value=1,
            )
            asset = database.get_video_asset("rnd_05")
            self.assertEqual(asset["frames_real"], 460)
            self.assertEqual(asset["low_value"], 1)

    def test_list_video_assets_and_get_by_camera(self):
        with temporary_database() as database:
            database.seed_video_asset(camera_id="a", rel_path="p/a.mp4", codec="h264")
            database.seed_video_asset(camera_id="b", rel_path="p/b.mp4", codec="hevc")
            self.assertEqual(len(database.list_video_assets()), 2)
            self.assertEqual(database.get_video_asset("a")["codec"], "h264")
            self.assertIsNone(database.get_video_asset("missing"))


class VideoCoordinateTests(unittest.TestCase):
    """轨迹行的视频内坐标：写入、读取、以及老数据的 NULL 降级。"""

    def _make_identity(self, database: Database) -> str:
        database.seed_video_asset(camera_id="rnd_08", rel_path="videos_low/rnd_08.mp4",
                                  fps_declared=25.0, frames_real=957,
                                  duration_real=38.28)
        asset = database.get_video_asset("rnd_08")
        gid = "test-gid-1"
        database.save_identity(
            global_id=gid, feature_dim=4,
            feature_blob=b"\x00" * 16, feature_bank_count=0, feature_bank_blob=b"",
            total_appearances=0, last_camera="rnd_08", last_seen=1000.0,
            feature_space="test:4", first_seen=1000.0,
        )
        return gid, asset["asset_id"]

    def test_record_appearance_persists_video_coordinates(self):
        with temporary_database() as database:
            gid, asset_id = self._make_identity(database)
            ok = database.record_appearance(
                global_id=gid, camera_id="rnd_08", timestamp=1005.0,
                bbox=[1, 2, 3, 4], total_appearances=1,
                asset_id=asset_id, video_frame=125, video_ts=5.0,
            )
            self.assertTrue(ok)
            total, rows = database.query_trajectory(gid, limit=10)
            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["asset_id"], asset_id)
            self.assertEqual(rows[0]["video_frame"], 125)
            self.assertAlmostEqual(rows[0]["video_ts"], 5.0)

    def test_legacy_rows_degrade_to_null_coordinates(self):
        """老数据三列为 NULL —— 检索侧必须能据此降级，不能报错也不能编造坐标。"""
        with temporary_database() as database:
            gid, _ = self._make_identity(database)
            database.record_appearance(
                global_id=gid, camera_id="rnd_08", timestamp=1005.0,
                bbox=[1, 2, 3, 4], total_appearances=1,
            )
            _, rows = database.query_trajectory(gid, limit=10)
            self.assertIsNone(rows[0]["asset_id"])
            self.assertIsNone(rows[0]["video_frame"])
            self.assertIsNone(rows[0]["video_ts"])

    def test_save_identity_new_appearance_carries_coordinates(self):
        with temporary_database() as database:
            gid, asset_id = self._make_identity(database)
            database.save_identity(
                global_id=gid, feature_dim=4, feature_blob=b"\x00" * 16,
                feature_bank_count=0, feature_bank_blob=b"",
                total_appearances=1, last_camera="rnd_08", last_seen=1010.0,
                feature_space="test:4", first_seen=1000.0,
                new_appearance={
                    "camera": "rnd_08", "time": 1010.0, "bbox": [0, 0, 8, 16],
                    "asset_id": asset_id, "video_frame": 250, "video_ts": 10.0,
                },
            )
            _, rows = database.query_trajectory(gid, limit=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_frame"], 250)
            self.assertAlmostEqual(rows[0]["video_ts"], 10.0)

    def test_merge_identities_repoints_appearance_rows(self):
        """归并后轨迹必须改挂主身份，否则检索断链（worklist 1.8）。"""
        with temporary_database() as database:
            gid, asset_id = self._make_identity(database)
            database.record_appearance(gid, "rnd_08", 1005.0, [1, 2, 3, 4], 1,
                                       asset_id=asset_id, video_frame=125, video_ts=5.0)
            database.merge_identities(keep_id=gid, drop_id="dup-gid")
            # 被并方的轨迹行也要跟着改挂
            database.record_appearance("dup-gid", "rnd_08", 1006.0, [1, 2, 3, 5], 1,
                                       asset_id=asset_id, video_frame=150, video_ts=6.0)
            # 先造一条挂在被并方名下的行
            with database._get_conn() as conn:
                conn.execute(
                    "UPDATE identity_appearances SET global_id = 'dup-gid' "
                    "WHERE video_frame = 150"
                )
                conn.commit()
            database.merge_identities(keep_id=gid, drop_id="dup-gid")
            _, rows = database.query_trajectory(gid, limit=10)
            frames = sorted(row["video_frame"] for row in rows)
            self.assertEqual(frames, [125, 150], "被并方的轨迹必须改挂主身份")
            with database._get_conn() as conn:
                orphan = conn.execute(
                    "SELECT COUNT(*) FROM identity_appearances WHERE global_id = 'dup-gid'"
                ).fetchone()[0]
            self.assertEqual(orphan, 0, "不能留下挂在被并方名下的轨迹")


class VideoCoordinateIndexTests(unittest.TestCase):
    def test_gid_cam_ts_index_exists(self):
        """检索接口按「身份 + 相机 + 时间」聚合，缺索引会全表扫描。"""
        with temporary_database() as database:
            with database._get_conn() as conn:
                names = {row["name"] for row in
                         conn.execute("PRAGMA index_list(identity_appearances)")}
            self.assertIn("idx_appearances_gid_cam_ts", names)


if __name__ == "__main__":
    unittest.main()
