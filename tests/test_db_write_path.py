"""
test_db_write_path.py — 身份持久化写路径回归测试（P2-10 / P2-13）

覆盖：轨迹增量写入、特征列节流与 flush() 补写、first_seen 单调保护、
last_seen 旧值守卫、索引存在性、以及重启后的身份/轨迹往返恢复。
所有用例都在 tempfile 临时库上跑，不会碰生产库 outputs/lab_monitor.db。
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.db import Database
from src.identity_store import IdentityStore


@contextmanager
def temporary_database(filename: str = "write_path.db"):
    with tempfile.TemporaryDirectory() as temp_dir:
        database = Database(Path(temp_dir) / filename)
        try:
            yield database
        finally:
            database.close()


def read_identity_row(database: Database, global_id: str) -> dict:
    with database._get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM identities WHERE global_id = ?", (global_id,)
        ).fetchone()
    return dict(row) if row is not None else {}


def make_feature(*values: float) -> np.ndarray:
    feature = np.array(values, dtype=np.float32)
    return feature / np.linalg.norm(feature)


class AppearanceWritePathTests(unittest.TestCase):
    def test_appearance_rows_are_queryable_and_json_column_is_not_written(self):
        with temporary_database() as database:
            store = IdentityStore(database=database, feature_space="test-model:4")
            global_id = store.register_if_new(make_feature(1, 0, 0, 0)).global_id
            for index in range(5):
                store.update_appearance(
                    global_id, "cam_a", make_feature(1, 0, 0, 0),
                    [index, 1, index + 2, 3], quality_score=1.0,
                )

            total, appearances = database.query_identity_appearances(global_id)
            self.assertEqual(total, 5)
            self.assertEqual(appearances[-1]["bbox"], [4, 1, 6, 3])
            self.assertEqual(appearances[-1]["camera"], "cam_a")

            row = read_identity_row(database, global_id)
            # 停写整段 JSON：新身份该列必须保持为空
            self.assertIn(row["appearances_json"], (None, ""))
            self.assertEqual(row["total_appearances"], 5)
            self.assertEqual(row["last_camera"], "cam_a")

    def test_feature_column_is_throttled_and_flush_persists_it(self):
        with temporary_database() as database:
            store = IdentityStore(database=database, feature_space="test-model:4")
            registered = make_feature(1, 0, 0, 0)
            global_id = store.register_if_new(registered).global_id
            baseline = read_identity_row(database, global_id)["feature_blob"]

            store.update_appearance(
                global_id, "cam_a", make_feature(0, 1, 0, 0),
                [0, 0, 2, 4], quality_score=1.0,
            )
            throttled = read_identity_row(database, global_id)
            # 节流窗口内特征列保持旧值，但轻量列已更新
            self.assertEqual(throttled["feature_blob"], baseline)
            self.assertGreater(throttled["last_seen"], 0.0)

            self.assertEqual(store.flush(), 1)
            flushed = read_identity_row(database, global_id)
            self.assertNotEqual(flushed["feature_blob"], baseline)
            in_memory = store.get(global_id).feature
            np.testing.assert_allclose(
                np.frombuffer(flushed["feature_blob"], dtype=np.float32),
                in_memory,
                rtol=1e-6,
            )
            # 已落盘后再次 flush 不应产生重复写入
            self.assertEqual(store.flush(), 0)

    def test_flush_is_noop_without_database(self):
        store = IdentityStore(feature_space="test-model:4")
        global_id = store.register_if_new(make_feature(1, 0, 0, 0)).global_id
        store.update_appearance(
            global_id, "cam_a", make_feature(1, 0, 0, 0), [0, 0, 1, 1]
        )
        self.assertEqual(store.flush(), 0)


class IdentityUpsertGuardTests(unittest.TestCase):
    def save(self, database: Database, **overrides) -> None:
        payload = {
            "global_id": "g1",
            "feature_dim": 2,
            "feature_blob": np.array([1, 0], dtype=np.float32).tobytes(),
            "feature_bank_count": 1,
            "feature_bank_blob": np.array([1, 0], dtype=np.float32).tobytes(),
            "total_appearances": 1,
            "last_camera": "cam_a",
            "last_seen": 100.0,
            "feature_space": "test-model:2",
            "first_seen": 100.0,
        }
        payload.update(overrides)
        self.assertTrue(database.save_identity(**payload))

    def test_first_seen_never_moves_forward(self):
        with temporary_database() as database:
            self.save(database, first_seen=100.0, last_seen=100.0)
            self.save(database, first_seen=900.0, last_seen=900.0)
            self.assertEqual(read_identity_row(database, "g1")["first_seen"], 100.0)
            # 更早的证据仍然可以把 first_seen 往前修正
            self.save(database, first_seen=50.0, last_seen=950.0)
            self.assertEqual(read_identity_row(database, "g1")["first_seen"], 50.0)

    def test_stale_write_cannot_roll_back_last_seen(self):
        with temporary_database() as database:
            self.save(database, last_seen=500.0, last_camera="cam_b", total_appearances=9)
            stale_feature = np.array([0, 1], dtype=np.float32).tobytes()
            self.save(
                database,
                last_seen=300.0,
                last_camera="cam_c",
                total_appearances=3,
                feature_blob=stale_feature,
            )
            row = read_identity_row(database, "g1")
            self.assertEqual(row["last_seen"], 500.0)
            self.assertEqual(row["last_camera"], "cam_b")
            self.assertEqual(row["total_appearances"], 9)
            # 时序守卫只保护时间戳与相机，特征列仍然写入，避免特征永久丢失
            self.assertEqual(row["feature_blob"], stale_feature)

    def test_record_appearance_guards_last_seen_and_reports_missing_row(self):
        with temporary_database() as database:
            self.save(database, last_seen=500.0, last_camera="cam_b", total_appearances=9)
            self.assertTrue(database.record_appearance(
                global_id="g1", camera_id="cam_old", timestamp=200.0,
                bbox=[1, 2, 3, 4], total_appearances=4,
            ))
            row = read_identity_row(database, "g1")
            self.assertEqual(row["last_seen"], 500.0)
            self.assertEqual(row["last_camera"], "cam_b")
            self.assertEqual(row["total_appearances"], 9)

            self.assertTrue(database.record_appearance(
                global_id="g1", camera_id="cam_new", timestamp=600.0,
                bbox=[5, 6, 7, 8], total_appearances=10,
            ))
            row = read_identity_row(database, "g1")
            self.assertEqual(row["last_seen"], 600.0)
            self.assertEqual(row["last_camera"], "cam_new")
            self.assertEqual(row["total_appearances"], 10)
            # 轨迹行不受时序守卫影响，两条都要落库
            total, _ = database.query_identity_appearances("g1")
            self.assertEqual(total, 2)

            self.assertFalse(database.record_appearance(
                global_id="missing", camera_id="cam_a", timestamp=1.0,
                bbox=[], total_appearances=1,
            ))


class SchemaIndexTests(unittest.TestCase):
    def test_required_indexes_exist(self):
        with temporary_database() as database:
            with database._get_conn() as conn:
                indexes = {
                    row["name"]: row["sql"] or ""
                    for row in conn.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type = 'index'"
                    )
                }
            for name in (
                "idx_identity_appearances_ts",
                "idx_identities_last_seen",
                "idx_alerts_cam_ts",
                "idx_identity_appearances_gid_ts",
            ):
                self.assertIn(name, indexes)
            self.assertIn("timestamp", indexes["idx_identity_appearances_ts"])
            self.assertIn("last_seen", indexes["idx_identities_last_seen"])
            self.assertIn("camera_id", indexes["idx_alerts_cam_ts"])

    def test_indexes_are_created_on_an_existing_legacy_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.db"
            legacy = Database(path)
            with legacy._get_conn() as conn:
                conn.execute("DROP INDEX idx_identities_last_seen")
                conn.execute("DROP INDEX idx_identity_appearances_ts")
                conn.commit()
            legacy.close()

            reopened = Database(path)
            try:
                with reopened._get_conn() as conn:
                    names = {
                        row["name"] for row in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'index'"
                        )
                    }
                self.assertIn("idx_identities_last_seen", names)
                self.assertIn("idx_identity_appearances_ts", names)
            finally:
                reopened.close()


class RestoreRoundTripTests(unittest.TestCase):
    def test_identity_and_trajectory_survive_restart_without_json_column(self):
        with temporary_database() as database:
            feature = make_feature(1, 0, 0, 0)
            store = IdentityStore(database=database, feature_space="test-model:4")
            global_id = store.register_if_new(feature).global_id
            for index in range(12):
                store.update_appearance(
                    global_id,
                    "cam_a" if index < 6 else "cam_b",
                    feature,
                    [index, 1, index + 2, 3],
                    quality_score=1.0,
                )
            store.flush()

            row = read_identity_row(database, global_id)
            self.assertIn(row["appearances_json"], (None, ""))

            restored = IdentityStore(database=database, feature_space="test-model:4")
            resolution = restored.register_if_new(feature)
            self.assertEqual(resolution.status, "matched")
            self.assertEqual(resolution.global_id, global_id)
            record = restored.get(global_id)
            self.assertEqual(record.total_appearances, 12)
            self.assertEqual(len(record.appearances), 12)
            self.assertEqual(record.appearances[-1]["bbox"], [11, 1, 13, 3])
            self.assertEqual(record.last_camera, "cam_b")
            self.assertEqual(restored.get_last_bbox(global_id), [11, 1, 13, 3])

    def test_legacy_json_column_still_restores_when_no_increment_rows(self):
        with temporary_database() as database:
            feature = make_feature(1, 0, 0, 0)
            legacy_appearances = [
                {"camera": "cam_legacy", "time": 10.0, "bbox": [1, 2, 3, 4]}
            ]
            with database._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO identities (
                        global_id, first_seen, last_seen, last_camera,
                        total_appearances, feature_dim, feature_blob,
                        feature_bank_count, feature_bank_blob,
                        appearances_json, feature_space, feature_schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "legacy-1", 10.0, 20.0, "cam_legacy", 1, 4,
                        feature.tobytes(), 0, b"",
                        json.dumps(legacy_appearances), "test-model:4", 1,
                    ),
                )
                conn.commit()

            restored = IdentityStore(database=database, feature_space="test-model:4")
            record = restored.get("legacy-1")
            self.assertIsNotNone(record)
            self.assertEqual(len(record.appearances), 1)
            self.assertEqual(record.appearances[-1]["camera"], "cam_legacy")


if __name__ == "__main__":
    unittest.main()
