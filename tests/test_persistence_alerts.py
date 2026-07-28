import csv
import io
import json
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import server
import src.db as db_module
from src.alerter import AlertBroadcaster, AlertManager
from src.db import Database
from src.identity_store import IdentityStore


@contextmanager
def temporary_database(filename: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        database = Database(root / filename)
        try:
            yield root, database
        finally:
            database.close()


def make_alert(alert_id: str, timestamp: float, **overrides) -> dict:
    alert = {
        "alert_id": alert_id,
        "timestamp": timestamp,
        "stage": "ALERT",
        "alert_type": "MISSING_PERSON",
        "risk_level": "HIGH",
        "global_id": "person-a",
        "last_camera": "cam_a",
        "elapsed_seconds": 12.5,
        "expected_cameras": ["cam_b"],
    }
    alert.update(overrides)
    return alert


class AlertPersistenceTests(unittest.TestCase):
    def test_legacy_jsonl_import_is_complete_and_idempotent(self):
        with temporary_database("alerts.db") as (root, database):
            legacy = make_alert("", 1.0)
            legacy.pop("alert_id")
            explicit = make_alert("existing-id", 2.0)
            log_path = root / "alerts.jsonl"
            log_path.write_text(
                "\n".join([
                    json.dumps(legacy),
                    json.dumps(legacy),
                    json.dumps(explicit),
                ]),
                encoding="utf-8",
            )

            self.assertEqual(database.import_alert_log(log_path), 2)
            self.assertEqual(database.import_alert_log(log_path), 0)
            total, alerts = database.query_alert_page(limit=10)
            self.assertEqual(total, 2)
            self.assertTrue(any(a["alert_id"].startswith("legacy_") for a in alerts))

    def test_history_filters_before_pagination_and_reports_true_total(self):
        with temporary_database("alerts.db") as (_, database):
            for index in range(6):
                database.insert_alert(make_alert(
                    f"a-{index}",
                    float(index),
                    risk_level="HIGH" if index % 2 == 0 else "LOW",
                    last_camera="cam_a" if index < 4 else "cam_b",
                ))

            total, first_page = database.query_alert_page(
                limit=1, offset=0, camera_id="cam_a", risk_level="HIGH"
            )
            _, second_page = database.query_alert_page(
                limit=1, offset=1, camera_id="cam_a", risk_level="HIGH"
            )
            self.assertEqual(total, 2)
            self.assertEqual(len(first_page), 1)
            self.assertEqual(len(second_page), 1)
            self.assertNotEqual(first_page[0]["alert_id"], second_page[0]["alert_id"])

    def test_tick_uses_one_id_for_return_log_database_broadcast_and_snapshot(self):
        with temporary_database("alerts.db") as (root, database):
            broadcaster = AlertBroadcaster()
            manager = AlertManager(
                alert_log=root / "alerts.jsonl",
                broadcaster=broadcaster,
                screenshot_dir=root / "screenshots",
                database=database,
            )
            frame = np.full((32, 48, 3), 127, dtype=np.uint8)
            manager.watch(
                "person-a", "cam_a", ["cam_b"], 0,
                snapshot_frame=frame,
            )

            emitted = manager.tick()
            self.assertEqual(len(emitted), 1)
            alert_id = emitted[0]["alert_id"]
            final_broadcast = [
                alert for alert in broadcaster.recent() if alert["stage"] == "ALERT"
            ]
            self.assertEqual(final_broadcast[0]["alert_id"], alert_id)
            total, rows = database.query_alert_page(limit=10)
            self.assertEqual(total, 1)
            self.assertEqual(rows[0]["alert_id"], alert_id)
            logged = json.loads((root / "alerts.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(logged["alert_id"], alert_id)
            screenshot = root / "screenshots" / f"{alert_id}.jpg"
            self.assertTrue(screenshot.exists())
            self.assertIsNotNone(cv2.imread(str(screenshot)))
            self.assertEqual(emitted[0]["screenshot_url"], f"/screenshots/{alert_id}.jpg")
            manager._close_log_file()

    def test_csv_export_round_trips_special_fields(self):
        with temporary_database("alerts.db") as (_, database):
            dangerous = '=SUM(1,2), "quoted"\nnext line'
            database.insert_alert(make_alert(
                "csv-1", 1.0, global_id=dangerous, last_camera="cam,one"
            ))
            original_db = db_module.db
            db_module.db = database
            try:
                response = server.export_alerts_csv()
            finally:
                db_module.db = original_db
            text = response.body.decode("utf-8-sig")
            rows = list(csv.reader(io.StringIO(text)))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[1]), 8)
            self.assertEqual(rows[1][5], "'" + dangerous)
            self.assertEqual(rows[1][6], "cam,one")

    def test_crowd_density_warning_is_emitted_with_snapshot_and_cooldown(self):
        with temporary_database("alerts.db") as (root, database):
            broadcaster = AlertBroadcaster()
            manager = AlertManager(
                alert_log=root / "alerts.jsonl",
                broadcaster=broadcaster,
                screenshot_dir=root / "screenshots",
                database=database,
            )
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            self.assertIsNone(manager.trigger_crowd_warning("cam_a", 4, frame=frame))
            warning = manager.trigger_crowd_warning("cam_a", 5, frame=frame)
            self.assertEqual(warning["alert_type"], "CROWD_DENSITY")
            self.assertTrue((
                root / "screenshots" / f"{warning['alert_id']}.jpg"
            ).exists())
            self.assertIsNone(manager.trigger_crowd_warning("cam_a", 6, frame=frame))
            self.assertEqual(len(broadcaster.recent()), 1)
            manager.close()

    def test_scene_exit_uses_one_id_across_all_delivery_channels(self):
        with temporary_database("scene-exit.db") as (root, database):
            feature = np.array([1.0, 0.0], dtype=np.float32)
            store = IdentityStore(database=database, feature_space="test-model:2")
            global_id = store.register(feature)
            with store._lock:
                record = store._records[global_id]
                record.last_seen = 100.0
                record.last_camera = "cam_terminal"
                record.appearances.append({"time": 100.0, "bbox": [1, 2, 3, 4]})

            frame = np.full((24, 32, 3), 96, dtype=np.uint8)
            broadcaster = AlertBroadcaster()
            notifier = Mock()
            manager = AlertManager(
                alert_log=root / "alerts.jsonl",
                notifier=notifier,
                broadcaster=broadcaster,
                identity_store=store,
                scene_exit_seconds=300,
                screenshot_dir=root / "screenshots",
                frame_provider=lambda camera_id: frame if camera_id == "cam_terminal" else None,
                database=database,
            )
            manager.mark_seen(global_id)

            with self.assertLogs("alerter", level="WARNING") as captured_logs:
                try:
                    with patch("src.alerter.time.time", return_value=400.0):
                        emitted = manager.tick()
                finally:
                    manager.close()

            self.assertEqual(["SCENE_EXIT"], [alert["alert_type"] for alert in emitted])
            alert_id = emitted[0]["alert_id"]
            self.assertTrue(any("SCENE_EXIT" in line for line in captured_logs.output))
            notifier.send.assert_called_once()
            self.assertEqual(alert_id, notifier.send.call_args.args[0]["alert_id"])
            self.assertEqual(alert_id, broadcaster.recent()[0]["alert_id"])
            total, rows = database.query_alert_page(limit=10, global_id=global_id)
            self.assertEqual(1, total)
            self.assertEqual(alert_id, rows[0]["alert_id"])
            logged = json.loads((root / "alerts.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(alert_id, logged["alert_id"])
            screenshot = root / "screenshots" / f"{alert_id}.jpg"
            self.assertTrue(screenshot.exists())
            self.assertIsNotNone(cv2.imread(str(screenshot)))
            self.assertEqual(f"/screenshots/{alert_id}.jpg", emitted[0]["screenshot_url"])


class IdentityPersistenceTests(unittest.TestCase):
    def test_identity_feature_bank_and_full_trajectory_survive_restart(self):
        with temporary_database("identities.db") as (_, database):
            feature = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            store = IdentityStore(database=database, feature_space="test-model:4")
            resolution = store.register_if_new(feature)
            global_id = resolution.global_id
            for index in range(205):
                store.update_appearance(
                    global_id,
                    "cam_a" if index < 100 else "cam_b",
                    feature,
                    [index, 1, index + 2, 3],
                    quality_score=1.0,
                )

            restored = IdentityStore(database=database, feature_space="test-model:4")
            match = restored.register_if_new(feature)
            self.assertEqual(match.status, "matched")
            self.assertEqual(match.global_id, global_id)
            record = restored.get(global_id)
            self.assertEqual(record.total_appearances, 205)
            self.assertEqual(len(record.appearances), 200)
            total, appearances = database.query_identity_appearances(global_id)
            self.assertEqual(total, 205)
            self.assertEqual(len(appearances), 205)
            self.assertEqual(appearances[-1]["bbox"], [204, 1, 206, 3])

            record.feature[:] = 0
            record.appearances.clear()
            unchanged = restored.get(global_id)
            self.assertGreater(float(np.linalg.norm(unchanged.feature)), 0.9)
            self.assertEqual(len(unchanged.appearances), 200)

    def test_incompatible_feature_space_is_not_loaded(self):
        with temporary_database("identities.db") as (_, database):
            feature = np.array([1.0, 0.0], dtype=np.float32)
            original = IdentityStore(database=database, feature_space="model-a:2")
            original.register_if_new(feature)
            incompatible = IdentityStore(database=database, feature_space="model-b:2")
            self.assertEqual(incompatible.all_ids(), [])


class BroadcasterTests(unittest.TestCase):
    def test_recent_after_returns_only_missing_events(self):
        broadcaster = AlertBroadcaster(maxlen=5)
        for index in range(3):
            broadcaster.push(make_alert(f"id-{index}", time.time() + index))
        self.assertEqual(
            [alert["alert_id"] for alert in broadcaster.recent_after("id-1")],
            ["id-2"],
        )
        self.assertEqual(len(broadcaster.recent_after("unknown")), 3)


if __name__ == "__main__":
    unittest.main()
