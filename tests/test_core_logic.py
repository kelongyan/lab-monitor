import json
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.alerter import AlertBroadcaster, AlertManager
from src.calibrator import TransitCalibrator
from src.identity_store import IdentityStore, ReIDMetrics
from src.reid_validator import ReIDValidator


def normalized(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TransitCalibratorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stats_path = Path(self.temp_dir.name) / "transit_stats.json"
        self.valid_edges = {("cam_a", "cam_b"), ("cam_a", "cam_c")}
        self.calibrator = TransitCalibrator(
            self.stats_path,
            valid_edges=self.valid_edges,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cross_camera_departure_is_shared_and_consumed_once(self):
        self.assertTrue(self.calibrator.record_departure(
            "person-1", "cam_a", ["cam_b"], timestamp=100.0
        ))
        self.assertTrue(self.calibrator.record_arrival(
            "person-1", "cam_b", timestamp=112.5
        ))
        self.assertFalse(self.calibrator.record_arrival(
            "person-1", "cam_b", timestamp=113.0
        ))

        stats = self.calibrator.stats()
        self.assertEqual(1, stats["cam_a→cam_b"]["count"])
        self.assertEqual(12.5, stats["cam_a→cam_b"]["mean_seconds"])

    def test_same_or_unexpected_camera_does_not_consume_departure(self):
        self.calibrator.record_departure(
            "person-1", "cam_a", ["cam_b"], timestamp=100.0
        )
        self.assertFalse(self.calibrator.record_arrival(
            "person-1", "cam_a", timestamp=105.0
        ))
        self.assertFalse(self.calibrator.record_arrival(
            "person-1", "cam_c", timestamp=106.0
        ))
        self.assertTrue(self.calibrator.record_arrival(
            "person-1", "cam_b", timestamp=107.0
        ))
        self.assertNotIn("cam_a→cam_a", self.calibrator.stats())
        self.assertNotIn("cam_a→cam_c", self.calibrator.stats())

    def test_concurrent_arrivals_record_only_one_sample(self):
        self.calibrator.record_departure(
            "person-1", "cam_a", ["cam_b"], timestamp=100.0
        )
        barrier = threading.Barrier(8)
        results = []
        result_lock = threading.Lock()

        def arrive(index):
            barrier.wait()
            result = self.calibrator.record_arrival(
                "person-1", "cam_b", timestamp=110.0 + index
            )
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=arrive, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(1, results.count(True))
        self.assertEqual(1, self.calibrator.stats()["cam_a→cam_b"]["count"])

    def test_load_removes_self_loops_and_non_topology_edges(self):
        self.stats_path.write_text(json.dumps({
            "cam_a→cam_a": [5],
            "cam_b→cam_c": [7],
            "cam_a→cam_b": [9, -1, 5000, "bad"],
        }), encoding="utf-8")

        cleaned = TransitCalibrator(self.stats_path, valid_edges=self.valid_edges)

        self.assertEqual({"cam_a→cam_b"}, set(cleaned.stats()))
        self.assertEqual(
            {"cam_a→cam_b": [9.0]},
            json.loads(self.stats_path.read_text("utf-8")),
        )


class IdentityResolutionTests(unittest.TestCase):
    def test_ambiguous_match_is_not_merged_or_registered(self):
        store = IdentityStore()
        first_id = store.register(normalized([1.0, 0.0, 0.0]))
        second_id = store.register(normalized([0.99, 0.1, 0.0]))
        query = normalized([0.997, 0.05, 0.0])

        result = store.register_if_new(query)

        self.assertEqual("ambiguous", result.status)
        self.assertIsNone(result.global_id)
        self.assertEqual({first_id, second_id}, set(store.all_ids()))

    def test_clear_match_reuses_existing_identity(self):
        store = IdentityStore()
        first_id = store.register(normalized([1.0, 0.0]))
        store.register(normalized([0.0, 1.0]))

        result = store.register_if_new(normalized([0.99, 0.1]))

        self.assertEqual("matched", result.status)
        self.assertEqual(first_id, result.global_id)
        self.assertFalse(result.is_new)

    def test_concurrent_registration_creates_one_identity(self):
        store = IdentityStore()
        feature = normalized([1.0, 0.0, 0.0])
        barrier = threading.Barrier(8)
        results = []
        result_lock = threading.Lock()

        def register():
            barrier.wait()
            result = store.register_if_new(feature)
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=register) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(1, len(store.all_ids()))
        self.assertEqual(1, sum(result.is_new for result in results))
        self.assertEqual(1, len({result.global_id for result in results}))

    def test_failed_search_is_not_counted_as_success(self):
        metrics = ReIDMetrics()
        validator = ReIDValidator(buffer_size=2, confirm_frames=1, threshold=0.75)
        query = normalized([0.0, 1.0])
        validator.add_feature(1, query)
        validator.add_feature(1, query)

        result = validator.get_confirmed_match(
            1,
            [("person-1", normalized([1.0, 0.0]))],
            metrics=metrics,
        )

        self.assertIsNone(result)
        summary = metrics.get_summary(gallery_size=1)
        self.assertEqual(1, summary["total_searches"])
        self.assertEqual(0, summary["successful_matches"])
        self.assertEqual(0.0, summary["match_rate"])


class AlertStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = AlertManager(
            alert_log=Path(self.temp_dir.name) / "alerts.jsonl"
        )

    def tearDown(self):
        self.manager._close_log_file()
        self.temp_dir.cleanup()

    def test_unexpected_camera_keeps_watch_until_expected_arrival(self):
        self.manager.watch("person-1", "cam_a", ["cam_b"], 30)

        self.assertFalse(self.manager.resolve("person-1", "cam_c"))
        self.assertIn("person-1", self.manager._watches)
        self.assertTrue(self.manager.resolve("person-1", "cam_b"))
        self.assertNotIn("person-1", self.manager._watches)

    def test_missing_person_escalates_to_scene_exit_after_grace_period(self):
        store = IdentityStore()
        global_id = store.register(normalized([1.0, 0.0]))
        with store._lock:
            record = store._records[global_id]
            record.last_seen = 700.0
            record.last_camera = "cam_a"
            record.appearances = deque([{"bbox": [1, 2, 3, 4]}], maxlen=200)

        broadcaster = AlertBroadcaster()
        manager = AlertManager(
            alert_log=Path(self.temp_dir.name) / "dedupe.jsonl",
            broadcaster=broadcaster,
            identity_store=store,
            scene_exit_seconds=300,
        )
        captured = []
        manager._write_log = captured.append
        with patch("src.alerter.time.time", return_value=700.0):
            manager.watch(global_id, "cam_a", ["cam_b"], deadline_offset=45)
        with patch("src.alerter.time.time", return_value=745.0):
            manager.tick()

        self.assertEqual(["MISSING_PERSON"], [a["alert_type"] for a in captured])
        self.assertNotIn("SCENE_EXIT", [
            alert["alert_type"] for alert in broadcaster.recent()
        ])

        manager._check_scene_exits(1044.9)
        self.assertEqual(["MISSING_PERSON"], [a["alert_type"] for a in captured])

        manager._check_scene_exits(1045.0)
        self.assertEqual(
            ["MISSING_PERSON", "SCENE_EXIT"],
            [a["alert_type"] for a in captured],
        )
        manager._check_scene_exits(1300.0)
        self.assertEqual(2, len(captured))
        manager._close_log_file()

    def test_active_topology_watch_suppresses_scene_exit(self):
        store = IdentityStore()
        global_id = store.register(normalized([1.0, 0.0]))
        with store._lock:
            record = store._records[global_id]
            record.last_seen = 100.0
            record.last_camera = "cam_a"

        manager = AlertManager(
            alert_log=Path(self.temp_dir.name) / "active-watch.jsonl",
            identity_store=store,
            scene_exit_seconds=300,
        )
        captured = []
        manager._write_log = captured.append
        with patch("src.alerter.time.time", return_value=100.0):
            manager.watch(global_id, "cam_a", ["cam_b"], deadline_offset=600)
        with patch("src.alerter.time.time", return_value=400.0):
            self.assertEqual([], manager.tick())

        self.assertEqual([], captured)
        self.assertIn(global_id, manager._watches)
        self.assertNotIn(global_id, manager._scene_exit_alerted)
        manager._close_log_file()

    def test_terminal_camera_scene_exit_resets_after_reappearance(self):
        store = IdentityStore()
        global_id = store.register(normalized([1.0, 0.0]))
        with store._lock:
            record = store._records[global_id]
            record.last_seen = 100.0
            record.last_camera = "cam_terminal"
            record.appearances = deque([{"bbox": [5, 6, 7, 8]}], maxlen=200)

        log_path = Path(self.temp_dir.name) / "scene-exit.jsonl"
        broadcaster = AlertBroadcaster()
        manager = AlertManager(
            alert_log=log_path,
            broadcaster=broadcaster,
            identity_store=store,
            scene_exit_seconds=300,
        )
        manager.mark_seen(global_id)

        with patch("src.alerter.time.time", return_value=399.9):
            self.assertEqual([], manager.tick())
        with patch("src.alerter.time.time", return_value=400.0):
            first_cycle = manager.tick()
        self.assertEqual(["SCENE_EXIT"], [a["alert_type"] for a in first_cycle])
        self.assertEqual([5, 6, 7, 8], first_cycle[0]["last_bbox"])
        self.assertEqual(first_cycle[0]["alert_id"], broadcaster.recent()[0]["alert_id"])

        manager.mark_seen(global_id)
        with store._lock:
            store._records[global_id].last_seen = 500.0
        with patch("src.alerter.time.time", return_value=799.9):
            self.assertEqual([], manager.tick())
        with patch("src.alerter.time.time", return_value=800.0):
            second_cycle = manager.tick()
        self.assertEqual(["SCENE_EXIT"], [a["alert_type"] for a in second_cycle])

        logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(2, len(logged))
        self.assertEqual(
            [alert["alert_id"] for alert in first_cycle + second_cycle],
            [alert["alert_id"] for alert in logged],
        )

        manager._close_log_file()

    def test_restored_stale_identity_is_not_alerted_before_current_session_sighting(self):
        store = IdentityStore()
        global_id = store.register(normalized([1.0, 0.0]))
        with store._lock:
            store._records[global_id].last_seen = 100.0
            store._records[global_id].last_camera = "cam_old"

        manager = AlertManager(
            alert_log=Path(self.temp_dir.name) / "restored.jsonl",
            identity_store=store,
            scene_exit_seconds=300,
        )
        with patch("src.alerter.time.time", return_value=1000.0):
            self.assertEqual([], manager.tick())

        with store._lock:
            store._records[global_id].last_seen = 1000.0
            store._records[global_id].last_camera = "cam_live"
        manager.mark_seen(global_id)
        with patch("src.alerter.time.time", return_value=1300.0):
            emitted = manager.tick()
        self.assertEqual(["SCENE_EXIT"], [alert["alert_type"] for alert in emitted])
        manager._close_log_file()

    def test_reappearance_generation_blocks_stale_scene_exit_snapshot(self):
        store = IdentityStore()
        global_id = store.register(normalized([1.0, 0.0]))
        with store._lock:
            store._records[global_id].last_seen = 100.0
            store._records[global_id].last_camera = "cam_old"

        manager = AlertManager(
            alert_log=Path(self.temp_dir.name) / "generation-race.jsonl",
            identity_store=store,
            scene_exit_seconds=300,
        )
        manager.mark_seen(global_id)
        original_get = store.get

        def reappear_during_snapshot(person_id):
            stale_snapshot = original_get(person_id)
            with store._lock:
                store._records[person_id].last_seen = 400.0
                store._records[person_id].last_camera = "cam_live"
            manager.mark_seen(person_id)
            return stale_snapshot

        with patch.object(store, "get", side_effect=reappear_during_snapshot):
            with patch("src.alerter.time.time", return_value=400.0):
                self.assertEqual([], manager.tick())
        self.assertNotIn(global_id, manager._scene_exit_alerted)
        manager._close_log_file()


if __name__ == "__main__":
    unittest.main()
