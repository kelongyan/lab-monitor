"""
tests/test_floorplan.py — 平面图点位映射与轨迹接口

注意（src/db.py 的老坑）：`import src/db.py` 会实例化 Database() 连上
outputs/lab_monitor.db，本用例读的是生产库。只做只读查询，不写入。
"""

import copy
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.floorplan import Floorplan, build_floorplan, DEFAULT_CAMERA_MAP_FILE


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class FloorplanLoadTest(unittest.TestCase):
    """加载与清洗：非法字段不能打死服务，只能降级。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "camera_map.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_degrades_to_empty(self):
        fp = Floorplan(path=Path(self.tmp.name) / "nope.json")
        self.assertEqual(fp.total_count(), 0)
        self.assertIsNone(fp.point("reg_01"))
        payload = fp.payload()
        self.assertEqual(payload["total"], 0)
        self.assertFalse(payload["complete"])

    def test_valid_entry_parsed(self):
        _write(self.path, {
            "reg_01": {"plan_id": "IP61", "desc": "东侧走廊",
                       "map_xy": [0.43, 0.52], "facing_deg": 90, "fov_deg": 80},
        })
        fp = Floorplan(path=self.path)
        self.assertEqual(fp.total_count(), 1)
        self.assertEqual(fp.mapped_count(), 1)
        self.assertAlmostEqual(fp.point("reg_01")[0], 0.43)
        meta = fp.meta("reg_01")
        self.assertEqual(meta["plan_id"], "IP61")

    def test_unmapped_camera_has_null_point(self):
        """未标注的相机 map_xy 为 None —— 前端应跳过而不是画到原点。"""
        _write(self.path, {"reg_01": {"map_xy": None}})
        fp = Floorplan(path=self.path)
        self.assertEqual(fp.total_count(), 1)
        self.assertEqual(fp.mapped_count(), 0)
        self.assertIsNone(fp.point("reg_01"))
        self.assertFalse(fp.payload()["complete"])

    def test_coords_are_clamped_to_unit_range(self):
        _write(self.path, {"reg_01": {"map_xy": [-0.5, 1.8]}})
        fp = Floorplan(path=self.path)
        self.assertEqual(fp.point("reg_01"), [0.0, 1.0])

    def test_malformed_coords_rejected(self):
        for bad in ["0.5", [0.5], [0.5, 0.5, 0.5], [None, 0.5], [True, 0.5], {"x": 1}, 5]:
            with self.subTest(bad=bad):
                _write(self.path, {"reg_01": {"map_xy": bad}})
                fp = Floorplan(path=self.path)
                self.assertIsNone(fp.point("reg_01"))

    def test_nan_and_inf_rejected(self):
        _write(self.path, {"reg_01": {"map_xy": [float("nan"), 0.5]}})
        self.assertIsNone(Floorplan(path=self.path).point("reg_01"))
        _write(self.path, {"reg_01": {"map_xy": [float("inf"), 0.5]}})
        self.assertIsNone(Floorplan(path=self.path).point("reg_01"))

    def test_angles_fall_back_to_defaults(self):
        _write(self.path, {"reg_01": {"map_xy": [0.1, 0.1],
                                      "facing_deg": "east", "fov_deg": -5}})
        fp = Floorplan(path=self.path)
        self.assertEqual(fp.meta("reg_01")["facing_deg"], 0.0)
        self.assertEqual(fp.meta("reg_01")["fov_deg"], 1.0)  # 钳到下限

    def test_fov_upper_bound(self):
        _write(self.path, {"reg_01": {"map_xy": [0.1, 0.1], "fov_deg": 720}})
        self.assertEqual(Floorplan(path=self.path).meta("reg_01")["fov_deg"], 360.0)

    def test_non_object_root_rejected(self):
        _write(self.path, [{"reg_01": {}}])
        fp = Floorplan(path=self.path)
        self.assertEqual(fp.total_count(), 0)

    def test_non_dict_entry_skipped(self):
        _write(self.path, {"reg_01": "not a dict", "reg_02": {"map_xy": [0.2, 0.3]}})
        fp = Floorplan(path=self.path)
        self.assertEqual(fp.total_count(), 1)
        self.assertIsNotNone(fp.point("reg_02"))

    def test_corrupt_file_keeps_last_good_map(self):
        """文件被写坏时保留上一次的有效映射，不能让平面图凭空消失。"""
        _write(self.path, {"reg_01": {"map_xy": [0.3, 0.4]}})
        fp = Floorplan(path=self.path)
        self.assertIsNotNone(fp.point("reg_01"))
        self.path.write_text("{ this is not json", encoding="utf-8")
        fp.reload()
        self.assertIsNotNone(fp.point("reg_01"), "损坏后应保留旧映射")


class FloorplanHotReloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "camera_map.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_edit_without_restart_is_picked_up(self):
        _write(self.path, {"reg_01": {"map_xy": [0.1, 0.1]}})
        fp = Floorplan(path=self.path)
        self.assertEqual(fp.point("reg_01"), [0.1, 0.1])
        _write(self.path, {"reg_01": {"map_xy": [0.9, 0.9]}})
        fp.maybe_reload()
        self.assertEqual(fp.point("reg_01"), [0.9, 0.9])

    def test_missing_file_does_not_raise(self):
        fp = Floorplan(path=Path(self.tmp.name) / "gone.json")
        fp.maybe_reload()  # 不应抛异常
        self.assertEqual(fp.total_count(), 0)


class FloorplanPayloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "camera_map.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_camera_ids_filter_drops_decommissioned(self):
        _write(self.path, {
            "reg_01": {"map_xy": [0.1, 0.1]},
            "rnd_03": {"map_xy": [0.2, 0.2]},  # 已摘除的相机
        })
        fp = Floorplan(path=self.path)
        payload = fp.payload(camera_ids={"reg_01"})
        self.assertEqual(payload["total"], 1)
        self.assertNotIn("rnd_03", payload["cameras"])

    def test_payload_reports_completion(self):
        _write(self.path, {
            "reg_01": {"map_xy": [0.1, 0.1]},
            "reg_02": {"map_xy": None},
        })
        fp = Floorplan(path=self.path)
        payload = fp.payload()
        self.assertEqual(payload["mapped"], 1)
        self.assertEqual(payload["total"], 2)
        self.assertFalse(payload["complete"])
        _write(self.path, {
            "reg_01": {"map_xy": [0.1, 0.1]},
            "reg_02": {"map_xy": [0.2, 0.2]},
        })
        fp.reload()
        self.assertTrue(fp.payload()["complete"])

    def test_empty_map_is_not_complete(self):
        _write(self.path, {})
        self.assertFalse(Floorplan(path=self.path).payload()["complete"])


class ShippedCameraMapTest(unittest.TestCase):
    """仓库里的 config/camera_map.json 必须覆盖全部在役相机（坐标可以待标）。"""

    def test_covers_all_configured_cameras(self):
        from src.floorplan import DEFAULT_CAMERA_MAP_FILE as MAP
        if not MAP.exists():
            self.skipTest("camera_map.json 未生成")
        sources = json.loads(
            (Path(__file__).resolve().parent.parent / "config" / "sources.json")
            .read_text(encoding="utf-8")
        )
        fp = Floorplan(path=MAP)
        missing = sorted(set(sources) - set(fp.payload()["cameras"]))
        self.assertEqual(missing, [], f"camera_map.json 缺少这些在役相机: {missing}")


# ------------------------------------------------------------------ #
# 轨迹查询（src/db.py 新增的 query_trajectory）                          #
# ------------------------------------------------------------------ #

class TrajectoryQueryTest(unittest.TestCase):
    """用内存库验证时间窗 / 相机过滤 / 上限，不碰生产数据。"""

    def setUp(self):
        from src.db import Database
        self.tmp = tempfile.mkdtemp()
        self.db = Database(Path(self.tmp) / "t.db")
        self.gid = "abcdef01"
        for n, (cam, ts) in enumerate([
            ("reg_01", 1000.0), ("reg_01", 1002.0), ("reg_01", 1004.0),
            ("rnd_08", 1100.0), ("rnd_08", 1102.0),
            ("reg_01", 1200.0),
        ], start=1):
            # total_appearances 是必填参数（record_appearance 用它更新 identities 汇总列）
            self.db.record_appearance(self.gid, cam, ts, [0, 0, 10, 10], n)

    def tearDown(self):
        # Database 用线程本地连接 + WAL，Windows 上 close() 后 -wal/-shm 释放有延迟，
        # 直接 rmtree 会撞 WinError 32。清理失败无害（临时目录在系统 TEMP 下）。
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_sequence_ordered(self):
        total, rows = self.db.query_trajectory(self.gid)
        self.assertEqual(total, 6)
        self.assertEqual([r["camera"] for r in rows],
                         ["reg_01", "reg_01", "reg_01", "rnd_08", "rnd_08", "reg_01"])
        self.assertEqual([r["time"] for r in rows],
                         [1000.0, 1002.0, 1004.0, 1100.0, 1102.0, 1200.0])

    def test_start_filter(self):
        total, rows = self.db.query_trajectory(self.gid, start=1100.0)
        self.assertEqual(total, 3)
        self.assertTrue(all(r["time"] >= 1100.0 for r in rows))

    def test_end_filter(self):
        total, rows = self.db.query_trajectory(self.gid, end=1004.0)
        self.assertEqual(total, 3)
        self.assertTrue(all(r["time"] <= 1004.0 for r in rows))

    def test_window_filter(self):
        total, rows = self.db.query_trajectory(self.gid, start=1002.0, end=1102.0)
        self.assertEqual(total, 4)
        self.assertEqual([r["time"] for r in rows], [1002.0, 1004.0, 1100.0, 1102.0])

    def test_camera_filter(self):
        total, rows = self.db.query_trajectory(self.gid, camera_id="rnd_08")
        self.assertEqual(total, 2)
        self.assertTrue(all(r["camera"] == "rnd_08" for r in rows))

    def test_combined_filters(self):
        total, rows = self.db.query_trajectory(
            self.gid, start=1000.0, end=1004.0, camera_id="reg_01")
        self.assertEqual(total, 3)

    def test_limit_truncates_but_total_is_unchanged(self):
        """total 是过滤后的真实行数，不受 limit 影响 —— 前端据此提示裁剪。"""
        total, rows = self.db.query_trajectory(self.gid, limit=2)
        self.assertEqual(total, 6)
        self.assertEqual(len(rows), 2)

    def test_bbox_roundtrip(self):
        _, rows = self.db.query_trajectory(self.gid, limit=1)
        self.assertEqual(rows[0]["bbox"], [0, 0, 10, 10])

    def test_unknown_identity_returns_empty(self):
        total, rows = self.db.query_trajectory("nosuchid")
        self.assertEqual(total, 0)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
