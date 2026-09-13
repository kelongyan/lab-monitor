"""
test_pipeline_stream_state.py — 覆盖 F2/F2b/F3/F10 在 pipeline 侧的行为：
  - 重开流时重置逐帧状态，但不得触碰 ByteTrack 的类级 track_id 计数器
  - 离场宽限期：一次关联失败不算离场，不误报 MISSING_PERSON、不清 ReID 缓冲
  - _read_loop 返回本轮解码帧数（坏源熔断依赖它）
  - ROI 配置清洗：字符串坐标、越界坐标、顶点数不足都不能让线程崩掉
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from ultralytics.trackers.basetrack import BaseTrack

from src.pipeline import CameraPipeline

# 行为测试固定用这个宽限值，不受 LAB_MONITOR_LEAVE_GRACE_FRAMES 环境变量影响
GRACE = 3


def _build_pipeline(source: str = "videos/none.mp4",
                    identity_store=None, personnel=None) -> CameraPipeline:
    return CameraPipeline(
        camera_id="cam_test",
        source=source,
        detector=Mock(),
        reid_extractor=Mock(),
        identity_store=identity_store if identity_store is not None else Mock(),
        topology=Mock(),
        alert_manager=Mock(),
        screenshot_dir=Path("outputs"),
        frame_hub=None,
        personnel=personnel,
    )


class AssetFpsRegistrationTests(unittest.TestCase):
    """登记视频资产后，帧率必须取自**刚登记的那一行**（2026-09-13 新增）。

    一个相机在库里有原片与低清片两行，`video_asset_for()` 按 `ORDER BY asset_id DESC`
    返回低清片那行；而 `_file_frame_idx` 数的是本路实际打开的文件（config/sources.json
    指向的原片）。帧率若来自另一行，`video_ts` 就会被静默算错 ——
    今天 44 行都是 25.0 fps 所以数值上看不出来。
    """

    def _store(self, fps_registered, fps_other):
        store = Mock()
        store.register_video_asset_stub.return_value = 7
        store.video_asset_by_id.return_value = {"fps_declared": fps_registered}
        store.video_asset_for.return_value = {"fps_declared": fps_other}
        return store

    def test_fps_comes_from_the_registered_row(self):
        store = self._store(fps_registered=30.0, fps_other=25.0)
        pipeline = _build_pipeline(identity_store=store)
        self.assertEqual(30.0, pipeline._video_fps)
        store.video_asset_by_id.assert_called_with(7)

    def test_insane_fps_falls_back_to_default(self):
        """损坏读数（rnd_05 报 351.56）必须被区间护栏挡住，回落到 25.0。"""
        store = self._store(fps_registered=351.56, fps_other=25.0)
        pipeline = _build_pipeline(identity_store=store)
        self.assertEqual(25.0, pipeline._video_fps)

    def test_falls_back_to_camera_row_when_registration_failed(self):
        """登记失败（返回 None）时退回按相机取，不能因此丢掉帧率。"""
        store = Mock()
        store.register_video_asset_stub.return_value = None
        store.video_asset_for.return_value = {"fps_declared": 30.0}
        pipeline = _build_pipeline(identity_store=store)
        self.assertEqual(30.0, pipeline._video_fps)
        store.video_asset_by_id.assert_not_called()


class NamedIdentityPriorityTests(unittest.TestCase):
    """
    底库命中并解析出"已绑定的实名身份"后，匿名匹配与注册**不得**覆盖它
    （2026-09-13 新增）。

    原实现里 `gid = known_gid` 之后，下面两处赋值（匿名确认 `confirmed_gid`、
    注册结果 `resolution.global_id`）都是无条件写回 `gid`，把"实名优先"这条
    写在注释里的契约架空了。同一个人常背 1~3 个重复匿名身份，
    validator 一旦确认了其中一个，画面标签就会从"张三 (P0007)"退回 "ID: #xxxx"。
    """

    TID = 7
    FEATURE = np.array([1.0] + [0.0] * 15, dtype=np.float32)

    def _pipeline(self):
        store = Mock()
        context = Mock()
        context.gallery = []
        context.prepare = lambda vector: vector
        store.build_match_context.return_value = context

        personnel = Mock()
        personnel.match.return_value = {"person_id": "P0001", "name": "张三",
                                        "score": 0.91}
        personnel.bound_gid.return_value = "gid-named"

        pipeline = _build_pipeline(identity_store=store, personnel=personnel)
        pipeline._detect_every_n = 1000          # 走"非检测帧"分支，不碰 detector/tracker
        pipeline._frame_idx = 1
        pipeline.reid.extract = Mock(return_value=self.FEATURE)
        pipeline._on_person_leave = Mock()

        validator = Mock()
        validator.get_avg_feature.return_value = self.FEATURE
        validator.get_confirmed_match.return_value = "gid-anon"   # 匿名库有"意见"
        validator.buffer_len.return_value = 8
        validator.buffer_size = 8
        pipeline._validator = validator
        return pipeline

    def _drive(self, pipeline):
        pipeline._last_tracks = [{"track_id": self.TID,
                                  "bbox": [1.0, 1.0, 20.0, 50.0],
                                  "conf": 0.9}]
        pipeline._process_frame(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_named_identity_is_not_overwritten_by_anonymous_match(self):
        pipeline = self._pipeline()
        self._drive(pipeline)
        self.assertEqual("gid-named", pipeline._track_to_global.get(self.TID),
                         "底库已解析出实名身份，匿名确认不得覆盖它")

    def test_anonymous_path_is_not_even_consulted(self):
        """既然已认出实名，就不该再走匿名确认/注册那条路（无事发生好过做错事）。"""
        pipeline = self._pipeline()
        self._drive(pipeline)
        pipeline._validator.get_confirmed_match.assert_not_called()
        pipeline.store.register_if_new.assert_not_called()

    def test_auto_naming_still_happens_when_person_has_no_bound_identity(self):
        """
        反向护栏：底库命中但名下还没有身份时，仍要走"注册 + 自动命名"闭环，
        否则路径 B 就只在"已绑过"的情况下有效，新人员永远进不了档。
        """
        pipeline = self._pipeline()
        pipeline.personnel.bound_gid.return_value = None
        pipeline._validator.get_confirmed_match.return_value = None
        resolution = Mock()
        resolution.global_id = "gid-new"
        resolution.is_new = True
        pipeline.store.register_if_new.return_value = resolution

        self._drive(pipeline)

        pipeline.store.register_if_new.assert_called_once()
        pipeline.store.bind_person.assert_called_once()
        self.assertEqual("P0001", pipeline.store.bind_person.call_args[0][1])
        self.assertEqual("gid-new", pipeline._track_to_global.get(self.TID))


class StreamStateResetTests(unittest.TestCase):
    def test_reset_clears_per_track_state(self):
        pipeline = _build_pipeline()
        pipeline._prev_track_ids = {1, 2}
        pipeline._absent_streak = {2: 1}
        pipeline._track_to_global = {1: "gid-a"}
        pipeline._reid_frame_counter = {1: 10}
        pipeline._last_tracks = [{"track_id": 1}]
        old_validator = pipeline._validator

        pipeline._reset_stream_state()

        self.assertEqual(set(), pipeline._prev_track_ids)
        self.assertEqual({}, pipeline._absent_streak)
        self.assertEqual({}, pipeline._track_to_global)
        self.assertEqual({}, pipeline._reid_frame_counter)
        self.assertEqual([], pipeline._last_tracks)
        self.assertIsNot(old_validator, pipeline._validator)

    def test_reset_does_not_touch_global_track_id_counter(self):
        """BaseTrack._count 是类级共享的：重建 tracker 会把所有摄像头的 id 一起归零"""
        pipeline = _build_pipeline()
        tracker_before = pipeline._tracker
        BaseTrack._count = 42

        pipeline._reset_stream_state()

        self.assertEqual(42, BaseTrack._count)
        self.assertIs(tracker_before, pipeline._tracker)

    def test_read_loop_returns_decoded_frame_count(self):
        pipeline = _build_pipeline()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        cap = Mock()
        cap.get.return_value = 25.0
        cap.read.side_effect = [(True, frame), (True, frame), (False, None)]

        with patch.object(pipeline, "_process_frame") as process:
            decoded = pipeline._read_loop(cap)

        self.assertEqual(2, decoded)
        self.assertEqual(2, process.call_count)

    def test_read_loop_returns_zero_for_broken_source(self):
        pipeline = _build_pipeline()
        cap = Mock()
        cap.get.return_value = 90000.0
        cap.read.return_value = (False, None)

        with patch.object(pipeline, "_process_frame"):
            self.assertEqual(0, pipeline._read_loop(cap))


class LeaveGraceTests(unittest.TestCase):
    """F2b：BYTETracker 一次关联失败就把 track 从输出里摘掉，不等于人走了"""

    def _pipeline(self) -> CameraPipeline:
        pipeline = _build_pipeline()
        # 让 _process_frame 走「非检测帧」分支：不去调 Mock 的 detector/tracker，
        # 直接消费我们塞进 _last_tracks 的轨迹（_frame_idx 不是 _detect_every_n 的倍数）
        pipeline._detect_every_n = 1000
        pipeline._frame_idx = 1
        return pipeline

    @staticmethod
    def _frame() -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    @staticmethod
    def _track(track_id: int) -> dict:
        return {"track_id": track_id, "bbox": [1.0, 1.0, 20.0, 50.0], "conf": 0.9}

    def _seen_once(self, pipeline: CameraPipeline, track_id: int) -> None:
        """先让该 track 正常在场一帧（并跳过 ReID 分支，本用例不关心特征提取）"""
        pipeline._reid_frame_counter[track_id] = pipeline._frame_idx
        pipeline._last_tracks = [self._track(track_id)]
        pipeline._process_frame(self._frame())

    @patch("src.pipeline._LEAVE_GRACE_FRAMES", GRACE)
    def test_single_frame_gap_does_not_report_leave(self):
        pipeline = self._pipeline()
        pipeline._on_person_leave = Mock()
        self._seen_once(pipeline, 7)
        pipeline._track_to_global[7] = "gid-7"

        pipeline._last_tracks = []
        pipeline._process_frame(self._frame())          # 抖动：缺席 1 帧

        self.assertEqual({7: 1}, pipeline._absent_streak)
        self.assertIn(7, pipeline._prev_track_ids)      # 宽限期内仍算在场

        pipeline._last_tracks = [self._track(7)]
        pipeline._process_frame(self._frame())          # 又回来了

        pipeline._on_person_leave.assert_not_called()
        self.assertEqual({}, pipeline._absent_streak)   # 缺席计数已清零
        self.assertEqual({7: "gid-7"}, pipeline._track_to_global)

    @patch("src.pipeline._LEAVE_GRACE_FRAMES", GRACE)
    def test_leave_reported_after_grace_frames(self):
        pipeline = self._pipeline()
        pipeline._on_person_leave = Mock()
        self._seen_once(pipeline, 7)

        pipeline._last_tracks = []
        for _ in range(GRACE - 1):
            pipeline._process_frame(self._frame())
            pipeline._on_person_leave.assert_not_called()

        pipeline._process_frame(self._frame())          # 第 GRACE 个缺席帧

        pipeline._on_person_leave.assert_called_once()
        self.assertEqual(7, pipeline._on_person_leave.call_args[0][0])

    @patch("src.pipeline._LEAVE_GRACE_FRAMES", GRACE)
    def test_leave_reported_only_once_then_forgotten(self):
        pipeline = self._pipeline()
        pipeline._on_person_leave = Mock()
        self._seen_once(pipeline, 7)

        pipeline._last_tracks = []
        for _ in range(GRACE + 5):
            pipeline._process_frame(self._frame())

        pipeline._on_person_leave.assert_called_once()
        self.assertEqual(set(), pipeline._prev_track_ids)
        self.assertEqual({}, pipeline._absent_streak)

    @patch("src.pipeline._LEAVE_GRACE_FRAMES", GRACE)
    def test_reid_buffer_survives_short_gap(self):
        """抖动期间不能清 ReID 缓冲，否则 8 帧缓冲永远填不满、无法注册身份"""
        pipeline = self._pipeline()
        feat = np.full(512, 1.0 / np.sqrt(512), dtype=np.float32)
        pipeline._validator.add_feature(7, feat)
        pipeline._validator.add_feature(7, feat)
        self._seen_once(pipeline, 7)

        pipeline._last_tracks = []
        for _ in range(GRACE - 1):
            pipeline._process_frame(self._frame())
        self.assertEqual(2, pipeline._validator.buffer_len(7))

        pipeline._process_frame(self._frame())          # 确认离场后才该清
        self.assertEqual(0, pipeline._validator.buffer_len(7))

    @patch("src.pipeline._LEAVE_GRACE_FRAMES", GRACE)
    def test_other_tracks_unaffected_by_one_departure(self):
        pipeline = self._pipeline()
        pipeline._on_person_leave = Mock()
        pipeline._reid_frame_counter = {7: pipeline._frame_idx, 8: pipeline._frame_idx}
        pipeline._last_tracks = [self._track(7), self._track(8)]
        pipeline._process_frame(self._frame())

        pipeline._last_tracks = [self._track(8)]        # 只有 7 消失
        for _ in range(GRACE):
            pipeline._process_frame(self._frame())

        pipeline._on_person_leave.assert_called_once()
        self.assertEqual(7, pipeline._on_person_leave.call_args[0][0])
        self.assertEqual({8}, pipeline._prev_track_ids)


class RoiSanitizeTests(unittest.TestCase):
    def test_string_coordinates_become_floats(self):
        pipeline = _build_pipeline()

        cleaned = pipeline._sanitize_rois(
            [{"name": "区域", "polygon": [["0.1", "0.2"], ["0.3", "0.4"], ["0.5", "0.6"]]}]
        )

        self.assertEqual(1, len(cleaned))
        for x, y in cleaned[0]["polygon"]:
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    def test_out_of_range_coordinates_are_clamped(self):
        pipeline = _build_pipeline()

        cleaned = pipeline._sanitize_rois(
            [{"name": "越界", "polygon": [[-1, 0.2], [5, 0.4], [0.5, 9]]}]
        )

        self.assertEqual([[0.0, 0.2], [1.0, 0.4], [0.5, 1.0]], cleaned[0]["polygon"])

    def test_degenerate_and_malformed_rois_are_dropped(self):
        pipeline = _build_pipeline()

        cleaned = pipeline._sanitize_rois(
            [
                {"name": "两点", "polygon": [[0.1, 0.1], [0.2, 0.2]]},
                {"name": "缺字段"},
                {"name": "坐标非法", "polygon": [["a", "b"], [0.2, 0.2], [0.3, 0.3]]},
                "不是对象",
                {"name": "合法", "polygon": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]},
            ]
        )

        self.assertEqual(["合法"], [roi["name"] for roi in cleaned])

    def test_non_list_payload_is_ignored(self):
        pipeline = _build_pipeline()
        self.assertEqual([], pipeline._sanitize_rois({"polygon": []}))


if __name__ == "__main__":
    unittest.main()
