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


def _build_pipeline(source: str = "videos/none.mp4") -> CameraPipeline:
    return CameraPipeline(
        camera_id="cam_test",
        source=source,
        detector=Mock(),
        reid_extractor=Mock(),
        identity_store=Mock(),
        topology=Mock(),
        alert_manager=Mock(),
        screenshot_dir=Path("outputs"),
        frame_hub=None,
    )


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
