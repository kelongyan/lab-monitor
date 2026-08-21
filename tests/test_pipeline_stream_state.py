"""
test_pipeline_stream_state.py — 覆盖第1批 F2/F3/F10 在 pipeline 侧的行为：
  - 重开流时重置逐帧状态，但不得触碰 ByteTrack 的类级 track_id 计数器
  - _read_loop 返回本轮解码帧数（坏源熔断依赖它）
  - ROI 配置清洗：字符串坐标、越界坐标、顶点数不足都不能让线程崩掉
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from ultralytics.trackers.basetrack import BaseTrack

from src.pipeline import CameraPipeline


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
        pipeline._track_to_global = {1: "gid-a"}
        pipeline._reid_frame_counter = {1: 10}
        pipeline._last_tracks = [{"track_id": 1}]
        old_validator = pipeline._validator

        pipeline._reset_stream_state()

        self.assertEqual(set(), pipeline._prev_track_ids)
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
