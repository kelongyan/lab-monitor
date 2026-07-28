import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.frame_hub import FrameHub
from src.pipeline import CameraPipeline


class FrameHubTests(unittest.TestCase):
    def test_first_failure_is_visible_and_offline_has_no_stale_jpeg(self):
        hub = FrameHub(display_width=32, display_height=18)
        hub.mark_offline("Cam_A", status_text="RECONNECTING", reconnect_count=1)

        status = hub.get_status()
        self.assertEqual(1, len(status))
        self.assertEqual("Cam_A", status[0]["camera_id"])
        self.assertFalse(status[0]["is_online"])
        self.assertIsNone(status[0]["frame_age_seconds"])
        self.assertIsNone(hub.get_jpeg("Cam_A"))

        hub.push_frame("Cam_A", np.zeros((18, 32, 3), dtype=np.uint8))
        self.assertIsNotNone(hub.get_jpeg("Cam_A"))
        hub.mark_offline("Cam_A", status_text="OFFLINE")
        self.assertIsNone(hub.get_jpeg("Cam_A"))

        hub.push_frame("Cam_A", np.full((18, 32, 3), 255, dtype=np.uint8))
        self.assertIsNotNone(hub.get_jpeg("Cam_A"))
        self.assertTrue(hub.get_status()[0]["is_online"])

    def test_resize_checks_both_width_and_height(self):
        hub = FrameHub(display_width=32, display_height=18)
        hub.push_frame("cam", np.zeros((20, 32, 3), dtype=np.uint8))

        with patch("src.frame_hub.cv2.resize", wraps=cv2.resize) as resize:
            jpeg = hub.get_jpeg("cam")

        self.assertIsNotNone(jpeg)
        resize.assert_called_once()
        self.assertEqual((32, 18), resize.call_args.args[1])

    def test_new_generation_wins_when_encoding_overlaps_push(self):
        class Encoded:
            def __init__(self, value):
                self.value = value

            def tobytes(self):
                return self.value

        hub = FrameHub(display_width=2, display_height=2)
        hub.push_frame("cam", np.zeros((2, 2, 3), dtype=np.uint8))
        encoding_started = threading.Event()
        release_first = threading.Event()
        calls = 0

        def encode(_extension, frame, _options):
            nonlocal calls
            calls += 1
            if calls == 1:
                encoding_started.set()
                release_first.wait(timeout=2)
            value = b"new" if int(frame[0, 0, 0]) == 255 else b"old"
            return True, Encoded(value)

        result = []
        with patch("src.frame_hub.cv2.imencode", side_effect=encode):
            thread = threading.Thread(target=lambda: result.append(hub.get_jpeg("cam")))
            thread.start()
            self.assertTrue(encoding_started.wait(timeout=2))
            hub.push_frame("cam", np.full((2, 2, 3), 255, dtype=np.uint8))
            release_first.set()
            thread.join(timeout=2)

            self.assertEqual([b"new"], result)
            self.assertEqual(b"new", hub.get_jpeg("cam"))


class PipelineLifecycleTests(unittest.TestCase):
    def test_file_processing_exception_releases_capture_and_marks_error(self):
        capture = Mock()
        capture.isOpened.return_value = True
        hub = FrameHub()
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline.camera_id = "cam_a"
        pipeline.source = "broken.mp4"
        pipeline.frame_hub = hub
        pipeline._stop_event = threading.Event()
        pipeline._read_loop = Mock(side_effect=RuntimeError("injected failure"))

        with patch("src.pipeline.cv2.VideoCapture", return_value=capture):
            pipeline._run_file()

        capture.release.assert_called_once()
        status = hub.get_status()[0]
        self.assertFalse(status["is_online"])
        self.assertEqual("PIPELINE_ERROR", status["status_text"])

    def test_rtsp_read_exception_reconnects_and_releases_each_capture(self):
        first_capture = Mock()
        first_capture.isOpened.return_value = True
        second_capture = Mock()
        second_capture.isOpened.return_value = True
        hub = FrameHub()
        pipeline = CameraPipeline.__new__(CameraPipeline)
        pipeline.camera_id = "cam_a"
        pipeline.source = "rtsp://example.invalid/stream"
        pipeline.frame_hub = hub
        pipeline._stop_event = threading.Event()
        pipeline._reconnect_count = 0

        def read_loop(_capture):
            if pipeline._read_loop.call_count == 1:
                raise RuntimeError("injected read failure")
            pipeline._stop_event.set()

        pipeline._read_loop = Mock(side_effect=read_loop)
        with patch(
            "src.pipeline.cv2.VideoCapture",
            side_effect=[first_capture, second_capture],
        ), patch.object(pipeline._stop_event, "wait", return_value=False):
            pipeline._run_rtsp()

        self.assertEqual(2, pipeline._read_loop.call_count)
        first_capture.release.assert_called_once()
        second_capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
