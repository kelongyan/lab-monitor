import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class RoiApiTests(unittest.TestCase):
    def test_polygon_count_and_coordinate_validation(self):
        async def exercise(root: Path):
            original_pipelines = server._pipelines
            original_lock = server._roi_file_lock
            server._pipelines = [SimpleNamespace(camera_id="cam_a")]
            server._roi_file_lock = asyncio.Lock()
            path_factory = lambda _: root / "server.py"
            try:
                with patch.object(server, "Path", path_factory):
                    empty = await server.save_roi(FakeRequest({
                        "camera_id": "cam_a", "polygon": [], "name": "zone",
                    }))
                    self.assertEqual(empty.status_code, 200)

                    for point_count in (1, 2, 65):
                        response = await server.save_roi(FakeRequest({
                            "camera_id": "cam_a",
                            "polygon": [[0.5, 0.5]] * point_count,
                            "name": "zone",
                        }))
                        self.assertEqual(response.status_code, 400)

                    valid = await server.save_roi(FakeRequest({
                        "camera_id": "cam_a",
                        "polygon": [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
                        "name": "zone",
                    }))
                    self.assertEqual(valid.status_code, 200)

                    invalid_coordinate = await server.save_roi(FakeRequest({
                        "camera_id": "cam_a",
                        "polygon": [[0.0, 0.0], [1.1, 0.0], [0.5, 1.0]],
                        "name": "zone",
                    }))
                    self.assertEqual(invalid_coordinate.status_code, 400)

                    invalid_shape = await server.save_roi(FakeRequest({
                        "camera_id": "cam_a", "polygon": "not-an-array",
                    }))
                    self.assertEqual(invalid_shape.status_code, 400)
            finally:
                server._pipelines = original_pipelines
                server._roi_file_lock = original_lock

            saved = json.loads((root / "config" / "roi.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["cam_a"][0]["polygon"]), 3)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config").mkdir()
            asyncio.run(exercise(root))


if __name__ == "__main__":
    unittest.main()
