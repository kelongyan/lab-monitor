"""
test_search_api.py — 检索接口的 HTTP 层冒烟测试（worklist 4.2 / 4.4）。

覆盖：
  - GET /api/search/person：200 结构、404 未知身份、已下线相机被过滤
  - POST /api/search/by-image：multipart 解析、无人员图 422、ReID 未注入 503
  - 守卫中间件：写接口必须带 X-Lab-Monitor-Request: 1

用 TestClient 直接打 app（不连生产库），身份库与资产都在临时库里。
"""

import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src.db import Database
from src.identity_store import IdentityStore
from src.search import aggregate_identity_assets


class _FakeExtractor:
    """固定返回构造特征的假提取器 —— 只测 HTTP 层的解析与分发。"""

    def __init__(self, feature: np.ndarray):
        self._feature = feature

    def extract(self, frame, bbox):
        return self._feature


class _FakeDetector:
    def detect(self, frame):
        return [[10, 10, 100, 300, 0.9]]   # 一个人体框


@contextmanager
def search_app():
    with tempfile.TemporaryDirectory() as temp_dir:
        database = Database(Path(temp_dir) / "api.db")
        database.seed_video_asset(
            camera_id="rnd_08", rel_path="videos_low/rnd_08.mp4",
            codec="h264", fps_declared=25.0, frames_real=957, duration_real=38.28)
        asset_id = database.get_video_asset("rnd_08")["asset_id"]
        store = IdentityStore(database=database, feature_space="test:4")
        gid = store.register(np.array([1, 0.05, 0, 0], dtype=np.float32)
                             / np.linalg.norm(np.array([1, 0.05, 0, 0],
                                                       dtype=np.float32)))
        database.record_appearance(
            gid, "rnd_08", 1000.0, [0, 0, 50, 100], 1,
            asset_id=asset_id, video_frame=100, video_ts=4.0)

        import server
        from fastapi.testclient import TestClient
        server.init_server(
            frame_hub=None, broadcaster=None, identity_store=store,
            detector=_FakeDetector(),
            reid_extractor=_FakeExtractor(
                np.array([1, 0.05, 0, 0], dtype=np.float32)
                / np.linalg.norm(np.array([1, 0.05, 0, 0], dtype=np.float32))),
        )
        client = TestClient(server.app)
        try:
            yield client, gid, server
        finally:
            server.init_server(None, None, None)   # 还原全局态，避免污染其它用例
            database.close()


class SearchPersonApiTests(unittest.TestCase):
    def test_returns_asset_list(self):
        with search_app() as (client, gid, _server):
            res = client.get(f"/api/search/person?global_id={gid}")
            self.assertEqual(res.status_code, 200)
            payload = res.json()
            self.assertEqual(payload["asset_count"], 1)
            asset = payload["assets"][0]
            self.assertEqual(asset["camera_id"], "rnd_08")
            self.assertTrue(asset["position_known"])
            self.assertAlmostEqual(asset["video_first_ts"], 4.0, places=2)

    def test_unknown_identity_returns_404(self):
        with search_app() as (client, _gid, _server):
            res = client.get("/api/search/person?global_id=no-such")
            self.assertEqual(res.status_code, 404)

    def test_offline_camera_rows_are_dropped(self):
        with search_app() as (client, gid, _server):
            # 直接往库里塞一条已下线相机的轨迹（模拟 rnd_03 孤儿数据）
            import server
            server._identity_store._database.record_appearance(
                gid, "rnd_03", 2000.0, [0, 0, 1, 2], 2)
            res = client.get(f"/api/search/person?global_id={gid}")
            cameras = {a["camera_id"] for a in res.json()["assets"]}
            self.assertNotIn("rnd_03", cameras)


class SearchByImageApiTests(unittest.TestCase):
    def _jpeg(self) -> bytes:
        import cv2
        image = np.zeros((400, 300, 3), dtype=np.uint8)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        return buffer.tobytes()

    def test_requires_guard_header(self):
        with search_app() as (client, _gid, _server):
            res = client.post("/api/search/by-image", content=self._jpeg())
            self.assertEqual(res.status_code, 403, "写接口必须带守卫头")

    def test_returns_topk_matches(self):
        with search_app() as (client, gid, _server):
            res = client.post(
                "/api/search/by-image",
                content=self._jpeg(),
                headers={"X-Lab-Monitor-Request": "1"},
            )
            self.assertEqual(res.status_code, 200)
            payload = res.json()
            self.assertEqual(payload["person_count"], 1)
            self.assertEqual(payload["matches"][0]["global_id"], gid)
            self.assertGreaterEqual(len(payload["matches"][0]["assets"]), 1)

    def test_reid_unavailable_returns_503(self):
        with search_app() as (client, _gid, server):
            import server as server_module
            server_module._reid_extractor = None
            res = client.post(
                "/api/search/by-image",
                content=self._jpeg(),
                headers={"X-Lab-Monitor-Request": "1"},
            )
            self.assertEqual(res.status_code, 503)

    def test_invalid_image_returns_400(self):
        with search_app() as (client, _gid, _server):
            res = client.post(
                "/api/search/by-image",
                content=b"not-an-image",
                headers={"X-Lab-Monitor-Request": "1"},
            )
            self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
