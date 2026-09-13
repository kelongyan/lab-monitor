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


class SearchPersonByIdTests(unittest.TestCase):
    """
    按实名档案检索（P1-1a）：/api/search/person?person_id=

    用独立的 app 夹具，因为要多一个 personnel 库（身份要绑定到档案上）。
    """

    @contextmanager
    def person_app(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "pid.db")
            database.seed_video_asset(
                camera_id="rnd_08", rel_path="videos_low/rnd_08.mp4",
                codec="h264", fps_declared=25.0, frames_real=957,
                duration_real=38.28)
            asset_id = database.get_video_asset("rnd_08")["asset_id"]
            store = IdentityStore(database=database, feature_space="test:4")

            gid_a = store.register(np.array([1, 0, 0, 0], dtype=np.float32))
            gid_b = store.register(np.array([0, 1, 0, 0], dtype=np.float32))
            database.record_appearance(gid_a, "rnd_08", 1000.0, [0, 0, 50, 100],
                                       1, asset_id=asset_id, video_frame=100,
                                       video_ts=4.0)
            database.record_appearance(gid_b, "rnd_08", 1100.0, [0, 0, 50, 100],
                                       2, asset_id=asset_id, video_frame=200,
                                       video_ts=8.0)

            from src.personnel import PersonnelGallery
            gallery = PersonnelGallery(database=database, threshold=0.68)
            person_id = gallery.create(name="张伟", employee_no="QLU-26-0001",
                                       department="网络运维")
            # 两个匿名身份都归到同一个人名下
            for gid in (gid_a, gid_b):
                database.save_identity(
                    global_id=gid, feature_dim=4, feature_blob=b"\x00" * 16,
                    feature_bank_count=0, feature_bank_blob=b"",
                    total_appearances=0, last_camera="rnd_08", last_seen=1000.0,
                    feature_space="test:4", first_seen=1000.0, person_id=person_id,
                )

            import server
            from fastapi.testclient import TestClient
            server.init_server(frame_hub=None, broadcaster=None,
                               identity_store=store, personnel=gallery)
            try:
                yield TestClient(server.app), person_id, gid_a, gid_b, database
            finally:
                server.init_server(None, None, None)
                database.close()

    def test_person_id_merges_all_identities(self):
        """核心语义：按档案检索要合并名下**全部**身份，不能只给第一个。"""
        with self.person_app() as (client, person_id, _a, _b, _db):
            res = client.get(f"/api/search/person?person_id={person_id}")
            self.assertEqual(res.status_code, 200, res.text[:200])
            payload = res.json()
            self.assertEqual(payload["scope"], "person")
            self.assertEqual(payload["person_name"], "张伟")
            self.assertEqual(len(payload["global_ids"]), 2,
                             "名下两个身份都要纳入")
            self.assertEqual(payload["asset_count"], 1)
            # 一个资产但来自两个身份
            entry = payload["assets"][0]
            self.assertEqual(len(entry["global_ids"]), 2)
            self.assertAlmostEqual(entry["video_first_ts"], 4.0, places=2)
            self.assertAlmostEqual(entry["video_last_ts"], 8.0, places=2)

    def test_known_camera_lookup_runs_off_the_event_loop(self):
        """
        `_known_camera_set()` 会 list_video_assets()（查库），必须留在 run_in_threadpool
        的闭包里。曾经为了复用参数把它提到协程顶层（`common = dict(..., known_cameras=...)`），
        于是每次检索都在事件循环上同步查库 —— 功能测试全绿，只在并发下拖慢所有接口。

        判据是"**调用时该线程有没有在跑事件循环**"，不能用线程名：
        TestClient 自己就在非主线程里跑 loop，按线程名断言会恒真（等于没测）。
        run_in_threadpool 的 worker 线程里没有 running loop，协程里必然有。
        """
        import asyncio
        import server

        on_loop = []
        original = server._known_camera_set

        def spy(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
            return original(*args, **kwargs)

        with self.person_app() as (client, person_id, gid_a, _b, _db):
            server._known_camera_set = spy
            try:
                for url in (f"/api/search/person?person_id={person_id}",
                            f"/api/search/person?global_id={gid_a}"):
                    res = client.get(url)
                    self.assertEqual(res.status_code, 200, res.text[:200])
            finally:
                server._known_camera_set = original

        self.assertTrue(on_loop, "_known_camera_set 未被调用，用例失去意义")
        self.assertNotIn(
            True, on_loop,
            f"_known_camera_set 在事件循环线程上被调用（共 {len(on_loop)} 次，"
            "其中若干次发生在 loop 上）—— 它查库，必须放回 run_in_threadpool 闭包内")


    def test_person_id_and_global_id_are_mutually_exclusive(self):
        """两个入口同时给 → 400，避免"到底按谁查"的歧义。"""
        with self.person_app() as (client, person_id, gid_a, _b, _db):
            res = client.get(
                f"/api/search/person?person_id={person_id}&global_id={gid_a}")
            self.assertEqual(res.status_code, 400)
            res = client.get("/api/search/person")
            self.assertEqual(res.status_code, 400, "两个都不给也应 400")

    def test_unknown_person_id_is_404(self):
        with self.person_app() as (client, _pid, _a, _b, _db):
            res = client.get("/api/search/person?person_id=no-such-person")
            self.assertEqual(res.status_code, 404)

    def test_person_without_identities_returns_empty_not_404(self):
        """
        档案刚建好、还没绑定身份 → 返回空清单。
        这不是错误：前端要显示"该档案暂无轨迹"，而不是弹 404 让人以为坏了。
        """
        with self.person_app() as (client, _pid, _a, _b, database):
            import server
            empty_pid = server._personnel.create(name="李四")
            res = client.get(f"/api/search/person?person_id={empty_pid}")
            self.assertEqual(res.status_code, 200, res.text[:200])
            payload = res.json()
            self.assertEqual(payload["asset_count"], 0)
            self.assertEqual(payload["global_ids"], [])
            self.assertEqual(payload["person_name"], "李四")


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
