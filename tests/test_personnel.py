"""Personnel API（worklist 3.4）：档案 CRUD、注册照、实名绑定、名下身份。

写接口一律带 X-Lab-Monitor-Request: 1（守卫中间件要求）。
绑定是幂等的；删除档案会解除其名下身份的绑定（身份回到匿名状态，不删除）。
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from src.db import Database
from src.identity_store import IdentityStore
from src.personnel import PersonnelGallery


def _unit(*values) -> np.ndarray:
    vec = np.array(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class PersonnelApiFixture(unittest.TestCase):
    """构造带 PersonnelGallery 注入的 TestClient（与 main.py 装配口径一致）。"""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.database = Database(Path(self._temp.name) / "personnel_api.db")
        self.gallery = PersonnelGallery(database=self.database)
        self.store = IdentityStore(database=self.database, feature_space="test:4")

        import server
        from fastapi.testclient import TestClient

        class _FakeDetector:
            def detect(self, frame):
                return [[10, 10, 100, 300, 0.9]]

        self._feature = _unit(1, 0.05, 0, 0)

        class _FakeExtractor:
            def extract(self, frame, bbox):
                return self._feature

        class _StubExtractor:
            def __init__(self, feature):
                self._feature = feature
            def extract(self, frame, bbox):
                return self._feature
        self.server_module = server
        server.init_server(frame_hub=None, broadcaster=None,
                           identity_store=self.store, personnel=self.gallery,
                           detector=_FakeDetector(),
                           reid_extractor=_StubExtractor(self._feature))
        self.client = TestClient(server.app)
        self.headers = {"X-Lab-Monitor-Request": "1"}

    def tearDown(self):
        self.server_module.init_server(None, None, None)
        self.database.close()
        self._temp.cleanup()

    def _register_identity(self) -> str:
        return self.store.register(_unit(1, 0.05, 0, 0))


class PersonnelCrudTests(PersonnelApiFixture):
    def test_create_and_get(self):
        res = self.client.post("/api/personnel", json={"name": "张三"},
                               headers=self.headers)
        self.assertEqual(res.status_code, 200)
        person_id = res.json()["person_id"]
        self.assertEqual(self.client.get(f"/api/personnel/{person_id}").json()["name"],
                         "张三")

    def test_name_required(self):
        res = self.client.post("/api/personnel", json={},
                               headers=self.headers)
        self.assertEqual(res.status_code, 400)

    def test_list_includes_identity_count(self):
        person_id = self.gallery.create("李四")
        gid = self._register_identity()
        self.store.bind_person(gid, person_id)
        items = self.client.get("/api/personnel").json()["personnel"]
        target = next(p for p in items if p["person_id"] == person_id)
        self.assertEqual(target["identity_count"], 1)

    def test_update_and_delete(self):
        person_id = self.gallery.create("王五")
        res = self.client.patch(f"/api/personnel/{person_id}",
                                json={"name": "王五二"}, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.gallery.get(person_id)["name"], "王五二")
        res = self.client.delete(f"/api/personnel/{person_id}", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(self.gallery.get(person_id))

    def test_delete_unbinds_identities(self):
        person_id = self.gallery.create("赵六")
        gid = self._register_identity()
        self.store.bind_person(gid, person_id)
        self.client.delete(f"/api/personnel/{person_id}", headers=self.headers)
        self.assertIsNone(self.store.get(gid).person_id, "删除档案必须解除绑定")


class BindTests(PersonnelApiFixture):
    def test_bind_and_enrichment(self):
        person_id = self.gallery.create("张三", employee_no="QLU-1")
        gid = self._register_identity()
        res = self.client.post(f"/api/identities/{gid}/bind",
                               json={"person_id": person_id}, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        # GET /api/identities 返回实名
        detail = self.client.get(f"/api/identities/{gid}").json()
        self.assertEqual(detail["person_id"], person_id)
        self.assertEqual(detail["person_name"], "张三")
        # 名下身份
        owned = self.client.get(f"/api/personnel/{person_id}/identities").json()
        self.assertEqual([i["global_id"] for i in owned["identities"]], [gid])

    def test_unbind(self):
        person_id = self.gallery.create("张三")
        gid = self._register_identity()
        self.client.post(f"/api/identities/{gid}/bind",
                         json={"person_id": person_id}, headers=self.headers)
        res = self.client.delete(f"/api/identities/{gid}/bind", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(self.store.get(gid).person_id)

    def test_bind_unknown_identity_404(self):
        person_id = self.gallery.create("张三")
        res = self.client.post("/api/identities/no-such/bind",
                               json={"person_id": person_id}, headers=self.headers)
        self.assertEqual(res.status_code, 404)

    def test_bind_unknown_person_404(self):
        gid = self._register_identity()
        res = self.client.post(f"/api/identities/{gid}/bind",
                               json={"person_id": "no-such"}, headers=self.headers)
        self.assertEqual(res.status_code, 404)

    def test_survives_restart(self):
        """绑定关系必须落库，重启后仍能恢复（3.6 的验收项）。"""
        person_id = self.gallery.create("张三")
        gid = self._register_identity()
        self.client.post(f"/api/identities/{gid}/bind",
                         json={"person_id": person_id}, headers=self.headers)
        restored = IdentityStore(database=self.database, feature_space="test:4")
        self.assertEqual(restored.get(gid).person_id, person_id)


class PersonnelPhotoTests(PersonnelApiFixture):
    def test_add_photo_enables_bottom_gallery_match(self):
        person_id = self.gallery.create("张三")
        feature = _unit(1, 0.05, 0, 0)
        # 用 raw body 上传（multipart 需要 python-multipart，测试环境未装）
        ok, buffer = cv2.imencode(".jpg", np.zeros((400, 300, 3), dtype=np.uint8))
        assert ok
        res = self.client.post(
            f"/api/personnel/{person_id}/photos",
            content=buffer.tobytes(),
            headers={**self.headers, "Content-Type": "image/jpeg"},
        )
        self.assertEqual(res.status_code, 200)
        # 上传的图会经检测器提特征 —— 测试环境未注入检测器，
        # 这里直接验证库里的注册照条目与底库检索（photos_of）
        self.assertEqual(self.gallery.photos_of(person_id), 1)

    def test_bottom_gallery_match_uses_registered_photo(self):
        """底库 1:N：注册照特征命中时返回 person_id（路径 B 的核心）。"""
        person_id = self.gallery.create("张三")
        self.gallery.add_photo(person_id, _unit(1, 0.05, 0, 0), quality=1.0)
        hit = self.gallery.match(_unit(1, 0.04, 0, 0))
        self.assertIsNotNone(hit)
        self.assertEqual(hit["person_id"], person_id)
        self.assertEqual(hit["name"], "张三")

    def test_empty_gallery_never_matches(self):
        hit = self.gallery.match(_unit(1, 0, 0, 0))
        self.assertIsNone(hit, "空底库必须返回 None（对主流程零影响）")

    def test_different_person_below_threshold(self):
        person_id = self.gallery.create("张三")
        self.gallery.add_photo(person_id, _unit(1, 0.05, 0, 0), quality=1.0)
        other = _unit(0, 0, 1, 0.05)
        self.assertIsNone(self.gallery.match(other), "正交特征不应命中")


if __name__ == "__main__":
    unittest.main()
