"""
test_alerts_api.py — 告警接口的 HTTP 层冒烟测试。

为什么专门建这个文件
--------------------
`GET /api/alerts/history` 此前**零测试覆盖**，因此一个 `NameError` 长期潜伏：

    summary = await run_in_threadpool(db.get_alert_summary)   # db 未定义

`db` 只在另一个函数里以 `from src.db import db` 惰性导入过，模块级并不存在该名字。
结果是 total/alerts 已经查出来了，却在下一步整体抛错被吞成 500 —— 告警历史面板
拿不到 summary，而日志里只有一行 "查询历史告警失败"。2026-09-15 启动服务时实测复现。

本文件的作用就是把这条路径钉住：**接口必须 200 且 summary 字段齐全**。
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from src.db import Database
from src.identity_store import IdentityStore

ALERTS = [
    {"alert_id": "a1", "timestamp": 1000.0, "last_camera": "rnd_04",
     "global_id": "g1111111", "alert_type": "MISSING_PERSON", "stage": "ALERT",
     "risk_level": "HIGH", "elapsed_seconds": 40.0, "expected_cameras": ["rnd_06"]},
    {"alert_id": "a2", "timestamp": 2000.0, "last_camera": "rnd_06",
     "global_id": "g2222222", "alert_type": "MISSING_PERSON", "stage": "WARNING",
     "risk_level": "LOW", "elapsed_seconds": 12.0, "expected_cameras": []},
    {"alert_id": "a3", "timestamp": 3000.0, "last_camera": "rnd_16",
     "global_id": "g3333333", "alert_type": "INTRUSION", "stage": "ALERT",
     "risk_level": "MEDIUM", "elapsed_seconds": 0.0, "expected_cameras": []},
]


@contextmanager
def _client():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "alerts.db")
        for alert in ALERTS:
            db.insert_alert(alert)

        import server
        from fastapi.testclient import TestClient

        # 库通过 IdentityStore 注入：_get_database() 的取法是 store → gallery →
        # 全局单例，而 init_server() 没有 database 参数（直接传会 TypeError）。
        store = IdentityStore(database=db, feature_space="test:4")
        server.init_server(frame_hub=None, broadcaster=None, identity_store=store)
        client = TestClient(server.app)
        try:
            yield client
        finally:
            # 必须先还原全局态并关库，再让 TemporaryDirectory 清理 ——
            # 否则 Windows 上库文件仍被占用，清理阶段抛 WinError 32。
            server.init_server(None, None, None)
            db.close()


class AlertHistoryApiTests(unittest.TestCase):
    def test_returns_alerts_and_summary(self):
        """summary 必须真的返回 —— 这正是 NameError 曾经吞掉的那一步。"""
        with _client() as client:
            resp = client.get("/api/alerts/history?limit=10")
            self.assertEqual(resp.status_code, 200,
                             f"接口不应 500: {resp.text[:300]}")
            body = resp.json()
            self.assertEqual(body["total"], 3)
            self.assertEqual(len(body["alerts"]), 3)
            self.assertIn("summary", body)
            self.assertIsInstance(body["summary"], dict)
            self.assertTrue(body["summary"], "summary 不应为空")

    def test_pagination_and_filters(self):
        with _client() as client:
            page = client.get("/api/alerts/history?limit=2&offset=0").json()
            self.assertEqual(len(page["alerts"]), 2)
            self.assertEqual(page["total"], 3)

            by_cam = client.get("/api/alerts/history?camera_id=rnd_04").json()
            self.assertEqual(by_cam["total"], 1)

            by_gid = client.get("/api/alerts/history?global_id=g2222222").json()
            self.assertEqual(by_gid["total"], 1)

            by_risk = client.get("/api/alerts/history?risk_level=LOW").json()
            self.assertEqual(by_risk["total"], 1)

    def test_limit_upper_bound_is_enforced(self):
        """limit 上限 500：不设上限会把整张告警表读进内存。"""
        with _client() as client:
            self.assertEqual(client.get("/api/alerts/history?limit=501").status_code, 422)
            self.assertEqual(client.get("/api/alerts/history?limit=0").status_code, 422)


if __name__ == "__main__":
    unittest.main()
