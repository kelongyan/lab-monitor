import base64
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from starlette.requests import Request

import server
from src.topology import CameraTopology, TopologyValidationError


CAMERAS = {"cam_01", "cam_02", "cam_03"}
INITIAL = {
    "cam_01": [
        {
            "next": "cam_02",
            "expected_seconds": 30,
            "tolerance_seconds": 10,
        }
    ]
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_request(payload: object) -> Request:
    body = json.dumps(payload).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/topology",
            "raw_path": b"/api/topology",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        },
        receive,
    )


class CameraTopologyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "topology.json"
        self.config_path.write_text(json.dumps(INITIAL), encoding="utf-8")
        self.topology = CameraTopology(self.config_path, CAMERAS)

    def tearDown(self):
        self.temp_dir.cleanup()

    def assert_rejected_without_mutation(self, payload: object):
        before_hash = file_hash(self.config_path)
        before_graph = self.topology.to_dict()
        with self.assertRaises(TopologyValidationError):
            self.topology.update_config(payload)
        self.assertEqual(before_hash, file_hash(self.config_path))
        self.assertEqual(before_graph, self.topology.to_dict())

    def test_invalid_payloads_do_not_change_disk_or_runtime(self):
        invalid_payloads = [
            [],
            {"cam_01": [{}]},
            {"cam_01": [{"next": "cam_99", "expected_seconds": 5}]},
            {"cam_01": [{"next": "cam_01", "expected_seconds": 5}]},
            {"cam_01<script>": []},
            {"cam_01": [{"next": "cam_02", "expected_seconds": 0}]},
            {"cam_01": [{"next": "cam_02", "expected_seconds": "30"}]},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.assert_rejected_without_mutation(payload)

    def test_valid_payload_is_normalized_persisted_and_loaded(self):
        payload = {
            "cam_02": [
                {
                    "next": "cam_03",
                    "expected_seconds": 12,
                    "tolerance_seconds": 0,
                }
            ]
        }
        normalized = self.topology.update_config(payload)

        self.assertEqual(normalized, self.topology.to_dict())
        self.assertEqual(normalized, json.loads(self.config_path.read_text("utf-8")))
        reloaded = CameraTopology(self.config_path, CAMERAS)
        self.assertEqual(normalized, reloaded.to_dict())

    def test_concurrent_updates_keep_disk_and_runtime_consistent(self):
        barrier = threading.Barrier(8)

        def update(index: int):
            barrier.wait()
            self.topology.update_config({
                "cam_01": [{
                    "next": "cam_02" if index % 2 else "cam_03",
                    "expected_seconds": index + 1,
                    "tolerance_seconds": index,
                }]
            })

        threads = [threading.Thread(target=update, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(
            self.topology.to_dict(),
            json.loads(self.config_path.read_text("utf-8")),
        )
        self.assertEqual([], list(self.config_path.parent.glob("*.tmp")))


class TopologyApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "topology.json"
        self.config_path.write_text(json.dumps(INITIAL), encoding="utf-8")
        self.previous_topology = server._topology
        self.previous_frame_hub = server._frame_hub
        server._topology = CameraTopology(self.config_path, CAMERAS)

    async def asyncTearDown(self):
        server._topology = self.previous_topology
        server._frame_hub = self.previous_frame_hub
        self.temp_dir.cleanup()

    async def test_api_rejects_xss_without_mutating_topology(self):
        before_hash = file_hash(self.config_path)
        before_graph = server._topology.to_dict()
        response = await server.save_topology(make_request({
            "cam_01</text><script>alert(1)</script>": []
        }))

        self.assertEqual(400, response.status_code)
        self.assertEqual(before_hash, file_hash(self.config_path))
        self.assertEqual(before_graph, server._topology.to_dict())

    async def test_api_persists_valid_topology_once(self):
        payload = {
            "cam_02": [{
                "next": "cam_03",
                "expected_seconds": 18,
                "tolerance_seconds": 4,
            }]
        }
        response = await server.save_topology(make_request(payload))

        self.assertEqual(200, response.status_code)
        persisted = json.loads(self.config_path.read_text("utf-8"))
        self.assertEqual(server._topology.to_dict(), persisted)

    async def test_health_schema_is_stable_for_camera_states(self):
        class FrameHubStub:
            def get_status(self):
                return [
                    {"camera_id": "cam_01", "is_online": True},
                    {"camera_id": "cam_02", "is_online": False},
                ]

        for frame_hub, expected_online in ((None, 0), (FrameHubStub(), 1)):
            with self.subTest(expected_online=expected_online):
                server._frame_hub = frame_hub
                response = await server.health_check()
                payload = json.loads(response.body)
                self.assertEqual(200, response.status_code)
                self.assertEqual("ok", payload["status"])
                self.assertEqual(expected_online, payload["cameras_online"])
                self.assertIsInstance(payload["timestamp"], float)


class ServerAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.old_username = os.environ.pop("LAB_MONITOR_USERNAME", None)
        self.old_password = os.environ.pop("LAB_MONITOR_PASSWORD", None)

    def tearDown(self):
        if self.old_username is not None:
            os.environ["LAB_MONITOR_USERNAME"] = self.old_username
        else:
            os.environ.pop("LAB_MONITOR_USERNAME", None)
        if self.old_password is not None:
            os.environ["LAB_MONITOR_PASSWORD"] = self.old_password
        else:
            os.environ.pop("LAB_MONITOR_PASSWORD", None)

    def test_auth_is_optional_for_default_loopback_mode(self):
        self.assertTrue(server._is_loopback_host("127.0.0.1"))
        self.assertTrue(server._is_loopback_host("::1"))
        self.assertFalse(server._is_loopback_host("0.0.0.0"))
        self.assertTrue(server._authorization_valid(None))

    def test_configured_credentials_use_constant_time_comparison(self):
        os.environ["LAB_MONITOR_USERNAME"] = "operator"
        os.environ["LAB_MONITOR_PASSWORD"] = "secret-value"
        token = base64.b64encode(b"operator:secret-value").decode("ascii")

        self.assertTrue(server._authorization_valid(f"Basic {token}"))
        self.assertFalse(server._authorization_valid("Basic invalid"))
        self.assertFalse(server._authorization_valid(None))

    def test_partial_credentials_are_rejected(self):
        os.environ["LAB_MONITOR_USERNAME"] = "operator"
        with self.assertRaises(RuntimeError):
            server._configured_basic_auth()

    def test_remote_bind_requires_credentials(self):
        with self.assertRaises(RuntimeError):
            server._ensure_secure_bind("0.0.0.0")

        os.environ["LAB_MONITOR_USERNAME"] = "operator"
        os.environ["LAB_MONITOR_PASSWORD"] = "secret-value"
        server._ensure_secure_bind("0.0.0.0")


if __name__ == "__main__":
    unittest.main()
