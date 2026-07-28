import asyncio
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server
from src.alerter import AlertBroadcaster, AlertManager
from src.db import Database
from src.pipeline import redact_source


def alert(alert_id: str) -> dict:
    return {
        "alert_id": alert_id,
        "timestamp": time.time(),
        "stage": "ALERT",
        "alert_type": "INTRUSION",
        "risk_level": "HIGH",
        "global_id": alert_id,
        "last_camera": "cam_a",
    }


class BroadcasterLifecycleTests(unittest.TestCase):
    def test_pre_start_messages_move_to_async_queue_once(self):
        async def exercise():
            broadcaster = AlertBroadcaster(queue_size=4)
            broadcaster.push(alert("early"))
            self.assertEqual(broadcaster.queue.qsize(), 1)
            broadcaster.set_event_loop(asyncio.get_running_loop())
            self.assertEqual(broadcaster.queue.qsize(), 0)
            migrated = await asyncio.wait_for(
                broadcaster._async_queue.get(), timeout=0.1
            )
            self.assertEqual(migrated["alert_id"], "early")

        asyncio.run(exercise())

    def test_no_web_mode_does_not_grow_delivery_queue(self):
        broadcaster = AlertBroadcaster(
            maxlen=5, queue_size=4, delivery_enabled=False
        )
        for index in range(100):
            broadcaster.push(alert(f"id-{index}"))
        self.assertEqual(broadcaster.queue.qsize(), 0)
        self.assertEqual(len(broadcaster.recent()), 5)


class WebSocketIsolationTests(unittest.TestCase):
    def test_slow_client_times_out(self):
        class SlowWebSocket:
            def __init__(self):
                self.closed = False

            async def send_text(self, _message):
                await asyncio.sleep(0.1)

            async def close(self, **_kwargs):
                self.closed = True

        async def exercise():
            client = SlowWebSocket()
            with patch.object(server, "_WS_SEND_TIMEOUT_SECONDS", 0.01):
                delivered = await server._send_ws_message(client, "message")
            self.assertFalse(delivered)
            self.assertTrue(client.closed)

        asyncio.run(exercise())


class OperationsSafetyTests(unittest.TestCase):
    def test_port_probe_rejects_an_existing_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with self.assertRaises(OSError):
                server._ensure_port_available("127.0.0.1", port)

    def test_shutdown_endpoint_is_loopback_only(self):
        original_callback = server._shutdown_callback
        calls = []
        server._shutdown_callback = lambda: calls.append(True)
        try:
            local_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
            response = asyncio.run(server.shutdown_service(local_request))
            self.assertEqual(response.status_code, 202)
            self.assertEqual(len(calls), 1)

            remote_request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.8"))
            response = asyncio.run(server.shutdown_service(remote_request))
            self.assertEqual(response.status_code, 403)
        finally:
            server._shutdown_callback = original_callback

    def test_rtsp_credentials_are_redacted(self):
        source = "rtsp://operator:secret@example.com:554/live?token=value"
        rendered = redact_source(source)
        self.assertNotIn("operator", rendered)
        self.assertNotIn("secret", rendered)
        self.assertIn("example.com:554", rendered)


class RetentionAndLogTests(unittest.TestCase):
    def test_database_retention_removes_only_expired_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database = Database(Path(temp_dir) / "retention.db")
            try:
                old = alert("old")
                old["timestamp"] = time.time() - 40 * 86400
                database.insert_alert(old)
                database.insert_alert(alert("new"))
                removed = database.apply_retention(30)
                total, remaining = database.query_alert_page(limit=10)
                self.assertEqual(removed["alerts"], 1)
                self.assertEqual(total, 1)
                self.assertEqual(remaining[0]["alert_id"], "new")
            finally:
                database.close()

    def test_concurrent_jsonl_writes_remain_parseable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "alerts.jsonl"
            manager = AlertManager(alert_log=path)
            try:
                threads = [
                    threading.Thread(
                        target=manager.trigger_intrusion,
                        args=("cam_a", f"person-{index}", "zone"),
                    )
                    for index in range(40)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                manager.close()
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(rows), 40)
                self.assertEqual(len({row["alert_id"] for row in rows}), 40)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
