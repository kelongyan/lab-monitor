"""F9a + F5 回归测试
- MJPEG 流：JPEG 编码卸出事件循环 + generation 增量推送 + 心跳兜底 + 相机白名单
- 写接口安全边界：Host 白名单（421）、跨站写保护（403）、GET 不受影响
"""

import asyncio
import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import server
from src.frame_hub import FrameHub


LOOPBACK_CLIENT = ("127.0.0.1", 12345)
WRITE_HEADER = {"X-Lab-Monitor-Request": "1"}


class ServerGlobalsMixin:
    """备份/恢复 server 模块的运行时全局对象，避免污染其他测试。"""

    def snapshot_globals(self):
        self._saved = {
            "_frame_hub": server._frame_hub,
            "_pipelines": server._pipelines,
            "_shutdown_callback": server._shutdown_callback,
            "_allowed_hosts": server._allowed_hosts,
            "_mjpeg_sleep": server._mjpeg_sleep,
        }

    def restore_globals(self):
        for name, value in self._saved.items():
            setattr(server, name, value)


class HostWhitelistHelperTests(unittest.TestCase):
    def test_allowed_hosts_are_derived_from_bind_port(self):
        allowed = server._build_allowed_hosts("127.0.0.1", 9123)
        self.assertIn("127.0.0.1:9123", allowed)
        self.assertIn("localhost:9123", allowed)
        self.assertNotIn("127.0.0.1:8000", allowed)   # 不得硬编码 8000

    def test_missing_host_header_is_allowed(self):
        with patch.object(server, "_allowed_hosts", {"127.0.0.1:9123"}):
            self.assertTrue(server._host_allowed(None))          # 内部探活不带 Host
            self.assertTrue(server._host_allowed("127.0.0.1:9123"))
            self.assertFalse(server._host_allowed("evil.example"))

    def test_unconfigured_whitelist_does_not_block(self):
        with patch.object(server, "_allowed_hosts", set()):
            self.assertTrue(server._host_allowed("anything"))

    def test_non_ascii_credentials_fail_auth_instead_of_raising(self):
        os.environ["LAB_MONITOR_USERNAME"] = "运维"
        os.environ["LAB_MONITOR_PASSWORD"] = "密码"
        try:
            import base64
            token = base64.b64encode("运维:密码".encode("utf-8")).decode("ascii")
            self.assertFalse(server._authorization_valid(f"Basic {token}"))
        finally:
            os.environ.pop("LAB_MONITOR_USERNAME", None)
            os.environ.pop("LAB_MONITOR_PASSWORD", None)


class WriteGuardApiTests(ServerGlobalsMixin, unittest.TestCase):
    def setUp(self):
        self.snapshot_globals()
        self.old_username = os.environ.pop("LAB_MONITOR_USERNAME", None)
        self.old_password = os.environ.pop("LAB_MONITOR_PASSWORD", None)
        self.shutdown_calls = []
        server._frame_hub = None
        server._pipelines = None
        server._allowed_hosts = set()          # Host 校验关闭，专测写保护
        server._shutdown_callback = lambda: self.shutdown_calls.append(1)
        self.client = TestClient(server.app, client=LOOPBACK_CLIENT)

    def tearDown(self):
        self.restore_globals()
        if self.old_username is not None:
            os.environ["LAB_MONITOR_USERNAME"] = self.old_username
        if self.old_password is not None:
            os.environ["LAB_MONITOR_PASSWORD"] = self.old_password

    def test_simple_cross_site_post_is_rejected(self):
        response = self.client.post("/api/admin/shutdown")
        self.assertEqual(403, response.status_code)
        self.assertIn("X-Lab-Monitor-Request", response.json()["error"])
        self.assertEqual([], self.shutdown_calls)

    def test_post_with_custom_header_is_accepted(self):
        response = self.client.post("/api/admin/shutdown", headers=WRITE_HEADER)
        self.assertEqual(202, response.status_code)
        self.assertEqual([1], self.shutdown_calls)

    def test_foreign_origin_is_rejected_even_with_header(self):
        response = self.client.post(
            "/api/admin/shutdown",
            headers={**WRITE_HEADER, "Origin": "http://evil.example"},
        )
        self.assertEqual(403, response.status_code)
        self.assertEqual([], self.shutdown_calls)

    def test_same_origin_is_accepted(self):
        response = self.client.post(
            "/api/admin/shutdown",
            headers={**WRITE_HEADER, "Origin": "http://testserver"},
        )
        self.assertEqual(202, response.status_code)
        self.assertEqual([1], self.shutdown_calls)

    def test_get_requests_are_not_affected(self):
        for path in ("/healthz", "/api/status", "/api/alerts"):
            with self.subTest(path=path):
                self.assertEqual(200, self.client.get(path).status_code)

    def test_guard_runs_before_authentication(self):
        os.environ["LAB_MONITOR_USERNAME"] = "operator"
        os.environ["LAB_MONITOR_PASSWORD"] = "secret-value"
        self.assertEqual(403, self.client.post("/api/admin/shutdown").status_code)
        self.assertEqual(401, self.client.get("/healthz").status_code)
        self.assertEqual([], self.shutdown_calls)

    def test_unknown_host_returns_421(self):
        server._allowed_hosts = server._build_allowed_hosts("127.0.0.1", 9123)
        self.assertEqual(421, self.client.get("/healthz").status_code)

        allowed_client = TestClient(
            server.app, base_url="http://127.0.0.1:9123", client=LOOPBACK_CLIENT
        )
        self.assertEqual(200, allowed_client.get("/healthz").status_code)


class StreamWhitelistTests(ServerGlobalsMixin, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.snapshot_globals()
        server._allowed_hosts = set()
        self.client = TestClient(server.app, client=LOOPBACK_CLIENT)

    def tearDown(self):
        self.restore_globals()

    def test_unknown_camera_returns_404_without_any_init(self):
        server._frame_hub = None
        server._pipelines = None
        response = self.client.get("/stream/whatever")
        self.assertEqual(404, response.status_code)   # 白名单为空也不能 fail-open

    def test_unknown_camera_returns_404_with_configured_pipelines(self):
        server._frame_hub = None
        server._pipelines = [SimpleNamespace(camera_id="cam_01")]
        self.assertEqual(404, self.client.get("/stream/cam_02").status_code)

    async def test_configured_camera_streams_jpeg(self):
        # 直接驱动路由与生成器：TestClient 对"永不结束"的流在关闭时会阻塞
        hub = FrameHub(display_width=32, display_height=18)
        hub.push_frame("cam_01", np.zeros((18, 32, 3), dtype=np.uint8))
        server._frame_hub = hub
        server._pipelines = [SimpleNamespace(camera_id="cam_01")]

        response = await server.video_stream("cam_01")
        self.assertIsInstance(response, StreamingResponse)
        chunk = await response.body_iterator.__anext__()
        await response.body_iterator.aclose()

        self.assertIn(b"Content-Type: image/jpeg", chunk)
        self.assertIn(b"\xff\xd8", chunk)


class MjpegGeneratorTests(ServerGlobalsMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.snapshot_globals()
        server._mjpeg_sleep = 0.01

    async def asyncTearDown(self):
        self.restore_globals()

    async def _collect(
        self,
        cam_id: str,
        seconds: float,
        after_first=None,
        min_chunks: int = 0,
        settle: float = 0.0,
    ):
        """
        驱动 MJPEG 生成器并收集推送出的 part。

        不要用固定的墙钟窗口去推断"应该收到几个 part"：本机跑着 22 路推理时事件循环
        会被饿到只调度一两次，曾导致心跳用例偶发失败（20 轮 1 次）。改为等到攒够
        min_chunks 个 part 为止（seconds 作为兜底上限），再可选地静置 settle 秒确认
        没有多余推送。
        """
        chunks = []
        generator = server._mjpeg_generator(cam_id)

        async def pump():
            async for chunk in generator:
                chunks.append(chunk)

        async def wait_for(target: int, deadline: float) -> None:
            loop_deadline = asyncio.get_running_loop().time() + deadline
            while len(chunks) < target:
                if asyncio.get_running_loop().time() > loop_deadline:
                    return
                await asyncio.sleep(0.01)

        task = asyncio.create_task(pump())
        try:
            if min_chunks:
                await wait_for(min_chunks, seconds)
            else:
                await asyncio.sleep(seconds)
            if after_first is not None:
                after_first()
                if min_chunks:
                    await wait_for(min_chunks + 1, seconds)
                else:
                    await asyncio.sleep(seconds)
            if settle:
                await asyncio.sleep(settle)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return chunks

    async def test_unchanged_frame_is_not_resent_and_encoding_is_offloaded(self):
        hub = FrameHub(display_width=32, display_height=18)
        hub.push_frame("cam_01", np.zeros((18, 32, 3), dtype=np.uint8))
        server._frame_hub = hub
        offloaded = []
        original = server.run_in_threadpool

        async def spy(func, *args, **kwargs):
            offloaded.append(getattr(func, "__name__", ""))
            return await original(func, *args, **kwargs)

        def push_new_frame():
            hub.push_frame("cam_01", np.full((18, 32, 3), 255, dtype=np.uint8))

        with patch.object(server, "run_in_threadpool", spy):
            chunks = await self._collect(
                "cam_01", 5.0, after_first=push_new_frame, min_chunks=1, settle=0.15,
            )

        # 只推送 2 帧：初始帧 + generation 变化后的新帧；静置期内不应再有重复推送
        self.assertEqual(2, len(chunks))
        self.assertNotEqual(chunks[0], chunks[1])
        self.assertIn("get_jpeg_with_generation", offloaded)   # 编码必须在线程池
        self.assertIn("_make_offline_frame", offloaded)

    async def test_heartbeat_resends_when_nothing_changes(self):
        hub = FrameHub(display_width=32, display_height=18)
        hub.push_frame("cam_01", np.zeros((18, 32, 3), dtype=np.uint8))
        server._frame_hub = hub

        with patch.object(server, "_MJPEG_HEARTBEAT_SECONDS", 0.05):
            chunks = await self._collect("cam_01", 5.0, min_chunks=3)

        self.assertGreaterEqual(len(chunks), 3)   # 心跳兜底：静默连接仍会重发
        self.assertEqual({chunks[0]}, set(chunks))

    async def test_offline_camera_still_receives_placeholder(self):
        hub = FrameHub(display_width=32, display_height=18)
        hub.mark_offline("cam_01", status_text="RECONNECTING", reconnect_count=1)
        server._frame_hub = hub

        chunks = await self._collect("cam_01", 0.15)

        self.assertEqual(1, len(chunks))
        self.assertIn(b"Content-Type: image/jpeg", chunks[0])
        self.assertIn(b"\xff\xd8", chunks[0])


class FrameHubGenerationTests(unittest.TestCase):
    def test_generation_tracks_pushes_and_offline(self):
        hub = FrameHub(display_width=32, display_height=18)
        self.assertEqual(0, hub.get_generation("cam_01"))       # 未知相机

        hub.push_frame("cam_01", np.zeros((18, 32, 3), dtype=np.uint8))
        jpeg, generation = hub.get_jpeg_with_generation("cam_01")
        self.assertIsNotNone(jpeg)
        self.assertEqual(generation, hub.get_generation("cam_01"))
        self.assertEqual(jpeg, hub.get_jpeg("cam_01"))          # 旧接口行为不变

        hub.push_frame("cam_01", np.full((18, 32, 3), 255, dtype=np.uint8))
        self.assertGreater(hub.get_generation("cam_01"), generation)

        hub.mark_offline("cam_01")
        offline_jpeg, offline_generation = hub.get_jpeg_with_generation("cam_01")
        self.assertIsNone(offline_jpeg)
        self.assertGreater(offline_generation, generation)


if __name__ == "__main__":
    unittest.main()
