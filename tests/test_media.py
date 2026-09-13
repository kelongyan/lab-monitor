"""
test_media.py — /media/{cam} 回放出片的 HTTP 层测试（P1-1b）。

为什么值得单独测：这是本项目**第一个把磁盘文件直接吐给浏览器**的端点。
四个真会咬人的点：
  1. Range —— <video> 拖动进度条全靠它；算错就是"能播但拖不动"或 416 黑屏。
  2. 路径穿越 —— 若拿 URL 里的 camera_id 去拼路径，`../` 就能读到项目外文件。
     这里相机 ID 只用于查表，真实路径只来自 video_assets.rel_path。
  3. Content-Length 不能读 size_bytes —— 低清转码行那一列是 NULL（实测 22/22），
     读它会得到长度 0，视频卡在第一帧且不报错。
  4. rel_path 只许落在 videos/ / videos_low/ 两个素材目录 —— 只校验"不逃出项目根"
     是不够的，库被写脏成 config/*.json 时项目根校验照样把配置文件吐出去。

测试用真实临时文件，因此走的是真正的 stat() / seek() / 分片读盘路径，而不是 mock。
用三个相机把关注点分开，每个用例只验证一件事：
  cam_hi   只索引 videos/ 下的原片          → Range 语义（内容确定）
  cam_low  只索引 videos_low 下的低清片     → 低清路径可出片
  cam_pair 两者都有                         → 默认选低清的偏好
"""

import tempfile
import unittest
from pathlib import Path

from src.db import Database

ROOT = Path(__file__).resolve().parent.parent
#: 原片测试文件放 videos/ 下（端点白名单内的素材目录，gitignored、可重建）
HI_REL = "videos/__test_media_hi__.mp4"
# 项目根之内但**不在**素材目录里的文件 —— 用于验证目录白名单
OUTSIDE_MEDIA_REL = "outputs/tmp/__inside_root_but_not_media__.mp4"


def _write_media(rel_path: str, payload: bytes) -> Path:
    """在项目根下按 rel_path 落一个真实文件（media 端点以项目根为基准）。"""
    target = ROOT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


class MediaEndpointTest(unittest.TestCase):
    """Range 语义 + 路径安全 + 低清优先。"""

    HI = bytes(range(256)) * 8                    # 2048 字节，内容可预测
    LOW = bytes(reversed(range(256))) * 8         # 与 HI 不同，用于区分选中了谁

    def setUp(self):
        from fastapi.testclient import TestClient
        import server

        self.server = server
        self._temp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._temp.name) / "media.db")

        # 原片必须放在 videos/ 前缀下 —— 端点只对 videos/、videos_low/ 开放
        self.rel_hi = HI_REL
        # 低清片必须放在 videos_low/ 前缀下 —— 端点的偏好就是认这个前缀
        # （与 scripts/transcode_lowres.py:141 的 rel_path 写法一致）。
        # 该目录已 gitignore，且是可由转码脚本重建的派生产物。
        self.rel_low = "videos_low/__test_media_low__.mp4"
        _write_media(self.rel_hi, self.HI)
        _write_media(self.rel_low, self.LOW)
        self.addCleanup(lambda: (ROOT / self.rel_hi).unlink(missing_ok=True))
        self.addCleanup(lambda: (ROOT / self.rel_low).unlink(missing_ok=True))

        # cam_hi：原片，size_bytes 有值
        self.db.seed_video_asset(camera_id="cam_hi", rel_path=self.rel_hi,
                                 size_bytes=len(self.HI), frames_real=100,
                                 duration_real=10.0, fps_declared=25.0)
        # cam_low：低清片，**故意不给 size_bytes** —— 复刻库里 22 行低清的真实状态
        self.db.seed_video_asset(camera_id="cam_low", rel_path=self.rel_low,
                                 frames_real=104, duration_real=10.4)
        # cam_pair：两行都有，用于验证默认偏好
        self.db.seed_video_asset(camera_id="cam_pair", rel_path=self.rel_hi,
                                 size_bytes=len(self.HI), frames_real=100,
                                 duration_real=10.0)
        self.db.seed_video_asset(camera_id="cam_pair", rel_path=self.rel_low,
                                 frames_real=104, duration_real=10.4)

        self.asset_hi_id = self._asset_id("cam_hi", self.rel_hi)
        self.asset_low_id = self._asset_id("cam_low", self.rel_low)

        server.init_server(frame_hub=None, broadcaster=None,
                           identity_store=None, personnel=None)
        # /media 用的是 src.db 的模块级单例；替换成临时库，避免碰生产库。
        # 注意"保存原值"不能直接读 db_module.db —— 惰性单例（PEP 562）下
        # 那次读取本身就会创建生产库实例。用哨兵区分"原本不存在"。
        import src.db as db_module
        self._db_module = db_module
        self._singleton_sentinel = object()
        self._orig_singleton = vars(db_module).get("db", self._singleton_sentinel)
        db_module.db = self.db
        self.client = TestClient(server.app)

    def _asset_id(self, camera: str, rel_path: str) -> int:
        for asset in self.db.list_video_assets():
            if asset["camera_id"] == camera and asset["rel_path"] == rel_path:
                return asset["asset_id"]
        raise AssertionError(f"资产未登记: {camera} {rel_path}")

    def tearDown(self):
        if self._orig_singleton is self._singleton_sentinel:
            vars(self._db_module).pop("db", None)   # 恢复"单例尚未创建"的惰性状态
        else:
            self._db_module.db = self._orig_singleton
        self.server.init_server(None, None, None)
        self.db.close()
        self._temp.cleanup()

    # ---------------------------------------------------------------- 完整请求 #
    def test_full_request_has_length_and_accept_ranges(self):
        """无 Range 时给整片，且必须声明 Accept-Ranges（否则播放器不敢拖）。"""
        res = self.client.get("/media/cam_hi")
        self.assertEqual(res.status_code, 200, res.text[:200])
        self.assertEqual(res.headers["accept-ranges"], "bytes")
        self.assertEqual(res.headers["content-type"], "video/mp4")
        self.assertEqual(int(res.headers["content-length"]), len(self.HI))
        self.assertEqual(res.content, self.HI)

    def test_length_is_measured_not_read_from_db(self):
        """
        回归：低清行 size_bytes 为 NULL，若拿它当长度会得到 0（视频卡第一帧且不报错）。
        这里断言真实 stat() 长度，等于钉死"不许读那一列"。
        """
        res = self.client.get("/media/cam_low")
        self.assertEqual(res.status_code, 200, res.text[:200])
        self.assertEqual(int(res.headers["content-length"]), len(self.LOW))
        self.assertGreater(int(res.headers["content-length"]), 0)
        self.assertEqual(res.content, self.LOW)

    def test_default_prefers_lowres_transcode(self):
        """
        同一相机两行时默认给低清片：体积约 1/30，且 identity_appearances.video_ts
        就是按它标定的 —— 给原片会让 seek 差几十毫秒（实测 239.12 vs 239.64）。

        两片内容不同，所以能按字节直接断言选了哪一个，
        而不是在测试里重抄一遍选择逻辑（那样只是自证）。
        """
        res = self.client.get("/media/cam_pair")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, self.LOW)

    def test_explicit_highres_asset_overrides_default(self):
        """显式指定原片时要能出片：低清优先只是默认，不是硬性替换。"""
        res = self.client.get(f"/media/cam_hi?asset_id={self.asset_hi_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content, self.HI)

    # ---------------------------------------------------------------- Range #
    def test_range_first_bytes(self):
        res = self.client.get("/media/cam_hi", headers={"Range": "bytes=0-99"})
        self.assertEqual(res.status_code, 206)
        self.assertEqual(res.headers["content-range"], f"bytes 0-99/{len(self.HI)}")
        self.assertEqual(int(res.headers["content-length"]), 100)
        self.assertEqual(res.content, self.HI[:100])

    def test_range_open_ended(self):
        res = self.client.get("/media/cam_hi", headers={"Range": "bytes=2000-"})
        self.assertEqual(res.status_code, 206)
        self.assertEqual(res.content, self.HI[2000:])
        self.assertEqual(res.headers["content-range"],
                         f"bytes 2000-{len(self.HI) - 1}/{len(self.HI)}")

    def test_range_suffix_reads_tail(self):
        """bytes=-N 表示最后 N 字节（不是从 N 开始）；搞反会导致拖到结尾黑屏。"""
        res = self.client.get("/media/cam_hi", headers={"Range": "bytes=-50"})
        self.assertEqual(res.status_code, 206)
        self.assertEqual(len(res.content), 50)
        self.assertEqual(res.content, self.HI[-50:])
        self.assertEqual(res.headers["content-range"],
                         f"bytes {len(self.HI) - 50}-{len(self.HI) - 1}/{len(self.HI)}")

    def test_range_on_suffix_larger_than_file(self):
        """bytes=-99999 超过文件长度时按整个文件返回，不能算出负数 start。"""
        res = self.client.get("/media/cam_hi", headers={"Range": "bytes=-99999"})
        self.assertEqual(res.status_code, 206)
        self.assertEqual(res.content, self.HI)
        self.assertEqual(res.headers["content-range"],
                         f"bytes 0-{len(self.HI) - 1}/{len(self.HI)}")

    def test_range_beyond_eof_is_416(self):
        res = self.client.get("/media/cam_hi", headers={"Range": "bytes=99999-"})
        self.assertEqual(res.status_code, 416)
        self.assertEqual(res.headers["content-range"], f"bytes */{len(self.HI)}")

    def test_malformed_range_never_produces_5xx(self):
        """
        畸形 Range **绝不允许 5xx**（一个怪头就把端点打崩是不可接受的）。

        注意 `bytes=` / `bytes=-` / `bytes=abc` 由 httpx→Starlette 传输层在更早处
        就拒成 400，根本到不了本端点，所以这里只断言"不是 5xx"。
        真正的端点内部分支由下面 test_parse_range_both_empty_* 直接覆盖。
        """
        for bad in ("bytes=-", "bytes=", "bytes=abc", "items=0-10", "bytes=--"):
            res = self.client.get("/media/cam_hi", headers={"Range": bad})
            self.assertLess(res.status_code, 500,
                            f"{bad} 不该 5xx（拿到 {res.status_code}）")

    def test_parse_range_both_empty_returns_whole_file(self):
        """
        `bytes=-` 两端皆空是真实踩过的 500：正则 `bytes=(\\d*)-(\\d*)` 会匹配它，
        但 `int('')` 抛 ValueError。传输层拦得住 TestClient 发的这种头，
        拦不住别的 HTTP 客户端，所以直接调 handler 把这条分支钉死。
        """
        import asyncio

        import server
        from starlette.requests import Request

        scope = {
            "type": "http", "method": "GET", "path": "/media/cam_hi",
            "query_string": b"", "headers": [(b"range", b"bytes=-")],
            "scheme": "http", "server": ("testserver", 80),
        }
        request = Request(scope)
        res = asyncio.run(server.get_media("cam_hi", request, asset_id=None,
                                          download=False))
        self.assertEqual(res.status_code, 200, "两端皆空的 Range 应回整片")
        # 返回的是整片 FileResponse（不是 206 分片）；Content-Length 由它
        # 在发送时按 stat 补，构造期读不到，所以这里断言类型与路径。
        self.assertEqual(Path(res.path).resolve(), (ROOT / self.rel_hi).resolve())

    def test_end_clamped_to_file_size(self):
        """bytes=2000-999999 的 end 超界要夹到 size-1，不能照抄回 Content-Range。"""
        res = self.client.get("/media/cam_hi", headers={"Range": "bytes=2000-999999"})
        self.assertEqual(res.status_code, 206)
        self.assertEqual(res.headers["content-range"],
                         f"bytes 2000-{len(self.HI) - 1}/{len(self.HI)}")
        self.assertEqual(res.content, self.HI[2000:])

    # ------------------------------------------------------------ 错误与安全 #
    def test_unknown_camera_is_404(self):
        self.assertEqual(self.client.get("/media/no_such_cam").status_code, 404)

    def test_traversal_in_camera_id_is_rejected(self):
        """
        相机 ID 只用于查表，绝不参与拼路径。这里断言穿越尝试被挡在 400/404，
        且**绝不**返回 200 —— 200 就意味着真的读到了项目外的文件。
        """
        for evil in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "cam_a/../../secret",
                     "..\\..\\windows\\win.ini", "a b", "cam;rm -rf /"):
            res = self.client.get(f"/media/{evil}")
            self.assertIn(res.status_code, (400, 404),
                          f"{evil} 不应被接受（拿到 {res.status_code}）")

    def test_poisoned_rel_path_cannot_escape_project_root(self):
        """
        兜底防线：即使库里 rel_path 被写脏（含 ../ 指向项目外），
        resolve()+is_relative_to() 也必须拒绝，返回 404 而不是把文件吐出去。
        """
        self.db.seed_video_asset(camera_id="cam_evil",
                                 rel_path="../../../../Windows/win.ini",
                                 frames_real=1, duration_real=1.0)
        res = self.client.get("/media/cam_evil")
        self.assertEqual(res.status_code, 404, res.text[:200])

    def test_missing_file_on_disk_is_404_not_500(self):
        """库里有行但磁盘文件被删（转码中断）→ 404，不能 500 打崩前端。"""
        self.db.seed_video_asset(camera_id="cam_gone",
                                 rel_path="videos/definitely_not_here.mp4",
                                 frames_real=1, duration_real=1.0)
        res = self.client.get("/media/cam_gone")
        self.assertEqual(res.status_code, 404)

    def test_rel_path_inside_root_but_outside_media_dirs_is_404(self):
        """
        目录白名单回归：rel_path 指向项目根**之内**、但不在 videos/、videos_low/
        里的真实文件（如被写脏成 outputs/ 下的任意文件）必须 404。
        只校验"不逃出项目根"的老逻辑会把这个文件当视频吐出去。
        """
        _write_media(OUTSIDE_MEDIA_REL, b"not-a-video")
        self.addCleanup(lambda: (ROOT / OUTSIDE_MEDIA_REL).unlink(missing_ok=True))
        self.db.seed_video_asset(camera_id="cam_dirty",
                                 rel_path=OUTSIDE_MEDIA_REL,
                                 frames_real=1, duration_real=1.0)
        res = self.client.get("/media/cam_dirty")
        self.assertEqual(res.status_code, 404, res.text[:200])

    def test_asset_id_must_match_camera(self):
        """asset_id 与 camera 不自洽时返回 409，避免前端拿错资产却播得"像对的"。"""
        res = self.client.get(f"/media/no_such_cam?asset_id={self.asset_hi_id}")
        self.assertEqual(res.status_code, 409, res.text[:200])

    def test_unknown_asset_id_is_404(self):
        self.assertEqual(self.client.get("/media/cam_hi?asset_id=999999").status_code, 404)

    def test_download_sets_content_disposition(self):
        res = self.client.get("/media/cam_hi?download=true")
        self.assertEqual(res.status_code, 200)
        self.assertIn("attachment", res.headers["content-disposition"])

    def test_get_is_not_blocked_by_write_guard(self):
        """GET 不该被 X-Lab-Monitor-Request 守卫拦掉（守卫只管写方法）。"""
        self.assertNotEqual(self.client.get("/media/cam_hi").status_code, 403)


class MediaUsesInjectedDatabaseTest(unittest.TestCase):
    """
    回归：/media 必须与检索共用**同一个注入库**，而不是 src.db 的模块级全局。

    临时/演示库场景里，检索（_identity_store.database）返回的是临时库的
    asset_id / 相机；/media 若仍查全局库，同一个 asset_id 会被拿到另一个
    库里解释（404，甚至指向完全无关的文件）。判据用生产库里不存在的相机：
    走注入库 → 200；走全局库 → 404 "Camera not found"。
    """

    def test_media_serves_asset_from_injected_store_db(self):
        import server
        from fastapi.testclient import TestClient
        from src.identity_store import IdentityStore

        rel = "videos_low/__test_media_unify__.mp4"
        _write_media(rel, b"unify")
        self.addCleanup(lambda: (ROOT / rel).unlink(missing_ok=True))
        # 先注册目录清理、后注册库关闭：addCleanup 是 LIFO，
        # 必须先 close 库再删目录，否则 Windows 上文件被占用（WinError 32）
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        database = Database(Path(temp.name) / "unify.db")
        self.addCleanup(database.close)
        database.seed_video_asset(
            camera_id="media_unify_cam", rel_path=rel,
            frames_real=10, duration_real=1.0)
        store = IdentityStore(database=database, feature_space="test:4")
        server.init_server(frame_hub=None, broadcaster=None,
                           identity_store=store, personnel=None)
        self.addCleanup(server.init_server, None, None, None)
        client = TestClient(server.app)
        res = client.get("/media/media_unify_cam")
        self.assertEqual(res.status_code, 200, res.text[:200])
        self.assertEqual(res.content, b"unify")


if __name__ == "__main__":
    unittest.main()
