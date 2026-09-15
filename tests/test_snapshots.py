"""
test_snapshots.py — 跨相机抓拍序列（src/snapshots.py 与 HTTP 层）。

覆盖：
  - pick_representative：面积/贴边/竖长比三项偏好，以及过小框的拒绝
  - read_crop：合成素材上按 video_ts 定位、bbox 坐标换算、越界钳制、缺失素材
  - build_identity_snapshots：落盘 + 侧车元数据 + 缓存命中 + 跳过原因
  - snapshot_url：只取文件名（路径穿越防护）
  - GET /api/identities/{gid}/snapshots：200 结构、非法 gid 400

关于坐标系：`bbox_json` 位于**处理帧** 960×540 空间（不是源片分辨率）。
本文件的合成素材用 480×270（一半尺寸），正是为了验证等比换算真的生效 ——
若实现里漏了换算，裁出来的框会偏到画面外，用例立刻失败。
"""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from src.db import Database
from src.identity_store import IdentityStore
from src.snapshots import (
    FRAME_H,
    FRAME_W,
    build_identity_snapshots,
    pick_representative,
    read_crop,
    render_strip,
)

GID = "aaaa1111"


def _write_clip(path: Path, width: int = FRAME_W, height: int = FRAME_H,
                frames: int = 40, fps: float = 25.0) -> None:
    """写一段合成素材：整幅深灰底 + 一个白色竖条，便于验证裁到的位置。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建合成素材（缺 mp4v 编码器）")
    try:
        for i in range(frames):
            frame = np.full((height, width, 3), 60, dtype=np.uint8)
            # 白条固定在画面左半部，位置随帧号轻微右移
            x0 = int(width * 0.2) + i
            cv2.rectangle(frame, (x0, int(height * 0.3)),
                          (x0 + int(width * 0.1), int(height * 0.9)), (240, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()


class PickRepresentativeTests(unittest.TestCase):
    def _row(self, bbox, area=0.0, camera="c1"):
        return {"camera": camera, "time": 0.0, "video_ts": 1.0,
                "bbox": list(bbox), "area": area}

    def test_prefers_larger_box(self):
        rows = [self._row([100, 100, 140, 220]), self._row([10, 10, 110, 300])]
        picked = pick_representative(rows)
        self.assertIsNotNone(picked)
        self.assertEqual(picked[1], [10.0, 10.0, 110.0, 300.0])

    def test_rejects_too_small_box(self):
        """过小的框裁出来没有辨识度，必须整条丢弃而不是硬裁。"""
        rows = [self._row([100, 100, 110, 130])]
        self.assertIsNone(pick_representative(rows))

    def test_prefers_box_not_touching_frame_edge(self):
        """两个框面积接近时，贴边的（人只进半身）应让位给完整框。"""
        edge = self._row([0, 0, 120, 300])
        inside = self._row([300, 100, 420, 400])
        picked = pick_representative([edge, inside])
        self.assertEqual(picked[1], [300.0, 100.0, 420.0, 400.0])

    def test_prefers_portrait_over_wide_box(self):
        """横向长条多是误检，同等面积下竖长的更像人。"""
        wide = self._row([100, 200, 400, 260])        # 300x60
        tall = self._row([500, 100, 560, 400])        # 60x300
        picked = pick_representative([wide, tall])
        self.assertEqual(picked[1], [500.0, 100.0, 560.0, 400.0])

    def test_handles_malformed_bbox(self):
        rows = [{"camera": "c1", "time": 0.0, "video_ts": 1.0, "bbox": [1, 2], "area": 0}]
        self.assertIsNone(pick_representative(rows))


class ReadCropTests(unittest.TestCase):
    def test_crop_scales_from_processing_frame_coordinates(self):
        """bbox 在处理帧 960×540 空间；素材是 480×270 时必须等比缩小。"""
        with tempfile.TemporaryDirectory() as tmp:
            low = Path(tmp)
            _write_clip(low / "cam1.mp4", width=480, height=270, frames=40, fps=25.0)
            # 处理帧坐标 (240,162)-(480,486) → 素材坐标应为 (120,81)-(240,243)
            bbox = [240.0, 162.0, 480.0, 486.0]
            crop = read_crop(low, "cam1", 0.5, bbox, pad_ratio=0.0)
            self.assertIsNotNone(crop)
            h, w = crop.shape[:2]
            self.assertEqual((w, h), (120, 162))
            # 白条在素材左半部 —— 若漏了坐标换算会裁到画面右半的灰底
            self.assertGreater(float(crop.mean()), 100.0)

    def test_missing_clip_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_crop(Path(tmp), "nope", 1.0, [0, 0, 100, 200]))

    def test_video_ts_beyond_end_is_clamped(self):
        """末帧 seek 常失败，实现应往前钳制而不是返回 None。"""
        with tempfile.TemporaryDirectory() as tmp:
            low = Path(tmp)
            _write_clip(low / "cam1.mp4", frames=20, fps=25.0)
            crop = read_crop(low, "cam1", 9999.0, [200, 100, 400, 400])
            self.assertIsNotNone(crop)

    def test_padding_expands_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            low = Path(tmp)
            _write_clip(low / "cam1.mp4", frames=20, fps=25.0)
            bbox = [300.0, 150.0, 500.0, 450.0]
            plain = read_crop(low, "cam1", 0.2, bbox, pad_ratio=0.0)
            padded = read_crop(low, "cam1", 0.2, bbox, pad_ratio=0.2)
            self.assertGreater(padded.shape[0], plain.shape[0])
            self.assertGreater(padded.shape[1], plain.shape[1])


class RenderStripTests(unittest.TestCase):
    def test_strip_layout_width_grows_with_tiles(self):
        tiles = [("a", "desc-a", np.full((100, 60, 3), 128, np.uint8)),
                 ("b", "desc-b", np.full((100, 90, 3), 200, np.uint8))]
        strip = render_strip(tiles, tile_h=60)
        self.assertEqual(strip.shape[0], 60 + 30 + 14 * 2)
        # 等比缩放到 tile_h=60 后宽度为 36 与 54；布局是「首尾各一个 PAD + 每格后一个 PAD」
        self.assertEqual(strip.shape[1], 14 * 3 + 36 + 54)


class BuildSnapshotsTests(unittest.TestCase):
    @contextmanager
    def _env(self, out_dir: Path, low_dir: Path, write_clips: bool = True):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "snap.db")
            if write_clips:
                _write_clip(low_dir / "cam1.mp4")
                _write_clip(low_dir / "cam2.mp4")
            for cam in ("cam1", "cam2"):
                db.seed_video_asset(camera_id=cam, rel_path=f"videos_low/{cam}.mp4",
                                    codec="h264", fps_declared=25.0,
                                    frames_real=40, duration_real=1.6)
            store = IdentityStore(database=db, feature_space="test:4")
            gid = store.register(np.array([1, 0, 0, 0], dtype=np.float32))
            aid1 = db.get_video_asset("cam1")["asset_id"]
            aid2 = db.get_video_asset("cam2")["asset_id"]
            db.record_appearance(gid, "cam1", 1000.0, [300, 150, 500, 450], 1,
                                 asset_id=aid1, video_frame=10, video_ts=0.4)
            db.record_appearance(gid, "cam1", 1001.0, [320, 150, 520, 450], 1,
                                 asset_id=aid1, video_frame=12, video_ts=0.5)
            db.record_appearance(gid, "cam2", 1002.0, [280, 140, 500, 460], 1,
                                 asset_id=aid2, video_frame=14, video_ts=0.6)
            try:
                yield db, gid, out_dir
            finally:
                db.close()

    def test_generates_crops_and_strip(self):
        with tempfile.TemporaryDirectory() as low_tmp, tempfile.TemporaryDirectory() as out_tmp:
            out = Path(out_tmp) / "snaps"
            low = Path(low_tmp)
            with self._env(out, low) as (db, gid, _):
                result = build_identity_snapshots(
                    db, gid, out, low, {"cam1": "一路", "cam2": "二路"})
            self.assertEqual(result["camera_count"], 2)
            self.assertFalse(result["cached"])
            self.assertEqual([c["camera"] for c in result["cameras"]], ["cam1", "cam2"])
            self.assertEqual(result["cameras"][0]["desc"], "一路")
            for item in result["cameras"]:
                self.assertTrue(Path(item["file"]).exists())
            self.assertTrue(Path(result["strip_file"]).exists())
            self.assertTrue((out / f"{gid}_meta.json").exists())

    def test_second_call_hits_cache(self):
        with tempfile.TemporaryDirectory() as low_tmp, tempfile.TemporaryDirectory() as out_tmp:
            out = Path(out_tmp) / "snaps"
            low = Path(low_tmp)
            with self._env(out, low) as (db, gid, _):
                build_identity_snapshots(db, gid, out, low)
                again = build_identity_snapshots(db, gid, out, low)
            self.assertTrue(again["cached"])
            self.assertEqual(again["camera_count"], 2)
            # 缓存命中时仍要能拼出可用的绝对路径（侧车只存文件名）
            for item in again["cameras"]:
                self.assertTrue(Path(item["file"]).is_absolute())
                self.assertTrue(Path(item["file"]).exists())

    def test_force_regenerates(self):
        with tempfile.TemporaryDirectory() as low_tmp, tempfile.TemporaryDirectory() as out_tmp:
            out = Path(out_tmp) / "snaps"
            low = Path(low_tmp)
            with self._env(out, low) as (db, gid, _):
                build_identity_snapshots(db, gid, out, low)
                forced = build_identity_snapshots(db, gid, out, low, force=True)
            self.assertFalse(forced["cached"])

    def test_missing_clip_is_reported_not_crashed(self):
        """素材缺失应进入 skipped 并在响应里说明，不能整体失败。"""
        with tempfile.TemporaryDirectory() as out_tmp:
            out = Path(out_tmp) / "snaps"
            with tempfile.TemporaryDirectory() as low_tmp:
                low = Path(low_tmp)          # 空的：没有 cam1.mp4 / cam2.mp4
                with self._env(out, low, write_clips=False) as (db, gid, _):
                    result = build_identity_snapshots(db, gid, out, low)
            self.assertEqual(result["camera_count"], 0)
            self.assertEqual(set(result["skipped"]), {"cam1", "cam2"})

    def test_unknown_identity_yields_empty_payload(self):
        with tempfile.TemporaryDirectory() as low_tmp, tempfile.TemporaryDirectory() as out_tmp:
            low = Path(low_tmp)
            with self._env(Path(out_tmp) / "snaps", low) as (db, gid, out):
                result = build_identity_snapshots(db, "nosuchid", out, low)
            self.assertEqual(result["camera_count"], 0)
            self.assertEqual(result["cameras"], [])


class SnapshotUrlTests(unittest.TestCase):
    def test_takes_basename_only(self):
        """只取文件名 —— 否则 gid 里塞 ../.. 就能读到目录外。"""
        import server
        self.assertEqual(server.snapshot_url("/a/b/c/gid_cam.jpg"),
                         "/identity-snapshots/gid_cam.jpg")
        self.assertEqual(server.snapshot_url("..\\..\\windows\\system32\\evil.jpg"),
                         "/identity-snapshots/evil.jpg")
        self.assertIsNone(server.snapshot_url(None))
        self.assertIsNone(server.snapshot_url(""))


@contextmanager
def _api_client():
    """打真实 app，但库与素材都在临时目录里。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = Database(root / "api.db")
        low = root / "videos_low"
        low.mkdir()
        _write_clip(low / "cam1.mp4")
        db.seed_video_asset(camera_id="cam1", rel_path="videos_low/cam1.mp4",
                            codec="h264", fps_declared=25.0,
                            frames_real=40, duration_real=1.6)
        aid = db.get_video_asset("cam1")["asset_id"]
        store = IdentityStore(database=db, feature_space="test:4")
        gid = store.register(np.array([1, 0, 0, 0], dtype=np.float32))
        db.record_appearance(gid, "cam1", 1000.0, [300, 150, 500, 450], 1,
                             asset_id=aid, video_frame=10, video_ts=0.4)

        import server
        from fastapi.testclient import TestClient

        old_low = os.environ.get("LAB_MONITOR_LOWRES_DIR")
        old_out = os.environ.get("LAB_MONITOR_IDENTITY_SNAPSHOTS")
        os.environ["LAB_MONITOR_LOWRES_DIR"] = str(low)
        os.environ["LAB_MONITOR_IDENTITY_SNAPSHOTS"] = str(root / "snaps")
        server.init_server(frame_hub=None, broadcaster=None, identity_store=store)
        client = TestClient(server.app)
        try:
            yield client, gid
        finally:
            server.init_server(None, None, None)
            for key, old in (("LAB_MONITOR_LOWRES_DIR", old_low),
                             ("LAB_MONITOR_IDENTITY_SNAPSHOTS", old_out)):
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
            db.close()


class SnapshotApiTests(unittest.TestCase):
    def test_returns_camera_list_with_urls(self):
        with _api_client() as (client, gid):
            resp = client.get(f"/api/identities/{gid}/snapshots")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["global_id"], gid)
            self.assertEqual(body["camera_count"], 1)
            item = body["cameras"][0]
            self.assertEqual(item["camera"], "cam1")
            self.assertEqual(item["url"], "/identity-snapshots/%s_cam1.jpg" % gid)
            self.assertNotIn("file", item)          # 磁盘路径不外泄
            self.assertEqual(body["strip_url"],
                             "/identity-snapshots/%s_strip.jpg" % gid)

    def test_rejects_illegal_gid(self):
        with _api_client() as (client, _):
            for bad in ("../etc/passwd", "a" * 65, "has space"):
                resp = client.get(f"/api/identities/{bad}/snapshots")
                self.assertIn(resp.status_code, (400, 404))

    def test_unknown_identity_returns_empty(self):
        with _api_client() as (client, _):
            resp = client.get("/api/identities/deadbeef/snapshots")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["camera_count"], 0)


class TrajectoryGroupsTests(unittest.TestCase):
    """轨迹接口新增的 groups / per_camera / flicker 字段。"""

    def test_payload_exposes_view_groups(self):
        with _api_client() as (client, gid):
            resp = client.get(f"/api/identities/{gid}/trajectory")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIn("groups", body)
            self.assertIn("per_camera", body)
            self.assertIn("flicker", body)
            self.assertEqual(body["flicker"]["raw_segments"], body["segment_count"])
            self.assertIn("cam1", body["per_camera"])
            self.assertEqual(body["per_camera"]["cam1"]["frames"], 1)


if __name__ == "__main__":
    unittest.main()
