"""
snapshots.py — 跨相机抓拍序列（身份在各路相机上的真实画面裁图）。

用途
----
管理界面要展示「同一个人出现在多路相机」，最直观的形式是把该身份在各路相机上的
**真实抓拍**并排对比。素材已烧录相机名与时间，图像全部来自真实帧、不含任何合成，
因此这种呈现方式本身即证据。

数据来源与坐标口径（实测确认，勿凭直觉改）
------------------------------------------
- 时间：`identity_appearances.video_ts`（**视频内秒数**）。不能用墙钟 `timestamp` ——
  素材循环播放会把墙钟拉成几天，拿墙钟去 seek 必然落到无关画面。
- 坐标：`bbox_json` 位于**处理帧**坐标系（960×540，即 `process_max_width=960`），
  **不是源片分辨率**（1920×1080 / 2560×1440）。实测依据：bbox 的 y2 恒等于 540.0。
  低清素材 `videos_low/<cam>.mp4` 恰为 960×540，故坐标可直接使用；
  若素材尺寸不同，本模块按 (w/960, h/540) 等比换算。

为什么走 `videos_low` 而不是原片
--------------------------------
- 体积：低清 28.5 MB / 原片 1 463 MB，解码一帧的开销差一个量级
- 回放口径与检索一致（`src/search.py::_asset_index` 同样归一到低清片）
- 抽帧精度：原片容器时间戳损坏（reg_06/reg_08 按时间 seek 会落到无关画面），低清转码
  产物时间戳正确

单点实现
--------
本模块是抓拍裁图的**唯一实现**，`scripts/render_cross_camera_strip.py` 与
`server.py` 的 `/api/identities/{gid}/snapshots` 都调用它。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

logger = logging.getLogger("snapshots")

# 处理帧坐标系（与 pipeline 的 process_max_width=960 对应）
FRAME_W = 960
FRAME_H = 540

MIN_BOX_W = 24          # 过小的框裁出来没有辨识度
MIN_BOX_H = 48
PAD_RATIO = 0.12        # 裁图外扩比例，留出上下文便于目视认人
MARGIN_TOLERANCE = 2.0  # 框贴边超过该像素数视为残缺样本，降权
TILE_H = 360            # 拼图中每格的高度
STRIP_PAD = 14
LABEL_H = 30
BG_COLOR = (245, 245, 245)
FG_COLOR = (30, 30, 30)
ACCENT_COLOR = (150, 80, 10)


@dataclass
class Snapshot:
    """单路相机的一张抓拍。"""

    camera: str
    desc: str
    video_ts: float
    wall_time: float
    bbox: list[float]
    frames: int
    file: str            # 拼图/裁图的磁盘路径（相对 outputs/ 或绝对路径由调用方决定）
    url: str = ""        # 可直接给 <img src> 的 URL，由调用方填充

    def as_dict(self) -> dict:
        return asdict(self)


def pick_representative(rows: list[dict]) -> tuple[dict, list[float]] | None:
    """
    从同一相机的候选行里挑一张最适合作图的样本。

    偏好（依次）：框面积大 → 不贴画面边缘（贴边说明人只进了半身）→ 竖长比接近人体
    （排除横向长条状的误检）。返回 (row, bbox) 或 None。
    """
    best = None
    best_score = -1.0
    for row in rows:
        bbox = row.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue
        margin = min(x1, y1, FRAME_W - x2, FRAME_H - y2)
        edge_penalty = 1.0 if margin < MARGIN_TOLERANCE else 0.0
        ratio_bonus = 1.0 if h / max(w, 1e-6) >= 1.2 else 0.0
        score = (w * h) * (1.0 - 0.6 * edge_penalty) * (1.0 + 0.5 * ratio_bonus)
        if score > best_score:
            best_score = score
            best = (row, [x1, y1, x2, y2])
    return best


def read_crop(
    low_dir: Path,
    camera: str,
    video_ts: float,
    bbox: list[float],
    pad_ratio: float = PAD_RATIO,
) -> np.ndarray | None:
    """
    按视频内秒数定位并裁出人体框。返回 BGR 图，失败返回 None。

    顺序解码 vs seek：低清转码产物的时间戳可靠，用 CAP_PROP_POS_MSEC 定位即可；
    原片相反（容器时间戳损坏），所以本函数只接受低清素材路径。
    """
    path = Path(low_dir) / f"{camera}.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # 末帧 seek 常失败，往前留 2 帧余量
        ts = max(0.0, min(float(video_ts), max(0.0, (frames - 2)) / fps))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    # bbox 在处理帧坐标系；按素材实际尺寸等比换算
    sx, sy = w / FRAME_W, h / FRAME_H
    x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
    pad_x, pad_y = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def _fit_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / max(h, 1)
    return cv2.resize(img, (max(1, int(w * scale)), height), interpolation=cv2.INTER_AREA)


def render_strip(tiles: list[tuple[str, str, np.ndarray]], tile_h: int = TILE_H) -> np.ndarray:
    """把 (camera, desc, crop) 列表拼成一张带标注的横向对比图。"""
    fitted = [(cam, desc, _fit_height(img, tile_h)) for cam, desc, img in tiles]
    total_w = STRIP_PAD + sum(t.shape[1] + STRIP_PAD for _, _, t in fitted)
    canvas = np.full((tile_h + LABEL_H + STRIP_PAD * 2, total_w, 3), BG_COLOR, dtype=np.uint8)
    x = STRIP_PAD
    for cam, desc, tile in fitted:
        canvas[STRIP_PAD : STRIP_PAD + tile_h, x : x + tile.shape[1]] = tile
        cv2.rectangle(canvas, (x, STRIP_PAD), (x + tile.shape[1], STRIP_PAD + tile_h),
                      (200, 200, 200), 1)
        cv2.putText(canvas, cam, (x, STRIP_PAD + tile_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT_COLOR, 1, cv2.LINE_AA)
        short = (desc or "")[:14]
        if short:
            cv2.putText(canvas, short, (x + 62, STRIP_PAD + tile_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, FG_COLOR, 1, cv2.LINE_AA)
        x += tile.shape[1] + STRIP_PAD
    return canvas


def build_identity_snapshots(
    database,
    global_id: str,
    out_dir: Path,
    low_dir: Path,
    desc_of: dict[str, str] | None = None,
    candidate_limit: int = 400,
    max_cameras: int = 8,
    force: bool = False,
) -> dict:
    """
    生成（或复用缓存）某身份的跨相机抓拍序列。

    缓存策略：裁图与拼图按 `<gid>_<cam>.jpg` / `<gid>_strip.jpg` 落盘，同时写
    `<gid>_meta.json` 侧车文件保存元数据（视频内秒数、墙钟时间等）。缓存命中时
    不重新解码 —— 解码是这条链路上最重的操作，而同一身份的抓拍在同一次演示里
    会被反复请求。

    返回结构（可直接作为接口响应体）：
      {"global_id", "camera_count", "cameras": [Snapshot.as_dict()...],
       "strip_file", "strip_url", "cached", "skipped": {camera: reason}}
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    desc_of = desc_of or {}

    meta_path = out_dir / f"{global_id}_meta.json"
    strip_path = out_dir / f"{global_id}_strip.jpg"

    if not force and meta_path.exists() and strip_path.exists():
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            # 侧车文件只存**文件名**，绝对路径在回读时用当前 out_dir 重建 ——
            # 否则换一个工作目录/部署路径后，缓存里的旧相对路径会失效。
            for item in cached.get("cameras", []):
                name = PurePosixPath(str(item.get("file") or "").replace("\\", "/")).name
                item["file"] = str(out_dir / name) if name else ""
            strip_name = PurePosixPath(str(cached.get("strip_file") or "").replace("\\", "/")).name
            cached["strip_file"] = str(out_dir / strip_name) if strip_name else ""
            cached["cached"] = True
            return cached
        except (json.JSONDecodeError, OSError):
            logger.warning("抓拍元数据损坏，重新生成: %s", meta_path)

    rows = database.top_appearances_by_area(global_id, limit=candidate_limit)
    by_camera: dict[str, list[dict]] = {}
    for row in rows:
        by_camera.setdefault(row["camera"], []).append(row)

    skipped: dict[str, str] = {}
    snapshots: list[Snapshot] = []
    tiles: list[tuple[str, str, np.ndarray]] = []

    # 相机顺序取「首次出现时间」，让拼图与通行链的时间顺序一致。
    # 逐个相机单独取最大框（而不是全局取前 N 大框再分组）—— 否则小框相机
    # （远距离机位）会被大框相机挤掉，表现为跨 4 路的身份只出 2 路图。
    camera_meta = database.identity_cameras(global_id)
    cameras = [c["camera"] for c in camera_meta][:max_cameras]

    for camera in cameras:
        candidates = by_camera.get(camera)
        if not candidates:
            candidates = database.top_appearances_by_area(
                global_id, limit=candidate_limit, camera_id=camera)
        picked = pick_representative(candidates) if candidates else None
        if picked is None:
            skipped[camera] = "无可用样本（框过小或贴边）"
            continue
        row, bbox = picked
        crop = read_crop(low_dir, camera, row["video_ts"], bbox)
        if crop is None:
            skipped[camera] = "素材缺失或解码失败"
            continue
        crop_path = out_dir / f"{global_id}_{camera}.jpg"
        cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        tiles.append((camera, desc_of.get(camera, ""), crop))
        snapshots.append(Snapshot(
            camera=camera,
            desc=desc_of.get(camera, ""),
            video_ts=round(float(row["video_ts"]), 2),
            wall_time=float(row["time"]),
            bbox=[round(v, 1) for v in bbox],
            frames=len(candidates),
            file=str(crop_path),
        ))

    strip_file = ""
    if tiles:
        strip = render_strip(tiles)
        cv2.imwrite(str(strip_path), strip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        strip_file = str(strip_path)

    payload = {
        "global_id": global_id,
        "camera_count": len(snapshots),
        "cameras": [s.as_dict() for s in snapshots],
        "strip_file": strip_file,
        "strip_url": "",
        "generated_at": time.time(),
        "skipped": skipped,
    }
    # 侧车文件只写文件名，绝对路径由读取方用 out_dir 重建（见上方缓存分支的说明）
    meta_payload = {
        **payload,
        "cameras": [{**c, "file": PurePosixPath(str(c["file"]).replace("\\", "/")).name}
                    for c in payload["cameras"]],
        "strip_file": PurePosixPath(str(strip_file).replace("\\", "/")).name if strip_file else "",
    }
    try:
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("抓拍元数据写入失败: %s", meta_path)
    return {**payload, "cached": False}
