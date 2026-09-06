"""
src/floorplan.py — 楼层平面图与摄像头点位坐标管理

职责边界：
- 只负责「camera_id ⇄ 平面图坐标」的映射与校验，不做轨迹计算、不做渲染。
- 与 ROI（config/roi.json，画面内归一化坐标）是两套独立坐标系，别混用：
  ROI 是「相机画面内」的归一化多边形，map_xy 是「楼层平面图」的归一化点位。

坐标系约定：
- map_xy = [x, y]，均归一化到 [0, 1]，原点在平面图左上角，x 向右、y 向下。
- facing_deg：摄像头朝向角度，0° = 正右（+x），顺时针增大（SVG 坐标系 y 向下）。
- fov_deg：水平视场角，用于前端画视锥扇形。

加载策略：
- 不提供模块级单例（src/db.py 的教训：import 即连库有副作用）。
- 调用方用 build_floorplan(path) 构造，server.py 在 init_server() 里注入。
- 文件缺失或字段非法时降级为空映射，不抛异常——平面图是可选增强，
  不能因为它把整个服务拖挂（与 topology 校验失败的处理口径一致）。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_MAP_FILE = Path(__file__).resolve().parent.parent / "config" / "camera_map.json"
# 底图托管在 /floorplan（server.py 挂载 assets/floorplan/），不入 Git（客户图纸）。
# 仓库里没有底图文件时，image_url 指向 404，前端应能容忍（只画点位与连线）。
DEFAULT_FLOORPLAN_IMAGE = "/floorplan/floorplan.jpg"

# 平面图底图像素尺寸（用于前端把归一化坐标还原成像素，仅作参考，不参与校验）
# 实际值由 static/floorplan.jpg 决定；这里只是给前端一个兜底宽高比。
FLOORPLAN_IMAGE_SIZE = (2420, 1711)

_COORD_MIN = 0.0
_COORD_MAX = 1.0
_FACING_MIN = 0.0
_FACING_MAX = 360.0
_FOV_MIN = 1.0
_FOV_MAX = 360.0


def _clean_coord(value) -> list[float] | None:
    """把 map_xy 清洗成 [x, y] 浮点并裁剪到 [0,1]；非法返回 None。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf
            return None
        out.append(min(_COORD_MAX, max(_COORD_MIN, v)))
    return out


def _clean_angle(value, low: float, high: float, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return min(high, max(low, v))


class Floorplan:
    """平面图点位映射表。所有读操作都在锁内，支持热重载。"""

    def __init__(self, path: Path | str = DEFAULT_CAMERA_MAP_FILE,
                 image_url: str = DEFAULT_FLOORPLAN_IMAGE):
        self.path = Path(path)
        self.image_url = image_url
        self._lock = threading.RLock()
        self._cameras: dict[str, dict] = {}
        self._mtime = 0.0
        self.reload()

    # ---------------- 加载 ---------------- #

    def reload(self) -> dict[str, dict]:
        """重新读盘。文件损坏时保留上一次的有效映射，不让服务降级。"""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            with self._lock:
                self._cameras = {}
            logger.warning("camera_map.json 不存在，平面图功能降级：%s", self.path)
            return {}
        except Exception:
            logger.exception("camera_map.json 解析失败，沿用上一次映射")
            with self._lock:
                return dict(self._cameras)

        if not isinstance(raw, dict):
            logger.error("camera_map.json 顶层必须是对象，实际是 %s", type(raw).__name__)
            with self._lock:
                return dict(self._cameras)

        cleaned: dict[str, dict] = {}
        for cam_id, item in raw.items():
            if not isinstance(cam_id, str) or not cam_id:
                continue
            if not isinstance(item, dict):
                continue
            cleaned[cam_id] = {
                "plan_id": str(item.get("plan_id") or ""),
                "desc": str(item.get("desc") or ""),
                "map_xy": _clean_coord(item.get("map_xy")),
                "facing_deg": _clean_angle(item.get("facing_deg"), _FACING_MIN, _FACING_MAX, 0.0),
                "fov_deg": _clean_angle(item.get("fov_deg"), _FOV_MIN, _FOV_MAX, 80.0),
            }

        with self._lock:
            self._cameras = cleaned
            try:
                self._mtime = self.path.stat().st_mtime
            except OSError:
                self._mtime = 0.0
        return cleaned

    def maybe_reload(self) -> None:
        """mtime 变化时热重载。手改 camera_map.json 无需重启。"""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        with self._lock:
            stale = mtime != self._mtime
        if stale:
            self.reload()

    # ---------------- 查询 ---------------- #

    def point(self, cam_id: str) -> list[float] | None:
        """返回该相机的 [x, y] 归一化坐标；未标注返回 None。"""
        if not cam_id:
            return None
        with self._lock:
            item = self._cameras.get(cam_id)
            return list(item["map_xy"]) if item and item.get("map_xy") else None

    def meta(self, cam_id: str) -> dict | None:
        with self._lock:
            item = self._cameras.get(cam_id)
            return dict(item) if item else None

    def mapped_count(self) -> int:
        with self._lock:
            return sum(1 for c in self._cameras.values() if c.get("map_xy"))

    def total_count(self) -> int:
        with self._lock:
            return len(self._cameras)

    # ---------------- 输出 ---------------- #

    def payload(self, camera_ids: set[str] | None = None) -> dict:
        """给前端的完整平面图包。camera_ids 用于过滤掉已摘除的相机。"""
        self.maybe_reload()
        with self._lock:
            cameras = dict(self._cameras)
        if camera_ids is not None:
            cameras = {k: v for k, v in cameras.items() if k in camera_ids}
        mapped = sum(1 for v in cameras.values() if v.get("map_xy"))
        return {
            "image": self.image_url,
            "image_size": list(FLOORPLAN_IMAGE_SIZE),
            "total": len(cameras),
            "mapped": mapped,
            "complete": mapped > 0 and mapped == len(cameras),
            "cameras": cameras,
        }


def build_floorplan(path: Path | str = DEFAULT_CAMERA_MAP_FILE,
                    image_url: str = DEFAULT_FLOORPLAN_IMAGE) -> Floorplan:
    """工厂函数。避免在模块级实例化产生 import 副作用。"""
    return Floorplan(path=path, image_url=image_url)
