"""
topology.py — 摄像头拓扑图 + 时间窗口管理
从 config/topology.json 读取图结构，提供期望出现位置和时间窗口查询
"""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class NextHop:
    camera_id: str
    expected_seconds: float
    tolerance_seconds: float = 15.0   # 宽容窗口 ±15s

    @property
    def deadline_window(self) -> tuple[float, float]:
        """返回 (最早, 最晚) 相对时间（秒）"""
        lo = max(0.0, self.expected_seconds - self.tolerance_seconds)
        hi = self.expected_seconds + self.tolerance_seconds
        return lo, hi


class CameraTopology:
    def __init__(self, config_path: str | Path):
        self._graph: dict[str, list[NextHop]] = {}
        self._load(Path(config_path))

    def _load(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            raw: dict = json.load(f)
        for cam_id, hops in raw.items():
            self._graph[cam_id] = [
                NextHop(
                    camera_id=h["next"],
                    expected_seconds=h["expected_seconds"],
                    tolerance_seconds=h.get("tolerance_seconds", 15.0),
                )
                for h in hops
            ]

    def next_hops(self, camera_id: str) -> list[NextHop]:
        """返回从 camera_id 出发后可能到达的摄像头列表"""
        return self._graph.get(camera_id, [])

    def all_cameras(self) -> list[str]:
        return list(self._graph.keys())
