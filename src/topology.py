import json
import threading
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
        self._lock = threading.Lock()
        self._config_path = Path(config_path)
        self._graph: dict[str, list[NextHop]] = {}
        self._load(self._config_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            raw: dict = json.load(f)
        with self._lock:
            self._graph.clear()
            for cam_id, hops in raw.items():
                self._graph[cam_id] = [
                    NextHop(
                        camera_id=h["next"],
                        expected_seconds=float(h["expected_seconds"]),
                        tolerance_seconds=float(h.get("tolerance_seconds", 15.0)),
                    )
                    for h in hops
                ]

    def next_hops(self, camera_id: str) -> list[NextHop]:
        """返回从 camera_id 出发后可能到达的摄像头列表"""
        with self._lock:
            return list(self._graph.get(camera_id, []))

    def all_cameras(self) -> list[str]:
        with self._lock:
            return list(self._graph.keys())

    def to_dict(self) -> dict:
        with self._lock:
            res = {}
            for cam_id, hops in self._graph.items():
                res[cam_id] = [
                    {
                        "next": h.camera_id,
                        "expected_seconds": h.expected_seconds,
                        "tolerance_seconds": h.tolerance_seconds,
                    }
                    for h in hops
                ]
            return res

    def update_config(self, raw_data: dict) -> None:
        """更新拓扑配置并持久化保存写回 config/topology.json"""
        with self._lock:
            self._graph.clear()
            for cam_id, hops in raw_data.items():
                self._graph[cam_id] = [
                    NextHop(
                        camera_id=h["next"],
                        expected_seconds=float(h["expected_seconds"]),
                        tolerance_seconds=float(h.get("tolerance_seconds", 15.0)),
                    )
                    for h in hops
                ]

        # 格式化持久化写回文件
        self._config_path.write_text(
            json.dumps(raw_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

