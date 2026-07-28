import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path


class TopologyValidationError(ValueError):
    """Raised when a topology payload does not match the accepted schema."""


@dataclass(frozen=True)
class NextHop:
    camera_id: str
    expected_seconds: float
    tolerance_seconds: float = 15.0

    @property
    def deadline_window(self) -> tuple[float, float]:
        """Return the earliest and latest expected arrival offsets."""
        lo = max(0.0, self.expected_seconds - self.tolerance_seconds)
        hi = self.expected_seconds + self.tolerance_seconds
        return lo, hi


class CameraTopology:
    _MAX_CAMERA_ID_LENGTH = 64
    _MAX_HOPS_PER_CAMERA = 64
    _MAX_SECONDS = 365 * 24 * 60 * 60

    def __init__(
        self,
        config_path: str | Path,
        allowed_camera_ids: set[str] | None = None,
    ):
        self._lock = threading.RLock()
        self._config_path = Path(config_path)
        self._allowed_camera_ids = (
            frozenset(allowed_camera_ids) if allowed_camera_ids is not None else None
        )
        self._graph: dict[str, list[NextHop]] = {}
        self._load(self._config_path)

    @property
    def allowed_camera_ids(self) -> frozenset[str] | None:
        return self._allowed_camera_ids

    @classmethod
    def _validate_camera_id(
        cls,
        value: object,
        allowed_camera_ids: frozenset[str] | None,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TopologyValidationError(f"{field_name} 必须是字符串")
        if not value or value != value.strip():
            raise TopologyValidationError(f"{field_name} 不能为空或包含首尾空白")
        if len(value) > cls._MAX_CAMERA_ID_LENGTH:
            raise TopologyValidationError(
                f"{field_name} 长度不能超过 {cls._MAX_CAMERA_ID_LENGTH}"
            )
        if any(ord(ch) < 32 or ch in "<>\"'`" for ch in value):
            raise TopologyValidationError(f"{field_name} 包含非法字符")
        if allowed_camera_ids is not None and value not in allowed_camera_ids:
            raise TopologyValidationError(f"未知摄像头 ID: {value}")
        return value

    @classmethod
    def _validate_seconds(
        cls,
        value: object,
        field_name: str,
        *,
        allow_zero: bool,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TopologyValidationError(f"{field_name} 必须是数字")
        number = float(value)
        minimum = 0.0 if allow_zero else 0.0
        if not math.isfinite(number) or number < minimum or (not allow_zero and number == 0):
            condition = "大于等于 0" if allow_zero else "大于 0"
            raise TopologyValidationError(f"{field_name} 必须是有限且{condition}的数字")
        if number > cls._MAX_SECONDS:
            raise TopologyValidationError(
                f"{field_name} 不能超过 {cls._MAX_SECONDS} 秒"
            )
        return number

    @classmethod
    def _build_graph(
        cls,
        raw_data: object,
        allowed_camera_ids: frozenset[str] | None,
    ) -> tuple[dict[str, list[NextHop]], dict[str, list[dict[str, float | str]]]]:
        if not isinstance(raw_data, dict):
            raise TopologyValidationError("数据格式必须为 JSON 对象")

        graph: dict[str, list[NextHop]] = {}
        normalized: dict[str, list[dict[str, float | str]]] = {}
        seen_edges: set[tuple[str, str]] = set()

        for raw_camera_id, raw_hops in raw_data.items():
            camera_id = cls._validate_camera_id(
                raw_camera_id, allowed_camera_ids, "起始摄像头 ID"
            )
            if not isinstance(raw_hops, list):
                raise TopologyValidationError(f"{camera_id} 的通道配置必须是数组")
            if len(raw_hops) > cls._MAX_HOPS_PER_CAMERA:
                raise TopologyValidationError(
                    f"{camera_id} 的通道数不能超过 {cls._MAX_HOPS_PER_CAMERA}"
                )

            graph[camera_id] = []
            normalized[camera_id] = []
            for index, raw_hop in enumerate(raw_hops):
                if not isinstance(raw_hop, dict):
                    raise TopologyValidationError(
                        f"{camera_id} 第 {index + 1} 条通道必须是 JSON 对象"
                    )
                unknown_fields = set(raw_hop) - {
                    "next", "expected_seconds", "tolerance_seconds"
                }
                if unknown_fields:
                    fields = ", ".join(sorted(map(str, unknown_fields)))
                    raise TopologyValidationError(f"拓扑通道包含未知字段: {fields}")
                if "next" not in raw_hop or "expected_seconds" not in raw_hop:
                    raise TopologyValidationError(
                        f"{camera_id} 第 {index + 1} 条通道缺少 next 或 expected_seconds"
                    )

                next_camera_id = cls._validate_camera_id(
                    raw_hop["next"], allowed_camera_ids, "目标摄像头 ID"
                )
                if next_camera_id == camera_id:
                    raise TopologyValidationError("拓扑不允许摄像头自循环")
                edge = (camera_id, next_camera_id)
                if edge in seen_edges:
                    raise TopologyValidationError(
                        f"拓扑通道重复: {camera_id} -> {next_camera_id}"
                    )
                seen_edges.add(edge)

                expected_seconds = cls._validate_seconds(
                    raw_hop["expected_seconds"], "expected_seconds", allow_zero=False
                )
                tolerance_seconds = cls._validate_seconds(
                    raw_hop.get("tolerance_seconds", 15.0),
                    "tolerance_seconds",
                    allow_zero=True,
                )
                hop = NextHop(next_camera_id, expected_seconds, tolerance_seconds)
                graph[camera_id].append(hop)
                normalized[camera_id].append({
                    "next": next_camera_id,
                    "expected_seconds": expected_seconds,
                    "tolerance_seconds": tolerance_seconds,
                })

        return graph, normalized

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open(encoding="utf-8") as file:
            raw = json.load(file)
        graph, _ = self._build_graph(raw, self._allowed_camera_ids)
        with self._lock:
            self._graph = graph

    def next_hops(self, camera_id: str) -> list[NextHop]:
        with self._lock:
            return list(self._graph.get(camera_id, []))

    def all_cameras(self) -> list[str]:
        with self._lock:
            return list(self._graph.keys())

    def edges(self) -> set[tuple[str, str]]:
        with self._lock:
            return {
                (camera_id, hop.camera_id)
                for camera_id, hops in self._graph.items()
                for hop in hops
            }

    def to_dict(self) -> dict[str, list[dict[str, float | str]]]:
        with self._lock:
            return {
                camera_id: [
                    {
                        "next": hop.camera_id,
                        "expected_seconds": hop.expected_seconds,
                        "tolerance_seconds": hop.tolerance_seconds,
                    }
                    for hop in hops
                ]
                for camera_id, hops in self._graph.items()
            }

    def update_config(
        self,
        raw_data: object,
    ) -> dict[str, list[dict[str, float | str]]]:
        """Validate, atomically persist, then replace the in-memory graph."""
        graph, normalized = self._build_graph(raw_data, self._allowed_camera_ids)
        payload = json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"

        with self._lock:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=self._config_path.parent,
                    prefix=f".{self._config_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_file.write(payload)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                    temp_path = Path(temp_file.name)
                os.replace(temp_path, self._config_path)
                temp_path = None
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            self._graph = graph
        return normalized
