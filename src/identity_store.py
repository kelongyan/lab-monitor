"""
identity_store.py — 线程安全的全局身份库
维护每个 global_id 对应的特征向量（滑动平均）和出现历史
"""

import threading
import uuid
import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PersonRecord:
    global_id: str
    feature: np.ndarray          # 平均特征向量（L2归一化）
    appearances: list[dict]      # [{"camera": str, "time": float, "bbox": list}, ...]
    last_camera: str = ""
    last_seen: float = 0.0


class IdentityStore:
    """线程安全的全局人员身份库"""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: dict[str, PersonRecord] = {}

    # ------------------------------------------------------------------ #
    # 查询                                                                  #
    # ------------------------------------------------------------------ #

    def get_gallery(self) -> list[tuple[str, np.ndarray]]:
        """返回所有身份的 (global_id, feature) 列表，供 ReID 匹配用"""
        with self._lock:
            return [(gid, rec.feature.copy()) for gid, rec in self._records.items()]

    def get(self, global_id: str) -> PersonRecord | None:
        with self._lock:
            rec = self._records.get(global_id)
            return rec  # 调用方需在锁外使用，feature 是 ndarray 不可变即可

    # ------------------------------------------------------------------ #
    # 更新                                                                  #
    # ------------------------------------------------------------------ #

    def register(self, feature: np.ndarray) -> str:
        """注册新身份，返回新 global_id"""
        gid = str(uuid.uuid4())[:8]
        with self._lock:
            self._records[gid] = PersonRecord(
                global_id=gid,
                feature=feature.copy(),
                appearances=[],
            )
        return gid

    def update_appearance(
        self,
        global_id: str,
        camera_id: str,
        feature: np.ndarray,
        bbox: list[float],
        alpha: float = 0.9,    # 滑动平均系数：偏向旧特征
    ) -> None:
        """
        记录出现事件，并用滑动平均更新特征向量
        alpha=0.9 表示 90% 旧特征 + 10% 新特征
        """
        with self._lock:
            rec = self._records.get(global_id)
            if rec is None:
                return
            rec.feature = alpha * rec.feature + (1 - alpha) * feature
            norm = np.linalg.norm(rec.feature)
            if norm > 1e-8:
                rec.feature /= norm
            rec.last_camera = camera_id
            rec.last_seen = time.time()
            rec.appearances.append({
                "camera": camera_id,
                "time": rec.last_seen,
                "bbox": bbox,
            })

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())
