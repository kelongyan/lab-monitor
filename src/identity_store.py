"""
identity_store.py — 线程安全的全局身份库
维护每个 global_id 对应的特征向量（滑动平均）和出现历史
"""

import threading
import uuid
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field

# 每个身份最多保留最近 N 条出现记录，防止长时间运行后内存耗尽
_MAX_APPEARANCES = 200


@dataclass
class PersonRecord:
    global_id: str
    feature: np.ndarray          # 平均特征向量（L2归一化）
    appearances: deque           # deque(maxlen=_MAX_APPEARANCES)，自动丢弃旧记录
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
                appearances=deque(maxlen=_MAX_APPEARANCES),
            )
        return gid

    def register_if_new(
        self,
        feature: np.ndarray,
        threshold: float = 0.75,
    ) -> tuple[str, bool]:
        """
        原子性查重+注册（P1-5）：在持有锁的情况下，先查询是否已有相似身份。
        - 有匹配 → 返回 (existing_id, False)，不重复注册
        - 无匹配 → 在锁内注册并返回 (new_id, True)
        防止多路 pipeline 并发时为同一人生成多个 global_id。
        """
        with self._lock:
            for gid, rec in self._records.items():
                sim = float(np.dot(feature, rec.feature))  # 均已 L2 归一化
                if sim >= threshold:
                    return gid, False
            # 未找到相似身份，在锁内注册，保证原子性
            gid = str(uuid.uuid4())[:8]
            self._records[gid] = PersonRecord(
                global_id=gid,
                feature=feature.copy(),
                appearances=deque(maxlen=_MAX_APPEARANCES),
            )
            return gid, True

    def update_appearance(
        self,
        global_id: str,
        camera_id: str,
        feature: np.ndarray,
        bbox: list[float],
        quality_score: float = 1.0,   # [0,1]，由 pipeline 传入，基于 bbox 面积+置信度
        base_alpha: float = 0.85,     # 基础衰减系数（质量满分时使用）
    ) -> None:
        """
        记录出现事件，并用质量加权的滑动平均更新特征向量（P1-3）。
        quality 高（目标大/清晰）→ alpha 小 → 积极吸收新特征
        quality 低（目标小/模糊）→ alpha 趋近 1 → 保守更新，避免噪声污染
        """
        quality = min(1.0, max(0.0, quality_score))
        alpha = base_alpha + (1.0 - base_alpha) * (1.0 - quality)
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
