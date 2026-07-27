"""
reid_validator.py — 多帧 ReID 确认器（降低单帧误匹配）
每个 track_id 维护最近 N 帧特征缓冲区，用平均特征做匹配，
要求连续 M 帧一致才确认身份——彻底消除单帧噪声引起的误报
"""

import time
import logging
import numpy as np
from collections import defaultdict, deque

from .reid import match_feature, match_feature_detailed

logger = logging.getLogger("reid_validator")


class ReIDValidator:
    """
    使用方式：
      1. 每帧调用 add_feature(track_id, feat) 积累特征
      2. 调用 get_confirmed_match(track_id, gallery) 获取确认的身份
         - 未达到确认条件时返回 None
         - 达到条件后返回 global_id（后续每帧都会返回，直到 track 被清除）
      3. track 消失时调用 clear(track_id) 释放资源
    """

    def __init__(
        self,
        buffer_size: int = 8,      # 特征缓冲帧数
        confirm_frames: int = 3,   # 连续匹配同一 ID 多少帧才确认
        threshold: float = 0.75,   # 余弦相似度阈值
    ):
        self._buffer_size = buffer_size
        self._confirm_frames = confirm_frames
        self._threshold = threshold

        # track_id → deque of feature vectors
        self._buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        # track_id → 候选 global_id
        self._candidate: dict[int, str] = {}
        # track_id → 候选连续命中次数
        self._hit_count: dict[int, int] = defaultdict(int)
        # track_id → 已确认的 global_id（一旦确认不再变更）
        self._confirmed: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # 主接口                                                                #
    # ------------------------------------------------------------------ #

    def add_feature(self, track_id: int, feat: np.ndarray) -> None:
        """积累特征帧"""
        self._buffers[track_id].append(feat)

    def get_confirmed_match(
        self,
        track_id: int,
        gallery: list[tuple[str, np.ndarray]],
        metrics=None,
    ) -> str | None:
        """
        尝试确认身份
        - 已确认：直接返回 global_id
        - 未确认：用平均特征做匹配，连续 confirm_frames 帧一致则确认
        """
        # 已确认过直接返回
        if track_id in self._confirmed:
            return self._confirmed[track_id]

        buf = self._buffers.get(track_id)
        if not buf or len(buf) < 2:
            return None  # 特征太少，等待积累

        # 用缓冲区内所有特征的平均值做匹配（比单帧稳定得多）
        avg_feat = np.mean(np.stack(list(buf)), axis=0)
        norm = np.linalg.norm(avg_feat)
        if norm < 1e-8:
            return None
        avg_feat /= norm

        t0 = time.perf_counter()
        detail = match_feature_detailed(avg_feat, gallery, threshold=self._threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if metrics is not None:
            metrics.record_search()
            metrics.record_match(
                best_sim=detail.best_sim,
                second_sim=detail.second_sim,
                is_ratio_blocked=detail.is_ratio_blocked,
                latency_ms=latency_ms,
            )

        matched_id = detail.matched_id

        if matched_id is None:
            # 没有匹配：重置候选计数
            self._candidate.pop(track_id, None)
            self._hit_count[track_id] = 0
            return None

        # 与上次候选一致：累加命中次数
        if self._candidate.get(track_id) == matched_id:
            self._hit_count[track_id] += 1
        else:
            # 候选变了：重置
            self._candidate[track_id] = matched_id
            self._hit_count[track_id] = 1

        # 达到确认阈值
        if self._hit_count[track_id] >= self._confirm_frames:
            self._confirmed[track_id] = matched_id
            logger.info(
                "Track %d → 身份确认: %s（%d帧一致，缓冲%d帧）",
                track_id, matched_id, self._confirm_frames, len(buf),
            )
            return matched_id

        return None  # 还在积累中

    def clear(self, track_id: int) -> None:
        """track 消失时释放资源"""
        self._buffers.pop(track_id, None)
        self._candidate.pop(track_id, None)
        self._hit_count.pop(track_id, None)
        self._confirmed.pop(track_id, None)

    def confirm(self, track_id: int, global_id: str) -> None:
        """
        外部强制确认身份（P1-5）：用于新身份注册后直接设定，替代直接写 _confirmed。
        同时清理候选状态，避免留下脏数据（_hit_count / _candidate 对应的 stale 记录）。
        """
        self._confirmed[track_id] = global_id
        self._candidate.pop(track_id, None)
        self._hit_count.pop(track_id, None)

    def get_avg_feature(self, track_id: int) -> np.ndarray | None:
        """获取当前缓冲区的平均特征（供注册新身份使用）"""
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        avg = np.mean(np.stack(list(buf)), axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 1e-8 else None
