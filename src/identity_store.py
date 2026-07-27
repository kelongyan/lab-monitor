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


class ReIDMetrics:
    """线程安全的 ReID 检索指标统计器"""

    def __init__(self, max_history: int = 100):
        self._lock = threading.Lock()
        self.total_searches = 0
        self.successful_matches = 0
        self.ratio_blocked_count = 0
        self.sim_history = deque(maxlen=max_history)
        self.margin_history = deque(maxlen=max_history)
        self.latency_history = deque(maxlen=max_history)
        self.quality_history = deque(maxlen=max_history)

    def record_search(self) -> None:
        with self._lock:
            self.total_searches += 1

    def record_match(
        self,
        best_sim: float,
        second_sim: float = 0.0,
        is_ratio_blocked: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        with self._lock:
            if is_ratio_blocked:
                self.ratio_blocked_count += 1
            else:
                self.successful_matches += 1
                self.sim_history.append(best_sim)
                if best_sim > 0 and second_sim > 0:
                    margin = 1.0 - (second_sim / best_sim)
                    self.margin_history.append(margin)
            if latency_ms > 0:
                self.latency_history.append(latency_ms)

    def record_quality(self, quality: float) -> None:
        with self._lock:
            self.quality_history.append(quality)

    def get_summary(self, gallery_size: int = 0) -> dict:
        with self._lock:
            avg_sim = round(float(np.mean(self.sim_history)), 4) if self.sim_history else 0.0
            avg_margin = round(float(np.mean(self.margin_history)), 4) if self.margin_history else 0.0
            avg_latency = round(float(np.mean(self.latency_history)), 2) if self.latency_history else 0.0
            avg_quality = round(float(np.mean(self.quality_history)), 4) if self.quality_history else 0.0
            match_rate = round(self.successful_matches / max(1, self.total_searches), 4)
            return {
                "gallery_size": gallery_size,
                "total_searches": self.total_searches,
                "successful_matches": self.successful_matches,
                "ratio_blocked_count": self.ratio_blocked_count,
                "match_rate": match_rate,
                "avg_top1_similarity": avg_sim,
                "avg_ratio_margin": avg_margin,
                "avg_latency_ms": avg_latency,
                "avg_feature_quality": avg_quality,
            }


@dataclass
class PersonRecord:
    global_id: str
    feature: np.ndarray          # 主平均特征向量（L2归一化）
    feature_bank: list           # 多姿态/多光照特征向量库 [np.ndarray, ...] (最多保留 5 个)
    appearances: deque           # deque(maxlen=_MAX_APPEARANCES)，自动丢弃旧记录
    last_camera: str = ""
    last_seen: float = 0.0


class IdentityStore:
    """线程安全的全局人员身份库"""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: dict[str, PersonRecord] = {}
        self.metrics = ReIDMetrics()

    # ------------------------------------------------------------------ #
    # 查询                                                                  #
    # ------------------------------------------------------------------ #

    def get_gallery(self) -> list[tuple[str, np.ndarray]]:
        """返回所有身份的 (global_id, feature) 列表，供 ReID 匹配用"""
        with self._lock:
            return [(gid, rec.feature.copy()) for gid, rec in self._records.items()]

    def get_full_gallery(self) -> list[tuple[str, list[np.ndarray]]]:
        """返回所有身份及其多姿态特征向量库，用于高精度多模态比对"""
        with self._lock:
            return [(gid, [f.copy() for f in rec.feature_bank]) for gid, rec in self._records.items()]

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
        feat_copy = feature.copy()
        with self._lock:
            self._records[gid] = PersonRecord(
                global_id=gid,
                feature=feat_copy,
                feature_bank=[feat_copy],
                appearances=deque(maxlen=_MAX_APPEARANCES),
            )
        return gid

    def register_if_new(
        self,
        feature: np.ndarray,
        threshold: float = 0.75,
    ) -> tuple[str, bool]:
        """
        原子性向量化查重+注册：
        在持有锁的情况下，利用矩阵乘法一次性计算 query 特征与所有记录主特征/特征库的相似度。
        - 有匹配 → 返回 (existing_id, False)，不重复注册
        - 无匹配 → 在锁内注册并返回 (new_id, True)
        """
        feat_copy = feature.copy()
        with self._lock:
            if self._records:
                # 提取所有主特征进行矩阵点积加速
                gids = list(self._records.keys())
                main_feats = np.stack([self._records[g].feature for g in gids]) # (N, D)
                sims = main_feats @ feature # (N,)
                max_idx = int(np.argmax(sims))
                if sims[max_idx] >= threshold:
                    return gids[max_idx], False

                # 深入检查特征库 (feature bank)
                for gid, rec in self._records.items():
                    if rec.feature_bank:
                        bank_feats = np.stack(rec.feature_bank) # (K, D)
                        bank_sims = bank_feats @ feature        # (K,)
                        if float(np.max(bank_sims)) >= threshold:
                            return gid, False

            # 未找到相似身份，在锁内注册，保证原子性
            gid = str(uuid.uuid4())[:8]
            self._records[gid] = PersonRecord(
                global_id=gid,
                feature=feat_copy,
                feature_bank=[feat_copy],
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
        同时动态维护多姿态特征向量库 (Feature Bank, max_size=5)。
        """
        quality = min(1.0, max(0.0, quality_score))
        self.metrics.record_quality(quality)
        alpha = base_alpha + (1.0 - base_alpha) * (1.0 - quality)
        feat_copy = feature.copy()
        with self._lock:
            rec = self._records.get(global_id)
            if rec is None:
                return
            
            # 1. 更新主平均特征
            rec.feature = alpha * rec.feature + (1 - alpha) * feat_copy
            norm = np.linalg.norm(rec.feature)
            if norm > 1e-8:
                rec.feature /= norm

            # 2. 动态维护多姿态特征库 (Bank)
            if quality > 0.6 and len(rec.feature_bank) < 5:
                # 若新特征与已有 bank 差异较明显 (sim < 0.92)，说明捕获到了新角度，加入 Bank
                bank_matrix = np.stack(rec.feature_bank)
                max_bank_sim = float(np.max(bank_matrix @ feat_copy))
                if max_bank_sim < 0.92:
                    rec.feature_bank.append(feat_copy)

            rec.last_camera = camera_id
            rec.last_seen = time.time()
            rec.appearances.append({
                "camera": camera_id,
                "time": rec.last_seen,
                "bbox": bbox,
            })

            # 同步更新 SQLite 数据库（方向三）
            try:
                from .db import db
                db.upsert_identity(global_id, camera_id, last_seen=rec.last_seen, appearances_count=len(rec.appearances))
            except Exception:
                pass

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())

    def get_metrics(self) -> dict:
        with self._lock:
            g_size = len(self._records)
        return self.metrics.get_summary(gallery_size=g_size)

