"""
identity_store.py — 线程安全的全局身份库
维护每个 global_id 对应的特征向量（滑动平均）和出现历史
"""

import threading
import uuid
import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field

from .reid import match_feature_detailed

# 每个身份最多保留最近 N 条出现记录，防止长时间运行后内存耗尽
_MAX_APPEARANCES = 200
_FEATURE_SCHEMA_VERSION = 1
logger = logging.getLogger("identity_store")


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
        matched: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        with self._lock:
            if is_ratio_blocked:
                self.ratio_blocked_count += 1
            if matched:
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
    total_appearances: int = 0
    last_camera: str = ""
    last_seen: float = 0.0


@dataclass(frozen=True)
class IdentityResolution:
    global_id: str | None
    status: str
    best_similarity: float = 0.0
    second_similarity: float = 0.0

    @property
    def is_new(self) -> bool:
        return self.status == "created"


class IdentityStore:
    """线程安全的全局人员身份库"""

    def __init__(
        self,
        database=None,
        feature_space: str = "unspecified",
        max_records: int = 10000,
    ):
        self._lock = threading.Lock()
        self._records: dict[str, PersonRecord] = {}
        self._database = database
        self._feature_space = feature_space
        self._max_records = max(1, max_records)
        self._feature_dim: int | None = None
        self.metrics = ReIDMetrics()
        if self._database is not None:
            self._restore()

    def _restore(self) -> None:
        restored = 0
        items = sorted(
            self._database.load_identities(),
            key=lambda item: item.get("last_seen", 0.0),
            reverse=True,
        )[:self._max_records]
        for item in items:
            try:
                if item["schema_version"] != _FEATURE_SCHEMA_VERSION:
                    logger.warning(
                        "跳过身份 %s：特征版本 %s 不受支持",
                        item["global_id"], item["schema_version"],
                    )
                    continue
                if item["feature_space"] != self._feature_space:
                    logger.warning(
                        "跳过身份 %s：特征空间 %r 与当前 %r 不一致",
                        item["global_id"], item["feature_space"], self._feature_space,
                    )
                    continue
                dim = int(item["feature_dim"])
                if dim <= 0 or len(item["feature_blob"]) != dim * 4:
                    raise ValueError("主特征字节长度不匹配")
                if self._feature_dim is not None and dim != self._feature_dim:
                    logger.warning(
                        "跳过身份 %s：特征维度 %d 与当前库 %d 不一致",
                        item["global_id"], dim, self._feature_dim,
                    )
                    continue
                feature = np.frombuffer(
                    item["feature_blob"], dtype=np.float32
                ).copy()
                bank_count = int(item["feature_bank_count"])
                bank_blob = item["feature_bank_blob"]
                if bank_count:
                    if len(bank_blob) != bank_count * dim * 4:
                        raise ValueError("Feature Bank 字节长度不匹配")
                    bank_matrix = np.frombuffer(
                        bank_blob, dtype=np.float32
                    ).reshape(bank_count, dim)
                    feature_bank = [row.copy() for row in bank_matrix]
                else:
                    feature_bank = [feature.copy()]
                appearances = deque(
                    (dict(entry) for entry in item["appearances"]),
                    maxlen=_MAX_APPEARANCES,
                )
                self._records[item["global_id"]] = PersonRecord(
                    global_id=item["global_id"],
                    feature=feature,
                    feature_bank=feature_bank,
                    appearances=appearances,
                    total_appearances=int(item["total_appearances"]),
                    last_camera=item["last_camera"],
                    last_seen=float(item["last_seen"]),
                )
                self._feature_dim = dim
                restored += 1
            except (KeyError, TypeError, ValueError) as error:
                logger.warning(
                    "跳过损坏的身份持久化记录 %s: %s",
                    item.get("global_id", "<unknown>"), error,
                )
        if restored:
            logger.info("已从 SQLite 恢复 %d 个 ReID 身份", restored)

    @staticmethod
    def _snapshot(rec: PersonRecord) -> PersonRecord:
        return PersonRecord(
            global_id=rec.global_id,
            feature=rec.feature.copy(),
            feature_bank=[feature.copy() for feature in rec.feature_bank],
            appearances=deque(
                (dict(entry) for entry in rec.appearances),
                maxlen=_MAX_APPEARANCES,
            ),
            total_appearances=rec.total_appearances,
            last_camera=rec.last_camera,
            last_seen=rec.last_seen,
        )

    def _persist(self, rec: PersonRecord, new_appearance: dict | None = None) -> None:
        if self._database is None:
            return
        feature = np.asarray(rec.feature, dtype=np.float32)
        bank = np.stack(rec.feature_bank).astype(np.float32, copy=False)
        self._database.save_identity(
            global_id=rec.global_id,
            feature_dim=int(feature.size),
            feature_blob=feature.tobytes(),
            feature_bank_count=len(bank),
            feature_bank_blob=bank.tobytes(),
            appearances=list(rec.appearances),
            total_appearances=rec.total_appearances,
            last_camera=rec.last_camera,
            last_seen=rec.last_seen,
            feature_space=self._feature_space,
            new_appearance=new_appearance,
            schema_version=_FEATURE_SCHEMA_VERSION,
        )

    def _make_room_locked(self) -> list[str]:
        if len(self._records) < self._max_records:
            return []
        victim = min(
            self._records.values(),
            key=lambda record: record.last_seen,
        )
        self._records.pop(victim.global_id, None)
        return [victim.global_id]

    def _delete_persisted(self, global_ids: list[str]) -> None:
        if self._database is not None and global_ids:
            self._database.delete_identities(global_ids)

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
            return self._snapshot(rec) if rec is not None else None

    def get_last_bbox(self, global_id: str) -> list[float]:
        """在锁内安全读取最后一次出现的 bbox，避免调用方在锁外访问可变的 appearances deque"""
        with self._lock:
            rec = self._records.get(global_id)
            if rec is None or not rec.appearances:
                return []
            return list(rec.appearances[-1].get("bbox", []))

    # ------------------------------------------------------------------ #
    # 更新                                                                  #
    # ------------------------------------------------------------------ #

    def register(self, feature: np.ndarray) -> str:
        """注册新身份，返回新 global_id"""
        feat_copy = np.asarray(feature, dtype=np.float32).copy()
        if feat_copy.ndim != 1 or np.linalg.norm(feat_copy) <= 1e-8:
            raise ValueError("身份特征必须是一维非零向量")
        feat_copy /= np.linalg.norm(feat_copy)
        with self._lock:
            if self._feature_dim is not None and feat_copy.size != self._feature_dim:
                raise ValueError("身份特征维度与当前特征库不一致")
            self._feature_dim = int(feat_copy.size)
            evicted = self._make_room_locked()
            gid = str(uuid.uuid4())[:8]
            rec = PersonRecord(
                global_id=gid,
                feature=feat_copy,
                feature_bank=[feat_copy.copy()],
                appearances=deque(maxlen=_MAX_APPEARANCES),
            )
            self._records[gid] = rec
            snapshot = self._snapshot(rec)
        self._persist(snapshot)
        self._delete_persisted(evicted)
        return gid

    def register_if_new(
        self,
        feature: np.ndarray,
        threshold: float = 0.75,
        ratio: float = 0.85,
    ) -> IdentityResolution:
        """
        原子性向量化查重+注册：
        在持有锁的情况下，利用矩阵乘法一次性计算 query 特征与所有记录主特征/特征库的相似度。
        返回 matched、created 或 ambiguous，歧义结果不会静默归并 Top-1。
        """
        feat_copy = feature.copy()
        norm = np.linalg.norm(feat_copy)
        if norm <= 1e-8:
            return IdentityResolution(global_id=None, status="invalid")
        feat_copy /= norm
        with self._lock:
            if self._feature_dim is not None and feat_copy.size != self._feature_dim:
                return IdentityResolution(global_id=None, status="invalid")
            if self._records:
                gallery = []
                for global_id, record in self._records.items():
                    candidates = [record.feature, *record.feature_bank]
                    best_feature = max(
                        candidates,
                        key=lambda candidate: float(candidate @ feat_copy),
                    )
                    gallery.append((global_id, best_feature))
                detail = match_feature_detailed(
                    feat_copy,
                    gallery,
                    threshold=threshold,
                    ratio=ratio,
                )
                if detail.matched_id is not None:
                    return IdentityResolution(
                        global_id=detail.matched_id,
                        status="matched",
                        best_similarity=detail.best_sim,
                        second_similarity=detail.second_sim,
                    )
                if detail.is_ratio_blocked:
                    return IdentityResolution(
                        global_id=None,
                        status="ambiguous",
                        best_similarity=detail.best_sim,
                        second_similarity=detail.second_sim,
                    )

            # 未找到相似身份，在锁内注册，保证原子性
            evicted = self._make_room_locked()
            gid = str(uuid.uuid4())[:8]
            rec = PersonRecord(
                global_id=gid,
                feature=feat_copy,
                feature_bank=[feat_copy.copy()],
                appearances=deque(maxlen=_MAX_APPEARANCES),
            )
            self._records[gid] = rec
            self._feature_dim = int(feat_copy.size)
            snapshot = self._snapshot(rec)
        self._persist(snapshot)
        self._delete_persisted(evicted)
        return IdentityResolution(global_id=gid, status="created")

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
        snapshot = None
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
            if quality > 0.6:
                bank_matrix = np.stack(rec.feature_bank)
                bank_sims = bank_matrix @ feat_copy
                max_bank_sim = float(np.max(bank_sims))
                # 新特征与已有 bank 差异明显（< 0.92）才值得加入，避免重复存储
                if max_bank_sim < 0.92:
                    if len(rec.feature_bank) < 5:
                        rec.feature_bank.append(feat_copy)
                    else:
                        # Bank 已满：淘汰与新特征最相似（最冗余）的那个，引入新角度
                        redundant_idx = int(np.argmax(bank_sims))
                        rec.feature_bank[redundant_idx] = feat_copy

            rec.last_camera = camera_id
            rec.last_seen = time.time()
            rec.appearances.append({
                "camera": camera_id,
                "time": rec.last_seen,
                "bbox": list(bbox),
            })
            rec.total_appearances += 1
            snapshot = self._snapshot(rec)
        self._persist(snapshot, new_appearance=dict(snapshot.appearances[-1]))

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())

    def get_metrics(self) -> dict:
        with self._lock:
            g_size = len(self._records)
        return self.metrics.get_summary(gallery_size=g_size)
