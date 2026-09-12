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
from .reid_config import REID_MATCH_THRESHOLD, REID_RATIO_TEST

# 每个身份最多保留最近 N 条出现记录，防止长时间运行后内存耗尽
_MAX_APPEARANCES = 200
_FEATURE_SCHEMA_VERSION = 1
# 特征列（feature_blob / feature_bank_blob）落盘节流阈值：
# 特征是滑动平均结果，单次 appearance 的变化极小，没必要每次都重写整行 BLOB。
# 同一 gid 满足「累计 M 次特征更新」或「距上次落盘超过 N 秒」之一即整行落盘，
# 其余时刻只插一条 identity_appearances 增量行 + 更新三个轻量列。
_FEATURE_FLUSH_EVERY_UPDATES = 50
_FEATURE_FLUSH_INTERVAL_SECONDS = 30.0

# 特征塌缩护栏（2026-09-12 新增）：
# 新身份与已有身份的相似度超过这个值时，几乎不可能是两个人 —— 说明特征空间已退化到
# 无法区分个体。此时必须告警而不是静默新建身份，否则身份表会无声膨胀。
# 判据取 0.999 而不是更低的值：比这更低的相似度有可能是"同一个人换了衣服"，
# 需要靠阈值标定解决，不能一概报警。
_COLLAPSE_SIMILARITY_ALERT = 0.999
_COLLAPSE_WARN_INTERVAL = 60.0   # 告警日志节流间隔（秒）
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
        # gid -> [距上次特征落盘的更新次数, 上次特征落盘时间]，只在持有 self._lock 时读写
        self._feature_flush_state: dict[str, list] = {}
        # 特征塌缩护栏计数（见 _COLLAPSE_SIMILARITY_ALERT）
        self._collapse_warnings = 0
        self._collapse_warned_at = 0.0
        self.metrics = ReIDMetrics()
        if self._database is not None:
            self._restore()

    def _restore(self) -> None:
        restored = 0
        space_mismatch = 0
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
                    # 逐条打 warning 会刷屏（换权重后旧身份是**全部**被跳过），
                    # 因此改为计数 + 末尾汇总一条可读的提示。
                    space_mismatch += 1
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
        if space_mismatch:
            logger.warning(
                "跳过 %d 个身份：特征空间与当前 %r 不一致（当前库共 %d 条）。"
                "**这是换 ReID 权重后的预期行为** —— 旧特征由别的权重产生，"
                "与新特征不可比，必须重新注册。不是数据丢失：identity_appearances "
                "里的轨迹仍在，可继续用于检索与统计。",
                space_mismatch, self._feature_space, len(items),
            )

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

    def _persist_payload(self, rec: PersonRecord) -> dict:
        """
        在锁内采集整行落盘所需的载荷。只复制特征向量（tobytes/stack 本身即拷贝），
        不深拷贝最多 200 条的 appearances deque，避免每次持久化都产生大对象。
        """
        feature = np.asarray(rec.feature, dtype=np.float32)
        bank = np.stack(rec.feature_bank).astype(np.float32, copy=False)
        if rec.appearances:
            first_seen = float(rec.appearances[0].get("time", rec.last_seen))
        elif rec.last_seen > 0:
            first_seen = rec.last_seen
        else:
            # 刚注册、还没有任何 appearance：用当前时间，避免写入 0（1970 年）
            first_seen = time.time()
        return {
            "global_id": rec.global_id,
            "feature_dim": int(feature.size),
            "feature_blob": feature.tobytes(),
            "feature_bank_count": len(bank),
            "feature_bank_blob": bank.tobytes(),
            "total_appearances": rec.total_appearances,
            "last_camera": rec.last_camera,
            "last_seen": rec.last_seen,
            "first_seen": first_seen,
        }

    def _write_identity_row(self, payload: dict, new_appearance: dict | None = None) -> None:
        """整行落盘（含特征列）。payload 必须由 _persist_payload 在锁内采集。"""
        if self._database is None:
            return
        self._database.save_identity(
            global_id=payload["global_id"],
            feature_dim=payload["feature_dim"],
            feature_blob=payload["feature_blob"],
            feature_bank_count=payload["feature_bank_count"],
            feature_bank_blob=payload["feature_bank_blob"],
            total_appearances=payload["total_appearances"],
            last_camera=payload["last_camera"],
            last_seen=payload["last_seen"],
            feature_space=self._feature_space,
            first_seen=payload["first_seen"],
            new_appearance=new_appearance,
            schema_version=_FEATURE_SCHEMA_VERSION,
        )

    def _should_persist_feature_locked(self, global_id: str, now: float) -> bool:
        """特征列节流判定（必须在持有 self._lock 时调用）"""
        state = self._feature_flush_state.setdefault(global_id, [0, now])
        state[0] += 1
        if (
            state[0] >= _FEATURE_FLUSH_EVERY_UPDATES
            or now - state[1] >= _FEATURE_FLUSH_INTERVAL_SECONDS
        ):
            state[0] = 0
            state[1] = now
            return True
        return False

    def flush(self) -> int:
        """
        把节流期内尚未落盘的特征列补写进 SQLite，返回补写的身份数。
        进程退出前必须调用（main.py 关停路径），否则最近一段滑动平均会丢失。
        """
        if self._database is None:
            return 0
        with self._lock:
            payloads = []
            now = time.time()
            for gid, state in self._feature_flush_state.items():
                if state[0] <= 0:
                    continue
                state[0] = 0
                state[1] = now
                rec = self._records.get(gid)
                if rec is not None:
                    payloads.append(self._persist_payload(rec))
        for payload in payloads:
            self._write_identity_row(payload)
        if payloads:
            logger.info("已补写 %d 个身份的 ReID 特征到 SQLite", len(payloads))
        return len(payloads)

    def _make_room_locked(self) -> list[str]:
        if len(self._records) < self._max_records:
            return []
        victim = min(
            self._records.values(),
            key=lambda record: record.last_seen,
        )
        self._records.pop(victim.global_id, None)
        self._feature_flush_state.pop(victim.global_id, None)
        return [victim.global_id]

    def _delete_persisted(self, global_ids: list[str]) -> None:
        if self._database is not None and global_ids:
            self._database.delete_identities(global_ids)

    # ------------------------------------------------------------------ #
    # 查询                                                                  #
    # ------------------------------------------------------------------ #

    def get_gallery(self) -> list[tuple[str, np.ndarray]]:
        """返回所有身份的 (global_id, 主特征) 列表 —— 严格一身份一行。

        注意：主特征是 EMA 滑动平均，会指数抹平身份特异残差
        （见 `scripts/diagnose_ema_collapse.py`）。**匹配请用 `get_match_gallery()`**，
        它会把每个身份的 feature_bank 一起展开，由 `match_feature_detailed()`
        按身份取 max，从而绕开 EMA 塌缩。本方法保留给"一身份一向量"的展示型用途。
        """
        with self._lock:
            return [(gid, rec.feature.copy()) for gid, rec in self._records.items()]

    def get_match_gallery(self) -> list[tuple[str, np.ndarray]]:
        """
        返回用于匹配的 (global_id, feature) 行 —— 同一身份可能多行。

        每行是主特征或 feature_bank 中的一个姿态特征。`feature_bank` 存的是**原始**
        特征且带多样性约束（与已有 bank 相似度 <0.92 才入池），因此不受 EMA 塌缩影响；
        取 max 可显著提升同一人跨姿态/跨镜头的召回。

        规模提示：行数 = 身份数 × (1 + bank 大小) ≤ 6 倍身份数。当前库只有几十个身份，
        开销可忽略；若将来 `LAB_MONITOR_MAX_IDENTITIES` 调到万级，需要改为
        预建特征矩阵或走 ANN 索引，否则每次匹配都要重算 6 万行。
        """
        with self._lock:
            return [
                (gid, feat.copy())
                for gid, rec in self._records.items()
                for feat in [rec.feature, *rec.feature_bank]
            ]

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
            self._feature_flush_state[gid] = [0, time.time()]
            payload = self._persist_payload(rec)
        self._write_identity_row(payload)
        self._delete_persisted(evicted)
        return gid

    def register_if_new(
        self,
        feature: np.ndarray,
        threshold: float = REID_MATCH_THRESHOLD,
        ratio: float = REID_RATIO_TEST,
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
                # 展开每个身份的 [主特征, *feature_bank]，交给 match_feature_detailed
                # 按身份去重（取 max）后再做阈值/Ratio 判定。
                # 这里的去重是**语义要求**而不是优化：不去重时同一身份会同时占据
                # best 与 second，Ratio Test 必然判"歧义"→ 调用方新建身份 →
                # 身份表无限膨胀。库内现存的 45 个身份里有 27 个分属 7 组、
                # 每组特征字节完全相同，就是这么长出来的。
                gallery = [
                    (global_id, feat)
                    for global_id, record in self._records.items()
                    for feat in [record.feature, *record.feature_bank]
                ]
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

                # 护栏必须放在 ambiguous 分支**之前**：塌缩的典型表现正是
                # "与已有身份几乎完全一致，却因 Ratio Test 判歧义而拒绝归并"，
                # 若放在 ambiguous 的 return 之后，这条护栏永远执行不到。
                #
                # 后果不是"错认"，而是"永远认不出"：本 track 拿不到 global_id，
                # 调用方（pipeline.py:527）只打一行 debug 就继续，攒满的 8 帧
                # ReID 缓冲被白白丢弃，下一帧从头再攒 —— 人员反复进出、身份表
                # 却不再增长，而库里 27 个身份分属 7 组、特征字节完全相同，
                # 就是这个状态留下的痕迹。
                #
                # 判据取 0.999 而不是更低：更低的相似度可能是"同一个人换了衣服"，
                # 属于阈值标定问题，不能一概报警。日志做 60 秒节流，避免每帧刷屏。
                if detail.best_sim >= _COLLAPSE_SIMILARITY_ALERT:
                    self._collapse_warnings += 1
                    now = time.time()
                    if now - self._collapse_warned_at >= _COLLAPSE_WARN_INTERVAL:
                        self._collapse_warned_at = now
                        logger.error(
                            "ReID 特征塌缩疑似：查询与已有身份相似度 %.4f（≥%.3f）"
                            "却被判为歧义，无法归并。累计 %d 次。"
                            "根因见 scripts/diagnose_ema_collapse.py（主特征的 EMA "
                            "滑动平均抹平身份特异残差）。修复方向见 "
                            "docs/TODO_2026-09-12_worklist.md 项 1.6。"
                            "在该问题解决前，身份识别结果不应采信。",
                            detail.best_sim, _COLLAPSE_SIMILARITY_ALERT,
                            self._collapse_warnings,
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
            self._feature_flush_state[gid] = [0, time.time()]
            payload = self._persist_payload(rec)
        self._write_identity_row(payload)
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
        持久化只走增量路径：整行（含特征 BLOB）按 _FEATURE_FLUSH_* 节流写入。
        """
        quality = min(1.0, max(0.0, quality_score))
        self.metrics.record_quality(quality)
        alpha = base_alpha + (1.0 - base_alpha) * (1.0 - quality)
        feat_copy = feature.copy()
        payload = None
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
            new_appearance = dict(rec.appearances[-1])
            total_appearances = rec.total_appearances
            if self._should_persist_feature_locked(global_id, rec.last_seen):
                payload = self._persist_payload(rec)
        if self._database is None:
            return
        if payload is not None:
            # 到期：整行落盘，顺带把这条 appearance 一并插入（同一事务）
            self._write_identity_row(payload, new_appearance=new_appearance)
            return
        recorded = self._database.record_appearance(
            global_id=global_id,
            camera_id=new_appearance["camera"],
            timestamp=new_appearance["time"],
            bbox=new_appearance["bbox"],
            total_appearances=total_appearances,
        )
        if not recorded:
            # identities 行缺失（例如被保留期清理掉）：退回整行写入，避免特征永久丢失
            self._persist_full(global_id)

    def _persist_full(self, global_id: str) -> None:
        """重新采集指定身份的特征载荷并整行落盘（轻量路径的兜底）"""
        if self._database is None:
            return
        with self._lock:
            rec = self._records.get(global_id)
            payload = self._persist_payload(rec) if rec is not None else None
        if payload is not None:
            self._write_identity_row(payload)

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._records.keys())

    def get_metrics(self) -> dict:
        with self._lock:
            g_size = len(self._records)
            collapse = self._collapse_warnings
        summary = self.metrics.get_summary(gallery_size=g_size)
        # 塌缩告警次数外露到 /api/metrics/reid 与 /api/system/metrics：
        # 这个失败模式此前完全不可观测（只能靠人肉发现"身份数在慢慢变多"）。
        summary["collapse_warnings"] = collapse
        return summary
