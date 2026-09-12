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

# 公共分量中心化（2026-09-12 新增，这是 ReID 识别失效的真正修复）
# ---------------------------------------------------------------------------
# 实测（scripts/diagnose_ema_collapse.py --center）：OSNet 无论用 ImageNet 还是
# ReID 度量学习权重，输出的单位特征都和一个**全局方向**高度共线 ——
# 全体特征的均值向量范数高达 0.80~0.83（随机方向的期望只有 ~0.1）。
# 这个分量对区分身份毫无贡献，却让跨身份余弦虚高到 0.50~0.73：
#     imagenet 原始   p50 0.4961 / 越过阈值 0.75 的异人对 21.4%
#     减去公共分量后  p50 -0.1021 / 越过阈值 0.0%
# 聚合（EMA 或窗口均值）之所以"有害"，只是因为它把这个残余的判别余量进一步挤掉
# （聚合后 p50 升到 0.73、越阈 46.4%）；**病根是公共分量，不是聚合方式**。
# 因此修复放在匹配路径上：query 与 gallery 统一减去中心向量再归一化。
# 中心取全体已注册特征（主特征 + feature_bank）的均值，零额外状态、重启后自动可得。
#
# 两个护栏是必需的：
#   · 身份太少时（特征数 < 下限）中心不可靠，且减完可能让范数趋零 —— 不启用
#   · 中心向量范数太小时说明本来就没有公共分量 —— 不启用
# 不启用时行为与改造前完全一致，因此冷启动是安全的。
_CENTER_MIN_FEATURES = 8
_CENTER_MIN_NORM = 0.35
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


@dataclass(frozen=True)
class MatchContext:
    """
    一次匹配所需的全部输入，**原子取得**。

    为什么打包成一个对象而不是两个方法：中心向量与 gallery 必须严格配套 ——
    gallery 里的每一行都是"减去该中心后归一化"的结果，query 若用另一个
    （哪怕只是稍旧一点的）中心处理，两者就不在同一坐标系，相似度全错且**不会报错**。
    若拆成 `get_match_gallery()` + `get_center()` 两次调用，中间另一个线程注册了
    新身份就会让中心变化，从而制造这种静默错配。打包返回即从 API 上排除该可能。
    """

    gallery: list[tuple[str, np.ndarray]]
    center: np.ndarray | None

    def prepare(self, vector: np.ndarray) -> np.ndarray:
        """把原始 query 特征变换到与 gallery 相同的坐标系。"""
        return IdentityStore._prepare_for_match(vector, self.center)

    @property
    def centering_enabled(self) -> bool:
        return self.center is not None


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
        # 公共分量中心（见 _feature_center_locked）与其缓存
        self._center_cache: np.ndarray | None = None
        self._center_cache_key = -1
        self._center_uses_centering = False
        self._gallery_version = 0
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
            self._invalidate_center_locked()
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
        self._invalidate_center_locked()
        return [victim.global_id]

    def _delete_persisted(self, global_ids: list[str]) -> None:
        if self._database is not None and global_ids:
            self._database.delete_identities(global_ids)

    # ------------------------------------------------------------------ #
    # 查询                                                                  #
    # ------------------------------------------------------------------ #

    def get_gallery(self) -> list[tuple[str, np.ndarray]]:
        """返回所有身份的 (global_id, 主特征) 列表 —— 严格一身份一行，**未中心化**。

        注意：主特征是 EMA 滑动平均，会指数抹平身份特异残差；而且未减去公共分量。
        **匹配请用 `get_match_gallery()`**，它展开 feature_bank 并做中心化。
        本方法保留给"一身份一向量"的展示型用途。
        """
        with self._lock:
            return [(gid, rec.feature.copy()) for gid, rec in self._records.items()]

    # ------------------------------------------------------------------ #
    # 公共分量中心化（匹配路径专用）                                          #
    # ------------------------------------------------------------------ #

    def _invalidate_center_locked(self) -> None:
        """
        标记公共分量中心缓存失效。必须在持有 self._lock 时、且**任何**会改动
        `_records` 或任一 `feature_bank` / 主特征之后调用。

        中心是全体特征的均值，只要有一个身份新增/淘汰/特征更新，它就变了。
        漏掉一处会导致 query 用新中心、gallery 用旧中心 —— 两者不在同一坐标系，
        相似度全错且**不会报错**，是本模块最难排查的一类缺陷。
        """
        self._gallery_version += 1

    def _feature_center_locked(self) -> np.ndarray | None:
        """
        计算/取缓存的全特征均值向量（公共分量方向）。必须在持有 self._lock 时调用。

        返回 None 表示"不启用中心化"（身份太少或本来就没有公共分量），
        此时匹配行为与改造前一致 —— 冷启动安全。
        """
        if self._center_cache_key == self._gallery_version and self._center_cache is not None:
            return self._center_cache
        if self._center_cache_key == self._gallery_version and self._center_cache is None:
            return None

        vectors = [
            vec for rec in self._records.values()
            for vec in (rec.feature, *rec.feature_bank)
        ]
        center: np.ndarray | None = None
        if len(vectors) >= _CENTER_MIN_FEATURES:
            candidate = np.mean(np.stack(vectors), axis=0)
            if float(np.linalg.norm(candidate)) >= _CENTER_MIN_NORM:
                center = candidate.astype(np.float32)

        if center is None and self._center_uses_centering:
            logger.warning(
                "ReID 公共分量中心化已停用（有效特征 %d 个，低于下限 %d，或中心范数不足）—— "
                "跨相机匹配退回改造前行为，准确率会明显下降",
                len(vectors), _CENTER_MIN_FEATURES,
            )
        elif center is not None and not self._center_uses_centering:
            logger.info(
                "ReID 公共分量中心化已启用：中心范数 %.4f（%d 个特征参与估计）",
                float(np.linalg.norm(center)), len(vectors),
            )
        self._center_uses_centering = center is not None
        self._center_cache = center
        self._center_cache_key = self._gallery_version
        return center

    @staticmethod
    def _prepare_for_match(vector: np.ndarray, center: np.ndarray | None) -> np.ndarray:
        """减去公共分量并重新 L2 归一化。center 为 None 时退化为普通归一化。"""
        vector = np.asarray(vector, dtype=np.float32)
        if center is not None:
            vector = vector - center
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            # 减完趋零（该向量几乎就是公共分量本身）→ 退回原方向，避免产生 NaN
            fallback = np.asarray(vector, dtype=np.float32)
            fallback_norm = float(np.linalg.norm(fallback))
            return fallback / fallback_norm if fallback_norm > 1e-8 else fallback
        return vector / norm

    def _build_match_context_locked(self) -> MatchContext:
        """
        构造匹配上下文（已中心化 + 已归一化的 gallery，连同所用中心）。
        必须在持有 self._lock 时调用。

        **单一实现点**：`build_match_context()` 与 `register_if_new()` 都用它，
        避免"注册时的 gallery 与匹配时的 gallery 不一致"这种极难发现的缺陷。
        """
        center = self._feature_center_locked()
        gallery = [
            (gid, self._prepare_for_match(vec, center))
            for gid, rec in self._records.items()
            for vec in (rec.feature, *rec.feature_bank)
        ]
        return MatchContext(gallery=gallery, center=center)

    def build_match_context(self) -> MatchContext:
        """
        取得一次匹配的完整上下文。调用方必须用返回的 `context.prepare()` 处理 query，
        不要自己拼装 —— 见 `MatchContext` 的说明。
        """
        with self._lock:
            return self._build_match_context_locked()

    def get_match_gallery(self) -> list[tuple[str, np.ndarray]]:
        """
        匹配用 gallery 的便捷访问（已中心化）。**返回的行必须与同一个中心配套使用**，
        因此实际匹配请优先用 `build_match_context()`，它把 gallery 与中心原子返回。

        每行 = 主特征或 feature_bank 中的一个姿态特征，都减去了公共分量中心并
        重新归一化（见 `_feature_center_locked`，这是识别能否工作的关键）。

        规模提示：行数 = 身份数 × (1 + bank 大小) ≤ 6 倍身份数。当前库只有几十个身份，
        开销可忽略；若将来 `LAB_MONITOR_MAX_IDENTITIES` 调到万级，需要改为
        预建特征矩阵或走 ANN 索引，否则每次匹配都要重算 6 万行。
        """
        with self._lock:
            return self._build_match_context_locked().gallery

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
            self._invalidate_center_locked()
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
                # gallery 与 query 必须用**同一个**中心向量处理，否则两者不在同一
                # 坐标系，相似度没有意义 —— MatchContext 把两者原子打包。
                context = self._build_match_context_locked()
                detail = match_feature_detailed(
                    context.prepare(feat_copy),
                    context.gallery,
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
                            "这个现象说明特征空间无法区分个体 —— 最常见的成因是"
                            "**未减去的公共分量**（全体特征均值范数可达 0.8，"
                            "远超随机方向的 ~0.1），其次是聚合把身份余量压掉。"
                            "诊断见 scripts/diagnose_ema_collapse.py，"
                            "修复方向见 docs/TODO_2026-09-12_worklist.md 项 1.6。"
                            "在中心化生效前，身份识别结果不应采信。",
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
            self._invalidate_center_locked()
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
            # 主特征与 feature_bank 都可能刚被改动 → 公共分量中心缓存失效
            self._invalidate_center_locked()
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
            center = self._feature_center_locked()
            center_norm = round(float(np.linalg.norm(center)), 4) if center is not None else 0.0
        summary = self.metrics.get_summary(gallery_size=g_size)
        # 塌缩告警次数与中心化状态外露到 /api/metrics/reid 与 /api/system/metrics：
        # 这两个失败模式此前都完全不可观测（只能靠人肉发现"身份数在慢慢变多"）。
        summary["collapse_warnings"] = collapse
        summary["center_enabled"] = center is not None
        summary["center_norm"] = center_norm
        return summary
