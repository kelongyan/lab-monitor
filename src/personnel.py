"""
personnel.py — 人员档案与底库检索（能力一 路径 B，worklist 3.3）。

两条路径的分工
--------------
- 路径 A（人工命名）：前端把 global_id 绑定到 person_id，走 IdentityStore.bind_person。
- 路径 B（底库 1:N 自动命名）：本模块。每人注册 1~N 张照片 → 提特征入库；
  实时检测到的人**先查底库**（实名信息价值高于匿名编号），命中即自动命名。

为什么底库检索必须插在全局身份库检索**之前**：
  ① 实名信息价值更高，命中即可直接报"张三"而不是"ID: c0ed5fd1"；
  ② 能抑制匿名身份膨胀 —— 旧库里两个"垃圾桶身份"吸收了 13.7 万 + 3.9 万次出现，
     根因就是无人可匹配时只好不停新建身份。

相似度必须**复用** match_feature_detailed()（按身份去重 + Ratio Test），
不要另写一套 —— 底库检索与实时匹配必须同一套判据，否则"标定"就失去了意义。

底库阈值独立于实时匹配阈值（LAB_MONITOR_PERSONNEL_THRESHOLD）：
底库特征来自**标准注册照**（正面、无遮挡、分辨率可控），质量远高于抓拍，
理论上可以用更严的阈值 —— 但默认先与实时阈值一致，等标注集扩充后再单独标定。
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np

from src.reid import match_feature_detailed
from src.reid_config import REID_MATCH_THRESHOLD

logger = logging.getLogger("personnel")


class PersonnelGallery:
    """人员底库：实名档案 + 注册照特征 + 1:N 检索。线程安全（内部锁）。"""

    def __init__(self, database, threshold: float = REID_MATCH_THRESHOLD):
        self._database = database
        self._threshold = float(threshold)
        self._lock_lock = __import__("threading").Lock()
        # person_id -> {"name": str, "employee_no":..., "features": [np.ndarray], "bound_gid": str|None}
        self._people: dict[str, dict] = {}
        self.reload()

    # ------------------------------------------------------------------ #
    # 加载                                                                  #
    # ------------------------------------------------------------------ #

    def reload(self) -> None:
        """从数据库重建内存底库（CRUD 之后必须调用，否则检索用的是旧数据）。"""
        people: dict[str, dict] = {}
        if self._database is not None:
            for person in self._database.list_personnel():
                people[person["person_id"]] = {
                    "name": person["name"],
                    "employee_no": person.get("employee_no"),
                    "department": person.get("department"),
                    "features": [],
                }
            for person_id in people:
                for photo in self._database.list_personnel_photos(person_id):
                    blob = photo.get("feature_blob")
                    dim = photo.get("feature_dim") or 0
                    if not blob or dim <= 0 or len(blob) != dim * 4:
                        logger.warning("跳过损坏的注册照 %s/%s", person_id, photo.get("photo_id"))
                        continue
                    vector = np.frombuffer(blob, dtype=np.float32).copy()
                    norm = float(np.linalg.norm(vector))
                    if norm > 1e-8:
                        people[person_id]["features"].append(vector / norm)

        with self._lock_lock:
            self._people = people
        logger.info("人员底库加载完成：%d 人 / %d 张注册照",
                    len(people), sum(len(p["features"]) for p in people.values()))

    # ------------------------------------------------------------------ #
    # 查询                                                                  #
    # ------------------------------------------------------------------ #

    def names(self) -> dict[str, str]:
        with self._lock_lock:
            return {pid: p["name"] for pid, p in self._people.items()}

    def get(self, person_id: str) -> dict | None:
        with self._lock_lock:
            person = self._people.get(person_id)
            return dict(person) if person else None

    def size(self) -> int:
        with self._lock_lock:
            return len(self._people)

    def threshold(self) -> float:
        return self._threshold

    def bound_gid(self, person_id: str, store) -> str | None:
        """该实名名下已绑定的 global_id（取出现次数最多者）。"""
        return store.bound_gid_map().get(person_id)

    # ------------------------------------------------------------------ #
    # 1:N 检索                                                              #
    # ------------------------------------------------------------------ #

    def match(self, feature: np.ndarray) -> dict | None:
        """
        底库 1:N 检索。命中（≥ 阈值且过 Ratio Test）返回：
            {"person_id", "name", "score"}
        未命中返回 None（调用方继续走匿名身份流程）。

        空底库返回 None —— 这是常态（没建档案就没有自动命名）。
        """
        with self._lock_lock:
            gallery = []
            meta = {}
            for person_id, person in self._people.items():
                features = person["features"]
                if not features:
                    continue
                best = max(features, key=lambda vec: float(vec @ feature))
                gallery.append((person_id, best))
                meta[person_id] = person["name"]
        if not gallery:
            return None

        detail = match_feature_detailed(
            np.asarray(feature, dtype=np.float32), gallery,
            threshold=self._threshold,
        )
        if detail.matched_id is None:
            return None
        return {
            "person_id": detail.matched_id,
            "name": meta.get(detail.matched_id),
            "score": round(detail.best_sim, 4),
        }

    # ------------------------------------------------------------------ #
    # CRUD（薄封装：写库 + reload，保证内存与库一致）                          #
    # ------------------------------------------------------------------ #

    def create(self, name: str, employee_no: str | None = None,
               department: str | None = None, note: str | None = None,
               person_id: str | None = None) -> str:
        pid = person_id or uuid.uuid4().hex[:8]
        if not self._database.upsert_personnel(pid, name, employee_no, department, note):
            raise RuntimeError(f"写人员档案失败: {pid}")
        self.reload()
        return pid

    def update(self, person_id: str, name: str, employee_no: str | None = None,
               department: str | None = None, note: str | None = None) -> bool:
        ok = self._database.upsert_personnel(person_id, name, employee_no, department, note)
        self.reload()
        return ok

    def delete(self, person_id: str) -> bool:
        """删除档案。数据库侧会同步解除名下身份的绑定（见 Database.delete_personnel）。"""
        ok = self._database.delete_personnel(person_id)
        self.reload()
        return ok

    def add_photo(self, person_id: str, feature: np.ndarray,
                  quality: float = 1.0, source_path: str | None = None) -> int | None:
        vector = np.asarray(feature, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise ValueError("注册照特征为零向量")
        vector = vector / norm
        photo_id = self._database.save_personnel_photo(
            person_id, int(vector.size), vector.tobytes(), float(quality), source_path
        )
        self.reload()
        return photo_id

    def photos_of(self, person_id: str) -> int:
        return len(self._database.list_personnel_photos(person_id))

    @staticmethod
    def default_person_id() -> str:
        return uuid.uuid4().hex[:8]
