"""
db.py — 基于 SQLite 的轻量级持久化数据库模块
零额外依赖，用于持久化存储告警日志、人员身份历史与系统事件
"""

import sqlite3
import json
import time
import logging
import threading
import hashlib
from pathlib import Path
from typing import Any

logger = logging.getLogger("db")

DB_PATH = Path(__file__).parent.parent / "outputs" / "lab_monitor.db"


class Database:
    """线程安全的 SQLite 数据库管理类（每个线程复用独立持久连接，消除连接创建开销）"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()   # 每个线程独立的连接槽
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """返回当前线程的持久连接，不存在则创建（一个线程只建一次连接）"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            # WAL 模式：写操作不阻塞并发读，适合多线程混合场景
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")   # 性能/安全折中
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """关闭当前线程持有的 SQLite 连接。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _init_db(self) -> None:
        """初始化数据库 Schema"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 告警事件表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE,
                    timestamp REAL,
                    camera_id TEXT,
                    global_id TEXT,
                    alert_type TEXT,
                    stage TEXT,
                    risk_level TEXT,
                    elapsed_seconds REAL,
                    expected_cameras TEXT,
                    details_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 身份历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    global_id TEXT PRIMARY KEY,
                    first_seen REAL,
                    last_seen REAL,
                    last_camera TEXT,
                    total_appearances INTEGER,
                    feature_dim INTEGER,
                    feature_blob BLOB,
                    feature_bank_count INTEGER DEFAULT 0,
                    feature_bank_blob BLOB,
                    appearances_json TEXT,
                    feature_space TEXT,
                    feature_schema_version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS identity_appearances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id TEXT NOT NULL,
                    camera_id TEXT,
                    timestamp REAL NOT NULL,
                    bbox_json TEXT NOT NULL
                )
            """)
            identity_columns = {
                row[1] for row in cursor.execute("PRAGMA table_info(identities)")
            }
            migrations = {
                "feature_dim": "INTEGER",
                "feature_blob": "BLOB",
                "feature_bank_count": "INTEGER DEFAULT 0",
                "feature_bank_blob": "BLOB",
                "appearances_json": "TEXT",
                "feature_space": "TEXT",
                "feature_schema_version": "INTEGER DEFAULT 1",
            }
            for column, definition in migrations.items():
                if column not in identity_columns:
                    cursor.execute(
                        f"ALTER TABLE identities ADD COLUMN {column} {definition}"
                    )
            # 创建索引加速检索
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_gid ON alerts(global_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cam ON alerts(camera_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_identity_appearances_gid_ts "
                "ON identity_appearances(global_id, timestamp DESC, id DESC)"
            )
            conn.commit()
        logger.info("SQLite 数据库初始化完成: %s", self.db_path)

    def insert_alert(self, alert_data: dict[str, Any], replace: bool = True) -> bool:
        """插入一条告警记录"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                conflict = "OR REPLACE" if replace else "OR IGNORE"
                cursor.execute(f"""
                    INSERT {conflict} INTO alerts (
                        alert_id, timestamp, camera_id, global_id,
                        alert_type, stage, risk_level, elapsed_seconds,
                        expected_cameras, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert_data.get("alert_id"),
                    alert_data.get("timestamp", time.time()),
                    alert_data.get("last_camera") or alert_data.get("camera_id"),
                    alert_data.get("global_id"),
                    alert_data.get("alert_type"),
                    alert_data.get("stage", "ALERT"),
                    alert_data.get("risk_level", "HIGH"),
                    alert_data.get("elapsed_seconds", 0.0),
                    json.dumps(alert_data.get("expected_cameras", [])),
                    json.dumps(alert_data)
                ))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error("写入告警到 SQLite 失败: %s", e)
            return False

    def upsert_identity(self, global_id: str, camera_id: str, first_seen: float = None, last_seen: float = None, appearances_count: int = 1) -> bool:
        """更新或注册人员身份"""
        now = time.time()
        f_seen = first_seen or now
        l_seen = last_seen or now
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO identities (global_id, first_seen, last_seen, last_camera, total_appearances)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(global_id) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        last_camera = excluded.last_camera,
                        total_appearances = total_appearances + 1
                """, (global_id, f_seen, l_seen, camera_id, appearances_count))
                conn.commit()
                return True
        except Exception as e:
            logger.error("写入身份到 SQLite 失败: %s", e)
            return False

    def query_alerts(self, limit: int = 50, offset: int = 0, camera_id: str = None, global_id: str = None) -> list[dict]:
        """分页与条件检索历史告警"""
        sql = "SELECT details_json FROM alerts WHERE 1=1"
        params = []
        if camera_id:
            sql += " AND camera_id = ?"
            params.append(camera_id)
        if global_id:
            sql += " AND global_id = ?"
            params.append(global_id)
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                rows = cursor.execute(sql, params).fetchall()
                result = []
                for r in rows:
                    try:
                        result.append(json.loads(r["details_json"]))
                    except Exception:
                        pass
                return result
        except Exception as e:
            logger.error("查询 SQLite 告警失败: %s", e)
            return []

    def query_alert_page(
        self,
        limit: int = 50,
        offset: int = 0,
        camera_id: str | None = None,
        global_id: str | None = None,
        risk_level: str | None = None,
    ) -> tuple[int, list[dict]]:
        """从唯一权威数据源返回过滤后的总数和分页结果。"""
        where = []
        params: list[Any] = []
        for column, value in (
            ("camera_id", camera_id),
            ("global_id", global_id),
            ("risk_level", risk_level),
        ):
            if value:
                where.append(f"{column} = ?")
                params.append(value)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""
        try:
            with self._get_conn() as conn:
                total = conn.execute(
                    f"SELECT COUNT(*) FROM alerts{where_sql}", params
                ).fetchone()[0]
                rows = conn.execute(
                    f"SELECT details_json FROM alerts{where_sql} "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    [*params, limit, offset],
                ).fetchall()
            alerts = []
            for row in rows:
                try:
                    alerts.append(json.loads(row["details_json"]))
                except (TypeError, json.JSONDecodeError):
                    logger.warning("跳过无法解析的告警记录")
            return int(total), alerts
        except Exception as e:
            logger.error("分页查询 SQLite 告警失败: %s", e)
            raise

    def import_alert_log(self, log_path: str | Path) -> int:
        """将旧 JSONL 记录幂等合并进 SQLite，之后查询只读取 SQLite。"""
        path = Path(log_path)
        if not path.exists():
            return 0
        imported = 0
        with path.open("r", encoding="utf-8") as log_file:
            for line in log_file:
                try:
                    alert = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(alert, dict):
                    continue
                if not alert.get("alert_id"):
                    canonical = json.dumps(
                        alert,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    alert["alert_id"] = (
                        "legacy_" + hashlib.sha256(canonical).hexdigest()[:20]
                    )
                if self.insert_alert(alert, replace=False):
                    imported += 1
        return imported

    def get_alert_summary(self) -> dict[str, int]:
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN stage = 'WARNING' THEN 0 ELSE 1 END) AS alerts,
                    SUM(CASE WHEN stage = 'WARNING' THEN 1 ELSE 0 END) AS warnings
                FROM alerts
                """
            ).fetchone()
        return {
            "alerts": int(row["alerts"] or 0),
            "warnings": int(row["warnings"] or 0),
        }

    def apply_retention(self, retention_days: int) -> dict[str, int]:
        """删除超过保留期的在线告警、轨迹和身份数据。"""
        cutoff = time.time() - max(1, retention_days) * 86400
        with self._get_conn() as conn:
            alerts = conn.execute(
                "DELETE FROM alerts WHERE timestamp < ?", (cutoff,)
            ).rowcount
            appearances = conn.execute(
                "DELETE FROM identity_appearances WHERE timestamp < ?", (cutoff,)
            ).rowcount
            identities = conn.execute(
                "DELETE FROM identities WHERE last_seen > 0 AND last_seen < ?",
                (cutoff,),
            ).rowcount
            conn.commit()
        return {
            "alerts": max(0, alerts),
            "appearances": max(0, appearances),
            "identities": max(0, identities),
        }

    def delete_identities(self, global_ids: list[str]) -> None:
        if not global_ids:
            return
        placeholders = ",".join("?" for _ in global_ids)
        with self._get_conn() as conn:
            conn.execute(
                f"DELETE FROM identity_appearances WHERE global_id IN ({placeholders})",
                global_ids,
            )
            conn.execute(
                f"DELETE FROM identities WHERE global_id IN ({placeholders})",
                global_ids,
            )
            conn.commit()

    def save_identity(
        self,
        global_id: str,
        feature_dim: int,
        feature_blob: bytes,
        feature_bank_count: int,
        feature_bank_blob: bytes,
        appearances: list[dict],
        total_appearances: int,
        last_camera: str,
        last_seen: float,
        feature_space: str,
        new_appearance: dict | None = None,
        schema_version: int = 1,
    ) -> bool:
        first_seen = appearances[0].get("time", last_seen) if appearances else last_seen
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO identities (
                        global_id, first_seen, last_seen, last_camera,
                        total_appearances, feature_dim, feature_blob,
                        feature_bank_count, feature_bank_blob,
                        appearances_json, feature_space, feature_schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(global_id) DO UPDATE SET
                        first_seen = excluded.first_seen,
                        last_seen = excluded.last_seen,
                        last_camera = excluded.last_camera,
                        total_appearances = excluded.total_appearances,
                        feature_dim = excluded.feature_dim,
                        feature_blob = excluded.feature_blob,
                        feature_bank_count = excluded.feature_bank_count,
                        feature_bank_blob = excluded.feature_bank_blob,
                        appearances_json = excluded.appearances_json,
                        feature_space = excluded.feature_space,
                        feature_schema_version = excluded.feature_schema_version
                    """,
                    (
                        global_id,
                        first_seen,
                        last_seen,
                        last_camera,
                        total_appearances,
                        feature_dim,
                        feature_blob,
                        feature_bank_count,
                        feature_bank_blob,
                        json.dumps(appearances, ensure_ascii=False),
                        feature_space,
                        schema_version,
                    ),
                )
                if new_appearance is not None:
                    conn.execute(
                        """
                        INSERT INTO identity_appearances (
                            global_id, camera_id, timestamp, bbox_json
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            global_id,
                            new_appearance.get("camera", ""),
                            float(new_appearance.get("time", last_seen)),
                            json.dumps(new_appearance.get("bbox", [])),
                        ),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.error("持久化身份特征失败: %s", e)
            return False

    def load_identities(self) -> list[dict[str, Any]]:
        """加载具有完整特征数据的身份；旧元数据行会保留但不参与 ReID。"""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT global_id, last_seen, last_camera, total_appearances, feature_dim,
                           feature_blob, feature_bank_count, feature_bank_blob,
                           appearances_json, feature_space, feature_schema_version
                    FROM identities
                    WHERE feature_dim > 0 AND feature_blob IS NOT NULL
                    ORDER BY created_at, global_id
                    """
                ).fetchall()
            result = []
            for row in rows:
                try:
                    appearances = json.loads(row["appearances_json"] or "[]")
                except json.JSONDecodeError:
                    appearances = []
                result.append({
                    "global_id": row["global_id"],
                    "last_seen": row["last_seen"] or 0.0,
                    "last_camera": row["last_camera"] or "",
                    "feature_dim": row["feature_dim"],
                    "feature_blob": row["feature_blob"],
                    "feature_bank_count": row["feature_bank_count"] or 0,
                    "feature_bank_blob": row["feature_bank_blob"] or b"",
                    "appearances": appearances if isinstance(appearances, list) else [],
                    "total_appearances": int(row["total_appearances"] or 0),
                    "feature_space": row["feature_space"] or "",
                    "schema_version": row["feature_schema_version"] or 1,
                })
                recent_rows = conn.execute(
                    """
                    SELECT camera_id, timestamp, bbox_json
                    FROM identity_appearances
                    WHERE global_id = ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 200
                    """,
                    (row["global_id"],),
                ).fetchall()
                if recent_rows:
                    result[-1]["appearances"] = [
                        {
                            "camera": appearance["camera_id"],
                            "time": appearance["timestamp"],
                            "bbox": json.loads(appearance["bbox_json"] or "[]"),
                        }
                        for appearance in reversed(recent_rows)
                    ]
                count_row = conn.execute(
                    "SELECT COUNT(*) FROM identity_appearances WHERE global_id = ?",
                    (row["global_id"],),
                ).fetchone()
                if count_row[0]:
                    result[-1]["total_appearances"] = int(count_row[0])
            return result
        except Exception as e:
            logger.error("恢复身份特征失败: %s", e)
            return []

    def query_identity_appearances(
        self,
        global_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> tuple[int, list[dict]]:
        with self._get_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM identity_appearances WHERE global_id = ?",
                (global_id,),
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT camera_id, timestamp, bbox_json
                FROM identity_appearances
                WHERE global_id = ?
                ORDER BY timestamp, id
                LIMIT ? OFFSET ?
                """,
                (global_id, limit, offset),
            ).fetchall()
        appearances = []
        for row in rows:
            try:
                bbox = json.loads(row["bbox_json"] or "[]")
            except json.JSONDecodeError:
                bbox = []
            appearances.append({
                "camera": row["camera_id"],
                "time": row["timestamp"],
                "bbox": bbox,
            })
        return int(total), appearances

    def get_stats(self) -> dict[str, int]:
        """获取数据库总统计数据"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                total_alerts = cursor.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
                total_identities = cursor.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
                return {
                    "db_total_alerts": total_alerts,
                    "db_total_identities": total_identities,
                    "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
                }
        except Exception as e:
            logger.error("查询 SQLite 统计失败: %s", e)
            return {"db_total_alerts": 0, "db_total_identities": 0, "db_size_bytes": 0}


# 全局单例数据库对象
db = Database()
