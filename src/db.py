"""
db.py — 基于 SQLite 的轻量级持久化数据库模块
零额外依赖，用于持久化存储告警日志、人员身份历史与系统事件
"""

import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("db")

DB_PATH = Path(__file__).parent.parent / "outputs" / "lab_monitor.db"


class Database:
    """线程安全的 SQLite 数据库管理类"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 创建索引加速检索
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_gid ON alerts(global_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cam ON alerts(camera_id)")
            conn.commit()
        logger.info("SQLite 数据库初始化完成: %s", self.db_path)

    def insert_alert(self, alert_data: dict[str, Any]) -> bool:
        """插入一条告警记录"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO alerts (
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
                return True
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
