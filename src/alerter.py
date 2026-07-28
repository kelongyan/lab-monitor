"""
alerter.py — 预警状态机
Phase 4 更新：两阶段告警（WARNING → ALERT），降低误报的打扰
  Stage 1 WARNING：deadline 的 70% 时刻 → 控制台预警，不发外部通知
  Stage 2 ALERT  ：deadline 时刻    → 全量告警 + 外部通知推送
"""

import time
import json
import queue
import threading
import logging
import uuid
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

logger = logging.getLogger("alerter")

_WARNING_RATIO = 0.70   # deadline 的这个比例触发 WARNING


def _new_alert_id(prefix: str = "alert") -> str:
    """生成全局唯一的告警 ID，避免同毫秒并发时的碰撞"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class WatchEntry:
    global_id: str
    last_camera: str
    last_seen: float
    expected_cameras: list[str]
    deadline: float
    last_bbox: list[float] = field(default_factory=list)
    triggered: bool = False   # ALERT 已触发
    warned: bool = False      # WARNING 已触发（Stage 1）


class AlertBroadcaster:
    """
    线程安全的告警收集器：
    - 保存最近 N 条告警记录（供 Web API 查询历史）
    - 提供 Queue 供 WebSocket 后台任务消费并推送
    """
    def __init__(self, maxlen: int = 50):
        self._recent: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        # sync → async 桥：server 里的后台任务从这里读告警推 WebSocket
        self.queue: queue.Queue[dict] = queue.Queue()

    def push(self, alert: dict) -> None:
        with self._lock:
            self._recent.append(alert)
        self.queue.put(alert)

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self._recent)


class AlertManager:
    """
    使用方法：
      1. 人员离开摄像头时调用 watch()
      2. 人员出现在摄像头时调用 resolve()
      3. 后台线程调用 tick() 检查是否超时
    """

    def __init__(
        self,
        alert_log: str | Path = "outputs/alerts.jsonl",
        notifier=None,           # src.notifier.BaseNotifier 实例
        broadcaster: AlertBroadcaster | None = None,
        identity_store=None,              # IdentityStore 实例（避免循环导入不做类型标注）
        scene_exit_seconds: float = 300.0,  # 人员从所有摄像头消失多久后报警
    ):
        self._lock = threading.Lock()
        self._watches: dict[str, WatchEntry] = {}
        self._log_path = Path(alert_log)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._notifier = notifier
        self._broadcaster = broadcaster
        self._identity_store = identity_store
        self._scene_exit_seconds = scene_exit_seconds
        # gid → 已触发 SCENE_EXIT 告警的时间戳；人员重新出现后清除
        self._scene_exit_alerted: dict[str, float] = {}
        # P0-3：冷却字典初始化于 __init__，避免 hasattr 动态创建 + 内存泄漏
        self._intrusion_cooldown: dict[str, float] = {}
        # P2-3：持久文件句柄（line buffering），避免每次 open/close 开销
        try:
            self._log_file = open(self._log_path, "a", encoding="utf-8", buffering=1)
            # 注册 atexit 确保进程正常/异常退出时都能 flush+close，不依赖不可靠的 __del__
            atexit.register(self._close_log_file)
        except OSError as e:
            logger.error("无法打开告警日志文件 %s: %s，将跳过日志写入", self._log_path, e)
            self._log_file = None

    def _close_log_file(self) -> None:
        """安全关闭日志文件句柄（atexit 注册 + __del__ 共用）"""
        f = getattr(self, "_log_file", None)
        if f and not f.closed:
            try:
                f.flush()
                f.close()
            except Exception:
                pass
            self._log_file = None

    def __del__(self):
        self._close_log_file()

    # ------------------------------------------------------------------ #
    # 开启监听                                                              #
    # ------------------------------------------------------------------ #

    def watch(
        self,
        global_id: str,
        last_camera: str,
        expected_cameras: list[str],
        deadline_offset: float,
        last_bbox: list[float] = None,
    ) -> None:
        if not expected_cameras:
            return
        entry = WatchEntry(
            global_id=global_id,
            last_camera=last_camera,
            last_seen=time.time(),
            expected_cameras=expected_cameras,
            deadline=time.time() + deadline_offset,
            last_bbox=last_bbox or [],
        )
        with self._lock:
            self._watches[global_id] = entry
        logger.debug(
            "Watch started: %s from %s, expects %s within %.0fs",
            global_id, last_camera, expected_cameras, deadline_offset,
        )

    # ------------------------------------------------------------------ #
    # 解除监听                                                              #
    # ------------------------------------------------------------------ #

    def resolve(self, global_id: str, seen_camera: str) -> bool:
        with self._lock:
            entry = self._watches.pop(global_id, None)
        if entry is None:
            return False
        hit = seen_camera in entry.expected_cameras
        if hit:
            logger.info("Resolved: %s appeared in %s as expected", global_id, seen_camera)
        else:
            logger.warning(
                "Person %s appeared in unexpected camera %s (expected %s)",
                global_id, seen_camera, entry.expected_cameras,
            )
        return hit

    # ------------------------------------------------------------------ #
    # 即时入侵告警 (ROI 电子围栏)                                            #
    # ------------------------------------------------------------------ #

    def trigger_intrusion(
        self,
        camera_id: str,
        global_id: str,
        roi_name: str,
        bbox: list[float] = None,
    ) -> dict | None:
        """针对电子围栏 / 危险区域越界，触发即时 INTRUSION 告警"""
        now = time.time()
        key = f"intrusion_{camera_id}_{global_id}_{roi_name}"
        with self._lock:
            # 10 秒冷却防刷屏；同时清理 >60s 的过期 key，防止 Trk_N 无限累积（内存泄漏修复）
            if now - self._intrusion_cooldown.get(key, 0) < 10.0:
                return None
            self._intrusion_cooldown[key] = now
            # 清理超过 60 秒的旧记录（仅当字典非空时执行，避免每帧开销）
            if len(self._intrusion_cooldown) > 200:
                expired = [k for k, ts in self._intrusion_cooldown.items() if now - ts > 60.0]
                for k in expired:
                    del self._intrusion_cooldown[k]

        alert_id = _new_alert_id("alert_roi")
        alert = {
            "alert_id": alert_id,
            "timestamp": now,
            "stage": "ALERT",
            "alert_type": "INTRUSION",
            "global_id": global_id,
            "last_camera": camera_id,
            "expected_cameras": [f"禁止进入: {roi_name}"],
            "elapsed_seconds": 0,
            "risk_level": "HIGH",
            "last_bbox": bbox or [],
        }

        # 写入日志文件
        self._write_log(alert)

        # 广播到 WebSocket
        if self._broadcaster:
            self._broadcaster.push(alert)

        logger.warning("[%s] 🚨 电子围栏越界告警! 人员 #%s 闯入 [%s]", camera_id, global_id, roi_name)
        return alert

    # ------------------------------------------------------------------ #
    # 超时检查（由后台线程定期调用）                                          #
    # ------------------------------------------------------------------ #

    def tick(self, screenshot_dir: Path | None = None) -> list[dict]:
        now = time.time()
        triggered = []
        warned = []

        # 将所有状态修改都放在锁内，避免与 resolve() 的竞态
        with self._lock:
            for entry in list(self._watches.values()):
                if entry.triggered:
                    continue
                elapsed = now - entry.last_seen
                window = entry.deadline - entry.last_seen  # 总窗口秒数

                # Stage 1: WARNING（到达 70% 时刻）
                if not entry.warned and elapsed >= window * _WARNING_RATIO:
                    entry.warned = True
                    warned.append(entry)

                # Stage 2: ALERT（超过 deadline）
                if now >= entry.deadline:
                    entry.triggered = True
                    triggered.append(entry)

            # 清理也在锁内完成，确保原子性
            self._watches = {gid: e for gid, e in self._watches.items() if not e.triggered}

        # 处理 WARNING
        for entry in warned:
            logger.warning(
                "⚡ WARNING: person %s 即将超时（%.0fs），预期在 %s",
                entry.global_id, entry.deadline - now, entry.expected_cameras,
            )
            if self._broadcaster:
                self._broadcaster.push(self._build_alert(entry, stage="WARNING"))

        # 处理 ALERT（IO 操作在锁外执行，不阻塞其他线程）
        for entry in triggered:
            alert = self._build_alert(entry, stage="ALERT")
            self._write_log(alert)
            logger.warning(
                "⚠ ALERT: person %s missing! Last seen at %s, expected in %s",
                entry.global_id, entry.last_camera, entry.expected_cameras,
            )
            if self._notifier:
                try:
                    self._notifier.send(alert)
                except Exception as e:
                    logger.error("通知发送失败: %s", e)
            if self._broadcaster:
                self._broadcaster.push(alert)

        # 场景消失检测：人员从所有摄像头消失超过阈值则触发 SCENE_EXIT
        if self._identity_store is not None:
            self._check_scene_exits(now)

        return [self._build_alert(e, stage="ALERT") for e in triggered]

    # ------------------------------------------------------------------ #

    def _check_scene_exits(self, now: float) -> None:
        """检测从所有摄像头消失超过 scene_exit_seconds 的人员并触发 SCENE_EXIT 告警"""
        all_ids = self._identity_store.all_ids()
        for gid in all_ids:
            rec = self._identity_store.get(gid)
            if rec is None or rec.last_seen == 0.0:
                continue

            elapsed = now - rec.last_seen

            # 仍在视野内 → 清除已报警标记（为下次消失做准备）
            if elapsed < self._scene_exit_seconds:
                self._scene_exit_alerted.pop(gid, None)
                continue

            # 正在被拓扑路径监听 → 交给 MISSING_PERSON 处理，避免冲突
            with self._lock:
                in_watch = gid in self._watches
            if in_watch:
                continue

            # 已报警过（同一次消失事件） → 跳过
            if gid in self._scene_exit_alerted:
                continue

            # ✅ 触发场景消失告警
            self._scene_exit_alerted[gid] = now
            alert = {
                "alert_id": _new_alert_id("alert_exit"),
                "alert_type": "SCENE_EXIT",
                "stage": "ALERT",
                "global_id": gid,
                "last_camera": rec.last_camera,
                "last_seen": rec.last_seen,
                "last_bbox": self._identity_store.get_last_bbox(gid),
                "elapsed_seconds": round(elapsed, 1),
                "risk_level": "MEDIUM",
                "timestamp": now,
                "expected_cameras": ["全域监控区"],
            }

            self._write_log(alert)
            logger.warning(
                "👻 SCENE_EXIT: 人员 %s 已从所有摄像头消失 %.0f 秒（最后位置: %s）",
                gid, elapsed, rec.last_camera,
            )
            if self._broadcaster:
                self._broadcaster.push(alert)
            if self._notifier:
                try:
                    self._notifier.send(alert)
                except Exception as e:
                    logger.error("通知发送失败: %s", e)

    # ------------------------------------------------------------------ #

    def _build_alert(self, entry: WatchEntry, stage: str = "ALERT") -> dict:
        elapsed = time.time() - entry.last_seen
        risk = "HIGH" if elapsed > 120 else ("MEDIUM" if elapsed > 60 else "LOW")
        return {
            "alert_id": _new_alert_id(),  # uuid 保证并发时不碰撞
            "alert_type": "MISSING_PERSON",
            "stage": stage,           # "WARNING" or "ALERT"
            "global_id": entry.global_id,
            "last_camera": entry.last_camera,
            "last_seen": entry.last_seen,
            "last_bbox": entry.last_bbox,
            "expected_cameras": entry.expected_cameras,
            "elapsed_seconds": round(elapsed, 1),
            "risk_level": risk,
            "timestamp": time.time(),
        }

    def _write_log(self, alert: dict) -> None:
        # 1. 写入 SQLite 数据库持久化（方向三）
        try:
            from .db import db
            db.insert_alert(alert)
        except Exception as e:
            logger.error("写入 SQLite 数据库失败: %s", e)

        # 2. 追加式 jsonl 文件日志
        if self._log_file is None:
            return
        try:
            self._log_file.write(json.dumps(alert, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("告警日志写入失败: %s", e)

    def trigger_crowd_warning(self, camera_id: str, count: int, threshold: int = 5) -> dict | None:
        """方向三业务扩展：人流密度/聚众预警"""
        if count < threshold:
            return None
        now = time.time()
        key = f"crowd_{camera_id}"
        with self._lock:
            if now - self._intrusion_cooldown.get(key, 0) < 30.0: # 30秒冷却
                return None
            self._intrusion_cooldown[key] = now

        alert = {
          "alert_id": _new_alert_id("alert_crowd"),
          "timestamp": now,
          "stage": "WARNING",
          "alert_type": "CROWD_DENSITY",
          "global_id": f"Crowd_{count}",
          "last_camera": camera_id,
          "expected_cameras": [f"人流密集 (>= {threshold} 人)"],
          "elapsed_seconds": 0,
          "risk_level": "MEDIUM",
          "last_bbox": [],
        }
        self._write_log(alert)
        if self._broadcaster:
            self._broadcaster.push(alert)
        logger.warning("[%s] 👥 人流密度预警: 当前区域检测到 %d 人在场", camera_id, count)
        return alert
