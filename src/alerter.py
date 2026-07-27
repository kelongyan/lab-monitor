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
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

logger = logging.getLogger("alerter")


_WARNING_RATIO = 0.70   # deadline 的这个比例触发 WARNING


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
    ):
        self._lock = threading.Lock()
        self._watches: dict[str, WatchEntry] = {}
        self._log_path = Path(alert_log)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._notifier = notifier
        self._broadcaster = broadcaster

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
    # 超时检查（由后台线程定期调用）                                          #
    # ------------------------------------------------------------------ #

    def tick(self, screenshot_dir: Path | None = None) -> list[dict]:
        now = time.time()
        triggered = []
        warned = []

        with self._lock:
            entries = list(self._watches.values())

        for entry in entries:
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

        # 处理 WARNING
        for entry in warned:
            logger.warning(
                "⚡ WARNING: person %s 即将超时（%.0fs），预期在 %s",
                entry.global_id, entry.deadline - now, entry.expected_cameras,
            )
            if self._broadcaster:
                self._broadcaster.push(self._build_alert(entry, stage="WARNING"))

        # 处理 ALERT
        for entry in triggered:
            alert = self._build_alert(entry, stage="ALERT")
            self._write_log(alert)
            triggered_alerts = triggered  # 用于返回
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

        with self._lock:
            self._watches = {gid: e for gid, e in self._watches.items() if not e.triggered}

        return [self._build_alert(e, stage="ALERT") for e in triggered]

    # ------------------------------------------------------------------ #

    def _build_alert(self, entry: WatchEntry, stage: str = "ALERT") -> dict:
        elapsed = time.time() - entry.last_seen
        risk = "HIGH" if elapsed > 120 else ("MEDIUM" if elapsed > 60 else "LOW")
        return {
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
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")
