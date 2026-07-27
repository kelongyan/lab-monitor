"""
notifier.py — 告警通知（控制台 + 邮件）
预警主要通过前端 WebSocket 实时推送，本模块仅负责服务端日志和可选邮件。
钉钉通知已移除：前端大屏已实时展示告警，无需第三方 IM 推送。
"""

import json
import logging
import smtplib
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

logger = logging.getLogger("notifier")


class BaseNotifier(ABC):
    @abstractmethod
    def send(self, alert: dict) -> None: ...


# ------------------------------------------------------------------ #
# 控制台（始终启用）                                                     #
# ------------------------------------------------------------------ #

class ConsoleNotifier(BaseNotifier):
    def send(self, alert: dict) -> None:
        stage = alert.get("stage", "ALERT")
        icon = "⚡" if stage == "WARNING" else "⚠ "
        logger.warning(
            "\n%s\n%s %s | ID=%-10s | 最后位置=%-8s | 超时=%.0fs | 风险=%s\n%s",
            "=" * 55, icon, stage,
            alert["global_id"], alert["last_camera"],
            alert["elapsed_seconds"], alert["risk_level"],
            "=" * 55,
        )


# ------------------------------------------------------------------ #
# 邮件通知（可选）                                                       #
# ------------------------------------------------------------------ #

class EmailNotifier(BaseNotifier):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def send(self, alert: dict) -> None:
        cfg = self.cfg
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[预警] 人员失踪 - {alert['global_id']} - {alert['risk_level']}"
        msg["From"] = cfg["from"]
        msg["To"] = ", ".join(cfg["to"])
        body = (
            f"人员失踪预警\n\n"
            f"身份 ID：{alert['global_id']}\n"
            f"最后出现：{alert['last_camera']}\n"
            f"预期出现：{', '.join(alert['expected_cameras'])}\n"
            f"超时时间：{alert['elapsed_seconds']} 秒\n"
            f"风险等级：{alert['risk_level']}\n"
        )
        msg.attach(MIMEText(body, "plain", "utf-8"))
        try:
            if cfg.get("use_ssl"):
                srv = smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], timeout=10)
            else:
                srv = smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=10)
            srv.login(cfg["username"], cfg["password"])
            srv.sendmail(cfg["from"], cfg["to"], msg.as_string())
            srv.quit()
            logger.info("邮件发送成功 → %s", cfg["to"])
        except Exception as e:
            logger.error("邮件发送失败: %s", e)


# ------------------------------------------------------------------ #
# 通知链                                                                #
# ------------------------------------------------------------------ #

class NotifierChain(BaseNotifier):
    def __init__(self, notifiers: list[BaseNotifier]):
        self._notifiers = notifiers

    def send(self, alert: dict) -> None:
        for n in self._notifiers:
            try:
                n.send(alert)
            except Exception as e:
                logger.error("通知发送异常 [%s]: %s", type(n).__name__, e)


def build_notifier(config_path: str | Path = "config/notify.json") -> NotifierChain:
    """从配置文件构建通知链；文件不存在时仅用控制台"""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return NotifierChain([ConsoleNotifier()])

    notifiers: list[BaseNotifier] = []

    if cfg.get("console", {}).get("enabled", True):
        notifiers.append(ConsoleNotifier())

    em = cfg.get("email", {})
    if em.get("enabled") and em.get("smtp_host"):
        notifiers.append(EmailNotifier(em))
        logger.info("邮件通知已启用 → %s", em.get("to"))

    return NotifierChain(notifiers)
