"""rebuild_identity_db.py — 重建生产身份库（修完防重复注册后的必须步骤）。

为什么需要重建（2026-10-07 取证，全部可直接复现）
--------------------------------------------------
1. `outputs/dev/probe_dupes.py`：rnd_04 一路累积 115 个身份，其中 39 个的
   feature_blob **字节完全相同**（md5 一致）；中心化后全体 p50 = 1.0000。
   它们观测的是同一批帧（video_frame 范围 75~363 逐帧一致）。
2. `outputs/dev/probe_match_replay.py`：用**存储的 bbox + 源视频帧 + 当前权重**
   重新提取特征（顺序解码，坐标与生产一致），与该身份存储主特征的相似度
   只有 0.42~0.78 raw / **centered -0.31~-0.53**；与存储 bank 条目最高
   0.70 raw —— **库内任何向量都无法被当前提取管线复现**。
3. 结论：存储特征已漂移成"谁也认不出"的 EMA 吸引子（数学本质是对宽锥形
   分布做长程平均，不动点落在锥心），**无法用归并或阈值技巧救回**，
   只能重建。这是"能力一（识别是谁）"的核心阻塞。

重建内容
--------
- `identity_appearances`（~143 万行，95% 以上是 20 秒素材循环播放的重复产物）
- `identities`（169 行）
- 保留：`alerts`（历史事件，global_id 引用变成孤儿但不删除 —— 事件审计
  不应因为身份重建而丢失）、`personnel` / `personnel_photos`（底库人员行
  保留，只是与旧 gid 的绑定随 identities 清空）、`video_assets`
- VACUUM 回收空间，备份数据库到 outputs/backups/

前置条件
--------
**必须在服务停止时运行**：IdentityStore 的内存状态会通过节流落盘
（_FEATURE_FLUSH_*），若服务在跑，删完会被内存副本重新写回。
脚本会检查 8000 端口与 outputs/server.pid，活着就拒绝执行。

用法
----
    # 干跑（默认，只报告）
    ./.venv/Scripts/python.exe scripts/rebuild_identity_db.py
    # 真正执行
    ./.venv/Scripts/python.exe scripts/rebuild_identity_db.py --confirm
"""

from __future__ import annotations

import argparse
import io
import shutil
import socket
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "outputs" / "lab_monitor.db"
BACKUP_DIR = ROOT / "outputs" / "backups"
SERVICE_PORT = 8000
PID_FILE = ROOT / "outputs" / "server.pid"


def service_is_running() -> tuple[bool, str]:
    """检查服务是否在跑（端口监听 或 PID 文件指向存活进程）。"""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.3)
    try:
        probe.connect(("127.0.0.1", SERVICE_PORT))
        return True, f"端口 {SERVICE_PORT} 有监听"
    except OSError:
        pass
    finally:
        probe.close()
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None
        if pid:
            try:
                import psutil
                if psutil.pid_exists(pid):
                    return True, f"PID 文件 {pid} 指向存活进程"
            except ImportError:
                return True, f"PID 文件存在（{pid}），无法验证进程存活，按在跑处理"
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="重建生产身份库（identities + identity_appearances）")
    parser.add_argument("--confirm", action="store_true",
                        help="真正执行删除（默认干跑）")
    args = parser.parse_args()

    if not DB.exists():
        print(f"✗ 数据库不存在: {DB}")
        return 1

    running, why = service_is_running()
    if running:
        print(f"✗ 服务正在运行（{why}）。IdentityStore 的内存状态会在落盘节流里"
              f"把删除的行重新写回。请先 ./stop.ps1 或停掉 main.py 再执行。")
        return 1

    conn = sqlite3.connect(str(DB))
    try:
        identities = conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
        appearances = conn.execute("SELECT COUNT(*) FROM identity_appearances").fetchone()[0]
        alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    finally:
        conn.close()

    print(f"当前库: identities={identities} / identity_appearances={appearances} / alerts={alerts}(保留)")
    print(f"将清空: identities + identity_appearances")
    print(f"备份到: {BACKUP_DIR}/")

    if not args.confirm:
        print("\n（干跑未执行）加 --confirm 真正执行")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = BACKUP_DIR / f"lab_monitor.db.{stamp}"
    shutil.copy2(str(DB), str(backup))
    backup_size = backup.stat().st_size / (1024 * 1024)
    print(f"✓ 已备份 {backup}（{backup_size:.1f} MB）")

    conn = sqlite3.connect(str(DB))
    try:
        conn.execute("DELETE FROM identity_appearances")
        conn.execute("DELETE FROM identities")
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
        after_id = conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
        after_app = conn.execute("SELECT COUNT(*) FROM identity_appearances").fetchone()[0]
        after_alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    finally:
        conn.close()
    print(f"✓ 已清空：identities {identities}→{after_id}，"
          f"identity_appearances {appearances}→{after_app}，"
          f"alerts 保留 {after_alerts}")
    print("\n下一步：用修好的代码启动服务，全量重跑语料（约 30 分钟）让身份库"
          "在防重复注册（bank 分层准入 + ratio 旁路 + 窗口主特征）下自然重建：")
    print("  ./.venv/Scripts/python.exe main.py")
    print("监控：GET /api/metrics/reid —— match_rate 应显著高于 0.26，"
          "created_near_miss 应趋近 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
