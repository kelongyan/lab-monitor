"""
trajectory.py — 轨迹分段与循环折叠（/api/.../trajectory 与检索接口共用的单一实现）。

为什么必须抽成公共模块
--------------------
素材是**循环播放**的（pipeline._run_file 遇到 EOF 就重开），部分素材只有几十秒。
一个在镜头里走一趟的人会被记录成上千段「A→B→A→B...」，直接画路线/算检索会得到
「这个人在两点之间来回跑了 545 次」的假象。折叠逻辑有 19 个测试用例保护
（tests/test_trajectory.py），**检索接口必须复用而不是重写** —— 重写必然倒退。

两个判据（缺一不可，互为兜底）：
  1. 差分周期检测 `seq[i] == seq[i - period]`（不是相位对齐！真实数据里
     A→B→A→B 中途会翻成 B→A→B→A，相位对齐在翻转点之后 100% 判不匹配，
     实测 9340 段严格交替序列只有 5% 匹配率）。
  2. 位置指纹：按 (camera, 首帧 bbox 中心 20px 量化) 去重。
     循环播放的本质是"同一段像素被反复播放"，段的起始画面位置几乎完全一致。

硬截断到第一轮前要做**覆盖校验**：若后续轮次出现了第一轮没有的相机，
降级用指纹法 —— 宁可少折叠也不丢轨迹。
"""

from __future__ import annotations

MAX_PERIOD = 200          # 一轮真实轨迹不会经过上百个停留段，同时避免 O(n×period) 打爆请求
PERIOD_MATCH_RATIO = 0.9  # 90% 匹配即可吸收相位翻转与漏检抖动
FINGERPRINT_QUANTUM = 20  # bbox 中心量化步长（px）：太细漏折叠、太粗误折叠


def detect_loop_period(seq: list[str]) -> int | None:
    """检测相机序列的最小重复周期（严格周期，快速路径）。无周期返回 None。"""
    n = len(seq)
    if n < 4:
        return None
    max_period = min(n // 2, MAX_PERIOD)
    for period in range(1, max_period + 1):
        checked = n - period
        if checked <= 0:
            break
        match = sum(1 for i in range(period, n) if seq[i] == seq[i - period])
        if match / checked >= PERIOD_MATCH_RATIO:
            return period
    return None


def collapse_by_fingerprint(segments: list[dict]) -> list[int]:
    """按 (camera, 首帧 bbox 中心量化) 去重，返回保留段的下标（保持首次出现顺序）。"""
    seen: set = set()
    keep: list[int] = []
    for i, seg in enumerate(segments):
        key = seg["camera"]
        bb = seg.get("bbox_start") or []
        if len(bb) >= 4:
            try:
                cx = (float(bb[0]) + float(bb[2])) / 2.0
                cy = (float(bb[1]) + float(bb[3])) / 2.0
                key = (seg["camera"], round(cx / FINGERPRINT_QUANTUM),
                       round(cy / FINGERPRINT_QUANTUM))
            except (TypeError, ValueError):
                pass  # bbox 异常时退化为只按相机去重
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def build_segments(rows: list[dict], split_gap_s: float = 30.0) -> list[dict]:
    """
    把时序点聚合成「连续停留段」：连续同相机合并；
    间隔超过 split_gap_s 视为重新进入视野，拆开。

    rows 需要 time / camera / bbox 字段（db.query_trajectory 的输出）。
    """
    segments: list[dict] = []
    for row in rows:
        cam, ts = row["camera"], row["time"]
        if (segments and segments[-1]["camera"] == cam
                and ts - segments[-1]["exit"] <= split_gap_s):
            seg = segments[-1]
            seg["exit"] = ts
            seg["frames"] += 1
            seg["bbox_end"] = row["bbox"]
        else:
            segments.append({
                "camera": cam,
                "enter": ts,
                "exit": ts,
                "frames": 1,
                "bbox_start": row["bbox"],
                "bbox_end": row["bbox"],
            })
    for seg in segments:
        seg["duration_s"] = round(seg["exit"] - seg["enter"], 2)
    return segments


def collapse_loop_segments(
    segments: list[dict],
    rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """
    折叠循环播放产生的重复轨迹。返回 (segments, rows, loop_info)。

    loop_info 结构（前端应展示 loop 字段说明是否折叠、用什么策略、折掉了多少）：
      {"detected": bool, "method": "period"|"fingerprint"|None,
       "period_segments": int, "loops": int|float, "raw_segments": int}
    """
    loop_info = {"detected": False, "method": None, "period_segments": 0,
                 "loops": 1, "raw_segments": len(segments)}
    if len(segments) < 4:
        return segments, rows, loop_info

    raw = len(segments)
    period = detect_loop_period([s["camera"] for s in segments])

    # 覆盖校验：硬截断到第一轮会丢掉「只在后续轮次出现的相机」。
    # 实测 9cd02946 的序列是 rnd_08→rnd_08→reg_08→…，检测到周期 2 后
    # 截断成 [rnd_08, rnd_08]，把只在第 3 段出现的 reg_08 整个丢了。
    # 宁可少折叠也不能丢轨迹，覆盖不全就降级到指纹法。
    if period and raw // period >= 2:
        cams_first = {s["camera"] for s in segments[:period]}
        cams_all = {s["camera"] for s in segments}
        if not cams_all <= cams_first:
            period = None

    if period and raw // period >= 2:
        # 严格周期且覆盖完整：截断到第一轮，时间线连续，播放动画不会瞬移
        loop_info = {"detected": True, "method": "period",
                     "period_segments": period, "loops": raw // period,
                     "raw_segments": raw}
        cut_at = segments[period]["enter"]
        segments = segments[:period]
        rows = [r for r in rows if r["time"] < cut_at]
        return segments, rows, loop_info

    keep = collapse_by_fingerprint(segments)
    if len(keep) < raw:
        # 非严格周期（或覆盖不全）：按画面位置指纹去重，只保留独特停留段的点
        windows = [(segments[i]["enter"], segments[i]["exit"]) for i in keep]
        segments = [segments[i] for i in keep]
        rows = [r for r in rows
                if any(a <= r["time"] <= b for a, b in windows)]
        loop_info = {"detected": True, "method": "fingerprint",
                     "period_segments": len(segments),
                     "loops": round(raw / max(1, len(segments)), 1),
                     "raw_segments": raw}
    return segments, rows, loop_info
