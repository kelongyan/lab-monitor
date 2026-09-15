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


def merge_flicker_segments(
    segments: list[dict],
    max_blip_s: float = 2.0,
    max_rounds: int = 8,
) -> tuple[list[dict], dict]:
    """
    把「循环素材 + 多相机并行确认」造成的往返抖动压成**视图组**。

    为什么要这一层
    --------------
    `collapse_loop_segments()` 解决的是「同一段像素被反复播放」（周期/指纹折叠），
    但折叠之后仍会剩下另一种噪声：**两条相机在同一时间窗内交替确认同一身份**。
    实测 5a7991c7（3 路相机）折叠后仍有 125 段，形态是

        rnd_04 (+0.0s → +132.7s)      ← 一段正常的长停留
        rnd_22 (+132.9s)               ← 此后 rnd_04 / rnd_22
        rnd_04 (+133.2s)                  每 0.2~0.5 秒交替一次
        rnd_22 (+133.4s)
        ...

    这一段里的 `duration_s` 几乎全是 0，成因不是人真的往返，而是两条走廊相机的
    画面被并行处理后各自写入了确认记录。**直接按 path 连折线会得到锯齿**，
    既不能表达路线，也会引出「轨迹为何来回跳」的质疑。

    `merge_flicker_segments()` 把它压成「视图组」：一组 = 一个连续时间窗内
    可见到的**相机集合**。上例的结果是一个 `cameras=[rnd_04, rnd_22]` 的组，
    语义是「该时段内在两路相机交替可见」——这是对数据如实的描述。

    合并规则（反复执行至稳定，最多 max_rounds 轮）
    ----------------------------------------------
    1. **A-B-A 抖动**：若 `groups[i]` 时长 ≤ `max_blip_s` 且其前后两组的相机集合
       相同，则三组合并为一组，相机集合取并集（这是判据的核心——只认"夹在
       相同相机之间"的短段，不会误并 A→B→C 这种真实的前后相继）。
    2. **同集合相邻合并**：相机集合相同的相邻两组直接合并。

    注意：真实的前后相继（A 段结束后进入 B 段，且顺序无反复）**不受影响**，
    因为不存在"夹在相同相机之间的短段"。

    rows 之外的调用方若只想看相机链，可直接取返回 groups 的 `cameras` 字段。

    返回 `(groups, info)`：
      groups 每项 = {"cameras": [按首次出现排序], "enter", "exit", "duration_s",
                     "frames", "segment_count", "multi_view"}
      info = {"raw_segments", "groups", "absorbed_segments", "multi_view_groups", "rounds"}

    单路相机的时间明细（进入/离开/驻留/命中帧数）用 `summarize_per_camera()`。
    """
    raw = len(segments)
    groups = [
        {
            "cameras": [seg["camera"]],
            "enter": seg["enter"],
            "exit": seg["exit"],
            "duration_s": round(seg["exit"] - seg["enter"], 2),
            "frames": seg.get("frames", 1),
            "segment_count": 1,
        }
        for seg in segments
    ]
    if len(groups) < 3:
        return _finalize_groups(groups, raw, 0)

    rounds = 0
    absorbed = 0
    for _ in range(max_rounds):
        changed = 0

        # 规则 1：A-B-A 抖动（夹在相同相机集合之间的短段）
        i = 1
        while i < len(groups) - 1:
            prev, cur, nxt = groups[i - 1], groups[i], groups[i + 1]
            if (cur["duration_s"] <= max_blip_s
                    and set(prev["cameras"]) == set(nxt["cameras"])):
                union_cams = _ordered_union(prev["cameras"], cur["cameras"])
                if prev["duration_s"] > max_blip_s:
                    # 前一组是**真实长停留**（不是抖动的一部分）：绝不能把它并进来 ——
                    # 否则会得出"他在 B 相机也待了 133 秒"的错误结论。
                    # 此时只把 cur 与 nxt 合成一个新的多视角组，起点从 cur 算起。
                    merged = {
                        "cameras": union_cams,
                        "enter": cur["enter"],
                        "exit": nxt["exit"],
                        "frames": cur["frames"] + nxt["frames"],
                        "segment_count": cur["segment_count"] + nxt["segment_count"],
                    }
                    merged["duration_s"] = round(merged["exit"] - merged["enter"], 2)
                    groups[i] = merged
                    del groups[i + 1]
                else:
                    prev["cameras"] = union_cams
                    prev["exit"] = nxt["exit"]
                    prev["frames"] += cur["frames"] + nxt["frames"]
                    prev["segment_count"] += cur["segment_count"] + nxt["segment_count"]
                    prev["duration_s"] = round(prev["exit"] - prev["enter"], 2)
                    del groups[i : i + 2]
                changed += 1
                absorbed += 2
                continue
            i += 1

        # 规则 1b：短段的相机已经属于前一组 —— 直接吸收。
        #
        # 为什么需要它：规则 1 把首个 A-B-A 合成 {A,B} 之后，其后残留的 B、A 短段
        # 与"下一组"的相机集合已不再相等（{A,B} vs {A}），规则 1 便不再命中，
        # 长交替链会在中途停下。判据改为"子集"即可把整条链收干净。
        i = 1
        while i < len(groups):
            prev, cur = groups[i - 1], groups[i]
            if (cur["duration_s"] <= max_blip_s
                    and set(cur["cameras"]) <= set(prev["cameras"])):
                prev["exit"] = max(prev["exit"], cur["exit"])
                prev["frames"] += cur["frames"]
                prev["segment_count"] += cur["segment_count"]
                prev["duration_s"] = round(prev["exit"] - prev["enter"], 2)
                del groups[i]
                changed += 1
                absorbed += 1
                continue
            i += 1

        # 规则 2：相机集合相同的相邻组
        j = 1
        while j < len(groups):
            if set(groups[j]["cameras"]) == set(groups[j - 1]["cameras"]):
                a, b = groups[j - 1], groups[j]
                a["exit"] = b["exit"]
                a["frames"] += b["frames"]
                a["segment_count"] += b["segment_count"]
                a["duration_s"] = round(a["exit"] - a["enter"], 2)
                del groups[j]
                changed += 1
                absorbed += 1
                continue
            j += 1

        rounds += 1
        if not changed:
            break

    return _finalize_groups(groups, raw, absorbed, rounds)


def _ordered_union(first: list[str], second: list[str]) -> list[str]:
    """按"先出现者在前"的规则取并集，保证结果稳定可测。"""
    out = list(first)
    for cam in second:
        if cam not in out:
            out.append(cam)
    return out


def _finalize_groups(
    groups: list[dict],
    raw: int,
    absorbed: int,
    rounds: int = 0,
) -> tuple[list[dict], dict]:
    """补齐 multi_view 统计并生成 info。per_camera 明细由 summarize_per_camera() 单独提供。"""
    for g in groups:
        g["multi_view"] = len(g["cameras"]) > 1
    info = {
        "raw_segments": raw,
        "groups": len(groups),
        "absorbed_segments": absorbed,
        "multi_view_groups": sum(1 for g in groups if g["multi_view"]),
        "rounds": rounds,
    }
    return groups, info


def summarize_per_camera(segments: list[dict]) -> dict[str, dict]:
    """
    按相机汇总停留信息，供「通行链」列表直接渲染。

    返回 {camera: {"visits", "enter", "exit", "frames", "duration_s"}}：
    - visits：该相机被打断成多少段（抖动会使这个数字很大，属预期）
    - duration_s：各段时长之和（真实驻留时间，与墙钟跨度区分开）
    """
    out: dict[str, dict] = {}
    for seg in segments:
        cam = seg["camera"]
        item = out.get(cam)
        if item is None:
            out[cam] = {
                "visits": 1,
                "enter": seg["enter"],
                "exit": seg["exit"],
                "frames": seg.get("frames", 1),
                "duration_s": round(seg["exit"] - seg["enter"], 2),
            }
        else:
            item["visits"] += 1
            item["enter"] = min(item["enter"], seg["enter"])
            item["exit"] = max(item["exit"], seg["exit"])
            item["frames"] += seg.get("frames", 1)
            item["duration_s"] = round(item["duration_s"] + seg["exit"] - seg["enter"], 2)
    for cam, item in out.items():
        out[cam] = {**item, "camera": cam}
    return out


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
