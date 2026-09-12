"""
search.py — 人员视频检索（能力二）的核心逻辑，HTTP 层只做参数解析与鉴权。

两个入口（worklist 4.2 / 4.4）：
  aggregate_identity_assets()   按编号检索：某个身份出现过的全部视频片段
  rank_identities_by_feature()  以图搜人：特征 → 与全库比对 → Top-K 身份

三条必须遵守的语义（都来自实测教训，改之前先看对应脚本/测试）：
  1. **双时间轴不能混用**：`timestamp`（墙钟）用于跨相机排序；
     `video_ts`（视频内秒数）用于回放定位。素材循环播放会把墙钟拉成几天，
     拿墙钟去视频里 seek 必然定位到错位置。
  2. **循环折叠复用 src/trajectory.py**，不要重写 —— 它有 19 个测试用例保护。
  3. **检索阈值独立于实时阈值**：离线检索返回 Top-K 排序可以放宽；
     实时匹配是二值判定必须保守。两者共用同一个常量是设计错误。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.trajectory import build_segments, collapse_loop_segments

logger = logging.getLogger("search")

APPEARANCE_QUERY_LIMIT = 20000


def _asset_index(database) -> tuple[dict, dict]:
    """
    构建 asset_id → 资产行 与 camera_id → 资产行 的索引。

    同一相机可能有两行（源文件 + 低清转码产物），按 camera_id 取时优先返回
    「有实测帧数」的那一行；trajectory 旧数据没有 asset_id，只能按相机兜底。
    """
    assets = database.list_video_assets()
    by_id: dict[int, dict] = {}
    by_cam: dict[str, list[dict]] = {}
    for asset in assets:
        if asset.get("asset_id") is not None:
            by_id[asset["asset_id"]] = asset
        by_cam.setdefault(asset["camera_id"], []).append(asset)
    for cam in by_cam:
        by_cam[cam].sort(key=lambda a: (bool(a.get("frames_real")), a["asset_id"] or 0),
                         reverse=True)
    return by_id, by_cam


def _resolve_asset(row: dict, by_id: dict, by_cam: dict) -> dict | None:
    asset_id = row.get("asset_id")
    if asset_id is not None and asset_id in by_id:
        return by_id[asset_id]
    candidates = by_cam.get(row["camera"]) or []
    return candidates[0] if candidates else None


def aggregate_identity_assets(
    database,
    global_id: str,
    *,
    start: float | None = None,
    end: float | None = None,
    camera: str | None = None,
    split_gap_s: float = 30.0,
    collapse: bool = True,
    known_cameras: set[str] | None = None,
) -> dict:
    """
    聚合某身份出现过的全部视频片段（worklist 4.2）。

    known_cameras：当前仍在 sources.json 里的相机集合（worklist 4.3）。
    库里有 rnd_03 的 6,447 条孤儿轨迹（历史配置残留），不按此过滤的话
    检索会返回一个已下线的相机，前端点进去就是 404。
    """
    total, rows = database.query_trajectory(
        global_id, start, end, camera, APPEARANCE_QUERY_LIMIT
    )

    if known_cameras:
        before = len(rows)
        rows = [row for row in rows if row["camera"] in known_cameras]
        dropped = before - len(rows)
    else:
        dropped = 0

    # 折叠前的逐相机原始命中数 —— 用于计算每个资产的循环倍数 loop_factor。
    # 语料只有 45 分钟却累计了 22 万条 appearance，同一段像素平均被循环记录数百次；
    # 不标注倍数，用户会得到"这个人在 rnd_01 出现了 3 万次"的度量假象。
    raw_counts: dict[str, int] = {}
    for row in rows:
        raw_counts[row["camera"]] = raw_counts.get(row["camera"], 0) + 1

    segments = build_segments(rows, split_gap_s)
    if collapse:
        segments, rows, loop_info = collapse_loop_segments(segments, rows)
    else:
        loop_info = {"detected": False, "method": None, "period_segments": 0,
                     "loops": 1, "raw_segments": len(segments)}

    kept_counts: dict[str, int] = {}
    for row in rows:
        kept_counts[row["camera"]] = kept_counts.get(row["camera"], 0) + 1

    by_id, by_cam = _asset_index(database)

    grouped: dict[tuple, dict] = {}
    for seg in segments:
        # 用段的代表行定位资产（段内 asset_id 理论上一致；异常时按首行）
        probe = next((r for r in rows
                      if r["camera"] == seg["camera"]
                      and seg["enter"] <= r["time"] <= seg["exit"]), None)
        asset_row = _resolve_asset(probe or {"camera": seg["camera"]}, by_id, by_cam)
        key = (asset_row["asset_id"] if asset_row else None, seg["camera"])
        entry = grouped.setdefault(key, {
            "asset": asset_row,
            "camera_id": seg["camera"],
            "asset_id": asset_row["asset_id"] if asset_row else None,
            "file": (asset_row or {}).get("rel_path"),
            "duration_real": (asset_row or {}).get("duration_real"),
            "position_known": False,
            "video_first_ts": None,
            "video_last_ts": None,
            "hit_count": 0,
            "loop_factor": 1.0,
            "segments": [],
        })
        entry["hit_count"] += seg["frames"]
        entry["segments"].append({
            "enter": seg["enter"],
            "exit": seg["exit"],
            "duration_s": seg["duration_s"],
            "video_enter_ts": None,
            "video_exit_ts": None,
        })

    # 段的视频内坐标：用落在该段时间窗内的轨迹行求 min/max
    for (asset_key, _cam), entry in grouped.items():
        for seg in entry["segments"]:
            inside = [r for r in rows
                      if r["camera"] == entry["camera_id"]
                      and seg["enter"] <= r["time"] <= seg["exit"]
                      and r.get("video_ts") is not None]
            if inside:
                seg["video_enter_ts"] = round(min(r["video_ts"] for r in inside), 2)
                seg["video_exit_ts"] = round(max(r["video_ts"] for r in inside), 2)
                entry["position_known"] = True
                if entry["video_first_ts"] is None:
                    entry["video_first_ts"] = seg["video_enter_ts"]
                entry["video_last_ts"] = seg["video_exit_ts"]

    for entry in grouped.values():
        kept = kept_counts.get(entry["camera_id"], 0)
        raw = raw_counts.get(entry["camera_id"], 0)
        entry["loop_factor"] = round(raw / kept, 1) if kept and raw else 1.0

    assets = sorted(grouped.values(),
                    key=lambda item: -(item["video_last_ts"] or 0))
    return {
        "global_id": global_id,
        "total_appearances": total,
        "returned_rows": len(rows),
        "dropped_offline_rows": dropped,
        "position_known_rows": sum(1 for r in rows if r.get("video_ts") is not None),
        "loop": loop_info,
        "assets": assets,
        "asset_count": len(assets),
        "truncated": total > APPEARANCE_QUERY_LIMIT,
    }


def rank_identities_by_feature(
    store,
    feature: np.ndarray,
    top_k: int = 5,
) -> dict:
    """
    以图搜人（worklist 4.4）：把查询特征与全库比对，返回 Top-K 身份。

    用 `build_match_context()` 保证 query 与 gallery 在同一坐标系（中心化）。
    返回排序结果而非二值判定 —— **离线检索的阈值语义与实时匹配独立**：
    这里照常给出 `matched` 字段（用实时阈值）方便前端标色，
    但调用方不应据此丢弃排名靠后但分数可观的候选。
    """
    from src.reid_config import REID_MATCH_THRESHOLD

    context = store.build_match_context()
    query = context.prepare(np.asarray(feature, dtype=np.float32))

    best: dict[str, float] = {}
    for gid, vector in context.gallery:
        score = float(query @ vector)
        if score > best.get(gid, -2.0):
            best[gid] = score
    ranked = sorted(best.items(), key=lambda item: -item[1])[:top_k]

    return {
        "galleries_compared": len(best),
        "threshold": REID_MATCH_THRESHOLD,
        "matches": [
            {"global_id": gid, "score": round(score, 4),
             "matched": score >= REID_MATCH_THRESHOLD}
            for gid, score in ranked
        ],
    }


def best_crop(detections: list[list[float]]) -> list[float] | None:
    """
    从一张图的多个人体框里挑"最可能是查询对象"的那个：取面积最大者。
    纯启发式 —— 上传图里有多人时无法自动知道用户指的是谁，取最大是
    最不坏的默认；同时把 person_count 返回给前端，由它提示用户框选。
    """
    if not detections:
        return None
    return max(detections, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))[:4]
