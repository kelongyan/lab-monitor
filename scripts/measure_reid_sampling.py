"""measure_reid_sampling.py — ReID 采样间隔 reid_every_n 的受控实验（Tier A 前置 P0）。

回答一个问题：在当前 22 路真实语料上，reid_every_n 取多少才能让
"大多数轨迹攒够身份确认/注册所需特征数"，代价（帧率损失）是多大？

背景（量化验证 main.py 性能注释里的公式）：
  - 特征填充时间 = buffer_size × reid_every_n / 实际处理 fps
  - 身份**注册**（匿名新人）需要 buffer_size=8 个特征；
  - 与已有身份**多帧确认**需要 ≥2 特征 + 连续 3 次一致命中（即 ≥4 个采样点）；
  - **底库实名命中只需第 1 个特征**（PersonnelGallery.match 在第一个采样点就跑）。
  - 22 路实测单路 ~1.77fps、轨迹中位 7.1s（≈12 处理帧）→ R=10 时一条中位
    轨迹只能采到 ~2 个特征：注册不可能，确认也悬。

方法：选 N 路本地视频，复用与生产完全一致的 detector/ReID/ByteTrack/pipeline
（差异只有：临时库、无通知器、无 Web、personnel=None），对每个 R 独立跑
--seconds 秒，统计：
  - 每路实际处理帧率（R 越小 ReID 越频繁，fps 掉多少）
  - 每条轨迹采到的特征数分布（≥2 / ≥4 / ≥8 的占比）
  - 轨迹起点 → 第 k 个特征的耗时（确认/注册延迟）
  - 实际发生的注册 / 归并 / 歧义 / 确认次数

结果追加写入 outputs/reports/reid_sampling_sweep.json 并打印对照表。
全程只写临时目录，生产 outputs/lab_monitor.db 零接触（依赖 src.db 惰性单例）。

用法：
  ./.venv/Scripts/python.exe scripts/measure_reid_sampling.py \
      --cameras reg_01 reg_08 reg_13 --seconds 75 --every-n 1 3 10
"""

from __future__ import annotations

import io
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

# 必须在导入 cv2 之前设置（与 main.py 同源）
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


class Recorder:
    """跨 pipeline 线程收集轨迹 / 特征事件（锁保护的极简记账）。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.seen: dict[tuple, list[float]] = defaultdict(list)      # (cam,tid) → 出现时刻
        self.features: dict[tuple, list[float]] = defaultdict(list)  # (cam,tid) → 采样时刻
        self.confirms: list[tuple[str, int, str]] = []               # (cam, tid, gid)
        self.registrations = {"new": 0, "merged": 0, "ambiguous": 0}

    def see(self, cam: str, tids: list[int]) -> None:
        now = time.monotonic()
        with self.lock:
            for tid in tids:
                self.seen[(cam, tid)].append(now)

    def feature(self, cam: str, tid: int) -> None:
        with self.lock:
            self.features[(cam, tid)].append(time.monotonic())

    def confirm(self, cam: str, tid: int, gid: str) -> None:
        with self.lock:
            self.confirms.append((cam, tid, gid))

    def register(self, resolution) -> None:
        key = ("new" if resolution.is_new
               else "ambiguous" if resolution.status == "ambiguous" else "merged")
        with self.lock:
            self.registrations[key] += 1


def instrument(pipeline, cam: str, recorder: Recorder) -> None:
    """给已构造、未启动的 pipeline 实例挂 spy（实例属性遮蔽方法，无侵入）。"""
    tracker = pipeline._tracker
    orig_update = tracker.update

    def spy_update(detections, shape):
        tracks = orig_update(detections, shape)
        recorder.see(cam, [t["track_id"] for t in tracks])
        return tracks

    tracker.update = spy_update

    validator = pipeline._validator
    orig_add = validator.add_feature
    orig_match = validator.get_confirmed_match

    def spy_add(tid, feat):
        recorder.feature(cam, tid)
        return orig_add(tid, feat)

    def spy_match(tid, gallery, **kwargs):
        gid = orig_match(tid, gallery, **kwargs)
        if gid:
            recorder.confirm(cam, tid, gid)
        return gid

    validator.add_feature = spy_add
    validator.get_confirmed_match = spy_match


def run_config(every_n: int, cameras: dict[str, str], seconds: float,
               detector, reid_extractor, device: str) -> dict:
    """跑一个 R 配置，返回指标 dict。模型池跨配置复用（推理无状态）。"""
    import threading

    from src.alerter import AlertManager, AlertBroadcaster
    from src.calibrator import TransitCalibrator
    from src.db import Database
    from src.frame_hub import FrameHub
    from src.identity_store import IdentityStore
    from src.pipeline import CameraPipeline
    from src.topology import CameraTopology, TopologyValidationError

    with tempfile.TemporaryDirectory(prefix=f"reid_sweep_r{every_n}_") as temp:
        temp_dir = Path(temp)
        database = Database(temp_dir / "sweep.db")
        try:
            store = IdentityStore(database=database,
                                  feature_space=reid_extractor.feature_space)
            try:
                topology = CameraTopology(
                    ROOT / "config" / "topology.json",
                    allowed_camera_ids=set(cameras))
            except TopologyValidationError:
                # 子集相机触发拓扑校验失败 → 空拓扑即可（本实验不测告警）
                topology = CameraTopology(
                    (ROOT / "config" / "topology.json").with_name(".sweep-disabled"),
                    allowed_camera_ids=set(cameras))
            frame_hub = FrameHub(jpeg_quality=50)
            frame_hub.register_cameras(cameras)
            broadcaster = AlertBroadcaster(delivery_enabled=False)
            alert_manager = AlertManager(
                alert_log=temp_dir / "alerts.jsonl",
                notifier=None,               # 绝不允许实验触发真实外部通知
                broadcaster=broadcaster,
                identity_store=store,
                screenshot_dir=temp_dir / "screenshots",
                database=database,
            )
            calibrator = TransitCalibrator(temp_dir / "transit_stats.json",
                                           valid_edges=topology.edges())

            recorder = Recorder()
            orig_register = store.register_if_new

            def spy_register(feat):
                resolution = orig_register(feat)
                recorder.register(resolution)
                return resolution

            store.register_if_new = spy_register

            pipelines = []
            for cam, source in cameras.items():
                p = CameraPipeline(
                    camera_id=cam, source=source,
                    detector=detector, reid_extractor=reid_extractor,
                    identity_store=store, topology=topology,
                    alert_manager=alert_manager,
                    screenshot_dir=temp_dir / "screenshots",
                    frame_hub=frame_hub, calibrator=calibrator,
                    display=False,
                    detect_every_n=1 if device == "cuda" else 3,
                    reid_every_n=every_n,
                    frame_rate_cap=10.0,
                    personnel=None,      # 隔离变量：只测匿名注册/确认链路
                )
                instrument(p, cam, recorder)
                pipelines.append(p)

            t0 = time.monotonic()
            for p in pipelines:
                p.start()
            time.sleep(seconds)
            for p in pipelines:
                p.stop()
            for p in pipelines:
                p.join(timeout=15)
            elapsed = time.monotonic() - t0
            try:
                store.flush()
            except Exception:
                pass
            # 与 main.py 关停路径同款：不关的话 alerts.jsonl 句柄还开着，
            # Windows 上 TemporaryDirectory 清理会 WinError 32
            alert_manager.close()
            calibrator.flush()

            with recorder.lock:
                seen = {k: list(v) for k, v in recorder.seen.items()}
                feats = {k: list(v) for k, v in recorder.features.items()}
                confirms = list(recorder.confirms)
                registrations = dict(recorder.registrations)

            per_track_feats = [len(feats.get(k, [])) for k in seen]
            spans = [(max(v) - min(v)) for v in seen.values() if len(v) >= 2]
            n = max(1, len(seen))
            # 轨迹起点 → 第 k 个特征的耗时（确认/注册的最低时间预算）
            latencies: dict[str, list[float]] = {"2": [], "4": [], "8": []}
            for key, times in feats.items():
                start = min(seen.get(key) or times)
                for k in (2, 4, 8):
                    if len(times) >= k:
                        latencies[str(k)].append(times[k - 1] - start)

            per_cam = {}
            for p in pipelines:
                per_cam[p.camera_id] = {
                    "processed_frames": p._frame_idx,
                    "fps": round(p._frame_idx / elapsed, 2),
                }

            return {
                "reid_every_n": every_n,
                "elapsed_s": round(elapsed, 1),
                "per_camera": per_cam,
                "mean_fps": round(statistics.mean(c["fps"] for c in per_cam.values()), 2),
                "tracks_total": len(seen),
                "tracks_with_any_feature": sum(1 for c in per_track_feats if c >= 1),
                "pct_tracks_ge2": round(100 * sum(1 for c in per_track_feats if c >= 2) / n, 1),
                "pct_tracks_ge4": round(100 * sum(1 for c in per_track_feats if c >= 4) / n, 1),
                "pct_tracks_ge8": round(100 * sum(1 for c in per_track_feats if c >= 8) / n, 1),
                "median_features_per_track": statistics.median(per_track_feats) if per_track_feats else 0,
                "median_track_span_s": round(statistics.median(spans), 2) if spans else 0,
                "median_time_to_2nd_feature_s": round(statistics.median(latencies["2"]), 2) if latencies["2"] else None,
                "median_time_to_8th_feature_s": round(statistics.median(latencies["8"]), 2) if latencies["8"] else None,
                "multi_frame_confirms": len(confirms),
                "registrations": registrations,
                "identities_in_store": len(store.all_ids()),
            }
        finally:
            database.close()


def main() -> int:
    import argparse

    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="ReID 采样间隔受控实验")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="相机 id 列表（默认全部 22 路 —— 单路帧率由总线程数决定，"
                             "只跑子集得到的 fps 与生产不可比）")
    parser.add_argument("--seconds", type=float, default=75.0,
                        help="每个配置的运行时长（默认 75s）")
    parser.add_argument("--every-n", type=int, nargs="+", default=[1, 3, 10],
                        help="要对比的 reid_every_n 取值")
    args = parser.parse_args()

    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    cameras = {cam: sources[cam] for cam in (args.cameras or list(sources))}
    missing = [c for c, p in cameras.items() if not Path(p).exists()]
    if missing:
        log(f"✗ 视频文件缺失: {missing}")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    log(f"device={device} · cameras={list(cameras)} · seconds/config={args.seconds}")

    from src.detector import PersonDetector
    from src.model_pool import PooledDetector, PooledReIDExtractor, resolve_pool_size
    from src.reid import build_reid_extractor

    pool_size = resolve_pool_size(device, len(cameras))
    detector = PooledDetector(
        lambda: PersonDetector(model_name="yolov8n.pt", conf_thresh=0.4, device=device),
        size=pool_size)
    reid_extractor = PooledReIDExtractor(
        lambda: build_reid_extractor(device=device), size=pool_size)
    log(f"模型池就绪: detector×{detector.pool_size} / reid×{reid_extractor.pool_size}")

    results = []
    for every_n in args.every_n:
        log(f"\n===== reid_every_n = {every_n} 运行中 … =====")
        result = run_config(every_n, cameras, args.seconds, detector,
                            reid_extractor, device)
        results.append(result)
        log(f"  实际帧率 {result['mean_fps']} fps/路 · 轨迹 {result['tracks_total']} 条 · "
            f"特征中位 {result['median_features_per_track']}")
        log(f"  ≥2特征(可确认) {result['pct_tracks_ge2']}% · "
            f"≥4特征(3次一致) {result['pct_tracks_ge4']}% · "
            f"≥8特征(可注册) {result['pct_tracks_ge8']}%")
        log(f"  多帧确认 {result['multi_frame_confirms']} 次 · "
            f"注册 {result['registrations']} · store 身份 {result['identities_in_store']}")

    report_path = ROOT / "outputs" / "reports" / "reid_sampling_sweep.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"runs": []}
    history["runs"].append({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "cameras": list(cameras),
        "seconds_per_config": args.seconds,
        "results": results,
    })
    history["runs"] = history["runs"][-10:]
    report_path.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    log(f"\n结果已追加 → {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
