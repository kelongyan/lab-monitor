"""
calibrator.py — 通行时间自动校准器
记录人员在摄像头间的实际通行时间，统计分布后自动修正 topology 时间窗口
持久化到 outputs/transit_stats.json，重启后加载继续积累
"""

import json
import math
import logging
import threading
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("calibrator")

# 最少需要多少条记录才启用自动修正（样本不足时保持原始配置）
MIN_SAMPLES = 5
# 时间窗口 = mean + Z * std（Z=1.65 覆盖约95%）
Z_SCORE = 1.65


class TransitCalibrator:
    """
    使用方法：
      - 人员从 cam_A 离开时记录离开时间戳
      - 人员在 cam_B 出现并匹配到同一身份时调用 record_transit()
      - 调用 calibrated_window() 获取动态修正后的 (expected_seconds, tolerance)
    """

    def __init__(self, stats_path: str | Path = "outputs/transit_stats.json"):
        self._lock = threading.Lock()
        self._path = Path(stats_path)
        # key: "cam_A→cam_B"  value: list of actual transit seconds
        self._data: dict[str, list[float]] = defaultdict(list)
        self._load()

    # ------------------------------------------------------------------ #
    # 记录实际通行时间                                                       #
    # ------------------------------------------------------------------ #

    def record_transit(self, cam_from: str, cam_to: str, actual_seconds: float) -> None:
        """人员从 cam_from 走到 cam_to 的实际耗时（秒）"""
        if actual_seconds <= 0 or actual_seconds > 3600:
            return  # 异常值过滤
        key = f"{cam_from}→{cam_to}"
        with self._lock:
            self._data[key].append(actual_seconds)
            # 保留最近200条，防止文件无限增大
            if len(self._data[key]) > 200:
                self._data[key] = self._data[key][-200:]
        logger.debug("记录通行时间 %s: %.1fs（共 %d 条）", key, actual_seconds, len(self._data[key]))
        self._save()

    # ------------------------------------------------------------------ #
    # 获取校准后的时间窗口                                                   #
    # ------------------------------------------------------------------ #

    def calibrated_window(
        self,
        cam_from: str,
        cam_to: str,
        default_expected: float,
        default_tolerance: float,
    ) -> tuple[float, float]:
        """
        返回 (expected_seconds, tolerance_seconds)
        样本不足时返回原始配置，样本充足时返回统计校准值
        """
        key = f"{cam_from}→{cam_to}"
        with self._lock:
            samples = list(self._data.get(key, []))

        if len(samples) < MIN_SAMPLES:
            return default_expected, default_tolerance

        mean = sum(samples) / len(samples)
        # 使用 Bessel 校正（样本方差），避免小样本下低估标准差
        variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
        std = math.sqrt(variance)

        calibrated_expected = mean
        calibrated_tolerance = max(Z_SCORE * std, 5.0)  # 最小5秒宽容
        logger.info(
            "校准 %s: mean=%.1fs std=%.1fs → window=[%.1f, %.1f]s (n=%d)",
            key, mean, std,
            calibrated_expected - calibrated_tolerance,
            calibrated_expected + calibrated_tolerance,
            len(samples),
        )
        return calibrated_expected, calibrated_tolerance

    def stats(self) -> dict:
        """返回所有路径的统计摘要，供 Web API 展示"""
        with self._lock:
            data = dict(self._data)
        result = {}
        for key, samples in data.items():
            if not samples:
                continue
            mean = sum(samples) / len(samples)
            # Bessel 校正：与 calibrated_window() 保持一致
            n = len(samples)
            variance = sum((x - mean) ** 2 for x in samples) / max(n - 1, 1)
            result[key] = {
                "count": len(samples),
                "mean_seconds": round(mean, 1),
                "std_seconds": round(math.sqrt(variance), 1),
                "min_seconds": round(min(samples), 1),
                "max_seconds": round(max(samples), 1),
            }
        return result

    # ------------------------------------------------------------------ #
    # 持久化                                                                #
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = dict(self._data)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("保存通行统计失败: %s", e)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            with self._lock:
                for k, v in raw.items():
                    self._data[k] = v
            total = sum(len(v) for v in self._data.values())
            logger.info("加载通行统计：%d 条路径，共 %d 条记录", len(self._data), total)
        except Exception as e:
            logger.error("加载通行统计失败: %s", e)
