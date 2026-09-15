"""
tests/test_trajectory.py — 循环播放折叠逻辑

背景：本地 MP4 素材会自动循环播放（pipeline._run_file），部分素材只有 40 秒。
一个在镜头里走一趟的人会被反复记录成上千段「A→B→A→B…」。不折叠的话，
前端画出来就是「这个人在两点之间来回跑了 4670 次」的假象。

两个折叠策略：
1. period   —— 严格周期，截断到第一轮，时间线连续（播放动画不瞬移）
2. fingerprint —— 按画面位置去重，覆盖非严格周期的场景（兜底，不丢相机）
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import _collapse_loop_period, _collapse_by_fingerprint
from src.trajectory import merge_flicker_segments, summarize_per_camera


def _seg(cam: str, cx: float = 0.0, cy: float = 0.0, half: float = 5.0) -> dict:
    """构造一个以 (cx, cy) 为中心、边长 2*half 的框。

    用中心点而非左上角，避免写出 x1 < x0 这种非法 bbox。
    """
    return {"camera": cam, "bbox_start": [cx - half, cy - half, cx + half, cy + half]}


class CollapsePeriodTest(unittest.TestCase):
    """周期检测：判据是差分 seq[i] == seq[i-period]，不是相位对齐。"""

    def test_strict_alternation(self):
        seq = ["a", "b"] * 50
        self.assertEqual(_collapse_loop_period(seq), 2)

    def test_three_step_cycle(self):
        seq = ["a", "b", "c"] * 40
        self.assertEqual(_collapse_loop_period(seq), 3)

    def test_single_camera_repeat_is_period_one(self):
        self.assertEqual(_collapse_loop_period(["a"] * 100), 1)

    def test_phase_flip_still_detected(self):
        """相位翻转是真实存在的：A→B→…→B→A→… 翻转点只有一两处。

        这正是朴素的 `seq[i] == seq[i % period]` 判据会全崩的场景
        （实测 9340 段严格交替序列在该判据下只有 5% 匹配率）。
        """
        seq = ["a", "b"] * 30 + ["b", "a"] * 30 + ["a", "b"] * 30
        self.assertEqual(_collapse_loop_period(seq), 2)

    def test_no_period_returns_none(self):
        """无固定重复的序列不应被判为周期。

        注意别用 `seq = [...] * 5` 这种写法构造——乘出来的序列天然带周期，
        会让它自己成为反例。
        """
        seq = ["a", "b", "c", "d", "e", "a", "c", "b", "e", "d",
               "b", "e", "d", "a", "c", "e", "d", "c", "b", "a",
               "d", "a", "e", "c", "b", "c", "e", "a", "d", "b",
               "e", "c", "a", "b", "d", "b", "d", "e", "a", "c"]
        self.assertIsNone(_collapse_loop_period(seq))

    def test_too_short_returns_none(self):
        for seq in ([], ["a"], ["a", "b"], ["a", "b", "a"]):
            with self.subTest(seq=seq):
                self.assertIsNone(_collapse_loop_period(seq))

    def test_noisy_cycle_tolerated_at_90pct(self):
        """90% 阈值：120 段里允许约 12 处抖动。"""
        seq = ["a", "b"] * 60
        for i in (5, 17, 40, 91):  # 4 处漏检
            seq[i] = "a" if seq[i] == "b" else "b"
        self.assertEqual(_collapse_loop_period(seq), 2)

    def test_heavy_noise_rejected(self):
        """抖动超过 10% 就不认定为周期——宁可不折叠也不能折错。

        随机打乱 1/3 的位置（种子固定保证可复现）。注意不能只翻偶数位，
        那会把 A→B 交替变成全 B，反而成了稳定的 period=1。
        """
        import random
        rnd = random.Random(1234)
        seq = ["a", "b"] * 60
        for i in rnd.sample(range(len(seq)), len(seq) // 3):
            seq[i] = "b" if seq[i] == "a" else "a"
        self.assertIsNone(_collapse_loop_period(seq))

    def test_period_upper_bound_caps_work(self):
        """周期上限 200，避免超长序列上 O(n×period) 打爆请求。"""
        seq = [f"cam{i:03d}" for i in range(500)] * 4  # 周期 500 > 上限
        self.assertIsNone(_collapse_loop_period(seq))


class CollapseFingerprintTest(unittest.TestCase):
    """指纹去重：按 (camera, 首帧 bbox 中心 20px 量化) 保留首次出现的段。"""

    def test_identical_segments_collapsed(self):
        segs = [_seg("a", 100, 200)] * 10
        self.assertEqual(_collapse_by_fingerprint(segs), [0])

    def test_different_positions_kept(self):
        """位置差 > 20px 视为不同停留点。"""
        segs = [_seg("a", 0, 0), _seg("a", 100, 0), _seg("a", 200, 0)]
        self.assertEqual(_collapse_by_fingerprint(segs), [0, 1, 2])

    def test_within_quantization_bucket_collapsed(self):
        """同一停留点的检测抖动应被折叠。

        20px 量化桶的边界在 ±10、±30…，所以「同桶」要求中心偏移落在
        [-10, 10) 内，不是「差值 < 20」——跨桶的两点会被判为不同位置。
        """
        segs = [_seg("a", 0, 0), _seg("a", 5, 4), _seg("a", 9, 8)]
        self.assertEqual(_collapse_by_fingerprint(segs), [0])

    def test_across_quantization_boundary_kept(self):
        """跨桶的两点（中心相距 15px，但分属 0 号与 1 号桶）判定为不同位置。"""
        segs = [_seg("a", 9, 0), _seg("a", 11, 0)]
        self.assertEqual(_collapse_by_fingerprint(segs), [0, 1])

    def test_same_position_different_camera_kept(self):
        segs = [_seg("a", 100, 100), _seg("b", 100, 100)]
        self.assertEqual(_collapse_by_fingerprint(segs), [0, 1])

    def test_alternating_pattern_keeps_two(self):
        segs = [_seg("a", 0, 0), _seg("b", 500, 500)] * 30
        self.assertEqual(_collapse_by_fingerprint(segs), [0, 1])

    def test_malformed_bbox_degrades_to_camera_only(self):
        """bbox 异常时退化为只按相机去重，不能把整段丢掉。"""
        segs = [
            {"camera": "a", "bbox_start": None},
            {"camera": "a", "bbox_start": []},
            {"camera": "a", "bbox_start": [1, 2]},
            {"camera": "a", "bbox_start": ["x", "y", "z", "w"]},
            {"camera": "b", "bbox_start": [0, 0, 10, 10]},
        ]
        self.assertEqual(_collapse_by_fingerprint(segs), [0, 4])

    def test_empty_input(self):
        self.assertEqual(_collapse_by_fingerprint([]), [])

    def test_missing_bbox_key(self):
        segs = [{"camera": "a"}, {"camera": "a"}, {"camera": "b"}]
        self.assertEqual(_collapse_by_fingerprint(segs), [0, 2])

    def test_keeps_first_occurrence_order(self):
        """保留的索引必须递增 —— 前端依赖顺序画连线。"""
        segs = [_seg("c", 0, 0), _seg("a", 0, 0), _seg("c", 0, 0),
                _seg("b", 0, 0), _seg("a", 0, 0)]
        keep = _collapse_by_fingerprint(segs)
        self.assertEqual(keep, [0, 1, 3])
        self.assertEqual(keep, sorted(keep))


class FlickerMergeTest(unittest.TestCase):
    """视图组合并：把「循环素材 + 多相机并行确认」造成的 A-B-A 抖动压成多视角组。

    实测形态（5a7991c7）：一段正常长停留之后，两条走廊相机每 0.2~0.5 秒交替写入
    确认记录，段时长几乎全为 0。直接连折线会得到锯齿，必须压成视图组。
    """

    @staticmethod
    def _s(cam: str, enter: float, exit_: float, frames: int = 1) -> dict:
        return {"camera": cam, "enter": enter, "exit": exit_, "frames": frames,
                "duration_s": round(exit_ - enter, 2)}

    def test_aba_blip_collapses_to_multiview(self):
        """长停留与抖动窗口应被**分开**：前者是真实驻留，后者是多视角窗口。"""
        segs = [
            self._s("rnd_04", 0.0, 132.7, 300),
            self._s("rnd_22", 132.9, 132.9),
            self._s("rnd_04", 133.2, 133.2),
            self._s("rnd_22", 133.4, 133.4),
            self._s("rnd_04", 133.8, 134.0, 5),
        ]
        groups, info = merge_flicker_segments(segs)
        self.assertEqual(info["raw_segments"], 5)
        self.assertEqual(info["groups"], 2)
        # 第一组：单独的长停留，不能被并进抖动窗口
        self.assertEqual(groups[0]["cameras"], ["rnd_04"])
        self.assertFalse(groups[0]["multi_view"])
        self.assertEqual(groups[0]["exit"], 132.7)
        self.assertEqual(groups[0]["frames"], 300)
        # 第二组：抖动窗口压成一个多视角组
        self.assertEqual(groups[1]["cameras"], ["rnd_04", "rnd_22"])
        self.assertTrue(groups[1]["multi_view"])
        self.assertEqual(groups[1]["enter"], 132.9)
        self.assertEqual(groups[1]["exit"], 134.0)
        self.assertEqual(groups[1]["frames"], 1 + 1 + 1 + 5)
        self.assertEqual(info["multi_view_groups"], 1)
        self.assertEqual(sum(g["frames"] for g in groups), 300 + 8)

    def test_sequential_cameras_are_not_merged(self):
        """真实的 A→B→C 前后相继不能被当成抖动合并 —— 中间不存在夹在相同相机之间的短段。"""
        segs = [
            self._s("reg_08", 0.0, 60.0, 10),
            self._s("reg_02", 120.0, 180.0, 10),
            self._s("reg_01", 240.0, 300.0, 10),
        ]
        groups, info = merge_flicker_segments(segs)
        self.assertEqual(info["groups"], 3)
        self.assertEqual([g["cameras"] for g in groups],
                         [["reg_08"], ["reg_02"], ["reg_01"]])
        self.assertFalse(any(g["multi_view"] for g in groups))

    def test_long_middle_segment_is_not_a_blip(self):
        """中间段够长说明是真实驻留，不是抖动，不应被吸收。"""
        segs = [
            self._s("a", 0.0, 10.0, 5),
            self._s("b", 11.0, 60.0, 50),
            self._s("a", 61.0, 70.0, 5),
        ]
        groups, info = merge_flicker_segments(segs, max_blip_s=2.0)
        self.assertEqual(info["groups"], 3)

    def test_alternating_chain_collapses_to_single_group(self):
        """长交替链应收敛到一个多视角组，且帧数一个不丢。"""
        segs = [self._s("a" if i % 2 == 0 else "b", float(i), float(i) + 0.1, 1)
                for i in range(10)]
        groups, info = merge_flicker_segments(segs)
        self.assertEqual(info["groups"], 1)
        self.assertEqual(groups[0]["cameras"], ["a", "b"])
        self.assertTrue(groups[0]["multi_view"])
        self.assertEqual(groups[0]["frames"], 10)
        self.assertEqual(groups[0]["segment_count"], 10)

    def test_short_input_returned_unmodified(self):
        """少于 3 段谈不上抖动，必须原样返回（不要为了统一形态而改写输入）。"""
        segs = [self._s("a", 0.0, 1.0), self._s("b", 1.0, 1.5)]
        groups, info = merge_flicker_segments(segs)
        self.assertEqual(info["groups"], 2)
        self.assertEqual(info["absorbed_segments"], 0)
        self.assertEqual(sum(g["frames"] for g in groups), 2)

    def test_frames_are_never_lost(self):
        """合并只能改变分组，不能丢帧 —— 帧数是"此人出现过"的计量。"""
        segs = [self._s("a", 0.0, 1.0, 7), self._s("b", 1.1, 1.1, 3),
                self._s("a", 1.2, 1.3, 4), self._s("b", 1.4, 1.4, 1),
                self._s("a", 1.5, 1.6, 2)]
        groups, _ = merge_flicker_segments(segs)
        self.assertEqual(sum(g["frames"] for g in groups), 7 + 3 + 4 + 1 + 2)

    def test_rounds_bounded_on_pathological_input(self):
        """病态输入（长交替链）必须在上限轮次内收敛，不能死循环。"""
        segs = [self._s("a" if i % 2 == 0 else "b", float(i), float(i)) for i in range(60)]
        groups, info = merge_flicker_segments(segs, max_rounds=8)
        self.assertLessEqual(info["rounds"], 8)
        self.assertGreaterEqual(info["groups"], 1)


class PerCameraSummaryTest(unittest.TestCase):
    """按相机汇总：duration_s 是各段时长之和，与墙钟跨度是两个不同含义。"""

    @staticmethod
    def _s(cam: str, enter: float, exit_: float, frames: int = 1) -> dict:
        return {"camera": cam, "enter": enter, "exit": exit_, "frames": frames,
                "duration_s": round(exit_ - enter, 2)}

    def test_aggregates_visits_and_sums_duration(self):
        segs = [self._s("a", 0.0, 10.0, 5), self._s("b", 20.0, 25.0, 2),
                self._s("a", 100.0, 104.0, 3)]
        out = summarize_per_camera(segs)
        self.assertEqual(out["a"]["visits"], 2)
        self.assertEqual(out["a"]["frames"], 8)
        self.assertEqual(out["a"]["enter"], 0.0)
        self.assertEqual(out["a"]["exit"], 104.0)
        # 驻留时长是 10 + 4 = 14，不是墙钟跨度 104
        self.assertEqual(out["a"]["duration_s"], 14.0)
        self.assertEqual(out["b"]["visits"], 1)

    def test_empty_input(self):
        self.assertEqual(summarize_per_camera([]), {})


if __name__ == "__main__":
    unittest.main()
