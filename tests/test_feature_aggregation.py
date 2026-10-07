"""
test_feature_aggregation.py — 主特征聚合策略（2026-09-13 新增，worklist 1.5 ②）

背景（实测）
------------
主特征原先是质量加权 EMA，α = base + (1-base)·(1-q)，新观测权重 (1-α) 随质量塌陷：

    q=1.00 → 0.150    q=0.50 → 0.075    q=0.20 → 0.030    q→0 → 0（不更新）

本语料实测 `avg_feature_quality` 只有 **0.12~0.28**（走廊素材里人是小目标）→
单次观测权重仅 3%~6%，主特征被最早的观测主导（30 次观测后最初的仍占 25%~40%）。
机制上这会让"同一个人重新入镜时新鲜特征与陈旧主特征差到阈值之外"。

⚠️ **2026-09-13 的 300 秒对照没有测出收益，曾因此默认关闭**（LAB_MONITOR_FEATURE_WINDOW=0）。
对照实验（scripts/ab_feature_window.py，同语料 / 同参数 / 各 300 秒 / 22 路）：

| | EMA(N=0) | 窗口(N=20) |
|---|---|---|
| 空库起步：新建身份数 | 13 | 14 |
| 空库起步：近邻擦肩（≥0.5） | 4 | 4 |
| 从生产库 45 身份起步：新增身份 | 1 | 1 |
| 从生产库起步：新建时最近邻相似度 | 0.6655 | 0.6655 |

两臂做了**同一个**注册决策 —— 因为 300 秒的臂每路只处理约 500 帧，同一人的观测
次数不足以让 EMA 收敛到吸引子，复现不了失效机制。

**2026-10-07 生产库取证补上了证据，默认已翻转为 20**：
  - rnd_04 一路累积 115 个身份、39 个 byte-identical（生产库 direct 测量）；
  - 重提该身份 appearance 记录的同帧特征（原 bbox/原视频/顺序解码），与存储主特征
    相似度只有 0.42~0.78 raw / **centered -0.31~-0.53** —— EMA 吸引子已漂移到
    "谁也认不出"的方向（数学本质：宽锥分布的长程平均落在锥心，与任何成员近正交）；
  - 同一人轨迹跨位置特征 raw 只有 0.5~0.6，而相邻位置 0.97 → 单向量表示必然失败，
    必须靠 bank 分层覆盖（见 _BANK_MIN_QUALITY）+ 窗口主特征保持在分布内。

本文件的定位：锁定**两种策略各自的行为**，并锁定"默认是有界窗口（20）"这条决定
（如需回退对照：LAB_MONITOR_FEATURE_WINDOW=0）。窗口策略必须在**独立进程**里
验证 —— `FEATURE_WINDOW_SIZE` 是 import 期常量，同进程改不了（这也顺带证明环境
变量真的生效）。为控制测试耗时，窗口侧的检查全部塞进**一次**子进程启动
（每启动一次要重新 import torch，约 8 秒）。
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from src.identity_store import FEATURE_WINDOW_SIZE, IdentityStore

ROOT = Path(__file__).resolve().parent.parent

PRELUDE = "\n".join([
    "import numpy as np",
    "from src.identity_store import IdentityStore, FEATURE_WINDOW_SIZE",
    "u = lambda *v: (lambda a: (a / np.linalg.norm(a)).astype(np.float32))"
    "(np.array(v, dtype=np.float32))",
    "print('WINDOW', FEATURE_WINDOW_SIZE)",
])

#: 一次子进程里跑完窗口策略的全部断言（每启动一个子进程要重新 import torch，约 8 秒）
WINDOW_PROBE = "\n".join([
    "s = IdentityStore(feature_space='t:4')",
    # 身份 1：注册后连续 40 次低质量(q=0.2)观测"另一个方向" → 主特征必须跟过去，且窗口有界
    "g1 = s.register(u(1, 0, 0, 0))",
    "print('SEED_LEN', len(s.get(g1).feature_window))",
    "for _ in range(40):",
    "    s.update_appearance(g1, 'c', u(0, 1, 0, 0), [0, 0, 10, 20], quality_score=0.2)",
    "r1 = s.get(g1)",
    "print('LEN', len(r1.feature_window))",
    "print('MAXLEN', r1.feature_window.maxlen)",
    "print('RECENT', round(float(r1.feature @ u(0, 1, 0, 0)), 4))",
    "print('INITIAL', round(float(r1.feature @ u(1, 0, 0, 0)), 4))",
    # 身份 2/3：一次高质量 vs 一次极低质量观测，比较对均值的推动
    "g2 = s.register(u(1, 0, 0, 0))",
    "s.update_appearance(g2, 'c', u(0, 1, 0, 0), [0, 0, 8, 16], quality_score=1.0)",
    "print('SIM_HIGH', round(float(s.get(g2).feature @ u(0, 1, 0, 0)), 4))",
    "g3 = s.register(u(1, 0, 0, 0))",
    "s.update_appearance(g3, 'c', u(0, 1, 0, 0), [0, 0, 8, 16], quality_score=0.01)",
    "print('SIM_LOW', round(float(s.get(g3).feature @ u(0, 1, 0, 0)), 4))",
])

#: 对照用：EMA 策略（默认）在同样输入下的行为
EMA_PROBE = "\n".join([
    "s = IdentityStore(feature_space='t:4')",
    "g = s.register(u(1, 0, 0, 0))",
    "for _ in range(30):",
    "    s.update_appearance(g, 'c', u(0.2, 1, 0, 0), [0, 0, 10, 20], quality_score=0.2)",
    "f = s.get(g).feature",
    "print('INITIAL', round(float(f @ u(1, 0, 0, 0)), 4))",
    "print('RECENT', round(float(f @ u(0.2, 1, 0, 0)), 4))",
])


def run_probe(window: str, script: str) -> dict:
    """在独立进程里跑一段最小脚本（`FEATURE_WINDOW_SIZE` 是 import 期常量）。"""
    env = {**os.environ, "LAB_MONITOR_FEATURE_WINDOW": window}
    result = subprocess.run(
        [sys.executable, "-c", PRELUDE + "\n" + script],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        raise AssertionError(f"子进程失败：{result.stderr[-600:]}")
    pairs = (line.split() for line in result.stdout.strip().splitlines() if " " in line)
    return {key: value for key, value in pairs}


class WindowPolicyTests(unittest.TestCase):
    """`LAB_MONITOR_FEATURE_WINDOW=20` 时必须表现出"跟随最近观测"的行为。"""

    @classmethod
    def setUpClass(cls):
        cls.values = run_probe("20", WINDOW_PROBE)

    def test_tracks_recent_observations_at_low_quality(self):
        """核心行为：低质量（q=0.2）下也必须跟随最近观测 —— 旧 EMA 正是在这里失效。"""
        self.assertGreater(float(self.values["RECENT"]), 0.95,
                           "窗口策略下主特征必须跟到最近观测上")
        self.assertLess(float(self.values["INITIAL"]), 0.3,
                        "40 次观测后最初那次不应再主导主特征")

    def test_window_length_is_bounded(self):
        """窗口必须有界，否则退化成"平均全部历史"，等于把陈旧问题换个名字。"""
        self.assertEqual(20, int(self.values["LEN"]), "40 次观测后窗口长度应被钳到 20")
        self.assertEqual(20, int(self.values["MAXLEN"]))

    def test_quality_still_downweights_within_window(self):
        """窗口内仍保留"少信模糊帧"的原意：低质量观测对均值的推动必须更小。"""
        self.assertLess(float(self.values["SIM_LOW"]), float(self.values["SIM_HIGH"]),
                        "低质量观测对主特征的推动必须小于高质量观测")

    def test_register_seeds_window_with_first_observation(self):
        self.assertEqual(1, int(self.values["SEED_LEN"]), "注册时应以初始特征作种子观测")


class WindowRecordShapeTests(unittest.TestCase):
    """不依赖策略开关的形状约定（同进程可测）。"""

    def test_record_window_maxlen_follows_constant(self):
        store = IdentityStore(feature_space="t:4")
        gid = store.register(np.array([1, 0, 0, 0], dtype=np.float32))
        self.assertEqual(max(1, FEATURE_WINDOW_SIZE),
                         store.get(gid).feature_window.maxlen)

    def test_first_identity_created_similarity_is_zero(self):
        """空库第一次注册没有近邻 —— created 的相似度必须是 0 而不是 None/异常。"""
        store = IdentityStore(feature_space="t:4")
        resolution = store.register_if_new(
            np.array([1, 0, 0, 0], dtype=np.float32))
        self.assertEqual("created", resolution.status)
        self.assertEqual(0.0, resolution.best_similarity)
        self.assertEqual(0, store.get_metrics()["created_near_miss"])


class ProductionDefaultTests(unittest.TestCase):
    """聚合策略的默认值锁定（2026-10-07 翻转为有界窗口）。"""

    def test_production_default_is_windowed(self):
        values = run_probe("", "print('DEFAULT', FEATURE_WINDOW_SIZE)")
        self.assertEqual(
            20, int(values["DEFAULT"]),
            "2026-10-07 取证后默认必须是有界窗口（20）：EMA 吸引子在生产库被"
            "实测证明与自身观测负相关（-0.53 centered），是重复注册与 26% "
            "match_rate 的根源。回退对照请显式设 LAB_MONITOR_FEATURE_WINDOW=0",
        )

    def test_ema_keeps_initial_observation_dominant(self):
        """
        反向锁定 EMA 的行为特征：它**确实存在**被最初观测主导的问题
        （30 次观测后与最初观测仍有 0.6+ 相似度）—— 这是窗口方案当初的动机。
        将来若要重新评估聚合策略，这条用例会立刻给出对照数字，不必靠记忆。
        """
        values = run_probe("0", EMA_PROBE)
        self.assertGreater(float(values["INITIAL"]), 0.6,
                           "EMA 在低质量下会被最初观测主导（这正是它的问题）")
        self.assertLess(float(values["RECENT"]), 0.95,
                        "EMA 的新观测权重远小于窗口策略（对照用）")


class CreatedObservabilityTests(unittest.TestCase):
    """新建身份的可观测性（2026-09-13 新增）：不能只看到身份变多，看不到为什么。"""

    def test_created_resolution_reports_nearest_similarity(self):
        def unit(*values):
            vector = np.array(values, dtype=np.float32)
            return vector / np.linalg.norm(vector)

        store = IdentityStore(feature_space="t:4")
        store.register_if_new(unit(1, 0, 0, 0))
        resolution = store.register_if_new(unit(0.60, 0.80, 0, 0))
        self.assertEqual("created", resolution.status)
        self.assertAlmostEqual(0.60, resolution.best_similarity, places=3,
                               msg="新建时必须回传『最近邻有多像』，否则不可观测")
        self.assertEqual(1, store.get_metrics()["created_near_miss"],
                         "相似度 ≥0.5 却仍新建 → 计入 created_near_miss")
        self.assertGreater(store.get_metrics()["avg_created_best_similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
