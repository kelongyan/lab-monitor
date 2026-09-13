"""
test_label_render.py — 画面中文标签渲染的回归测试（批次五遗留断点修复）。

锁三件事：
  1. 中文实名必须真的画上帧（Hershey 时代是 "?????"）；
  2. 相同标签重复绘制必须命中缓存（每帧一次区块拷贝，不是每帧 PIL 重绘）；
  3. 边界输入（越界坐标 / 超长文本 / 缓存上限）绝不抛异常 —— 这个函数跑在
     22 个 pipeline 线程的每帧热路径上，抛一次异常就打死一路摄像头。
"""

import unittest

import numpy as np

from src.label_render import LabelRenderer, has_non_ascii


def _frame(w=480, h=270):
    return np.zeros((h, w, 3), dtype=np.uint8)


class AsciiPathTest(unittest.TestCase):
    """ASCII 快路径：几何与旧 cv2 实现同款，且不依赖 PIL。"""

    def test_has_non_ascii(self):
        self.assertFalse(has_non_ascii("ID: #a1b2 [3/8]"))
        self.assertTrue(has_non_ascii("张伟 | #a1b2"))
        self.assertTrue(has_non_ascii("全角！"))

    def test_ascii_draw_changes_frame(self):
        frame = _frame()
        renderer = LabelRenderer()
        renderer.draw(frame, "ID: #a1b2c3d4", 40, 60, (248, 189, 56))
        self.assertTrue(frame.any(), "ASCII 标签应画上像素")

    def test_ascii_draw_does_not_touch_cache(self):
        renderer = LabelRenderer()
        frame = _frame()
        renderer.draw(frame, "Trk: #12 [3/8]", 10, 40, (129, 185, 16))
        self.assertEqual(len(renderer._cache), 0, "ASCII 快路径不该进 PIL 缓存")


@unittest.skipUnless(LabelRenderer().cjk_capable,
                     "PIL 或中文字体不可用（CI/无字体环境），跳过中文渲染用例")
class ChinesePathTest(unittest.TestCase):
    """中文路径：画上帧 + 命中缓存 + 边界安全。"""

    def setUp(self):
        self.renderer = LabelRenderer()

    def test_chinese_draw_changes_frame(self):
        frame = _frame()
        self.renderer.draw(frame, "张伟 | #a1b2c3d4", 40, 60, (248, 189, 56))
        self.assertTrue(frame.any(), "中文标签应画上像素（此前是 ?????）")

    def test_same_text_hits_cache(self):
        frame = _frame()
        self.renderer.draw(frame, "张伟 | #a1b2c3d4", 40, 60, (248, 189, 56))
        self.renderer.draw(frame, "张伟 | #a1b2c3d4", 40, 60, (248, 189, 56))
        self.assertEqual(len(self.renderer._cache), 1,
                         "相同 (文本, 颜色) 必须命中同一张缓存位图")
        # 颜色状态不同（绿→金）是不同的 key
        self.renderer.draw(frame, "张伟 | #a1b2c3d4", 40, 60, (129, 185, 16))
        self.assertEqual(len(self.renderer._cache), 2)

    def test_cache_eviction_bounded(self):
        # cache_limit 下限钳到 16（避免病态小缓存），用 16 验证上限语义
        renderer = LabelRenderer(cache_limit=16)
        frame = _frame()
        for i in range(60):
            renderer.draw(frame, f"人员{i}号 | #gid{i:04d}", 20, 50,
                          (248, 189, 56))
        self.assertLessEqual(len(renderer._cache), 16,
                             "缓存必须受限，长时间运行不得无界增长")

    def test_hostile_inputs_never_raise(self):
        """跑在 22 个 pipeline 线程的每帧热路径上，任何输入都不许抛异常。"""
        renderer = self.renderer
        cases = [
            ("", 0, 0),                                   # 空文本
            ("超" * 200, 10, 10),                          # 超长文本
            ("张三", -50, -20),                            # 越界坐标
            ("李四", 470, 265),                            # 右下角贴边
            ("王五", 10000, 10000),                        # 完全出界
        ]
        for text, x, y in cases:
            frame = _frame()
            renderer.draw(frame, text, x, y, (0, 0, 239))   # 不应抛异常

    def test_banner_has_ink(self):
        """渲染出的位图不能是纯黑块（黑底上要有彩色文字像素）。"""
        banner = self.renderer._banner_cached("测试标签", (248, 189, 56),
                                              (255, 255, 255))
        self.assertIsNotNone(banner)
        self.assertTrue((banner > 40).any(), "位图全黑说明文字没画上")


if __name__ == "__main__":
    unittest.main()
