"""
label_render.py — 画面人员标签渲染（支持中文），带按文本缓存。

为什么需要这个模块
------------------
pipeline.py 此前用 cv2.putText（Hershey 字体）画人员标签，而 Hershey **不含
任何非 ASCII 字形**：实名"张三"画出来是 "?? | #a1b2c3d4"。批次五打通了
自动命名链路之后，实时画面上最重要的信息就是中文姓名 —— 显示成问号等于
白认了。这是 docs/PLAN_2026-09-12_personnel_mock_data.md 明确遗留的断点。

方案与代价
----------
- 标签含非 ASCII 字符时改用 PIL 渲染（Windows 自带 msyh/simhei/simsun），
  把「整条标签 + 黑底 + 彩色边框」渲染成一张 BGR 位图，按 (文本, 颜色) 缓存。
  标签文本在一条轨迹生命周期内几乎不变（姓名/gid 稳定），命中缓存时每帧
  只剩一次 numpy 区块拷贝（微秒级），未命中才走 PIL 绘制（毫秒级）。
- 纯 ASCII 标签走 cv2 快路径，几何与旧实现完全一致，零额外开销。
- PIL 缺失 / 字体不可用时优雅回退 cv2.putText：中文退化成 "?"，但绝不
  打崩 pipeline 线程（与 seed 脚本头像的降级策略同思路）。

缓存上限：标签文本里有 track_id 与缓冲计数（"Trk: #12 [3/8]"），长时间
运行会积累大量不同字符串，必须设上限防止内存无界增长。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("label_render")

#: 中文字体候选（顺序即优先级）。与 scripts/seed_personnel_mock.py 的头像
#: 字体保持同一组候选，任一可用即可。
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/simsun.ttc",   # 宋体
)


def _load_font(size: int):
    """加载任一可用中文字体；全失败返回 None（调用方降级 cv2）。"""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return None


def has_non_ascii(text: str) -> bool:
    """标签里是否有 Hershey 画不出的字符（中文、全角符号等）。"""
    return any(ord(ch) > 127 for ch in text)


class LabelRenderer:
    """黑底描边标签条渲染器。

    用法（pipeline 每路一个实例，避免跨线程共享 PIL 对象）::

        renderer = LabelRenderer()
        renderer.draw(frame, label_text, x1, y1, color, text_color)
    """

    def __init__(self, font_size: int = 18, cache_limit: int = 256):
        self._font_size = font_size
        self._font = _load_font(font_size)
        self._cache: dict[tuple, np.ndarray] = {}   # (text, border, fill) → BGR 位图
        self._cache_limit = max(16, int(cache_limit))
        if self._font is None:
            logger.info("PIL 或中文字体不可用，画面标签回退 cv2（中文将显示为 ?）")

    @property
    def cjk_capable(self) -> bool:
        """是否具备中文渲染能力（PIL + 至少一个候选字体都可用）。"""
        return self._font is not None

    # ------------------------------------------------------------------ #
    # PIL 路径                                                              #
    # ------------------------------------------------------------------ #

    def _render_banner(self, text: str, border_bgr, fill_bgr) -> np.ndarray | None:
        """PIL 渲染整条标签：黑底 + 1px 彩色边框 + 彩色文字。失败返回 None。"""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None
        if self._font is None:
            return None
        try:
            left, top, right, bottom = self._font.getbbox(text)
            tw = right - left
            th = bottom - top
            pad_x, pad_top, pad_bottom = 6, 4, 6
            width = tw + pad_x * 2
            height = th + pad_top + pad_bottom
            img = Image.new("RGB", (max(2, width), max(2, height)), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, width - 1, height - 1],
                           outline=self._to_rgb(border_bgr), width=1)
            draw.text((pad_x - left, pad_top - top), text,
                      font=self._font, fill=self._to_rgb(fill_bgr))
            return cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001 - 渲染失败必须降级而不是打崩 pipeline
            logger.debug("PIL 标签渲染失败，回退 cv2", exc_info=True)
            return None

    @staticmethod
    def _to_rgb(bgr) -> tuple:
        b, g, r = (int(c) for c in bgr[:3])
        return r, g, b

    def _banner_cached(self, text: str, border_bgr, fill_bgr) -> np.ndarray | None:
        key = (text, tuple(border_bgr[:3]), tuple(fill_bgr[:3]))
        banner = self._cache.get(key)
        if banner is not None:
            return banner
        banner = self._render_banner(text, border_bgr, fill_bgr)
        if banner is None:
            return None
        # 简单 FIFO：过限时丢掉最老的一半，避免长时间运行内存无界
        if len(self._cache) >= self._cache_limit:
            for old in list(self._cache)[: self._cache_limit // 2]:
                self._cache.pop(old, None)
        self._cache[key] = banner
        return banner

    # ------------------------------------------------------------------ #
    # 对外接口                                                              #
    # ------------------------------------------------------------------ #

    def draw(self, frame: np.ndarray, text: str, x1: int, y1: int,
             color, text_color=(255, 255, 255)) -> None:
        """
        在 frame 上画标签条（黑底 + 彩边 + 文字），贴在 bbox 左上角外侧。

        ASCII 文本（含纯数字/英文标签）直接走 cv2，几何与历史实现一致；
        含中文等非 ASCII 字符时走 PIL 位图缓存。PIL 不可用时同样回落 cv2 ——
        代价只是中文显示成问号，不影响任何跟踪/识别逻辑。
        """
        if has_non_ascii(text) and self.cjk_capable:
            banner = self._banner_cached(str(text), color, text_color)
            if banner is not None:
                self._paste(frame, banner, x1, y1)
                return
        self._draw_cv2(frame, str(text), x1, y1, color, text_color)

    @staticmethod
    def _paste(frame: np.ndarray, banner: np.ndarray, x1: int, y1: int) -> None:
        fh, fw = frame.shape[:2]
        bh, bw = banner.shape[:2]
        # 底边贴着 bbox 顶（与 cv2 路径的 text_y=max(th+10, y1) 语义一致）；
        # y1 太靠上时整体下压到画面内，右/下越界直接裁掉。
        top = max(0, min(int(y1), fh) - bh)
        left = max(0, min(int(x1) - 1, fw - 1))
        bottom = min(fh, top + bh)
        right = min(fw, left + bw)
        if bottom > top and right > left:
            frame[top:bottom, left:right] = banner[: bottom - top, : right - left]

    @staticmethod
    def _draw_cv2(frame, text, x1, y1, color, text_color) -> None:
        """cv2 路径：与 pipeline.py 原实现逐像素同款（含双描边）。"""
        font_scale = 0.6
        font_thick = 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, font_thick)
        text_y = max(th + 10, int(y1))
        x1 = int(x1)
        cv2.rectangle(frame, (x1 - 1, text_y - th - 8), (x1 + tw + 12, text_y + 6),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (x1 - 1, text_y - th - 8), (x1 + tw + 12, text_y + 6),
                      color, 1)
        cv2.putText(frame, text, (x1 + 5, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x1 + 5, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, font_thick, cv2.LINE_AA)
