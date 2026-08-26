"""
model_pool.py — 有界模型实例池：打破全局推理锁，让多路流水线真正并发推理

改动前 main.py 只创建 1 个 PersonDetector 与 1 个 ReID 提取器供全部 pipeline 共用，
而 src/detector.py 与 src/reid.py 内部各用一把实例锁把整个 predict / extract 包住，
22 个 pipeline 线程因此完全串行（实测每路约 0.45 fps，batch 恒为 1）。

本模块用「有界实例池」替代单实例：预建 size 个彼此独立的模型实例，并发度 = size；
池空时借出会阻塞，形成天然背压（等价于原来的实例锁，只是并发度不再恒为 1）。
不做「每路一个实例」：22 个 YOLO + 22 个 OSNet 常驻内存不可控，且 Python 侧预处理受
GIL 限制，实例数超过 2 后收益就没了（实测数据见 MAX_AUTO_POOL_SIZE 注释）。池大小 = 1 时行为与改动前完全一致
（LAB_MONITOR_MODEL_POOL=1 可一键回退）。
"""

import logging
import os
import queue
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger("model_pool")

# 借出等待多久算异常：只记 warning 提示背压，不抛异常
DEFAULT_BORROW_TIMEOUT = 30.0
# 自动推导池大小的上限。原设 6；2026-08-25 在 22 路真实素材上实测（解码 + detect）：
#     池=1 → 22.9 帧/s    池=2 → 38.0    池=4 → 38.4    池=6 → 38.9
# 池 ≥2 之后曲线就平了 —— 瓶颈已从「实例锁」转移到 GIL，多出来的实例只增显存占用
# 与启动耗时（每个实例都要单独加载一份权重）。要更大就显式设 LAB_MONITOR_MODEL_POOL。
MAX_AUTO_POOL_SIZE = 2


def resolve_pool_size(
    device: str,
    source_count: int,
    env_var: str = "LAB_MONITOR_MODEL_POOL",
) -> int:
    """
    解析模型池大小：环境变量优先（取值 > 0 时直接采用），为 0（默认）时按设备自动推导。
    GPU：min(6, 视频源数)；CPU：1（只有一份算力，池化只增内存不增吞吐）。
    """
    raw = os.getenv(env_var, "0")
    try:
        configured = int(raw)
    except ValueError:
        logger.warning("%s=%r 不是合法整数，改为自动推导池大小", env_var, raw)
        configured = 0
    if configured > 0:
        return configured
    if str(device).startswith("cuda"):
        return max(1, min(MAX_AUTO_POOL_SIZE, source_count))
    return 1


class ModelPool:
    """有界模型实例池：构造时用 factory 预建 size 个实例，borrow() 借出并保证归还"""

    def __init__(
        self,
        factory: Callable[[], object],
        size: int,
        name: str = "model",
        borrow_timeout: float = DEFAULT_BORROW_TIMEOUT,
    ):
        if size < 1:
            raise ValueError(f"模型池大小必须 >= 1，收到 {size}")
        self.name = name
        self._borrow_timeout = borrow_timeout
        self._queue: queue.Queue = queue.Queue(maxsize=size)
        self._instances: list = []
        started = time.perf_counter()
        for idx in range(size):
            one_started = time.perf_counter()
            instance = factory()   # 加载失败直接上抛：启动阶段暴露比运行时暴露好
            self._instances.append(instance)
            self._queue.put(instance)
            # 逐个打印进度：池大小 × 单实例加载耗时可达数十秒，否则运维会以为服务卡死
            logger.info(
                "%s 池实例 %d/%d 就绪（%.1fs）",
                name, idx + 1, size, time.perf_counter() - one_started,
            )
        logger.info(
            "%s 池创建完成：%d 个实例，总耗时 %.1fs",
            name, size, time.perf_counter() - started,
        )

    @property
    def size(self) -> int:
        """池容量（实例总数）"""
        return len(self._instances)

    @property
    def available(self) -> int:
        """当前空闲实例数（近似值，仅用于监控与测试）"""
        return self._queue.qsize()

    @property
    def first(self) -> object:
        """池内第一个实例：仅供代理只读公开属性（device / feature_space 等）使用"""
        return self._instances[0]

    @contextmanager
    def borrow(self, timeout: float | None = None) -> Iterator[object]:
        """
        借出一个实例，退出上下文时用 try/finally 保证归还
        （少了 finally，一次推理异常就永久漏掉一个实例，池最终会被抽干）。
        池空时阻塞 —— 这正是我们要的背压。等待超过 timeout 秒只记 warning 后继续等，
        绝不抛异常，避免把 pipeline 线程打死。
        """
        wait = self._borrow_timeout if timeout is None else timeout
        try:
            instance = self._queue.get(timeout=wait)
        except queue.Empty:
            logger.warning(
                "%s 池满载：等待空闲实例超过 %.1fs（池大小=%d），继续等待中",
                self.name, wait, self.size,
            )
            instance = self._queue.get()   # 兜底无限等待：宁可慢，不可崩
        try:
            yield instance
        finally:
            self._queue.put(instance)

    def close(self) -> None:
        """
        释放池内实例（仅在停止服务后调用）。ultralytics / torchreid 模型没有显式释放接口，
        内存与显存随对象被 GC 回收即可，因此这里只清空引用，留作将来扩展。
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._instances.clear()


class _PooledModel:
    """
    池化代理基类：方法调用转给借出的实例，公开只读属性代理到池内第一个实例
    （池内实例配置完全相同，只读属性取任意一个都等价）。
    """

    def __init__(self, pool: ModelPool):
        self._pool = pool

    @property
    def pool_size(self) -> int:
        return self._pool.size

    @property
    def pool_available(self) -> int:
        return self._pool.available

    def close(self) -> None:
        self._pool.close()

    def __getattr__(self, name: str):
        """兜底属性代理：实例字典里找不到的公开属性一律读第一个池内实例"""
        if name.startswith("_"):
            raise AttributeError(name)
        pool = self.__dict__.get("_pool")
        if pool is None:
            raise AttributeError(name)
        return getattr(pool.first, name)


class PooledDetector(_PooledModel):
    """
    与 PersonDetector 接口等价的池化检测器。
    调用方 src/pipeline.py:423 只用 detect()；device / conf_thresh / use_fp16 是
    PersonDetector 的全部公开属性，此处显式代理，避免有人读到 AttributeError。
    """

    def __init__(
        self,
        factory: Callable[[], object],
        size: int,
        borrow_timeout: float = DEFAULT_BORROW_TIMEOUT,
    ):
        super().__init__(
            ModelPool(factory, size, name="detector", borrow_timeout=borrow_timeout)
        )

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """与 PersonDetector.detect 完全一致：[[x1, y1, x2, y2, conf], ...]"""
        with self._pool.borrow() as detector:
            return detector.detect(frame)

    @property
    def device(self):
        return self._pool.first.device

    @property
    def conf_thresh(self) -> float:
        return self._pool.first.conf_thresh

    @property
    def use_fp16(self) -> bool:
        return self._pool.first.use_fp16


class PooledReIDExtractor(_PooledModel):
    """
    与 ReIDExtractorOSNet / ReIDExtractor 接口等价的池化提取器。
    feature_space 是必须代理的关键属性：main.py 用它构造 IdentityStore，
    src/identity_store.py:140 对每条历史身份做特征空间全等比较，
    代理错会导致全部历史身份被静默跳过。
    """

    def __init__(
        self,
        factory: Callable[[], object],
        size: int,
        borrow_timeout: float = DEFAULT_BORROW_TIMEOUT,
    ):
        super().__init__(
            ModelPool(factory, size, name="reid", borrow_timeout=borrow_timeout)
        )
        # build_reid_extractor() 在 OSNet 初始化失败时会静默回退 ResNet50（2048 维），
        # 池内因此可能混装 512/2048 两种骨干，而 feature_space 只报第一个实例的值 →
        # 那些维度不符的实例提取出的特征会被 IdentityStore 判为 invalid 而静默失效。
        # 宁可启动即失败，也不要带着一半瘫掉的 ReID 上线。
        spaces = {
            getattr(instance, "feature_space", None)
            for instance in self._pool._instances
        }
        if len(spaces) > 1:
            raise RuntimeError(f"ReID 池实例特征空间不一致: {spaces}")

    def extract(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        """与 ReIDExtractor*.extract 完全一致：L2 归一化特征向量或 None"""
        with self._pool.borrow() as extractor:
            return extractor.extract(frame, bbox)

    @property
    def feature_space(self) -> str:
        return self._pool.first.feature_space

    @property
    def device(self):
        return self._pool.first.device


