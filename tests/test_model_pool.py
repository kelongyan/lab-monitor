"""测试 src/model_pool.py：不加载任何真实 YOLO / OSNet 模型，全部使用假工厂"""

import logging
import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

from src.model_pool import (
    MAX_AUTO_POOL_SIZE,
    ModelPool,
    PooledDetector,
    PooledReIDExtractor,
    resolve_pool_size,
)


def setUpModule():
    # 池每建一个实例都会打一条 INFO 进度日志；测试里只保留 WARNING（超时用例需要它）
    logging.getLogger("model_pool").setLevel(logging.WARNING)


class FakeDetector:
    """假检测器：公开属性与 PersonDetector 对齐"""

    def __init__(self, index: int = 0):
        self.index = index
        self.device = "cuda"
        self.conf_thresh = 0.4
        self.use_fp16 = True
        self.result = [[1.0, 2.0, 3.0, 4.0, 0.9]]

    def detect(self, frame):
        return self.result


class FakeExtractor:
    """假 ReID 提取器：公开属性与 ReIDExtractorOSNet 对齐"""

    def __init__(self, index: int = 0):
        self.index = index
        self.feature_space = "fake-space:8"
        self.device = "cuda:0"
        self.result = np.ones(8, dtype=np.float32)

    def extract(self, frame, bbox):
        return self.result


def _counting_factory(cls):
    """返回依次编号的假实例工厂，便于确认池内是多个不同实例"""
    counter = iter(range(1000))
    return lambda: cls(next(counter))


class ModelPoolTests(unittest.TestCase):
    def test_at_most_size_instances_borrowed_concurrently(self):
        pool = ModelPool(_counting_factory(FakeDetector), size=2, name="t")
        holding = threading.Semaphore(0)
        release = threading.Event()

        def worker():
            with pool.borrow():
                holding.release()
                release.wait(timeout=5)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
        for thread in threads:
            thread.start()
        try:
            self.assertTrue(holding.acquire(timeout=2))      # 第 1 个借到
            self.assertTrue(holding.acquire(timeout=2))      # 第 2 个借到
            self.assertFalse(holding.acquire(timeout=0.3))   # 第 3 个必须阻塞
            self.assertEqual(0, pool.available)
        finally:
            release.set()
        self.assertTrue(holding.acquire(timeout=2))          # 归还后第 3 个才拿到
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(2, pool.available)
        self.assertEqual(2, pool.size)

    def test_pool_holds_distinct_instances(self):
        pool = ModelPool(_counting_factory(FakeDetector), size=3, name="t")
        seen = set()
        for _ in range(3):
            with pool.borrow() as instance:
                seen.add(id(instance))
                # 借出期间不归还，强制拿到不同实例
                with pool.borrow() as second:
                    seen.add(id(second))
        self.assertGreaterEqual(len(seen), 2)
        self.assertEqual(3, pool.available)

    def test_instance_is_returned_when_body_raises(self):
        pool = ModelPool(_counting_factory(FakeDetector), size=1, name="t")
        with self.assertRaises(ValueError):
            with pool.borrow():
                raise ValueError("业务异常")
        self.assertEqual(1, pool.available)

    def test_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            ModelPool(lambda: FakeDetector(), size=0, name="t")

    def test_close_releases_instances(self):
        pool = ModelPool(_counting_factory(FakeDetector), size=2, name="t")
        pool.close()
        self.assertEqual(0, pool.available)
        self.assertEqual(0, pool.size)


class BorrowTimeoutTests(unittest.TestCase):
    def test_timeout_only_warns_and_keeps_waiting(self):
        pool = ModelPool(
            _counting_factory(FakeDetector), size=1, name="t", borrow_timeout=0.05
        )
        borrowed = threading.Event()
        release = threading.Event()
        warned = threading.Event()
        got = []

        def holder():
            with pool.borrow():
                borrowed.set()
                release.wait(timeout=5)

        def waiter():
            with pool.borrow() as instance:
                got.append(instance)

        class _WarnProbe(logging.Handler):
            def emit(self, record):
                warned.set()

        probe = _WarnProbe(level=logging.WARNING)
        pool_logger = logging.getLogger("model_pool")
        pool_logger.addHandler(probe)
        holder_thread = threading.Thread(target=holder, daemon=True)
        waiter_thread = threading.Thread(target=waiter, daemon=True)
        try:
            holder_thread.start()
            self.assertTrue(borrowed.wait(timeout=2))
            waiter_thread.start()
            self.assertTrue(warned.wait(timeout=3))   # 超时只告警
            self.assertEqual([], got)                 # 且没有抛异常打死线程
        finally:
            release.set()
            pool_logger.removeHandler(probe)
        waiter_thread.join(timeout=5)
        holder_thread.join(timeout=5)
        self.assertEqual(1, len(got))                 # 归还后照常拿到实例
        self.assertEqual(1, pool.available)


class PooledWrapperTests(unittest.TestCase):
    def test_instances_returned_after_many_sequential_calls(self):
        detector = PooledDetector(_counting_factory(FakeDetector), size=3)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        for _ in range(60):
            detector.detect(frame)
        self.assertEqual(3, detector.pool_size)
        self.assertEqual(3, detector.pool_available)

    def test_instance_returned_when_detect_raises(self):
        class BoomDetector:
            device = "cpu"

            def detect(self, frame):
                raise RuntimeError("推理炸了")

        detector = PooledDetector(lambda: BoomDetector(), size=1)
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                detector.detect(None)
            # 归还失败会在这里立即暴露（否则下一次调用会永久阻塞）
            self.assertEqual(1, detector.pool_available)

    def test_instance_returned_when_extract_raises(self):
        class BoomExtractor:
            feature_space = "fake-space:8"

            def extract(self, frame, bbox):
                raise RuntimeError("OSNet 炸了")

        extractor = PooledReIDExtractor(lambda: BoomExtractor(), size=1)
        with self.assertRaises(RuntimeError):
            extractor.extract(None, [0, 0, 1, 1])
        self.assertEqual(1, extractor.pool_available)

    def test_detector_attributes_are_proxied(self):
        detector = PooledDetector(_counting_factory(FakeDetector), size=2)
        self.assertEqual("cuda", detector.device)
        self.assertEqual(0.4, detector.conf_thresh)
        self.assertTrue(detector.use_fp16)
        self.assertEqual(0, detector.index)   # 兜底代理读第一个实例
        with self.assertRaises(AttributeError):
            detector.no_such_attribute

    def test_extractor_feature_space_is_proxied_exactly(self):
        extractor = PooledReIDExtractor(_counting_factory(FakeExtractor), size=3)
        self.assertEqual("fake-space:8", extractor.feature_space)
        self.assertEqual("cuda:0", extractor.device)
        self.assertEqual(3, extractor.pool_size)

    def test_return_values_pass_through_unchanged(self):
        fake_detector = FakeDetector()
        fake_extractor = FakeExtractor()
        detector = PooledDetector(lambda: fake_detector, size=1)
        extractor = PooledReIDExtractor(lambda: fake_extractor, size=1)
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertIs(fake_detector.result, detector.detect(frame))
        self.assertIs(fake_extractor.result, extractor.extract(frame, [0, 0, 4, 4]))


class PoolSizeTests(unittest.TestCase):
    def test_env_override_wins(self):
        with patch.dict(os.environ, {"LAB_MONITOR_MODEL_POOL": "3"}):
            self.assertEqual(3, resolve_pool_size("cuda", 22))
            self.assertEqual(3, resolve_pool_size("cpu", 22))

    def test_auto_size_by_device_and_source_count(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("LAB_MONITOR_MODEL_POOL", None)
            self.assertEqual(MAX_AUTO_POOL_SIZE, resolve_pool_size("cuda", 22))
            self.assertEqual(2, resolve_pool_size("cuda:0", 2))   # 源少于上限时不浪费
            self.assertEqual(1, resolve_pool_size("cuda", 0))
            self.assertEqual(1, resolve_pool_size("cpu", 22))     # CPU 保持串行

    def test_invalid_env_falls_back_to_auto(self):
        with patch.dict(os.environ, {"LAB_MONITOR_MODEL_POOL": "abc"}):
            logging.getLogger("model_pool").setLevel(logging.CRITICAL)
            try:
                self.assertEqual(MAX_AUTO_POOL_SIZE, resolve_pool_size("cuda", 22))
            finally:
                logging.getLogger("model_pool").setLevel(logging.WARNING)


if __name__ == "__main__":
    unittest.main()
