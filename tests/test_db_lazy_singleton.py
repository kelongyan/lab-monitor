"""
test_db_lazy_singleton.py — src.db 模块级单例惰性化的回归测试。

此前 `db = Database()` 写在模块尾部，任何 `import src.db`（哪怕只想要
Database 这个类去建临时库）都会先打开并迁移生产库 outputs/lab_monitor.db
—— seed 脚本 `--db outputs/mock_demo.db` 就是这样"顺手"碰了生产库的。

惰性化（PEP 562 __getattr__）后必须守住两条：
  1. `from src.db import Database` 只取类：不创建单例、不连库；
  2. `from src.db import db` / 属性访问仍能拿到单例（main.py 等的原语义不变）。

用子进程跑：套件里其他测试可能早已触发过单例创建，主进程内断言
"db 不在模块命名空间"会受用例执行顺序影响；子进程才是干净的可复现环境。
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class LazySingletonTest(unittest.TestCase):

    def _run(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(ROOT), timeout=120,
        )

    def test_importing_class_does_not_create_singleton(self):
        """`from src.db import Database` 之后，模块命名空间里不得出现 db 实例。"""
        code = (
            "import sys; sys.path.insert(0, r'{root}'); "
            "from src.db import Database; "
            "import src.db as m; "
            "assert 'db' not in vars(m), 'import Database 类时单例已被创建（惰性化失效）'; "
            "assert callable(getattr(m, '__getattr__', None)), '缺少 PEP 562 惰性钩子'"
        ).format(root=ROOT.as_posix())
        proc = self._run(code)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)

    def test_attribute_access_still_yields_singleton(self):
        """正向语义：访问 src.db.db 仍要拿到单例并缓存（把 __init__ 打桩以免碰真实库）。"""
        code = (
            "import sys; sys.path.insert(0, r'{root}'); "
            "from unittest.mock import patch; "
            "import src.db as m; "
            "p = patch.object(m.Database, '__init__', lambda self, *a, **k: None); "
            "p.start(); "
            "inst = m.db; p.stop(); "
            "assert isinstance(inst, m.Database), '属性访问应创建单例'; "
            "assert m.db is inst, '单例应缓存为模块属性，二次访问不得重建'"
        ).format(root=ROOT.as_posix())
        proc = self._run(code)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
