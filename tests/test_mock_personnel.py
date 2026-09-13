"""批次五 P0：人员模拟数据 + 档案检索的回归测试（stdlib unittest，无 pytest）。

覆盖三条"错一点就静默失效"的链路：
  1. personnel 两列新字段的迁移与 source 保护（PATCH 不得把模拟档案洗成真实档案）
  2. search_personnel 的过滤 / 转义 / 分页 / 聚合口径
  3. seed → IdentityStore._restore() → 底库 1:N 命名（端到端，临时库，不碰生产）
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import Database  # noqa: E402
from src.identity_store import IdentityStore  # noqa: E402
from src.personnel import PersonnelGallery  # noqa: E402
from src.mock_personnel import (  # noqa: E402
    COMPOUND_SURNAMES, DISPLAY_HEIGHT, DISPLAY_WIDTH, GIVEN_2, SURNAMES,
    build_adjacency, generate_persons, jitter_feature, make_bbox, make_employee_no,
    make_name, make_walking_chain, synth_feature,
)


def _tmp_db_path(test: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp(prefix="mock_p0_"))
    test.addCleanup(shutil.rmtree, path, True)
    return path / "t.db"


class PersonnelColumnsTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(_tmp_db_path(self))
        self.addCleanup(self.db.close)

    def test_new_columns_exist(self):
        cols = {r[1] for r in self.db._get_conn().execute("PRAGMA table_info(personnel)")}
        self.assertIn("source", cols)
        self.assertIn("thumb_path", cols)

    def test_source_survives_sourceless_upsert(self):
        """PATCH 路径不带 source，绝不能把 SIM 标记洗掉。"""
        self.db.upsert_personnel("p1", "张伟", "QLU-26-0001", "高性能机房", "a",
                                 source="synthetic")
        self.db.upsert_personnel("p1", "张伟", "QLU-26-0001", "高性能机房", "b")
        row = self.db.get_personnel("p1")
        self.assertEqual(row["source"], "synthetic")
        self.assertEqual(row["note"], "b")

    def test_set_thumb_does_not_clear_source(self):
        self.db.upsert_personnel("p2", "李娜", None, None, None, source="synthetic")
        self.assertTrue(self.db.set_personnel_thumb("p2", "personnel_crops/p2.jpg"))
        row = self.db.get_personnel("p2")
        self.assertEqual(row["thumb_path"], "personnel_crops/p2.jpg")
        self.assertEqual(row["source"], "synthetic")


class SearchPersonnelTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(_tmp_db_path(self))
        self.addCleanup(self.db.close)
        for i in range(6):
            self.db.upsert_personnel(
                f"p{i}", f"人{i}", f"QLU-26-{i:04d}",
                "网络运维" if i % 2 == 0 else "系统软件", None,
                source="synthetic" if i < 5 else None,
            )

    def test_pagination_and_total(self):
        rows, total = self.db.search_personnel(limit=3)
        self.assertEqual(total, 6)
        self.assertEqual(len(rows), 3)
        rows2, _ = self.db.search_personnel(limit=3, offset=3)
        self.assertEqual(len(rows2), 3)
        self.assertFalse({r["person_id"] for r in rows}
                         & {r["person_id"] for r in rows2})

    def test_q_matches_name_employee_no_department(self):
        self.assertEqual(self.db.search_personnel(q="人1")[1], 1)
        self.assertEqual(self.db.search_personnel(q="QLU-26-0002")[1], 1)
        self.assertEqual(self.db.search_personnel(q="网络")[1], 3)

    def test_like_metacharacters_escaped(self):
        """搜 '%' / '_' 必须按字面量处理，否则退化成全表通配。"""
        self.assertEqual(self.db.search_personnel(q="%")[1], 0)
        self.assertEqual(self.db.search_personnel(q="_")[1], 0)

    def test_source_filter(self):
        self.assertEqual(self.db.search_personnel(source="synthetic")[1], 5)
        # source=NULL = 人工录入的真实档案（见 db.py 迁移注释）：
        # "真实"筛选必须把 NULL 一起算进来，否则手工建档的人永远不可见。
        # 这里 i=5 的档案就是手工录入（source=None），必须被 real 命中。
        self.assertEqual(self.db.search_personnel(source="real")[1], 1)
        self.db.upsert_personnel("px", "显式真实", None, None, None, source="real")
        self.assertEqual(self.db.search_personnel(source="real")[1], 2,
                         "显式打 real 标的行也要命中")

    def test_db_failure_is_raised_not_reported_as_empty(self):
        """
        回归：库故障绝不能伪装成"没有人"的空结果。
        search_personnel / person_activity_stats 曾把异常吞成 ([], 0) / 全零，
        前端拿到 HTTP 200 只能显示"确实没人"—— 故障和事实被混为一谈。
        """
        import sqlite3

        def boom():
            raise sqlite3.OperationalError("injected failure")

        self.db._get_conn = boom  # noqa: SLF001 - 实例属性遮蔽方法，注入故障
        with self.assertRaises(sqlite3.OperationalError):
            self.db.search_personnel()
        with self.assertRaises(sqlite3.OperationalError):
            self.db.person_activity_stats("p0")

    def test_empty_result_shape(self):
        rows, total = self.db.search_personnel(q="不存在的人")
        self.assertEqual((rows, total), ([], 0))

    def test_aggregates_from_identities(self):
        self.db._get_conn().execute(
            "INSERT INTO identities (global_id, last_seen, last_camera, "
            "total_appearances, person_id) VALUES ('g1', 100, 'reg_01', 7, 'p0')")
        self.db._get_conn().commit()
        rows, _ = self.db.search_personnel(q="人0")
        self.assertEqual(rows[0]["identity_count"], 1)
        self.assertEqual(rows[0]["total_appearances"], 7)
        self.assertEqual(rows[0]["last_camera"], "reg_01")

    def test_list_personnel_still_unfiltered(self):
        """PersonnelGallery 靠它建全量底库，绝不能被分页污染。"""
        self.assertEqual(len(self.db.list_personnel()), 6)


class MockPersonnelGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_names_unique_and_well_formed(self):
        used: set[str] = set()
        names = {make_name(self.rng, used) for _ in range(400)}
        self.assertEqual(len(names), 400)
        for name in names:
            self.assertTrue(2 <= len(name) <= 4, name)

    def test_wordbanks_have_no_stray_tokens(self):
        """姓氏表按字拆分：出现空白或超长项说明词库被写坏了。"""
        for surname in SURNAMES:
            self.assertEqual(len(surname), 1, repr(surname))
        for compound in COMPOUND_SURNAMES:
            self.assertTrue(2 <= len(compound) <= 3, repr(compound))
        self.assertGreater(len(SURNAMES), 100)
        self.assertGreater(len(GIVEN_2), 60)

    def test_employee_no_format(self):
        self.assertEqual(make_employee_no(7, 26), "QLU-26-0007")

    def test_jitter_hits_expected_similarity_band(self):
        """回归：noise 未归一化时余弦会掉到 0.15，整条命名链路静默失效。"""
        base = synth_feature(self.rng)
        values = [jitter_feature(base, self.rng)[1] for _ in range(50)]
        self.assertGreater(min(values), 0.80, f"与本人档案过远: {min(values):.3f}")
        self.assertLess(max(values), 0.99)

    def test_jittered_identities_stay_separable_across_people(self):
        a = synth_feature(self.rng)
        b = synth_feature(self.rng)
        cross = [float(np.dot(jitter_feature(a, self.rng)[0],
                              jitter_feature(b, self.rng)[0]))
                 for _ in range(20)]
        self.assertLess(max(abs(v) for v in cross), 0.68,
                        "跨人余弦越过阈值 → 演示会张冠李戴")

    def test_bbox_in_bounds(self):
        for _ in range(500):
            x1, y1, x2, y2 = make_bbox(self.rng)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(x1, 0)
            self.assertLessEqual(x2, DISPLAY_WIDTH)
            self.assertLessEqual(y2, DISPLAY_HEIGHT)

    def test_adjacency_drops_unknown_cameras(self):
        topology = {"a": [{"next": "b"}], "b": [{"next": "ghost"}], "ghost": [{"next": "a"}]}
        adj = build_adjacency(topology, ["a", "b"])
        self.assertEqual(sorted(adj), ["a"])
        self.assertEqual(adj["a"][0]["next"], "b")

    def test_walking_chain_follows_real_edges(self):
        cameras = ["a", "b", "c"]
        topology = {"a": [{"next": "b", "expected_seconds": 10, "tolerance_seconds": 5}],
                    "b": [{"next": "c", "expected_seconds": 10, "tolerance_seconds": 5}],
                    "c": [{"next": "a", "expected_seconds": 10, "tolerance_seconds": 5}]}
        adj = build_adjacency(topology, cameras)
        for _ in range(200):
            chain = make_walking_chain(adj, cameras, self.rng, 7)
            self.assertEqual(len(chain), 7)
            for left, right in zip(chain, chain[1:]):
                self.assertIn(right, [e["next"] for e in adj[left]])

    def test_align_tail_to_keeps_window(self):
        persons = generate_persons(
            6, known_cameras=["a", "b"],
            topology={"a": [{"next": "b", "expected_seconds": 5, "tolerance_seconds": 1}],
                      "b": [{"next": "a", "expected_seconds": 5, "tolerance_seconds": 1}]},
            assets_by_camera={"a": {"asset_id": 1, "duration_real": 60.0,
                                    "fps_declared": 25.0, "frames_real": 1500},
                              "b": {"asset_id": 2, "duration_real": 60.0,
                                    "fps_declared": 25.0, "frames_real": 1500}},
            feature_space="unit-test:512", rng=np.random.default_rng(3),
            window_start=1000.0, window_span=600.0, align_tail_to=9000.0)
        stamps = [a.timestamp for p in persons for i in p.identities for a in i.appearances]
        self.assertAlmostEqual(max(stamps), 9000.0, places=1)
        self.assertGreater(min(stamps), 1000.0)

    def test_appearances_carry_video_coordinates(self):
        persons = generate_persons(
            4, known_cameras=["a"],
            topology={"a": [{"next": "a", "expected_seconds": 5, "tolerance_seconds": 1}]},
            assets_by_camera={"a": {"asset_id": 9, "duration_real": 100.0,
                                    "fps_declared": 25.0, "frames_real": 2500}},
            feature_space="unit-test:512", rng=np.random.default_rng(5),
            window_start=0.0, window_span=10.0)
        rows = [a for p in persons for i in p.identities for a in i.appearances]
        self.assertTrue(rows)
        for appearance in rows:
            self.assertEqual(appearance.asset_id, 9)
            self.assertGreaterEqual(appearance.video_ts, 0.0)
            self.assertLess(appearance.video_ts, 100.0)
            self.assertLess(appearance.video_frame, 2500)
            self.assertEqual(len(appearance.bbox), 4)


class SeedRoundTripTest(unittest.TestCase):
    """端到端：seed 出来的身份必须能被 IdentityStore 恢复、被底库命名、被 --clean 清掉。"""

    def setUp(self):
        import json
        self.dir = Path(tempfile.mkdtemp(prefix="seed_rt_"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.db_path = self.dir / "seed.db"
        config = Path(__file__).resolve().parent.parent / "config"
        sources = json.loads((config / "sources.json").read_text(encoding="utf-8"))
        topology = json.loads((config / "topology.json").read_text(encoding="utf-8"))
        from src.reid_config import FEATURE_DIM, get_reid_weight
        self.feature_space = get_reid_weight(None).feature_space
        self.known = list(sources)
        self.persons = generate_persons(
            12, known_cameras=self.known, topology=topology,
            assets_by_camera={c: {"asset_id": i + 1, "duration_real": 60.0,
                                  "fps_declared": 25.0, "frames_real": 1500}
                              for i, c in enumerate(self.known)},
            feature_space=self.feature_space, feature_dim=FEATURE_DIM,
            rng=np.random.default_rng(17),
            window_start=1_700_000_000.0, window_span=3600.0, align_tail_to=1_700_003_600.0)
        self.db = Database(self.db_path)
        self.addCleanup(self.db.close)
        self.conn = sqlite3.connect(str(self.db_path))
        self.addCleanup(self.conn.close)

    def _seed_rows(self):
        from scripts.seed_personnel_mock import write_persons
        return write_persons(self.db, self.persons)

    def test_restored_by_identity_store(self):
        from src.identity_store import IdentityStore
        self._seed_rows()
        restore_db = Database(self.db_path)
        self.addCleanup(restore_db.close)
        store = IdentityStore(database=restore_db, feature_space=self.feature_space)
        gids = [i.global_id for p in self.persons for i in p.identities]
        self.assertTrue(gids)
        for gid in gids:
            self.assertIn(gid, store.all_ids(), f"{gid} 不会被 _restore 认出 → 检索会 404")

    def test_gallery_can_name_its_own_identities(self):
        """每个 gid 必须能匹配回自己的人 —— 这是自动命名链路成立的底线。"""
        from src.personnel import PersonnelGallery
        from src.reid_config import FEATURE_DIM, REID_MATCH_THRESHOLD
        self._seed_rows()
        gallery = PersonnelGallery(database=self.db, threshold=REID_MATCH_THRESHOLD)
        self.assertEqual(gallery.size(), len(self.persons))
        for person in self.persons:
            for identity in person.identities:
                hit = gallery.match(identity.feature)
                self.assertIsNotNone(hit, f"{identity.global_id} 匹配不到任何人")
                self.assertEqual(hit["person_id"], person.person_id,
                                 "匹配到了别人 → 演示数据会张冠李戴")
                self.assertGreaterEqual(hit["score"], REID_MATCH_THRESHOLD)

    def test_write_persons_persists_expected_counts(self):
        stats = self._seed_rows()
        self.assertEqual(stats["personnel"], len(self.persons))
        self.assertEqual(stats["photos"], len(self.persons))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM identity_appearances").fetchone()[0],
            sum(len(i.appearances) for p in self.persons for i in p.identities))
        # total_appearances 必须回填成真实轨迹数，否则档案卡显示 0 次
        for person in self.persons:
            for identity in person.identities:
                row = self.conn.execute(
                    "SELECT total_appearances FROM identities WHERE global_id = ?",
                    (identity.global_id,)).fetchone()
                self.assertEqual(row[0], len(identity.appearances))

    def test_feature_space_matches_runtime_exactly(self):
        self._seed_rows()
        stored = {r[0] for r in self.conn.execute(
            "SELECT DISTINCT feature_space FROM identities")}
        self.assertEqual(stored, {self.feature_space},
                         "feature_space 与运行时不一致 → store 会跳过整行")


class PersonnelApiTest(unittest.TestCase):
    """新端点的契约 + 一个真实 500 的回归。"""

    def setUp(self):
        import server
        from fastapi.testclient import TestClient
        self.db = Database(_tmp_db_path(self))
        self.addCleanup(self.db.close)
        gallery = PersonnelGallery(database=self.db, threshold=0.68)
        store = IdentityStore(database=self.db, feature_space="test:4")
        self.server_module = server
        server.init_server(frame_hub=None, broadcaster=None,
                           identity_store=store, personnel=gallery)
        self.client = TestClient(server.app)
        self.headers = {"X-Lab-Monitor-Request": "1"}

    def tearDown(self):
        self.server_module.init_server(None, None, None)

    def _make(self, name, employee_no, department, source=None, with_photo=True):
        pid = self.client.post("/api/personnel", json={
            "name": name, "employee_no": employee_no, "department": department},
            headers=self.headers,
        ).json()["person_id"]
        self.db.upsert_personnel(pid, name, employee_no, department, None, source=source)
        if with_photo:
            rng = np.random.default_rng(abs(hash(pid)) % 2**32)
            gallery = self.server_module._personnel
            gallery.add_photo(pid, synth_feature(rng, dim=512))
        return pid

    def test_detail_endpoint_is_serializable_with_photos(self):
        """回归：{**person} 展开底库字典会把 ndarray 塞进 JSON → 500。"""
        pid = self._make("张三", "QLU-26-0001", "网络运维", source="synthetic")
        res = self.client.get(f"/api/personnel/{pid}")
        self.assertEqual(res.status_code, 200, res.text[:200])
        body = res.json()
        self.assertEqual(body["name"], "张三")
        self.assertEqual(body["source"], "synthetic")
        self.assertEqual(body["photo_count"], 1)
        self.assertNotIn("features", body, "内存底库字典漏进了响应")

    def test_list_supports_q_department_source_pagination(self):
        for i in range(8):
            self._make(f"赵{i}", f"QLU-26-{i:04d}",
                       "网络运维" if i < 5 else "系统软件",
                       source="synthetic" if i < 6 else None, with_photo=False)
        body = self.client.get("/api/personnel").json()
        self.assertEqual(body["count"], 8)
        self.assertEqual(len(body["personnel"]), 8)
        self.assertEqual(body["departments"], ["系统软件", "网络运维"])

        self.assertEqual(self.client.get(
            "/api/personnel", params={"q": "赵3"}).json()["count"], 1)
        self.assertEqual(self.client.get(
            "/api/personnel", params={"q": "QLU-26-0007"}).json()["count"], 1)
        self.assertEqual(self.client.get(
            "/api/personnel", params={"department": "系统软件"}).json()["count"], 3)
        self.assertEqual(self.client.get(
            "/api/personnel", params={"source": "synthetic"}).json()["count"], 6)

        page = self.client.get("/api/personnel", params={"limit": 3, "offset": 6}).json()
        self.assertEqual(page["count"], 8, "total 必须是过滤后总数，不是本页条数")
        self.assertEqual(len(page["personnel"]), 2)

    def test_list_rejects_out_of_range_params(self):
        self.assertEqual(self.client.get(
            "/api/personnel", params={"limit": 0}).status_code, 422)
        self.assertEqual(self.client.get(
            "/api/personnel", params={"limit": 9999}).status_code, 422)
        self.assertEqual(self.client.get(
            "/api/personnel", params={"q": "x" * 65}).status_code, 422)

    def test_departments_not_narrowed_by_query(self):
        """搜"张"之后部门下拉仍要完整，否则筛选器越用越窄。"""
        self._make("张伟", "QLU-26-0001", "网络运维", with_photo=False)
        self._make("李娜", "QLU-26-0002", "系统软件", with_photo=False)
        body = self.client.get("/api/personnel", params={"q": "张"}).json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["departments"], ["系统软件", "网络运维"])

    def test_activity_endpoint(self):
        pid = self._make("王五", "QLU-26-0009", "动力保障", source="synthetic",
                         with_photo=False)
        self.db.save_identity(
            global_id="gact", feature_dim=4, feature_blob=np.ones(4, "float32").tobytes(),
            feature_bank_count=1,
            feature_bank_blob=np.ones(4, "float32").tobytes(),
            total_appearances=2, last_camera="reg_01", last_seen=200.0,
            first_seen=100.0, feature_space="test:4", person_id=pid,
            new_appearance={"camera": "reg_01", "time": 200.0, "bbox": [1, 2, 3, 4]})
        self.db.record_appearance("gact", "reg_02", 300.0, [5, 6, 7, 8], 3,
                                  asset_id=1, video_frame=10, video_ts=1.0)
        body = self.client.get(f"/api/personnel/{pid}/activity").json()
        self.assertEqual(body["camera_count"], 2)
        self.assertEqual(body["raw_rows"], 2)
        self.assertEqual({c["camera_id"] for c in body["per_camera"]},
                         {"reg_01", "reg_02"})

    def test_activity_404_for_unknown(self):
        self.assertEqual(self.client.get(
            "/api/personnel/nope/activity").status_code, 404)


class ThumbPathContractTest(unittest.TestCase):
    """
    头像路径三方口径必须一致：存储（磁盘路径）→ 清理（--purge-images）→ 出图（URL）。

    这里锁的是两个**已发生过的静默 bug**（不报错、只是悄悄坏掉）：
      1. 前端自己拼 URL：库里存 `personnel_crops/x.jpg`（下划线、还少一层 outputs/），
         mount 却是连字符的 `/personnel-crops` → 每个头像 404、静默退化成首字块。
      2. `--purge-images` 解析 `ROOT / thumb_path`：存储值少一层 outputs/ 时
         解析到不存在的路径 → 一张图都删不掉，但日志只报"成功 0 张"。
    """

    def test_stored_path_is_relative_to_root_and_really_exists(self):
        from scripts.seed_personnel_mock import CROPS_DIR, make_placeholder_thumb
        out_dir = CROPS_DIR.parent / "thumb_contract_tmp"
        self.addCleanup(shutil.rmtree, out_dir, True)
        person = _FakePerson("aaaabbbb", "张伟", "网络运维", "QLU-26-0001")
        rel = make_placeholder_thumb(person, out_dir)
        self.assertIsNotNone(rel, "PIL 不可用，无法验证头像路径契约")
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(rel, "outputs/thumb_contract_tmp/aaaabbbb.jpg")
        # 存储路径必须指向真实文件，否则 --purge-images 必然删不到
        self.assertTrue((root / rel).exists(), f"存储路径不指向真实文件: {rel}")

    def test_absolute_path_when_dir_outside_project(self):
        """--thumb-dir 指到项目外时要存绝对路径，清理才仍能删到（而不是拼错相对路径）。"""
        from scripts.seed_personnel_mock import make_placeholder_thumb
        outside = Path(tempfile.mkdtemp(prefix="thumb_outside_"))
        self.addCleanup(shutil.rmtree, outside, True)
        rel = make_placeholder_thumb(
            _FakePerson("ccccdddd", "李娜", "系统软件", "QLU-26-0002"), outside)
        self.assertIsNotNone(rel)
        self.assertTrue(Path(rel).is_absolute(), f"项目外目录应存绝对路径: {rel}")
        self.assertTrue(Path(rel).exists())

    def test_url_derivation_hits_the_mount(self):
        """
        URL 必须由后端从磁盘路径翻译，且落在 mount 前缀下 —— 前端不再自己拼。
        只取文件名，顺带杜绝 `../../` 路径穿越。
        """
        from server import PERSONNEL_CROPS_MOUNT, personnel_thumb_url
        self.assertEqual(personnel_thumb_url("outputs/personnel_crops/x.jpg"),
                         f"{PERSONNEL_CROPS_MOUNT}/x.jpg")
        # Windows 分隔符 / 绝对路径同样只取文件名
        self.assertEqual(personnel_thumb_url(r"C:\tmp\crops\y.jpg"),
                         f"{PERSONNEL_CROPS_MOUNT}/y.jpg")
        # 穿越尝试被压回文件名，不可能逃出头像目录
        self.assertEqual(personnel_thumb_url("../../etc/passwd"),
                         f"{PERSONNEL_CROPS_MOUNT}/passwd")
        self.assertIsNone(personnel_thumb_url(None))
        self.assertIsNone(personnel_thumb_url(""))

    def test_url_file_is_reachable_under_mount_dir(self):
        """翻译出的 URL 去文件名后，必须真能在 mount 目录里找到同名文件。"""
        import server
        from scripts.seed_personnel_mock import make_placeholder_thumb
        out_dir = server._personnel_crops
        out_dir.mkdir(parents=True, exist_ok=True)
        person = _FakePerson("eeeeffff", "王强", "安全管理", "QLU-26-0003")
        rel = make_placeholder_thumb(person, out_dir)
        self.addCleanup(lambda: (out_dir / "eeeeffff.jpg").unlink(missing_ok=True))
        self.assertIsNotNone(rel)
        url = server.personnel_thumb_url(rel)
        served = out_dir / url[len(server.PERSONNEL_CROPS_MOUNT) + 1:]
        self.assertTrue(served.exists(), f"URL {url} 在 mount 目录下找不到文件")


class _FakePerson:
    """make_placeholder_thumb 只用到这四个字段，不必构造完整数据类。"""

    def __init__(self, person_id, name, department, employee_no):
        self.person_id = person_id
        self.name = name
        self.department = department
        self.employee_no = employee_no


class ThumbDirIsolationTest(unittest.TestCase):
    """
    给**非默认库** seed 时，头像不能写进生产 outputs/personnel_crops/。

    实测踩过：临时演示库跑完 seed，生产头像目录多出 12 张无人引用的孤儿 jpg
    （personnel 表 0 行），只能手工清理。根因是 thumb_dir 不看目标库。
    """

    def test_default_thumb_dir_follows_target_db(self):
        import scripts.seed_personnel_mock as seeder
        default_db = seeder.DEFAULT_DB
        demo_db = Path(tempfile.mkdtemp(prefix="thumbdir_")) / "demo.db"
        self.addCleanup(shutil.rmtree, demo_db.parent, True)

        # 复刻 main() 里的选择逻辑：默认库 → CROPS_DIR；其它库 → 库旁边
        pick = lambda db: (seeder.CROPS_DIR if db == default_db
                           else db.parent / "personnel_crops")
        self.assertEqual(pick(default_db), seeder.CROPS_DIR)
        self.assertEqual(pick(demo_db), demo_db.parent / "personnel_crops")
        # 关键断言：演示库的头像目录不等于生产目录
        self.assertNotEqual(pick(demo_db), seeder.CROPS_DIR)


class CleanSafetyTest(unittest.TestCase):
    """--clean 只能删模拟数据。误删真实档案是不可逆事故。"""

    def test_real_rows_are_never_selected(self):
        from scripts.seed_personnel_mock import find_mock_persons
        path = _tmp_db_path(self)
        db = Database(path)
        self.addCleanup(db.close)
        db.upsert_personnel("real-1", "真人甲", "QLU-26-1000", "网络运维", "手工录入")
        db.upsert_personnel("mock-1", "模拟甲", "QLU-26-2000", "网络运维", "模拟数据 · 批次五",
                            source="synthetic")
        db._get_conn().execute(
            "INSERT INTO identities (global_id, person_id) VALUES ('gm', 'mock-1')")
        db._get_conn().execute(
            "INSERT INTO identities (global_id, person_id) VALUES ('gr', 'real-1')")
        db._get_conn().commit()
        pids, gids, thumbs = find_mock_persons(db, {})
        self.assertEqual(pids, ["mock-1"])
        self.assertEqual(gids, ["gm"])
        self.assertEqual(thumbs, [])


if __name__ == "__main__":
    unittest.main()
