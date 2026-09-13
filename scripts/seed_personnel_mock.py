"""seed_personnel_mock.py — 生成行人姓名模拟数据并落库（批次五 P0-1）。

为什么需要这一步
----------------
`personnel` / `personnel_photos` 实测 0 行、6 个身份 0 个绑名：批次三的实名档案与
批次四的视频检索从未被真实数据执行过。没有数据就无法验收，也无法继续做缩略图。

⚠ 这批数据是**演示用**的。`--mode synthetic` 的特征来自随机数，1:N 匹配会表现完美 ——
那是随机数的性质，不是识别能力。前端对 source='synthetic' 强制打 SIM 徽标。
要演示真实识别请用 `--mode real`（对 videos_low 跑真实 detector + ReID）。

用法：
    # 先看会生成什么（不写库）
    ./.venv/Scripts/python.exe scripts/seed_personnel_mock.py --count 40 --dry-run

    # 正式写入生产库（自动备份 + 落 manifest）
    ./.venv/Scripts/python.exe scripts/seed_personnel_mock.py --count 40

    # 写进临时库（联调前端用，不碰生产）
    ./.venv/Scripts/python.exe scripts/seed_personnel_mock.py --db outputs/mock_demo.db --count 40

    # 演示前把 mock 数据整体平移到"现在附近"（对抗 30 天保留策略）
    ./.venv/Scripts/python.exe scripts/seed_personnel_mock.py --refresh-timestamps

    # 精确回滚（按 manifest 删除本次 seed 的人 / 身份 / 轨迹 / 图片）
    ./.venv/Scripts/python.exe scripts/seed_personnel_mock.py --clean

约束详见 src/mock_personnel.py 的模块 docstring 与
docs/PLAN_2026-09-12_personnel_mock_data.md §0.1。
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import Database  # noqa: E402
from src.mock_personnel import (  # noqa: E402
    AGGREGATE_IDENTITY_THRESHOLD, generate_persons,
)
from src.reid_config import FEATURE_DIM, get_reid_weight  # noqa: E402

DEFAULT_DB = ROOT / "outputs" / "lab_monitor.db"
CROPS_DIR = ROOT / "outputs" / "personnel_crops"
NOTE_TAG = "模拟数据 · 批次五"


def manifest_path_for(db_path: Path) -> Path:
    """manifest 跟着目标库走。

    写死成 outputs/mock_seed_manifest.json 的话，给临时演示库 seed 会覆盖
    生产库的回滚清单 —— 那次 --clean 就不知道该删谁了。
    """
    return db_path.with_name(f"{db_path.stem}.mock_manifest.json")


def relative_thumb_path(target: Path) -> str:
    """
    头像的存储路径，**相对项目根**，例：outputs/personnel_crops/9f2c.jpg。

    两个消费方必须共用这一个口径（此前它们各写各的、拼法还不一样）：
      - 清理：cmd_clean 的 --purge-images 做 ROOT / thumb_path 当文件路径用
      - 出图：server.py 的 personnel_thumb_url() 把它翻译成 /personnel-crops/<file>
              （前端只消费后端给的 thumb_url，不再自己拼）

    历史 bug（本次修掉）:
      1. 存在库里的值被硬编码成 "personnel_crops/<file>"（少一层 outputs/），
         --purge-images 解析成 ROOT/personnel_crops/<file> → 文件根本不在那，
         静默删不掉任何图片（它只 log 成功数量，不报错）。
      2. 那个字符串又被前端拼成 URL。真正能提供文件的是 server.py 的
         StaticFiles mount /personnel-crops（连字符），而存储值里是下划线
         personnel_crops → 每个头像都 404，全部退化成首字块。
      两个 bug 同源：一个字符串被两个消费方各自解析，且谁都不知道对方怎么解析。

    所以这里统一返回相对 ROOT 的真实磁盘位置，并由 server.py 的
    personnel_thumb_url() 负责"磁盘路径 → URL"的翻译。
    """
    try:
        return target.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        # 目录在项目外（--thumb-dir 指到别处）：存绝对路径，保证 --purge-images 仍能删
        return target.resolve().as_posix()


#: 写入走 SQLite 批量事务。逐条 record_appearance 每行都 commit 一次，
#: 40 人 ~4k 行在 Windows 上实测要几十秒；这里按 gid 分组，一个事务写完
#: 一个身份的全部轨迹（语义与逐条完全一致，只是少 commit）。
BATCH_SIZE = 500


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# manifest                                                                     #
# --------------------------------------------------------------------------- #

def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        log(f"⚠ manifest 读取失败（{exc}），改用 source 列识别 mock 数据")
        return {}


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# 资产读取                                                                      #
# --------------------------------------------------------------------------- #

def load_assets_by_camera(database: Database, prefer_low: bool = True) -> dict[str, dict]:
    """每路相机挑一个可用资产行。

    优先 videos_low（体积小、seek 快，且 /media 回放默认也用它）。关键：必须选
    **有 frames_real 实测值**的行 —— 容器元数据不可信（见 seed_video_assets.py 模块
    docstring），没有 frames_real 就无法生成合法的 video_frame。
    """
    chosen: dict[str, dict] = {}
    for asset in database.list_video_assets():
        cam = asset.get("camera_id")
        if not cam:
            continue
        rel = str(asset.get("rel_path") or "")
        is_low = rel.startswith("videos_low")
        row = {
            "asset_id": asset.get("asset_id"),
            "camera_id": cam,
            "rel_path": rel,
            "duration_real": float(asset.get("duration_real") or 0.0),
            "fps_declared": float(asset.get("fps_declared") or 25.0) or 25.0,
            "frames_real": int(asset.get("frames_real") or 0),
            "is_low": is_low,
        }
        if row["duration_real"] <= 0:
            continue
        prev = chosen.get(cam)
        if prev is None:
            chosen[cam] = row
            continue
        # 排序键：先满足"低清偏好"，再要求 frames_real>0，最后比 asset_id 稳定
        def rank(item: dict) -> tuple:
            low_ok = (item["is_low"] == prefer_low) if prefer_low else True
            return (low_ok, item["frames_real"] > 0, -(item["asset_id"] or 0))
        if rank(row) > rank(prev):
            chosen[cam] = row
    return chosen


# --------------------------------------------------------------------------- #
# 写入                                                                          #
# --------------------------------------------------------------------------- #

def write_persons(database: Database, persons: list, progress_every: int = 10) -> dict:
    """把生成好的人写入 personnel / identities / identity_appearances / personnel_photos。"""
    conn = database._get_conn()  # noqa: SLF001 - 需要批量事务，公开 API 只有逐条 commit
    stats = {"personnel": 0, "identities": 0, "appearances": 0, "photos": 0, "thumbs": 0}
    t0 = time.time()

    for idx, person in enumerate(persons, start=1):
        ok = database.upsert_personnel(
            person.person_id, person.name, person.employee_no,
            person.department, person.note, source=person.source,
        )
        if not ok:
            raise RuntimeError(f"写档案失败: {person.name}")
        stats["personnel"] += 1

        if person.gallery_feature is not None:
            photo_id = database.save_personnel_photo(
                person.person_id, FEATURE_DIM,
                person.gallery_feature.astype("float32").tobytes(),
                0.8, person.thumb_path,
            )
            if photo_id:
                stats["photos"] += 1

        for ident in person.identities:
            if not ident.appearances:
                continue  # 空轨迹身份不写：会造成"档案里有人但永远搜不到"的困惑
            first = ident.appearances[0]
            last = ident.appearances[-1]
            database.save_identity(
                global_id=ident.global_id,
                feature_dim=ident.feature_dim,
                feature_blob=ident.blob(),
                feature_bank_count=1,
                feature_bank_blob=ident.blob(),
                total_appearances=0,          # 下面用批量轨迹行回填真实计数
                last_camera=last.camera_id,
                last_seen=last.timestamp,
                first_seen=first.timestamp,
                feature_space=ident.feature_space,
                schema_version=1,
                person_id=person.person_id,
                name_confidence=ident.name_confidence,
                new_appearance=None,
            )
            stats["identities"] += 1

            rows = [
                (ident.global_id, a.camera_id, a.timestamp, json.dumps(a.bbox),
                 a.asset_id, a.video_frame, a.video_ts)
                for a in ident.appearances
            ]
            for chunk_start in range(0, len(rows), BATCH_SIZE):
                batch = rows[chunk_start:chunk_start + BATCH_SIZE]
                conn.executemany(
                    """
                    INSERT INTO identity_appearances (
                        global_id, camera_id, timestamp, bbox_json,
                        asset_id, video_frame, video_ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                stats["appearances"] += len(batch)
            conn.execute(
                "UPDATE identities SET total_appearances = ?, last_camera = ?, "
                "last_seen = ?, first_seen = ? WHERE global_id = ?",
                (len(rows), last.camera_id, last.timestamp, first.timestamp,
                 ident.global_id),
            )
            conn.commit()

        if person.thumb_path:
            database.set_personnel_thumb(person.person_id, person.thumb_path)
            stats["thumbs"] += 1

        if idx % progress_every == 0 and sys.stdout.isatty():
            log(f"  … {idx}/{len(persons)} 人 · {stats['appearances']} 条轨迹 · "
                f"{time.time() - t0:.1f}s")

    return stats


# --------------------------------------------------------------------------- #
# 动作：seed / clean / refresh / report                                         #
# --------------------------------------------------------------------------- #

_CJK_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)


def _load_font(size: int):
    """按字号加载任意一个可用中文字体；全部失败返回 PIL 默认位图字体。"""
    from PIL import ImageFont
    for candidate in _CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def department_color(department: str) -> tuple[int, int, int]:
    """部门 → 稳定的 (r, g, b)。

    必须用 md5 而不是内置 hash()：后者按进程随机化（PYTHONHASHSEED），
    同一部门在两次运行里会拿到不同颜色，前后端配色也就对不上了。
    饱和度/明度固定在中低档，保证深色与浅色主题下白字都可读。
    """
    digest = hashlib.md5((department or "").encode("utf-8")).digest()
    r, g, b = colorsys.hsv_to_rgb(digest[0] / 255.0, 0.45, 0.42)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def make_placeholder_thumb(person, out_dir: Path) -> str | None:
    """
    Tier B 的头像：PIL 画「姓氏首字 + 部门色 + SIM 水印」占位块。

    为什么不用纯 CSS 首字块：档案卡要有"人像"的形状才直观，且 CSS 块无法
    带 SIM 水印 —— 截图外流时必须一眼能看出是演示数据。

    PIL 缺失或字体不可用时返回 None（前端退化成姓名首字块）。
    绝不让 seed 因为头像失败而整体中断 —— 数据比图片重要。
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        width, height = 96, 120
        bg = department_color(person.department)
        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)

        initial = person.name[:1]
        font = _load_font(52)
        box = draw.textbbox((0, 0), initial, font=font)
        draw.text(
            ((width - (box[2] - box[0])) / 2 - box[0],
             (height - 22 - (box[3] - box[1])) / 2 - box[1]),
            initial, font=font, fill=(255, 255, 255),
        )
        draw.rectangle([0, height - 22, width, height], fill=(17, 24, 39))
        draw.text((5, height - 18), "SIM · " + person.employee_no,
                  font=_load_font(11), fill=(226, 232, 240))

        target = out_dir / f"{person.person_id}.jpg"
        image.save(target, "JPEG", quality=82)
        return relative_thumb_path(target)
    except Exception as exc:  # noqa: BLE001 - 头像失败不阻断 seed
        log(f"  ⚠ 生成头像失败 {person.person_id}: {exc}")
        return None


def backup_database(db_path: Path) -> Path | None:
    """
    备份目标库，返回备份文件路径（库不存在返回 None）。

    必须走 SQLite backup API 而不是 shutil.copy2：库是 WAL 模式（src/db.py），
    运行中直接 copy 主文件会漏掉尚未 checkpoint 进主库的 -wal 事务，
    得到一个"看起来完整、恢复后丢最近写入"的备份。backup API 经由连接
    把主库 + WAL 的已提交内容一起搬过去，天然一致。项目此前已经为
    "复制 WAL 数据库"交过一次学费（TODO_2026-09-12_worklist.md）。
    """
    if not db_path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = db_path.with_name(f"{db_path.stem}.bak-mock-{stamp}{db_path.suffix}")
    source = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(str(target))
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()
    log(f"已备份数据库（WAL 一致性快照）→ {target.name}")
    return target


def cmd_seed(args: argparse.Namespace) -> int:
    if args.mode == "real":
        # 宁可拒绝，也不能把随机数标成 real。
        # generate_persons 目前只会产 synth_feature；若照常写 source='real'，
        # 前端就不打 SIM 徽标，一份"看起来是真人识别结果"的假数据会一路流到汇报里。
        # P1-2 接入真实 detector + ReID（src/mock_personnel 的 real 分支）后再放开。
        log("✗ --mode real 尚未实现（P1-2）：当前生成器只会产随机特征，")
        log("  标成 real 会让假数据看起来像真实识别结果。")
        log("  现在请用默认的 --mode synthetic（前端会强制显示 [模拟] 徽标）。")
        return 3

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 备份必须在 Database(db_path) 之前：构造函数会对目标库跑 schema 迁移，
    # 先开库再备份会把迁移结果一起打进"回滚点"，备份就不再是纯回滚点了。
    if db_path == DEFAULT_DB:
        if not args.dry_run:
            backup_database(db_path)
    else:
        log(f"目标不是默认库（{db_path.name}），跳过备份")

    sources = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8"))
    topology = json.loads((ROOT / "config" / "topology.json").read_text(encoding="utf-8"))
    known_cameras = list(sources)

    feature_space = get_reid_weight(None).feature_space
    log(f"feature_space = {feature_space}（运行时取值，未硬编码）")

    now = time.time()
    if not 1 <= args.days <= 28:
        log(f"⚠ --days {args.days} 超出安全区间 (1~28)："
            f"main.py 每次启动无条件跑 apply_retention(30)，超过 30 天的数据会被删")
    window_days = min(max(1, args.days), 6)

    database = Database(db_path)
    try:
        # 幂等护栏：工号由固定序号生成（QLU-26-0001…），person_id 却是新 UUID。
        # 重复 seed 会写出一批"工号/姓名与旧数据重复"的新档案，manifest 只追加
        # 不拦截，--clean 按 manifest 回滚也会对不上账 —— 所以直接拒绝。
        conn = database._get_conn()  # noqa: SLF001 - 与 write_persons/find_mock_persons 同口径
        existing = conn.execute(
            "SELECT COUNT(*) FROM personnel WHERE source = 'synthetic' OR note LIKE ?",
            (f"%{NOTE_TAG}%",),
        ).fetchone()[0]
        if existing and not args.dry_run:
            log(f"✗ 目标库已存在 {existing} 条模拟档案，拒绝重复写入。")
            log("  再次 seed 会生成新 person_id，但工号/姓名与旧数据重复；")
            log("  请先回滚再种，或换一个库：")
            log(f"    python scripts/seed_personnel_mock.py --db {args.db} --clean --yes")
            return 4
        if existing:
            log(f"⚠ 目标库已存在 {existing} 条模拟档案（当前是 dry-run，不写入）")

        assets = load_assets_by_camera(database)
        usable = [c for c in known_cameras if c in assets]
        if not usable:
            log("✗ video_assets 表里没有可用资产行。先跑 scripts/seed_video_assets.py")
            return 2
        log(f"相机：sources {len(known_cameras)} 路 / 有资产 {len(usable)} 路 "
            f"/ 拓扑有出边 {len([c for c in usable if topology.get(c)])} 路")
        if len(usable) < len(known_cameras):
            missing = sorted(set(known_cameras) - set(usable))
            log(f"  ⚠ 无资产（将不生成其轨迹）: {' '.join(missing)}")

        persons = generate_persons(
            args.count,
            known_cameras=usable,       # 只在**有资产**的相机上生成轨迹：
            topology=topology,          # 传全量 known_cameras 会让生成器选中
            assets_by_camera=assets,    # 无资产行的相机，写出 video_ts=NULL 的轨迹
            feature_space=feature_space,
            feature_dim=FEATURE_DIM,
            rng=np.random.default_rng(args.seed),
            source=args.mode,
            window_start=now - window_days * 86400,
            window_span=(window_days - 0.2) * 86400,
            loops=args.loops,
            align_tail_to=now - 120.0,      # 最晚的人停在 2 分钟前，让"刚刚"有数据
        )

        gids = [i.global_id for p in persons for i in p.identities]
        apps = [a for p in persons for i in p.identities for a in i.appearances]
        cams_used = sorted({a.camera_id for a in apps})
        log(f"\n计划生成：{len(persons)} 人 / {len(gids)} 身份 / {len(apps)} 轨迹行 "
            f"/ {len(cams_used)} 路相机")
        log(f"时间跨度：{(now - min((a.timestamp for a in apps), default=now)) / 3600:.1f}h "
            f"~ {(now - max((a.timestamp for a in apps), default=now)) / 60:.1f}min 前")

        if args.dry_run:
            log("\n[dry-run] 未写入任何数据。示例：")
            for p in persons[:5]:
                log(f"  {p.name:6} {p.employee_no}  {p.department:6} "
                    f"{len(p.identities)} 身份 / {p.total_appearances:4} 行 / "
                    f"最后 {time.strftime('%m-%d %H:%M', time.localtime(p.last_seen))}")
            return 0

        # 头像必须在写库前生成：thumb_path 要进 personnel 行和注册照的 source_path
        #
        # 目录必须**跟着目标库走**：给演示库 seed 时若仍写生产 outputs/personnel_crops/，
        # 那个库里的人一旦被 --clean 删掉，这些图就变成无人引用的孤儿
        # （实测踩过一次：临时库跑完，生产头像目录多出 12 张孤儿 jpg）。
        # 所以：未显式指定 --thumb-dir 时，把图放在 <目标库同级>/personnel_crops/。
        default_thumb_dir = (CROPS_DIR if db_path == DEFAULT_DB
                             else db_path.parent / "personnel_crops")
        thumb_dir = Path(args.thumb_dir) if args.thumb_dir else default_thumb_dir
        if not thumb_dir.is_absolute():
            thumb_dir = ROOT / thumb_dir
        thumbs_ok = 0
        if not args.no_thumbs:
            for p in persons:
                p.thumb_path = make_placeholder_thumb(p, thumb_dir)
                thumbs_ok += 1 if p.thumb_path else 0
            log(f"头像：{thumbs_ok}/{len(persons)} 张 → {thumb_dir}"
                f"{'（PIL 不可用，前端将退化为姓名首字块）' if thumbs_ok == 0 else ''}")

        t0 = time.time()
        stats = write_persons(database, persons)
        log(f"\n写入完成 · {time.time() - t0:.1f}s：{stats}")

        manifest = {
            "seeded_at": now,
            "mode": args.mode,
            "db": str(db_path),
            "feature_space": feature_space,
            "person_ids": [p.person_id for p in persons],
            "employee_nos": [p.employee_no for p in persons],
            "global_ids": gids,
            "appearance_rows": len(apps),
            "thumb_paths": [p.thumb_path for p in persons if p.thumb_path],
            "thumb_dir": str(thumb_dir),
            "note_tag": NOTE_TAG,
        }
        manifest_path = manifest_path_for(db_path)
        existing = read_manifest(manifest_path)
        runs = existing.get("runs") or []
        runs.append(manifest)
        write_manifest(manifest_path, {"runs": runs[-20:], "latest": manifest})
        log(f"manifest → {manifest_path.name}（--clean 依据它回滚）")

        log("\n⚠ 生效前提：seed 的身份要能被检索/展示，必须**重启服务** ——")
        log("   IdentityStore 只在启动时 _restore()，/api/search/person 只查内存 store。")
        log("   人员档案列表（/api/personnel）读库，无需重启即可看到。")
        return 0
    finally:
        database.close()


def find_mock_persons(database: Database, manifest: dict) -> tuple[list[str], list[str]]:
    """manifest 优先，source 列兜底。双保险：手改过姓名也仍能清干净。

    只认 'synthetic'：--mode real（P1-2）写入的是真人抓拍特征，
    与用户手工建的 real 档案无法区分，一律删会误伤真实数据。
    要删 real 档请用 --person-id 指名。
    """
    person_ids = list(manifest.get("person_ids") or [])
    global_ids = list(manifest.get("global_ids") or [])
    thumb_paths = [t for t in (manifest.get("thumb_paths") or []) if isinstance(t, str)]
    conn = database._get_conn()  # noqa: SLF001
    rows = conn.execute(
        "SELECT person_id FROM personnel WHERE source = 'synthetic' OR note LIKE ?",
        (f"%{NOTE_TAG}%",),
    ).fetchall()
    for row in rows:
        if row[0] not in person_ids:
            person_ids.append(row[0])
    if person_ids:
        placeholders = ",".join("?" * len(person_ids))
        extra = conn.execute(
            "SELECT global_id FROM identities WHERE person_id IN "
            f"({placeholders})",
            person_ids,
        ).fetchall()
        for row in extra:
            if row[0] not in global_ids:
                global_ids.append(row[0])
        # thumb_path 也存在于库里（--refresh 之类的操作可能重建过 manifest）
        for row in conn.execute(
            f"SELECT thumb_path FROM personnel WHERE person_id IN ({placeholders}) "
            "AND thumb_path IS NOT NULL",
            person_ids,
        ).fetchall():
            if row[0] and row[0] not in thumb_paths:
                thumb_paths.append(row[0])
    return person_ids, global_ids, thumb_paths


def cmd_clean(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    if not db_path.exists():
        log(f"✗ 数据库不存在: {db_path}")
        return 2

    database = Database(db_path)
    try:
        manifest_path = manifest_path_for(db_path)
        manifest = read_manifest(manifest_path)
        runs = manifest.get("runs") or ([manifest] if manifest.get("person_ids") else [])
        all_pids: list[str] = []
        all_gids: list[str] = []
        all_thumbs: list[str] = []
        for run in runs:
            pids, gids, thumbs = find_mock_persons(database, run)
            all_pids += [p for p in pids if p not in all_pids]
            all_gids += [g for g in gids if g not in all_gids]
            all_thumbs += [t for t in thumbs if t not in all_thumbs]
        if args.person_id:
            all_pids = [args.person_id]
            all_gids = [r["global_id"] for r in
                        database.identities_for_person(args.person_id)]

        if not all_pids and not all_gids:
            log("没有找到模拟数据（manifest 为空且无 source 标记的行）。未做任何改动。")
            return 0
        if not args.yes:
            log(f"将删除：{len(all_pids)} 人 / {len(all_gids)} 身份（连带其轨迹与注册照）")
            log("  这是不可逆操作。确认请加 --yes")
            return 1

        backup_database(db_path)
        conn = database._get_conn()  # noqa: SLF001
        deleted = {"photos": 0, "personnel": 0, "identities": 0, "appearances": 0}
        for pid in all_pids:
            deleted["photos"] += conn.execute(
                "DELETE FROM personnel_photos WHERE person_id = ?", (pid,)).rowcount
        if all_gids:
            ph = ",".join("?" * len(all_gids))
            deleted["appearances"] += conn.execute(
                f"DELETE FROM identity_appearances WHERE global_id IN ({ph})",
                all_gids).rowcount
            deleted["identities"] += conn.execute(
                f"DELETE FROM identities WHERE global_id IN ({ph})", all_gids).rowcount
        if all_pids:
            ph = ",".join("?" * len(all_pids))
            deleted["personnel"] += conn.execute(
                f"DELETE FROM personnel WHERE person_id IN ({ph})", all_pids).rowcount
        conn.commit()

        removed_files = 0
        if args.purge_images:
            # 按 manifest / 库里记录的相对路径删，而不是 glob 某个写死的目录：
            # 用户可能用 --thumb-dir 指到了别处，glob 既会漏删也可能删到别人的文件。
            for rel in all_thumbs:
                candidate = (ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
                if candidate.exists():
                    try:
                        candidate.unlink()
                        removed_files += 1
                    except OSError as exc:
                        log(f"  ⚠ 删图失败 {candidate.name}: {exc}")

        write_manifest(manifest_path, {"runs": [], "latest": {},
                                       "last_clean": time.time()})
        log(f"已清理：{deleted} · 图片 {removed_files} 张 · manifest 已清空")
        log("⚠ 重启服务后，内存 store 才会忘掉这些身份（重启前 /api/identities 仍会列出它们）")
        return 0
    finally:
        database.close()


def cmd_refresh_timestamps(args: argparse.Namespace) -> int:
    """把模拟数据整体平移到近期，对抗 apply_retention(30)。纯 UPDATE，不改生产语义。"""
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    database = Database(db_path)
    try:
        conn = database._get_conn()  # noqa: SLF001
        mock_gids = [r[0] for r in conn.execute(
            "SELECT i.global_id FROM identities i JOIN personnel p "
            "ON p.person_id = i.person_id WHERE p.source = 'synthetic'"
        ).fetchall()]
        if not mock_gids:
            log("没有找到模拟身份。先跑 seed。")
            return 1
        ph = ",".join("?" * len(mock_gids))
        latest = conn.execute(
            f"SELECT MAX(timestamp) FROM identity_appearances "
            f"WHERE global_id IN ({ph})", mock_gids).fetchone()[0]
        if latest is None:
            log("模拟身份没有轨迹行。")
            return 1
        now = time.time()
        shift = (now - 120.0) - float(latest)
        if args.dry_run:
            log(f"[dry-run] 将把 {len(mock_gids)} 个模拟身份的轨迹整体平移 "
                f"{shift / 3600:+.1f} 小时（{len(mock_gids)} gid）")
            return 0
        backup_database(db_path)
        rows = conn.execute(
            f"UPDATE identity_appearances SET timestamp = timestamp + ? "
            f"WHERE global_id IN ({ph})", (shift, *mock_gids)).rowcount
        conn.execute(
            f"UPDATE identities SET first_seen = first_seen + ?, "
            f"last_seen = last_seen + ? WHERE global_id IN ({ph})",
            (shift, shift, *mock_gids))
        conn.commit()
        log(f"已平移 {rows} 条轨迹 + {len(mock_gids)} 行身份，"
            f"偏移 {shift:+.1f} 秒（{shift / 3600:+.2f} 小时）")
        log("⚠ 需要重启服务让内存 store 的 last_seen 与新库一致")
        return 0
    finally:
        database.close()


def cmd_report(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        def one(sql: str) -> int:
            try:
                return int(conn.execute(sql).fetchone()[0] or 0)
            except sqlite3.Error:
                return 0

        total_p = one("SELECT COUNT(*) FROM personnel")
        mock_p = one("SELECT COUNT(*) FROM personnel WHERE source IS NOT NULL")
        print(f"人员档案  {mock_p:6} 模拟 / {total_p:6} 总计")
        print(f"注册照    {one('SELECT COUNT(*) FROM personnel_photos'):6}")
        total_i = one("SELECT COUNT(*) FROM identities")
        bound = one("SELECT COUNT(*) FROM identities WHERE person_id IS NOT NULL")
        print(f"身份      {bound:6} 已绑定 / {total_i:6} 总计")
        print(f"轨迹行    {one('SELECT COUNT(*) FROM identity_appearances'):6} "
              f"(带视频坐标 {one('SELECT COUNT(*) FROM identity_appearances WHERE video_ts IS NOT NULL')})")
        lo, hi = conn.execute(
            """SELECT MIN(a.timestamp), MAX(a.timestamp) FROM identity_appearances a
               JOIN identities i ON i.global_id = a.global_id
               JOIN personnel p ON p.person_id = i.person_id
               WHERE p.source IS NOT NULL""").fetchone()
        if lo and hi:
            now = time.time()
            print(f"模拟时间跨度  {time.strftime('%m-%d %H:%M', time.localtime(lo))}"
                  f" ~ {time.strftime('%m-%d %H:%M', time.localtime(hi))} "
                  f"（最晚 {(now - hi) / 60:.0f} 分钟前；保留期 30 天）")
        top = conn.execute(
            """SELECT p.name, p.employee_no, p.department, p.source,
                      COUNT(DISTINCT i.global_id) gids,
                      COALESCE(SUM(i.total_appearances),0) apps
               FROM personnel p LEFT JOIN identities i ON i.person_id = p.person_id
               GROUP BY p.person_id ORDER BY apps DESC LIMIT 5""").fetchall()
        if top:
            print("\n最活跃档案 top5:")
            for r in top:
                flag = " [SIM]" if r["source"] == "synthetic" else ""
                over = "  ⚠ 超聚合阈值" if r["apps"] > AGGREGATE_IDENTITY_THRESHOLD else ""
                print(f"  {r['name']:6} {r['employee_no'] or '-':13} "
                      f"{r['department'] or '-':6} {r['gids']} 身份 {r['apps']:6} 行{flag}{over}")
        return 0
    finally:
        conn.close()


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="行人姓名模拟数据生成器（批次五）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="目标数据库路径")
    parser.add_argument("--count", type=int, default=40, help="生成人数（默认 40）")
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic",
                        help="synthetic=随机特征(秒级,仅演示 UI)；real=真实抓拍(需模型)")
    parser.add_argument("--days", type=float, default=7,
                        help="把轨迹铺进最近 N 天（默认 7；>28 会被 30 天保留策略吃掉）")
    parser.add_argument("--loops", type=int, default=2,
                        help="每人行程重复几遍，用于产生 loop_factor（默认 2）")
    parser.add_argument("--seed", type=int, default=20260912, help="随机种子（可复现）")
    parser.add_argument("--thumb-dir", default=None,
                        help=f"头像输出目录（默认 {CROPS_DIR.name}/）")
    parser.add_argument("--no-thumbs", action="store_true",
                        help="不生成头像（前端退化为姓名首字块）")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写库")
    parser.add_argument("--clean", action="store_true", help="删除模拟数据（按 manifest）")
    parser.add_argument("--yes", action="store_true", help="配合 --clean 跳过确认")
    parser.add_argument("--purge-images", action="store_true", help="配合 --clean 删抓拍图")
    parser.add_argument("--person-id", default=None, help="配合 --clean 只删某一人")
    parser.add_argument("--refresh-timestamps", action="store_true",
                        help="把模拟数据整体平移到近期（演示前跑一次）")
    parser.add_argument("--report-only", action="store_true", help="只打印现状统计")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report_only:
        return cmd_report(args)
    if args.clean:
        return cmd_clean(args)
    if args.refresh_timestamps:
        return cmd_refresh_timestamps(args)
    if args.count < 1:
        log("✗ --count 必须 >= 1")
        return 2
    return cmd_seed(args)


if __name__ == "__main__":
    raise SystemExit(main())
