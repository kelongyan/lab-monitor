# 行人姓名模拟数据 + 人员档案可视化检索 设计方案

> 2026-09-12 · 基于对 `main` @ `e237944` 的代码通读 + 四项本机实测
> 需求：能否随机生成行人姓名模拟数据并存储，在前台管理系统中更直观地显示，且可以检索行人信息？
> **答案：能，而且必须这么做才能推进 —— 因为 `personnel` 表当前是 0 行，整条实名链路从来没有真实数据流过。**

---

## 0. 先说结论

批次三把实名档案的**后端**做完了（CRUD 齐全、`test_personnel.py` 14 例通过），批次四把检索**后端**做完了。但四件事实测后暴露出来：

| 实测项 | 结果 | 判定 |
|---|---|---|
| `select count(*) from personnel` / `personnel_photos` | **0 / 0** | 档案系统空转。前端即使写完也没有一张卡可渲染 |
| `identities` 中 `person_id is not null` 的行数 | **0 / 6** | 底库 1:N 自动命名、`person_label()`、「姓名 (工号)」标签，全部处于**从未被执行**的状态 |
| `identity_appearances` 中 `asset_id is not null` | **0 / 1718** | 现有轨迹全部是「老数据」，`/api/search/person` 每一行都会显示 `position_known=False` → 前端只能回「无视频内坐标」 |
| 前端对 `/api/personnel*`、`/api/search/by-image` 的调用 | **零匹配** | 后端 8 个端点无一个前端消费者；`#card-stat-persons`（文案「人员档案数据库」）实际打开的是匿名 gid 九宫格 |

**所以这不是"造点假数据好看"，而是：没有数据就无法验证已交付的两个批次，也无法继续做 4.4（缩略图）和以图搜人。**

### 0.1 四个必须先记住的硬约束（实测得出，不是推测）

1. **`/api/search/person` 会 404 掉 seed 出来的身份。**
   `server.py:1023` 是 `if _identity_store.get(global_id) is None: 404` —— 只查**内存 store**，不查库。
   而 store 的 `_restore()`（`identity_store.py:202-281`）只恢复满足 `feature_dim>0` 且 `len(blob)==dim*4` 且 `feature_space` **全等** 且 `schema_version==1` 的行。
   → **模拟身份必须写合法的 512 维 float32 特征，且 `feature_space` 必须与运行时 extractor 的字符串全等**（实测当前库内 6 行都是 `osnet-x0.25-market1501:512`；这个值**不能硬编码**，理由见 §4.2 第 4 步），否则检索链路直接不可用。这是本方案对数据格式的第一约束，不是可选项。

2. **每次启动都会删数据。** `main.py:285-286` 无条件调 `db.apply_retention(30)`，按 `timestamp` 墙钟删 `identity_appearances` / `identities`（`db.py:421-440`）。
   → 模拟时间戳必须落在 **now - 7 天** 之内；并提供 `--refresh-timestamps` 一键把历史 mock 数据整体平移到近期（纯 UPDATE，不改生产语义）。

3. **相机 id 必须来自 `sources.json`。** `search.py:81-86` 按 `known_cameras` 丢弃孤儿行。有效 id 共 22 个：
   `reg_01 reg_02 reg_05 reg_06 reg_08 reg_10` + `rnd_01 02 04 05 06 07 08 10 11 12 16 17 18 19 21 22`。

4. **中文名烧不进画面。** `pipeline.py:705-706` 用 `cv2.putText(FONT_HERSHEY_SIMPLEX)` 绘制 `person_label(gid)`＝「姓名 (person_id)」，该字体族无中文字形 → 一旦绑上中文名，画面标签会退化成 `????`。
   → 这是**已经存在、只是被空表掩盖的 bug**，本方案的模拟数据会立刻暴露它。必须同期修（见 §5.3）。

### 0.2 不要拿合成特征演示检索精度

`docs/PLAN_2026-09-12_identity_search_resolution.md:15` 实测：45 个真实身份之间余弦相似度 **p50 0.984**，990 对异人中 988 对越阈（阈值现为 `reid_config.py:54` 的 **0.68**，不是 AGENTS.md 里那个过期的 0.75）。
公共分量中心化已修，但**用 `rng.random(512)` 造出来的特征天然互相正交，1:N 会表现完美**。那是在演示一个不存在的结论。
→ 因此本方案分两档，且**明确标注哪一档能演示什么**（§2）。

---

## 1. 设计原则

- **不新增依赖。** `requirements.txt` 保持不含 `faker`；中文姓名词库自己写（约 120 行常量），PIL 已在环境中（实测 12.3.0），`C:\Windows\Fonts\simhei.ttf` / `msyh.ttc` 可用。
- **不改生产语义。** 不动 `apply_retention`、不动阈值、不动 `PersonTracker`（AGENTS.md 的 class-level `_count` 陷阱）。模拟数据靠**标记列**与真实数据共存。
- **复用既有写入路径。** 一律走 `db.save_identity()` / `db.record_appearance()` / `db.upsert_personnel()` / `db.save_personnel_photo()` / `db.set_identity_person()`，绝不自己拼 SQL 插表 —— 否则 `load_identities()` 的兼容分支会踩空。
- **前端不新增第三方库、不新增轮询。** 浏览器同域 HTTP/1.1 只有 6 个槽，已被 5 路 MJPEG + 1 WS 占满（`stream_manager.js:18`、`server.py:1314-1317`）。人员头像必须是**静态 JPG**，不能是 `<img src="/stream/...">`。

---

## 2. 数据分两档：`real` 与 `synthetic`

| | **Tier A · real（推荐默认）** | **Tier B · synthetic（开发/演示兜底）** |
|---|---|---|
| 特征来源 | 对 `videos_low/*.mp4` 离线跑 `PersonDetector`+`build_reid_extractor()`，取真人 crop 的**真实 512 维特征** | `rng` 生成 L2 归一化向量（姓氏基向量近似正交） |
| 头像 | 真人抓拍裁图 → `outputs/personnel_crops/{person_id}.jpg` | PIL 画「姓氏首字 + 部门色」占位块 |
| 能否演示 1:N 自动命名 / 以图搜人 | **能**（用的是同一套 `match_feature_detailed`） | **不能**，UI 上必须打 `SIM` 徽标 |
| 前置条件 | 加载模型（torch + 权重），22 路按 2s 步长采样约 1350 帧 | 无模型，秒级完成 |
| 实测可行性 | 低清 seek 34 ms/帧、1080p 230~330 ms/帧（`videos_low/reg_01.mp4` 5991 帧 25fps） | — |
| 落库字段 | `personnel.source='real'` | `personnel.source='synthetic'` |

两档**写同一套 schema、同一套校验**，前端只按 `source` 显示徽标。这样 UI/检索/接口开发完全不被模型阻塞：先 Tier B 打通全链路，再切 Tier A 出可信演示。

---

## 3. 数据模型改动（最小面）

沿用 `db.py:180-190` 的迁移字典机制（老库自动 `ALTER TABLE`），**只加 2 列**：

```python
# src/db.py:180 附近的 migrations 字典追加
"source": "TEXT",            # 'real' | 'synthetic' | NULL(=人工/线上真实)
"thumb_path": "TEXT",        # 相对 outputs/ 的抓拍图路径，可空
```

`personnel` 现有列 `person_id / name / employee_no / department / note / created_at / updated_at`（`db.py:156-165`）**已经够用**，检索需要的字段一个都不用新加。

### 3.1 编号可读性（「可检索」的前提）

`person_id` 目前是 `uuid4().hex[:8]`（`personnel.py:148`），`docs/PLAN_...:341` 设想的 `P0007` 从未实现。
**不动 `person_id`**（它被 `identities.person_id`、`personnel_photos.person_id` 按约定关联，改格式风险大于收益），改用已存在却没人填的 **`employee_no`** 承载人眼可读编号：

```
employee_no = "QLU-26-0001"   # 齐鲁工业大学 + 年 + 顺序号，天然唯一、可搜、可口播
```

界面主展示 `姓名 + 工号`，`person_id` 降级为详情里的 `<span class="id-link">`。检索框支持 `姓名 / 工号 / 部门` 三合一模糊匹配。

### 3.2 垃圾身份保护

实测 `identities` 里 `c156549c` 单身份 `total_appearances=1561`，`last_camera=rnd_21` —— 这正是文档 §0.1 描述的「垃圾桶身份」形态。
→ 生成器**绝不给这类身份绑名**（`--max-gid-appearances` 默认 300 过滤），前端对超阈值身份显示 `⚠ 疑似聚合身份` 而非姓名。

---

## 4. 生成器设计

### 4.1 新文件 `src/mock_personnel.py`（纯函数 + 数据类，可单测）

```python
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜..."  # 80+
GIVEN_1 = "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀霞平刚桂英华玉萍红飞玲桂"
GIVEN_2 = ["建国","志强","秀英","海燕","文博","子轩","梓涵","一诺","思远","雨桐", ...]
DEPARTMENTS = ["高性能机房","网络运维","系统软件","存储系统","动力保障","安全管理","科研支持","行政后勤"]

@dataclass
class MockPerson:        # 一个人 = 1 条 personnel + 1~3 条 identities + N 条 appearances
    person_id: str; name: str; employee_no: str; department: str
    global_ids: list[str]; features: dict[str, np.ndarray]
    thumb_path: str | None; source: str

def make_name(rng, used: set[str]) -> str          # 姓 + 1~2 字名，重名重试 + 后缀去重
def make_walking_chain(rng, topology, n_stops) -> list[str]
    # 沿 config/topology.json 的真图走：reg_08→reg_02→reg_01→reg_05/reg_06→reg_10→reg_08
    # 相邻两站的停留间隔 = expected_seconds ± tolerance_seconds（让轨迹看起来是"人走的"）
def make_bbox(rng, frame_w=480, frame_h=270, person_h=90..170) -> list[float]
    # 按 frame_hub.DISPLAY_WIDTH/HEIGHT=480×270（frame_hub.py:40-41）生成合法 bbox，
    # 纵向随机、横向沿走廊方向平移，保证 x1<x2、y1<y2、不越界
def synth_feature(rng, dim=512, seed_basis=None) -> np.ndarray   # Tier B
```

`bbox` 与 `video_ts` 必须自洽：`video_ts = video_frame / fps_declared`，`video_frame ∈ [0, frames_real]`，墙钟 `timestamp` 与 `video_ts` 是**两条独立轴**（`db.py:129-131`：素材循环播放，墙钟只表示第几轮）。生成器按 `loop_factor` 把同一段 `video_ts` 区间复制若干轮，这样 `search.py:88-104` 算出的 `loop_factor` 才有意义、前端才会显示「循环 ×N」而不是「出现 3 万次」。

### 4.2 新文件 `scripts/seed_personnel_mock.py`（CLI，风格对齐 `scripts/seed_video_assets.py`）

```
.venv\Scripts\python.exe scripts\seed_personnel_mock.py --count 40 --mode real \
        [--db outputs/lab_monitor.db] [--days 7] [--seed 20260912]
        [--with-photos] [--clean] [--refresh-timestamps] [--dry-run] [--report-only]
```

流程（幂等）：
1. **备份**：`shutil.copy2(db, db.with_name(f"lab_monitor.bak-{ts}.db"))`。`src/db.py:1084` 是模块级单例，任何误写都可能污染生产库，这一步不可跳过（AGENTS.md 已警告测试碰生产库的问题）。
2. 打开 `Database(db_path)`，读 `video_assets` 拿每路相机的 `asset_id / frames_real / duration_real / rel_path`；优先 `videos_low/` 行（体积小、seek 快），并读 `config/topology.json`。
3. 生成 N 个 `MockPerson`：先 `upsert_personnel()` 建档案（含 `employee_no`/`department`/`note='模拟数据 · 批次五'`），再每人 1~3 个 gid。
4. **写身份**：`save_identity(gid, feature_dim=512, feature_blob=vec.astype(np.float32).tobytes(), feature_bank_count=1, feature_bank_blob=<同一向量>, total_appearances=0, last_camera=..., last_seen=..., feature_space='osnet-x0.25-market1501:512', first_seen=..., schema_version=1, person_id=pid, name_confidence=0.68~0.95)`
   —— `feature_space` 字符串**必须在运行时从 extractor 实例取**（`reid_extractor.feature_space`，与 `main.py:296` 同一口径）。
   它不是常量：`reid_config.py:116` 是 `ReIDWeight` 的 **property**，值是 `f"osnet-x0.25-{self.dataset}:{FEATURE_DIM}"`，
   实际可能是 `osnet-x0.25-market1501:512` 或 `osnet-x0.25-msmt17:512`（取决于哪个权重文件存在，`reid_config.py:50` 两个都标定过）。
   写死字符串 → 权重一换，`_restore()` 的全等比较（`identity_store.py:218-222`）会把 seed 身份**全部静默跳过**，
   表现为"数据没丢但检索 404"。`--mode synthetic` 下不加载 torch 时，用 `get_reid_weight(None).feature_space`（纯标准库，`reid_config.py:26` 明确该模块不引 torch/cv2）；
   注意 `resolve_reid_weight_path()` 只返回 `(ReIDWeight, Path)` 且权重缺失会抛异常，别拿它当"取字符串"的入口。
   另一个坑：权重缺失时线上会**降级**成 `resnet50-imagenet1k-v1:2048`（`reid.py:60`）或 ImageNet OSNet（`reid.py:168`），
   此时正在跑的 store 用的是降级字符串 —— seed 前先看启动日志里的 `特征空间=` 那行（`reid.py:154-157`）确认一致，否则要等补好权重再重启。
5. **写轨迹**：按行走链逐条 `record_appearance(gid, cam, ts, bbox, total, asset_id, video_frame, video_ts)`。
6. **写底库**：`save_personnel_photo(pid, 512, blob, quality=0.7~0.95, source_path=<crop 路径>)` + 回写 `thumb_path`。
7. **绑定**：`set_identity_person(gid, pid, confidence)`。
8. 落 `outputs/mock_seed_manifest.json`：`{person_ids, global_ids, photo_files, seeded_at, mode}` → `--clean` 据此精确回滚（先 `clear_identity_person`，再 `delete_identities(gids)`，最后 `delete_personnel(pid)`）。清单 + `source` 列双保险，改名后仍能清干净。
9. 报告：各表新增行数、覆盖相机数、时间跨度、最忙身份 top5、耗时。

`--refresh-timestamps` 的实现是**一条纯 UPDATE 平移**（三张相关列整体加同一个 offset），不改任何生产代码。

### 4.3 后端生效方式

seed 完必须让内存 store 看到。三条路，推荐第 ①：

① **重启服务**（零代码改动，最诚实）；
② `scripts/` 里跑完后打印提示「重启后生效」；
③ 新增 `POST /api/admin/reload-personnel`（需 `X-Lab-Monitor-Request`）→ `_personnel.reload()` + `_identity_store.set_person_names(_personnel.names())`。**只重读 personnel，不重读 identities**（`IdentityStore` 没有热重载入口，硬加会牵动锁内状态，风险不划算）。
→ 本方案取 ①+③：档案列表/底库能热更新，身份要重启。写进 `--report` 提示里，别让使用者以为数据丢了。

---

## 5. 前端「更直观」怎么落

### 5.1 新增 `static/js/modules/personnel.js`（把 1119 行的 `modals.js` 继续撑大是错路）

```
export async function openPersonnelModal(preset = {})   // 列表页：搜索 + 筛选 + 卡片网格
export async function showPersonDetail(person_id)        // 详情页：档案 + 名下身份 + 活动统计 + 视频检索
```

接线：`app.js:84-97` 目前把 `card-stat-persons`(header) 和 `card-stat-persons-btn`(统计卡) 都绑到 `openIdentitySearchModal`。改为
- header「人员档案数据库」→ `openPersonnelModal()`（**这才是它文案承诺的东西**）
- 统计卡 → 仍开 gid 九宫格，但标题改成「ReID 匿名身份」，并把 `stat-ids` 的口径说明补进 tooltip
`index.html` 增加第 7 个 `.modal-overlay` `#personnel-modal`（复用 trajectory-modal 会让 `beginModalRequest` 与「返回」栈互相打架），并在 `app.js:115-128` 的 `data-close-modal` 委托里登记。

### 5.2 列表页（核心界面）

```
┌─ 人员档案数据库 ────────────────────────────────────────────────┐
│ [🔍 姓名 / 工号 / 部门]  [部门 ▾]  [来源: 全部|模拟|真实]  [排序: 最近出现] │
│  共 40 人 · 12 人已绑定身份 · 覆盖 18 路相机 · 数据为演示用途        │
├───────────────────────────────────────────────────────────────┤
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                  │
│ │ 抓拍图  │ │ 抓拍图  │ │ 抓拍图  │ │ 姓氏块  │  ← 头像 64×80，     │
│ │ 张 伟   │ │ 李海燕  │ │ 王 磊  │ │ SIM    │     圆角 8px，      │
│ │QLU-26-  │ │QLU-26- │ │...     │ │        │     mono 工号       │
│ │ 0001    │ │ 0002   │ │ 机房运维│ │        │     + 部门徽标       │
│ │ 机房运维 │ │ 3天前   │ │ ⚠聚合   │ │        │     + 相对时间       │
│ │ 412 次  │ │ 2 身份  │ │        │ │        │     + 出现次数       │
│ └────────┘ └────────┘ └────────┘ └────────┘                  │
└───────────────────────────────────────────────────────────────┘
```

- 卡片基类复用 `.meta-card`（`modals.css:166-174`）+ 新 `.person-card`；徽标复用 `.id-link`（`alert-list.css:169-191`），**新增 `.person-badge`** —— 现有 `.badge` 家族全是硬编码一次性类，没有通用 badge 可用。
- 头像用 `<img loading="lazy">`，URL **由后端 `thumb_url` 给出**（`server.py: personnel_thumb_url()`；新 StaticFiles mount `/personnel-crops`，与 `server.py:42` 的 `/screenshots` 同一手法）。**绝不能用 `/stream/`**，否则挤占 5 路 MJPEG 预算。
  → **前端不得自己拼 URL**。这是已发生过的真实 bug（见 §5.4）：库里存的是磁盘路径 `outputs/personnel_crops/x.jpg`（下划线），mount 却是连字符的 `/personnel-crops`，前端拼 `'/' + thumb_path` 会让**每个头像都 404 并静默退化成首字块**，不报任何错。
- 搜索防抖 250 ms；前端只请求当前页（`limit=24&offset=`），不做全量渲染。
- **`SIM` 徽标必须有**：`source='synthetic'` 的人在卡片右上角打「模拟」，防止演示时被当成真实识别结果。
- CSS 落点：新 `static/css/components/personnel.css` → 加进 `main.css` 的 `@import`（**8 处 `?v=` 一起 bump**，`index.html:19` + `main.css` 内 7 行，两边不一致就命中旧缓存）。同时**必须在 `variables.css` 补 `html[data-theme="light"]` 覆盖**（那里 260 行手写逐类覆盖，漏写会在亮色主题下留下不可读暗块）。

### 5.3 顺手修的既有 bug（中文名渲染）

`pipeline.py:679-690` 的 `label_text` 含中文。改为 PIL 贴图：

```python
# 模块级：_FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)  —— 找不到时回退 cv2.putText
# 新增 src/text_overlay.py: draw_label(frame, xy, text, color) 
#   1) 按 text 做 LRU 补丁缓存（姓名集合有限 → 命中率≈100%，这是不拖垮 22 路 pipeline 的关键）
#   2) 缓存里存 RGBA ndarray，命中后只做一次 alpha 合成
```

**验证口径**：改前 `cv2.putText` 画「张伟」得 `???`；改后必须逐路对比首帧耗时（CPU 模式预算很紧）。若单帧增量 > 0.5 ms，退回「ASCII 化标签 + 前端 DOM 覆盖层显示中文」的方案（前端在 `<img>` 上叠 div，`grid.js:130` 的 `buildSingleCamCardHTML` 有 `data-cam-id` 可定位，且 `grid.js:150` 只在 camIds 变化时重建，所以逐帧更新卡内子元素是安全的）。

### 5.4 实施期发现并修掉的三个真实 bug（头像路径）

三个 bug **同源**：一个路径字符串被多处各自解析，且谁都不知道别人怎么解析。
`scripts/seed_personnel_mock.py` 曾把缩略图路径硬编码成 `f"personnel_crops/{name}"`（见 `make_placeholder_thumb`），
于是：

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | **每个头像都 404**，全部静默退化成姓名首字块，控制台不报错 | 库里存 `personnel_crops/x.jpg`（下划线、且少一层 `outputs/`），而对外提供文件的是 StaticFiles mount `/personnel-crops`（连字符）。前端 `thumbUrl()` 做 `'/' + thumb_path` → 404 | 存储侧改 `relative_thumb_path()` 返回**相对项目根的真实磁盘位置**；URL 由后端 `personnel_thumb_url()` 翻译并随 `thumb_url` 字段下发；前端只消费 `thumb_url`（保留按文件名兜底） |
| 2 | **`--purge-images` 一张图都删不掉**，且只打印"图片 0 张"不报错 | `cmd_clean` 做 `ROOT / thumb_path` → `ROOT/personnel_crops/x.jpg`，而文件实际在 `ROOT/outputs/personnel_crops/` | 同上，存储值补全 `outputs/` 层级 |
| 3 | 给**示范/临时库** seed 后，生产 `outputs/personnel_crops/` 里多出**无人引用的孤儿 jpg**（实测 12 张，`personnel` 表 0 行） | `thumb_dir` 默认写死 `CROPS_DIR`，**不看目标库** —— `--db /tmp/demo.db` 也照样往生产目录写图；该库被 `--clean` 删掉后，图就永久变成孤儿 | 默认目录跟着目标库走：非默认库时用 `<目标库同级>/personnel_crops/`；`server.py` 的 mount 目录加 `LAB_MONITOR_PERSONNEL_CROPS` 环境变量可覆盖，让为演示库起的服务也能取到图 |

附带修掉 `--thumb-dir` 的隐患：旧实现无论 `out_dir` 指到哪都硬拼 `personnel_crops/` 前缀，
自定义目录时会存下一个**不描述文件真实位置**的路径（表现为"图生成了但清理/展示都找不到"）。
现在 `relative_thumb_path()` 对项目外目录退化为绝对路径，保证 `--purge-images` 仍能删到。

**教训（值得写进 review checklist）**：凡是"一个字符串被存下来、由多处各自解释"的字段（路径、URL、特征空间标识都属此类），
必须指定**唯一翻译点**；否则拼法一旦不一致，失败方式必然是**静默降级**而不是报错。
第 3 条是同一教训的另一面：**"数据写到哪"必须由"数据属于谁"决定**，不能让默认值越过作用域
（默认库→生产目录、其它库→库旁边）。三处都补了回归用例，见
`tests/test_mock_personnel.py::ThumbPathContractTest` 与 `::ThumbDirIsolationTest`。
`personnel_thumb_url()` 只取文件名（头像目录扁平、按 `person_id` 命名），顺带天然杜绝 `../../` 路径穿越。

---

## 6. 检索能力（后端已有 + 3 个真缺口）

| 需求 | 现状 | 动作 |
|---|---|---|
| 按姓名/工号找人 | `GET /api/personnel` 返回全量、无过滤无分页（`db.py:592-607`） | **A. 加 `q`/`department`/`limit`/`offset`**（无参时行为完全不变，向后兼容） |
| 点某人 → 他出现在哪些视频 | `/api/search/person` **只吃 `global_id`** | **B. 新增 `person_id` 入口** → `src/search.py: aggregate_person_assets()`，跨该人名下所有 gid 合并 `segments` 后按 `video_last_ts` 重排 |
| 结果里那 3.2s~8.7s 能点开吗 | 算出来了但不可点（`modals.js:324-325`） | **C. `GET /media/{camera_id}?asset_id=`** 用 `FileResponse` 出片（实测 starlette 1.6 支持 Range，浏览器 `<video>` 可直接 seek）。**必须做路径校验**：`rel_path` 只能来自 `video_assets` 表、解析后必须落在 ROOT/`videos_low` 或 ROOT/`videos` 内，防穿越 |
| 时间范围/相机过滤 | 后端 `server.py:1002-1004` 已支持，前端一个没暴露 | 详情页加时间窗预设（今天/7 天/30 天）+ 相机下拉，纯前端 |
| 以图搜人 | `POST /api/search/by-image`（`server.py:1041-1106`）就绪 | 前端上传控件；Tier A 数据下才有意义 |

**以图搜人的三个实测注意点**（`api.js:33-70` 已核）：
- `fetchJson` **不强制 `Content-Type`**（它把 `rawOptions` 直接并进 `fetch`），所以 `body: new FormData()` 可以正常走 —— 但**绝不要手写** `Content-Type: multipart/form-data`，那样会丢掉浏览器自动生成的 boundary。
- 默认 `timeoutMs = 5000`，而上传后要跑 YOLO 检测 + ReID 提特征，CPU 模式轻松超 5 s → 必须显式传 `{ timeoutMs: 20000 }`。
- `fetchJson` **返回的是已解析对象**，不是 `Response`。`modals.js:276-277` 连续两处踩空：`await res.json()` 抛 TypeError（`res` 是普通对象），下一行的 `res.status === 200` 也永远不成立 —— 所以现在**「命名并绑定」按钮即使后端成功建了档案，前端也会显示「绑定失败: res.json is not a function」**。这是一条独立的既有 bug，批次五实施时顺手修（直接用 `res.name` / `res.person_id`，并删掉冗余的守卫头 —— `fetchJson` 会自动加）。

### 6.1 一个必须提前定死的坑

`search.py:46` 的资产排序键是 `(bool(frames_real), asset_id)`，`_resolve_asset()` `:51-56` 在 `asset_id` 为 NULL 时**取该相机第一个候选资产**。现在每路相机有 2 行资产（`videos/` 原始 + `videos_low/` 低清，实测 44 行/22 相机）。
→ seed 时**必须显式写 `asset_id`**（不要留 NULL 走 fallback），且要和 `/media/{cam}` 实际出片的那一行对上，否则前端显示的 `video_ts` 会定位到另一个文件上 —— 时长差 0.5 s 看似无关，但 239.6 s vs 239.12 s 的循环素材里就是一次错帧。

---

## 6.2 P1-1 检索闭环（已实施并验收）

闭环 = **按人搜 → 看到片段 → 点开定位**。三块都已落地：

| 块 | 落点 | 关键设计 |
|---|---|---|
| 入口 | `GET /api/search/person?person_id=` | 与 `global_id=` **二选一**（都给/都不给 → 400）。档案名下无身份 → 返回空清单而非 404（不是错误，是新档案还没绑定 gid） |
| 聚合 | `src/search.py:aggregate_person_assets()` | **先按身份各自聚合，再合并结果**，不在 SQL 层 `IN (...)` 一次查完，理由见下 |
| 出片 | `GET /media/{cam}?asset_id=` | 支持 Range；默认给 `videos_low/`；`Content-Length` 走 `stat()` 不读 `size_bytes` |
| 前端 | `personnel.js` 详情页片段列表 + 内联 `<video>` | 点「播放」定位到 `video_first_ts`；越界显式夹到末尾并提示 |

**为什么不能 SQL 一次查完再折叠循环。** 一个人常被拆成多个匿名身份，而**每个身份都在走同一条相机拓扑**（都是走廊 A→B→C）。把多个身份的相机序列首尾拼成一条长链再交给 `collapse_loop_segments()`，它天然呈现"严格周期"，`detect_loop_period()` 会判成循环播放，把**真正不同的多次到访折成一次** —— 用户看到"这人只走过一趟"。按身份分别聚合则各自折叠，合并后段数相加。代价是每人 N 次查询（N = 名下身份数，通常 1~3）。回归用例：`tests/test_search.py::AggregatePersonTests::test_merges_without_fake_loop`（断言每个相机保留 **2** 段；拼后折叠会变成 1 段）。

**/media 的两个实测约束**（都写进 docstring 了，改之前先看）：
1. `videos_low` 的 22 行 `size_bytes` **全是 NULL**（转码是后续批量做的，没回填）→ `Content-Length` 必须 `stat()`，读库会算错长度、视频卡在第一帧。
2. 原片与低清片时长不同（实测 239.12 vs 239.64），而 `identity_appearances.video_ts` 是按**低清片**标定的 → 前端播放 URL 必须带 `asset_id`，不能只给相机让后端自己挑，否则 seek 会错帧。

**实施期发现并修掉的两个真实 bug**：

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | **接口失败被说成"确实没有轨迹"**：`/api/personnel/{id}/activity` 挂掉时，详情页照旧显示"该档案名下身份没有轨迹记录。" | 详情页只区分了"有没有 `per_camera` 行"，没区分 `activity == null`（请求失败）与"成功了但为空"。前者是故障、后者是事实，混在一起用户会以为系统查过了 | 与片段列表统一口径：失败说"活动量读取失败（可重试）"，为空才说"没有轨迹记录"；统计卡片失败时显示 `—` 而不是误导性的 `0` |
| 2 | **每次检索都在事件循环上同步查库**（功能全绿，只在并发下拖慢所有接口） | 为复用参数把 `known_cameras` 提到协程顶层的 `common = dict(...)` 里，而 `_known_camera_set()` 内部要 `list_video_assets()` 查库 | `known_cameras` 留在 `run_in_threadpool` 的闭包内。回归用例见下 |

**伪测试的教训（值得单独记）**：bug 2 的第一版回归用例按"调用线程名 ≠ 主线程"断言 —— **它在有 bug 的代码上照样通过**（`TestClient` 自己就在非主线程跑 loop，断言恒真）。换成"调用时该线程有没有 `asyncio.get_running_loop()`"才真正区分：threadpool worker 没有 running loop，协程里必然有。已按"先确认它在 bug 版上失败、再确认在修好的版本上通过"两步验证过（`tests/test_search_api.py::test_known_camera_lookup_runs_off_the_event_loop`）。**任何回归用例都必须先在 bug 上验红。**

---

## 7. 风险与不做清单

| 风险 | 处置 |
|---|---|
| seed 脚本误写生产库 | 自动备份 + `--dry-run` + manifest 精确回滚；脚本只连 `Database(path)`，不 `import src.db`（避开 `db.py:1084` 单例） |
| 模拟数据与真实运行数据混在同一张表，之后无法分辨 | `personnel.source` + `note='模拟数据 · 批次五'` + `SIM` 徽标三重标记；`--clean` 一条命令回滚 |
| 30 天保留策略清掉演示数据 | 时间戳限制在 7 天内 + `--refresh-timestamps` 平移 |
| 合成特征让人误判 ReID 已可用 | Tier B 强制 `SIM` 徽标；报告里打印真实组内/组间余弦分布，与 `scripts/probe_reid_separability.py` 同口径 |
| 新增 22 路视频出片带宽 | `/media` 默认指向 `videos_low/`（29 MB 全量 vs 983 MB），并限制并发不新增 |
| 头像目录膨胀 | `outputs/personnel_crops/` 按 person_id 覆盖写；`--clean --purge-images` 按 manifest 记录精确删（不是 glob 猜目录） |
| 给临时/演示库 seed 时把头像写进**生产**目录（留下孤儿图） | 默认头像目录**跟着目标库走**（非默认库 → `<库同级>/personnel_crops/`）；服务侧 `LAB_MONITOR_PERSONNEL_CROPS` 可覆盖 mount 目录。→ 该风险曾在实施中真实发生，详见 §5.4 第 3 条 |

**明确不做**：不引入 faker/echarts/前端框架；不给 `person_id` 换格式；不改 `apply_retention` 语义；不做真正的数据看板大屏（本批次目标是"能看能搜"）；不在 `personnel` 上加 `FOREIGN KEY`（全库 0 个 REFERENCES 是既有约定，单独立项）。

---

## 8. 实施顺序（每步可独立验收）

1. **P0-1 schema + 生成器（纯后端，半天）**
   `db.py` 加 2 列 → `src/mock_personnel.py` → `scripts/seed_personnel_mock.py --mode synthetic`
   验收：`--dry-run` 输出统计；正式跑完 `personnel>0`、`identities` 有 `person_id`、重启后 `GET /api/search/person?global_id=<seed出的gid>` **不再 404**。
2. **P0-2 API 扩展**：`GET /api/personnel?q=&department=&limit=&offset=` + `Database.person_activity_stats()`（分相机出现次数/首末时间/停留段数）
   验收：curl 三组过滤组合，无参响应与改前逐字段一致。
3. **P0-3 前端档案列表**：`personnel.js` + `#personnel-modal` + `personnel.css`（含 light 覆盖）+ 8 处 `?v=` bump（✅ 已做，当前 **12.0**）
   验收：header 按钮打开列表；输入「张」出 3 人；切亮色主题无暗块；`__labStreamStats()` 仍 ≤5。
4. **P1-1 检索闭环**：`person_id` 入口 + 详情页视频片段列表 + `/media/{cam}` seek
   验收：点某人的片段 → 视频跳到 `video_first_ts` 秒；`position_known=True`。
5. **P1-2 Tier A 真人数据**：离线跑 detector+ReID 出真实特征与抓拍图，替换 demo 集
   验收：底库 1:N 自动命名在实时链路上真的产生「姓名 (工号)」标签。
6. **P1-3 中文标签**：`src/text_overlay.py` + PIL 补丁缓存
   验收：画面上中文名可读；逐路 fps 相比改前降幅 <5%。
7. **P2**：以图搜人上传控件、实时卡片的「在场人员」姓名覆盖层、`/api/personnel` 拼音/首字母检索。

**测试**：`tests/test_mock_personnel.py`（unittest，`Database(tmp_path)` + 断言 seed 行数/特征可被 `_restore()` 认/`--clean` 归零）；`tests/test_personnel_frontend.mjs`（Node DOM 桩件，19 项）；扩 `tests/test_search.py` 加 `person_id` 聚合用例；`node tests/test_stream_manager_frontend.mjs` 回归。
⚠ 跑 Python 测试前先备份 `outputs/lab_monitor.db`（AGENTS.md 的 `import src.db` 陷阱）。

### 实施结果（2026-09-12 已完成并验收）

| 步骤 | 产物 | 验收证据 |
|---|---|---|
| P0-1 schema | `src/db.py` 迁移字典加 `source` / `thumb_path` | 全新库 `PRAGMA table_info(personnel)` 含两列；老库自动 `ALTER TABLE` |
| P0-1 生成器 | `src/mock_personnel.py` | 40 人 / 80 身份 / 4996 行轨迹 / 22 路相机，0.1s 生成；姓名 500 次抽样零重复 |
| P0-1 CLI | `scripts/seed_personnel_mock.py` | `--dry-run` 可预览；`--clean` 精确回滚到基线；`--mode real` 主动拒绝（退出码 3）；头像目录跟着目标库走（§5.4 第 3 条） |
| P0-1 端到端 | — | seed → `IdentityStore._restore()` **全部认出**（0 未识别）→ `/api/search/person` 不再 404 → `position_known_rows == returned_rows`（视频内坐标齐全，无孤儿相机） |
| P0-2 检索 | `Database.search_personnel()` + `person_activity_stats()` | `q`/`department`/`source`/分页三组组合验证；LIKE 元字符已转义；分页无重复行；PATCH 不会冲掉 `source` |
| P0-3 前端 | `static/js/modules/personnel.js` + `#personnel-modal` + `personnel.css` | Node 桩件 19 项全通过（列表/检索防抖/分页/详情/两步删除/XSS 转义/头像 URL）；`.js` 语法检查通过；`?v=` 全量 bump 到 11.8（`index.html` + `main.css` 8 条 `@import`） |
| P0-3 头像 | `relative_thumb_path()` + `personnel_thumb_url()` + `#card-stat-persons` | 真 HTTP 实测 5 张头像全 **200**（改前必 404）；`thumb_url` 由后端下发，路径穿越输入被压回文件名 |
| **P1-1 闭环** | `aggregate_person_assets()` + `?person_id=` + `/media/{cam}` + 详情页片段列表/播放器 | 见 §6.2。真 HTTP 端到端：合并 2 个身份 → `video_first_ts=12.5 / video_last_ts=30.25` → `/media` 出片 200、Range 206（首片/开区间/后缀）、越界 416、穿越 404、asset 与 camera 不符 409 |
| 回归 | — | Python **281 / 281**、Node **28 + 19** 全通过；生产库跑前跑后均为 `personnel=0 / identities=6 / appearances=1718 / assets=44`（未动） |

已知待办（本批次**刻意不做**）：
1. **中文名烧不进画面**（§5.3）——`cv2.putText` 无中文字形，绑上中文名会显示 `????`。这是被空表掩盖的既有 bug，模拟数据会立刻暴露它。
2. **P1-2 Tier A 真人数据**：从 `videos_low` 抽真人特征与抓拍图（`--mode real` 现在会主动拒绝，等这一步落地才放开）。
3. 详情页片段列表未做时间窗/相机过滤（后端 `/api/search/person` 已支持 `start`/`end`/`camera`，前端还没暴露）。

`/media` 已于 P1-1 实现（原待办 2 已完成），带回环 Range、低清优先与穿越防护。

---

## 9. 一句话总结

**能不能做？能。** 后端 CRUD 与检索已就位，缺的正是"有数据可看的前端"：
先用 `--mode synthetic` 秒级造 40 人把 **列表→详情→按人检索视频→可 seek 回放** 全链路打通并验收；
再用 `--mode real` 从 `videos_low` 抽真人特征与抓拍图，把演示升级为可信结果。
同期必须修掉中文名 `cv2.putText` 这个被空表掩盖的老 bug，并在 UI 上给模拟数据打 `SIM` 徽标 —— 否则这套系统会在某次汇报里，把随机数说成识别能力。
