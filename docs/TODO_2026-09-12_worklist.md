# 三项能力实施工作清单

> 2026-09-12 · 依据已批准的 `docs/PLAN_2026-09-12_identity_search_resolution.md`
> 标注：**S** 小（单文件局部改动）· **M** 中（跨 2~4 文件，含测试）· **L** 大（新模块 + API + 前端）
> 勾选状态随实施推进更新。**批次一全部是阻塞项，未完成前批次三及之后不应开工。**

---

## 批次一 · 修特征层（阻塞项，7 项）

判定依据：实测 990 对异人特征余弦中位 0.984、99.8% 越过阈值 0.75。**特征层不修，后面所有工作都是错的。**

> ⚠️ **2026-09-12 实施中的重大修正**：开工后实测发现，根因**不止是权重**。
> 库内那批塌缩特征不是模型吐出来的原始特征（本机实测原始特征均值范数 0.796 / 余弦 p50 0.629，
> 并未塌缩），而是被 `update_appearance()` 的 EMA 滑动平均**加工后**的结果。
> 因此 1.1（换权重）是**必要但不充分**的，真正的主因是新增的 **1.6（聚合方式）**。
> 详见 1.4 与 1.6 两条。

- [x] **1.1 换 ReID 度量学习权重** —— **S**（2026-09-12 完成）
  - 已做：新增 `src/reid_config.py` 作为权重注册表（数据集 → 文件名 → `feature_space` → 分类头维度 → drive id）；
    `ReIDExtractorOSNet` 改为从本地 `.pth` 显式加载（`pretrained=False` + `load_state_dict`），
    并用**分类头维度**校验数据集一致性（751=market1501 / 1041=msmt17 / 702=dukemtmc / 1000=imagenet）
    —— 这是唯一能证明"这份权重训在哪个数据集上"的证据，参数量与张量数三项太接近、区分不了。
  - 已做：新增 `scripts/fetch_reid_weights.py`（走 Clash 代理下载 torchreid 官方权重，
    幂等 + 校验），`market1501`（Rank-1 91.2）与 `msmt17`（61.4 域内）两份已就位。
    默认 `LAB_MONITOR_REID_WEIGHTS=msmt17`，可切 market1501；最终选谁由 1.3 的标注集实测决定。
  - 已做：降级链改为**显式**：ReID 权重 → ImageNet OSNet（打 ERROR）→ ResNet50（打 ERROR），
    不再静默；`LAB_MONITOR_ALLOW_IMAGENET_FALLBACK=0` 可切严格模式。
  - ⚠️ **但本项单独不足以修复**：本机实测三种权重的原始特征质量差异很小
    （见 `scripts/compare_reid_weights.py`），库内塌缩的主因是聚合方式，见 1.6。

- [x] **1.2 收拢散落 6 处的匹配阈值** —— **S**（2026-09-12 完成）
  - `src/reid_config.py` 提供 `REID_MATCH_THRESHOLD` / `REID_RATIO_TEST`，
    支持 `LAB_MONITOR_REID_THRESHOLD` / `LAB_MONITOR_REID_RATIO` 覆盖（取值须在 (0,1) 开区间，
    非法只回落默认值并打 warning）。
  - 6 个调用点全部改为引用：`reid.py:171/208`、`reid_validator.py:31`、
    `identity_store.py:371`、`pipeline.py:138/238`；`demo.py` 与两个诊断脚本也一并对齐。
  - 验收已达成：`grep -rn "0\.75\|0\.85" src/` 只剩 `reid_config.py` 的常量定义与注释
    （另有 `identity_store.py` 的 `base_alpha=0.85`，是滑动平均系数，与阈值无关）。

- [ ] **1.3 建人工标注对 + 阈值标定（本批的验收基线）** —— **M**
  - 新建 `config/labeled/pairs.json`：正样本对（同人跨相机）、负样本对（异人）。可半自动挑候选：同一 global_id 跨相机的两帧 = 候选正；不同 global_id 同相机同时刻 = 候选负
  - 新建 `scripts/build_label_set.py`（挑候选 + 导出缩略图供人工确认）
  - 新建 `scripts/tune_reid_threshold.py`（扫 threshold × ratio，输出 P/R/F1 与 ROC，落 `outputs/reports/reid_threshold_sweep.csv`）
  - 验收：报告给出推荐工作点，且满足 **p95(异人余弦) < 选定阈值 < p5(同人余弦)**；这一步不做，任何"识别准确率"都无法证伪

- [x] **1.4 澄清 `feature_blob` 重复写入的性质 + 加护栏** —— **M**（2026-09-12 完成）
  - **结论修正**：原判 "写库缺陷" 不成立。加查后的证据：27 个重复身份分属 7 组，
    每组的 `total_appearances` 完全相同（230 / 236）、`last_camera` 全是 `rnd_19`、
    创建时刻彼此相差约 33 分钟、特征**字节完全相同**。这是**同一段循环素材被反复
    注册**留下的指纹，不是同一份数组被写进多行。已排除的路径：`register()` 在生产
    路径中从不被调用（仅 demo 与测试用）、`register_if_new()` 锁内生成独立 uuid、
    `save_identity` SQL 列序、`_persist_full()` 兜底。
  - 已完成：`register_if_new()` 的**特征塌缩护栏** —— 查询与已有身份相似度 ≥ 0.999
    却被判歧义时打 `ERROR`（60s 节流）并累加 `collapse_warnings`，该计数外露到
    `/api/metrics/reid` 与 `/api/system/metrics`。此前这个失败模式完全不可观测。
  - 已完成：`tests/test_reid_identity_integrity.py`（9 例）覆盖并发注册 blob 唯一性、
    按身份去重语义、feature_bank 救回漂移主特征、护栏触发与指标同构。
  - 遗留：精确的创建时点需当次运行日志才能定论，不在本批范围内。

- [ ] **1.6 修特征聚合方式（EMA 抹平身份特异残差）** —— **L** ｜**新增，是本批最关键的一项**
  - 证据：`scripts/diagnose_ema_collapse.py` —— 用**同一人**的时间连续样本跑真实更新式，
    跨身份余弦 p50 随更新次数 0→5→20→50→200 变化为 **0.496 → 0.653 → 0.733 → 0.728 → 0.734**，
    越过阈值 0.75 的异人对比例从 **21.4% → 46.4%**（饱和）。若把不同人混进同一身份
    （匹配失败后的实际情形），p50 直接冲到 **0.95+**，与库内实测的 0.984 吻合。
  - 机理：`update_appearance()` 的 `feature = alpha*feature + (1-alpha)*feat` + 重新归一化
    是个带持续再注入的递归低通滤波 —— 公共分量每轮被新样本补回，**身份特异残差**
    每轮只保留 `alpha` 倍且无来源补充，按 `alpha^k` 指数衰减（alpha=0.85 时 0.85^50 ≈ 3e-4）。
  - 已做的一半：`match_feature_detailed()` 改为**按身份去重**后再做阈值/Ratio 判定，
    主匹配路径（`pipeline`）改用 `get_match_gallery()` 展开 `feature_bank`
    （bank 存的是原始特征且带多样性约束，天然不受 EMA 塌缩影响）。
  - 待做：决定主特征本身的取舍 —— 候选方案（需在 1.3 的标注集上比）：
    ① 主特征改为**有界窗口均值**（如最近 20 个原始特征，`deque(maxlen=20)`）替代无界 EMA；
    ② 调低 `base_alpha`（0.85 → 0.5）缩短记忆；
    ③ 干脆把 `feature_bank` 当唯一匹配依据，主特征只用于展示。
  - 验收：库内任意两身份的余弦 p95 < 匹配阈值；`collapse_warnings` 长时间保持 0。

- [ ] **1.7 顺带修掉的既有缺陷（本批发现）** —— **S**（已完成）
  - `Database.close()` 原先只关**当前线程**的连接，22 个 pipeline 线程各自的连接会存活到
    进程退出 → Windows 上库文件被占用（临时目录删不掉、备份后无法 rename，实测触发
    `PermissionError: [WinError 32]`）。现改为登记全部线程连接并统一回收，
    连接以 `check_same_thread=False` 建立（每条连接仍只被创建它的线程使用）。
  - `tests/test_stream_manager_frontend.mjs` 的轮转周期断言停留在 `12000ms`，
    而 `1a5b17b` 已把模块常量改成 `6000ms` —— 该用例自那次提交起**一直失败**却没人发现。
    已改为镜像常量 `EXPECTED_ROTATE_MS` / `EXPECTED_FILL_MS` 并注明需与模块同步。

- [ ] **1.5 全量重跑语料验证** —— **S**（约 30 分钟机时）
  - 动作：清空/备份 `outputs/lab_monitor.db` → 跑 `main.py` → 等 22 路走完一轮（67576 帧 ÷ 38 帧/s ≈ 30 分钟）
  - 验收：重跑 `probe_reid_separability.py`，组间余弦 p95 < 阈值、组内 p5 > 阈值；`alerts` 表里 MISSING_PERSON 的数量级明显下降（当前 7066 条含大量误报）

---

## 批次二 · 数据基础（6 项）

- [ ] **2.1 `video_assets` 表** —— **M**
  - 建表（含 `sha1_8` / `frames_real` / `duration_real` / `loop_detected` / `low_value`），`_init_db` 的 `migrations` 字典补列
  - 22 路的 `frames_real` / `duration_real` **直接落库**（`probe_media_frames.py` 已跑完，见方案 §2），不要重跑
  - 新建 `tests/test_video_assets.py`
  - 验收：`GET /api/assets` 返回 22 行，`duration_real` 合计约 45 分钟

- [ ] **2.2 轨迹表补「视频内时间」三列** —— **M**
  - `identity_appearances` 加 `asset_id` / `video_frame` / `video_ts`，加索引 `(global_id, camera_id, timestamp)`
  - 理由：素材循环播放，墙钟 `timestamp` 无法反查源视频位置，检索结果就没法"跳转到那一段"
  - 老数据这三列为 NULL → 检索返回 `position_known: false`，前端降级为只显示相机 + 墙钟时间，**不做无根据的回填猜测**

- [ ] **2.3 采集侧写入视频内坐标** —— **S**
  - 位置：`src/pipeline.py:_read_loop` 与 `_process_frame` → `IdentityStore.update_appearance`
  - 动作：把 `self._frame_idx`（或 `cap.get(CAP_PROP_POS_FRAMES)`）与 `asset_id` 一路透传到 `db.record_appearance` / `save_identity`
  - 注意：`update_appearance` 的轻量路径（只插增量行）与 `_persist_full` 兜底路径**都要带上**，漏一条就会出现半截数据

- [ ] **2.4 运行时缩放兜底 `process_max_width`** —— **S**
  - `CameraPipeline.__init__` 增参数（默认 960），`_read_loop` 里 `cap.read()` 后按需 `cv2.resize`
  - 用途：RTSP 在线流无法离线转码，只能走运行时缩放。**这条无论本次做不做都必须建**，是能力三唯一确定的价值
  - 注意：缩放要在 `push_frame` 之前，否则 MJPEG 与告警截图拿到的是原始大帧

- [ ] **2.5 `scripts/transcode_lowres.py` + A/B 实验** —— **M**（优先级可延后）
  - 参数 `--scale 960:540` / `--crf 28` / `--only reg_01` / `--dry-run`；按 `sha1_8` 幂等跳过
  - A/B 抽 3 路：`reg_01`（1080p 常规）、`rnd_21`（1440p 随机）、`rnd_16`（唯一含 ROI）。原片 vs 低分片各跑 300 秒，比人数召回 / 注册身份数 / 每路 fps，落 `outputs/reports/resolution_ab.csv`
  - 定档规则：小目标召回下降 > 5% 就退到 `1280x720`；≤ 2% 可再试 `854x480`

- [ ] **2.6 `rnd_05` 标记 `low_value`** —— **S**
  - 真实内容仅 1.3 秒 / 460 帧（fps 元数据报 351.56，属损坏读数）。索引里打标记，**不删除**（采集点不能随意摘除），但 A/B 与检索结果默认排除

---

## 批次三 · 能力一：人员身份识别（6 项）

前置认知：当前 `global_id = uuid4()[:8]` 是**随机编号，不含实名信息**。"识别出具体是谁"必须靠 L2 实名层。

- [ ] **3.1 `personnel` / `personnel_photos` 表 + `identities` 补列** —— **M**
  - 新表 `personnel(person_id, name, employee_no, department, note, created_at, updated_at)`
  - 新表 `personnel_photos(photo_id, person_id, feature_dim, feature_blob, quality, source_path, created_at)`
  - `identities` 加 `person_id` / `name_confidence`

- [ ] **3.2 `IdentityStore.bind_person()` + 采集侧带出实名** —— **S**
  - 绑定后 `pipeline` 写 appearance 时一并带上 `person_id` / `name`，供前端标签与检索复用

- [ ] **3.3 `PersonnelGallery`（路径 B 的底库检索）** —— **M**
  - 新建 `src/personnel.py`，**复用 `match_feature_detailed()`**，不要另写一套相似度逻辑
  - 插入位置：`pipeline._process_frame` 中**在**现有 `get_confirmed_match()` **之前**——实名信息价值高于匿名编号，且能抑制匿名身份膨胀
  - 阈值独立于实时匹配阈值（用 1.2 的 `reid_config`，但可配不同值）

- [ ] **3.4 API 层** —— **M**
  - `GET/POST /api/personnel`、`PATCH/DELETE /api/personnel/{pid}`
  - `POST /api/personnel/{pid}/photos`（multipart → 提特征入库）
  - `POST /api/identities/{gid}/bind`（路径 A 的核心）
  - `GET /api/personnel/{pid}/identities`
  - 扩展 `GET /api/identities` 返回 `person_id` / `name`
  - 全部写接口带 `X-Lab-Monitor-Request: 1`

- [ ] **3.5 前端命名入口** —— **M**
  - 人员档案弹窗加姓名/工号编辑与绑定按钮；视频卡片标签由 `ID: #c0ed5fd1` 变为 `张三 (P0007)`
  - 改完 bump `static/index.html` 的 `?v=`（当前 CSS `11.5` / `app.js` `11.6`，`main.css` 的 7 条 `@import` 要一起改）

- [ ] **3.6 `tests/test_personnel.py`** —— **S**
  - 覆盖建档案、绑定、重启后绑定关系仍能恢复

---

## 批次四 · 能力二：人员视频检索（6 项）

- [ ] **4.1 抽出 `collapse_loop_segments()` 到公共模块** —— **M**
  - 把 `server.py:752-923` 的循环折叠逻辑（差分判据 `seq[i]==seq[i-period]`、覆盖校验、bbox 20px 指纹降级）抽到 `src/trajectory.py`，`/trajectory` 与检索共用
  - 硬约束：**不要在检索接口里重写一遍**，该逻辑有 `tests/test_trajectory.py` 的 19 个用例保护，重写必然倒退
  - 验收：`test_trajectory.py` 19 例全绿

- [ ] **4.2 `GET /api/search/person`** —— **M**
  - 纯 SQL 聚合，无新算法。按 `(asset_id, global_id)` 聚合 → `split_gap_s` 切停留段 → 关联 `video_assets`
  - **必须返回 `loop.loop_factor` 与 `unique_segments`**，默认展示折叠结果（语料仅 45 分钟却已累计 227,124 条 appearance，不标注倍数会得到"出现 3 万次"的度量假象）

- [ ] **4.3 过滤已下线相机** —— **S**
  - 已查实：库里存在 **`rnd_03` 的 6,447 条孤儿轨迹**，但该相机已不在 `sources.json`（历史配置残留）
  - 动作：检索与轨迹接口一律按 `sources.json` 过滤，或返回时标注"已下线"，避免前端点进去 404

- [ ] **4.4 `POST /api/search/by-image` + `src/search.py`** —— **L**
  - 上传图片 → OSNet 提特征 → 与 `identities`（`feature` + `feature_bank`）全库比对 → Top-K 排序（**非二值判定**，离线检索可放宽阈值）
  - 为每个命中段抽 1 帧存缩略图（seek 解码，可复用 `FrameHub.get_frame`）

- [ ] **4.5 `GET /api/assets` + 前端视频清单** —— **M**
  - 人员档案弹窗内加「该人出现过的视频」表格（相机 / 文件名 / 真实时长 / 出现段数 / 循环倍数 / 跳转）

- [ ] **4.6 `tests/test_search.py`** —— **S**

---

## 批次五 · 能力一 路径 B 收尾（3 项）

- [ ] **5.1 底库检索接入实时链路**（3.3 的落地）+ 自动命名优先级调优
- [ ] **5.2 前端注册照上传入口** —— 每人 1~N 张，含质量校验（复用 `quality = min(1, area/128×256) × conf` 口径）
- [ ] **5.3 标注集回归** —— 用 1.3 的同一套标注对验证路径 B 相对路径 A 的增益

---

## 并行清理项（不阻塞主线，可随时插入）

- [ ] **22 路里 9 路零身份数据，需诊断** —— 已查实无数据的 9 路：`reg_01`、`reg_02`、`rnd_01`、`rnd_02`、`rnd_05`、`rnd_12`、`rnd_17`、`rnd_21`、`rnd_22`
  - 两个假设：① 该路视频确实无人；② 有人但 `_validator` 攒不满 8 帧缓冲 → 永不注册身份（短素材如 `rnd_05` 1.3 秒必然如此，但 `reg_01`/`reg_02` 各有 239 秒，零数据很反常）
  - 做法：抽 `reg_01` 单跑，打开 `DEBUG` 看 `buffer_len` 是否始终 < 8。若是 ②，批次一的修复会顺带解决
- [ ] **数据分布极度倾斜，需确认是否有意为之** —— `rnd_08` 占 135,679 条（全库 59.7%），前 4 路占约 91%；尾部 `reg_10`/`rnd_10` 各仅 33 条
- [ ] **`camera_map.json` 的 `map_xy` 全为 null** —— 22 路点位一个都没标，是轨迹回放与平面图渲染的前置（你需用 `outputs/dev/calibrate_floorplan.html` 标点）
- [ ] **`roi.json` 只有 `rnd_16` 一条围栏** —— 其余 21 路永远不会报 INTRUSION
- [ ] **前端未消费的后端接口**：`/api/floorplan`、`/api/identities/{gid}/trajectory`、`/api/system/metrics` —— 后端已实现且有测试，前端零代码
- [ ] **INTRUSION 驻留状态机** —— 当前 `LAB_MONITOR_INTRUSION_COOLDOWN=120` 只是复报间隔，语义上"进围栏一次报一条"未实现
- [ ] **测试碰生产库** —— `src/db.py:646` 模块级 `db = Database()`，import 即连 `outputs/lab_monitor.db`
- [ ] **ReID 特征节流无监控** —— `identity_store.py:23-24` 的 50 次/30s 节流静默丢数据，建议在 `/api/system/metrics` 暴露待落盘计数

---

## 需要你决策/提供的 3 项

| # | 事项 | 我的建议 |
|---|---|---|
| 1 | **ReID 权重获取渠道** | `osnet_x0_25_msmt17.pth` 在 torchreid 官方 Google Drive 上（首次需联网/代理）。若下载不通，退而用 `osnet_x0_25_market1501.pth`，效果同量级。**不建议**继续用 ImageNet 权重 |
| 2 | **标注集工作量** | 建议规模：**正样本对 ≥ 150、负样本对 ≥ 300**。我可以先半自动挑好候选并导出缩略图，你只需确认/剔除，预计 20~30 分钟人工 |
| 3 | **是否现在就开批次一** | 建议**只做 1.1 + 1.2 + 1.4 三项**（换权重 + 收拢阈值 + 排查重复写入），然后再做 1.3 标定——因为标定必须在特征修好之后才有意义 |

---

## 进度看板

| 批次 | 项数 | 已完成 | 阻塞关系 |
|---|---|---|---|
| 一 · 修特征层 | 7 | 4（1.1 / 1.2 / 1.4 / 1.7） | **阻塞批次三 / 四 / 五**；剩 1.3 标定、1.5 重跑、**1.6 聚合方式（最关键）** |
| 二 · 数据基础 | 6 | 0 | 2.1~2.3 阻塞批次四 |
| 三 · 能力一 身份识别 | 6 | 0 | 依赖批次一 |
| 四 · 能力二 视频检索 | 6 | 0 | 依赖批次一 + 2.1~2.3 |
| 五 · 能力一 路径 B | 3 | 0 | 依赖批次三 |
| 并行清理 | 8 | 0 | 无 |

**下一步建议顺序**：1.6（改聚合方式）→ 1.3（标注集标定，选权重 + 定阈值）→ 1.5（全量重跑验证）。
1.6 排在最前是因为它是唯一能真正解除塌缩的一项：不改聚合方式，换任何权重、扫任何阈值，
测出来的都是"在一个已经失去判别力的特征空间里找最优切点"。
