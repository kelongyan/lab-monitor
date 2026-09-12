# 三项能力实施工作清单

> 2026-09-12 · 依据已批准的 `docs/PLAN_2026-09-12_identity_search_resolution.md`
> 标注：**S** 小（单文件局部改动）· **M** 中（跨 2~4 文件，含测试）· **L** 大（新模块 + API + 前端）
> 勾选状态随实施推进更新。**批次一全部是阻塞项，未完成前批次三及之后不应开工。**

---

## 批次一 · 修特征层（阻塞项，5 项）

判定依据：实测 990 对异人特征余弦中位 0.984、99.8% 越过阈值 0.75；45 行身份只有 25 个唯一 `feature_blob`。**特征层不修，后面所有工作都是错的。**

- [ ] **1.1 换 ReID 度量学习权重** —— **S**
  - 位置：`src/reid.py:77-131`（`ReIDExtractorOSNet`）
  - 动作：`build_model(name="osnet_x0_25", num_classes=1000, pretrained=True)` 改为显式加载 ReID 权重（torchreid 支持 `model_path=`），用 `osnet_x0_25_msmt17.pth`（约 3MB）；`feature_space` 字符串改为 `osnet-x0.25-msmt17:512`
  - 连带：`IdentityStore._restore()`（`src/identity_store.py:148`）按 `feature_space` 全等过滤 → 旧身份会被静默跳过。**必须在启动日志里显式打"跳过 N 个旧特征空间身份"**，否则运维会误判为数据丢失
  - 验收：启动后 `/api/metrics/reid` 的 `gallery_size` 归零且不报错；日志有跳过计数；`feature_space` 端到端一致

- [ ] **1.2 收拢散落 5 处的匹配阈值** —— **S**
  - 现状：`0.75` 散落在 `reid.py:171`、`reid.py:208`、`reid_validator.py:31`、`identity_store.py:371`、`pipeline.py:138` 与 `:238`（共 6 处调用点，改一处无效）
  - 动作：新建 `src/reid_config.py`，定义 `REID_MATCH_THRESHOLD` / `REID_RATIO_TEST`，支持 `LAB_MONITOR_REID_THRESHOLD` / `LAB_MONITOR_REID_RATIO` 环境变量覆盖（沿用 `main.py:_env_positive` 的"非法即回落 + 打 warning"口径）；全部调用点改为引用；启动时打印生效值
  - 验收：`grep -rn "0\.75\|0\.85" src/` 只在 `reid_config.py` 命中；启动日志可见生效阈值

- [ ] **1.3 建人工标注对 + 阈值标定（本批的验收基线）** —— **M**
  - 新建 `config/labeled/pairs.json`：正样本对（同人跨相机）、负样本对（异人）。可半自动挑候选：同一 global_id 跨相机的两帧 = 候选正；不同 global_id 同相机同时刻 = 候选负
  - 新建 `scripts/build_label_set.py`（挑候选 + 导出缩略图供人工确认）
  - 新建 `scripts/tune_reid_threshold.py`（扫 threshold × ratio，输出 P/R/F1 与 ROC，落 `outputs/reports/reid_threshold_sweep.csv`）
  - 验收：报告给出推荐工作点，且满足 **p95(异人余弦) < 选定阈值 < p5(同人余弦)**；这一步不做，任何"识别准确率"都无法证伪

- [ ] **1.4 排查 `feature_blob` 重复写入** —— **M**
  - 现状：45 行身份仅 25 个唯一 md5，7 组重复共 27 个身份，最严重一组 7 个身份共享同一份特征。已排除 `register()`/`register_if_new()`（锁内独立 uuid + `feature_bank` copy）与 `save_identity` 的 SQL 列序
  - 排查方向：`identity_store.py:490-505` 的**锁外写库分支**，以及 `_persist_full()` 兜底路径（`:507`）是否用了过期引用
  - 加护栏：`register_if_new()` 返回前，若新身份主特征与 gallery 中任一主特征 cosine ≥ 0.999 → 打 `ERROR` 日志并附两个 gid（不静默通过）
  - 新建 `tests/test_identity_blob_unique.py`：并发注册 N 个互异特征，断言 `feature_blob` md5 唯一数 == N
  - 验收：单测通过；重跑语料后 `scripts/diagnose_reid_degeneracy.py` 的唯一 md5 数 == 身份数

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
| 一 · 修特征层 | 5 | 0 | **阻塞批次三 / 四 / 五** |
| 二 · 数据基础 | 6 | 0 | 2.1~2.3 阻塞批次四 |
| 三 · 能力一 身份识别 | 6 | 0 | 依赖批次一 |
| 四 · 能力二 视频检索 | 6 | 0 | 依赖批次一 + 2.1~2.3 |
| 五 · 能力一 路径 B | 3 | 0 | 依赖批次三 |
| 并行清理 | 8 | 0 | 无 |
