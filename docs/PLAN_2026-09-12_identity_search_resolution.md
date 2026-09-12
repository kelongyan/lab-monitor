# 三项能力实施方案（身份识别 / 视频检索 / 分辨率降低）

> 2026-09-12 · 基于对 `main` @ `3a8df5f` 的全量代码通读 + 四项本机实测
> 范围：本次只做①人员身份识别 ②人员视频检索 ③分辨率降低。
> **明确不做**：基于 A→B→C 行程时间的串联超时预警（缺少摄像头连通关系与点位对应数据），仅作后续需求保留。

---

## 0. 先说结论

三项能力**不能直接开工**，有两个阻塞项必须先修。这不是估算，是实测结论：

| 实测项 | 结果 | 判定 |
|---|---|---|
| 45 个身份之间，不同人的特征余弦相似度 | mean **0.950**、中位 **0.984**、max 1.000；990 对异人里 **988 对（99.8%）** 越过阈值 0.75 | **特征已塌缩，当前无法区分人** |
| 45 行身份记录的 `feature_blob` 唯一数 | 只有 **25** 个唯一值；7 组重复共 27 个身份，最严重 **7 个身份共享同一份特征** | 塌缩的**指纹**（见 0.2，非写库缺陷） |
| ReID 权重来源 | 缓存里只有 `osnet_x0_25_imagenet.pth`，`src/reid.py:93-98` 用 `pretrained=True` 加载它 | 用的是 **ImageNet 分类权重**，且是**次要**原因 |
| 视频容器元数据 | ffprobe 与 OpenCV 一致报出 11948~71617 秒时长，`nb_frames` 缺失 | **元数据不可信**，检索定位不能依赖它 |

> ⚠️ **2026-09-12 实施期修正（重要）**：下文 0.1 与 0.2 的原始因果判断需要修正。
> 开工后实测发现：**同一份 ImageNet 权重直接提原始特征并不塌缩**
> （均值范数 0.796 / 余弦 p50 0.629，见 `scripts/compare_reid_weights.py`）；
> 库内那批 0.98 相似的特征是**被 `update_appearance()` 的 EMA 滑动平均加工过**的结果。
> 真正的主因是**特征聚合方式**，不是权重。详见 §0.1a 与 worklist 项 1.6。
> 原文判断保留在下方，以便对照——它记录了我们为什么会先怀疑权重。

**一句话**：身份识别这条链路的"特征层"目前是失效的，直接叠加功能只会得到一串看似成功、实则互相串号的身份。所以实施顺序必须是 **先修特征 → 再建身份 → 最后做检索**（降分辨率的优先级实测后被下调，见 §2「预期收益要说实话」）。

**修正后的因果链**（三门独立缺陷叠加，按影响力排序）：

```
① 特征聚合（EMA + 重新归一化）  →  抹平身份特异残差，余弦 p50 0.50 → 0.73（同一人样本）
                                    混入不同人后冲到 0.95+，与库内 0.984 吻合  ← 主因
② ImageNet 分类权重             →  原始嵌入公共分量偏大（均值范数 0.80、异人 p50 0.63），
                                    本身已不可用，但尚未塌缩                      ← 次要
③ feature_blob 重复 / 身份重复  →  是 ① 的指纹（同一段循环素材反复注册），非写库缺陷
```

---

## 1. 公共前置（三项能力的共同底座）

### 0.1 修正 ReID 特征退化【阻塞项】

**证据**（`scripts/probe_reid_separability.py`、`scripts/diagnose_reid_degeneracy.py`）：

- 组内相似度（同一身份 feature_bank 内部）：mean 0.794 / p5 0.579 / min 0.345
- 组间相似度（不同身份主特征）：mean 0.950 / p50 0.984 / p95 1.000
- **判别间隔为负**：组间均值（0.950）比组内均值（0.794）还高，任何阈值都无法分开
- 全体均值向量范数 **0.9754**（单位向量下越接近 1 越说明所有人共用一个主导分量）；去均值后每个身份的残差范数仅 mean **0.166** —— 身份信息几乎全被公共分量吞掉

**根因**：`src/reid.py:93-98` 的 `pretrained=True` 不加载 ReID 权重，而是 ImageNet 分类预训练权重（`feature_space` 字符串 `osnet-x0.25-imagenet:512` 即为自证）。分类任务的嵌入空间天然不保证"同人靠近、异人远离"。

**做法**：

1. 换用 ReID 度量学习权重 `osnet_x0_25_msmt17`（或 `market1501`），用 torchreid 的 `model_path=` 显式加载本地权重文件。
2. 同步把 `feature_space` 改成 `osnet-x0.25-msmt17:512`。这一步**自动完成旧数据隔离**：`IdentityStore._restore()`（`src/identity_store.py:148`）按 `feature_space` 全等过滤，旧身份会被静默跳过，无需手工清库；但要在启动日志里明确提示"跳过 N 个旧特征空间身份"，避免误判为数据丢失。
3. **阈值必须重新标定**。`0.75` 散落在 5 处（`reid.py:171`、`reid.py:208`、`reid_validator.py:31`、`identity_store.py:371`、`pipeline.py:138` 与 `:238`），换权重后要一起改。
4. 不需清空 `outputs/transit_stats.json`。通行时间统计的是**时间**，与特征维度/权重无关（该文件只在特征维度变化时才需重置，本次维度仍是 512）。这点与 `CLAUDE.md` 的表述不一致，已在代码注释中留痕。

**验收**：异人对余弦 p95 < 匹配阈值 < 同人对余弦 p5，两者之间留出可观测间隔。

> ⚠️ **实施期修正（2026-09-12）**：本项**必要但不充分**。换权重已实施（见 worklist 1.1），
> 权重来自 torchreid 官方模型库，用**分类头维度**校验数据集一致性
> （751=market1501 / 1041=msmt17 / 702=dukemtmc / 1000=imagenet）。
> 但本机实测（`scripts/compare_reid_weights.py`，同一批 199 个真实人体 crop）：

| 权重 | 均值向量范数 | 去均值残差范数 | 两两余弦 p50 | 越过阈值 0.75 的比例 |
|---|---|---|---|---|
| imagenet（原） | 0.7958 | 0.5983 | 0.6293 | 20.15% |
| market1501 | 0.8148 | 0.5719 | 0.6571 | 27.09% |
| **msmt17** | 0.7938 | 0.5971 | 0.6234 | 22.16% |

> 三者差异很小，且**都远好于库内那批 0.9754 / 0.984 的特征** —— 这直接说明库内塌缩不是
> 模型吐出来的原始特征固有的，而是流程加工出来的（见 §0.1a）。
> 因此"换权重"不能单独解决身份识别问题；选哪个权重也**必须**用 §1.3 的标注集实测，
> 不能在纸面上按公开指标拍定。本项保留为必做，但优先级让位于 §0.1a。

### 0.1a 修特征聚合方式（EMA 塌缩）【阻塞项 · 实施期新发现，优先级最高】

**证据**（`scripts/diagnose_ema_collapse.py`）：用**同一人**的时间连续样本（同相机、采样间隔
0.4s，按 12 帧切块）跑 `update_appearance()` 的真实更新式，跨身份余弦随更新次数变化：

| EMA 更新次数 | 0（原始） | 5 | 20 | 50 | 200 |
|---|---|---|---|---|---|
| 跨身份余弦 p50 | 0.496 | 0.653 | 0.733 | 0.728 | 0.734 |
| 异人对越过阈值 0.75 的比例 | 21.4% | 25.0% | 46.4% | 46.4% | 46.4% |

若把**不同人**混进同一身份（匹配失败后的实际情形），p50 直接冲到 0.95+，与库内实测 0.984 吻合。

**机理**：`feature = alpha*feature + (1-alpha)*feat` 后重新归一化，是一个**带持续再注入的递归
低通滤波**。公共分量每轮被新样本补回；而身份特异残差每轮只保留 `alpha` 倍、没有来源补充，
按 `alpha^k` 指数衰减（alpha=0.85 时 `0.85^50 ≈ 3e-4`）。因此观测次数越多的身份塌缩越彻底
——这解释了库里为什么会出现 `total_appearances = 137,439` 的"垃圾桶身份"。

**做法**：候选方案需在 §0.3 的标注集上对比后定稿 ——
1. 主特征改为**有界窗口均值**（`deque(maxlen=20)` 的原始特征均值）替代无界 EMA；
2. 调低 `base_alpha`（0.85 → 0.5）缩短记忆；
3. 匹配只依据 `feature_bank`（已存原始特征、带 <0.92 多样性约束），主特征降级为展示用途。

**已做的一半**：`match_feature_detailed()` 改为**按身份去重**后再做阈值与 Ratio 判定
（不去重时同一身份会同时占据 best 与 second，Ratio Test 必然误判为歧义）；
主匹配路径改用 `IdentityStore.get_match_gallery()`，把 `feature_bank` 一起展开参与 max 比较。
这两步已让 bank 里未被 EMA 加工过的原始特征进入判定，缓解但未根治。

**验收**：库内任意两身份的余弦 p95 < 匹配阈值；`/api/metrics/reid` 的 `collapse_warnings` 长期为 0。

### 0.2 `feature_blob` 重复的性质澄清（原判"写库缺陷"已排除）

**证据**：45 行身份只有 25 个唯一 `feature_blob`，典型重复组：

- 7 个身份共享同一份 blob：`c0ed5fd1 / bbd36f00 / ebe85b3a / 5127431d / 546f3b8f / 73981c3f / 987523f1`
- 7 个身份共享同一份 blob：`a3558536 / 76a0be92 / 681257fb / 5800de71 / ea374ead / 90868125 ...`
- 5 个身份共享同一份 blob：`8b4e741e / 06266ae0 / 935ff4a7 / 6eee70b5 / e646c22b`

md5 完全相同**不是**"同一个数组被写进多行"，而是**同一段循环素材被反复注册后，各自的 EMA
收敛到同一个不动点**。加查到的证据：

- 27 个重复身份分属 7 组，每组的 `total_appearances` **完全相同**（230 / 236）
- 每组的 `last_camera` 全是 `rnd_19`（素材仅 39 秒，循环播放）
- 同组身份的 `first_seen` 彼此相差约 **33 分钟**（如 21:28 → 22:09 → 22:48 ...）
- 每组的 `feature_bank_count` 都是 5（满）

即：同一段像素在不同播放轮次/不同运行里各注册了一个身份，而它们的特征最终长成了同一个向量。
已排除的路径：`register()` 在**生产路径中从不被调用**（只有 `demo.py` 与测试用，因此
"歧义 → 新建身份"这条链不成立）、`register_if_new()` 锁内生成独立 uuid、
`save_identity` 的 SQL 列序、`_persist_full()` 兜底路径。

**歧义的实际后果不是"错认"而是"永远认不出"**：`register_if_new()` 返回 `ambiguous` 后不注册，
`pipeline.py:527` 只打一行 debug，攒满的 8 帧 ReID 缓冲被丢弃，下一帧从头再攒。

**做法**（已完成）：

1. 在 `register_if_new()` 的 **ambiguous 分支之前**加塌缩护栏：相似度 ≥ 0.999 却判为歧义时
   打 `ERROR`（60 秒节流）并累加 `collapse_warnings`，计数外露到 `/api/metrics/reid`
   与 `/api/system/metrics`。**护栏必须放在 ambiguous 的 return 之前** —— 塌缩的表现正是
   "相似度极高却被判歧义"，放在 return 之后这条护栏永远执行不到（首版实现就踩了这个坑）。
2. 新增 `tests/test_reid_identity_integrity.py`（9 例）：并发注册 blob 唯一性、
   按身份去重语义、`feature_bank` 救回漂移主特征、护栏触发、指标字段同构。
3. 精确的创建时点需要当次运行日志才能定论，不在本方案范围内。

### 0.3 视频资产索引（能力 2 的数据基础）

容器元数据不可信 → 自建索引表，`frames_real` / `duration_real` 由实解码取得（`scripts/probe_media_frames.py` **已跑完**，22 路结果见 §2「现状实测」，并应直接落库而不是重新跑一遍）：

```sql
CREATE TABLE video_assets (
  asset_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  camera_id       TEXT NOT NULL,
  rel_path        TEXT NOT NULL,
  file_name       TEXT,
  sha1_8          TEXT,            -- 转码幂等判据
  size_bytes      INTEGER,
  width           INTEGER, height INTEGER, codec TEXT,
  fps_declared    REAL,            -- 元数据值，仅记录、不做计算依据
  frames_real     INTEGER,         -- ffprobe -count_frames 实测
  duration_real   REAL,            -- frames_real / real_fps
  loop_detected   INTEGER DEFAULT 0,
  low_value       INTEGER DEFAULT 0,  -- 内容过短（如 rnd_05 仅 1.3s），统计时排除
  ingest_ts       REAL
);
CREATE UNIQUE INDEX idx_assets_cam_path ON video_assets(camera_id, rel_path);
```

同时给轨迹表补**视频内坐标**（这是能力 2 能否"定位回放"的关键）：

```sql
ALTER TABLE identity_appearances ADD COLUMN asset_id    INTEGER;
ALTER TABLE identity_appearances ADD COLUMN video_frame INTEGER;  -- 源视频第几帧
ALTER TABLE identity_appearances ADD COLUMN video_ts    REAL;     -- 源视频内秒数
CREATE INDEX idx_appearances_gid_cam_ts
  ON identity_appearances(global_id, camera_id, timestamp);
```

**为什么必须双时间轴**：素材是**循环播放**的离线文件（`pipeline._run_file`），一条 appearance 的墙钟 `timestamp` 代表"第几轮播放的第几秒"，无法反查源视频位置。要让检索结果可跳转，只能落 `video_frame`/`video_ts`。老数据这三列为 NULL，检索接口返回 `position_known: false`，前端降级为"仅显示相机 + 墙钟时间"，**不做无根据的回填猜测**。

---

## 2. 能力三：分辨率降低

> 建议**第一个动手**——它把后续所有离线实验（重跑 22 路语料）的耗时压下来。

### 范围

- **做**：离线素材的一次性转码预处理 + 运行时缩放兜底（覆盖 RTSP 与未转码源）。
- **不做**：不改 MJPEG 输出侧（`FrameHub` 已是 480×270 / q50，`main.py` 的 `jpeg_quality=50`，无需再压）。

### 现状实测

| 项 | 数值 |
|---|---|
| 分辨率分布 | `1920x1080` × 13 路，`2560x1440` × 9 路 |
| 编码分布 | `h264` × 10 路，`hevc` × 12 路 |
| 声明帧率 | 25.00 fps（`rnd_05` 报 351.56，属损坏读数） |
| 总体积 | **983 MB** |
| 单路体积区间 | 4.6 MB（rnd_05）~ 174.3 MB（reg_05） |
| **实解码总帧数** | **67,576 帧** |
| **实解码总时长** | **约 2,700 秒 = 45 分钟**（22 路合计） |
| 单路真实时长区间 | 1.3 秒（rnd_05）~ 480 秒（reg_05，8 分钟） |
| 元数据虚高倍数 | 容器报 599,232 秒 vs 实解 2,700 秒 → **虚高 222 倍** |

**单路真实时长**（实解码，`scripts/probe_media_frames.py`）：

| 时长档 | 路数 | 相机 |
|---|---|---|
| ≥ 180s | 4 | reg_05 480s、reg_08 358s、reg_01 239s、reg_02 240s |
| 100~180s | 5 | reg_06 118s、reg_10 118s、rnd_06 178s、rnd_10 180s、rnd_12 178s、rnd_21 120s |
| 30~100s | 12 | rnd_16 69s、rnd_18 60s、rnd_07 58s、rnd_11 58s、rnd_17 38s、rnd_08 38s、rnd_19 39s、rnd_02 35s、rnd_01 33s、rnd_22 29s、rnd_04 20s |
| < 10s | 1 | **rnd_05 仅 1.3 秒 / 460 帧**（且 fps 元数据报 351.56） |

### ⚠️ 由此暴露的一个关键事实：语料被循环播放了约 300 倍

- 22 路素材**真实总时长只有 45 分钟、67,576 帧**
- 而库里已有 **227,124 条 `identity_appearances`**，平均每帧产出 **3.4 条**轨迹行
- 素材自 7 月 31 日采集，服务累计循环播放约 9 天

**这对能力二的影响是决定性的**：检索返回的"出现段"绝大多数是**同一段像素的重复播放**。所以 `video_assets` 除了 `duration_real`，检索接口还必须输出**循环倍数**，前端要默认折叠并明确标注"该片段在素材中只出现过 1 次，被循环记录了 N 次"。否则用户会得到"这个人在 rnd_01 出现了 3 万次"的荒谬结论——这正是 `collapse_loops` 已经在解决的问题，检索接口必须复用它。

### 另一个好消息：全量重跑语料只要半小时

45 分钟素材 = 67,576 帧。按当前实测吞吐 **38 帧/s**（22 路单进程）计算：

```
67576 帧 ÷ 38 帧/s ≈ 1,778 秒 ≈ 30 分钟
```

也就是说**改一版代码、全量重跑 22 路语料并重新生成身份库，只要约半小时**。这让能力一/二的迭代验证完全可行——之前担心的"重跑一次要几小时甚至一天"不成立。这也是能力三降分辨率最大的间接价值所在。

### 目标分辨率：960×540

**依据**：YOLOv8n 内部会把输入统一 resize 到 **640**；OSNet 输入是 **256×128**。也就是说模型侧的"有效分辨率上限"由 640/256 决定，往下游喂 1080p 纯属浪费解码与显存带宽。960×540 对"人体在画面中占中等比例"的场景几乎无损。

**唯一风险**：远距离小目标。若人体框高度 < 80px，降到 540p 后可能掉到 YOLO 的有效感受野以下 → **必须用 A/B 实验定档，不能拍脑袋**。

### 输入 / 输出

- 输入：`videos/常规路线/*.mp4`、`videos/随机路线/*.mp4` + `config/sources.json`
- 输出：`videos_low/<cam_id>.mp4`（新目录，gitignore；**原片保留归档**）、`video_assets` 索引行、`outputs/reports/resolution_ab.csv`

### 处理流程

```
sources.json
   ↓ 逐路 ffmpeg 转码（H.264 CRF 28 / preset medium / -an / 960x540 / -r 25）
videos_low/<cam_id>.mp4
   ↓ ffprobe -count_frames 实测帧数 → duration_real（容器时长不可信）
video_assets 索引行（含 sha1_8，供幂等跳过）
   ↓ 更新 sources.json 指向 videos_low/
运行时：`cap.read()` 后按 process_max_width 兜底缩放（仅 RTSP / 未转码源触发）
```

### 步骤

1. **`scripts/transcode_lowres.py`**：读 `sources.json` → 逐路 ffmpeg → 落 `videos_low/` → 实测帧数 → 写索引。参数 `--scale 960:540` / `--crf 28` / `--only reg_01` / `--dry-run`；已有输出且 `sha1_8` 一致则跳过（幂等）。逐个打印进度（单路 1440p HEVC 转码耗时可达分钟级）。
   - 全量转码预计 **10~20 分钟**（源共 983 MB，最长的 reg_05 为 480 秒 1440p）。
   - **`rnd_05` 单独处理**：真实内容只有 1.3 秒 / 460 帧，且 fps 元数据报 351.56（损坏读数）。在索引里标记 `low_value = 1`，**不删除**（采集点位不能随意摘除），但 A/B 实验与检索结果里默认排除它，否则统计口径会被这 460 帧污染。
2. **A/B 实验**（先做，再全量）：抽 3 路 —— `reg_01`（1080p 常规）、`rnd_21`（1440p 随机）、`rnd_16`（含唯一 ROI 围栏）。原片与低分片各跑 300 秒，比三件事：**检测人数召回**、**身份注册数**、**每路实际 fps**。写 `outputs/reports/resolution_ab.csv`。
3. **定档**：若小目标召回下降 > 5%，退到 `1280x720`；若下降 ≤ 2% 且 fps 提升明显，可再试 `854x480`。
4. **全量转码 + 切源**：`sources.json` 改指 `videos_low/`，原 `sources.json` 备份为 `config/sources_high.json`（保留可回退）。
5. **运行时兜底**：`CameraPipeline.__init__` 增 `process_max_width`（默认 960），`_read_loop` 里 `ret, frame = cap.read()` 之后按需缩放。保证 RTSP 与本地文件两条路径行为一致。
6. **回归**：`unittest discover -s tests -t .` 全绿 + 抽查大屏 22 路。

### 验收

- 22 路转码成功，`videos_low/` 总体积 ≤ 200 MB（预期 110~150 MB）
- A/B 报告中人数召回下降 ≤ 5%，每路 fps 提升 ≥ 10%
- `tests/` 全绿；`/api/status` 的 `display_width` 仍为 480（MJPEG 输出不受影响）

### 预期收益要说实话

实测数据出来后，这个能力的价值需要**下调**，理由如下：

1. **吞吐提升有限**：瓶颈是**单进程 GIL**，不是解码（22 路实测 38 帧/s，GPU 只用 35~40%）。降分辨率预期只带来 **+10~20%**。
2. **"全量重跑语料"本来就是可行的**：实解码显示 22 路素材**总共只有 45 分钟 / 67,576 帧**，按 38 帧/s 算**约 30 分钟就能重跑一遍**。所以"降分辨率才能重跑语料"这个论证**不成立**——它不需要降分辨率也只要半小时。
3. **API 层面无收益**：MJPEG 输出走的是 `FrameHub`（已是 480×270 / q50），与源分辨率无关。

**所以这个能力真正的价值只剩两条**：

- **工程整洁与分发**：983 MB → 约 130 MB，语料可随项目归档/分发，不必依赖 `videos/`（当前 `.gitignore` 排除）。
- **为将来接入 RTSP 实时流做铺垫**：实时流无法离线转码，只能靠运行时的 `process_max_width` 缩放，这条路径无论本次做不做都必须建。

**建议降级优先级**：能力三从"建议第一个动手"改为**与批次二合并、且可延后**。真正该先做的是批次一（修 ReID 权重 + 排查重复写入）——那才是决定"身份识别"能否成立的一步。

---

## 3. 能力一：人员身份识别

### 范围（必须先分清两层，否则需求会被误解）

| 层级 | 内容 | 现状 |
|---|---|---|
| **L1 匿名身份归一** | 跨相机把同一个人归一到一个编号 | **代码已实现**（`IdentityStore` + `global_id`），但被 0.1/0.2 阻塞，实际不可用 |
| **L2「具体是谁」** | 说出姓名/工号 | **未实现**。当前 `global_id = uuid4()[:8]`，是**随机编号，不含任何实名信息** |

所以"识别出其具体是谁"必须落成两条路径，**建议先做 A**：

- **路径 A · 人工命名（低成本，推荐先做）**：前端给 `global_id` 绑定姓名/工号。适合"现场只想知道目标人物是谁"。
- **路径 B · 底库 1:N 自动命名**：建人员底库（每人 1~N 张注册照 → OSNet 特征），实时检测先查底库，命中即命名，未命中才建匿名身份。适合"已有人员名单"。

### 输入

- 实时帧：现有 `/stream/{cam_id}` 链路，无需新增采集
- 命名来源（路径 A）：前端表单 + `person_id`
- 底库（路径 B）：证件照/抓拍图，每人 ≥ 1 张

### 输出（统一结构，供前端与能力 2 复用）

```json
{
  "track_id": 17,
  "global_id": "c0ed5fd1",
  "person_id": "P0007",
  "name": "张三",
  "employee_no": "QLU-1234",
  "camera_id": "rnd_16",
  "timestamp": 1757000000.0,
  "video_ts": 41.2,
  "bbox": [x1, y1, x2, y2],
  "similarity": 0.83,
  "match_source": "enrolled|gallery|new",
  "confidence": 0.91,
  "quality": 0.44
}
```

### 处理流程（在现有链路上扩展，不重写）

```
读帧 → YOLOv8 → ByteTrack
   ├─ ① 底库检索（路径 B）：特征 vs personnel_gallery，命中且 ≥ 阈值 → 直接命名
   ├─ ② 全局身份库检索（现状）：match_feature_detailed + Ratio Test + 3 帧确认 → global_id
   ├─ ③ 都未命中且缓冲满（8 帧）→ register_if_new() 建匿名身份
   └─ ④ 人工命名（路径 A）：POST /api/identities/{gid}/bind
            ↓
   写 identity_appearances（本轮新增 video_frame / video_ts / asset_id）
```

底库检索**插在现有 gallery 查询之前**：实名信息的价值高于匿名编号，且能顺带抑制匿名身份的膨胀（当前已有 45 个身份，其中两个 `total_appearances` 高达 13.7 万 / 3.9 万，明显是循环播放反复累加的"垃圾桶身份"）。

### 数据模型

```sql
CREATE TABLE personnel (
  person_id TEXT PRIMARY KEY, name TEXT NOT NULL, employee_no TEXT,
  department TEXT, note TEXT, created_at REAL, updated_at REAL
);
CREATE TABLE personnel_photos (
  photo_id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT NOT NULL,
  feature_dim INTEGER, feature_blob BLOB, quality REAL,
  source_path TEXT, created_at REAL
);
ALTER TABLE identities ADD COLUMN person_id TEXT;
ALTER TABLE identities ADD COLUMN name_confidence REAL;
```

按现有约定：新表加在 `db.py:_init_db`，新列加进该函数的 `migrations` 字典（保证老库自动升级）。

### API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET/POST | `/api/personnel` | 人员档案列表 / 新建 |
| PATCH/DELETE | `/api/personnel/{pid}` | 修改 / 删除 |
| POST | `/api/personnel/{pid}/photos` | multipart 上传注册照 → 提特征入库 |
| POST | `/api/identities/{gid}/bind` | 把匿名身份绑定到实名（路径 A 的核心） |
| GET | `/api/personnel/{pid}/identities` | 该人名下所有 `global_id` |
| — | `GET /api/identities` | 扩展返回 `person_id` / `name` |

写接口一律带 `X-Lab-Monitor-Request: 1`（现有守卫中间件要求）。

### 步骤

1. **先解 0.1 + 0.2**。特征不可分时，后续所有工作都是错的。
2. **标定阈值（本能力验收基线，不可跳过）**：在 `config/labeled/` 放人工标注对（同人跨相机 = 正样本对，异人 = 负样本对），写 `scripts/tune_reid_threshold.py` 扫阈值与 ratio，输出给定阈值下的 Precision / Recall 曲线，选定工作点。**没有这一步，"识别准确率"就无法证伪。**
3. **Schema 迁移** + `tests/test_personnel.py`（建档案、绑定、启停重启后仍能恢复绑定关系）。
4. `IdentityStore.bind_person(gid, person_id)`；`pipeline` 在确认身份后把 `person_id/name` 一并写进 appearance。
5. **路径 B**：新增 `src/personnel.py` 的 `PersonnelGallery`，复用 `match_feature_detailed()`（不要另写一套相似度逻辑）。
6. **前端**：人员档案弹窗加姓名/工号编辑与绑定入口；视频卡片标签由 `ID: #c0ed5fd1` 变为 `张三 (P0007)`。改完按项目约定 bump `static/index.html` 的 `?v=`（当前 CSS `11.5` / `app.js` `11.6`，`main.css` 的 7 条 `@import` 要一起改）。
7. **验收**：固定测试片段跑 3 遍，正样本对匹配率与误归并率以步骤 2 的标定结果为线。

---

## 4. 能力二：人员视频检索

### 范围

给定"一个人"，返回其出现过的**全部视频片段清单**。三种查询入口：

| 入口 | 依赖 | 说明 |
|---|---|---|
| ① `global_id` | 能力 1 的 L1 | 已在系统里识别过的人 |
| ② `person_id` | 能力 1 的 L2 | 实名聚合名下所有 `global_id` |
| ③ **以图搜人** | 仅依赖 0.1/0.3 | 上传一张图 → 提特征 → 全库检索。**唯一需要新算法的部分，也最有实用价值** |

### 输入

`person_id` / `global_id` / 图片文件，加上可选的时间窗、相机过滤、相似度下限。

### 输出

```json
{
  "query": {"type": "image", "person_id": null, "global_id": null},
  "galleries_compared": 45,
  "matched_identities": [
    {
      "global_id": "c0ed5fd1", "name": "张三", "score": 0.87,
      "assets": [
        {
          "asset_id": 12, "camera_id": "rnd_16",
          "file": "videos_low/rnd_16.mp4", "duration_real": 68.7,
          "hit_count": 37, "first_ts": 41.2, "last_ts": 68.9,
          "best_score": 0.87, "position_known": true,
          "loop": {"detected": true, "method": "period", "loop_factor": 288,
                   "unique_segments": 1, "raw_segments": 288},
          "thumb_url": "/screenshots/xxx.jpg"
        }
      ]
    }
  ],
  "total_assets": 3, "truncated": false
}
```

`loop` 字段直接复用 `/trajectory` 的 `loop` 结构（`detected` / `method` / `raw_segments`），只增补 `loop_factor` 与 `unique_segments`。**前端必须默认展示折叠后的结果**，并在 UI 上写明原始倍数。

### 处理流程

```
查询展开
  ├─ 图片查询 → OSNet 提特征 → 与 identities(feature + feature_bank) 全库比对
  │             → 取 score ≥ 阈值的身份集合（Top-K 排序，而非二值判定）
  └─ 实名/编号 → 直接取该 person_id 名下的 global_id 集合
        ↓
按 (asset_id, global_id) 聚合 identity_appearances
        ↓
split_gap_s 切停留段（复用 /trajectory 已有分段逻辑）
        ↓
关联 video_assets → 文件名 / 真实时长 / video_ts 起止（可回放定位串）
        ↓
可选：为每段抽 1 帧存缩略图（seek 解码，复用 FrameHub.get_frame）
```

### 四个必须写进代码注释的设计决定

1. **双时间轴不能混用**：`timestamp`（墙钟）用于跨相机排序；`video_ts`（视频内秒数）用于回放定位。循环播放会把墙钟时间拉成几天，拿墙钟去视频里 seek 必然定位到错位置。
2. **循环折叠复用而非重写**：`GET /api/identities/{gid}/trajectory` 已验证过的折叠逻辑（差分判据 `seq[i]==seq[i-period]`、覆盖校验、bbox 20px 指纹降级）要抽成公共函数 `collapse_loop_segments()` 放到 `src/trajectory.py`，让两个接口共用。**不要在检索接口里再写一遍**——`server.py` 里那段逻辑有 19 个测试保护，重写必然倒退。
3. **检索阈值独立于实时阈值**：离线检索返回 Top-K 排序，可以放宽阈值；实时匹配是二值判定，必须保守。两者共用同一个常量是设计错误。
4. **必须返回循环倍数 `loop_factor`，不能只返回命中段数**：语料真实总时长仅 45 分钟，却已累计 227,124 条 appearance，同一段像素平均被循环记录了数百次。若不折叠、不标注倍数，用户看到的"出现 3 万次"是纯粹的度量假象。**检索接口的默认输出必须是折叠后的结果**，`raw_segments` 与 `loop_factor` 作为解释字段附带返回。

### API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/search/person` | 按 `person_id`/`global_id` + 时间窗/相机检索 |
| POST | `/api/search/by-image` | multipart 以图搜人 |
| GET | `/api/assets` | 视频资产索引（列出语料范围与真实时长） |

### 步骤

1. 完成 **0.3**（`video_assets` + appearance 三列）+ `tests/test_video_assets.py`。
2. **老数据降级**：历史 appearance 的 `video_ts` 为 NULL → 检索结果标 `position_known: false`，前端只显示相机与墙钟时间。
3. 抽 `collapse_loop_segments()` 到 `src/trajectory.py`，`/trajectory` 与检索共用；确保现有 `tests/test_trajectory.py`（19 例）全绿。
4. 实现 `GET /api/search/person`（纯 SQL 聚合，无新算法）。
5. 实现 `POST /api/search/by-image` + `src/search.py`（特征检索 + Top-K + 缩略图）。
6. **前端**：人员档案弹窗内加「该人出现过的视频」表格（相机 / 文件名 / 时长 / 出现段数 / 跳转）；点击跳现场画面或平面图回放。
7. **验收**：从 45 个已有身份随机抽 10 个，人工核对检索出的视频清单是否覆盖已知出现相机；以图搜人用留出集测 Top-1 / Top-5 命中率。

---

## 5. 依赖关系

```
0.1 修 ReID 权重 ──┬──► 能力1 身份识别 ──► 能力2 的「实名/编号查询」入口
0.2 修持久化重复 ──┘                        （「以图搜人」入口不依赖能力1）
0.3 视频资产索引 ─────► 能力2 视频检索（硬依赖）
能力3 降分辨率 ───────► 提速项，非硬依赖；但离线实验强烈建议先做
```

- **能力 1 硬依赖 0.1 + 0.2**：特征不可分 → 身份必错；持久化重复 → 身份表本身脏。
- **能力 2 硬依赖 0.1 + 0.3**；对能力 1 是**弱依赖**：以图搜人可独立工作，"按姓名检索"必须有能力 1 的实名绑定。
- **能力 3 与其余是加速关系**，非硬依赖。
- **反向依赖（有用的一点）**：**能力 2 是能力 1 的验收工具** —— 用检索结果人工核对身份有没有串号，比看统计指标更直观。

### 推荐实施顺序（增量交付，每批可验证）

| 批次 | 内容 | 交付物 |
|---|---|---|
| 一 | 0.1 换 ReID 权重 + 0.2 排查重复写入 | 阈值标定报告 + 全绿测试 |
| 二 | 0.3 视频资产索引（**能力 3 转码并入本批，可延后**） | `video_assets` 表 + 真实时长基线 + A/B 报告 |
| 三 | 能力 1 路径 A（人工命名） + L1 修复验证 | `personnel` 表 + 绑定 API + 前端命名入口 |
| 四 | 能力 2（`global_id` 查询 → 以图搜人） | `/api/search/*` + 前端视频清单 |
| 五 | 能力 1 路径 B（底库 1:N 自动命名） | 注册照上传 + 自动命名 |

### 后续需求（本次不做）

A→B→C 行程时间串联超时预警。**当前阻塞原因**：缺少「哪些摄像头物理相连」与「视频文件 ↔ 摄像头点位」的对应关系。补齐后可直接复用现有 `CameraTopology` + `TransitCalibrator`（`config/topology.json` 已有 22 节点的边定义，但预期时间来自点位表推算而非实测，且 `camera_map.json` 的 `map_xy` 目前**全为 null**，点位一个都没标）。

---

## 附：本次新增的实测脚本

| 脚本 | 用途 |
|---|---|
| `scripts/probe_media_ffprobe.py` | ffprobe 取分辨率/编码/声明帧率，暴露容器元数据不可信 |
| `scripts/probe_media_frames.py` | `-count_frames` 实解码，得到可用于设计的真实时长基线（**已执行**：22 路合计 67,576 帧 / 约 45 分钟，耗时 10.5 分钟） |
| `scripts/probe_reid_separability.py` | 组内/组间余弦分布 + 阈值代价统计（读**库内**特征） |
| `scripts/diagnose_reid_degeneracy.py` | 区分「特征退化」与「持久化重复」，按 blob md5 分组 |
| `scripts/compare_reid_weights.py` | 同一批真实 crop 上对比各权重的嵌入质量（读**现提**特征），并输出每路检测密度 |
| `scripts/diagnose_ema_collapse.py` | 判定 EMA 滑动平均是否为塌缩主因（同一人 vs 混人两组对照） |
| `scripts/fetch_reid_weights.py` | 下载 torchreid 官方 ReID 权重（走代理、幂等、按分类头校验数据集） |

全部为只读脚本（`fetch_reid_weights.py` 只写 `~/.cache/torch/checkpoints/`），不修改项目文件。
