# 三项能力实施工作清单

> 2026-09-12 · 依据已批准的 `docs/PLAN_2026-09-12_identity_search_resolution.md`
> 标注：**S** 小（单文件局部改动）· **M** 中（跨 2~4 文件，含测试）· **L** 大（新模块 + API + 前端）
> 勾选状态随实施推进更新。**批次一全部是阻塞项，未完成前批次三及之后不应开工。**

---

## 批次一 · 修特征层（阻塞项，8 项）

判定依据：实测 990 对异人特征余弦中位 0.984、99.8% 越过阈值 0.75。**特征层不修，后面所有工作都是错的。**

> ⚠️ **2026-09-12 实施中的两次重大修正**（每一轮都是先做对照实验再下结论）：
> **第一次**：库内那批塌缩特征不是模型吐出的原始特征（本机实测原始特征均值范数 0.796 /
> 余弦 p50 0.629，并未塌缩），而是被 `update_appearance()` 的聚合加工后的结果
> → 一度判断"主因是聚合方式，权重是次要"。
> **第二次（推翻了第一次的修复方案）**：实测"有界窗口均值"与 EMA 的塌缩**完全一样**
> （20 次观测跨身份 p50：0.7330 vs 0.7366）→ 聚合方式不是杠杆。
> 真根因是**特征里的公共分量**（全体均值范数 0.80~0.83）：
> 减去它之后跨身份余弦从 +0.50 掉到 **-0.10**、越阈比例从 21~46% 掉到 **0~7%**。
> 修复已实施为匹配路径上的中心化（见 1.6）。
> **教训：连续两次"显得很有道理"的因果推断都被实验否掉了 —— 只保留可测的结论。**

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

- [x] **1.3 自动标注集 + 阈值标定** —— **M**（2026-09-12 完成，全自动、零人工）
  - 标签来源（两条运动学约束，物理上不可能错，因此**无需人工确认**）：
    正样本 = 同相机相邻采样帧(0.4s) IoU>=0.25 的检测框（0.4s 内一人最多移动 ~0.5m）；
    负样本 = 同相机同帧 IoU<=0.05 的两框 + 「时间区间重叠的不同轨迹簇」（一人不能同时在两处）。
    规模：正 180 对 / 自动负 6 对（走廊素材极少两人同框，属预期）。
  - 新增 scripts/build_label_set.py：扫描视频 + 并查集聚簇 + 配对，
    落盘 config/labeled/pairs.json 与 outputs/labeled/crops.npz。
  - 自动负样本不足，由「重建库 22 身份的跨身份对(226 个)」作代理负样本补足；
    其中少数对可能是同一人（1.8 欠归并），会把负样本高位略抬高 → 选出的阈值偏保守。
  - 新增 scripts/tune_reid_threshold.py：扫 0.20~0.90 每 0.02 档，输出 P/R/F1/FPR，
    在误报率 <=5% 约束下取 F1 最大。明细 outputs/reports/reid_threshold_sweep.{csv,json}。
  - **标定结论**：

    | 权重 | 阈值 | P | R | F1 | FPR |
    |---|---|---|---|---|---|
    | msmt17 | 0.68 | 0.899 | 0.544 | 0.678 | 4.7% |
    | market1501 | 0.68 | 0.913 | 0.639 | **0.752** | 4.7% |

    **market1501 胜出**（同阈值同误报率下召回高 9.5 个百分点）；
    **阈值 0.68（下调 —— 方向与直觉相反）**：中心化把异人分布 p95 压到 0.673，
    沿用多年的 0.75 会漏掉大量同人匹配（同人对 p50 只有 0.69~0.75）。
    另：precision 在 0.68~0.80 之间几乎持平（0.913→0.920），升高阈值只损失召回、不提升精度。
  - 已回填 src/reid_config.py：_DEFAULT_THRESHOLD=0.68、DEFAULT_REID_WEIGHTS=market1501。
  - **本方法的盲区**：正样本全部来自同相机相邻帧，跨相机同人样本无法自动标注；
    将来若发现跨相机漏配，优先怀疑这里，需要人工标注补一批跨相机正样本。

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

- [x] **1.6 修公共分量（ReID 识别失效的真正根因）** —— **L** ｜**已完成** ｜原定"改聚合方式"已被实测否决
  - **原判断被推翻**：原以为根因是 EMA 指数抹平身份残差，候选方案是"有界窗口均值替代 EMA"。
    实测对比（`scripts/diagnose_ema_collapse.py`，同一人样本、8 个身份）后**否决**该方案：
    | 聚合策略 | 20 次观测跨身份 p50 | 200 次 |
    |---|---|---|
    | EMA(α=0.85) | 0.7330 | 0.7344 |
    | 窗口均值 N=5 | 0.7313 | 0.7313 |
    | 窗口均值 N=20 | **0.7366** | 0.7366 |
    | 窗口均值 N=50 | 0.7358 | 0.7410 |

    三者几乎相同 → **塌缩来自"聚合"本身，不是 EMA 的指数衰减**。换成窗口均值毫无收益。
  - **真根因（实测）**：OSNet 无论用 ImageNet 还是 ReID 度量学习权重，输出的单位特征都与
    一个**全局方向**高度共线 —— 全体特征均值向量范数 **0.80~0.83**（随机方向的期望只有 ~0.1）。
    这个分量对区分身份毫无贡献，却让跨身份余弦虚高：

    | 特征 | 跨身份余弦 p50 | 越过阈值 0.75 的异人对 |
    |---|---|---|
    | imagenet 原始 | 0.4961 | 21.4% |
    | imagenet EMA 200 次 | 0.7344 | 46.4% |
    | msmt17 原始 | 0.4978 | 14.3% |
    | **减去公共分量（imagenet）** | **-0.1021** | **0.0%** |
    | **减去公共分量（msmt17）** | **-0.1319** | **0.0%** |
    | **聚合后再减公共分量（msmt17 EMA 200 次）** | **-0.2823** | **3.6%** |

    聚合之所以"有害"，只是它把这个残余的判别余量进一步挤掉；**病根是公共分量**。
  - **已实施**：中心化放在**匹配路径**上（不动 `update_appearance`，改动面最小）——
    `IdentityStore._feature_center_locked()` 用全体已注册特征（主特征 + feature_bank）的均值
    作为中心，`MatchContext` 把"已中心化的 gallery"与"所用中心"**原子打包**返回，
    调用方必须用 `context.prepare(query)` 变换查询向量。
    - 为什么必须打包：gallery 与 query 若用了不同的中心（哪怕只是稍旧一点的），
      两者就不在同一坐标系，**相似度全错且不报错**。拆成两个方法调用时，
      中间另一个线程注册新身份就会制造这种静默错配。首版实现就是这个隐患，被单测抓出来。
    - 两个护栏保证冷启动安全：有效特征数 < 8 或中心范数 < 0.35 时**不启用**中心化，
      行为与改造前完全一致。
    - `center_enabled` / `center_norm` 外露到 `/api/metrics/reid` 与 `/api/system/metrics`。
  - **附带发现：库内历史特征无法离线修复**。对生产库 45 个身份做离线中心化，
    越阈比例只从 99.80% 降到 33.64%（主特征）/ 11.85%（含 bank），
    且组内相似度变成 **-0.130** —— 说明这些身份的身份特异残差已被"混人 + 海量 EMA"
    彻底抹掉，中心化只能去掉公共分量、救不回已丢失的信号。
    **结论：必须重建身份库（见 1.5），这不是可选优化。**
  - 验收：新库上 `over_threshold` 应为个位数百分比、`collapse_warnings` 为 0。

- [x] **1.7 顺带修掉的既有缺陷（本批发现）** —— **S**（已完成）
  - `Database.close()` 原先只关**当前线程**的连接，22 个 pipeline 线程各自的连接会存活到
    进程退出 → Windows 上库文件被占用（临时目录删不掉、备份后无法 rename，实测触发
    `PermissionError: [WinError 32]`）。现改为登记全部线程连接并统一回收，
    连接以 `check_same_thread=False` 建立（每条连接仍只被创建它的线程使用）。
  - `tests/test_stream_manager_frontend.mjs` 的轮转周期断言停留在 `12000ms`，
    而 `1a5b17b` 已把模块常量改成 `6000ms` —— 该用例自那次提交起**一直失败**却没人发现。
    已改为镜像常量 `EXPECTED_ROTATE_MS` / `EXPECTED_FILL_MS` 并注明需与模块同步。

- [x] **1.8 身份归并（consolidation）** —— **M**（2026-09-12 完成）
  - 现象：msmt17 重建后的 22 个身份里，a093d0a6(rnd_01) 与 rnd_02/rnd_05/rnd_11/rnd_17/rnd_22
    共 6 个身份的中心化特征**完全相同（sim=1.0000）**，全部在同一条走廊链上。
  - 成因：中心化的冷启动护栏（有效特征 < 8 时不启用）是必要的安全设计，但运行初
    gallery 很小，早期身份在**未中心化**空间里各自注册；之后中心化生效，
    却**没有回溯归并机制**，已分开的身份永不合并。
  - **已实施**：
    - IdentityStore.consolidate(threshold, dry_run) / consolidation_candidates(threshold)：
      以与线上完全一致的语义（中心化 + 重新归一化 + max-over-bank）找候选并归并；
      保留 total_appearances 较多的一方、feature_bank 取并集去重、appearances 合并截断；
      **数据库侧把 identity_appearances 的 global_id 改挂到主身份**再删除被并方
      （Database.merge_identities），否则轨迹断链；幂等（重复执行结果一致）。
    - scripts/consolidate_identities.py（执行前自动备份，支持 --dry-run）。
    - 归并阈值**直接用匹配阈值**：否则存储状态与实时匹配行为互相矛盾
      （实时匹配认为同一个人、存储里却是两个身份）。
  - **执行结果**（market1501 重建后）：10 个身份，归并 4 组，剩 **6 个身份**，
    归并后高于阈值的对 0 组。
  - **最终验收**（scripts/verify_rebuilt_identities.py）：6 身份 / 6 唯一 blob /
    0 重复组；15 对身份两两中心化相似度 p50=0.287、p95=0.573、**max=0.613**，
    **0 对越过阈值 0.68** —— 全部身份彼此可分，且相对阈值留有 0.067 余量。
  - **教训记录**：本轮两次用内联 python -c 做离线分析得到互相矛盾的结果
    （同一对身份一次 1.0000、一次 0.6366），根因是漏了"中心化后重新归一化"——
    不归一化的点积可以超过 1，与线上语义不一致。已固化成带自检的
    scripts/verify_rebuilt_identities.py，**离线分析一律走脚本、不走内联**。

- [x] **1.5 重建身份库并验证** —— **S**（2026-09-12 完成，正式库已重建并复核）
  - 为什么必需：库内历史特征**无法离线修复**（见 1.6 的附带发现）——
    越阈比例只能从 99.80% 降到 11.85~33.64%，组内相似度变成负值。
    这些身份的身份特异残差已被"混人 + 海量 EMA"抹掉，只能重建。
  - 工具已就位：`scripts/eval_identity_pipeline.py`（**新增**）——
    走真实链路（PersonDetector / build_reid_extractor / IdentityStore / CameraPipeline），
    有界运行指定秒数后停机并输出验收指标，落 `outputs/reports/identity_pipeline_eval.json`。
    比直接跑 `main.py` 更可控：`main.py` 是无限循环，跑一轮约 30 分钟且不会自己停。
  - **评测库已验证（msmt17 / 5 路 / 600s）**：

    | 指标 | 数值 | 判读 |
    |---|---|---|
    | 身份数 / 唯一 feature_blob | 6 / 6 | ✓ 一致，无重复 |
    | 原始特征余弦 p50 | 0.8722 | 公共分量未减时的落库形态（虚高符合预期） |
    | **中心化后余弦 p50** | **-0.2354** | ✓ 识别真正可用 |
    | **中心化后越阈比例** | **6.67%** | ✓ 个位数（改造前 99.80%） |
    | 中心化后 p95 | 0.7477 | ✓ 恰好低于阈值 0.75 |
    | 匹配率 success/total | 69/147 = **46.94%** | ✓ 能复用已有身份 |
    | Ratio 判歧义 / 塌缩告警 | 0 / **0** | ✓ |
    | 中心范数 | 0.8746 | 公共分量真实存在 |

    结论：**修复在真实链路上生效**。`match_rate 46.94%` + `collapse_warnings 0` 说明
    "塌缩→歧义→丢弃缓冲"的死循环已被打破。
  - **正式库已重建（2026-09-12 11:04–11:35，22 路 / msmt17 / GPU 档 / 31 分钟）**：
    备份 `outputs/lab_monitor.db.bak-before-rebuild-20260912-110244` 后清空
    identities（45）与 identity_appearances（227,124），告警审计日志保留（7,066 条）。
    停机走 `POST /api/admin/shutdown`，关停路径正常补写 20 个身份的特征。

    | 指标 | 旧库（重建前） | **新库（重建后）** |
    |---|---|---|
    | 身份数 / 唯一 blob | 45 / 25 | **22 / 21** |
    | 原始跨身份余弦 p50 | 0.984 | 0.838 |
    | **中心化后跨身份余弦 p50** | **0.778** | **-0.031** |
    | **中心化后越阈异人对** | **53.43%** | **1.30%**（3 / 231 对） |
    | 中心化后组内相似度均值 | **-0.130**（无身份信号） | **+0.284**（有身份信号） |
    | Ratio 判歧义对 | 193 / 231 | **0 / 231** |
    | 本轮塌缩告警 / 跨摄归并 / 身份确认 | —（为 0） | **0 / 6 次 / 106 次** |

    **核心结论：跨相机识别已经工作** —— 中心化后 231 对异人里只有 3 对越阈，
    且 Ratio 判歧义为 0（旧库 193 对全判歧义、跨摄归并 0 次）。
  - **遗留的精确问题（3 对越阈全是同一相机内的重复注册）**：
    `8f708fd8(rnd_17)↔bab050cd(rnd_17)` 字节相同、`total_appearances` 都是 34；
    另两对也在同相机（rnd_16 / reg_10）。**没有任何一对是跨相机的**。
    成因：主特征是 EMA，会随观测漂移——同一个人重新入镜时的"新鲜特征"
    与已漂移的存储特征差到阈值之外（当时 <0.75），于是再次注册；之后两者又收敛到同一点。
    对策二选一（归 1.3 / 1.6②）：**①** 调低阈值（但会增加误归并风险，需标注集定夺）；
    **②** 把主特征改成有界窗口均值，让它追踪近期外观而不是无限记忆。
    注意：方案 ② 在"降低跨身份塌缩"上已被实测否决，但对"漂移导致的同相机重复注册"
    是对症的 —— 两者是不同的问题。
  - 验收：`identities == unique_blobs`（差 1，见上）、`centered.over_threshold` 1.30%、
    `collapse_warnings == 0`、跨摄归并 6 次 ✓。

---

## 批次二 · 数据基础（6 项，2026-09-12 全部完成）

- [x] **2.1 `video_assets` 表 + 实测元数据** —— **M**（完成）
  - 建表（含 sha1_8 / frames_real / duration_real / loop_detected / low_value），
    `UNIQUE(camera_id, rel_path)` 允许同一相机同时有"源文件"与"低清转码产物"两行。
  - `scripts/seed_video_assets.py`（新增）：`ffprobe -count_frames` **实解码**逐路实测，
    结果落库后复用（--refresh 才重测）。实测 22 路耗时 22 分钟（与转码并行时被拖慢）。
  - `GET /api/assets`（新增）：列出语料范围与实测时长，标注 `measured` 与 `low_value`。
  - 实测结果：22 路合计 **67,576 帧 / 45.1 分钟**，与早前独立探测完全吻合。

- [x] **2.2 轨迹表补「视频内时间」三列** —— **M**（完成）
  - `identity_appearances` 加 `asset_id` / `video_frame` / `video_ts`（迁移走
    `_init_db` 的 appearance_migrations 字典，老库自动升级），
    加复合索引 `idx_appearances_gid_cam_ts(global_id, camera_id, timestamp)`。
  - **为什么必须双时间轴**：素材循环播放，墙钟 `timestamp` 表示"第几轮播放的第几秒"，
    无法反查源视频位置；检索回放只能靠这三列。
  - 老数据三列为 NULL → 检索返回 `position_known: false` 降级，**不做无根据的回填猜测**。

- [x] **2.3 采集侧写入视频内坐标** —— **S**（完成，端到端已验证）
  - 新增 `CameraPipeline._file_frame_idx`：**当前文件内帧号**。
    关键坑：原有的 `_frame_idx` 是进程生命周期累计值，**不随素材循环复位**，
    直接拿它当"源视频第几帧"是错的 —— 必须在 `_reset_stream_state()` 里归零。
  - `video_ts = _file_frame_idx / _video_fps`，fps 取自资产索引
    （元数据帧率可能损坏，只接受 [10,60] 区间，否则回落 25）。
  - **端到端验证**（eval 75 秒 / 2 路）：37/37 行三列全部非空，
    `video_frame=497 → video_ts=19.88s` 与 25fps 换算精确一致。
  - 顺带修掉评测脚本的一个顺序 bug：`--keep-db` 的复制原先发生在 `database.close()`
    **之前**，WAL 未 checkpoint，拷出去的库缺最近事务（实测拷出来是空库）。

- [x] **2.4 运行时缩放兜底 `process_max_width`** —— **S**（完成）
  - `CameraPipeline` 新增参数（默认 960，`LAB_MONITOR_PROCESS_MAX_WIDTH` 可覆盖）；
    在 `_process_frame` 与 `push_frame` **之前**等比缩小（否则 YOLO/ReID/MJPEG/截图
    拿到的还是原始大帧）。本地素材已转码时不触发（宽度已 <= 上限），零开销。
  - 用途：RTSP 在线流无法离线转码，只能靠这条路径。

- [x] **2.5 `scripts/transcode_lowres.py`** —— **M**（完成；A/B 实验暂缓，见下）
  - 支持参数：--scale（默认 960:-2，两种源分辨率都恰好落在 540p 无黑边）/ --crf /
    --only / --dry-run / --refresh / --seed-only / --switch-sources。
  - **已全量执行**：22 路全部转码成功，**983MB → 29MB（压缩 34 倍）**，
    且统一 `-r 25` 顺带修复了 rnd_05 损坏的 351.56 元数据。
    转码产物已全部登记进 video_assets（44 行 = 22 源 + 22 低清）。
  - 切换开关 `--switch-sources` 已就绪但**未执行**（会改变所有资产的 rel_path 语义，
    需要先决定是否重建身份库，见下）。
  - **A/B 实验暂缓的理由**：吞吐瓶颈是 GIL 不是解码，降分辨率的吞吐收益本来就有限
    （预期 +10~20%）；而现在身份库是按源视频的资产索引建的，切换到 videos_low 会
    使已有身份的 asset_id 指向旧路径。建议把 A/B 与"是否切换 sources"一起推迟到
    批次四（检索）落地后，用真实检索效果来定 —— 届时有业务收益可依据。

- [x] **2.6 rnd_05 修正** —— **S**（完成，结论修正）
  - **原结论有误**：之前写"rnd_05 仅 1.3 秒"，那是用损坏的元数据帧率 351.56 除出来的。
    按 fps 修正规则（只接受 [10,60] 区间，否则回落 25）实解 460 帧 → **真实时长 18.4 秒**。
  - 因此 `low_value` 阈值（<5s）在当前 22 路上**一个都不命中** —— 22 路最短 18.4s，
    全部保留。low_value 字段保留（将来接入更短素材时有用），当前无排除对象。

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

## 批次四 · 能力二：人员视频检索（6 项，2026-09-12 全部完成）

- [x] **4.1 抽出 `collapse_loop_segments()` 到公共模块 `src/trajectory.py`** —— **M**（完成）
  - 迁移：`detect_loop_period` / `collapse_by_fingerprint` / `build_segments` /
    `collapse_loop_segments`（分段 + 折叠一体）。
  - server.py 保留 `_collapse_loop_period` / `_collapse_by_fingerprint` 别名
    —— tests/test_trajectory.py 通过 `from server import ...` 引用，改名会打断 19 个用例。
  - 轨迹路由改为复用公共模块（顺带把分段逻辑也抽了出去，检索接口同样要用）。
  - 验收：`test_trajectory.py` 19 例全绿 ✓

- [x] **4.2 `GET /api/search/person`** —— **M**（完成）
  - 核心逻辑在 `src/search.py::aggregate_identity_assets()`（HTTP 层只做参数解析）。
  - 按 (asset_id, camera) 聚合 → 切停留段 → 循环折叠 → 关联 video_assets。
  - 返回 `loop_factor`（逐资产：折叠前原始命中数 / 折叠后保留数）——
    语料只有 45 分钟却累计 22 万条 appearance，不标注倍数会得到"出现 3 万次"的假象。
  - 返回 `position_known`：老数据缺视频内坐标时降级为"仅相机 + 墙钟时间"。
  - **真实库冒烟**：单身份命中 22 个视频资产，loop_factor x1.0~x10.4；
    position_known 全 False（这批轨迹是在 2.3 落地前写入的，属预期降级；
    服务重启后新轨迹会带坐标）。

- [x] **4.3 过滤已下线相机** —— **S**（完成）
  - 库里有 rnd_03 的 6,447 条孤儿轨迹（历史配置残留）。检索按
    `_known_camera_set()`（pipelines ∪ frame_hub ∪ video_assets 索引里的相机）过滤。
  - 注意：不能只用 video_assets 索引（rnd_03 没有资产行，会被正确滤掉），
    也不能只看 pipelines（单测未注入时会失效）—— 取并集两个场景都对。

- [x] **4.4 `POST /api/search/by-image` + `src/search.py::rank_identities_by_feature`** —— **L**（完成）
  - 上传图 → cv2 解码 → 检测人体框 → 取**面积最大**者（多人时纯启发式，
    响应带 person_count 供前端提示框选）→ OSNet 提特征 →
    `build_match_context()` 全库比对（中心化坐标系）→ Top-K 排序。
  - 返回 `matched` 字段（用实时阈值 0.68 标色），但**调用方不应据此丢弃**排名靠后
    但分数可观的候选 —— 离线检索阈值语义与实时匹配独立。
  - `init_server()` 新增 detector / reid_extractor 注入（main.py 传入模型池代理）。
  - **缩略图暂未实现**（需按 video_ts seek 解码源视频），响应里用 /stream/{cam} 代替。

- [x] **4.5 前端入口** —— **M**（完成，最小可用版）
  - 人员轨迹弹窗（跨镜头通行轨迹链）底部新增「🔎 检索该人出现过的全部视频」按钮：
    调 GET /api/search/person，渲染命中视频表格（相机 / 文件 / 时长 / 命中帧数与循环倍数 /
    视频内位置 / 实时画面链接），position_known=False 的行明确标注"无视频内坐标（老数据）"。
  - 版本号已 bump：`main.css?v=11.7`、`app.js?v=11.7`（含 7 处 @import 同步）。
  - 以图搜人的前端入口（上传图）暂未加 —— 命令行/curl 可直接调，等 4.4 的
    缩略图与多人框选交互一起设计更合适。

- [x] **4.6 `tests/test_search.py`（11 例）+ `tests/test_search_api.py`（7 例）** —— **S**（完成）
  - 聚合：按文件分组、循环折叠（25 段折叠成 1 轮）、视频内坐标、老数据降级、
    时间窗过滤、未知身份、**已下线相机过滤**
  - 以图搜人：Top-1 命中、降序排序、正交方向不崩
  - HTTP 层：200 结构、404 未知身份、守卫头 403、ReID 未注入 503、坏图 400

- **接口层缺陷（过程中发现并修复）**：检索端点原先用 `from src.db import db`
  全局单例查轨迹，而身份库可能挂在另一个库上（测试里是临时库）→ 检索永远为空。
  改为 `_identity_store.database`（身份库自带引用），保证"身份与其轨迹同库"。

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
| 一 · 修特征层 | 8 | **8 全部完成** | 批次一收口 |
| 二 · 数据基础 | 6 | **6 全部完成** | 完成 |
| 三 · 能力一 身份识别 | 6 | 0 | 依赖批次一（已满足）；personnel 实名绑定 |
| 四 · 能力二 视频检索 | 6 | **6 全部完成**（4.5 仅最小前端；缩略图未做） | 完成 |
| 五 · 能力一 路径 B | 3 | 0 | 依赖批次三 |
| 并行清理 | 8 | 0 | 无 |

**批次一、二、四均已收口。** 当前能力状态：
- 身份库：6 个身份、两两中心化相似度 max 0.613、0 对越过阈值 0.68，全部彼此可分；
- 检索：`GET /api/search/person?global_id=...`（按编号检索，真实库冒烟命中 22 个视频、
  loop_factor x1.0~x10.4）与 `POST /api/search/by-image`（以图搜人 Top-K）均已可用；
- 前端：人员轨迹弹窗新增「检索该人出现过的全部视频」按钮（版本号 11.7）；
- 低清语料 29MB 已转码登记（44 行资产索引）。

**下一步是批次三（能力一：人员身份识别，personnel 表 + 绑定）**
与批次四（能力二：人员视频检索）。批次三依赖批次一（已满足）；
批次四依赖 2.1~2.3（已满足）。两者可以并行推进，建议先做批次四 ——
检索是验收身份识别最直接的工具。

**遗留的三个可选优化**（不阻塞）：
① consolidate() 挂成周期任务（现在只能手动跑脚本）；
② 中心预热（用本库特征预置中心，让重建后第一帧就有中心化）；
③ 跨相机人工标注补盲区（自动正样本全是同相机相邻帧）。
另：`--switch-sources`（切到低清语料）已就绪但未执行 —— 会改变资产 rel_path 语义，
建议与"是否重建身份库"一起在批次四落地后再决定。
