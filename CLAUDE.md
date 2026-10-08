# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Lab-Monitor** 是基于 YOLOv8 + ReID 的多摄像头实时监控与智能告警系统。纯 Python 项目（无 Node.js、Docker），使用 FastAPI 提供 Web 服务，前端为无构建流程的模块化静态资源（`static/index.html` + ES module JS + 拆分 CSS）。在"检测-跟踪-识别"之上，另提供三项业务能力：**实名人员档案**（底库 1:N 自动命名）、**跨摄像头视频检索**（按人找片 / 以图搜人）、**轨迹回放与路线图**（循环播放折叠），见「核心架构」末尾的业务能力一节。

## 常用命令

### 依赖安装

**必须按顺序安装**（PyTorch 依赖需先于其他库）：

```bash
# CPU 版本（通用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU 版本（当前开发机实测：Python 3.13.14 + torch 2.13.0+cu126 + RTX 3090，CUDA 可用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> **依赖清单**：`numpy`、`opencv-python`、`psutil` 已写入 `requirements.txt`，无需手动安装；`requirements.txt` 里**没有** `torch` / `torchvision`，必须按上面的顺序先单独装。`scipy` 已在 P1 优化中移除。
>
> **OSNet 首次运行**：预训练权重（~3MB）自动从 Google Drive 下载缓存到 `~/.cache/torch/checkpoints/`，需联网（或配置代理）。

### 运行服务

```bash
# 前台运行（推荐开发时使用，可直接看日志）
./.venv/Scripts/python.exe main.py    # 系统 PATH 里的 python 是 Microsoft Store 存根，会静默退出

# 后台运行（Windows PowerShell）
.\start.ps1    # 用 .venv\Scripts\python.exe 拉起 main.py（main.py 自己写真实 PID 到 outputs/server.pid）
.\stop.ps1     # 停止服务

# 访问 Web 界面
http://localhost:8000
```

> **监听地址与鉴权**：默认只监听 `127.0.0.1`（`main.py --host` 可改）。`LAB_MONITOR_USERNAME` / `LAB_MONITOR_PASSWORD` 两个环境变量都为空时**所有接口无鉴权**（此时 `server.py` 会拒绝绑定非本机地址）。另存在 `POST /api/admin/shutdown`（仅接受回环地址调用）可直接停服，调试时勿误触。

> **所有写接口（POST/PUT/PATCH/DELETE）必须带请求头 `X-Lab-Monitor-Request: 1`**，否则被守卫中间件 403 拒绝；`Origin` 存在时还必须同源，`Host` 必须在白名单内（否则 421）。前端已在 `static/js/utils/api.js` 统一注入，`stop.ps1` 也已补头。**手工 curl 写接口时别忘了带**，且中文字段要以 UTF-8 发送（Git Bash 直接 `-d` 会按 GBK 编码导致 400）：
>
> ```bash
> curl -X POST http://127.0.0.1:8000/api/roi -H "Content-Type: application/json" -H "X-Lab-Monitor-Request: 1" --data-binary @payload.json
> ```

**环境变量一览**（全部可选）：

| 变量 | 默认 | 作用 |
|------|------|------|
| `LAB_MONITOR_MODEL_POOL` | `0`（自动） | 推理模型实例池大小。0 时 GPU 取 `min(2, 源数)`、CPU 取 1（上限 2026-08-25 从 6 降到 2，实测 22 路下池=2/4/6 吞吐都是 38 帧/s，池=1 才掉到 22.9）。设 `1` 可回退到改造前的串行行为（但线程钳制不随之回退，见 `main.py`） |
| `LAB_MONITOR_ALLOWED_HOSTS` | 空（按绑定地址推导） | 逗号分隔的 Host 白名单，绑定通配地址时必须显式配置，否则 Host 校验会被关闭 |
| `LAB_MONITOR_USERNAME` / `_PASSWORD` | 空 | Basic 鉴权凭据；都为空即无鉴权 |
| `LAB_MONITOR_MAX_IDENTITIES` | `10000` | 全局身份库上限，超出按 `last_seen` 淘汰最旧 |
| `LAB_MONITOR_RETENTION_DAYS` | `30` | 告警/截图保留天数（仅在启动时执行一次清理） |
| `LAB_MONITOR_RTSP_OPEN_TIMEOUT_MS` / `_READ_TIMEOUT_MS` | `10000` | RTSP 连接/读取超时 |
| `LAB_MONITOR_FRAME_RATE_CAP` | GPU `10` / CPU `15` | 取帧上限 fps，覆盖设备默认档（`main.py:_env_positive`）。非法或非正值只打 warning 并回落默认 |
| `LAB_MONITOR_REID_EVERY_N` | GPU **`1`** / CPU `15` | 每个 track 每 N 个处理帧提一次 ReID 特征。**必须和实际达到的帧率一起调**，见「性能档位」下的缓冲填充算式。GPU 档已按 22 路受控实验从 10 定到 1（R=10 时一半轨迹攒不满 8 帧缓冲，见 `outputs/reports/reid_sampling_sweep.json`） |
| `LAB_MONITOR_MJPEG_FPS` | GPU `30` / CPU `15` | MJPEG 推流轮询频率 |
| `LAB_MONITOR_LEAVE_GRACE_FRAMES` | `12` | track_id 从 tracker 输出里连续缺席多少个**处理帧**才算人员离场（`src/pipeline.py:39-62`）。上限钳到 30（= `track_buffer`），设 `1` 可回退到改造前"当帧即判离场"的行为 |
| `LAB_MONITOR_INTRUSION_COOLDOWN` | `120` | 同一（相机, 身份, 围栏）的 INTRUSION 复报间隔秒数（`src/alerter.py:27`）。**这只是复报间隔，不是驻留判定**——按"进入/离开围栏"配对的状态机尚未实现 |
| `LAB_MONITOR_PROCESS_MAX_WIDTH` | `960` | 运行时解码后缩放上限（像素）。本地素材已离线转码到 960 宽，不触发；RTSP 在线流靠它兜底（`main.py:252`）。要关闭设一个大于所有源宽度的值（如 4096） |
| `LAB_MONITOR_REID_THRESHOLD` | `0.68` | ReID 匹配阈值（`src/reid_config.py` 的 `REID_MATCH_THRESHOLD`，2026-09 用自动标注集标定）。详见「常见开发任务」 |
| `LAB_MONITOR_REID_RATIO` | `0.85` | Ratio Test 判歧义阈值（`second_sim / best_sim` 超过此值拒绝匹配） |
| `LAB_MONITOR_RATIO_BYPASS_SIM` | `0.95` | 高相似度旁路：Top-1 相似度 ≥ 此值时跳过 Ratio Test，防"两个候选都是同一人的重复注册"造成的死亡螺旋。设 `≥1.0` 等价关闭旁路 |
| `LAB_MONITOR_REID_WEIGHTS` | `market1501` | OSNet 权重选择（注册表 `src/reid_config.py`：market1501 / msmt17 / dukemtmc） |
| `LAB_MONITOR_ALLOW_IMAGENET_FALLBACK` | `1` | OSNet 权重不可用时是否允许用 ImageNet 分类权重降级（会打 ERROR 明示身份比对不可采信） |
| `LAB_MONITOR_FEATURE_WINDOW` | `20` | 主特征聚合：有界窗口内的质量加权平均。**0 回退改造前的质量加权 EMA**（供对照实验，已知有不动点漂移问题） |
| `LAB_MONITOR_PERSONNEL_CROPS` / `_IDENTITY_SNAPSHOTS` | 空时按输出目录推导 | `/personnel-crops` 与 `/identity-snapshots` 静态挂载的根目录；给临时/演示库 seed 时用来隔离头像与抓拍缓存 |

### 测试与验证

**stdlib `unittest`（venv 里没装 pytest，别写 pytest 专有语法）**，`tests/` 下 25 个 Python 文件共约 320 个用例（全量 359 例含参数化展开，约 57 秒跑完）；另有 3 个纯 Node 的前端用例（无浏览器依赖，自定义 `check()` 框架 + DOM 桩 + fetch 打桩）：

```bash
# 全量
./.venv/Scripts/python.exe -m unittest discover -s tests -t .

# 单个文件 / 单个类 / 单个用例（点号路径，不是文件路径）
./.venv/Scripts/python.exe -m unittest tests.test_core_logic -v
./.venv/Scripts/python.exe -m unittest tests.test_stream_security.StreamSecurityTest
./.venv/Scripts/python.exe -m unittest tests.test_model_pool.ModelPoolTest.test_borrow_returns_instance

# 前端（Node，无依赖）：MJPEG 连接管理器（并发上限/轮转/冻结/后台断流）
node tests/test_stream_manager_frontend.mjs
# 人员档案库弹窗（姓名检索/SIM 徽标/注册照上传/以图搜人）
node tests/test_personnel_frontend.mjs
# 轨迹站点图（站点合并/区域识别/正交路由；直接断言仓库里的真实 topology.json + camera_map.json，
# 配置或站点判据一漂移就失败）
node tests/test_traj_graph_frontend.mjs
```

> **注意**：`src/db.py` 尾部的全局单例已**惰性化**（PEP 562 `__getattr__`）：`from src.db import Database` 只取类、不连库（脚本/测试的安全路径）；`from src.db import db` 或属性访问 `src.db.db` 才会创建并连上生产库 `outputs/lab_monitor.db`。测试一律用临时库注入；`server.py` 统一经 `_get_database()`（store → gallery → 全局单例）取库。

补充人工验证：
1. 运行 `./.venv/Scripts/python.exe main.py` 查看控制台是否报错
2. 访问 `http://localhost:8000` 检查 Web 界面是否正常
3. 检查 `outputs/alerts.jsonl` 是否正常写入告警日志

## 核心架构

### 多线程架构

```
main.py (主线程)
  ├─ FastAPI Web Server Thread (daemon)
  │   └─ 提供 REST API、MJPEG 流、WebSocket
  ├─ Alert Ticker Thread (0.5秒轮询)
  │   └─ AlertManager.tick() → 检测滞留/MISSING_PERSON/SCENE_EXIT
  └─ N × CameraPipeline Threads (每个摄像头一个)
      └─ 读帧 → 检测 → 跟踪 → ROI越界(INTRUSION) → ReID → 更新身份库 → 推送到 FrameHub
```

### 性能档位（GPU / CPU 自动切换）

`main.py` 启动时用 `torch.cuda.is_available()` 选一整套参数（`main.py` 的 `# ---- 性能配置` 段）。`frame_rate_cap` / `reid_every_n` / `mjpeg_fps` 现在可用环境变量覆盖，调优时不必改代码；`detect_every_n` 与 `jpeg_quality` 仍需改代码：

| 参数 | GPU | CPU | 影响 |
|------|-----|-----|------|
| `detect_every_n` | 1 | 3 | YOLO 跳帧；非检测帧复用上次 tracker 输出（**不能给 ByteTracker 传空列表**，会瞬间清空所有 track） |
| `reid_every_n` | **1** | 15 | 每个 track 每 N 帧才提一次 ReID 特征。**这个值不能单独调**：`ReIDValidator` 要攒满 `buffer_size=8` 才会 `register_if_new()`，填满耗时 = `8 × reid_every_n / 实际fps` 秒，而单相机内轨迹时长中位只有 **7.1s**（实测）。2026-09-13 的 22 路 90s×3 配置受控实验（`scripts/measure_reid_sampling.py` → `outputs/reports/reid_sampling_sweep.json`）：R=10 时 50% 轨迹攒不满（第 8 特征要 25.9s）；**R=1 时 100% 轨迹 3.4s 攒满，帧率只掉 9%**（2.48 vs 2.73 fps——瓶颈是 GIL，ReID 的 CUDA 推理释放 GIL，"R 调小白烧算力"的旧假设被实测推翻）。要保守可设 3（83% 注册、8.2s） |
| `frame_rate_cap` | **10** | 15 | `_read_loop` 里 sleep 补齐到该帧率。GPU 档 2026-08-25 从 30 降到 10：素材全是 25fps，抽到 10fps 时 ByteTrack 相邻帧 IoU 中位 0.87、关联失败率 0.1%，追 25fps 纯属浪费 |
| `mjpeg_fps` | **30** | 15 | MJPEG 推流间隔，取略高于 `frame_rate_cap` 以免节拍抖动叠加延迟 |
| `jpeg_quality` | **50** | 65 | `FrameHub` 编码质量 |

同时做了**线程钳制**：`cv2.setNumThreads(1)` + `torch.set_num_threads(2)`。22 路解码 + 池化并发推理已经能打满 CPU，OpenCV/torch 内部线程池再抢核只会加剧上下文切换。**`LAB_MONITOR_MODEL_POOL=1` 只回退池大小，不回退这两行。**

> **当前实测吞吐（2026-09-13，22 路 / RTX 3090 / 14 核，R=1 配置，受控实验无 Web 负载）**：每路均值 **2.48 fps**（R=10 对照为 2.73）。瓶颈仍是**单进程 GIL**，不是 GPU（GPU 利用率 35~40%，13 个核空转）：单线程 detect 天花板就是 56 帧/s（17.6ms/帧，其中 inference 14.3ms），22 路挤一把 GIL 后远低于此。**加大模型池无用**（2026-08-25 实测池 2/4/6 都是 38 帧/s，池=1 才掉到 22.9），**降 `frame_rate_cap` 也无用**（限速 sleep 压根不触发）。要突破必须减少单进程内的线程数 —— 见 `docs/` 里的优化方案。

### 启动期的两个降级路径

两者都**不会让服务起不来**，只会静默丢功能，排查时优先看启动日志：

1. **拓扑校验失败** → `main.py:308-321` 捕获 `TopologyValidationError`，换成空拓扑继续启动，**MISSING_PERSON 全线失效**（`next_hops()` 恒为空 → `watch()` 直接 return）。改好 `config/topology.json` 后必须重启。
2. **视频源文件不存在** → `main.py:186-197` 直接从 `sources` 里剔除该相机，只打一条 warning。前端网格里那一路会连卡片都没有。

### 全局共享组件

以下对象在 `main.py` 中实例化，被所有 pipeline 线程和 Web 服务共享：

| 组件 | 文件 | 作用 | 线程安全 |
|------|------|------|----------|
| `PooledDetector` / `PooledReIDExtractor` | `src/model_pool.py` | 有界模型实例池（默认 GPU `min(2, 源数)`），并发度 = 池大小 | 队列借还，`try/finally` 保证归还；池空时阻塞形成背压 |
| `IdentityStore` | `src/identity_store.py` | 全局 ReID 身份库（含 feature_bank 与 ReIDMetrics） | 内置锁保护（含 register_if_new 原子操作） |
| `FrameHub` | `src/frame_hub.py` | JPEG 帧缓冲区（供 MJPEG 流消费） | 内置锁保护 |
| `AlertManager` | `src/alerter.py` | 告警逻辑与历史记录 | 锁内原子修改状态 |
| `AlertBroadcaster` | `src/alerter.py` | WebSocket 告警推送 | 线程安全（asyncio.Queue / sync queue） |
| `TransitCalibrator` | `src/calibrator.py` | 相机间穿越时延统计 | 内置锁保护 |
| `Database` | `src/db.py` | SQLite 数据库持久化（`outputs/lab_monitor.db`，见 `src/db.py:17`） | 单例模式与独立连接 |
| `LabelRenderer` | `src/label_render.py` | 画面人员标签（支持中文实名：PIL 位图按文本缓存，ASCII 走 cv2 快路径，无字体优雅回退） | 每路 pipeline 独立实例，无共享 |
| `Floorplan` | `src/floorplan.py` | 平面图点位映射（`config/camera_map.json`） | `RLock` 保护，按 mtime 热重载 |
| `PersonnelGallery` | `src/personnel.py` | 实名人员底库（注册照 1:N 自动命名 + CRUD）。空库时 `match()` 恒为 None，对主流程零影响 | 内存索引由 `reload()` 重建，写库即重载 |

### ReID 身份识别流程

```
CameraPipeline (src/pipeline.py)
  ↓
检测到新人员 → 提取特征向量 (ReIDExtractorOSNet — OSNet-x0.25, 512维)
  ↓
积累多帧缓冲 (ReIDValidator) → 计算平均特征
  ↓
先查实名底库 (PersonnelGallery.match — 命中已知人员直接复用其绑定的 global_id)
  ↓
查询身份库 (match_feature_detailed + Ratio Test，均在公共分量中心化坐标系下)
  ↓
匹配成功(3帧一致) → 返回已有 global_id
匹配失败(缓冲区满) → IdentityStore.register_if_new() 原子性查重+注册 (支持多姿态 feature_bank)
  ↓
更新特征 (有界窗口 FEATURE_WINDOW_SIZE=20 内质量加权平均，自动维护 feature_bank)
```

**关键参数**：
- ReID 模型：**OSNet-x0.25**（512维），通过 `build_reid_extractor()` 工厂函数实例化（三级降级：OSNet+ReID 权重 → OSNet+ImageNet 权重（降级，比对不可采信）→ ResNet50 2048维）
  - **当前加载的是 Market-1501 ReID 专用权重**（批次一换装，`feature_space` 形如 `osnet-x0.25-market1501:512`），权威配置在 `src/reid_config.py`
- 相似度阈值：`REID_MATCH_THRESHOLD`（默认 **0.68**，单一来源 `src/reid_config.py`，可用环境变量 `LAB_MONITOR_REID_THRESHOLD` 覆盖；旧文档所称"0.75 散落 5 处"已单源化，勿再硬编码）
- Ratio Test：`second_sim / best_sim > 0.85`（`REID_RATIO_TEST`）时拒绝歧义匹配（`src/reid.py:match_feature_detailed`；**按身份去重后再做 Ratio**，否则同一身份占满 best/second 必判歧义 → 身份表膨胀）
- 高相似度旁路：`REID_RATIO_BYPASS_SIMILARITY`（0.95）——Top-1 ≥ 0.95 时跳过 Ratio，防"两个候选都是同一人的重复注册"造成的死亡螺旋（新观测永远拿不到身份、旧重复身份永不回收）
- 多帧确认：连续 3 帧匹配同一 ID 才确认（`confirm_frames=3`）
- 注册防重：`register_if_new()` 在锁内原子执行查重+注册，防多摄像头并发重复注册；Ratio 判歧义时返回 `ambiguous` 而**不静默归并 Top-1**
- 多姿态特征库：`feature_bank` 保存最多 5 个差异明显特征（入库门限质量 ≥0.02、相似度 < 0.92），提升大视角变化下的检索召回率
- ⚠️ **公共分量中心化是匹配路径的硬前提**：OSNet 单位特征彼此余弦高达 0.98 的根因是公共分量；gallery 与 center 必须通过 `IdentityStore.build_match_context()` **原子配套**取用（拆两次 getter 会被别的线程注册新身份搞成"query 用新中心、gallery 用旧中心"的静默错配）。**`PersonnelGallery.match()` 必须传 context**——底库阈值在中心化空间标定，不传等于在原空间用错阈值（实测误越阈 54.9% vs 1.7%）

### 告警触发机制

`AlertManager.tick()` 每 0.5 秒扫描所有活跃身份，支持以下告警类型：

1. **MISSING_PERSON (超时/失踪)**：人员离开某相机后，未在预期时间窗口内到达拓扑相邻相机
2. **INTRUSION (即时越界/围栏)**：人体框的 **4 个采样点**（底边中心/底边左角/底边右角 + 身体中心，`src/pipeline.py:667-672`）任一落入 ROI 电子围栏即触发（`cv2.pointPolygonTest`），不是只判脚下一点
3. **SCENE_EXIT (全域消失)**：人员从所有摄像头消失超过设定期限（默认 300 秒）
4. **CROWD_DENSITY (聚众预警)**：区域内检测到的人数超过设定阈值（默认 5 人）

告警分两阶段触发：
- **WARNING**（70% deadline）：控制台预警，WebSocket 推送紫色/黄色标志，不触发外部通知
- **ALERT**（100% deadline / INTRUSION）：全量告警 + WebSocket 推送 + 日志/数据库持久化

> **外部通知（notifier）只有两条路径**：MISSING_PERSON 的 ALERT（`src/alerter.py:380`）与 SCENE_EXIT（`src/alerter.py:461`）。**INTRUSION 与 CROWD_DENSITY 只写日志 + WebSocket，不发邮件/外部通知。**

> 告警逻辑在 `src/alerter.py` 中实现，时间窗口由 `config/topology.json` + `TransitCalibrator` 动态校准。

### 业务能力：身份档案 / 视频检索 / 轨迹回放（2026-09-12 后逐步交付）

在"识别出人"之上叠加的三层能力，对应 `docs/PLAN_2026-09-12_identity_search_resolution.md` 的三项能力：

| 模块 | 职责 | 关键不变量 |
|------|------|-----------|
| `src/personnel.py`（PersonnelGallery） | 实名底库：注册照 1:N 自动命名（"认出熟人"）、档案 CRUD | `match()` **必须传 `build_match_context()`**；检索先按身份各自聚合再合并（SQL `IN(...)` 会把多个身份的相机序列拼成假周期） |
| `src/search.py` | 视频检索：按 gid 或实名档案聚合出现片段，`best_crop` 选最佳裁图；以图搜人 Top-K | 离线检索返回**排序**而非二值判定，阈值语义独立于实时匹配 |
| `src/snapshots.py` | 跨相机抓拍裁图（唯一实现，server 与脚本共用） | 只用低清素材 + `CAP_PROP_POS_MSEC` 定位；侧车 `_meta.json` 只存文件名，换部署目录后用当前 out_dir 重建路径 |
| `src/trajectory.py` | 轨迹分段与循环折叠（详见「注意事项 16」） | 周期检测用**差分判据**；截断前做覆盖校验，失败降级指纹法 |
| `src/mock_personnel.py` + `scripts/seed_personnel_mock.py` | 模拟行人姓名/档案数据 | `source` 必须标 `synthetic`，前端强制打 SIM 徽标——合成特征近似正交，用它演示检索精度等于造假 |
| `scripts/`（22 个脚本，本地保留不入库） | ReID 标定/诊断/修复链（标注→标定→A/B 实验→归并/重建）、媒体探测（ffprobe/实测帧数）、转码、数据播种 | 详见各脚本头注释；`rebuild_identity_db.py` 2026-10-07 重建过一次生产身份库 |
| `videos_low/`（960×540 CRF28 转码副本，gitignore） | 检索/回放/抓拍的实际素材：原片容器时间戳损坏，seek 会落到无关画面；体积 1.5GB→29MB 可归档 | `sources.json` 指向原片（`sources_high.json` 是切回高清的备份）；`video_assets` 表按 `UNIQUE(camera_id, rel_path)` 同时登记两行 |

数据流：pipeline 每次观测写 `identity_appearances` 增量行（含视频内坐标 `asset_id`/`video_frame`/`video_ts`，解决循环播放下墙钟无法反查源视频位置）→ 检索/轨迹端点读 `video_assets` 实测帧数索引 → 在低清片上定位出片。

## Web 服务（server.py）

模块级全局变量（`_frame_hub` / `_identity_store` / `_pipelines` / `_topology` ...）由 `main.py` 调 `init_server()` 注入，**导入 `server` 不等于服务可用**，单测里直接 `TestClient(app)` 时这些是 `None`（各路由都做了 None 分支）。

⚠️ **两个中间件的定义顺序不能动**（`server.py:303-307` 有注释说明）：Starlette 的 `add_middleware` 是 `insert(0)`，**后注册的在最外层**。`guard_request` 必须定义在 `require_basic_auth` **之后**，才能先于鉴权执行（实际顺序 guard → auth → 路由）。新增中间件时想清楚要插在哪一层。守卫规则：Host 不在白名单 → **421**；写方法 Origin 跨源 → **403**；写方法缺 `X-Lab-Monitor-Request: 1` 头 → **403**。绑定通配地址（0.0.0.0 等）且未配 `LAB_MONITOR_ALLOWED_HOSTS` 时 Host 校验**整体关闭**（有 warning）。

接口分组（全部在 `server.py`）。✍ = 写方法，需守卫头 + 同源 Origin + Host 白名单：

| 分组 | 端点 |
|------|------|
| 页面 / 流 | `GET /`（每次请求读盘托管 index.html）、`GET /stream/{cam_id}`（MJPEG，增量推送 + 5s 心跳，空白名单 fail-closed 404）、`WS /ws`（告警推送 + 状态请求-应答，`?since=<alert_id>` 断线续传） |
| 身份 / 轨迹 | `GET /api/identities`、`GET /api/identities/{global_id}`（详情 + 压缩轨迹 + 实名反查）、`GET /api/identities/{global_id}/trajectory`（**轨迹回放**：`start`/`end`/`camera`/`max_points`≤20000/`min_gap_s`/`split_gap_s`/`collapse_loops`，默认开）、`GET /api/identities/{global_id}/snapshots`（跨相机抓拍序列，`force`/`max_cameras`） |
| 检索 | `GET /api/search/person`（按 gid/person_id 检索视频片段，`split_gap_s`/`collapse_loops`）、`POST /api/search/by-image`（以图搜人 Top-K，✍，`top_k`≤20） |
| 实名档案 | `GET /api/personnel`（`q`/`department`/`source`/分页）、`GET /api/personnel/{person_id}`、`GET /api/personnel/{person_id}/activity`、`GET /api/personnel/{person_id}/identities`、`POST /api/personnel`（✍）、`PATCH /api/personnel/{person_id}`（✍）、`DELETE /api/personnel/{person_id}`（✍，名下身份解绑）、`POST /api/personnel/{person_id}/photos`（✍，注册照：检测→提特征→入库）、`POST /api/identities/{global_id}/bind`（✍）、`DELETE /api/identities/{global_id}/bind`（✍） |
| 状态 / 统计 | `/api/status`、`/api/metrics/reid`、`/api/alerts`、`/api/system/metrics`、`/api/stats`（校准统计）、`/api/floorplan`（平面图包）、`/api/roi`、`/api/topology`（注入优先、读文件兜底） |
| 历史 / 导出 | `/api/alerts/history`（分页，`limit`≤500/`offset`/`risk_level`/`camera_id`/`global_id`）、`/api/alerts/export`（CSV，上限 5 万行，`X-Export-*` 头） |
| 媒体 | `/api/assets`（视频资产索引，含实测帧数）、`/media/{camera_id}`（回放出片，支持 HTTP Range，`asset_id`/`download`） |
| 写接口（需守卫头） | `POST /api/roi`（3~64 顶点、坐标 `[0,1]`，空 polygon=删除；写盘后遍历 `_pipelines` 热刷新）、`POST /api/topology`、`POST /api/admin/shutdown`（仅回环） |
| 运维 | `GET /healthz`（`start.ps1` 靠它判断启动成功） |

> **`/ws` 协议**：客户端发 `{"type": "status"|"identities"|"reid"|"calib"}` 请求状态（把原本的 REST 轮询并入 WS，连接预算只剩 5 MJPEG + 1 WS）；服务端回 `{"type": <同名>, "data": {...}}`；告警推送是**无 type 字段**的告警对象（前端按有无 `type` 区分应答与告警）。WS 也过 Basic 鉴权。

**组件未注入时的降级**：各注入组件为 `None` 时（裸 `TestClient(app)`）所有路由有 None 分支——空载荷降级（`/api/status` → `{"cameras": []}` 等）或显式 503；`/stream/{cam_id}` 相机白名单为空时 fail-closed 404；`_get_database()` 按 store → gallery → 全局单例逐级兜底取库。生产部署（`main.py` 装配注入）不受影响。

静态挂载：`/static`、`/screenshots`、`/floorplan`（`assets/floorplan/`，客户图纸不入 Git）、`/personnel-crops`（`LAB_MONITOR_PERSONNEL_CROPS`）、`/identity-snapshots`（`LAB_MONITOR_IDENTITY_SNAPSHOTS`）。

## 配置文件

所有配置使用 JSON 格式（无 `.env` 文件）：

### config/sources.json
超算中心 L2 层实拍录像，原始素材 **32 路**：`reg_01..reg_10`（常规巡检环）+ `rnd_01..rnd_22`（随机路线）。相机 ID 即 key，值为视频路径或 RTSP 地址：
```json
{
  "reg_01": "videos/常规路线/L2东侧走廊南北向南_20260731141700-20260731142100_1.mp4",
  "reg_02": "videos/常规路线/L2东侧走廊南南向北_20260731141700-20260731142100_1.mp4",
  "rnd_01": "videos/随机路线/L2东侧走廊北北向南_20260731143025-20260731143100_1.mp4",
  "cam_rtsp": "rtsp://192.168.1.100:554"
}
```

> **当前实际启用 22 路**，已摘除 10 条：
> - **9 路源文件损坏**（`reg_03`/`reg_04`/`reg_07`/`reg_09`、`rnd_09`/`rnd_13`/`rnd_14`/`rnd_15`/`rnd_20`）：`isOpened()` 返回 True 但首帧 `read()` 失败（H.264 缺 PPS / HEVC PPS out of range，元数据 fps=90000、frame_count=INT64_MIN）
> - **1 路重复**：`rnd_03` 与 `rnd_04` 的 MD5 完全相同，是同一份素材

> ⚠️ **改 `sources.json` 必须同步改 `topology.json`**：`CameraTopology` 用 `allowed_camera_ids=set(sources)` 校验，拓扑里出现未配置的相机 ID 会抛 `TopologyValidationError`。**它不会让服务崩**——`main.py` 捕获后降级为空拓扑继续启动，代价是 MISSING_PERSON 静默全线失效（见「启动期的两个降级路径」）。所以这类错配**只能在启动日志里看到**。

### config/topology.json
按起始摄像头 ID 映射下游相邻摄像头列表，定义相机间的邻接关系和预期穿越时间（秒）。**实际按巡检路线的点位顺序串联配置**：共位反向相机对（同一走廊两个朝向）用 `5 / 10` 秒，走廊相邻点位按实际步行距离取 `20~60` 秒；因中间点位视频损坏被摘除而形成的跨区长边用 `120~180` 秒（容忍 `60~90` 秒）：
```json
{
  "reg_02": [
    {
      "next": "reg_01",
      "expected_seconds": 5,
      "tolerance_seconds": 10
    }
  ],
  "reg_01": [
    {
      "next": "reg_05",
      "expected_seconds": 180,
      "tolerance_seconds": 90
    }
  ]
}
```

### config/camera_map.json

**摄像头在楼层平面图上的点位，仅供「路线可视化」使用**。按 system_id（`reg_XX` / `rnd_XX`）映射：

```json
{
  "rnd_19": {
    "plan_id": "IP79",
    "desc": "L2高性能机房04通道东南向北",
    "map_xy": [0.43, 0.52],
    "facing_deg": 90,
    "fov_deg": 80
  }
}
```

- ⚠️ **`map_xy` 与 `roi.json` 的 polygon 是两套坐标系，别混用**：`map_xy` 是「楼层平面图」上的归一化点位；ROI 是「相机画面内」的归一化多边形（用于 INTRUSION）
- `map_xy` 为 `null` 表示该相机**尚未标注**，接口照常返回但坐标是 null，前端应跳过而不是画到原点
- 加载器 `src/floorplan.py` 会清洗非法值（裁剪到 `[0,1]`、拒绝 NaN/Inf/非数值），文件损坏时**保留上一次的有效映射**
- 手改文件后无需重启（`Floorplan.maybe_reload()` 按 mtime 热重载）
- 标注工具：`outputs/dev/calibrate_floorplan.html`（单文件，浏览器打开点图落点后导出 JSON）
- **现有坐标来自 `scripts/derive_camera_map.py` 的自动推导**（按 `desc` 命名规则 + topology 共位对，把 22 台合并为 15 个物理点位），**不是图纸标注**；`facing_deg` 全为 0、`plan_id` 全为空，要真实朝向需手工补

### config/roi.json

**INTRUSION 电子围栏的唯一来源**。按相机 ID 映射多边形列表，坐标是**归一化的 `[0,1]` 浮点**（`pipeline.py` 里再乘画面宽高）：

```json
{
  "rnd_16": [
    {
      "name": "机房02通道尽端受限区",
      "polygon": [[0.395, 0.015], [0.565, 0.015], [0.585, 0.255], [0.375, 0.255]]
    }
  ]
}
```

- **当前有 2 条围栏**：`rnd_16`「机房02通道尽端受限区」与 `rnd_01`「核心机房防护区」（后者 2026-10 提交 `52b69b4` 加入）；其余 20 路没有 ROI，永远不会报 INTRUSION
- **热更新**：`POST /api/roi` 写盘后会遍历 `_pipelines` 调 `reload_rois()`，不用重启。手改文件则必须重启
- 写接口校验：相机 ID 必须在已配置的 `_pipelines` 里、顶点 3~64 个、坐标 `[0,1]`（`POST /api/roi`，`server.py:460` 附近）
- `pipeline._sanitize_rois()` 会二次清洗（裁剪到 `[0,1]`、丢掉非法 ROI）。历史上字符串坐标会让 `int(p[0] * w)` 抛 `ValueError` **打死整条摄像头线程**，所以两侧都校验
- 传空 `polygon` 即删除该相机的围栏

### config/notify.json
告警通知渠道配置（支持 `console` 和 `email`）：
```json
{
  "console": {
    "enabled": true
  },
  "email": {
    "enabled": false,
    "smtp_host": "smtp.qq.com",
    "smtp_port": 465,
    "use_ssl": true,
    "username": "your@qq.com",
    "password": "your_auth_code",
    "from": "your@qq.com",
    "to": ["admin@example.com"]
  }
}
```

## 运行时输出

`outputs/` 目录（自动创建，`.gitignore` 已忽略；仓库中**没有** `data/` 目录）：
- `outputs/lab_monitor.db`：SQLite 数据库文件，持久化身份档案与告警日志（路径定义见 `src/db.py:17`）
- `outputs/alerts.jsonl`：追加式 JSONL 告警日志文件
- `outputs/transit_stats.json`：穿越时延统计（用于异常检测自适应校准）
- `outputs/screenshots/`：告警截图（按 `alert_id.jpg` 命名）
- `outputs/reports/`：标定与 A/B 实验产物（`reid_threshold_sweep.json`、`reid_sampling_sweep.json`、`feature_window_ab.json`、`resolution_ab*.json` 等，调参结论的权威来源）
- `outputs/personnel_crops/`、`outputs/identity_snapshots/`：注册照头像与跨相机抓拍缓存（前者供 `/personnel-crops`，后者含 `<gid>_meta.json` 侧车）
- `outputs/dev/`：开发辅助工具（如 `calibrate_floorplan.html` 标注器）
- `outputs/backups/`：库维护备份（如身份库重建前）
- `outputs/server.log`：Python logging 写入，`main.py` 用 `RotatingFileHandler` 轮转（单个 64MB，保留 5 份），无需手工清理
- `outputs/server.stderr.log` / `outputs/server.stdout.log`：`start.ps1` 重定向的 stderr / stdout，只兜底捕获 C 层（FFmpeg）输出与 `print`；启动时若超过 64MB 会被自动改名 `.old`
- `outputs/server.pid`：真实服务进程 PID，由 `main.py` 自己写入（`start.ps1` 启动前先清残留），`stop.ps1` 以监听端口为主、PID 文件为辅

## 常见开发任务

### 修改检测模型（YOLOv8）
- YOLOv8 权重文件：`yolov8n.pt`（6.5MB，在工作区里，但**没有**入 Git——`.gitignore` 的 `*.pt` 把它排除了）。新克隆的仓库首次运行时由 ultralytics 自动下载同名权重
- 检测器实例化：`src/detector.py:12`（`PersonDetector.__init__`），`YOLO(model_name)` 在 `:19`；实际传参在 `main.py:275`（`conf_thresh=0.4`）
- 要换模型（如 yolov8s.pt）：改 `main.py:275` 的 `model_name`，或替换同名权重文件

### 修改 ReID 模型
- 当前：**OSNet-x0.25**（512维，自动在首次运行时下载）
- 工厂函数：`src/reid.py` 中的 `build_reid_extractor()`，优先 OSNet，OSNet 不可用时回退 ResNet50（2048维）
- 切换为其他 torchreid 模型（如 osnet_x1_0）：修改 `ReIDExtractorOSNet.__init__` 中的 `name` 参数
- **注意**：切换模型后特征维度可能变化（512→2048），需清空 `outputs/transit_stats.json` 重新校准

### 调整 ReID 匹配阈值
- 阈值已**单源化**到 `src/reid_config.py` 的 `REID_MATCH_THRESHOLD`（默认 0.68，2026-09 由 `config/labeled/pairs.json` 自动标注集标定，P=0.913 / R=0.639），所有调用点（reid / pipeline / identity_store / personnel）都从它导入；临时调整用环境变量 `LAB_MONITOR_REID_THRESHOLD`，不要在任何调用点写死数字
- Ratio Test 阈值：`src/reid_config.py` 的 `REID_RATIO_TEST`（0.85，env `LAB_MONITOR_REID_RATIO`）；高相似度旁路 `REID_RATIO_BYPASS_SIMILARITY`（0.95，env `LAB_MONITOR_RATIO_BYPASS_SIM`）
- 权重选择：`DEFAULT_REID_WEIGHTS`（market1501，env `LAB_MONITOR_REID_WEIGHTS`），注册表含 msmt17 / dukemtmc
- 阈值越高 → 匹配越严格 → 更易产生新身份（适合外貌差异大的场景）
- **换权重/换模型后**特征维度或 `feature_space` 变化，`IdentityStore._restore()` 会按 `feature_space` **全等**校验跳过旧身份（换权重后旧身份静默不加载，属预期保护），需清空 `outputs/transit_stats.json` 重新校准；必要时用 `scripts/rebuild_identity_db.py` 重建身份库

### 修改告警时间窗口
- 基础预期时间：`config/topology.json` 中各边的 `expected_seconds` 和 `tolerance_seconds`
- 自动校准：运行足够多样本（≥5条）后 `TransitCalibrator` 会覆盖静态配置
- 最小容忍时间：`src/calibrator.py:217` 的 `max(Z_SCORE * std, 5.0)`（`Z_SCORE = 1.65` 定义在 `:22`）

### 添加新的告警类型
1. 在 `AlertManager.tick()` 中添加检测逻辑
2. 调用 `self._build_alert(entry, stage="ALERT")` 构造告警对象
3. 前端会通过 WebSocket 实时接收（无需修改前端代码）

## 前端代码

**已模块化**（不再是单文件）：
- `static/index.html`：370 行，只剩结构与资源引用
- `static/css/`：10 个文件（`main.css` / `variables.css` / `layout.css` / `utilities.css` + `components/` 下 6 个：alert-list / cards / modals / personnel / roi-canvas / traj-graph）
- `static/js/`：11 个文件，原生 **ES module**（`app.js` + `modules/` 8 个：grid / modals / personnel / roi / stream_manager / theme / traj_graph / websocket + `utils/` 2 个：api / formatter）

- 无构建流程，由 FastAPI 在 `server.py:437` 的 `index()` 路由每次请求直接读盘托管（改 HTML 不用重启，但 `?v=` 仍决定 CSS/JS 是否走缓存）
- 使用原生 JavaScript + WebSocket + MJPEG `<img>` 标签；模块间无事件总线，全部显式 import/export 调用（数据流主干：`websocket.js` 推送 → `grid.js` 渲染网格 / `modals.js` 弹窗；`app.js` 绑定全部 DOM 事件 + 三个轮询兜底）
- **`modules/stream_manager.js` 管理 MJPEG 长连接**：浏览器对同域并发连接上限为 6，22 路全量建流会把连接池占满（只有约 6 路能出图，REST 轮询也会挨饿）。它用 `IntersectionObserver` 只给视口内的卡片建流、**同时建流上限 5**（`MAX_CONCURRENT_STREAMS`）、超限时整批轮转（6s）、断流前把最后一帧冻结成占位图、故障 5s 冷却重试；焦点大屏常驻，标签页切后台全部断流，弹窗有独立流时挂起网格。**改网格/弹窗相关代码时注意调用 `resetStreamRegistry()` / `registerStreamImage()` 的时机**，否则会出现"卡片可见但永不建流"或连接泄漏。纯 Node 用例：`node tests/test_stream_manager_frontend.mjs`
- **改任何 CSS/JS 后必须同步 bump `static/index.html` 里的 `?v=` 版本号**（当前 CSS `?v=12.1` / `app.js?v=12.5`；`main.css` 的 `@import` 子路径也带版本号，三处要一起改）。注意 `app.js` 里的 `import './modules/*.js'` **不带版本号**，改子模块后需要 `Ctrl+F5` 硬刷新才能看到效果
- 只改前端资源时刷新浏览器即可（无需重启后端）

## 文档与审计台账

- `docs/CODE_AUDIT_2026-07-28.md`：**代码里 `P0-1` / `P1-4` / `F5` / `F10` / `P2-13` 这类注释编号的字典**。41 组确定性缺陷 + 7 组条件性风险的编号台账，看不懂某处防御性代码为什么存在就查这里
- `docs/SYSTEM_PRINCIPLES.md`：系统原理讲解（面向汇报，非实现细节）
- `docs/deployment.md`：完整部署 / 排错手册，比本文件详细
- `docs/PLAN_2026-09-12_identity_search_resolution.md` + `TODO_2026-09-12_worklist.md`：三项能力（身份识别 / 视频检索 / 分辨率降低）的实施方案与批次工作清单（批次一~五的进度看板）
- `docs/PLAN_2026-09-12_personnel_mock_data.md`：模拟行人数据与人员档案检索设计（real/synthetic 两档数据的约束）
- `docs/技术总结_跨摄像头行人检索.md`：检索能力技术总结（公共分量中心化、判据、精度标定）
- `docs/方案_跨相机轨迹可视化演示.md`：轨迹可视化方案（2026-09-15 转为实施记录，含三项前置阻塞的分析）
- `.plans/scene-exit-alert.md`：SCENE_EXIT 的设计记录（已入库）
- `docs/*.xlsx`（**已 gitignore**）：现场点位表，含真实摄像头 IP 与图纸点号，**是 `config/topology.json` 的唯一依据**。仓库公开，不要提交；本地丢了就只能从 `常规路线.7z` / `随机路线.7z` 里取

> ⚠️ **`docs/`、`.plans/`、`config/labeled/pairs.json` 已转为本地保留**（提交 `18823a6` / `238d796`：远端已移除，commit history 仍可见）。它们是设计记录与标定集，不属于代码资产；新克隆的仓库没有这些文件，引用编号时注意这一点。

## Git 工作流

- **主分支**：`main`（当前分支，默认推送目标）
- **提交规范**：使用语义化前缀 `feat:` / `fix:` / `docs:` / `style:` / `refactor:`
- **视频文件**：放在 `videos/` 目录，已在 `.gitignore` 中排除（不要提交大文件）
- **模型权重**：`.gitignore` 里 `*.pt` 排除了全部权重，`yolov8n.pt` **不在仓库里**（ultralytics 首次运行自动下载）。换更大模型也别提交，保持自动下载或走 Git LFS
- **大文件**：根目录 `常规路线.7z` / `随机路线.7z`（约 1.5GB）、`docs/*.pdf`、`docs/*.xlsx` 均已 gitignore

## 网络代理

开发机运行 Clash Verge 代理（`127.0.0.1:7897`）。如遇依赖下载失败：

```bash
# 临时启用代理（仅当前命令有效）
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 注意事项

1. **终端环境**：Windows 10。Git Bash 与 PowerShell 5.1 都可用，但**分工固定**：日常 `git` / `grep` / 跑 Python 用 Git Bash；`start.ps1` / `stop.ps1` 只能在 PowerShell 里跑。PowerShell 5.1 **没有** `&&` / `||` / 三元运算符，写 `.ps1` 时用 `; if ($?) { ... }`
2. **Python 解释器**：必须用 `./.venv/Scripts/python.exe`（系统 PATH 里的 `python` 是 Microsoft Store 存根，静默退出）。该解释器不认 `/c/...` 路径，传路径请用 `C:\...`；stdout 默认 GBK，脚本打印中文需包 `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`
3. **路径格式**：代码中使用正斜杠 `/`（Python 跨平台兼容），Git Bash 路径用 `/c/Users/...`
4. **依赖清单**：`numpy` / `opencv-python` / `psutil` 已在 `requirements.txt` 中；`torch` / `torchvision` 不在，需单独先装；`scipy` 已在 P1 优化中移除，不再需要
5. **OSNet 权重**：首次运行自动从 Google Drive 下载（~3MB），缓存至 `~/.cache/torch/checkpoints/`；网络不通时可配置代理后重试
6. **FastAPI 自动重载**：默认未启用，修改后端代码需手动重启服务
7. **视频循环播放**：本地 MP4 文件会自动循环（`pipeline.py` `_run_file()` 中检测到流结束后重新打开）
8. **appearances 类型**：`PersonRecord.appearances` 是 `deque(maxlen=200)`，不是 `list`，但支持 `list()` 转换和负索引访问
9. **`.ps1` 必须存成 UTF-8 with BOM**：PowerShell 5.1 否则按 GBK 解析中文注释，会吃掉注释后续的语句（已踩过坑）
10. **`outputs/server.pid` 现在是可信的**：`main.py` 启动时写入自己的真实 PID（venv 的 `python.exe` 是 launcher，`start.ps1` 拿到的 `$process.Id` 并非工作进程，因此不再由脚本写入）。`stop.ps1` 仍以监听端口为主依据、PID 文件为辅；手工排查用：`Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess`
11. **绝不要在 `_reset_stream_state()` 里重建 `PersonTracker`**（`src/pipeline.py:272-287` 有长注释）：`BYTETracker.__init__` 会调 `reset_id()`，而 `BaseTrack._count` 是**类级共享**计数器，重建会把**所有**摄像头线程的 `track_id` 一起归零 → 与其它路正在活跃的 id 撞号 → `_track_to_global` 把新来的人认成旧身份。旧轨迹交给 `track_buffer` 自然淘汰即可
12. **流重开（文件循环 / RTSP 重连）必须重置跟踪状态**：不重置时旧 `track_id` 会集体从 tracker 输出里消失，被 `_process_frame` 当成人员离场 → `watch()` → 一片 MISSING_PERSON 误报
13. **离场判定带宽限期（F2b）**：ultralytics 8.4 的 `BYTETracker._format_output()` 只返回 `is_activated` 的轨迹，一次 IoU 关联失败该 track 当帧就从输出里消失（内部仍按 `track_buffer=30` 帧保留、可复活）。所以 `_process_frame` **不能**看到 id 消失就判离场——那会误报 MISSING_PERSON 并清空攒了一半的 ReID 缓冲（`buffer_size=8` 永远填不满 → 无法注册身份）。现在改成连续缺席 `_LEAVE_GRACE_FRAMES`（默认 12）个处理帧才调 `_on_person_leave()`，状态记在 `_absent_streak` 里，`_prev_track_ids` 的语义也随之变成"在场（含宽限期内）"而不是"上一帧输出"
14. **ReID 特征落盘是节流的**：`IdentityStore` 对特征列做「累计 50 次更新或间隔 30 秒」节流（`src/identity_store.py:29-30` 的 `_FEATURE_FLUSH_*`），不调 `flush()` 就会静默丢掉最近一段滑动平均结果。`main.py` 的 `flush_identity_features()` 在关停路径上补写，新增退出分支时别漏掉它。历史教训：存储特征曾漂移成"谁也认不出"的 EMA 吸引子，2026-10-07 用 `scripts/rebuild_identity_db.py` **重建过一次生产身份库**——这正是主特征从 EMA 翻转为有界窗口（`FEATURE_WINDOW_SIZE=20`）的动因
15. **`demo.py` 不读真实录像**：它生成合成视频（矩形模拟人员移动）跑通检测→跟踪→ReID→告警链路，适合在没有素材或想快速验证改动时用
16. **轨迹数据里绝大多数是循环播放的产物，不是真实路线**：本地 MP4 会自动循环（`pipeline._run_file`），而部分素材只有 **40 秒**（如 `rnd_08`）。一个在镜头里走一趟的人会被反复记录 9 天——实测 `32c70435` 在 `rnd_03`↔`rnd_04` 之间产生 **9340 段**（11.8 万行）。**直接画路线会得到「这个人在两点之间来回跑了 4670 次」的假象**
    - `GET /api/identities/{global_id}/trajectory` 默认开启 `collapse_loops=true` 折叠，两种策略（外加一档降级路径）：
      - `period`：相机序列严格周期时截断到第一轮（时间线连续，播放动画不瞬移）
      - `fingerprint`：按 (camera, 首帧 bbox 中心 20px 量化) 去重，覆盖非严格周期
    - **周期检测必须用差分判据 `seq[i] == seq[i - period]`，不能用相位对齐 `seq[i] == seq[i % period]`**。真实数据里 A→B→A→B 中途会翻成 B→A→B→A，相位对齐在翻转点之后会 100% 判不匹配（实测 9340 段严格交替序列只有 **5%** 匹配率）
    - 硬截断到第一轮前会做**覆盖校验**：若后续轮次出现了第一轮没有的相机，降级用 fingerprint，宁可少折叠也不丢轨迹
    - 返回值里的 `loop` 字段说明是否折叠、用什么策略、折掉了多少，前端应展示提示
17. **`src/floorplan.py` 不提供模块级单例**，用 `build_floorplan()` 构造（`src/db.py` 的教训是 import 即连生产库）。`server.py` 在 `init_server(floorplan=...)` 注入，未注入时 `GET /api/floorplan` 惰性构造，单测裸 `TestClient(app)` 也能用
18. **匹配路径必须用公共分量中心化，且 gallery 与 center 原子配套**：`PersonnelGallery.match()` / `ReIDValidator.get_confirmed_match()` 都必须传 `IdentityStore.build_match_context()` 的 `context`（gallery + center + prepare）。拆成两次 getter 会在中间被别的线程注册新身份，造成"query 用新中心、gallery 用旧中心"的**静默错配**（不报错、相似度全错）。同理：底库、实时匹配、以图搜人共用同一中心化坐标系与同一套 `match_feature_detailed` 判据，分岔即失准。塌缩护栏 `0.999`（`_COLLAPSE_SIMILARITY_ALERT`）刻意放在 `ambiguous` 分支之前——特征塌缩的典型表现正是"几乎全相似却被 Ratio 判歧义"
