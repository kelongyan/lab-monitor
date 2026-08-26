# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Lab-Monitor** 是基于 YOLOv8 + ReID 的多摄像头实时监控与智能告警系统。纯 Python 项目（无 Node.js、Docker），使用 FastAPI 提供 Web 服务，前端为无构建流程的模块化静态资源（`static/index.html` + ES module JS + 拆分 CSS）。

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
| `LAB_MONITOR_REID_EVERY_N` | GPU `5` / CPU `15` | 每个 track 每 N 个处理帧提一次 ReID 特征。**必须和实际达到的帧率一起调**，见「性能档位」下的缓冲填充算式 |
| `LAB_MONITOR_MJPEG_FPS` | GPU `15` / CPU `15` | MJPEG 推流轮询频率 |
| `LAB_MONITOR_LEAVE_GRACE_FRAMES` | `12` | track_id 从 tracker 输出里连续缺席多少个**处理帧**才算人员离场（`src/pipeline.py:38-56`）。上限钳到 30（= `track_buffer`），设 `1` 可回退到改造前"当帧即判离场"的行为 |
| `LAB_MONITOR_INTRUSION_COOLDOWN` | `120` | 同一（相机, 身份, 围栏）的 INTRUSION 复报间隔秒数（`src/alerter.py:27`）。**这只是复报间隔，不是驻留判定**——按"进入/离开围栏"配对的状态机尚未实现 |

### 测试与验证

**stdlib `unittest`（venv 里没装 pytest，别写 pytest 专有语法）**，`tests/` 下 10 个 Python 文件共 113 个用例；另有一个纯 Node 的前端用例（19 项，测 MJPEG 连接管理器与写请求守卫头）：

```bash
# 全量
./.venv/Scripts/python.exe -m unittest discover -s tests -t .

# 单个文件 / 单个类 / 单个用例（点号路径，不是文件路径）
./.venv/Scripts/python.exe -m unittest tests.test_core_logic -v
./.venv/Scripts/python.exe -m unittest tests.test_stream_security.StreamSecurityTest
./.venv/Scripts/python.exe -m unittest tests.test_model_pool.ModelPoolTest.test_borrow_returns_instance

# 前端（Node，无依赖）：19 项断言，整体 pass/fail
node tests/test_stream_manager_frontend.mjs
```

> **注意**：`src/db.py` 末尾有模块级单例 `db = Database()`，import 即连上生产库 `outputs/lab_monitor.db`——跑测试会碰生产数据，必要时先备份。

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
| `reid_every_n` | 5 | 15 | 每个 track 每 N 帧才提一次 ReID 特征。**这个值不能单独调**：`ReIDValidator` 要攒满 `buffer_size=8` 才会 `register_if_new()`，填满耗时 = `8 × reid_every_n / 实际fps` 秒，而单相机内轨迹时长中位只有 **7.1s**（实测）。10fps+R=3 → 2.4s ✓；1.77fps+R=5 → 22.6s ✗ 永远填不满 |
| `frame_rate_cap` | **10** | 15 | `_read_loop` 里 sleep 补齐到该帧率。GPU 档 2026-08-25 从 30 降到 10：素材全是 25fps，抽到 10fps 时 ByteTrack 相邻帧 IoU 中位 0.87、关联失败率 0.1%，追 25fps 纯属浪费 |
| `mjpeg_fps` | **15** | 15 | MJPEG 推流间隔，取略高于 `frame_rate_cap` 以免节拍抖动叠加延迟 |
| `jpeg_quality` | 85 | 65 | `FrameHub` 编码质量 |

同时做了**线程钳制**：`cv2.setNumThreads(1)` + `torch.set_num_threads(2)`。22 路解码 + 池化并发推理已经能打满 CPU，OpenCV/torch 内部线程池再抢核只会加剧上下文切换。**`LAB_MONITOR_MODEL_POOL=1` 只回退池大小，不回退这两行。**

> **当前实测吞吐（2026-08-25，22 路 / RTX 3090 / 14 核）**：每路 **1.77 fps**、合计 38.9 帧/s、frame_age 0.28s、RSS 6.4GB。瓶颈是**单进程 GIL**，不是 GPU（GPU 利用率 35~40%，13 个核空转）：单线程 detect 天花板就是 56 帧/s（17.6ms/帧，其中 inference 14.3ms），22 路挤一把 GIL 后只剩 38.9。**加大模型池无用**（池 2/4/6 都是 38 帧/s），**降 `frame_rate_cap` 也无用**（限速 sleep 压根不触发）。要突破必须减少单进程内的线程数 —— 见 `docs/` 里的优化方案。

### 启动期的两个降级路径

两者都**不会让服务起不来**，只会静默丢功能，排查时优先看启动日志：

1. **拓扑校验失败** → `main.py:255-269` 捕获 `TopologyValidationError`，换成空拓扑继续启动，**MISSING_PERSON 全线失效**（`next_hops()` 恒为空 → `watch()` 直接 return）。改好 `config/topology.json` 后必须重启。
2. **视频源文件不存在** → `main.py:164-174` 直接从 `sources` 里剔除该相机，只打一条 warning。前端网格里那一路会连卡片都没有。

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

### ReID 身份识别流程

```
CameraPipeline (src/pipeline.py)
  ↓
检测到新人员 → 提取特征向量 (ReIDExtractorOSNet — OSNet-x0.25, 512维)
  ↓
积累多帧缓冲 (ReIDValidator) → 计算平均特征
  ↓
查询身份库 (match_feature + Ratio Test)
  ↓
匹配成功(3帧一致) → 返回已有 global_id
匹配失败(缓冲区满) → IdentityStore.register_if_new() 原子性查重+注册 (支持多姿态 feature_bank)
  ↓
更新特征 (质量加权滑动平均，alpha 随帧质量动态调整，自动维护 feature_bank)
```

**关键参数**：
- ReID 模型：**OSNet-x0.25**（512维），通过 `build_reid_extractor()` 工厂函数实例化（自动回退 ResNet50）
  - **当前加载的是 ImageNet 预训练权重（非 ReID 专用权重）**：`src/reid.py:93-98` 用 `pretrained=True`，从未加载 Market-1501 等 ReID 数据集权重，`feature_space` 字符串 `osnet-x0.25-imagenet:512` 即为自证。因此跨镜头判别力有限，属已知待办项
- 相似度阈值：`0.75`（**散落在 5 处，改一处无效**，清单见下方「调整 ReID 匹配阈值」）
- Ratio Test：`second_sim / best_sim > 0.85` 时拒绝歧义匹配（`src/reid.py`）
- 多帧确认：连续 3 帧匹配同一 ID 才确认（`confirm_frames=3`）
- 注册防重：`register_if_new()` 在锁内原子执行查重+注册，防多摄像头并发重复注册
- 多姿态特征库：`feature_bank` 保存最多 5 个差异明显特征（相似度 < 0.92），提升大视角变化下的检索召回率

### 告警触发机制

`AlertManager.tick()` 每 0.5 秒扫描所有活跃身份，支持以下告警类型：

1. **MISSING_PERSON (超时/失踪)**：人员离开某相机后，未在预期时间窗口内到达拓扑相邻相机
2. **INTRUSION (即时越界/围栏)**：人体框的 **4 个采样点**（底边中心/底边左角/底边右角 + 身体中心，`src/pipeline.py:512-517`）任一落入 ROI 电子围栏即触发（`cv2.pointPolygonTest`），不是只判脚下一点
3. **SCENE_EXIT (全域消失)**：人员从所有摄像头消失超过设定期限（默认 300 秒）
4. **CROWD_DENSITY (聚众预警)**：区域内检测到的人数超过设定阈值（默认 5 人）

告警分两阶段触发：
- **WARNING**（70% deadline）：控制台预警，WebSocket 推送紫色/黄色标志，不触发外部通知
- **ALERT**（100% deadline / INTRUSION）：全量告警 + WebSocket 推送 + 日志/数据库持久化

> **外部通知（notifier）只有两条路径**：MISSING_PERSON 的 ALERT（`src/alerter.py:380`）与 SCENE_EXIT（`src/alerter.py:461`）。**INTRUSION 与 CROWD_DENSITY 只写日志 + WebSocket，不发邮件/外部通知。**

> 告警逻辑在 `src/alerter.py` 中实现，时间窗口由 `config/topology.json` + `TransitCalibrator` 动态校准。

## Web 服务（server.py）

模块级全局变量（`_frame_hub` / `_identity_store` / `_pipelines` / `_topology` ...）由 `main.py` 调 `init_server()` 注入，**导入 `server` 不等于服务可用**，单测里直接 `TestClient(app)` 时这些是 `None`（各路由都做了 None 分支）。

⚠️ **两个中间件的定义顺序不能动**（`server.py:175-179` 有注释说明）：Starlette 的 `add_middleware` 是 `insert(0)`，**后注册的在最外层**。`guard_request` 必须定义在 `require_basic_auth` **之后**，才能先于鉴权执行。新增中间件时想清楚要插在哪一层。

接口分组（全部在 `server.py`，共约 18 个）：

| 分组 | 端点 |
|------|------|
| 页面 / 流 | `GET /`、`GET /stream/{cam_id}`（MJPEG）、`WS /ws`（告警推送） |
| 读接口 | `/api/status`、`/api/alerts`、`/api/alerts/history`、`/api/alerts/export`（CSV，上限 5 万行）、`/api/identities`、`/api/identities/{global_id}`、`/api/stats`、`/api/roi`、`/api/topology`、`/api/metrics/reid`、`/api/system/metrics` |
| 写接口（需守卫头） | `POST /api/roi`、`POST /api/topology`、`POST /api/admin/shutdown`（仅回环） |
| 运维 | `GET /healthz`（`start.ps1` 靠它判断启动成功） |

静态资源由 `StaticFiles` 挂在 `/static`，告警截图挂在 `/screenshots`。

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

- **当前只有 `rnd_16` 一条围栏**，其余 21 路没有 ROI，永远不会报 INTRUSION
- **热更新**：`POST /api/roi` 写盘后会遍历 `_pipelines` 调 `reload_rois()`，不用重启。手改文件则必须重启
- 写接口校验：相机 ID 必须在已配置的 `_pipelines` 里、顶点 3~64 个、坐标 `[0,1]`（`server.py:309-394`）
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
- `outputs/server.log`：Python logging 写入，`main.py` 用 `RotatingFileHandler` 轮转（单个 64MB，保留 5 份），无需手工清理
- `outputs/server.stderr.log` / `outputs/server.stdout.log`：`start.ps1` 重定向的 stderr / stdout，只兜底捕获 C 层（FFmpeg）输出与 `print`；启动时若超过 64MB 会被自动改名 `.old`
- `outputs/server.pid`：真实服务进程 PID，由 `main.py` 自己写入（`start.ps1` 启动前先清残留），`stop.ps1` 以监听端口为主、PID 文件为辅

## 常见开发任务

### 修改检测模型（YOLOv8）
- YOLOv8 权重文件：`yolov8n.pt`（6.5MB，在工作区里，但**没有**入 Git——`.gitignore` 的 `*.pt` 把它排除了）。新克隆的仓库首次运行时由 ultralytics 自动下载同名权重
- 检测器实例化：`src/detector.py:12`（`PersonDetector.__init__`），`YOLO(model_name)` 在 `:19`；实际传参在 `main.py:226`（`conf_thresh=0.4`）
- 要换模型（如 yolov8s.pt）：改 `main.py:226` 的 `model_name`，或替换同名权重文件

### 修改 ReID 模型
- 当前：**OSNet-x0.25**（512维，自动在首次运行时下载）
- 工厂函数：`src/reid.py` 中的 `build_reid_extractor()`，优先 OSNet，OSNet 不可用时回退 ResNet50（2048维）
- 切换为其他 torchreid 模型（如 osnet_x1_0）：修改 `ReIDExtractorOSNet.__init__` 中的 `name` 参数
- **注意**：切换模型后特征维度可能变化（512→2048），需清空 `outputs/transit_stats.json` 重新校准

### 调整 ReID 匹配阈值
- **`0.75` 这个阈值散落在 5 处，只改一处不生效**，需同步修改：
  - `src/reid.py:171`、`src/reid.py:208`（`match_feature` / 相关函数默认参数）
  - `src/reid_validator.py:31`
  - `src/pipeline.py:110` 与 `:210`（调用处显式传参）
  - `src/identity_store.py:371`（`register_if_new`）
- Ratio Test 阈值：`src/reid.py` `match_feature()` 的 `ratio=0.85` 参数
- 阈值越高 → 匹配越严格 → 更易产生新身份（适合外貌差异大的场景）

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
- `static/index.html`：316 行，只剩结构与资源引用
- `static/css/`：8 个文件（`main.css` / `variables.css` / `layout.css` / `utilities.css` + `components/` 下 4 个）
- `static/js/`：9 个文件，原生 **ES module**（`app.js` + `modules/` 6 个 + `utils/` 2 个）

- 无构建流程，由 FastAPI 在 `server.py:286-290` 的 `index()` 路由每次请求直接读盘托管（改 HTML 不用重启，但 `?v=` 仍决定 CSS/JS 是否走缓存）
- 使用原生 JavaScript + WebSocket + MJPEG `<img>` 标签
- **`modules/stream_manager.js` 管理 MJPEG 长连接**：浏览器对同域并发连接上限为 6，22 路全量建流会把连接池占满（只有约 6 路能出图，REST 轮询也会挨饿）。它用 `IntersectionObserver` 只给视口内的卡片建流、同时建流上限 4、超限时整批轮转、断流前把最后一帧冻结成占位图；焦点大屏常驻，标签页切后台全部断流。**改网格/弹窗相关代码时注意调用 `resetStreamRegistry()` / `registerStreamImage()` 的时机**，否则会出现"卡片可见但永不建流"或连接泄漏。纯 Node 用例：`node tests/test_stream_manager_frontend.mjs`
- **改任何 CSS/JS 后必须同步 bump `static/index.html` 里的 `?v=` 版本号**（当前 `?v=11.2`）。注意 `app.js` 里的 `import './modules/*.js'` 与 `main.css` 的 `@import` 子路径**不带版本号**，改子模块后需要 `Ctrl+F5` 硬刷新才能看到效果
- 只改前端资源时刷新浏览器即可（无需重启后端）

## 文档与审计台账

- `docs/CODE_AUDIT_2026-07-28.md`：**代码里 `P0-1` / `P1-4` / `F5` / `F10` / `P2-13` 这类注释编号的字典**。41 组确定性缺陷 + 7 组条件性风险的编号台账，看不懂某处防御性代码为什么存在就查这里
- `docs/SYSTEM_PRINCIPLES.md`：系统原理讲解（面向汇报，非实现细节）
- `docs/deployment.md`：完整部署 / 排错手册，比本文件详细
- `.plans/scene-exit-alert.md`：SCENE_EXIT 的设计记录（已入库）
- `docs/*.xlsx`（**已 gitignore**）：现场点位表，含真实摄像头 IP 与图纸点号，**是 `config/topology.json` 的唯一依据**。仓库公开，不要提交；本地丢了就只能从 `常规路线.7z` / `随机路线.7z` 里取

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
11. **绝不要在 `_reset_stream_state()` 里重建 `PersonTracker`**（`src/pipeline.py:222-229` 有长注释）：`BYTETracker.__init__` 会调 `reset_id()`，而 `BaseTrack._count` 是**类级共享**计数器，重建会把**所有**摄像头线程的 `track_id` 一起归零 → 与其它路正在活跃的 id 撞号 → `_track_to_global` 把新来的人认成旧身份。旧轨迹交给 `track_buffer` 自然淘汰即可
12. **流重开（文件循环 / RTSP 重连）必须重置跟踪状态**：不重置时旧 `track_id` 会集体从 tracker 输出里消失，被 `_process_frame` 当成人员离场 → `watch()` → 一片 MISSING_PERSON 误报
13. **离场判定带宽限期（F2b）**：ultralytics 8.4 的 `BYTETracker._format_output()` 只返回 `is_activated` 的轨迹，一次 IoU 关联失败该 track 当帧就从输出里消失（内部仍按 `track_buffer=30` 帧保留、可复活）。所以 `_process_frame` **不能**看到 id 消失就判离场——那会误报 MISSING_PERSON 并清空攒了一半的 ReID 缓冲（`buffer_size=8` 永远填不满 → 无法注册身份）。现在改成连续缺席 `_LEAVE_GRACE_FRAMES`（默认 12）个处理帧才调 `_on_person_leave()`，状态记在 `_absent_streak` 里，`_prev_track_ids` 的语义也随之变成"在场（含宽限期内）"而不是"上一帧输出"
14. **ReID 特征落盘是节流的**：`IdentityStore` 对特征列做「累计 50 次更新或间隔 30 秒」节流（`src/identity_store.py:23-24`），不调 `flush()` 就会静默丢掉最近一段滑动平均结果。`main.py` 的 `flush_identity_features()` 在关停路径上补写，新增退出分支时别漏掉它
15. **`demo.py` 不读真实录像**：它生成合成视频（矩形模拟人员移动）跑通检测→跟踪→ReID→告警链路，适合在没有素材或想快速验证改动时用
