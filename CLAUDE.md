# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Lab-Monitor** 是基于 YOLOv8 + ReID 的多摄像头实时监控与智能告警系统。纯 Python 项目（无 Node.js、Docker），使用 FastAPI 提供 Web 服务，前端为无构建流程的模块化静态资源（`static/index.html` + ES module JS + 拆分 CSS）。

## 常用命令

## 常用命令

### 依赖安装

**必须按顺序安装**（PyTorch 依赖需先于其他库）：

```bash
# CPU 版本（通用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU 版本（CUDA 11.8，推荐用于高帧率场景）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
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

### 测试与验证

**stdlib `unittest`（没有 pytest）**，`tests/` 下 7 个文件共 60 个用例：

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -t .
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

### 全局共享组件

以下对象在 `main.py` 中实例化，被所有 pipeline 线程和 Web 服务共享：

| 组件 | 文件 | 作用 | 线程安全 |
|------|------|------|----------|
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

> **外部通知（notifier）只有两条路径**：MISSING_PERSON 的 ALERT（`src/alerter.py:374`）与 SCENE_EXIT（`src/alerter.py:455`）。**INTRUSION 与 CROWD_DENSITY 只写日志 + WebSocket，不发邮件/外部通知。**

> 告警逻辑在 `src/alerter.py` 中实现，时间窗口由 `config/topology.json` + `TransitCalibrator` 动态校准。

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
- YOLOv8 权重文件：`yolov8n.pt`（6.5MB，已提交到 Git）
- 检测器实例化：`src/detector.py` 第 8 行
- 要换模型（如 yolov8s.pt）：替换文件 + 修改 `PersonDetector.__init__` 中的路径

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
  - `src/identity_store.py:299`
- Ratio Test 阈值：`src/reid.py` `match_feature()` 的 `ratio=0.85` 参数
- 阈值越高 → 匹配越严格 → 更易产生新身份（适合外貌差异大的场景）

### 修改告警时间窗口
- 基础预期时间：`config/topology.json` 中各边的 `expected_seconds` 和 `tolerance_seconds`
- 自动校准：运行足够多样本（≥5条）后 `TransitCalibrator` 会覆盖静态配置
- 最小容忍时间：`src/calibrator.py` 第 81 行 `max(Z_SCORE * std, 5.0)` 中的 `5.0` 秒下限

### 添加新的告警类型
1. 在 `AlertManager.tick()` 中添加检测逻辑
2. 调用 `self._build_alert(entry, stage="ALERT")` 构造告警对象
3. 前端会通过 WebSocket 实时接收（无需修改前端代码）

## 前端代码

**已模块化**（不再是单文件）：
- `static/index.html`：311 行，只剩结构与资源引用
- `static/css/`：8 个文件（`main.css` / `variables.css` / `layout.css` / `utilities.css` + `components/` 下 4 个）
- `static/js/`：8 个文件，原生 **ES module**（`app.js` + `modules/` 5 个 + `utils/` 2 个）

- 无构建流程，由 FastAPI 在 `server.py:174-175` 的 `index()` 路由直接读取托管
- 使用原生 JavaScript + WebSocket + MJPEG `<img>` 标签
- **改任何 CSS/JS 后必须同步 bump `static/index.html` 里的 `?v=` 版本号**（当前 `?v=11.1`，见 index.html 第 13、309 行），否则浏览器会用旧缓存
- 只改前端资源时刷新浏览器即可（无需重启后端）

## Git 工作流

- **主分支**：`main`（当前分支，默认推送目标）
- **提交规范**：使用语义化前缀 `feat:` / `fix:` / `docs:` / `style:` / `refactor:`
- **视频文件**：放在 `videos/` 目录，已在 `.gitignore` 中排除（不要提交大文件）
- **模型权重**：`yolov8n.pt` 已提交（6.5MB 可接受），更大模型（如 yolov8x.pt）应使用 Git LFS

## 网络代理

开发机运行 Clash Verge 代理（`127.0.0.1:7897`）。如遇依赖下载失败：

```bash
# 临时启用代理（仅当前命令有效）
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 注意事项

1. **终端环境**：Windows + Git Bash（不是 PowerShell），禁用 PowerShell 专有命令
2. **Python 解释器**：必须用 `./.venv/Scripts/python.exe`（系统 PATH 里的 `python` 是 Microsoft Store 存根，静默退出）。该解释器不认 `/c/...` 路径，传路径请用 `C:\...`；stdout 默认 GBK，脚本打印中文需包 `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`
3. **路径格式**：代码中使用正斜杠 `/`（Python 跨平台兼容），Git Bash 路径用 `/c/Users/...`
4. **依赖清单**：`numpy` / `opencv-python` / `psutil` 已在 `requirements.txt` 中；`torch` / `torchvision` 不在，需单独先装；`scipy` 已在 P1 优化中移除，不再需要
5. **OSNet 权重**：首次运行自动从 Google Drive 下载（~3MB），缓存至 `~/.cache/torch/checkpoints/`；网络不通时可配置代理后重试
6. **FastAPI 自动重载**：默认未启用，修改后端代码需手动重启服务
7. **视频循环播放**：本地 MP4 文件会自动循环（`pipeline.py` `_run_file()` 中检测到流结束后重新打开）
8. **appearances 类型**：`PersonRecord.appearances` 是 `deque(maxlen=200)`，不是 `list`，但支持 `list()` 转换和负索引访问
9. **`.ps1` 必须存成 UTF-8 with BOM**：PowerShell 5.1 否则按 GBK 解析中文注释，会吃掉注释后续的语句（已踩过坑）
10. **`outputs/server.pid` 现在是可信的**：`main.py` 启动时写入自己的真实 PID（venv 的 `python.exe` 是 launcher，`start.ps1` 拿到的 `$process.Id` 并非工作进程，因此不再由脚本写入）。`stop.ps1` 仍以监听端口为主依据、PID 文件为辅；手工排查用：`Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess`
