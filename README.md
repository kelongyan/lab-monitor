# <img src="docs/images/qlu.png" alt="齐鲁工业大学 Logo" height="38" valign="middle"> <img src="docs/images/chaosuan.png" alt="国家超算中心 Logo" height="38" valign="middle"> Lab-Monitor 超算中心智能监控预警系统

<p align="center">
  <b>基于计算机视觉与 ReID 的多摄像头实时监控、跨视角目标追踪及智能化告警平台</b>
</p>



---

## 📸 系统效果展示

![System Overview](docs/images/readme.png)

---

## ✨ 核心功能亮点

- 🎥 **多路视频源实时管理**
  - 支持本地视频文件及 RTSP 网络摄像头视频流的并发接入与多线程实时推流处理。
- 🔍 **智能目标检测与追踪**
  - 基于 **YOLOv8** 模型完成高精度人员检测（`PersonDetector`）。
  - 单相机内追踪基于 ultralytics 内置 **BYTETracker** 的轻量封装（`src/tracker.py`），实现轨迹平滑与目标连贯标识。
- 🆔 **跨视角 ReID 身份重识别**
  - 基于 **OSNet-x0.25** 骨干网络（`ReIDExtractor`）提取 512 维特征向量。加载的是 **Market-1501 度量学习权重**（`src/reid_config.py` 的 `DEFAULT_REID_WEIGHTS`，可切 msmt17），匹配阈值 0.68 由自动标注集标定（P=0.913 / R=0.639）。
  - 匹配前会减去全体特征共同的方向（**公共分量**）并重新归一化 —— OSNet 输出的单位特征彼此余弦高达 0.98 的根因就是这个公共分量；减掉后异人越阈比例从 54.9% 降到 1.7%（`src/identity_store.py:_feature_center_locked`）。
  - 跨不同摄像头视角建立全局身份库（`IdentityStore`），解决视角遮挡与离场重进识别难题。
  - 多帧确认机制（`ReIDValidator`）结合 Ratio Test，有效降低误识别率。
- 🪪 **实名人员档案**
  - 注册照底库 1:N 自动命名：画面里的人被"认出"后直接显示真实姓名（`PersonnelGallery`），支持档案增删改查与注册照上传。
  - 模拟数据与真实数据分档管理（`synthetic` 源强制打 SIM 徽标，避免用合成特征冒充识别精度）。
- 🔎 **跨摄像头视频检索与以图搜人**
  - 按身份编号或实名档案检索"这个人出现在哪些相机的哪些时间段"（`search.py`），按低清转码副本（`videos_low/`）定位并回放出片。
  - 以图搜人：上传照片提取特征，返回 Top-K 相似身份（离线检索返回排序，阈值语义有别于实时匹配）。
- 🛰️ **轨迹回放与路线图**
  - 跨相机轨迹分段展示 + 平面图路线图（站点自动合并共位反向相机对）。
  - 自动折叠**循环播放**产生的假轨迹（差分周期检测 + 位置指纹两档策略 + 覆盖校验降级），避免"同一人在两点间来回数千次"的假象。
- 🚧 **ROI 电子围栏（INTRUSION）**
  - 画面内归一化多边形区域，人体框 4 采样点任一落入即触发即时告警（`config/roi.json`）。支持 Web 端手绘画板热更新。
- 🗺️ **相机拓扑与穿越时延校验**
  - 可配置的摄像头空间拓扑模型（`CameraTopology`），支持相邻区域转移时延统计与概率校准（`TransitCalibrator`），及时发现异常路径或留存。
- 🚨 **智能化多渠道告警机制**
  - 后台多线程自动化巡检（`AlertManager`），针对滞留超时、未授权越界、异常转移等行为触发告警。
  - 支持 WebSocket 实时推流告警（`AlertBroadcaster`），并扩展邮件/终端等通知方式（`notifier`）。
- 🖥️ **Web 可视化大屏与 API 接口**
  - 基于 **FastAPI** 架构构建 Web 服务，提供 MJPEG 实时监控视频流、人员身份档案查询、轨迹回放、拓扑状态统计以及系统控制面板（REST + WebSocket 双通道）。

---

## 📁 项目目录结构

```text
lab-monitor/
├── config/                  # 系统配置文件目录（全 JSON，无 .env）
│   ├── sources.json         # 视频源配置（本地视频 / RTSP，当前启用 22 路）
│   ├── sources_high.json    # 高清源备份（切回原片时用，与 sources.json 同结构）
│   ├── topology.json        # 摄像头拓扑与预估穿越时间配置
│   ├── camera_map.json      # 摄像头在楼层平面图上的点位（路线可视化）
│   ├── roi.json             # ROI 电子围栏（画面内归一化多边形，INTRUSION 唯一来源）
│   ├── notify.json          # 告警通知渠道配置（Console / Email）
│   └── labeled/             # ReID 自动标注对（本地保留，标定阈值用）
├── src/                     # 核心源码目录
│   ├── detector.py          # YOLOv8 目标检测器
│   ├── tracker.py           # ByteTrack 单路跟踪
│   ├── reid.py              # ReID 特征提取（OSNet-x0.25 工厂 + Ratio Test）
│   ├── reid_config.py       # ReID 阈值 / 权重注册表（唯一来源，勿在调用点硬编码）
│   ├── reid_validator.py    # 多帧确认（buffer_size=8 / confirm_frames=3）
│   ├── identity_store.py    # 全局 ReID 身份数据库（多姿态 Feature Bank + 节流落盘）
│   ├── personnel.py         # 实名人员底库（注册照 1:N 自动命名、档案 CRUD）
│   ├── search.py            # 跨摄像头视频检索 + 以图搜人
│   ├── snapshots.py         # 跨相机抓拍裁图（检索证据链）
│   ├── trajectory.py        # 轨迹分段与循环播放折叠
│   ├── floorplan.py         # 平面图点位映射（轨迹回放底图）
│   ├── alerter.py           # 告警状态机与 WebSocket 广播器
│   ├── calibrator.py        # 轨迹转移时延校准器
│   ├── topology.py          # 相机拓扑关系（schema 校验 + 原子持久化）
│   ├── pipeline.py          # 单路视频流水线（每路一个线程）
│   ├── model_pool.py        # 有界模型实例池（打破单实例推理锁）
│   ├── frame_hub.py         # 视频帧共享缓冲区（MJPEG 源）
│   ├── label_render.py      # 画面人员标签（中文实名渲染）
│   ├── mock_personnel.py    # 模拟行人数据生成（标注 synthetic，演示用）
│   ├── notifier.py          # 告警通知发送器（Console / Email）
│   └── db.py                # SQLite 数据库操作层（惰性单例）
├── scripts/                 # ReID 标定 / 诊断 / 修复链 + 媒体探测 + 数据播种（22 个，本地保留）
│   ├── tune_reid_threshold.py / build_label_set.py / fetch_reid_weights.py
│   ├── probe_reid_separability.py / diagnose_ema_collapse.py / ab_*.py
│   ├── rebuild_identity_db.py / verify_rebuilt_identities.py
│   ├── probe_media_*.py / seed_video_assets.py / transcode_lowres.py
│   └── seed_personnel_mock.py / derive_camera_map.py / render_floorplan.py …
├── static/                  # Web 前端静态资源（index.html + css/ + js/ ES module，无构建）
├── tests/                   # 25 个 stdlib unittest 文件 + 3 个 Node 前端用例，共 359 项断言
├── videos/                  # 原始素材（1080p，gitignore）
├── videos_low/              # 960×540 CRF28 转码副本（检索 / 回放 / 抓拍实际素材，gitignore）
├── docs/                    # 项目文档与资源（本地保留：审计台账 / 三轮能力方案 / 技术总结）
├── .plans/                  # 设计记录（本地保留，如 SCENE_EXIT 告警）
├── outputs/                 # 运行时输出目录（自动生成，已被 .gitignore 排除）
│   ├── lab_monitor.db       # SQLite 数据库（身份履历 / 告警 / 视频资产）
│   ├── alerts.jsonl         # 追加式告警日志
│   ├── transit_stats.json   # 穿越时延统计
│   ├── reports/             # 标定与 A/B 实验产物（调参结论的权威来源）
│   └── screenshots/         # 告警截图
├── main.py                  # 系统主入口
├── server.py                # FastAPI Web 服务器（REST + MJPEG + WebSocket）
├── demo.py                  # 快速演示脚本（合成视频，无需真实素材）
├── start.ps1                # PowerShell 一键后台启动脚本
├── stop.ps1                 # PowerShell 一键停止后台服务脚本
├── requirements.txt         # Python 依赖清单（torch / torchvision 需单独先装）
└── .gitignore               # Git 忽略配置
```

---

## ⚙️ 环境部署与配置

### 1. 环境要求

- **操作系统**: Windows / Linux / macOS
- **Python 版本**: Python 3.10+（当前实测环境为 **Python 3.13**）
- **PyTorch**: 建议安装支持当前环境的 PyTorch 和 Torchvision（当前实测环境为 **torch CUDA 12.6 + NVIDIA RTX 3090**）

### 2. 安装依赖

推荐使用 `conda` 或 `venv` 虚拟环境。根据硬件选择 **CPU** 或 **GPU (CUDA)** 依赖：

#### 选项 A：CPU 版本安装 (通用)
```bash
# 1. 安装 PyTorch CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 安装项目依赖（含 OSNet ReID 模型依赖）
pip install -r requirements.txt
```

#### 选项 B：NVIDIA GPU (CUDA) 版本安装 (推荐，高帧率)
```bash
# CUDA 12.6 版本（当前实测环境：Python 3.13 + RTX 3090）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 或按驱动支持的版本改用 cu118 / cu121
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖（含 OSNet ReID 模型依赖：torchreid + gdown + tensorboard）
pip install -r requirements.txt
```

> **首次运行说明**：OSNet-x0.25 预训练权重（~3MB）将自动从 Google Drive 下载并缓存到 `~/.cache/torch/checkpoints/`，首次启动需联网。

> 📖 **完整部署指引**：关于视频文件存放规范、RTSP 摄像头接入配置、相机拓扑规则及详细 GPU 调优排错说明，请查阅 [docs/deployment.md](docs/deployment.md)。


---

## 🚀 快速启动指南

### 1. 配置视频源与拓扑

编辑 `config/` 目录下的配置文件：

- **视频源配置** (`config/sources.json`):
  超算中心 L2 层实拍录像，原始素材 32 路：`reg_01..reg_10`（常规巡检环）+ `rnd_01..rnd_22`（随机路线）。
  ```json
  {
    "reg_01": "videos/常规路线/L2东侧走廊南北向南_20260731141700-20260731142100_1.mp4",
    "reg_02": "videos/常规路线/L2东侧走廊南南向北_20260731141700-20260731142100_1.mp4",
    "rnd_01": "videos/随机路线/L2东侧走廊北北向南_20260731143025-20260731143100_1.mp4",
    "cam_rtsp": "rtsp://admin:password@192.168.1.100:554/stream1"
  }
  ```
  *(注：视频路径支持本地视频文件及 RTSP 视频流 `rtsp://...`；其中 9 路原始录像解码损坏、1 路与其他点位重复，已从配置摘除，当前实际启用 22 路，详见 [docs/deployment.md](docs/deployment.md)。)*

- **拓扑关系配置** (`config/topology.json`):
  配置各摄像头之间的连通关系及预计通行时间（单位：秒）。

### 2. 准备视频文件/数据

将需要测试的视频文件放置在 `videos/` 目录下（视频文件不会被 Git 提交推送到仓库）。

### 3. 运行服务

#### 方式一：直接运行 (前台控制台)

```bash
# 使用项目自带虚拟环境的解释器（Windows 上不要用系统 python）
./.venv/Scripts/python.exe main.py
```

#### 方式二：一键后台运行 (Windows PowerShell)

系统提供了方便后台守护运行与安全停止的 PowerShell 脚本：

```powershell
# 启动服务
.\start.ps1

# 停止服务
.\stop.ps1
```

### 4. 访问 Web 监控大屏

服务启动后，在浏览器中打开：

👉 **[http://localhost:8000](http://localhost:8000)**

> **注意**：服务默认只监听 `127.0.0.1`，局域网内其他设备无法直接访问。确需远程访问请先配置 `LAB_MONITOR_USERNAME` / `LAB_MONITOR_PASSWORD` 登录凭据，再用 `--host` 显式指定监听地址（详见 [docs/deployment.md](docs/deployment.md)）。

在监控大屏中可实时查看多路摄像头推流、人脸/身份识别轨迹、实时告警面板及拓扑数据分析。

---

## 📝 贡献与许可

欢迎提交 Issue 和 Pull Request 完善本项目！
