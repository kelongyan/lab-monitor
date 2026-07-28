# Lab-Monitor 系统详细部署与配置指南

本文档提供 **Lab-Monitor 实验室/超算中心智能监控预警系统** 的全面部署指引，涵盖 CPU / GPU 两种运行模式、视频源准备规范、配置详解以及故障排查。

---

## 目录

1. [环境与硬件要求](#1-环境与硬件要求)
2. [视频源管理与存放规范](#2-视频源管理与存放规范)
3. [依赖环境安装 (CPU / GPU)](#3-依赖环境安装-cpu--gpu)
   - [CPU 版本安装](#cpu-版本安装)
   - [GPU CUDA 版本安装 (推荐)](#gpu-cuda-版本安装-推荐)
4. [系统配置文件解析](#4-系统配置文件解析)
   - [视频源配置 (`config/sources.json`)](#视频源配置-configsourcesjson)
   - [相机拓扑配置 (`config/topology.json`)](#相机拓扑配置-configtopologyjson)
   - [告警通知配置 (`config/notify.json`)](#告警通知配置-confignotifyjson)
5. [系统启动与后台服务运行](#5-系统启动与后台服务运行)
6. [常见问题与故障排查 (Troubleshooting)](#6-常见问题与故障排查-troubleshooting)
7. [数据保留与敏感信息](#7-数据保留与敏感信息)

---

## 1. 环境与硬件要求

| 组件 | CPU 运行模式 | GPU 运行模式 (推荐) |
| :--- | :--- | :--- |
| **操作系统** | Windows 10/11 / Linux (Ubuntu 20.04+) | Windows 10/11 / Linux (Ubuntu 20.04+) |
| **处理器 (CPU)** | Intel Core i5 8代以上 / AMD Ryzen 5 | Intel Core i7 10代以上 / AMD Ryzen 7 |
| **显卡 (GPU)** | N/A | NVIDIA GTX 1060 (6GB) 及以上，推荐 RTX 3060/4060+ |
| **内存 (RAM)** | 至少 8 GB | 推荐 16 GB 及以上 |
| **显存 (VRAM)**| N/A | 至少 4 GB，多路 4K 建议 8GB+ |
| **Python 版本**| Python 3.8 ~ 3.11 | Python 3.8 ~ 3.11 |

---

## 2. 视频源管理与存放规范

系统支持**本地离线视频文件**与**远程 RTSP 实时网络摄像头视频流**两种视频源。

### 2.1 本地视频文件存放规范

- **存放目录**：项目根目录下的 `videos/` 文件夹（例如 `F:\lab-monitor\videos\`）。
- **支持格式**：`.mp4`、`.avi`、`.mkv`、`.mov`。
- **推荐编码**：H.264 / AVC 编码（兼容性最佳），推荐分辨率 1080p (1920x1080) 或 720p。
- **放置示例**：
  ```text
  lab-monitor/
  └── videos/
      ├── people_sample.mp4   # 摄像头 1 示例视频
      ├── store_sample.mp4    # 摄像头 2 示例视频
      ├── street_sample.mp4   # 摄像头 3 示例视频
      └── hall_sample.mp4     # 摄像头 4 示例视频
  ```
  *(注：`videos/` 目录中的大文件视频默认已被 `.gitignore` 排除，不会提交到 Git 远程仓库。)*

### 2.2 RTSP 网络摄像头规范

- 如果接入海康威视、大华、宇视等网络摄像头，需获取相机的 RTSP URL。
- **RTSP 标准 URL 格式**：
  - 海康威视：`rtsp://admin:password@192.168.1.64:554/h264/ch1/main/av_stream`
  - 大华：`rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`

RTSP URL 中的用户名和密码会在应用日志中自动脱敏。连接和读取默认各使用 10 秒超时，可按网络条件在当前进程中调整：

```powershell
$env:LAB_MONITOR_RTSP_OPEN_TIMEOUT_MS = "15000"
$env:LAB_MONITOR_RTSP_READ_TIMEOUT_MS = "15000"
```

---

## 3. 依赖环境安装 (CPU / GPU)

首先建议使用 `conda` 或 `venv` 创建干净的虚拟环境：

```bash
# 使用 conda 创建虚拟环境
conda create -n lab-monitor python=3.10 -y
conda activate lab-monitor
```

### CPU 版本安装

如果你在没有独立显卡的机器或普通服务器上部署：

```bash
# 1. 安装 PyTorch CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 安装项目基础依赖
pip install -r requirements.txt
```

### GPU CUDA 版本安装 (推荐)

在包含 NVIDIA 显卡的机器上部署 GPU 版本可以大幅提升 YOLOv8 人员检测和 ReID 特征提取的帧率。

#### 步骤 1：检查 NVIDIA 驱动与 CUDA 版本
在终端运行 `nvidia-smi` 确认显卡驱动支持的最高 CUDA 版本（如 11.8 或 12.1）。

#### 步骤 2：安装 PyTorch GPU (CUDA) 版本

- **CUDA 11.8 版本**：
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **CUDA 12.1 版本**：
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

#### 步骤 3：安装项目依赖
```bash
pip install -r requirements.txt
```

#### 步骤 4：运行系统（自动触发 GPU 加速）
`main.py` 内部会自动调用 `torch.cuda.is_available()` 检测 GPU 硬件，并自动完成 CUDA + FP16 半精度加速配置和优化参数切换（如 30fps 高帧率模式），无需手动修改代码逻辑：

```bash
python main.py
```

---

## 4. 系统配置文件解析

所有的配置文件存放在 `config/` 目录中。

### 视频源配置 (`config/sources.json`)

配置各摄像头 ID 与对应视频源（文件路径或 RTSP 地址）：

```json
{
  "cam_01": "videos/people_sample.mp4",
  "cam_02": "videos/store_sample.mp4",
  "cam_03": "videos/street_sample.mp4",
  "cam_04": "rtsp://admin:123456@192.168.1.100:554/stream1"
}
```

### 相机拓扑配置 (`config/topology.json`)

定义多摄像头之间的逻辑物理连通关系与期望通行时延（单位：秒），系统将根据该拓扑进行跨视角转移时延校验与滞留预警：

```json
{
  "cam_01": [
    {
      "next": "cam_02",
      "expected_seconds": 30,
      "tolerance_seconds": 15
    },
    {
      "next": "cam_03",
      "expected_seconds": 45,
      "tolerance_seconds": 15
    }
  ]
}
```

### 告警通知配置 (`config/notify.json`)

系统支持控制台输出以及 SMTP 邮件等实时告警推送：

```json
{
  "console": {
    "enabled": true
  },
  "email": {
    "enabled": true,
    "smtp_host": "smtp.qq.com",
    "smtp_port": 465,
    "use_ssl": true,
    "username": "your_email@qq.com",
    "password": "your_smtp_auth_code",
    "from": "your_email@qq.com",
    "to": ["admin@example.com"]
  }
}
```

---

## 5. 系统启动与后台服务运行

### 方式一：前台控制台调试模式
直接在终端中运行：
```bash
python main.py
```

### 方式二：Windows PowerShell 后台后台守护运行
系统提供了内置的后台管理脚本：
- **启动服务**：
  ```powershell
  .\start.ps1
  ```
- **停止服务**：
  ```powershell
  .\stop.ps1
  ```

`start.ps1` 只有在 `/healthz` 返回本项目的健康响应后才提示启动成功。Python logging 写入 `outputs/server.log`，标准输出写入 `outputs/server.stdout.log`。`stop.ps1` 优先调用本机安全停止接口，等待 pipeline、校准文件、JSONL 和 SQLite 完成收尾；只有超时后才强制终止已通过 PID、命令行和监听端口共同校验的项目进程。

### 访问 Web 监控面板
服务启动后，使用浏览器访问：
👉 **http://localhost:8000**

默认只监听 `127.0.0.1`，局域网内其他设备无法直接访问。确需远程访问时，必须同时配置登录凭据，再显式指定监听地址：

```powershell
$env:LAB_MONITOR_USERNAME = "operator"
$env:LAB_MONITOR_PASSWORD = "请替换为高强度密码"
python main.py --host 0.0.0.0 --port 8000
```

浏览器首次访问时会显示 HTTP Basic 登录框。远程部署必须通过 Nginx、Caddy 等可信反向代理启用 HTTPS；Basic 认证本身不加密用户名、密码和监控数据。不要把密码写入仓库文件或 PowerShell 脚本。

---

## 6. 常见问题与故障排查 (Troubleshooting)

### Q1: 提示 `以下摄像头视频文件不存在，已跳过`？
- **原因**：`config/sources.json` 中配置的文件路径不存在。
- **解决**：请确认已将 mp4 视频文件放入 `videos/` 文件夹中，或检查 json 中的文件名拼写是否完全一致。

### Q2: CUDA out of memory (显存溢出)？
- **原因**：同时并发处理的多路高分辨率视频流超出了显存上限。
- **解决**：
  1. 使用更轻量的 YOLO 模型权重（默认 `yolov8n.pt` 已是轻量版）。
  2. 降低 `config/sources.json` 中并发运行的摄像头路数。
  3. 适当降低视频流帧率或分辨率。

### Q3: 端口 8000 被占用，无法启动？
- **原因**：上一次运行的服务未完全退出，或其它程序占用了 8000 端口。
- **解决**：若是本项目旧实例，运行 `.\stop.ps1` 安全停止；若是其他程序，请先确认归属后自行处理，或运行 `python main.py --port 8001` 使用其他端口。停止脚本不会终止无法确认归属的进程。

---

## 7. 数据保留与敏感信息

系统默认保留最近 30 天的 SQLite 告警、身份轨迹、JSONL 镜像和告警截图；启动时会先把旧 JSONL 幂等合并进 SQLite，再执行过期清理。可通过环境变量调整保留期和内存身份上限：

```powershell
$env:LAB_MONITOR_RETENTION_DAYS = "30"
$env:LAB_MONITOR_MAX_IDENTITIES = "10000"
python main.py
```

`outputs/lab_monitor.db` 中的 ReID 主特征和 Feature Bank 属于敏感生物特征数据。部署时应限制 `outputs/` 的文件系统访问权限，备份必须加密，不应上传到公共对象存储或代码仓库。切换 ReID 模型时，系统会按 `feature_space` 隔离不兼容特征，不会把同维度但不同模型的向量混入同一身份库。

CSV 和 JSONL 是审计数据副本；在线历史查询以 SQLite 为唯一权威源。调整保留期前应按组织审计要求确认，缩短保留期会在下次启动时删除过期数据和截图。
