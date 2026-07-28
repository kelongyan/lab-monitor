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

### 访问 Web 监控面板
服务启动后，使用浏览器访问：
👉 **http://localhost:8000**

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
- **解决**：运行 `.\stop.ps1` 强行释放端口，或在 `main.py` 中自定义 `web_port` 端口号。
