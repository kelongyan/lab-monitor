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
| **Python 版本**| Python 3.10 ~ 3.13 | Python 3.10 ~ 3.13（当前实测：3.13） |

---

## 2. 视频源管理与存放规范

系统支持**本地离线视频文件**与**远程 RTSP 实时网络摄像头视频流**两种视频源。

### 2.1 本地视频文件存放规范

- **存放目录**：项目根目录下的 `videos/` 文件夹（例如 `F:\lab-monitor\videos\`）。
- **支持格式**：`.mp4`、`.avi`、`.mkv`、`.mov`。
- **推荐编码**：H.264 / AVC 编码（兼容性最佳），推荐分辨率 1080p (1920x1080) 或 720p。
- **放置示例**（当前项目为超算中心 L2 层实拍录像，按巡检路线分两个子目录）：
  ```text
  lab-monitor/
  └── videos/
      ├── 常规路线/
      │   ├── L2东侧走廊南北向南_20260731141700-20260731142100_1.mp4   # reg_01
      │   ├── L2东侧走廊南南向北_20260731141700-20260731142100_1.mp4   # reg_02
      │   └── ...                                                      # 其余 reg_xx 点位
      └── 随机路线/
          ├── L2东侧走廊北北向南_20260731143025-20260731143100_1.mp4   # rnd_01
          └── ...                                                      # 其余 rnd_xx 点位
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

**本项目已自带虚拟环境 `.venv`（Python 3.13）**，日常运行与测试请直接使用其解释器，不要用系统 `python`（Windows 上 PATH 里的 `python` 往往是 Microsoft Store 存根，会静默退出）：

```bash
./.venv/Scripts/python.exe main.py
./.venv/Scripts/python.exe -m unittest discover -s tests -t .
```

若需在新机器上重建环境：

```bash
# 创建虚拟环境（Python 3.10 ~ 3.13）
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
```

### CPU 版本安装

如果你在没有独立显卡的机器或普通服务器上部署：

```bash
# 1. 安装 PyTorch CPU 版本
./.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 安装项目基础依赖
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### GPU CUDA 版本安装 (推荐)

在包含 NVIDIA 显卡的机器上部署 GPU 版本可以大幅提升 YOLOv8 人员检测和 ReID 特征提取的帧率。

#### 步骤 1：检查 NVIDIA 驱动与 CUDA 版本
在终端运行 `nvidia-smi` 确认显卡驱动支持的最高 CUDA 版本（如 11.8 / 12.1 / 12.6）。

#### 步骤 2：安装 PyTorch GPU (CUDA) 版本

- **CUDA 12.6 版本（当前实测环境：RTX 3090 + Python 3.13）**：
  ```bash
  ./.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```
- **CUDA 11.8 版本**：
  ```bash
  ./.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

#### 步骤 3：安装项目依赖
```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

#### 步骤 4：运行系统（自动触发 GPU 加速）
`main.py` 内部会自动调用 `torch.cuda.is_available()` 检测 GPU 硬件，并自动完成 CUDA + FP16 半精度加速配置和优化参数切换（如 30fps 高帧率模式），无需手动修改代码逻辑：

```bash
./.venv/Scripts/python.exe main.py
```

---

## 4. 系统配置文件解析

所有的配置文件存放在 `config/` 目录中。

### 视频源配置 (`config/sources.json`)

配置各摄像头 ID 与对应视频源（文件路径或 RTSP 地址）。当前项目为超算中心 L2 层实拍录像，原始素材 32 路：`reg_01..reg_10` 为常规巡检环、`rnd_01..rnd_22` 为随机路线；其中 9 路解码损坏（见 Q4）、`rnd_03` 与 `rnd_04` 内容重复，均已摘除，实际启用 22 路：

```json
{
  "reg_01": "videos/常规路线/L2东侧走廊南北向南_20260731141700-20260731142100_1.mp4",
  "reg_02": "videos/常规路线/L2东侧走廊南南向北_20260731141700-20260731142100_1.mp4",
  "rnd_01": "videos/随机路线/L2东侧走廊北北向南_20260731143025-20260731143100_1.mp4",
  "cam_rtsp": "rtsp://admin:123456@192.168.1.100:554/stream1"
}
```

### 相机拓扑配置 (`config/topology.json`)

定义多摄像头之间的逻辑物理连通关系与期望通行时延（单位：秒），系统将根据该拓扑进行跨视角转移时延校验与滞留预警。实际按巡检路线的点位顺序串联：共位反向相机对（同一走廊两个朝向）用 `5 / 10` 秒，走廊相邻点位按步行距离取 `20~60` 秒，跨区长边（中间点位因视频损坏被摘除）用 `120~180` 秒：

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
直接在终端中运行（必须用项目自带虚拟环境的解释器）：
```bash
./.venv/Scripts/python.exe main.py
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

`start.ps1` 只有在 `/healthz` 返回本项目的健康响应后才提示启动成功。Python logging 写入 `outputs/server.log`（`RotatingFileHandler`，单个 64MB 保留 5 份）；进程的 stdout/stderr 由脚本兜底重定向到 `outputs/server.stdout.log` 与 `outputs/server.stderr.log`（超过 64MB 自动改名 `.old`），只用于捕获 FFmpeg 等 C 层输出。`stop.ps1` 优先调用本机安全停止接口，等待 pipeline、校准文件、JSONL 和 SQLite 完成收尾；只有超时后才强制终止已通过 PID、命令行和监听端口共同校验的项目进程。

### 访问 Web 监控面板
服务启动后，使用浏览器访问：
👉 **http://localhost:8000**

默认只监听 `127.0.0.1`，局域网内其他设备无法直接访问。确需远程访问时，必须同时配置登录凭据，再显式指定监听地址：

```powershell
$env:LAB_MONITOR_USERNAME = "operator"
$env:LAB_MONITOR_PASSWORD = "请替换为高强度密码"
.\.venv\Scripts\python.exe main.py --host 0.0.0.0 --port 8000
```

> **无鉴权风险**：`LAB_MONITOR_USERNAME` 与 `LAB_MONITOR_PASSWORD` 两者都为空时，所有 API 接口（含 `POST /api/admin/shutdown` 停服接口）**均无任何鉴权**，因此绝不要在未配置凭据的情况下绑定非本机地址（程序会主动拒绝启动）。

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
- **解决**：若是本项目旧实例，运行 `.\stop.ps1` 安全停止；若是其他程序，请先确认归属后自行处理，或运行 `.\.venv\Scripts\python.exe main.py --port 8001` 使用其他端口。停止脚本不会终止无法确认归属的进程。

### Q4: 摄像头日志显示打开成功，但一帧都读不出来？
- **现象**：`cv2.VideoCapture.isOpened()` 返回 `True`，首帧 `read()` 却直接失败；用 ffprobe 看元数据是 `fps=90000`、`frame_count=INT64_MIN`。
- **原因**：录像文件本身在导出/切片时损坏——H.264 码流缺少 PPS，或 HEVC 报 `PPS id out of range`，属于源文件问题，不是代码 bug。
- **实例**：本项目 32 路素材中有 9 路属此情况（`reg_03`/`reg_04`/`reg_07`/`reg_09`、`rnd_09`/`rnd_13`/`rnd_14`/`rnd_15`/`rnd_20`），已从 `config/sources.json` 摘除。
- **解决**：重新从录像机导出该点位视频，或用 `ffmpeg -i 坏文件.mp4 -c copy 修复.mp4` 尝试重封装后再验证首帧可读。

### Q5: 改完 `.ps1` 脚本后，PowerShell 报语法错误或部分语句被莫名跳过？
- **原因**：PowerShell 5.1 读取无 BOM 的 UTF-8 文件时按 GBK 解析，中文注释会被解码成乱码并"吃掉"其后的语句。
- **解决**：`start.ps1` / `stop.ps1` 必须保存为 **UTF-8 with BOM**（带签名）。修改后先跑一次确认脚本能完整执行。

### Q6: `outputs/server.pid` 里的 PID 和实际服务进程对不上？
- **原因**：虚拟环境的 `.venv\Scripts\python.exe` 是一层 launcher，`Start-Process` 返回的 PID 是启动器而非真正跑 `main.py` 的工作进程。
- **现状（已修复）**：`main.py` 启动时自己把真实 PID 写入 `outputs/server.pid`，`start.ps1` 启动前先清理残留、健康检查通过后读回真实值打印；`stop.ps1` 以监听端口为主依据，端口无监听时才回查 PID 文件（并校验进程名与命令行含 `main.py`，防误杀）。
- **手工排查**：仍可按端口反查交叉验证：
  ```powershell
  Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -ExpandProperty OwningProcess
  Get-Process -Id <上面得到的PID> | Select-Object Id, ProcessName, Path
  ```

---

## 7. 数据保留与敏感信息

系统默认保留最近 30 天的 SQLite 告警、身份轨迹、JSONL 镜像和告警截图；启动时会先把旧 JSONL 幂等合并进 SQLite，再执行过期清理。可通过环境变量调整保留期和内存身份上限：

```powershell
$env:LAB_MONITOR_RETENTION_DAYS = "30"
$env:LAB_MONITOR_MAX_IDENTITIES = "10000"
.\.venv\Scripts\python.exe main.py
```

`outputs/lab_monitor.db` 中的 ReID 主特征和 Feature Bank 属于敏感生物特征数据。部署时应限制 `outputs/` 的文件系统访问权限，备份必须加密，不应上传到公共对象存储或代码仓库。切换 ReID 模型时，系统会按 `feature_space` 隔离不兼容特征，不会把同维度但不同模型的向量混入同一身份库。

CSV 和 JSONL 是审计数据副本；在线历史查询以 SQLite 为唯一权威源。调整保留期前应按组织审计要求确认，缩短保留期会在下次启动时删除过期数据和截图。
