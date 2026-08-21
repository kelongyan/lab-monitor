# F:\lab-monitor\start.ps1
# 一键后台启动超算中心监控预警服务（关闭终端不影响后台运行）

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

# 检查 8000 端口是否有进程在 Listen 监听
$portOccupied = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($portOccupied) {
    Write-Host "[!] 端口 8000 已被占用，请先运行 .\stop.ps1 停止旧服务！" -ForegroundColor Yellow
    exit 1
}

Write-Host "[+] 正在后台启动超算中心监控预警服务..." -ForegroundColor Green

# 创建 outputs 目录
$outputsDir = Join-Path $scriptDir "outputs"
if (!(Test-Path $outputsDir)) {
    New-Item -ItemType Directory -Path $outputsDir | Out-Null
}
$stderrLogFile = Join-Path $outputsDir "server.stderr.log"
$stdoutLogFile = Join-Path $outputsDir "server.stdout.log"

# Python 端已自行轮转写入 outputs/server.log，这里的重定向只兜底捕获 C 层（FFmpeg）输出与 print，
# 因此不能再指向 server.log（会与轮转抢同一个文件）。启动前做体积保护，避免无限膨胀。
foreach ($redirectLog in @($stderrLogFile, $stdoutLogFile)) {
    if ((Test-Path -LiteralPath $redirectLog) -and ((Get-Item -LiteralPath $redirectLog).Length -gt 64MB)) {
        Move-Item -LiteralPath $redirectLog -Destination "$redirectLog.old" -Force
    }
}

# 使用项目虚拟环境 (venv) 的 Python 解释器
$pythonPath = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "[X] 未找到虚拟环境 Python: $pythonPath" -ForegroundColor Red
    Write-Host "[i] 请先创建虚拟环境并安装依赖：python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor Gray
    exit 1
}

# PID 文件由 Python 进程自己写入真实 PID（venv 的 python.exe 可能只是 launcher，
# $process.Id 不一定是最终服务进程），启动前先清掉上次残留，确保读到的是本次的值。
$pidFile = Join-Path $outputsDir "server.pid"
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue

# 使用 Start-Process 在后台独立启动 Python 进程，stdout/stderr 兜底重定向到 outputs/
$process = Start-Process -FilePath $pythonPath -ArgumentList "main.py" -WorkingDirectory $scriptDir -RedirectStandardOutput $stdoutLogFile -RedirectStandardError $stderrLogFile -WindowStyle Hidden -PassThru

$headers = @{}
if ($env:LAB_MONITOR_USERNAME -and $env:LAB_MONITOR_PASSWORD) {
    $rawCredentials = "$($env:LAB_MONITOR_USERNAME):$($env:LAB_MONITOR_PASSWORD)"
    $encodedCredentials = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($rawCredentials))
    $headers["Authorization"] = "Basic $encodedCredentials"
}

$healthy = $false
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    if ($process.HasExited) { break }
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -Headers $headers -TimeoutSec 2
        if ($health.service -eq "Lab-Monitor" -and $health.status -eq "ok") {
            $healthy = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $healthy) {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
    # 兜底：launcher 与真实服务进程可能不同，按 PID 文件清理残留（校验命令行含 main.py，防误杀）
    if (Test-Path -LiteralPath $pidFile) {
        $recordedPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
        if ($recordedPid -match '^\d+$') {
            $orphan = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedPid" -ErrorAction SilentlyContinue
            if ($orphan -and $orphan.Name -match '^python(\.exe)?$' -and $orphan.CommandLine -like '*main.py*') {
                Stop-Process -Id $recordedPid -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "[X] 服务未通过健康检查，请查看 outputs/server.log！" -ForegroundColor Red
    exit 1
} else {
    $realPid = $process.Id
    if (Test-Path -LiteralPath $pidFile) {
        $recordedPid = (Get-Content -LiteralPath $pidFile -Raw).Trim()
        if ($recordedPid -match '^\d+$') { $realPid = $recordedPid }
    }
    Write-Host "[✓] 服务已成功在后台启动 (PID: $realPid)" -ForegroundColor Green
    Write-Host "[✓] 监控大屏地址: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "[i] 你现在可以放心关闭此终端窗口。" -ForegroundColor Gray
}
