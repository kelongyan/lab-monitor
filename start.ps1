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
$logFile = Join-Path $outputsDir "server.log"
$stdoutLogFile = Join-Path $outputsDir "server.stdout.log"

# 使用 Start-Process 在后台独立启动 Python 进程，并将输出日志定向到 outputs/server.log
$process = Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $scriptDir -RedirectStandardOutput $stdoutLogFile -RedirectStandardError $logFile -WindowStyle Hidden -PassThru

# 保存 PID 到 outputs/server.pid
$process.Id | Out-File -FilePath (Join-Path $outputsDir "server.pid") -Encoding utf8

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
    Remove-Item -LiteralPath (Join-Path $outputsDir "server.pid") -Force -ErrorAction SilentlyContinue
    Write-Host "[X] 服务未通过健康检查，请查看 outputs/server.log！" -ForegroundColor Red
    exit 1
} else {
    Write-Host "[✓] 服务已成功在后台启动 (PID: $($process.Id))" -ForegroundColor Green
    Write-Host "[✓] 监控大屏地址: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "[i] 你现在可以放心关闭此终端窗口。" -ForegroundColor Gray
}
