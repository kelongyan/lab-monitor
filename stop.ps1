# F:\lab-monitor\stop.ps1
# 优先通过本机管理接口安全停止；超时后仅强制终止已验证属于本项目的 PID。

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

$pidFile = Join-Path $scriptDir "outputs\server.pid"
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "[i] 未发现 outputs/server.pid，未执行停止操作。" -ForegroundColor Gray
    exit 0
}

# 以实际监听 8000 端口的进程为准（最可靠）；outputs/server.pid 由 Python 进程写入真实 PID，
# 在端口已无监听时作为回查残留进程的依据。
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $listener) {
    # 端口无监听 ≠ 服务已退出：Web 线程可能已死而流水线仍在跑（孤儿进程会与新实例同写一个 DB）。
    # 因此先按 PID 文件回查，确认是本项目的 main.py 进程才终止；确认无残留后才清理 PID 文件。
    $recordedPid = (Get-Content -LiteralPath $pidFile -Raw -ErrorAction SilentlyContinue)
    if ($recordedPid) { $recordedPid = $recordedPid.Trim() }
    if ($recordedPid -match '^\d+$') {
        $orphan = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedPid" -ErrorAction SilentlyContinue
        if ($orphan -and $orphan.Name -match '^python(\.exe)?$' -and $orphan.CommandLine -like '*main.py*') {
            Stop-Process -Id $recordedPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 500
            if (Get-Process -Id $recordedPid -ErrorAction SilentlyContinue) {
                Write-Host "[X] PID $recordedPid（无监听的残留流水线进程）终止失败，请手动检查。" -ForegroundColor Red
                exit 1
            }
            Write-Host "[!] 8000 端口无监听，已终止残留的项目进程 PID $recordedPid。" -ForegroundColor Yellow
            Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
            exit 0
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "[i] 8000 端口无监听进程，且无残留项目进程，已清理过期 PID 文件。" -ForegroundColor Gray
    exit 0
}
$pidToStop = $listener.OwningProcess

$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $pidToStop" -ErrorAction SilentlyContinue
if (-not $processInfo) {
    Write-Host "[X] 无法获取监听进程 $pidToStop 信息，已拒绝终止。" -ForegroundColor Red
    exit 1
}

# 校验：进程必须是 python 且命令行包含 main.py，防止误杀其他占用 8000 的服务
$isPythonMain = $processInfo.Name -match '^python(\.exe)?$' -and $processInfo.CommandLine -like '*main.py*'
if (-not $isPythonMain) {
    Write-Host "[X] PID $pidToStop 不是本项目进程（main.py），已拒绝终止。" -ForegroundColor Red
    exit 1
}

$headers = @{}
if ($env:LAB_MONITOR_USERNAME -and $env:LAB_MONITOR_PASSWORD) {
    $rawCredentials = "$($env:LAB_MONITOR_USERNAME):$($env:LAB_MONITOR_PASSWORD)"
    $encodedCredentials = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($rawCredentials))
    $headers["Authorization"] = "Basic $encodedCredentials"
}

try {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/admin/shutdown" -Headers $headers -TimeoutSec 5 | Out-Null
    Write-Host "[i] 已发送安全停止请求，等待流水线和持久化收尾..." -ForegroundColor Cyan
} catch {
    Write-Host "[!] 安全停止接口无响应，将等待后再决定是否强制停止。" -ForegroundColor Yellow
}

$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline) {
    if (-not (Get-Process -Id $pidToStop -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 500
}

if (Get-Process -Id $pidToStop -ErrorAction SilentlyContinue) {
    Stop-Process -Id $pidToStop -Force
    Write-Host "[!] 安全停止超时，已强制终止已验证的项目进程 PID $pidToStop。" -ForegroundColor Yellow
} else {
    Write-Host "[✓] 服务已安全停止，日志和校准数据已完成落盘。" -ForegroundColor Green
}

Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
