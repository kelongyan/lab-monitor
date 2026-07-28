# F:\lab-monitor\stop.ps1
# 优先通过本机管理接口安全停止；超时后仅强制终止已验证属于本项目的 PID。

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

$pidFile = Join-Path $scriptDir "outputs\server.pid"
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "[i] 未发现 outputs/server.pid，未执行停止操作。" -ForegroundColor Gray
    exit 0
}

$pidText = (Get-Content -LiteralPath $pidFile -Raw -ErrorAction SilentlyContinue).Trim()
$pidToStop = 0
if (-not [int]::TryParse($pidText, [ref]$pidToStop)) {
    Write-Host "[X] PID 文件内容无效，已拒绝终止任何进程。" -ForegroundColor Red
    exit 1
}

$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $pidToStop" -ErrorAction SilentlyContinue
if (-not $processInfo) {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "[i] 记录的进程已退出，已清理过期 PID 文件。" -ForegroundColor Gray
    exit 0
}

$ownsPort = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.OwningProcess -eq $pidToStop }
$isPythonMain = $processInfo.Name -match '^python(\.exe)?$' -and $processInfo.CommandLine -like '*main.py*'
if (-not $ownsPort -or -not $isPythonMain) {
    Write-Host "[X] PID $pidToStop 未通过项目进程校验，已拒绝终止。" -ForegroundColor Red
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
