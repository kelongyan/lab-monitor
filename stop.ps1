# F:\lab-monitor\stop.ps1
# 一键停止超算中心监控预警服务并精准释放端口

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

$pidFile = Join-Path $scriptDir "outputs\server.pid"
$stopped = $false

# 1. 尝试从 PID 文件关闭
if (Test-Path $pidFile) {
    $pidToKill = Get-Content -Path $pidFile -ErrorAction SilentlyContinue
    if ($pidToKill) {
        $proc = Get-Process -Id $pidToKill -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
            Write-Host "[✓] 已精准停止 PID $pidToKill 对应的后台服务。" -ForegroundColor Green
            $stopped = $true
        }
    }
    Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
}

# 2. 检查并清理所有占用 8000 端口的残余进程
$portConns = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($portConns) {
    foreach ($conn in $portConns) {
        $owningPid = $conn.OwningProcess
        if ($owningPid -gt 0) {
            Stop-Process -Id $owningPid -Force -ErrorAction SilentlyContinue
            Write-Host "[✓] 已清理占用 8000 端口的残余进程 (PID: $owningPid)。" -ForegroundColor Green
            $stopped = $true
        }
    }
}

if ($stopped) {
    Write-Host "[✓] 服务已成功完全关闭！" -ForegroundColor Green
} else {
    Write-Host "[i] 未发现正在运行的服务或端口占用。" -ForegroundColor Gray
}
