Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RepoRoot "backend"
$EnvFile    = Join-Path $BackendDir ".env.worker"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"

# 0. Kiem tra file env
if (-not (Test-Path $EnvFile)) {
    Write-Error "Khong tim thay $EnvFile. Hay copy tu backend/.env.example va dien gia tri that."
    exit 1
}

$envContent = Get-Content $EnvFile -Raw
if ($envContent -match "<PASSWORD>|<HOST>|<ACCESS_KEY>|<SECRET_KEY>") {
    Write-Error "backend/.env.worker con chua gia tri placeholder (<...>). Hay dien day du truoc khi chay."
    exit 1
}

# 1. Kiem tra venv
if (-not (Test-Path $VenvPython)) {
    Write-Error "Venv chua duoc setup. Chay install_worker_deps.ps1 truoc."
    exit 1
}

# 2. Kiem tra / khoi dong Qdrant Docker
Write-Host "[Qdrant] Kiem tra container..." -ForegroundColor Cyan
$qdrantStatus = docker inspect --format "{{.State.Status}}" "video-understanding-qdrant-1" 2>$null

if ($qdrantStatus -eq "running") {
    Write-Host "[Qdrant] Dang chay." -ForegroundColor Green
} elseif ($qdrantStatus -eq "exited" -or $qdrantStatus -eq "created") {
    Write-Host "[Qdrant] Khoi dong container video-understanding-qdrant-1 ..." -ForegroundColor Yellow
    docker start video-understanding-qdrant-1 | Out-Null
    Start-Sleep -Seconds 3
    Write-Host "[Qdrant] Da khoi dong." -ForegroundColor Green
} else {
    Write-Host "[Qdrant] Khong tim thay container, chay moi..." -ForegroundColor Yellow
    docker run -d --name video-understanding-qdrant-1 -p 6333:6333 -p 6334:6334 -v video-understanding_qdrant_data:/qdrant/storage -e QDRANT__SERVICE__HTTP_PORT=6333 qdrant/qdrant:latest | Out-Null
    Start-Sleep -Seconds 5
    Write-Host "[Qdrant] Da khoi dong container moi." -ForegroundColor Green
}

# 3. Load env
Write-Host "[Env] Load tu $EnvFile ..." -ForegroundColor Cyan
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $idx = $line.IndexOf("=")
        if ($idx -gt 0) {
            $key   = $line.Substring(0, $idx).Trim()
            $value = $line.Substring($idx + 1).Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# 4. PYTHONPATH va bien moi truong worker
$env:PYTHONPATH = $BackendDir
$env:SB_DISABLE_K2 = "1"

$promDir = Join-Path $BackendDir "prometheus_multiproc"
if (-not (Test-Path $promDir)) { New-Item -ItemType Directory -Path $promDir | Out-Null }
$env:PROMETHEUS_MULTIPROC_DIR = $promDir

# 5. Khoi dong worker
Write-Host ""
Write-Host "=== Khoi dong Celery GPU worker ===" -ForegroundColor Cyan
Write-Host "    QDRANT   : localhost:6333"
Write-Host "    DEVICE   : $($env:DEVICE)"
Write-Host "    Pool     : solo (GPU serial, tranh VRAM OOM)"
Write-Host ""
Write-Host "Nhan Ctrl+C de dung worker." -ForegroundColor DarkGray
Write-Host ""

Push-Location $BackendDir
try {
    & $VenvPython -m celery -A workers.celery_app worker --loglevel=info --pool=solo --concurrency=1 --prefetch-multiplier=1 --without-gossip --without-mingle
} finally {
    Pop-Location
}
