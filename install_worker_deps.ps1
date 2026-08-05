Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RepoRoot "backend"
$VenvDir    = Join-Path $BackendDir ".venv"
$PythonExe  = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "=== GPU Worker - dependency installer ===" -ForegroundColor Cyan

# 1. Tao / kiem tra venv Python 3.11
#    uv venv hay tao venv voi Python 3.12 (managed) thay vi Python 3.11 tren system.
#    De dam bao dung 3.11, dung truc tiep "python" (3.11.9 tren PATH).
$needCreate = $false
if (-not (Test-Path $PythonExe)) {
    $needCreate = $true
} else {
    $ver = & $PythonExe -c "import sys; print(sys.version_info.major, sys.version_info.minor)"
    if ($ver -notmatch "^3 11") {
        Write-Host "[1/3] Venv dang dung Python $ver (can 3.11), xoa va tao lai..." -ForegroundColor Yellow
        Remove-Item $VenvDir -Recurse -Force
        $needCreate = $true
    }
}

if ($needCreate) {
    Write-Host "[1/3] Tao venv Python 3.11 tai $VenvDir ..." -ForegroundColor Yellow
    # Dung "python" tren PATH (3.11.9) de tao venv co pip
    python -m venv $VenvDir
} else {
    Write-Host "[1/3] Venv Python 3.11 da ton tai - bo qua." -ForegroundColor Green
}

# 2. Cai PyTorch CUDA 12.6 voi uv pip (khong can pip trong venv)
Write-Host "[2/3] Cai PyTorch CUDA 12.6 ... (co the mat 5-15 phut)" -ForegroundColor Yellow

uv pip install --python $PythonExe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

$cudaOk = & $PythonExe -c "import torch; print(torch.cuda.is_available())"
if ($cudaOk -eq "True") {
    $gpuName = & $PythonExe -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "    CUDA OK - GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Warning "torch.cuda.is_available() = False. Kiem tra lai driver NVIDIA."
}

# 3. Cai requirements.txt
Write-Host "[3/3] Cai requirements.txt ..." -ForegroundColor Yellow
$RequirementsFile = Join-Path $BackendDir "requirements.txt"
uv pip install --python $PythonExe -r $RequirementsFile

Write-Host ""
Write-Host "=== Xong! Chay start_worker.ps1 de bat dau worker. ===" -ForegroundColor Cyan
