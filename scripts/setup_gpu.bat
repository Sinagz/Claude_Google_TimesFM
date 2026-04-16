@echo off
:: ============================================================
:: scripts/setup_gpu.bat  (Feature A)
:: Detects CUDA availability and installs the matching PyTorch
:: wheel into the project virtual environment.
::
:: Supported CUDA versions auto-detected via nvidia-smi:
::   CUDA 12.x  → torch cu121 wheel (latest stable)
::   CUDA 11.x  → torch cu118 wheel
::   No GPU     → CPU-only wheel (already installed by setup.bat)
:: ============================================================

setlocal EnableDelayedExpansion

set PROJECT_DIR=%~dp0..
set VENV_PIP=%PROJECT_DIR%\venv\Scripts\pip.exe
set VENV_PYTHON=%PROJECT_DIR%\venv\Scripts\python.exe

echo.
echo  ============================================================
echo   GPU / CUDA Setup for Stock Forecast Pipeline
echo  ============================================================
echo.

if not exist "%VENV_PIP%" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

:: ── Detect CUDA via nvidia-smi ───────────────────────────────
set CUDA_VER=none
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] nvidia-smi not found — no NVIDIA GPU detected.
    goto :cpu_install
)

for /f "tokens=9" %%v in ('nvidia-smi ^| findstr /i "CUDA Version"') do (
    set CUDA_VER=%%v
)

if "!CUDA_VER!"=="none" (
    echo [INFO] Could not parse CUDA version from nvidia-smi output.
    goto :cpu_install
)

echo [INFO] Detected CUDA Version: !CUDA_VER!

:: Major version only
for /f "tokens=1 delims=." %%m in ("!CUDA_VER!") do set CUDA_MAJOR=%%m

:: ── Choose wheel index ───────────────────────────────────────
if !CUDA_MAJOR! GEQ 12 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    set TORCH_TAG=cu121
    goto :gpu_install
)
if !CUDA_MAJOR! == 11 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    set TORCH_TAG=cu118
    goto :gpu_install
)

echo [INFO] CUDA !CUDA_VER! is older than 11.x — falling back to CPU build.
goto :cpu_install

:gpu_install
echo.
echo [INFO] Installing PyTorch with CUDA !TORCH_TAG! support ...
echo        This may take several minutes (wheel is ~2 GB).
echo.
"%VENV_PIP%" install --upgrade ^
    torch torchvision torchaudio ^
    --index-url %TORCH_INDEX%

if %errorlevel% neq 0 (
    echo [ERROR] GPU torch install failed — falling back to CPU build.
    goto :cpu_install
)

echo.
echo [SUCCESS] PyTorch installed with CUDA !TORCH_TAG! support.
goto :verify

:cpu_install
echo.
echo [INFO] Installing CPU-only PyTorch ...
"%VENV_PIP%" install --upgrade torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cpu
echo [INFO] CPU-only PyTorch installed.
goto :verify

:verify
echo.
echo [INFO] Verifying torch installation ...
"%VENV_PYTHON%" -c "import torch; print('  torch version :', torch.__version__); print('  CUDA available:', torch.cuda.is_available()); print('  Device        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo.
pause
endlocal
