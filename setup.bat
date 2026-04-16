@echo off
REM ============================================================
REM  Setup script — Multi-Agent Stock Forecasting System
REM  Run once from the project root: setup.bat
REM ============================================================

echo [SETUP] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.10+ and try again.
    exit /b 1
)

echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip

echo [SETUP] Installing dependencies (this may take several minutes)...
pip install -r requirements.txt
pip install plotly

echo [SETUP] Downloading NLTK data...
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

echo [SETUP] Creating required directories...
if not exist data         mkdir data
if not exist outputs      mkdir outputs
if not exist logs         mkdir logs
if not exist models\cache mkdir models\cache

echo.
echo ============================================================
echo  SETUP COMPLETE
echo ============================================================
echo  To run the system:
echo    venv\Scripts\activate
echo    python main.py
echo.
echo  For GPU acceleration (CUDA auto-detection):
echo    scripts\setup_gpu.bat
echo.
echo  To schedule daily runs (Mon-Fri 07:00 AM):
echo    Run schedule_setup.bat as Administrator
echo ============================================================
pause
