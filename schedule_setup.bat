@echo off
:: ============================================================
:: schedule_setup.bat  (Feature F)
:: Registers a Windows Task Scheduler job to run the forecasting
:: pipeline every weekday at 07:00 AM.
:: Run this script once as Administrator.
:: ============================================================

setlocal

:: ── Configuration ────────────────────────────────────────────
set TASK_NAME=StockForecastPipeline
set PROJECT_DIR=%~dp0
:: Strip trailing backslash
if "%PROJECT_DIR:~-1%"=="\" set PROJECT_DIR=%PROJECT_DIR:~0,-1%

set VENV_PYTHON=%PROJECT_DIR%\venv\Scripts\python.exe
set MAIN_SCRIPT=%PROJECT_DIR%\main.py
set RUN_TIME=07:00
set SCHEDULE_DAYS=MON,TUE,WED,THU,FRI

echo.
echo  ============================================================
echo   Stock Forecast Pipeline -- Task Scheduler Setup
echo  ============================================================
echo   Project dir : %PROJECT_DIR%
echo   Python      : %VENV_PYTHON%
echo   Schedule    : %SCHEDULE_DAYS% at %RUN_TIME%
echo  ============================================================
echo.

:: ── Check Python exists ──────────────────────────────────────
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at:
    echo         %VENV_PYTHON%
    echo.
    echo  Please run setup.bat first to create the virtual environment.
    pause
    exit /b 1
)

:: ── Delete existing task if present ─────────────────────────
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Removing existing task "%TASK_NAME%" ...
    schtasks /delete /tn "%TASK_NAME%" /f >nul
)

:: ── Register new task ────────────────────────────────────────
echo [INFO] Registering scheduled task ...

schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "\"%VENV_PYTHON%\" \"%MAIN_SCRIPT%\"" ^
    /sc weekly ^
    /d %SCHEDULE_DAYS% ^
    /st %RUN_TIME% ^
    /ru "%USERNAME%" ^
    /rl HIGHEST ^
    /f

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to create scheduled task.
    echo         Try running this script as Administrator.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Scheduled task "%TASK_NAME%" created.
echo           The pipeline will run every weekday at %RUN_TIME%.
echo.
echo  To view the task:   schtasks /query /tn "%TASK_NAME%" /fo LIST
echo  To run it now:      schtasks /run /tn "%TASK_NAME%"
echo  To delete it:       schtasks /delete /tn "%TASK_NAME%" /f
echo.
pause
endlocal
