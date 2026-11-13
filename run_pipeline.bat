@echo off
REM ============================================================
REM  run_pipeline.bat
REM  One-click CMD launcher for the PPG Arrhythmia Detection
REM  pipeline.  Double-click or run from any CMD window.
REM
REM  Usage (from project root in CMD):
REM    run_pipeline.bat                          full run
REM    run_pipeline.bat --quick                  smoke test
REM    run_pipeline.bat --skip-preprocess        skip Phase 1
REM    run_pipeline.bat --skip-features          skip Phase 2
REM    run_pipeline.bat --skip-train             skip Phase 3
REM    run_pipeline.bat --skip-benchmark         skip Phase 4
REM    run_pipeline.bat --skip-export            skip Phase 5
REM    run_pipeline.bat --no-gpu                 force CPU
REM    run_pipeline.bat --models xgb cnn         subset of models
REM    run_pipeline.bat --epochs 50 --batch 32
REM    run_pipeline.bat --legacy                 also run old pipeline
REM
REM  All extra arguments are forwarded directly to run_pipeline.py
REM ============================================================

setlocal EnableDelayedExpansion

REM ── Move to the project root (same folder as this .bat) ───────────────────
cd /d "%~dp0"

echo.
echo ============================================================
echo   PPG Arrhythmia Detection Pipeline
echo   CB.SC.P2AIE25005_DARSHAN_B
echo ============================================================
echo.

REM ── Locate the venv Python ────────────────────────────────────────────────
set VENV_PYTHON=venv_gpu\Scripts\python.exe

if exist "%VENV_PYTHON%" (
    echo [OK] Found venv: %CD%\%VENV_PYTHON%
) else (
    echo [WARN] venv_gpu not found at %CD%\%VENV_PYTHON%
    echo        Trying to find venv_gpu\Scripts\activate.bat ...
    if exist "venv_gpu\Scripts\activate.bat" (
        call venv_gpu\Scripts\activate.bat
        set VENV_PYTHON=python
    ) else (
        echo.
        echo [ERROR] Virtual environment ^'venv_gpu^' not found.
        echo         Create it first:
        echo           python -m venv venv_gpu
        echo           venv_gpu\Scripts\activate.bat
        echo           pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

REM ── Activate the venv so subprocesses inherit it ──────────────────────────
call venv_gpu\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate venv_gpu.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

REM ── Sanity check: numpy must be importable ────────────────────────────────
"%VENV_PYTHON%" -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] numpy is not installed in venv_gpu.
    echo         Run:  venv_gpu\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo [OK] numpy found in venv.

REM ── Print active Python path for confirmation ─────────────────────────────
echo [OK] Python: 
"%VENV_PYTHON%" -c "import sys; print('       ' + sys.executable)"
echo.

REM ── Log start time ────────────────────────────────────────────────────────
echo Start time: %DATE% %TIME%
echo Arguments : %*
echo.
echo ============================================================
echo   Launching run_pipeline.py ...
echo ============================================================
echo.

REM ── Run the pipeline, forwarding all arguments ────────────────────────────
"%VENV_PYTHON%" run_pipeline.py %*

REM ── Capture exit code ─────────────────────────────────────────────────────
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================
echo   End time: %DATE% %TIME%
if %EXIT_CODE% EQU 0 (
    echo   Status  : SUCCESS
) else (
    echo   Status  : FAILED  ^(exit code %EXIT_CODE%^)
)
echo ============================================================
echo.

REM ── Keep window open so the user can read the output ─────────────────────
REM (Remove the next two lines if running from a CI/automated environment)
echo Press any key to close this window ...
pause >nul

exit /b %EXIT_CODE%
