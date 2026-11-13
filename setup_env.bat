ls
@echo off
REM ============================================================
REM  setup_env.bat  –  Create and activate the project venv
REM  Run once from the project root:  .\setup_env.bat
REM ============================================================

echo [1/4] Creating virtual environment at .\venv ...
REM Use Python 3.11 (adjust path if your Python is installed elsewhere)
set PYTHON="C:\Users\darsh\AppData\Local\Programs\Python\Python311\python.exe"
if not exist %PYTHON% (
    echo Trying system python ...
    set PYTHON=python
)
%PYTHON% -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found or venv creation failed.
    exit /b 1
)

echo [2/4] Activating virtual environment ...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip ...
python -m pip install --upgrade pip

echo [4/4] Installing project dependencies from requirements.txt ...
pip install -r requirements.txt

echo.
echo ============================================================
echo  Setup complete!
echo  To activate in a new terminal run:
echo    venv\Scripts\activate.bat
echo ============================================================
