@echo off
:: ╔══════════════════════════════════════════════════════╗
:: ║     AutoValue AI - One-Click Setup (Windows)        ║
:: ╚══════════════════════════════════════════════════════╝

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║       🚗  AUTOVALUE AI  -  SETUP               ║
echo  ╚══════════════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ❌ Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo  ✅ Python found

:: Create venv
if not exist "venv" (
    echo  📦 Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

echo  📥 Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo  🤖 Training ML models...
python model_trainer.py

echo.
echo  ✅ Setup complete!
echo  🚀 Launching AutoValue AI...
echo.
streamlit run app.py

pause
