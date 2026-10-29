@echo off
cd /d "%~dp0backend"
if not exist .venv (
  py -3.11 -m venv .venv
  if errorlevel 1 pause & exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
pause
