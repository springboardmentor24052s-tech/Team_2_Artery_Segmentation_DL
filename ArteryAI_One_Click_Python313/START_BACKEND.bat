@echo off
setlocal
cd /d "%~dp0backend"
if not exist venv\Scripts\python.exe (
  py -3.11 -m venv venv
  if errorlevel 1 py -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Backend dependency installation failed. Review the error above.
  pause
  exit /b 1
)
python app.py
pause
