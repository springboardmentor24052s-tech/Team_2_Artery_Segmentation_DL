@echo off
setlocal EnableExtensions
title ArteryAI - Full Project Launcher
set "ROOT=%~dp0"
echo.
echo  ==========================================
echo       ArteryAI - One Click Launcher
echo  ==========================================
echo.

where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_CMD=py -3.13"
  ) else (
    py -3.11 --version >nul 2>nul
    if not errorlevel 1 (
      set "PYTHON_CMD=py -3.11"
    ) else (
      set "PYTHON_CMD=py"
    )
  )
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Install Python 3.13 or 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
  )
  set "PYTHON_CMD=python"
)

where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: Node.js is not installed or not in PATH.
  echo Install Node.js 20 or newer from https://nodejs.org/
  pause
  exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
  echo ERROR: npm was not found. Reinstall Node.js with npm included.
  pause
  exit /b 1
)

if not exist "%ROOT%backend\app.py" (
  echo ERROR: backend\app.py not found. Extract the complete ZIP and run this file from the project folder.
  pause
  exit /b 1
)
if not exist "%ROOT%frontend\package.json" (
  echo ERROR: frontend\package.json not found. Extract the complete ZIP and run this file from the project folder.
  pause
  exit /b 1
)

echo [1/5] Preparing Python environment...
if not exist "%ROOT%backend\venv\Scripts\python.exe" (
  pushd "%ROOT%backend"
  %PYTHON_CMD% -m venv venv
  if errorlevel 1 (
    echo ERROR: Could not create Python virtual environment.
    popd
    pause
    exit /b 1
  )
  popd
)

echo [2/5] Installing/checking backend packages...
pushd "%ROOT%backend"
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Backend dependency installation failed.
  popd
  pause
  exit /b 1
)
call venv\Scripts\deactivate.bat
popd

echo [3/5] Installing/checking frontend packages...
pushd "%ROOT%frontend"
if not exist node_modules (
  call npm install
) else (
  echo node_modules already exists; skipping npm install.
)
if errorlevel 1 (
  echo ERROR: Frontend dependency installation failed.
  popd
  pause
  exit /b 1
)
popd

echo [4/5] Starting Flask backend in a separate window...
start "ArteryAI Backend" cmd /k "cd /d ""%ROOT%backend"" && call venv\Scripts\activate.bat && python app.py"

echo Waiting for backend startup...
timeout /t 4 /nobreak >nul

echo [5/5] Starting React frontend...
echo.
echo When Vite is ready, open http://localhost:5173 in your browser.
echo Keep both terminal windows open while using ArteryAI.
echo.
pushd "%ROOT%frontend"
call npm run dev
set "FRONTEND_EXIT=%ERRORLEVEL%"
popd

echo.
echo Frontend process stopped with exit code %FRONTEND_EXIT%.
echo The backend window may still be running. Close it separately when finished.
pause
exit /b %FRONTEND_EXIT%
