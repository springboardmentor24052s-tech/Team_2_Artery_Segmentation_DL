@echo off
setlocal
cd /d "%~dp0frontend"
where node >nul 2>nul
if errorlevel 1 (
  echo Node.js 20 LTS or newer is required. Install it from https://nodejs.org/
  pause
  exit /b 1
)
if not exist node_modules (
  npm install
  if errorlevel 1 (
    echo Frontend dependency installation failed. Review the error above.
    pause
    exit /b 1
  )
)
npm run dev -- --host 127.0.0.1
pause
