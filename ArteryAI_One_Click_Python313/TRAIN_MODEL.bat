@echo off
setlocal
cd /d "%~dp0backend"
if not exist venv\Scripts\python.exe (
  echo Backend virtual environment not found.
  echo Run START_BACKEND.bat once first, then run this file again.
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
echo Training U-Net using dataset\images and dataset\masks ...
python -m training.train --epochs 60 --batch-size 4
if errorlevel 1 (
  echo Training failed. Read the error above.
  pause
  exit /b 1
)
echo Training complete. Check backend\model_weights\unet.pth
pause
