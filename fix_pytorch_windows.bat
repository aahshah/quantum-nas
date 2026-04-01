@echo off
REM Fix PyTorch DLL loading issue on Windows

echo ========================================
echo PyTorch DLL Fix for Windows
echo ========================================
echo.

echo Step 1: Uninstalling current PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 2: Clearing pip cache...
pip cache purge

echo.
echo Step 3: Installing CPU-only PyTorch (most stable on Windows)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Step 4: Testing PyTorch import...
python -c "import torch; print('SUCCESS: PyTorch', torch.__version__, 'installed correctly')"

echo.
echo ========================================
echo If you still get errors:
echo 1. Install Visual C++ Redistributables:
echo    https://aka.ms/vs/17/release/vc_redist.x64.exe
echo 2. Restart your computer
echo 3. Run this script again
echo ========================================
pause

