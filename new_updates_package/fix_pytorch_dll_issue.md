# Fixing PyTorch DLL Loading Error on Windows

## Error
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "c10.dll" or one of its dependencies.
```

## Solutions (Try in Order)

### Solution 1: Install Visual C++ Redistributables
PyTorch requires Visual C++ Redistributables. Download and install:
- **Visual C++ Redistributable 2015-2022**: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Solution 2: Reinstall PyTorch
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall PyTorch (CPU version - more stable on Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR if you need CUDA (if you have NVIDIA GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Solution 3: Use Conda (Recommended for Windows)
```bash
# Install Miniconda/Anaconda, then:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Solution 4: Check for Conflicting Packages
```bash
# List all installed packages
pip list | findstr torch

# Remove any conflicting packages
pip uninstall torch torchvision torchaudio
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Solution 5: Environment Variables
Add to system PATH (if missing):
- `C:\Windows\System32`
- `C:\Windows\SysWOW64`

### Solution 6: Run as Administrator
Sometimes DLL loading requires admin privileges:
1. Right-click Command Prompt/PowerShell
2. Select "Run as Administrator"
3. Try running the script again

## Quick Test
After fixing, test PyTorch:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should be False for CPU version
```

## Alternative: Use CPU-Only Version
If GPU is not critical, use CPU-only PyTorch (more stable on Windows):
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

