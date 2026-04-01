"""
Wrapper script to run QFI correlation experiment.
Run this from the project root directory.
"""

import sys
from pathlib import Path

# Ensure we're in the right directory
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Change to project root
import os
original_cwd = os.getcwd()
os.chdir(project_root)

try:
    # Now run the experiment
    import subprocess
    result = subprocess.run(
        [sys.executable, 'experiments/validate_qfi_correlation.py'],
        cwd=project_root,
        capture_output=False
    )
    sys.exit(result.returncode)
finally:
    os.chdir(original_cwd)


