# Quick Start Guide - Running GQAS Experiments

## Fixing Import Issues

If you get import errors when running experiments, here's how to fix them:

### Option 1: Run from Project Root (Recommended)

```bash
# Make sure you're in the project root directory
cd C:\Users\shahs\OneDrive\Desktop\TRA-Energy_Effecient\quantum_nas

# Run the experiment
python experiments/validate_qfi_correlation_fixed.py
```

### Option 2: Use the Wrapper Script

```bash
# From project root
python run_qfi_experiment.py
```

### Option 3: Manual Import Test

```python
# test_import.py (run from project root)
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

# Test import
from quantum_nas import HARDWARE_SPECS
print(f"Success! Found {len(HARDWARE_SPECS)} hardware specs")
```

## File Structure

```
quantum_nas/
├── quantum_nas.py          # Main library
├── qfi_calculator.py        # QFI calculator (NEW)
├── gradient_flow.py         # Gradient flow analyzer (NEW)
├── experiments/
│   ├── validate_qfi_correlation_fixed.py  # Fixed experiment script
│   └── run_all_experiments.py
└── run_qfi_experiment.py    # Wrapper script
```

## Common Issues

### Issue: "cannot import name 'HARDWARE_SPECS'"
**Solution**: Make sure you're running from the project root, not from the experiments/ directory.

### Issue: "qfi_calculator not found"
**Solution**: The script has fallbacks - it will use simplified QFI calculation if the module isn't found.

### Issue: "gradient_flow not found"
**Solution**: The script has fallbacks - it will use simplified gradient flow analysis.

## Running Experiments

1. **Test imports first**:
   ```bash
   python -c "from quantum_nas import HARDWARE_SPECS; print('OK')"
   ```

2. **Run QFI correlation experiment**:
   ```bash
   python experiments/validate_qfi_correlation_fixed.py
   ```

3. **Check results**:
   - `experiments/qfi_correlation_results.json`
   - `experiments/qfi_correlation_data.csv`
   - `experiments/qfi_correlation_results.png`

## Next Steps

Once imports work:
1. Run Experiment 1 (QFI correlation)
2. Implement Experiments 2-4
3. Generate paper figures
4. Write paper


