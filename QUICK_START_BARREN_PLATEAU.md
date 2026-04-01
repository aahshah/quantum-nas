# Quick Start: Barren Plateau Experiments

## Status Check

✅ **PyTorch is working!** The DLL issue appears to be resolved.

## Run Experiments

### Option 1: Full Experiment (Recommended)
```bash
python experiments/barren_plateau_comparison.py
```

This will:
- Generate 20 Depth-2 and 20 Depth-6 architectures
- Compute gradient variance for each
- Perform statistical comparison
- Generate visualization plots

### Option 2: Simplified Version (No Full Dependencies)
```bash
python experiments/run_without_torch.py
```

This uses only PennyLane and NumPy (no full quantum_nas system).

### Option 3: All Experiments
```bash
python experiments/run_barren_plateau_experiments.py
```

## If You Still Get DLL Errors

### Quick Fix (Windows)
1. **Run the batch file**:
   ```bash
   fix_pytorch_windows.bat
   ```

2. **Or manually**:
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip cache purge
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Visual C++ Redistributables** (if needed):
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Install and restart computer

## Expected Output

When running `barren_plateau_comparison.py`, you should see:

```
[1/4] Analyzing Depth-2 architectures...
  Progress: 5/20
  Progress: 10/20
  ...

[2/4] Analyzing Depth-6 architectures...
  ...

[3/4] Statistical Analysis...
Results:
  Depth-2 Gradient Variance: 0.000123 ± 0.000045
  Depth-6 Gradient Variance: 0.000001 ± 0.000000
  Improvement Factor: 123.0×
  T-test p-value: 0.000001
  ✓ Statistically significant difference (p < 0.05)

[4/4] Generating plots...
  Saved plot to experiments/barren_plateau_comparison.png
```

## Output Files

- `experiments/barren_plateau_comparison.png` - Comparison plots
- `experiments/barren_plateau_results.json` - Statistical results
- `experiments/barren_plateau_depth2_data.csv` - Depth-2 data
- `experiments/barren_plateau_depth6_data.csv` - Depth-6 data

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'barren_plateau_analyzer'"
**Solution**: Make sure you're in the project root directory:
```bash
cd C:\Users\shahs\OneDrive\Desktop\TRA-Energy_Effecient\quantum_nas
python experiments/barren_plateau_comparison.py
```

### Issue: "ImportError: cannot import name 'BarrenPlateauAnalyzer'"
**Solution**: Check that `barren_plateau_analyzer.py` exists in the root directory.

### Issue: NaN values in gradient variance
**Solution**: This can happen if:
- Circuit has no parameters
- PennyLane device setup issue
- Try reducing `n_samples` in `BarrenPlateauAnalyzer` (default: 50)

## Next Steps

1. ✅ Run the comparison experiment
2. ✅ Generate 3D Pareto visualizations: `python experiments/visualize_pareto_3d.py`
3. ✅ Review results in `experiments/barren_plateau_results.json`

## Need Help?

See `BARREN_PLATEAU_INTEGRATION.md` for detailed documentation.

