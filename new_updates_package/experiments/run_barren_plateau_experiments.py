"""
Master script for running all barren plateau experiments.

This orchestrates:
1. Depth-2 vs Depth-6 comparison
2. 3D Pareto front visualization
3. Statistical analysis
"""

import sys
from pathlib import Path
import logging

# Setup
script_file = Path(__file__).resolve()
parent_dir = script_file.parent.parent
sys.path.insert(0, str(parent_dir))

from experiments.barren_plateau_comparison import run_barren_plateau_comparison
from experiments.visualize_pareto_3d import main as visualize_pareto

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run all barren plateau experiments."""
    print("="*80)
    print("BARREN PLATEAU EXPERIMENTS")
    print("="*80)
    
    Path('experiments').mkdir(exist_ok=True)
    
    # Experiment 1: Depth-2 vs Depth-6 comparison
    print("\n[Experiment 1] Depth-2 vs Depth-6 Comparison")
    print("-" * 80)
    try:
        results = run_barren_plateau_comparison(n_samples_per_depth=20, seed=42)
        print("✓ Experiment 1 completed")
    except Exception as e:
        logger.error(f"Experiment 1 failed: {e}", exc_info=True)
    
    # Experiment 2: 3D Pareto front visualization
    print("\n[Experiment 2] 3D Pareto Front Visualization")
    print("-" * 80)
    try:
        visualize_pareto()
        print("✓ Experiment 2 completed")
    except Exception as e:
        logger.error(f"Experiment 2 failed: {e}", exc_info=True)
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    print("\nGenerated files:")
    print("  - experiments/barren_plateau_comparison.png")
    print("  - experiments/barren_plateau_results.json")
    print("  - experiments/barren_plateau_*_data.csv")
    print("  - experiments/pareto_3d.png")
    print("  - experiments/pareto_2d_projections.png")
    print("  - experiments/pareto_data.csv")

if __name__ == "__main__":
    main()

