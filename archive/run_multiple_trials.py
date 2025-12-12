"""
Run multiple verification trials with different random seeds
This gives us statistically meaningful results with error bars
"""
import subprocess
import json
import numpy as np
from pathlib import Path

def run_single_trial(seed):
    """Run verification with a specific seed"""
    print(f"\n{'='*60}")
    print(f"TRIAL {seed+1}/10 (Seed={seed})")
    print(f"{'='*60}")
    
    # Modify verify_real.py to use this seed
    # For now, just run it and parse output
    cmd = f"python verify_real.py --seed {seed}"
    
    # This is a placeholder - you'll run verify_real.py manually 10 times
    # and record results
    return {
        'seed': seed,
        'random_acc': None,
        'gnn_acc': None,
        'trans_scalar_acc': None,
        'grid_acc': None,
        'bo_acc': None,
        'ours_acc': None,
        'random_depth': None,
        'gnn_depth': None,
        'trans_scalar_depth': None,
        'grid_depth': None,
        'bo_depth': None,
        'ours_depth': None
    }

def analyze_results(results):
    """Compute statistics across trials"""
    methods = ['random', 'gnn', 'trans_scalar', 'grid', 'bo', 'ours']
    
    stats = {}
    for method in methods:
        accs = [r[f'{method}_acc'] for r in results if r[f'{method}_acc'] is not None]
        depths = [r[f'{method}_depth'] for r in results if r[f'{method}_depth'] is not None]
        
        stats[method] = {
            'acc_mean': np.mean(accs) if accs else 0,
            'acc_std': np.std(accs) if accs else 0,
            'depth_mean': np.mean(depths) if depths else 0,
            'depth_std': np.std(depths) if depths else 0
        }
    
    return stats

def print_summary(stats):
    """Print final summary"""
    print("\n" + "="*60)
    print("FINAL RESULTS (10 TRIALS)")
    print("="*60)
    
    for method, s in stats.items():
        print(f"{method.upper():15s}: Acc={s['acc_mean']:.1f}±{s['acc_std']:.1f}%, Depth={s['depth_mean']:.1f}±{s['depth_std']:.1f}")
    
    # Check if Ours beats baselines
    ours = stats['ours']
    baselines = ['random', 'gnn', 'trans_scalar', 'grid', 'bo']
    
    print("\n--- Statistical Comparison ---")
    for baseline in baselines:
        b = stats[baseline]
        if ours['acc_mean'] > b['acc_mean']:
            diff = ours['acc_mean'] - b['acc_mean']
            print(f"✓ Ours > {baseline}: +{diff:.1f}% accuracy")

if __name__ == "__main__":
    print("="*60)
    print("MULTI-TRIAL VERIFICATION SETUP")
    print("="*60)
    print("\nThis script will help you run 10 trials.")
    print("\nMANUAL STEPS:")
    print("1. Run verify_real.py 10 times")
    print("2. Record results in results.json")
    print("3. Run this script to analyze")
    print("\nEstimated time: 15-20 hours total (can run overnight)")
