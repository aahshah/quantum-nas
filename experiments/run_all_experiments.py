"""
Master Experiment Runner for GQAS Publication

Runs all 4 validation experiments:
1. QFI vs Actual Trainability Correlation
2. Architecture Search Efficiency Comparison
3. Hardware Validation
4. Ablation Study

Author: GQAS Research Team
Target: NeurIPS 2026 / Nature Quantum Information
"""

import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime

# Add parent directory to path
sys.path.append('..')

def run_experiment_1():
    """QFI vs Actual Trainability Correlation"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: QFI vs Actual Trainability Correlation")
    print("="*80)
    
    from validate_qfi_correlation import run_validation_experiment
    
    results, df = run_validation_experiment(n_samples=200, seed=42)
    
    return {
        'experiment': 'qfi_correlation',
        'status': 'completed',
        'results': results,
        'success': results.get('success_criteria_met', False)
    }

def run_experiment_2():
    """Architecture Search Efficiency Comparison"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Architecture Search Efficiency")
    print("="*80)
    
    # Import and run search comparison
    from search_efficiency_comparison import run_search_comparison
    
    results = run_search_comparison(
        n_runs=5,
        pop_size=50,
        generations=50,
        seed=42
    )
    
    return {
        'experiment': 'search_efficiency',
        'status': 'completed',
        'results': results,
        'success': results.get('speedup', 1.0) >= 1.3  # 30%+ speedup
    }

def run_experiment_3():
    """Hardware Validation"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Hardware Validation")
    print("="*80)
    
    # Import and run hardware validation
    from hardware_validation import run_hardware_validation
    
    results = run_hardware_validation(
        n_architectures=10,
        hardware_platforms=['ibm_quantum', 'ionq_aria', 'simulator']
    )
    
    return {
        'experiment': 'hardware_validation',
        'status': 'completed',
        'results': results,
        'success': results.get('gqas_degradation', 1.0) < 0.25  # <25% degradation
    }

def run_experiment_4():
    """Ablation Study"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Ablation Study")
    print("="*80)
    
    # Import and run ablation
    from ablation_study import run_ablation_study
    
    results = run_ablation_study(
        n_samples=50,
        seed=42
    )
    
    return {
        'experiment': 'ablation',
        'status': 'completed',
        'results': results,
        'success': True  # Ablation is descriptive, not pass/fail
    }

def main():
    """Run all experiments and generate final report."""
    print("="*80)
    print("GQAS PUBLICATION EXPERIMENTS")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create results directory
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    start_time = time.time()
    
    try:
        # Experiment 1: QFI Correlation
        print("\n[1/4] Starting Experiment 1...")
        exp1_results = run_experiment_1()
        all_results['experiment_1'] = exp1_results
        
        # Experiment 2: Search Efficiency
        print("\n[2/4] Starting Experiment 2...")
        exp2_results = run_experiment_2()
        all_results['experiment_2'] = exp2_results
        
        # Experiment 3: Hardware Validation
        print("\n[3/4] Starting Experiment 3...")
        exp3_results = run_experiment_3()
        all_results['experiment_3'] = exp3_results
        
        # Experiment 4: Ablation
        print("\n[4/4] Starting Experiment 4...")
        exp4_results = run_experiment_4()
        all_results['experiment_4'] = exp4_results
        
    except Exception as e:
        print(f"\nERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['error'] = str(e)
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    success_count = sum(1 for exp in all_results.values() 
                       if isinstance(exp, dict) and exp.get('success', False))
    total_experiments = len([k for k in all_results.keys() if k.startswith('experiment')])
    
    print(f"\nExperiments Completed: {total_experiments}/4")
    print(f"Successful: {success_count}/{total_experiments}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    
    # Publication readiness check
    print("\nPublication Readiness Checklist:")
    checks = {
        'QFI Correlation (ρ > 0.70)': all_results.get('experiment_1', {}).get('success', False),
        'Search Speedup (≥30%)': all_results.get('experiment_2', {}).get('success', False),
        'Hardware Robustness (<25% degradation)': all_results.get('experiment_3', {}).get('success', False),
        'Ablation Study Complete': all_results.get('experiment_4', {}).get('status') == 'completed'
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    all_passed = all(checks.values())
    
    # Save final results
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'experiments': all_results,
        'summary': {
            'total_experiments': total_experiments,
            'successful': success_count,
            'publication_ready': all_passed
        },
        'checks': checks
    }
    
    report_path = results_dir / 'final_report.json'
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\nFinal report saved to: {report_path}")
    
    if all_passed:
        print("\n" + "="*80)
        print("✓ ALL EXPERIMENTS PASSED - READY FOR PUBLICATION!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠ SOME EXPERIMENTS NEED ATTENTION")
        print("="*80)
    
    return final_report

if __name__ == "__main__":
    report = main()

