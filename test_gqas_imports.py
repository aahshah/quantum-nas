"""
Test script to verify all GQAS imports work correctly.
Run this from the project root directory.
"""

import sys
from pathlib import Path

print("Testing GQAS imports...")
print(f"Current directory: {Path.cwd()}")
print(f"Python path (first 3): {sys.path[:3]}")

# Test 1: quantum_nas imports
print("\n[1/4] Testing quantum_nas imports...")
try:
    from quantum_nas import (
        HARDWARE_SPECS, ArchitectureSampler, QuantumCircuitSimulator,
        HybridArchitecture, NSGA2Search, GraphTransformerPredictor
    )
    print(f"✓ quantum_nas imports successful")
    print(f"  Found {len(HARDWARE_SPECS)} hardware specs")
except ImportError as e:
    print(f"✗ quantum_nas import failed: {e}")
    sys.exit(1)

# Test 2: qfi_calculator imports
print("\n[2/4] Testing qfi_calculator imports...")
try:
    from qfi_calculator import QuantumFisherInformationCalculator, compute_qfi_for_architecture
    print("✓ qfi_calculator imports successful")
except ImportError as e:
    print(f"⚠ qfi_calculator import failed (optional): {e}")
    print("  Will use simplified QFI calculation")

# Test 3: gradient_flow imports
print("\n[3/4] Testing gradient_flow imports...")
try:
    from gradient_flow import GradientFlowAnalyzer
    print("✓ gradient_flow imports successful")
except ImportError as e:
    print(f"⚠ gradient_flow import failed (optional): {e}")
    print("  Will use simplified gradient flow analysis")

# Test 4: Experiment script import
print("\n[4/4] Testing experiment script setup...")
try:
    # Simulate what the experiment script does
    script_file = Path('experiments/validate_qfi_correlation_fixed.py')
    if script_file.exists():
        print(f"✓ Experiment script found: {script_file}")
    else:
        print(f"⚠ Experiment script not found: {script_file}")
except Exception as e:
    print(f"⚠ Error checking experiment script: {e}")

print("\n" + "="*80)
print("IMPORT TEST COMPLETE")
print("="*80)
print("\nTo run experiments:")
print("  python experiments/validate_qfi_correlation_fixed.py")
print("\nOr use the wrapper:")
print("  python run_qfi_experiment.py")


