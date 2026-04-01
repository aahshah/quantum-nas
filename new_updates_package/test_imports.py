"""
Quick test to verify all imports work correctly.
Run this from the project root directory.
"""

import sys
from pathlib import Path

print("Testing imports...")
print(f"Python path: {sys.path[0]}")
print(f"Current directory: {Path.cwd()}")

try:
    from quantum_nas import (
        HARDWARE_SPECS, ArchitectureSampler, QuantumCircuitSimulator,
        HybridArchitecture, NSGA2Search
    )
    print("✓ quantum_nas imports successful")
    print(f"  Found {len(HARDWARE_SPECS)} hardware specs")
except ImportError as e:
    print(f"✗ quantum_nas import failed: {e}")
    sys.exit(1)

try:
    from qfi_calculator import QuantumFisherInformationCalculator
    print("✓ qfi_calculator import successful")
except ImportError as e:
    print(f"⚠ qfi_calculator import failed (optional): {e}")

try:
    from gradient_flow import GradientFlowAnalyzer
    print("✓ gradient_flow import successful")
except ImportError as e:
    print(f"⚠ gradient_flow import failed (optional): {e}")

print("\nAll critical imports successful!")


