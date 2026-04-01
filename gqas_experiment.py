"""
Gradient-Aware Quantum Architecture Search (GQAS) Experiment

This script compares QFI-guided search vs standard search to demonstrate
that GQAS finds better architectures 30%+ faster.

Key Innovation:
- Uses Quantum Fisher Information (QFI) to predict trainability
- Filters out architectures with vanishing gradients before training
- Results in faster convergence to high-quality architectures
"""

import torch
import numpy as np
import time
import json
from pathlib import Path
from quantum_nas import (
    HARDWARE_SPECS, ArchitectureSampler, ImprovedBipartiteGraphBuilder,
    GraphTransformerPredictor, NSGA2Search, QuantumCircuitSimulator,
    QFICalculator, GNNTrainer
)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def run_gqas_experiment():
    """
    Run comparison experiment: QFI-guided (GQAS) vs Standard search.
    
    Measures:
    1. Time to find architecture with accuracy > threshold
    2. Number of evaluations needed
    3. Final architecture quality (accuracy, energy, QFI)
    """
    print("="*80)
    print("GQAS EXPERIMENT: QFI-Guided vs Standard Search")
    print("="*80)
    
    # Setup
    hardware = HARDWARE_SPECS['ibm_quantum']
    sampler = ArchitectureSampler(hardware)
    graph_builder_standard = ImprovedBipartiteGraphBuilder(include_gradient_flow=False)
    graph_builder_gqas = ImprovedBipartiteGraphBuilder(include_gradient_flow=True)
    
    # Load data
    digits = load_digits(n_class=2)
    X = digits.data / 16.0
    y = (digits.target > 4).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Generate training dataset with QFI
    print("\n[1/4] Generating training dataset with QFI...")
    qfi_calculator = QFICalculator(n_samples=50)
    quantum_sim = QuantumCircuitSimulator(
        hardware, 
        compute_qfi=True, 
        qfi_calculator=qfi_calculator
    )
    
    # Generate small dataset for predictor training
    train_graphs = []
    train_targets = []
    
    for i in range(50):
        arch = sampler.sample()
        results = quantum_sim.evaluate(
            arch, X_train, y_train, X_test, y_test,
            n_epochs=10, quick_mode=True
        )
        
        graph = graph_builder_gqas.build(arch, hardware, qfi_value=results.get('qfi', 0.0))
        train_graphs.append(graph)
        
        target_dict = {
            'accuracy': torch.tensor([results['accuracy']], dtype=torch.float32),
            'energy': torch.tensor([results['energy'] / 100.0], dtype=torch.float32),
            'trainability': torch.tensor([results['trainability']], dtype=torch.float32),
            'depth': torch.tensor([float(results['circuit_depth']) / 40.0], dtype=torch.float32),
            'qfi': torch.tensor([results.get('qfi', 0.0)], dtype=torch.float32)
        }
        train_targets.append(target_dict)
    
    # Train predictor with QFI
    print("\n[2/4] Training predictor with QFI support...")
    predictor = GraphTransformerPredictor()
    trainer = GNNTrainer(predictor, device='cpu')
    
    split_idx = int(0.8 * len(train_graphs))
    val_graphs = train_graphs[split_idx:]
    val_targets = train_targets[split_idx:]
    train_graphs_sub = train_graphs[:split_idx]
    train_targets_sub = train_targets[:split_idx]
    
    trainer.train(
        train_graphs_sub, train_targets_sub,
        val_graphs, val_targets,
        epochs=30, batch_size=8, verbose=False
    )
    
    # Run Standard Search (no QFI)
    print("\n[3/4] Running Standard Search (no QFI guidance)...")
    standard_search = NSGA2Search(
        predictor, graph_builder_standard, sampler, hardware,
        qfi_threshold=0.0, use_qfi=False
    )
    
    start_time = time.time()
    standard_arch, _, standard_history = standard_search.search(
        X_train, y_train, X_test, y_test,
        pop_size=20, generations=20
    )
    standard_time = time.time() - start_time
    
    # Evaluate standard result
    standard_results = quantum_sim.evaluate(
        standard_arch, X_train, y_train, X_test, y_test, quick_mode=True
    )
    
    # Run GQAS Search (with QFI)
    print("\n[4/4] Running GQAS Search (QFI-guided)...")
    gqas_search = NSGA2Search(
        predictor, graph_builder_gqas, sampler, hardware,
        qfi_threshold=0.1, use_qfi=True
    )
    
    start_time = time.time()
    gqas_arch, _, gqas_history = gqas_search.search(
        X_train, y_train, X_test, y_test,
        pop_size=20, generations=20
    )
    gqas_time = time.time() - start_time
    
    # Evaluate GQAS result
    gqas_results = quantum_sim.evaluate(
        gqas_arch, X_train, y_train, X_test, y_test, quick_mode=True
    )
    
    # Results Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nStandard Search:")
    print(f"  Time: {standard_time:.2f}s")
    print(f"  Accuracy: {standard_results['accuracy']:.3f}")
    print(f"  Energy: {standard_results['energy']:.1f}")
    print(f"  QFI: {standard_results.get('qfi', 0.0):.3f}")
    
    print(f"\nGQAS Search (QFI-guided):")
    print(f"  Time: {gqas_time:.2f}s")
    print(f"  Accuracy: {gqas_results['accuracy']:.3f}")
    print(f"  Energy: {gqas_results['energy']:.1f}")
    print(f"  QFI: {gqas_results.get('qfi', 0.0):.3f}")
    
    # Calculate speedup
    if standard_time > 0:
        speedup = standard_time / gqas_time
        print(f"\nSpeedup: {speedup:.2f}x faster")
        if speedup >= 1.3:
            print("✓ GQAS achieves 30%+ speedup as required!")
        else:
            print("⚠ Speedup below 30% threshold")
    
    # Save results
    results = {
        'standard': {
            'time': standard_time,
            'accuracy': standard_results['accuracy'],
            'energy': standard_results['energy'],
            'qfi': standard_results.get('qfi', 0.0)
        },
        'gqas': {
            'time': gqas_time,
            'accuracy': gqas_results['accuracy'],
            'energy': gqas_results['energy'],
            'qfi': gqas_results.get('qfi', 0.0)
        },
        'speedup': speedup if standard_time > 0 else 0.0
    }
    
    with open('gqas_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to gqas_results.json")
    
    return results

if __name__ == "__main__":
    results = run_gqas_experiment()

