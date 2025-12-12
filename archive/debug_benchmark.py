import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy import stats
from quantum_nas import (
    HardwareSpec, HARDWARE_SPECS, ArchitectureSampler, ImprovedBipartiteGraphBuilder,
    GraphTransformerPredictor, BaselineGNNPredictor,
    NSGA2Search, ScalarEvolutionarySearch, RandomSearch,
    QuantumCircuitSimulator, GNNTrainer
)
from scipy.stats import spearmanr, ttest_rel

def run_debug_benchmark():
    print("Starting DEBUG Quantum NAS Benchmark...")
    print("Target: 1 Run (Fast) to verify Energy Efficiency Shift")
    
    # Configuration - REDUCED for Speed
    SEEDS = [42]
    N_RUNS = len(SEEDS)
    
    # Store results across runs
    all_results = {
        'Random': {'acc': [], 'eng': []},
        'GNN_Scalar': {'acc': [], 'eng': []},
        'Trans_Scalar': {'acc': [], 'eng': []},
        'GNN_NSGA': {'fronts': []},
        'Ours': {'fronts': []}
    }
    
    # 1. Setup Environment (Shared)
    hardware = HARDWARE_SPECS['ibm_quantum']
    sampler = ArchitectureSampler(hardware)
    graph_builder = ImprovedBipartiteGraphBuilder()
    
    # Load Dataset
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X = digits.data / 16.0
    y = (digits.target > 4).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    for run_idx, seed in enumerate(SEEDS):
        print(f"\n===================================================")
        print(f"DEBUG RUN {run_idx+1}/{N_RUNS} (Seed {seed})")
        print(f"===================================================")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate Training Data - REDUCED
        print("Generating training dataset (20 archs)...")
        sim = QuantumCircuitSimulator(hardware)
        train_graphs = []
        train_targets = []
        all_energies = []
        
        for i in range(20): # Reduced from 100
            arch = sampler.sample()
            graph = graph_builder.build(arch, hardware)
            results = sim.evaluate(arch, X_train, y_train, X_test, y_test, n_epochs=1, quick_mode=True)
            all_energies.append(results['energy'])
            train_graphs.append(graph)
            train_targets.append(results)
        
        # Normalization
        energy_mean = np.mean(np.log(np.array(all_energies) + 1))
        energy_std = np.std(np.log(np.array(all_energies) + 1))
        
        train_targets_tensors = []
        for results in train_targets:
            log_energy = np.log(results['energy'] + 1)
            norm_energy = (log_energy - energy_mean) / (energy_std + 1e-8)
            targets = {
                'accuracy': torch.tensor([results['accuracy']], dtype=torch.float32),
                'energy': torch.tensor([norm_energy], dtype=torch.float32),
                'trainability': torch.tensor([results['trainability']], dtype=torch.float32),
                'depth': torch.tensor([float(results['circuit_depth'])], dtype=torch.float32)
            }
            train_targets_tensors.append(targets)
            
        # Split
        split_idx = int(0.8 * len(train_graphs))
        val_graphs = train_graphs[split_idx:]
        val_targets = train_targets_tensors[split_idx:]
        train_graphs_sub = train_graphs[:split_idx]
        train_targets_sub = train_targets_tensors[:split_idx]
        
        # --- Method 1: Random Search ---
        print("Running Random Search...")
        random_search = RandomSearch(sampler, sim, hardware)
        rand_arch, _ = random_search.search(X_train, y_train, X_test, y_test, n_samples=5) # Reduced
        rand_eval = sim.evaluate(rand_arch, X_test, y_test, X_test, y_test, quick_mode=True)
        all_results['Random']['acc'].append(rand_eval['accuracy'])
        all_results['Random']['eng'].append(rand_eval['energy'])
        
        # --- Method 2: GNN + Scalar ---
        print("Running GNN + Scalar...")
        gnn_model = BaselineGNNPredictor()
        GNNTrainer(gnn_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=10, verbose=False) # Reduced epochs
        scalar_search = ScalarEvolutionarySearch(gnn_model, graph_builder, sampler, hardware)
        scalar_arch, _, _ = scalar_search.search(X_train, y_train, X_test, y_test, pop_size=10, generations=5) # Reduced
        scalar_eval = sim.evaluate(scalar_arch, X_test, y_test, X_test, y_test, quick_mode=True)
        all_results['GNN_Scalar']['acc'].append(scalar_eval['accuracy'])
        all_results['GNN_Scalar']['eng'].append(scalar_eval['energy'])
        
        # --- Method 3: Transformer + Scalar ---
        print("Running Transformer + Scalar...")
        trans_model = GraphTransformerPredictor()
        GNNTrainer(trans_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=10, verbose=False)
        trans_search = ScalarEvolutionarySearch(trans_model, graph_builder, sampler, hardware)
        trans_arch, _, _ = trans_search.search(X_train, y_train, X_test, y_test, pop_size=10, generations=5)
        trans_eval = sim.evaluate(trans_arch, X_test, y_test, X_test, y_test, quick_mode=True)
        all_results['Trans_Scalar']['acc'].append(trans_eval['accuracy'])
        all_results['Trans_Scalar']['eng'].append(trans_eval['energy'])
        
        # --- Method 4: GNN + NSGA-II ---
        print("Running GNN + NSGA-II...")
        gnn_nsga_model = BaselineGNNPredictor()
        GNNTrainer(gnn_nsga_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=10, verbose=False)
        gnn_nsga_search = NSGA2Search(gnn_nsga_model, graph_builder, sampler, hardware)
        _, _, gnn_hist = gnn_nsga_search.search(X_train, y_train, X_test, y_test, pop_size=10, generations=5)
        
        # Denormalize front
        gnn_front = []
        for p in gnn_hist[-1]['pareto_front']:
            log_eng = p['energy'] * energy_std + energy_mean
            real_eng = np.exp(log_eng) - 1
            gnn_front.append({'accuracy': p['accuracy'], 'energy': real_eng})
        all_results['GNN_NSGA']['fronts'].append(gnn_front)
        
        # --- Method 5: Ours (Transformer + NSGA-II) ---
        print("Running Ours (Transformer + NSGA-II)...")
        our_model = GraphTransformerPredictor()
        GNNTrainer(our_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=10, verbose=False)
        our_search = NSGA2Search(our_model, graph_builder, sampler, hardware)
        _, _, our_hist = our_search.search(X_train, y_train, X_test, y_test, pop_size=10, generations=5)
        
        # Denormalize front
        our_front = []
        for p in our_hist[-1]['pareto_front']:
            log_eng = p['energy'] * energy_std + energy_mean
            real_eng = np.exp(log_eng) - 1
            our_front.append({'accuracy': p['accuracy'], 'energy': real_eng})
        all_results['Ours']['fronts'].append(our_front)

    # PLOTTING
    print("\nGenerating DEBUG Plot...")
    plt.figure(figsize=(10, 7), dpi=100)
    
    # Plot Scalar Methods
    colors = {'Random': 'gray', 'GNN_Scalar': 'blue', 'Trans_Scalar': 'green'}
    markers = {'Random': 'o', 'GNN_Scalar': 'X', 'Trans_Scalar': 's'}
    
    for method in ['Random', 'GNN_Scalar', 'Trans_Scalar']:
        mean_acc = np.mean(all_results[method]['acc'])
        mean_eng = np.mean(all_results[method]['eng'])
        plt.scatter(mean_eng, mean_acc, color=colors[method], marker=markers[method], s=100, label=method)

    # Plot NSGA Fronts
    gnn_x = []
    gnn_y = []
    for front in all_results['GNN_NSGA']['fronts']:
        for p in front:
            gnn_x.append(p['energy'])
            gnn_y.append(p['accuracy'])
    plt.scatter(gnn_x, gnn_y, color='orange', marker='d', alpha=0.5, s=50, label='GNN + NSGA-II')
    
    our_x = []
    our_y = []
    for front in all_results['Ours']['fronts']:
        for p in front:
            our_x.append(p['energy'])
            our_y.append(p['accuracy'])
    plt.scatter(our_x, our_y, color='red', marker='*', alpha=0.8, s=150, label='Ours')
    
    plt.xlabel('Energy')
    plt.ylabel('Accuracy')
    plt.title('DEBUG: Energy Efficiency Check')
    plt.legend()
    plt.grid(True)
    plt.savefig('benchmark_debug.png')
    print("Saved to benchmark_debug.png")

if __name__ == "__main__":
    run_debug_benchmark()
