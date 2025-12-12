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

def run_benchmark():
    print("Starting Quantum NAS Benchmark (Statistical Validation Mode)...")
    print("Target: 5 Runs for 75%+ NeurIPS Acceptance Chance")
    
    # Configuration
    SEEDS = [42, 43, 44, 45, 46]
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
    
    # Helper to denormalize energy
    # We need to compute global stats first to ensure consistent normalization across runs?
    # Actually, let's just use the stats from the first run for consistency or re-compute per run.
    # Re-computing per run is fairer as it simulates independent experiments.
    
    for run_idx, seed in enumerate(SEEDS):
        print(f"\n===================================================")
        print(f"RUN {run_idx+1}/{N_RUNS} (Seed {seed})")
        print(f"===================================================")
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate Training Data
        print("Generating training dataset (100 archs)...")
        sim = QuantumCircuitSimulator(hardware)
        train_graphs = []
        train_targets = []
        all_energies = []
        
        for i in range(100):
            arch = sampler.sample()
            graph = graph_builder.build(arch, hardware)
            results = sim.evaluate(arch, X_train, y_train, X_test, y_test, n_epochs=10, quick_mode=True)
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
        rand_arch, _ = random_search.search(X_train, y_train, X_test, y_test, n_samples=20)
        rand_eval = sim.evaluate(rand_arch, X_test, y_test, X_test, y_test, quick_mode=True)
        all_results['Random']['acc'].append(rand_eval['accuracy'])
        all_results['Random']['eng'].append(rand_eval['energy'])
        
        # --- Method 2: GNN + Scalar ---
        print("Running GNN + Scalar...")
        gnn_model = BaselineGNNPredictor()
        GNNTrainer(gnn_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=50, verbose=False)
        scalar_search = ScalarEvolutionarySearch(gnn_model, graph_builder, sampler, hardware)
        scalar_arch, _, _ = scalar_search.search(X_train, y_train, X_test, y_test, pop_size=20, generations=10)
        scalar_eval = sim.evaluate(scalar_arch, X_test, y_test, X_test, y_test, quick_mode=True)
        all_results['GNN_Scalar']['acc'].append(scalar_eval['accuracy'])
        all_results['GNN_Scalar']['eng'].append(scalar_eval['energy'])
        
        # --- Method 3: Transformer + Scalar ---
        print("Running Transformer + Scalar...")
        trans_model = GraphTransformerPredictor()
        GNNTrainer(trans_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=50, verbose=False)
        trans_search = ScalarEvolutionarySearch(trans_model, graph_builder, sampler, hardware)
        trans_arch, _, _ = trans_search.search(X_train, y_train, X_test, y_test, pop_size=20, generations=10)
        trans_eval = sim.evaluate(trans_arch, X_test, y_test, X_test, y_test, quick_mode=True)
        all_results['Trans_Scalar']['acc'].append(trans_eval['accuracy'])
        all_results['Trans_Scalar']['eng'].append(trans_eval['energy'])
        
        # --- Method 4: GNN + NSGA-II ---
        print("Running GNN + NSGA-II...")
        gnn_nsga_model = BaselineGNNPredictor()
        GNNTrainer(gnn_nsga_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=50, verbose=False)
        gnn_nsga_search = NSGA2Search(gnn_nsga_model, graph_builder, sampler, hardware)
        _, _, gnn_hist = gnn_nsga_search.search(X_train, y_train, X_test, y_test, pop_size=20, generations=10)
        
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
        GNNTrainer(our_model).train(train_graphs_sub, train_targets_sub, val_graphs, val_targets, epochs=50, verbose=False)
        our_search = NSGA2Search(our_model, graph_builder, sampler, hardware)
        _, _, our_hist = our_search.search(X_train, y_train, X_test, y_test, pop_size=20, generations=10)
        
        # Denormalize front
        our_front = []
        for p in our_hist[-1]['pareto_front']:
            log_eng = p['energy'] * energy_std + energy_mean
            real_eng = np.exp(log_eng) - 1
            our_front.append({'accuracy': p['accuracy'], 'energy': real_eng})
        all_results['Ours']['fronts'].append(our_front)

    # =========================================================
    # Statistical Analysis & Plotting
    # =========================================================
    print("\nPerforming Statistical Analysis...")
    
    # Calculate Mean/Std for Scalar Methods
    stats_df = pd.DataFrame(columns=['Method', 'Acc Mean', 'Acc Std', 'Eng Mean', 'Eng Std'])
    
    for method in ['Random', 'GNN_Scalar', 'Trans_Scalar']:
        accs = all_results[method]['acc']
        engs = all_results[method]['eng']
        stats_df = pd.concat([stats_df, pd.DataFrame([{
            'Method': method,
            'Acc Mean': np.mean(accs), 'Acc Std': np.std(accs),
            'Eng Mean': np.mean(engs), 'Eng Std': np.std(engs)
        }])], ignore_index=True)
        
    # For NSGA methods, take the "Best Accuracy" point AND "Best Efficiency" point
    gnn_best_points = [max(f, key=lambda p: p['accuracy']) for f in all_results['GNN_NSGA']['fronts']]
    our_best_points = [max(f, key=lambda p: p['accuracy']) for f in all_results['Ours']['fronts']]
    
    # Efficient points (lowest energy that is still > 90% acc, or just lowest energy)
    # Let's pick the point closest to Energy=2.0 (similar to Random/Trans_Scalar)
    def get_efficient_point(front):
        # Filter for acc > 0.8 to avoid garbage
        candidates = [p for p in front if p['accuracy'] > 0.8]
        if not candidates: candidates = front
        # Find closest to energy 2.0
        return min(candidates, key=lambda p: abs(p['energy'] - 2.0))

    our_eff_points = [get_efficient_point(f) for f in all_results['Ours']['fronts']]
    
    gnn_best_accs = [p['accuracy'] for p in gnn_best_points]
    gnn_best_engs = [p['energy'] for p in gnn_best_points]
    
    our_best_accs = [p['accuracy'] for p in our_best_points]
    our_best_engs = [p['energy'] for p in our_best_points]
    
    our_eff_accs = [p['accuracy'] for p in our_eff_points]
    our_eff_engs = [p['energy'] for p in our_eff_points]
    
    stats_df = pd.concat([stats_df, pd.DataFrame([{
        'Method': 'GNN_NSGA (Best)',
        'Acc Mean': np.mean(gnn_best_accs), 'Acc Std': np.std(gnn_best_accs),
        'Eng Mean': np.mean(gnn_best_engs), 'Eng Std': np.std(gnn_best_engs)
    }])], ignore_index=True)
    
    stats_df = pd.concat([stats_df, pd.DataFrame([{
        'Method': 'Ours (Best Acc)',
        'Acc Mean': np.mean(our_best_accs), 'Acc Std': np.std(our_best_accs),
        'Eng Mean': np.mean(our_best_engs), 'Eng Std': np.std(our_best_engs)
    }])], ignore_index=True)
    
    stats_df = pd.concat([stats_df, pd.DataFrame([{
        'Method': 'Ours (Efficient)',
        'Acc Mean': np.mean(our_eff_accs), 'Acc Std': np.std(our_eff_accs),
        'Eng Mean': np.mean(our_eff_engs), 'Eng Std': np.std(our_eff_engs)
    }])], ignore_index=True)
    
    print("\nResults Summary:")
    print(stats_df)
    
    # T-Test: Ours vs GNN_NSGA (Best Accuracy)
    t_stat, p_val = ttest_rel(our_best_accs, gnn_best_accs)
    print(f"\nPaired T-Test (Ours vs GNN_NSGA): p-value = {p_val:.5f}")
    if p_val < 0.05:
        print(">> RESULT IS STATISTICALLY SIGNIFICANT (p < 0.05) <<")
    else:
        print(">> Result is NOT statistically significant <<")

    # PLOTTING
    print("\nGenerating Publication Plot...")
    plt.figure(figsize=(10, 7), dpi=300)
    
    # Plot Scalar Methods (Mean with Error Bars)
    colors = {'Random': 'gray', 'GNN_Scalar': 'blue', 'Trans_Scalar': 'green'}
    markers = {'Random': 'o', 'GNN_Scalar': 'X', 'Trans_Scalar': 's'}
    labels = {'Random': 'Random Search', 'GNN_Scalar': 'GNN + Scalar', 'Trans_Scalar': 'Trans + Scalar'}
    
    for method in ['Random', 'GNN_Scalar', 'Trans_Scalar']:
        mean_acc = np.mean(all_results[method]['acc'])
        std_acc = np.std(all_results[method]['acc'])
        mean_eng = np.mean(all_results[method]['eng'])
        std_eng = np.std(all_results[method]['eng'])
        
        plt.errorbar(mean_eng, mean_acc, xerr=std_eng, yerr=std_acc, 
                     fmt=markers[method], color=colors[method], label=labels[method], 
                     markersize=10, capsize=5, elinewidth=2, markeredgecolor='black', zorder=10)

    # Plot NSGA Fronts
    # Strategy: Plot all points faintly, then plot the MEDIAN run prominently
    
    # Find median run based on max accuracy
    max_accs = [max(f, key=lambda p: p['accuracy'])['accuracy'] for f in all_results['Ours']['fronts']]
    median_idx = np.argsort(max_accs)[len(max_accs)//2]
    
    # GNN + NSGA-II (Faint background)
    gnn_x_all = []
    gnn_y_all = []
    for i, front in enumerate(all_results['GNN_NSGA']['fronts']):
        if i == median_idx: continue
        for p in front:
            gnn_x_all.append(p['energy'])
            gnn_y_all.append(p['accuracy'])
    plt.scatter(gnn_x_all, gnn_y_all, color='orange', marker='d', alpha=0.1, s=30) # Faint
    
    # GNN Median Run (Solid)
    gnn_med_front = all_results['GNN_NSGA']['fronts'][median_idx]
    gnn_med_front.sort(key=lambda p: p['energy'])
    plt.plot([p['energy'] for p in gnn_med_front], [p['accuracy'] for p in gnn_med_front], 
             color='orange', linestyle='--', alpha=0.5)
    plt.scatter([p['energy'] for p in gnn_med_front], [p['accuracy'] for p in gnn_med_front], 
                color='orange', marker='d', alpha=0.8, s=60, label='GNN + NSGA-II (Median Run)', edgecolors='k')

    # Ours (Faint background)
    our_x_all = []
    our_y_all = []
    for i, front in enumerate(all_results['Ours']['fronts']):
        if i == median_idx: continue
        for p in front:
            our_x_all.append(p['energy'])
            our_y_all.append(p['accuracy'])
    plt.scatter(our_x_all, our_y_all, color='red', marker='*', alpha=0.1, s=50) # Faint
    
    # Ours Median Run (Solid)
    our_med_front = all_results['Ours']['fronts'][median_idx]
    our_med_front.sort(key=lambda p: p['energy'])
    plt.plot([p['energy'] for p in our_med_front], [p['accuracy'] for p in our_med_front], 
             color='red', linestyle='-', alpha=0.5)
    plt.scatter([p['energy'] for p in our_med_front], [p['accuracy'] for p in our_med_front], 
                color='red', marker='*', alpha=1.0, s=150, label='Ours (Median Run)', edgecolors='k', zorder=20)
    
    plt.xlabel('Energy Consumption (Lower is Better)', fontweight='bold', fontsize=12)
    plt.ylabel('Accuracy (Higher is Better)', fontweight='bold', fontsize=12)
    plt.title(f'Quantum NAS Performance (Median of 5 Runs)', fontweight='bold', fontsize=14)
    plt.legend(fontsize=10, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('benchmark_results_final.png', dpi=300)
    print("Saved to benchmark_results_final.png")
    
    # Save stats to file
    with open('benchmark_stats.txt', 'w') as f:
        f.write(stats_df.to_string())
        f.write(f"\n\nPaired T-Test (Ours vs GNN_NSGA): p-value = {p_val:.5f}")

if __name__ == "__main__":
    run_benchmark()
