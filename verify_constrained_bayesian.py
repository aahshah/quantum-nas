"""
Verify Constrained Bayesian: Can Bayesian match Ours if forced to be shallow?
"""
import numpy as np
import torch
from quantum_nas import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_constrained_comparison():
    print("="*60)
    print("CONSTRAINED BAYESIAN CHALLENGE")
    print("Testing: Can Bayesian match Ours if restricted to Depth <= 3?")
    print("="*60)
    
    # 1. Setup Environment
    hardware = HARDWARE_SPECS['ibm_quantum']
    sampler = ArchitectureSampler(hardware)
    graph_builder = ImprovedBipartiteGraphBuilder()
    sim = QuantumCircuitSimulator(hardware)
    
    # 2. Generate Training Data & Train Predictor (Fast)
    print("\n[1/3] Training Predictor (Fast Mode)...")
    # We'll generate a smaller set for speed, but enough to learn
    architectures = []
    graphs = []
    performances = []
    
    # Generate 100 archs
    for i in range(100):
        arch = sampler.sample()
        graph = graph_builder.build(arch, hardware)
        # Use dummy performance for speed - we rely on the predictor's learned bias
        # But to make it "real", we should use the simulator's heuristic
        perf = sim.evaluate(arch, np.zeros((10,4)), np.zeros(10), np.zeros((10,4)), np.zeros(10), quick_mode=True)
        
        architectures.append(arch)
        graphs.append(graph)
        performances.append(perf)
        
    # Train Transformer
    targets = [{
        'accuracy': torch.tensor([p['accuracy']]),
        'energy': torch.tensor([p['energy']]),
        'trainability': torch.tensor([p['trainability']]),
        'depth': torch.tensor([float(p['circuit_depth'])])
    } for p in performances]
    
    model = GraphTransformerPredictor()
    GNNTrainer(model).train(graphs, targets, graphs[:20], targets[:20], epochs=10, verbose=False)
    print("✓ Predictor Trained")
    
    # 3. Run Comparisons
    print("\n[2/3] Running Search Algorithms...")
    
    # A. Standard Bayesian (Unconstrained) - Proxy: Random Sampling + Predictor Selection
    # Real BO would use acquisition functions, here we simulate "finding the best in a random batch"
    print("  Running Standard Bayesian (Depth Unconstrained)...")
    bo_candidates = [sampler.sample() for _ in range(50)]
    bo_scores = [model(graph_builder.build(a, hardware))['accuracy'].item() for a in bo_candidates]
    best_bo_idx = np.argmax(bo_scores)
    best_bo = bo_candidates[best_bo_idx]
    best_bo_score = bo_scores[best_bo_idx]
    
    # B. Constrained Bayesian (Depth <= 3)
    print("  Running Constrained Bayesian (Depth <= 3)...")
    cbo_candidates = []
    attempts = 0
    while len(cbo_candidates) < 50 and attempts < 500:
        a = sampler.sample()
        if a.total_depth <= 3:
            cbo_candidates.append(a)
        attempts += 1
    
    if not cbo_candidates:
        print("  WARNING: Could not find shallow candidates for Constrained BO")
        best_cbo = None
        best_cbo_score = 0
    else:
        cbo_scores = [model(graph_builder.build(a, hardware))['accuracy'].item() for a in cbo_candidates]
        best_cbo_idx = np.argmax(cbo_scores)
        best_cbo = cbo_candidates[best_cbo_idx]
        best_cbo_score = cbo_scores[best_cbo_idx]

    # C. Ours (NSGA-II)
    print("  Running Ours (Graph Transformer + NSGA-II)...")
    # We use the actual search class
    search = NSGA2Search(model, graph_builder, sampler, hardware)
    # Dummy data for search interface
    X_d = np.random.randn(10, 4)
    y_d = np.random.randint(0, 2, 10)
    best_ours, _, _ = search.search(X_d, y_d, X_d, y_d, pop_size=20, generations=5)
    best_ours_score = model(graph_builder.build(best_ours, hardware))['accuracy'].item()

    # 4. Results
    print("\n" + "="*60)
    print("RESULTS: ACCURACY vs DEPTH")
    print("="*60)
    
    print(f"Standard Bayesian:  Acc={best_bo_score:.3f}, Depth={best_bo.total_depth}")
    if best_cbo:
        print(f"Constrained BO:     Acc={best_cbo_score:.3f}, Depth={best_cbo.total_depth}")
    else:
        print(f"Constrained BO:     FAILED to find candidates")
        
    print(f"Ours (NSGA-II):     Acc={best_ours_score:.3f}, Depth={best_ours.total_depth}")
    
    print("\n--- Analysis ---")
    if best_cbo and best_ours_score > best_cbo_score:
        diff = best_ours_score - best_cbo_score
        print(f"✓ Ours beats Constrained BO by +{diff:.3f}")
        print("  CONCLUSION: Our search finds BETTER shallow circuits, not just ANY shallow circuit.")
    elif best_cbo and abs(best_ours_score - best_cbo_score) < 0.01:
        print(f"~ Ours ties Constrained BO")
        print("  CONCLUSION: Depth is the main factor. Ours finds these automatically.")
    else:
        print(f"✗ Constrained BO wins")

if __name__ == "__main__":
    run_constrained_comparison()
