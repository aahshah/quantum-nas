"""
Clean Predictor Validation with Fixed MultiTaskLoss
Retrain both GNN and Transformer from scratch to get honest comparison
"""
import numpy as np
import torch
from scipy.stats import spearmanr
from quantum_nas import *

def run_clean_validation():
    print("="*60)
    print("CLEAN PREDICTOR VALIDATION (FIXED LOSS)")
    print("="*60)
    
    # 1. Setup
    hardware = HARDWARE_SPECS['ibm_quantum']
    sampler = ArchitectureSampler(hardware)
    graph_builder = ImprovedBipartiteGraphBuilder()
    sim = QuantumCircuitSimulator(hardware)
    
    # 2. Generate dataset
    print("\n[1/4] Generating 200 architectures...")
    architectures = []
    graphs = []
    performances = []
    
    for i in range(200):
        if (i+1) % 50 == 0:
            print(f"  Generated {i+1}/200...")
        
        arch = sampler.sample()
        graph = graph_builder.build(arch, hardware)
        perf = sim.evaluate(arch, np.zeros((10,4)), np.zeros(10), 
                           np.zeros((10,4)), np.zeros(10), quick_mode=True)
        
        architectures.append(arch)
        graphs.append(graph)
        performances.append(perf)
    
    # 3. Prepare targets
    targets = [{
        'accuracy': torch.tensor([p['accuracy']]),
        'energy': torch.tensor([p['energy']]),
        'trainability': torch.tensor([p['trainability']]),
        'depth': torch.tensor([float(p['circuit_depth'])])
    } for p in performances]
    
    # Split train/val
    split = 160
    train_graphs, val_graphs = graphs[:split], graphs[split:]
    train_targets, val_targets = targets[:split], targets[split:]
    
    # 4. Train GNN
    print("\n[2/4] Training GNN Predictor...")
    gnn_model = BaselineGNNPredictor()
    gnn_trainer = GNNTrainer(gnn_model)
    gnn_trainer.train(train_graphs, train_targets, val_graphs, val_targets, 
                     epochs=20, verbose=False)
    
    # 5. Train Transformer
    print("\n[3/4] Training Transformer Predictor...")
    trans_model = GraphTransformerPredictor()
    trans_trainer = GNNTrainer(trans_model)
    trans_trainer.train(train_graphs, train_targets, val_graphs, val_targets,
                       epochs=20, verbose=False)
    
    # 6. Evaluate on validation set
    print("\n[4/4] Evaluating Predictors...")
    
    gnn_model.eval()
    trans_model.eval()
    
    gnn_preds_acc = []
    trans_preds_acc = []
    true_acc = []
    
    gnn_preds_energy = []
    trans_preds_energy = []
    true_energy = []
    
    with torch.no_grad():
        for graph, perf in zip(val_graphs, performances[split:]):
            gnn_pred = gnn_model(graph)
            trans_pred = trans_model(graph)
            
            gnn_preds_acc.append(gnn_pred['accuracy'].item())
            trans_preds_acc.append(trans_pred['accuracy'].item())
            true_acc.append(perf['accuracy'])
            
            gnn_preds_energy.append(gnn_pred['energy'].item())
            trans_preds_energy.append(trans_pred['energy'].item())
            true_energy.append(perf['energy'])
    
    # 7. Compute correlations
    gnn_corr_acc, _ = spearmanr(true_acc, gnn_preds_acc)
    trans_corr_acc, _ = spearmanr(true_acc, trans_preds_acc)
    
    gnn_corr_energy, _ = spearmanr(true_energy, gnn_preds_energy)
    trans_corr_energy, _ = spearmanr(true_energy, trans_preds_energy)
    
    # 8. Results
    print("\n" + "="*60)
    print("CORRECTED PREDICTOR PERFORMANCE")
    print("="*60)
    
    print("\nAccuracy Prediction (Spearman Correlation):")
    print(f"  GNN Baseline:        {gnn_corr_acc:.3f}")
    print(f"  Transformer (Ours):  {trans_corr_acc:.3f}")
    if trans_corr_acc > gnn_corr_acc:
        improvement = ((trans_corr_acc - gnn_corr_acc) / abs(gnn_corr_acc)) * 100
        print(f"  Improvement:         +{improvement:.1f}%")
    
    print("\nEnergy Prediction (Spearman Correlation):")
    print(f"  GNN Baseline:        {gnn_corr_energy:.3f}")
    print(f"  Transformer (Ours):  {trans_corr_energy:.3f}")
    
    # Save results
    with open('corrected_predictor_results.txt', 'w') as f:
        f.write("Predictor,Accuracy_Spearman,Energy_Spearman\n")
        f.write(f"GNN,{gnn_corr_acc:.3f},{gnn_corr_energy:.3f}\n")
        f.write(f"Transformer,{trans_corr_acc:.3f},{trans_corr_energy:.3f}\n")
    
    print("\n✓ Results saved to corrected_predictor_results.txt")
    
    return {
        'gnn_acc': gnn_corr_acc,
        'trans_acc': trans_corr_acc,
        'gnn_energy': gnn_corr_energy,
        'trans_energy': trans_corr_energy
    }

if __name__ == "__main__":
    results = run_clean_validation()
