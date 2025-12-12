"""
Quick Predictor Validation - Reduced Epochs for Faster Results
"""
import numpy as np
import torch
from scipy.stats import spearmanr
import pickle
import os
import json
from quantum_nas import *

def main():
    print("="*60)
    print("QUICK PREDICTOR VALIDATION (20 epochs)")
    print("="*60)
    
    # Load cached dataset
    print("\n[1/3] Loading cached dataset...")
    with open('dataset_1000.pkl', 'rb') as f:
        archs, graphs, perfs = pickle.load(f)
    print(f"✓ Loaded {len(archs)} architectures")
    
    # Split train/val
    split = int(0.8 * len(archs))
    train_graphs, val_graphs = graphs[:split], graphs[split:]
    train_perfs, val_perfs = perfs[:split], perfs[split:]
    
    # Prepare targets
    train_targets = [{
        'accuracy': torch.tensor([p['accuracy']]),
        'energy': torch.tensor([p['energy']]),
        'trainability': torch.tensor([0.5]),
        'depth': torch.tensor([5.0])
    } for p in train_perfs]
    
    val_targets = [{
        'accuracy': torch.tensor([p['accuracy']]),
        'energy': torch.tensor([p['energy']]),
        'trainability': torch.tensor([0.5]),
        'depth': torch.tensor([5.0])
    } for p in val_perfs]
    
    # Train GNN
    print("\n[2/3] Training predictors (20 epochs each)...")
    print("  Training GNN predictor...")
    gnn_model = BaselineGNNPredictor()
    GNNTrainer(gnn_model).train(train_graphs, train_targets, 
                                val_graphs, val_targets, epochs=20, batch_size=64, verbose=False)
    
    # Train Transformer
    print("  Training Transformer predictor...")
    trans_model = GraphTransformerPredictor()
    GNNTrainer(trans_model).train(train_graphs, train_targets,
                                   val_graphs, val_targets, epochs=20, batch_size=64, verbose=False)
    
    print("✓ Training complete")
    
    # Validate
    print("\n[3/3] Validating Predictors...")
    
    gnn_preds_acc = []
    trans_preds_acc = []
    true_acc = []
    
    gnn_preds_energy = []
    trans_preds_energy = []
    true_energy = []
    
    for graph, perf in zip(val_graphs, val_perfs):
        gnn_pred = gnn_model(graph)
        trans_pred = trans_model(graph)
        
        gnn_preds_acc.append(gnn_pred['accuracy'].item())
        trans_preds_acc.append(trans_pred['accuracy'].item())
        true_acc.append(perf['accuracy'])
        
        gnn_preds_energy.append(gnn_pred['energy'].item())
        trans_preds_energy.append(trans_pred['energy'].item())
        true_energy.append(perf['energy'])
    
    # Compute correlations
    gnn_corr_acc = spearmanr(true_acc, gnn_preds_acc)[0]
    trans_corr_acc = spearmanr(true_acc, trans_preds_acc)[0]
    
    gnn_corr_energy = spearmanr(true_energy, gnn_preds_energy)[0]
    trans_corr_energy = spearmanr(true_energy, trans_preds_energy)[0]
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nAccuracy Prediction:")
    print(f"  GNN Spearman Correlation: {gnn_corr_acc:.3f}")
    print(f"  Transformer Spearman Correlation: {trans_corr_acc:.3f}")
    
    print(f"\nEnergy Prediction:")
    print(f"  GNN Spearman Correlation: {gnn_corr_energy:.3f}")
    print(f"  Transformer Spearman Correlation: {trans_corr_energy:.3f}")
    
    # Save results
    results = {
        'gnn_acc_corr': gnn_corr_acc,
        'trans_acc_corr': trans_corr_acc,
        'gnn_energy_corr': gnn_corr_energy,
        'trans_energy_corr': trans_corr_energy,
        'epochs': 20,
        'n_samples': 1000
    }
    
    with open('quick_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to quick_validation_results.json")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
