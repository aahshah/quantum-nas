"""
Debug GNN Predictor - Check if predictions are actually backwards
"""
import numpy as np
import torch
from scipy.stats import spearmanr
from quantum_nas import *

def debug_gnn_predictions():
    print("="*60)
    print("DEBUGGING GNN PREDICTOR")
    print("="*60)
    
    # 1. Setup
    hardware = HARDWARE_SPECS['ibm_quantum']
    sampler = ArchitectureSampler(hardware)
    graph_builder = ImprovedBipartiteGraphBuilder()
    sim = QuantumCircuitSimulator(hardware)
    
    # 2. Generate test data
    print("\n[1/4] Generating test architectures...")
    architectures = []
    graphs = []
    true_accuracies = []
    
    for i in range(50):
        arch = sampler.sample()
        graph = graph_builder.build(arch, hardware)
        perf = sim.evaluate(arch, np.zeros((10,4)), np.zeros(10), 
                           np.zeros((10,4)), np.zeros(10), quick_mode=True)
        
        architectures.append(arch)
        graphs.append(graph)
        true_accuracies.append(perf['accuracy'])
    
    # 3. Train GNN
    print("\n[2/4] Training GNN...")
    targets = [{
        'accuracy': torch.tensor([acc]),
        'energy': torch.tensor([10.0]),
        'trainability': torch.tensor([0.5]),
        'depth': torch.tensor([5.0])
    } for acc in true_accuracies]
    
    gnn_model = BaselineGNNPredictor()
    trainer = GNNTrainer(gnn_model)
    
    # Train with verbose to see loss
    history = trainer.train(graphs[:40], targets[:40], 
                           graphs[40:], targets[40:], 
                           epochs=20, verbose=True)
    
    print("\n[3/4] Checking predictions...")
    
    # 4. Get predictions on validation set
    gnn_model.eval()
    gnn_predictions = []
    
    with torch.no_grad():
        for graph in graphs[40:]:
            pred = gnn_model(graph)
            gnn_predictions.append(pred['accuracy'].item())
    
    true_val = true_accuracies[40:]
    
    # 5. Manual inspection
    print("\n[4/4] Manual Inspection:")
    print("\nTop 3 TRUE high-accuracy circuits:")
    sorted_indices = np.argsort(true_val)[::-1][:3]
    for idx in sorted_indices:
        print(f"  Circuit {idx}: True={true_val[idx]:.3f}, GNN Predicted={gnn_predictions[idx]:.3f}")
    
    print("\nTop 3 TRUE low-accuracy circuits:")
    sorted_indices = np.argsort(true_val)[:3]
    for idx in sorted_indices:
        print(f"  Circuit {idx}: True={true_val[idx]:.3f}, GNN Predicted={gnn_predictions[idx]:.3f}")
    
    # 6. Correlation
    corr, _ = spearmanr(true_val, gnn_predictions)
    print(f"\nSpearman Correlation: {corr:.3f}")
    
    # 7. Check if flipping helps
    flipped_predictions = [1.0 - p for p in gnn_predictions]
    corr_flipped, _ = spearmanr(true_val, flipped_predictions)
    print(f"Spearman (Flipped): {corr_flipped:.3f}")
    
    # 8. Training loss check
    print("\n--- Training Loss Analysis ---")
    if 'train_loss' in history:
        print(f"Initial Loss: {history['train_loss'][0]:.4f}")
        print(f"Final Loss: {history['train_loss'][-1]:.4f}")
        if history['train_loss'][-1] < history['train_loss'][0]:
            print("✓ Loss decreased (training worked)")
        else:
            print("✗ Loss increased or stayed flat (training failed)")
    
    return {
        'correlation': corr,
        'correlation_flipped': corr_flipped,
        'true_val': true_val,
        'predictions': gnn_predictions,
        'history': history
    }

if __name__ == "__main__":
    results = debug_gnn_predictions()
