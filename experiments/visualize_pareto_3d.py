"""
3D Pareto Front Visualization: Accuracy vs Depth vs Gradient Variance

Creates publication-quality 3D plots showing the Pareto front in
(accuracy, depth, gradient_variance) space.

This demonstrates the trade-off between accuracy, circuit depth, and trainability.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json

# Import setup
script_file = Path(__file__).resolve()
parent_dir = script_file.parent.parent
sys.path.insert(0, str(parent_dir))

from quantum_nas import HARDWARE_SPECS, ArchitectureSampler, QuantumCircuitSimulator
from barren_plateau_analyzer import BarrenPlateauAnalyzer

def generate_pareto_data(n_samples=100, seed=42):
    """Generate architecture data for Pareto front visualization."""
    print("Generating architecture data...")
    
    np.random.seed(seed)
    hardware = HARDWARE_SPECS['ibm_quantum']
    sampler = ArchitectureSampler(hardware)
    sim = QuantumCircuitSimulator(hardware, analyze_barren_plateau=True)
    bp_analyzer = BarrenPlateauAnalyzer(n_samples=30)  # Fewer samples for speed
    
    # Load data
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    digits = load_digits(n_class=2)
    X = digits.data / 16.0
    y = (digits.target > 4).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_sample = X_train[0:1] if len(X_train) > 0 else np.zeros((1, 8))
    
    results = []
    
    for i in range(n_samples):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        arch = sampler.sample()
        
        # Evaluate architecture
        try:
            eval_result = sim.evaluate(arch, X_train, y_train, X_test, y_test, quick_mode=True)
            
            # Get gradient variance
            if 'gradient_variance' in eval_result:
                grad_var = eval_result['gradient_variance']
            else:
                # Compute if not available
                bp_result = bp_analyzer.analyze_architecture(arch, sim, X_sample)
                grad_var = bp_result.gradient_variance
            
            results.append({
                'accuracy': eval_result['accuracy'],
                'depth': arch.total_depth,
                'gradient_variance': grad_var,
                'energy': eval_result['energy'],
                'n_qubits': arch.total_qubits
            })
        except Exception as e:
            if (i + 1) % 10 == 0:
                print(f"  Warning: Evaluation failed: {e}")
            continue
    
    return pd.DataFrame(results)

def plot_3d_pareto(df, save_path='experiments/pareto_3d.png'):
    """Create 3D Pareto front visualization."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(
        df['depth'],
        df['gradient_variance'],
        df['accuracy'],
        c=df['accuracy'],
        s=df['energy'] * 2,  # Size by energy
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidths=0.5
    )
    
    ax.set_xlabel('Circuit Depth', fontweight='bold', fontsize=12)
    ax.set_ylabel('Gradient Variance (log scale)', fontweight='bold', fontsize=12)
    ax.set_zlabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('3D Pareto Front: Accuracy vs Depth vs Gradient Variance', 
                 fontweight='bold', fontsize=14)
    
    # Set log scale for gradient variance
    ax.set_yscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Accuracy', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D plot to {save_path}")

def plot_2d_projections(df, save_path='experiments/pareto_2d_projections.png'):
    """Create 2D projections of the 3D Pareto front."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Depth vs Accuracy
    axes[0, 0].scatter(df['depth'], df['accuracy'], 
                      c=df['gradient_variance'], cmap='viridis',
                      s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[0, 0].set_xlabel('Circuit Depth', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_title('Depth vs Accuracy\n(color = gradient variance)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Depth vs Gradient Variance
    axes[0, 1].scatter(df['depth'], df['gradient_variance'],
                      c=df['accuracy'], cmap='plasma',
                      s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[0, 1].set_xlabel('Circuit Depth', fontweight='bold')
    axes[0, 1].set_ylabel('Gradient Variance', fontweight='bold')
    axes[0, 1].set_title('Depth vs Gradient Variance\n(color = accuracy)', fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient Variance vs Accuracy
    axes[1, 0].scatter(df['gradient_variance'], df['accuracy'],
                      c=df['depth'], cmap='coolwarm',
                      s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[1, 0].set_xlabel('Gradient Variance', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[1, 0].set_title('Gradient Variance vs Accuracy\n(color = depth)', fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy vs Accuracy (with gradient variance overlay)
    scatter = axes[1, 1].scatter(df['energy'], df['accuracy'],
                                c=df['gradient_variance'], cmap='viridis',
                                s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    axes[1, 1].set_xlabel('Energy', fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy', fontweight='bold')
    axes[1, 1].set_title('Energy vs Accuracy\n(color = gradient variance)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Gradient Variance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D projections to {save_path}")

def main():
    """Generate Pareto front visualizations."""
    print("="*80)
    print("3D PARETO FRONT VISUALIZATION")
    print("="*80)
    
    Path('experiments').mkdir(exist_ok=True)
    
    # Generate data
    df = generate_pareto_data(n_samples=100, seed=42)
    
    print(f"\nGenerated {len(df)} architecture evaluations")
    print(f"  Accuracy range: {df['accuracy'].min():.3f} - {df['accuracy'].max():.3f}")
    print(f"  Depth range: {df['depth'].min()} - {df['depth'].max()}")
    print(f"  Gradient variance range: {df['gradient_variance'].min():.2e} - {df['gradient_variance'].max():.2e}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_3d_pareto(df)
    plot_2d_projections(df)
    
    # Save data
    df.to_csv('experiments/pareto_data.csv', index=False)
    print("\nSaved data to experiments/pareto_data.csv")
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)

if __name__ == "__main__":
    main()

