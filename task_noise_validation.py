"""
Task-Based Noise Resilience Experiment (MNIST Classification)
Demonstrates that Depth 2 (Ours) and Depth 6 (Bayesian) have similar ideal accuracy,
but Depth 2 is far more robust to noise.
"""
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt

# --- 1. Data Setup (Harder Task: 3 vs 8) ---
def load_data():
    digits = load_digits()
    # Select 3 and 8 (Hard pair)
    mask = (digits.target == 3) | (digits.target == 8)
    X = digits.data[mask]
    y = digits.target[mask]
    
    # Reduce to 4 features (PCA-like or just center crop) for 4 qubits
    # For simplicity, we'll take top 4 features by variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Simple dimensionality reduction: take 4 columns with highest variance
    variances = np.var(X, axis=0)
    top_4_idx = np.argsort(variances)[-4:]
    X = X[:, top_4_idx]
    
    # Normalize to [0, pi] for angle embedding
    X = (X - X.min()) / (X.max() - X.min()) * np.pi
    
    # Change labels to -1, 1 for PauliZ measurement
    # 3 -> -1, 8 -> 1
    y = np.where(y == 3, -1, 1)
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. Circuit Definitions ---
def apply_noise(noise_prob, wires):
    if noise_prob > 0:
        for w in wires:
            qml.DepolarizingChannel(noise_prob, wires=w)

def get_circuit(depth, noise_prob=0.0):
    dev = qml.device("default.mixed", wires=4)
    
    @qml.qnode(dev)
    def circuit(inputs, params):
        # Encoding
        for i in range(4):
            qml.RY(inputs[i], wires=i)
        
        # Layers
        for l in range(depth):
            apply_noise(noise_prob, range(4))
            
            # Rotation
            for i in range(4):
                qml.RX(params[l, i], wires=i)
                qml.RZ(params[l, i+4], wires=i) # Add RZ for expressivity
            
            # Entanglement
            if depth == 6: # Sparse (Linear)
                for i in range(3):
                    qml.CNOT(wires=[i, i+1])
                    if noise_prob > 0:
                        qml.DepolarizingChannel(noise_prob, wires=i)
                        qml.DepolarizingChannel(noise_prob, wires=i+1)
            else: # Dense (Full) - Ours
                for i in range(4):
                    for j in range(i+1, 4):
                        qml.CNOT(wires=[i, j])
                        if noise_prob > 0:
                            qml.DepolarizingChannel(noise_prob, wires=i)
                            qml.DepolarizingChannel(noise_prob, wires=j)
                            
        return qml.expval(qml.PauliZ(0))
    
    return circuit

# --- 3. Training Loop ---
def train_model(depth, X_train, y_train, steps=50):
    print(f"\nTraining Depth {depth} model...")
    
    # Initialize params: layers x (4 RX + 4 RZ)
    params = pnp.random.uniform(0, 2*np.pi, (depth, 8), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    circuit = get_circuit(depth, noise_prob=0.0) # Train on ideal simulator
    
    def cost(p):
        preds = [circuit(x, p) for x in X_train]
        # MSE Loss
        return pnp.mean((pnp.stack(preds) - y_train) ** 2)
    
    for i in range(steps):
        params, loss = opt.step_and_cost(cost, params)
        if (i+1) % 10 == 0:
            # Calculate accuracy
            preds = [np.sign(circuit(x, params)) for x in X_train]
            acc = np.mean(preds == y_train)
            print(f"  Step {i+1}: Loss = {loss:.4f}, Train Acc = {acc:.1%}")
            
    return params

# --- 4. Main Experiment ---
def run_experiment(seed=42):
    print("="*60)
    print(f"TASK-BASED NOISE RESILIENCE (MNIST 3 vs 8) - Seed {seed}")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_data()
    print(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train both models on IDEAL simulator
    # Use fixed seed for reproducibility
    np.random.seed(seed)
    pnp.random.seed(seed)
    
    # Increased samples to 250 for harder task
    print("\n[Training Phase - Ideal Simulator]")
    params6 = train_model(6, X_train[:250], y_train[:250], steps=50) 
    params2 = train_model(2, X_train[:250], y_train[:250], steps=50)
    
    # Evaluate IDEAL performance first
    print("\n[Ideal Performance Check]")
    circ6_ideal = get_circuit(6, 0.0)
    circ2_ideal = get_circuit(2, 0.0)
    
    preds6_ideal = [np.sign(circ6_ideal(x, params6)) for x in X_test]
    preds2_ideal = [np.sign(circ2_ideal(x, params2)) for x in X_test]
    
    acc6_ideal = np.mean(preds6_ideal == y_test)
    acc2_ideal = np.mean(preds2_ideal == y_test)
    
    print(f"  Depth 6 Test Accuracy (0% noise): {acc6_ideal:.1%}")
    print(f"  Depth 2 Test Accuracy (0% noise): {acc2_ideal:.1%}")
    
    # Evaluate on Test Set with Increasing Noise
    noise_levels = [0.0, 0.02, 0.05, 0.10]
    acc_results = {'depth6': [], 'depth2': []}
    
    print("\n[Noisy Performance Evaluation]")
    for noise in noise_levels:
        # Depth 6
        circ6 = get_circuit(6, noise)
        preds6 = [np.sign(circ6(x, params6)) for x in X_test]
        acc6 = np.mean(preds6 == y_test)
        acc_results['depth6'].append(acc6)
        
        # Depth 2
        circ2 = get_circuit(2, noise)
        preds2 = [np.sign(circ2(x, params2)) for x in X_test]
        acc2 = np.mean(preds2 == y_test)
        acc_results['depth2'].append(acc2)
        
        # Calculate degradation
        deg6 = (acc6_ideal - acc6) / acc6_ideal * 100 if acc6_ideal > 0 else 0
        deg2 = (acc2_ideal - acc2) / acc2_ideal * 100 if acc2_ideal > 0 else 0
        
        print(f"  Noise {noise*100:>4.0f}%: D6={acc6:.1%} (↓{deg6:.0f}%), D2={acc2:.1%} (↓{deg2:.0f}%)")
        
    # Save Results
    output_file = f'task_noise_results_seed{seed}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'seed': seed,
            'noise_levels': noise_levels, 
            'results': acc_results,
            'ideal_acc': {'depth6': acc6_ideal, 'depth2': acc2_ideal}
        }, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")
        
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, acc_results['depth6'], 'o--', label='Bayesian (Depth 6)', color='#e74c3c', linewidth=2, markersize=8)
    plt.plot(noise_levels, acc_results['depth2'], 's-', label='Ours (Depth 2)', color='#2ecc71', linewidth=2, markersize=8)
    
    plt.xlabel('Noise Probability (p)', fontweight='bold')
    plt.ylabel('Test Accuracy (MNIST 3 vs 8)', fontweight='bold')
    plt.title(f'Task Performance vs Noise (Seed {seed})', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0.4, 1.0)
    
    plot_file = f'figure_task_noise_seed{seed}.png'
    plt.savefig(plot_file, dpi=300)
    print(f"✓ Created {plot_file}")
    
    return acc_results

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_experiment(seed=seed)
