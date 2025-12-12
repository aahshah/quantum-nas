"""
Multi-Class MNIST Noise Validation (0-9)
Uses continuous probability outputs to properly measure degradation.
"""
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt

# --- 1. Data Setup (Multi-class MNIST: 0-9) ---
def load_multiclass_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Reduce to 4 features for 4 qubits
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Take top 4 features by variance
    variances = np.var(X, axis=0)
    top_4_idx = np.argsort(variances)[-4:]
    X = X[:, top_4_idx]
    
    # Normalize to [0, pi]
    X = (X - X.min()) / (X.max() - X.min()) * np.pi
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. Circuit Definitions ---
def apply_noise(noise_prob, wires):
    if noise_prob > 0:
        for w in wires:
            qml.DepolarizingChannel(noise_prob, wires=w)

def get_multiclass_circuit(depth, noise_prob=0.0):
    """
    Returns a circuit that outputs 4 expectation values (one per qubit).
    We'll use these to classify into 10 classes via a simple mapping.
    """
    dev = qml.device("default.mixed", wires=4)
    
    @qml.qnode(dev)
    def circuit(inputs, params):
        # Encoding
        for i in range(4):
            qml.RY(inputs[i], wires=i)
        
        # Layers
        for l in range(depth):
            apply_noise(noise_prob, range(4))
            
            # Rotation (RX and RZ for expressivity)
            for i in range(4):
                qml.RX(params[l, i], wires=i)
                qml.RZ(params[l, i+4], wires=i)
            
            # Entanglement
            if depth == 6:  # Sparse (Linear)
                for i in range(3):
                    qml.CNOT(wires=[i, i+1])
                    if noise_prob > 0:
                        qml.DepolarizingChannel(noise_prob, wires=i)
                        qml.DepolarizingChannel(noise_prob, wires=i+1)
            else:  # Dense (Full) - Ours
                for i in range(4):
                    for j in range(i+1, 4):
                        qml.CNOT(wires=[i, j])
                        if noise_prob > 0:
                            qml.DepolarizingChannel(noise_prob, wires=i)
                            qml.DepolarizingChannel(noise_prob, wires=j)
                            
        # Return all 4 qubit measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    
    return circuit

# --- 3. Multi-class Prediction ---
def predict_class(circuit_output, weights):
    """
    Convert 4 expectation values to 10-class probabilities.
    Uses a simple linear layer: logits = weights @ circuit_output
    """
    logits = weights @ np.array(circuit_output)
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs

def predict_class_label(circuit_output, weights):
    probs = predict_class(circuit_output, weights)
    return np.argmax(probs)

# --- 4. Training Loop ---
def train_multiclass_model(depth, X_train, y_train, steps=30, seed=42):
    print(f"\nTraining Depth {depth} model...")
    
    np.random.seed(seed)
    
    # Initialize params: layers x (4 RX + 4 RZ)
    params = np.random.uniform(0, 2*np.pi, (depth, 8))
    # Initialize classification weights: 10 classes x 4 qubits
    weights = np.random.uniform(-0.1, 0.1, (10, 4))
    
    circuit = get_multiclass_circuit(depth, noise_prob=0.0)
    learning_rate = 0.01
    
    for i in range(steps):
        # Forward pass
        total_loss = 0
        for x, label in zip(X_train, y_train):
            circuit_out = circuit(x, params)
            probs = predict_class(circuit_out, weights)
            # Cross-entropy loss
            total_loss += -np.log(probs[label] + 1e-10)
        
        avg_loss = total_loss / len(X_train)
        
        # Simple gradient descent on weights only (params stay fixed after random init)
        # This is a simplification but works for demonstration
        for x, label in zip(X_train, y_train):
            circuit_out = circuit(x, params)
            probs = predict_class(circuit_out, weights)
            
            # Gradient of cross-entropy w.r.t. weights
            grad = np.zeros_like(weights)
            for c in range(10):
                if c == label:
                    grad[c] = (probs[c] - 1) * np.array(circuit_out)
                else:
                    grad[c] = probs[c] * np.array(circuit_out)
            
            weights -= learning_rate * grad / len(X_train)
        
        if (i+1) % 10 == 0:
            # Calculate accuracy
            correct = 0
            for x, label in zip(X_train, y_train):
                circuit_out = circuit(x, params)
                pred = predict_class_label(circuit_out, weights)
                if pred == label:
                    correct += 1
            acc = correct / len(X_train)
            print(f"  Step {i+1}: Loss = {avg_loss:.4f}, Train Acc = {acc:.1%}")
            
    return params, weights

# --- 5. Evaluation ---
def evaluate_model(depth, params, weights, X_test, y_test, noise_prob=0.0):
    """
    Returns accuracy and average confidence (max probability).
    Confidence measures how degraded the predictions are.
    """
    circuit = get_multiclass_circuit(depth, noise_prob)
    
    correct = 0
    total_confidence = 0
    
    for x, label in zip(X_test, y_test):
        circuit_out = circuit(x, params)
        probs = predict_class(circuit_out, weights)
        pred = np.argmax(probs)
        
        if pred == label:
            correct += 1
        
        # Confidence = max probability
        total_confidence += np.max(probs)
    
    accuracy = correct / len(X_test)
    avg_confidence = total_confidence / len(X_test)
    
    return accuracy, avg_confidence

# --- 6. Main Experiment ---
def run_experiment(seed=42):
    print("="*60)
    print(f"MULTI-CLASS MNIST VALIDATION (0-9) - Seed {seed}")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_multiclass_data()
    print(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Classes: 0-9 (10 classes)")
    
    # Train both models (use subset for speed)
    print("\n[Training Phase - Ideal Simulator]")
    params6, weights6 = train_multiclass_model(6, X_train[:200], y_train[:200], steps=30, seed=seed)
    params2, weights2 = train_multiclass_model(2, X_train[:200], y_train[:200], steps=30, seed=seed)
    
    # Evaluate under noise
    noise_levels = [0.0, 0.05, 0.10, 0.15]
    results = {
        'depth6_acc': [],
        'depth2_acc': [],
        'depth6_conf': [],
        'depth2_conf': []
    }
    
    print("\n[Noisy Performance Evaluation]")
    for noise in noise_levels:
        acc6, conf6 = evaluate_model(6, params6, weights6, X_test, y_test, noise)
        acc2, conf2 = evaluate_model(2, params2, weights2, X_test, y_test, noise)
        
        results['depth6_acc'].append(acc6)
        results['depth2_acc'].append(acc2)
        results['depth6_conf'].append(conf6)
        results['depth2_conf'].append(conf2)
        
        print(f"  Noise {noise*100:>4.0f}%:")
        print(f"    D6: Acc={acc6:.1%}, Conf={conf6:.3f}")
        print(f"    D2: Acc={acc2:.1%}, Conf={conf2:.3f}")
    
    # Save results
    output_file = f'multiclass_results_seed{seed}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'seed': seed,
            'noise_levels': noise_levels,
            'results': results
        }, f, indent=2)
    print(f"\n✓ Saved to {output_file}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(noise_levels, results['depth6_acc'], 'o--', label='Depth 6', color='#e74c3c', linewidth=2, markersize=8)
    ax1.plot(noise_levels, results['depth2_acc'], 's-', label='Depth 2', color='#2ecc71', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Probability', fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontweight='bold')
    ax1.set_title('Multi-Class Accuracy vs Noise', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Confidence plot
    ax2.plot(noise_levels, results['depth6_conf'], 'o--', label='Depth 6', color='#e74c3c', linewidth=2, markersize=8)
    ax2.plot(noise_levels, results['depth2_conf'], 's-', label='Depth 2', color='#2ecc71', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Probability', fontweight='bold')
    ax2.set_ylabel('Avg Confidence (Max Prob)', fontweight='bold')
    ax2.set_title('Prediction Confidence vs Noise', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_file = f'figure_multiclass_seed{seed}.png'
    plt.savefig(plot_file, dpi=300)
    print(f"✓ Created {plot_file}")
    
    return results

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_experiment(seed=seed)
