"""
Multi-Class MNIST Noise Validation (0-9) - Feature Extraction Approach
Uses quantum circuit as a fixed feature extractor + trained classical classifier.
This avoids gradient issues and ensures convergence.
"""
import pennylane as qml
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

def get_circuit_features(depth, params, inputs, noise_prob=0.0):
    """
    Returns 4 expectation values (features) from the circuit.
    """
    dev = qml.device("default.mixed", wires=4)
    
    @qml.qnode(dev)
    def circuit():
        # Encoding
        for i in range(4):
            qml.RY(inputs[i], wires=i)
        
        # Layers
        for l in range(depth):
            apply_noise(noise_prob, range(4))
            
            # Rotation
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
                            
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    
    return circuit()

# --- 3. Training Loop (Feature Extraction + Logistic Regression) ---
def train_and_evaluate(depth, X_train, y_train, X_test, y_test, seed=42):
    print(f"\nTraining Depth {depth} model...")
    
    np.random.seed(seed)
    
    # Initialize fixed random parameters for the quantum circuit
    params = np.random.uniform(0, 2*np.pi, (depth, 8))
    
    # Extract features for training set (Ideal Simulator)
    print("  Extracting training features...")
    X_train_features = []
    for x in X_train:
        feats = get_circuit_features(depth, params, x, noise_prob=0.0)
        X_train_features.append(feats)
    X_train_features = np.array(X_train_features)
    
    # Train classical classifier on top
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train_features, y_train)
    
    train_acc = clf.score(X_train_features, y_train)
    print(f"  Train Accuracy: {train_acc:.1%}")
    
    # Evaluate under noise
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    accuracies = []
    confidences = []
    
    print("  Evaluating under noise...")
    for noise in noise_levels:
        X_test_features = []
        for x in X_test:
            feats = get_circuit_features(depth, params, x, noise_prob=noise)
            X_test_features.append(feats)
        X_test_features = np.array(X_test_features)
        
        acc = clf.score(X_test_features, y_test)
        
        # Calculate average confidence (max probability)
        probs = clf.predict_proba(X_test_features)
        conf = np.mean(np.max(probs, axis=1))
        
        accuracies.append(acc)
        confidences.append(conf)
        print(f"    Noise {noise*100:>2.0f}%: Acc={acc:.1%}, Conf={conf:.3f}")
        
    return accuracies, confidences

# --- 4. Main Experiment ---
def run_experiment(seed=42):
    print("="*60)
    print(f"MULTI-CLASS MNIST VALIDATION (Feature Extraction) - Seed {seed}")
    print("="*60)
    
    X_train, X_test, y_train, y_test = load_multiclass_data()
    print(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    
    # Use subset for speed
    subset = 500
    X_train, y_train = X_train[:subset], y_train[:subset]
    X_test, y_test = X_test[:200], y_test[:200]
    
    # Run for both depths
    acc6, conf6 = train_and_evaluate(6, X_train, y_train, X_test, y_test, seed=seed)
    acc2, conf2 = train_and_evaluate(2, X_train, y_train, X_test, y_test, seed=seed)
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    # Save results
    results = {
        'seed': seed,
        'noise_levels': noise_levels,
        'depth6_acc': acc6, 'depth2_acc': acc2,
        'depth6_conf': conf6, 'depth2_conf': conf2
    }
    
    output_file = f'multiclass_features_seed{seed}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved to {output_file}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(noise_levels, acc6, 'o--', label='Depth 6', color='#e74c3c', linewidth=2)
    ax1.plot(noise_levels, acc2, 's-', label='Depth 2', color='#2ecc71', linewidth=2)
    ax1.set_xlabel('Noise Probability')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Multi-Class Accuracy vs Noise')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Confidence
    ax2.plot(noise_levels, conf6, 'o--', label='Depth 6', color='#e74c3c', linewidth=2)
    ax2.plot(noise_levels, conf2, 's-', label='Depth 2', color='#2ecc71', linewidth=2)
    ax2.set_xlabel('Noise Probability')
    ax2.set_ylabel('Avg Confidence')
    ax2.set_title('Prediction Confidence vs Noise')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'figure_multiclass_features_seed{seed}.png', dpi=300)
    print(f"✓ Created figure")

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_experiment(seed=seed)
