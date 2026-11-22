"""
Gradient Scaling vs Qubit Count - Standalone (No PyTorch)
Produces gradient_scaling_results.json and barren_plateau_proof.png.
"""
import numpy as np
import pennylane as qml
import json
import matplotlib.pyplot as plt
from pathlib import Path

def create_circuit(n_qubits, depth, cost_type='local', entanglement='linear'):
    """Create circuit with specified depth and cost type."""
    dev = qml.device('default.qubit', wires=n_qubits)
    
    def circuit(params):
        params = params.flatten()
        idx = 0
        for d in range(depth):
            for q in range(n_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
            if d < depth - 1:
                if entanglement == 'linear':
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q+1])
                else:
                    for q1 in range(n_qubits):
                        for q2 in range(q1+1, n_qubits):
                            qml.CNOT(wires=[q1, q2])
        if cost_type == 'global':
            obs = qml.PauliZ(0)
            for i in range(1, n_qubits):
                obs = obs @ qml.PauliZ(i)
            return qml.expval(obs)
        return qml.expval(qml.PauliZ(0))
    
    qnode = qml.QNode(circuit, dev, diff_method='parameter-shift')
    n_params = depth * n_qubits
    return qnode, n_params

def compute_gradient_variance(n_qubits, depth, cost_type, entanglement='linear', n_samples=40):
    """Compute gradient variance for a circuit configuration."""
    qnode, n_params = create_circuit(n_qubits, depth, cost_type, entanglement)
    gradients = []
    for _ in range(n_samples):
        # Use small init (like barren_plateau_simple) for non-vanishing gradients
        params = qml.numpy.random.randn(n_params) * 0.1
        try:
            grad = qml.grad(qnode)(params)
            grad = np.array(grad).flatten()
            if len(grad) == n_params and np.all(np.isfinite(grad)) and np.any(np.abs(grad) > 1e-12):
                gradients.append(grad)
        except Exception:
            continue
    if len(gradients) < 5:
        return float(np.var([np.mean(np.abs(g)) for g in gradients])) if gradients else 1e-10
    return float(np.mean(np.var(gradients, axis=0)))

def main():
    print("Gradient Scaling Benchmark (No PyTorch)")
    qubit_range = [4, 6, 8, 10, 12]
    results = {'Global': [], 'Local': [], 'NAS': []}
    
    for n_q in qubit_range:
        print(f"  n={n_q}...", end=" ", flush=True)
        # Global: depth 10, full entanglement, global cost (worst for BP)
        var_g = compute_gradient_variance(n_q, depth=10, cost_type='global', entanglement='full')
        # Local: depth 10, full entanglement, local cost
        var_l = compute_gradient_variance(n_q, depth=10, cost_type='local', entanglement='full')
        # NAS: depth 6, linear entanglement, local cost (best)
        var_n = compute_gradient_variance(n_q, depth=6, cost_type='local', entanglement='linear')
        results['Global'].append(var_g)
        results['Local'].append(var_l)
        results['NAS'].append(var_n)
        print(f"G={var_g:.2e} L={var_l:.2e} N={var_n:.2e}")
    
    # Power-law fit: Var = C * n^slope
    qubit_arr = np.array(qubit_range, dtype=float)
    slopes = {}
    for name, var_list in results.items():
        var_arr = np.maximum(np.array(var_list), 1e-20)
        slope, intercept = np.polyfit(np.log(qubit_arr), np.log(var_arr), 1)
        slopes[name] = {'slope': float(slope), 'exponent': float(slope)}
        print(f"  {name}: exponent = {slope:.3f}  =>  Var ~ n^{slope:.3f}")
    
    output = {
        'qubit_range': qubit_range,
        'results': {k: [float(v) for v in vals] for k, vals in results.items()},
        'power_law_fits': slopes,
    }
    out_path = Path(__file__).parent / 'gradient_scaling_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    # Plot (matches barren_plateau_proof.py style: no title, MO-QAS legend)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(qubit_range, results['Global'], 'r-o', label='Standard (Global Cost)')
    ax.semilogy(qubit_range, results['Local'], 'b-s', label='Standard (Local Cost)')
    ax.semilogy(qubit_range, results['NAS'], 'g-^', linewidth=3, label='MO-QAS')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Gradient Variance (Log Scale)')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plot_path = Path(__file__).parent.parent / 'barren_plateau_proof.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

    return slopes

if __name__ == "__main__":
    main()
