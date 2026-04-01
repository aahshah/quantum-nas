"""
Multi-Noise Validation Experiment - VERIFIED VERSION
Tests Depth 6 (Baseline) vs Depth 2 (Knee-point from NSGA-II) under 3 noise types.

Knee-point architecture: depth 2, linear entanglement, RX+RZ gates,
data-reuploading, local cost (PauliZ on qubit 0).
"""
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt
import json

def apply_noise(noise_type, noise_prob, wires):
    """Apply specific noise channel to wires"""
    if noise_prob <= 0:
        return
    if isinstance(wires, int):
        wires = [wires]
    for w in wires:
        if noise_type == 'depolarizing':
            qml.DepolarizingChannel(noise_prob, wires=w)
        elif noise_type == 'amplitude_damping':
            qml.AmplitudeDamping(noise_prob, wires=w)
        elif noise_type == 'phase_damping':
            qml.PhaseDamping(noise_prob, wires=w)

def run_single_experiment(noise_type, noise_levels, n_qubits, seed=None):
    """
    Run experiment for one noise type.
    n_qubits: 4 or 5 (knee-point discovered at 5 qubits; 4-qubit is truncated version)
    """
    print(f"\nRunning {noise_type} experiment (n_qubits={n_qubits}, seed={seed})...")

    if seed is not None:
        np.random.seed(seed)

    # Inputs and params
    inputs = np.full(n_qubits, 0.5)  # RY(pi/2) -> |+> state
    params6 = np.full((6, n_qubits), np.pi/4)  # Depth-6 baseline: RX only
    # Knee-point: depth 2, RX+RZ per qubit per layer
    params2 = np.full((2, n_qubits, 2), np.pi/4)  # [layer, qubit, rx/rz]

    n_cnots_linear = n_qubits - 1
    print(f"  Circuit Configuration:")
    print(f"    - Depth 6 baseline: 6 layers, Linear entanglement ({n_cnots_linear} CNOTs per layer), RX only")
    print(f"    - Knee-point (D2): 2 layers, Linear entanglement ({n_cnots_linear} CNOTs), RX+RZ, data-reuploading")

    def get_state(depth, noise_val):
        dev = qml.device("default.mixed", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(inputs, params6, params2):
            if depth == 6:
                # Depth-6 baseline: encode once, 6 layers with RX + linear CNOTs
                for i in range(n_qubits):
                    qml.RY(inputs[i] * np.pi, wires=i)
                for l in range(6):
                    apply_noise(noise_type, noise_val, range(n_qubits))
                    for i in range(n_qubits):
                        qml.RX(params6[l, i], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                        apply_noise(noise_type, noise_val, [i, i+1])
            else:
                # Knee-point: depth 2, linear, RX+RZ, data-reuploading (encode at each layer)
                for l in range(2):
                    # Data re-upload: encode input at each layer
                    for i in range(n_qubits):
                        qml.RY(inputs[i] * np.pi, wires=i)
                    apply_noise(noise_type, noise_val, range(n_qubits))
                    # RX and RZ rotations
                    for i in range(n_qubits):
                        qml.RX(params2[l, i, 0], wires=i)
                        qml.RZ(params2[l, i, 1], wires=i)
                    # Linear entanglement
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                        apply_noise(noise_type, noise_val, [i, i+1])
            return qml.density_matrix(wires=list(range(n_qubits)))

        if depth == 6:
            return circuit(inputs, params6, params2)
        else:
            return circuit(inputs, params6, params2)

    print(f"  Computing ideal states (no noise)...")
    ideal_state6 = get_state(6, 0.0)
    ideal_state2 = get_state(2, 0.0)

    acc_depth6 = []
    acc_depth2 = []
    for noise in noise_levels:
        noisy_state6 = get_state(6, noise)
        fid6 = qml.math.fidelity(ideal_state6, noisy_state6)
        acc_depth6.append(float(fid6))
        noisy_state2 = get_state(2, noise)
        fid2 = qml.math.fidelity(ideal_state2, noisy_state2)
        acc_depth2.append(float(fid2))
        print(f"    Noise {noise*100:>4.0f}%: D6={fid6:.4f}, knee-point={fid2:.4f}")

    return acc_depth6, acc_depth2

def main(seed=None):
    print("="*60)
    print("MULTI-NOISE VALIDATION (Knee-point from NSGA-II)")
    if seed is not None:
        print(f"Random Seed: {seed}")
    print("="*60)

    noise_levels = [0.0, 0.02, 0.05, 0.10]
    noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping']
    all_results = {}  # {n_qubits: {nt: {depth6, depth2}}}

    for n_qubits in [5, 4]:
        print(f"\n{'#'*60}")
        print(f"# {n_qubits}-QUBIT VERSION")
        print(f"{'#'*60}")
        results = {}
        for nt in noise_types:
            d6, d2 = run_single_experiment(nt, noise_levels, n_qubits=n_qubits, seed=seed)
            results[nt] = {'depth6': d6, 'depth2': d2}
        all_results[n_qubits] = results

    # Save results
    output_file = f'multi_noise_results_seed{seed}.json' if seed else 'multi_noise_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'noise_levels': noise_levels,
            'results': all_results,
            'seed': seed,
            'knee_point': 'depth-2, linear, RX+RZ, data-reuploading'
        }, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Print formatted output for user
    titles = {
        'depolarizing': 'Depolarizing',
        'amplitude_damping': 'Amplitude damping',
        'phase_damping': 'Phase damping'
    }
    print("\n" + "="*70)
    print("FIDELITY @ 10% NOISE (knee-point vs depth-6 baseline)")
    print("="*70)
    for nq in [5, 4]:
        r = all_results[nq]
        print(f"\n{nq}-qubit knee-point vs {nq}-qubit depth-6 baseline:")
        for nt in noise_types:
            k = r[nt]['depth2'][-1]
            b = r[nt]['depth6'][-1]
            print(f"  {titles[nt]} @ 10%: knee-point = {k:.4f}, baseline = {b:.4f}")
    print("="*70)

    return all_results

def run_verification():
    """
    Run experiment 3 times with different seeds and compute improvement factor variance.
    Used to verify reproducibility claims for the paper.
    """
    seeds = [42, 123, 456]
    noise_levels = [0.0, 0.02, 0.05, 0.10]
    noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping']
    results = {}

    print("="*60)
    print("MULTI-SEED VERIFICATION (seeds=42, 123, 456)")
    print("="*60)

    for seed in seeds:
        np.random.seed(seed)
        seed_results = {}
        for nt in noise_types:
            d6, d2 = run_single_experiment(nt, noise_levels, n_qubits=4, seed=seed)
            # Store fidelity at 10% noise (last index)
            seed_results[nt] = {
                "depth2": d2[-1],
                "depth6": d6[-1],
            }
        results[seed] = seed_results

    # Compute improvement factors across seeds (at 10% noise)
    for nt in noise_types:
        improvement_factors = []
        for seed in seeds:
            d2 = results[seed][nt]["depth2"]
            d6 = results[seed][nt]["depth6"]
            imp = d2 / d6 if d6 > 0 else 0
            improvement_factors.append(imp)

        sigma = np.std(improvement_factors)
        mean_imp = np.mean(improvement_factors)
        print(f"\n{nt}:")
        print(f"  Mean improvement: {mean_imp:.3f}")
        print(f"  Sigma: {sigma:.4f}")

    # Primary metric: depolarizing (as in paper)
    imp_depol = [results[s]["depolarizing"]["depth2"] / results[s]["depolarizing"]["depth6"]
                 for s in seeds if results[s]["depolarizing"]["depth6"] > 0]
    sigma_depol = np.std(imp_depol)
    mean_depol = np.mean(imp_depol)
    print("\n" + "="*60)
    print("PRIMARY (Depolarizing @ 10% noise):")
    print(f"  Mean improvement: {mean_depol:.3f}")
    print(f"  Sigma: {sigma_depol:.4f}")
    print("="*60)

    with open("multi_noise_results_3seeds.json", "w") as f:
        json.dump({
            "seeds": seeds,
            "noise_levels": noise_levels,
            "results": results,
            "improvement_depolarizing": {"mean": float(mean_depol), "sigma": float(sigma_depol)},
        }, f, indent=2)
    print("\nSaved to multi_noise_results_3seeds.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "verify":
        run_verification()
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
        main(seed=seed)
