"""
Multi-Noise Validation Experiment - VERIFIED VERSION
Tests Depth 6 (Bayesian) vs Depth 2 (Ours) under 3 noise types.
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

def run_single_experiment(noise_type, noise_levels, seed=None):
    """Run experiment for one noise type"""
    print(f"\nRunning {noise_type} experiment (seed={seed})...")
    
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Define inputs and params (fixed for consistency)
    # Use superposition to be sensitive to all noise types
    inputs = np.full(4, 0.5)  # RY(pi/2) -> |+> state
    params6 = np.full((6, 4), np.pi/4)  # Some rotations
    params2 = np.full((2, 4), np.pi/4)
    
    print(f"  Circuit Configuration:")
    print(f"    - Depth 6: {6} layers, Linear entanglement (3 CNOTs per layer)")
    print(f"    - Depth 2: {2} layers, Full entanglement (6 CNOTs per layer)")
    
    # --- Helper to run circuit and get state ---
    def get_state(depth, noise_val):
        dev = qml.device("default.mixed", wires=4)
        
        @qml.qnode(dev)
        def circuit(inputs, params):
            # Encoding
            for i in range(4):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Layers
            for l in range(depth):
                # Noise after layer
                apply_noise(noise_type, noise_val, range(4))
                
                # Rotation
                for i in range(4):
                    qml.RX(params[l, i], wires=i)
                
                # Entanglement
                if depth == 6:  # Linear (Bayesian-like)
                    for i in range(3):
                        qml.CNOT(wires=[i, i+1])
                        apply_noise(noise_type, noise_val, [i, i+1])
                else:  # Full (Ours-like, Depth 2)
                    for i in range(4):
                        for j in range(i+1, 4):
                            qml.CNOT(wires=[i, j])
                            apply_noise(noise_type, noise_val, [i, j])
                            
            return qml.density_matrix(wires=[0, 1, 2, 3])
            
        if depth == 6:
            return circuit(inputs, params6)
        else:
            return circuit(inputs, params2)

    # 1. Get Ideal States (Noise = 0)
    print(f"  Computing ideal states (no noise)...")
    ideal_state6 = get_state(6, 0.0)
    ideal_state2 = get_state(2, 0.0)
    
    # 2. Run Noise Levels
    acc_depth6 = []
    acc_depth2 = []
    
    for noise in noise_levels:
        # Depth 6
        noisy_state6 = get_state(6, noise)
        fid6 = qml.math.fidelity(ideal_state6, noisy_state6)
        acc_depth6.append(float(fid6))
        
        # Depth 2
        noisy_state2 = get_state(2, noise)
        fid2 = qml.math.fidelity(ideal_state2, noisy_state2)
        acc_depth2.append(float(fid2))
        
        print(f"    Noise {noise*100:>4.0f}%: D6={fid6:.4f}, D2={fid2:.4f}")
        
    return acc_depth6, acc_depth2

def main(seed=None):
    print("="*60)
    print("MULTI-NOISE VALIDATION EXPERIMENT")
    if seed is not None:
        print(f"Random Seed: {seed}")
    print("="*60)
    
    noise_levels = [0.0, 0.02, 0.05, 0.10]
    noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping']
    
    results = {}
    
    for nt in noise_types:
        d6, d2 = run_single_experiment(nt, noise_levels, seed=seed)
        results[nt] = {
            'depth6': d6,
            'depth2': d2
        }
    
    # Save results
    output_file = f'multi_noise_results_seed{seed}.json' if seed else 'multi_noise_results.json'
    with open(output_file, 'w') as f:
        json.dump({'noise_levels': noise_levels, 'results': results, 'seed': seed}, f, indent=2)
    print(f"\n✓ Saved results to {output_file}")
    
    # Print Summary Table
    print("\n" + "="*70)
    print(f"{'Noise Model':<25} | {'Level':<6} | {'Bayesian (D6)':<12} | {'Ours (D2)':<12} | {'Improvement':<10}")
    print("-" * 70)
    
    titles = {
        'depolarizing': 'Depolarizing',
        'amplitude_damping': 'Amplitude Damping (T1)',
        'phase_damping': 'Phase Damping (T2)'
    }
    
    for nt in noise_types:
        d6 = results[nt]['depth6'][-1]
        d2 = results[nt]['depth2'][-1]
        imp = d2 / d6 if d6 > 0 else 0
        print(f"{titles[nt]:<25} | 10%    | {d6:<12.4f} | {d2:<12.4f} | {imp:<10.2f}x")
    print("="*70)
    
    return results

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(seed=seed)
