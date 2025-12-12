"""
Multi-Noise Validation Experiment
Tests Depth 6 (Bayesian) vs Depth 2 (Ours) under Depolarizing, Amplitude Damping, and Phase Damping noise.
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

def run_single_experiment(noise_type, noise_levels):
    print(f"Running {noise_type} experiment...")
    
    acc_depth6 = []
    acc_depth2 = []
    
def run_single_experiment(noise_type, noise_levels):
    print(f"Running {noise_type} experiment...")
    
    acc_depth6 = []
    acc_depth2 = []
    
    # Define inputs and params (fixed for consistency)
    # Use superposition to be sensitive to all noise types
    inputs = np.full(4, 0.5) # RY(pi/2) -> |+> state
    params6 = np.full((6, 4), np.pi/4) # Some rotations
    params2 = np.full((2, 4), np.pi/4)
    
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
                if depth == 6: # Linear
                    for i in range(3):
                        qml.CNOT(wires=[i, i+1])
                        apply_noise(noise_type, noise_val, [i, i+1])
                else: # Full (Depth 2)
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
    ideal_state6 = get_state(6, 0.0)
    ideal_state2 = get_state(2, 0.0)
    
    # 2. Run Noise Levels
    for noise in noise_levels:
        # Depth 6
        noisy_state6 = get_state(6, noise)
        fid6 = qml.math.fidelity(ideal_state6, noisy_state6)
        acc_depth6.append(float(fid6))
        
        # Depth 2
        noisy_state2 = get_state(2, noise)
        fid2 = qml.math.fidelity(ideal_state2, noisy_state2)
        acc_depth2.append(float(fid2))
        
    return acc_depth6, acc_depth2
        
    return acc_depth6, acc_depth2

def main():
    print("="*60)
    print("MULTI-NOISE VALIDATION EXPERIMENT")
    print("="*60)
    
    noise_levels = [0.0, 0.02, 0.05, 0.10]
    noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping']
    
    results = {}
    
    for nt in noise_types:
        d6, d2 = run_single_experiment(nt, noise_levels)
        results[nt] = {
            'depth6': d6,
            'depth2': d2
        }
        print(f"  {nt} @ 10%: D6={d6[-1]:.3f}, D2={d2[-1]:.3f}")
        
    # Save results
    with open('multi_noise_results.json', 'w') as f:
        json.dump({'noise_levels': noise_levels, 'results': results}, f, indent=2)
    print("\n✓ Saved results to multi_noise_results.json")
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    titles = {
        'depolarizing': 'Depolarizing Noise',
        'amplitude_damping': 'Amplitude Damping (T1)',
        'phase_damping': 'Phase Damping (T2)'
    }
    
    for i, nt in enumerate(noise_types):
        ax = axes[i]
        d6 = results[nt]['depth6']
        d2 = results[nt]['depth2']
        
        ax.plot(noise_levels, d6, 'o--', label='Bayesian (Depth 6)', color='#e74c3c', linewidth=2)
        ax.plot(noise_levels, d2, 's-', label='Ours (Depth 2)', color='#2ecc71', linewidth=2)
        
        ax.set_title(titles[nt], fontweight='bold')
        ax.set_xlabel('Noise Probability')
        ax.set_ylabel('Signal Fidelity')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        if i == 0:
            ax.legend()
            
        # Annotate improvement at 10%
        imp = d2[-1] / d6[-1] if d6[-1] > 0 else 0
        ax.annotate(f'{imp:.1f}x Better', 
                    xy=(0.1, d2[-1]), xytext=(0.05, d2[-1]+0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig('figure_multi_noise.png', dpi=300)
    print("✓ Created figure_multi_noise.png")
    
    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Noise Model':<20} | {'Level':<6} | {'Bayesian':<10} | {'Ours':<10} | {'Improvement':<10}")
    print("-" * 66)
    for nt in noise_types:
        d6 = results[nt]['depth6'][-1]
        d2 = results[nt]['depth2'][-1]
        imp = d2 / d6 if d6 > 0 else 0
        print(f"{titles[nt]:<20} | 10%    | {d6:<10.3f} | {d2:<10.3f} | {imp:<10.1f}x")
    print("="*60)

if __name__ == "__main__":
    main()
