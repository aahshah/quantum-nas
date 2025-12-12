"""
Generate Publication-Quality Multi-Noise Figure
Uses verified results from Seed 42.
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_figure():
    # Load results
    with open('multi_noise_results_seed42.json', 'r') as f:
        data = json.load(f)
    
    noise_levels = data['noise_levels']
    results = data['results']
    
    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    
    titles = {
        'depolarizing': 'Depolarizing Noise',
        'amplitude_damping': 'Amplitude Damping (T1)',
        'phase_damping': 'Phase Damping (T2)'
    }
    
    noise_types = ['depolarizing', 'amplitude_damping', 'phase_damping']
    
    for i, nt in enumerate(noise_types):
        ax = axes[i]
        d6 = results[nt]['depth6']
        d2 = results[nt]['depth2']
        
        # Plot lines
        ax.plot(noise_levels, d6, 'o--', label='Bayesian (Depth 6)', 
                color='#e74c3c', linewidth=2.5, markersize=8)
        ax.plot(noise_levels, d2, 's-', label='Ours (Depth 2)', 
                color='#2ecc71', linewidth=2.5, markersize=8)
        
        # Styling
        ax.set_title(titles[nt], fontweight='bold', pad=15)
        ax.set_xlabel('Noise Probability (p)', fontweight='bold')
        ax.set_ylabel('Signal Fidelity', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(noise_levels)
        
        # Add improvement annotation at 10%
        imp = d2[-1] / d6[-1] if d6[-1] > 0 else 0
        ax.annotate(f'{imp:.1f}x Better', 
                    xy=(0.1, d2[-1]), xytext=(0.06, d2[-1]+0.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        if i == 0:
            ax.legend(loc='lower left', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('figure_multi_noise_verified.png', dpi=300, bbox_inches='tight')
    print("✓ Created figure_multi_noise_verified.png")

if __name__ == "__main__":
    generate_figure()
