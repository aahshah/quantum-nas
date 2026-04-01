"""
Generate noise sweep figure: Fidelity vs noise probability (0-10%)
for Depth-2 vs Depth-6 across all three noise models.

Uses data from multi_noise_results.json or multi_noise_results_seed42.json.
Run from project root: python generate_noise_sweep_figure.py

Suggested caption: Signal fidelity vs noise probability for Depth-2 (MO-QAS)
and Depth-6 (baseline) circuits under three noise models. The advantage of
the shallow circuit persists across all noise levels, not only at 10%.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data():
    """Load multi-noise results from JSON (project root or archive)."""
    candidates = [
        Path('multi_noise_results.json'),
        Path('multi_noise_results_seed42.json'),
        Path('archive/multi_noise_results.json'),
        Path('archive/multi_noise_results_seed42.json'),
    ]
    for p in candidates:
        if p.exists():
            with open(p, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(
        "No multi_noise_results*.json found. Run multi_noise_validation_verified.py first."
    )

def main():
    data = load_data()
    noise_levels = np.array(data['noise_levels']) * 100  # Convert to percent for x-axis
    results = data['results']

    titles = {
        'depolarizing': 'Depolarizing',
        'amplitude_damping': 'Amplitude Damping (T1)',
        'phase_damping': 'Phase Damping (T2)',
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.3)

    for i, nt in enumerate(['depolarizing', 'amplitude_damping', 'phase_damping']):
        ax = axes[i]
        d6 = results[nt]['depth6']
        d2 = results[nt]['depth2']

        ax.plot(noise_levels, d6, 'o--', label='Depth 6 (Baseline)', color='#e74c3c',
                linewidth=2, markersize=8)
        ax.plot(noise_levels, d2, 's-', label='Depth 2 (MO-QAS)', color='#2ecc71',
                linewidth=2, markersize=8)

        ax.set_xlabel('Noise Probability (%)', fontweight='bold')
        ax.set_ylabel('Signal Fidelity', fontweight='bold')
        ax.set_title(titles[nt], fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-1, 11)
        ax.set_xticks(noise_levels)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out_path = Path('figure_fidelity_vs_noise.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
