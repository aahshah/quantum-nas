"""
Figure 2: Noise Sweep for arXiv/IEEE submission.
Generates figures/noise_sweep.pdf

Requires: pip install SciencePlots
Run from project root: python figures/figure2_noise_sweep.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee'])
except Exception:
    for _fallback in ('seaborn-v0_8-whitegrid', 'ggplot'):
        try:
            plt.style.use(_fallback)
            break
        except Exception:
            pass
# Set to False if LaTeX (MiKTeX/TeX Live) is not installed
USE_USETEX = False  # Set True if LaTeX (MiKTeX/TeX Live) is installed
plt.rcParams['text.usetex'] = USE_USETEX
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.alpha'] = 0.5

# Colorblind-safe
COLOR_D6 = '#d62728'   # red
COLOR_D2 = '#2ca02c'  # green

def main():
    # Load data (run from project root)
    root = Path(__file__).parent.parent
    json_path = root / 'multi_noise_results_seed42.json'
    if not json_path.exists():
        json_path = root / 'archive' / 'multi_noise_results_seed42.json'
    with open(json_path) as f:
        data = json.load(f)

    noise_levels = np.array(data['noise_levels']) * 100  # to percent
    # New format: results['4'] or results['5']; fallback to flat results
    raw = data['results']
    if '4' in raw:
        results = raw['4']  # 4-qubit knee-point (matches paper scale)
    elif '5' in raw:
        results = raw['5']
    else:
        results = raw

    # Print values for verification
    print("=== Figure 2: State Fidelity (for verification) ===")
    print("Noise levels (%):", noise_levels.tolist())
    for nt in ['depolarizing', 'amplitude_damping', 'phase_damping']:
        print(f"  {nt}:")
        print(f"    Depth-6: {[f'{v:.6f}' for v in results[nt]['depth6']]}")
        print(f"    Depth-2: {[f'{v:.6f}' for v in results[nt]['depth2']]}")
    imp_depol = results['depolarizing']['depth2'][-1] / results['depolarizing']['depth6'][-1]
    print(f"Depolarizing improvement @ 10%: {imp_depol:.2f}x")
    print()

    # Figure: double column IEEE (7.16 x 2.8 inches)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.25)

    titles = {
        'depolarizing': 'Depolarizing',
        'amplitude_damping': 'Amplitude Damping (T1)',
        'phase_damping': 'Phase Damping (T2)',
    }

    for i, nt in enumerate(['depolarizing', 'amplitude_damping', 'phase_damping']):
        ax = axes[i]
        d6 = results[nt]['depth6']
        d2 = results[nt]['depth2']

        # Use both line style and marker shape so curves remain distinct in grayscale.
        ax.plot(noise_levels, d6, marker='o', linestyle='-.', color=COLOR_D6, linewidth=1.5, markersize=5,
                label='Depth-6 Baseline')
        ax.plot(noise_levels, d2, marker='s', linestyle='-', color=COLOR_D2, linewidth=1.6, markersize=5,
                label='Depth-2 Knee-style Template')

        ax.set_xlabel('Noise Probability (%)')
        if i == 0:
            ax.set_ylabel(r'State Fidelity $F$')
            ax.legend(loc='upper right', fontsize=8)
        ax.set_title(titles[nt], fontweight='normal')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-1, 11)
        ax.set_xticks(noise_levels)
        ax.grid(True, linewidth=0.4, alpha=0.5)

        # Improvement annotation at 10% for each panel.
        imp = d2[-1] / d6[-1] if d6[-1] > 0 else 0
        mid_y = (d2[-1] + d6[-1]) / 2
        ax.annotate(rf'{imp:.2f}$\times$', xy=(10, mid_y), xytext=(7.4, min(0.95, mid_y + 0.14)),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                    fontsize=9, ha='center')

    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'noise_sweep.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'noise_sweep.png', dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
