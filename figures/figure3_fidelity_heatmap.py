"""
Figure 3: Fidelity Heatmap for arXiv/IEEE submission.
Generates figures/fidelity_heatmap.pdf and figures/fidelity_heatmap.png

Requires: pip install SciencePlots
Run from project root: python figures/figure3_fidelity_heatmap.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
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
plt.rcParams['axes.grid'] = False

noise_levels = [0, 2, 5, 10]  # percent
NOISE_MODELS = ['Depolarizing', 'Amplitude Damping', 'Phase Damping']
NOISE_KEYS = ['depolarizing', 'amplitude_damping', 'phase_damping']

COL_TITLES = ['Depolarizing', r'Amplitude Damping ($T_1$)', r'Phase Damping ($T_2$)']
ROW_LABELS = ['Depth-6 Baseline', 'MO-QAS Knee-style Template']


def load_fidelity_data(root: Path):
    """Load fidelity from multi_noise_results_seed42.json (4-qubit knee-point)."""
    json_path = root / 'multi_noise_results_seed42.json'
    if not json_path.exists():
        json_path = root / 'archive' / 'multi_noise_results_seed42.json'
    with open(json_path) as f:
        data = json.load(f)
    raw = data['results']
    r = raw.get('4', raw.get('5', raw))
    fidelity = {}
    improvement = {}
    for name, key in zip(NOISE_MODELS, NOISE_KEYS):
        d6 = [round(v, 3) for v in r[key]['depth6']]
        d2 = [round(v, 3) for v in r[key]['depth2']]
        fidelity[name] = {'Depth-6': d6, 'Depth-2': d2}
        imp = d2[-1] / d6[-1] if d6[-1] > 0 else 0
        improvement[name] = f'{imp:.2f}×'
    return fidelity, improvement


def luminance(r, g, b):
    """Approximate luminance for choosing black vs white text."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def main():
    root = Path(__file__).parent.parent
    fidelity, IMPROVEMENT = load_fidelity_data(root)

    fig_width = 8.5
    fig_height = 3.4
    fig = plt.figure(figsize=(fig_width, fig_height))
    # GridSpec: top row, thin improvement row, bottom row (heights ~1 : 0.4 : 1)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 0.5, 1.2], hspace=0.35, wspace=0.5)
    axes = np.array([
        [fig.add_subplot(gs[0, j]) for j in range(3)],
        [fig.add_subplot(gs[2, j]) for j in range(3)],
    ])

    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=vmin, vmax=vmax)

    for row_idx, row_label in enumerate(ROW_LABELS):
        depth_key = 'Depth-6' if row_idx == 0 else 'Depth-2'
        for col_idx, noise_model in enumerate(NOISE_MODELS):
            ax = axes[row_idx, col_idx]
            vals = np.array(fidelity[noise_model][depth_key]).reshape(1, 4)
            # aspect=2 makes each cell wider so tick labels don't overlap
            im = ax.imshow(vals, cmap=cmap, norm=norm, aspect=2,
                           extent=[0, 4, 1, 0])  # explicit extent for correct cell centers

            # Tick labels — horizontal, wider cells give room
            ax.set_xticks(np.arange(4) + 0.5)
            ax.set_xticklabels([f'{n}%' for n in noise_levels], fontsize=7)
            ax.tick_params(axis='x', pad=3)
            ax.set_yticks([])
            ax.set_xlabel('')

            # Column titles on top row only
            if row_idx == 0:
                title = COL_TITLES[col_idx]
                ax.set_title(title, fontsize=9, pad=8)

            # Annotate each cell with fidelity value — center in its own cell
            for j in range(4):
                v = vals[0, j]
                rgb = cmap(norm(v))[:3]
                lum = luminance(*rgb)
                text_color = 'white' if lum < 0.5 else 'black'
                # Cell center: (j + 0.5, 0.5) with extent [0,4,1,0]
                ax.text(j + 0.5, 0.5, f'{v:.3f}', ha='center', va='center', fontsize=7,
                        color=text_color, fontweight='bold')

    # Row labels on the left — reserve space so text isn't clipped
    for row_idx, row_label in enumerate(ROW_LABELS):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=8, labelpad=4)

    # Dedicated improvement row (middle of GridSpec) — centered in each column
    imp_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    for col_idx in range(3):
        ax_imp = imp_axes[col_idx]
        ax_imp.set_axis_off()
        imp = IMPROVEMENT[NOISE_MODELS[col_idx]]
        ax_imp.text(0.5, 0.5, f'Improvement\n{imp}', ha='center', va='center', fontsize=8,
                    fontweight='bold', style='italic', transform=ax_imp.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8', edgecolor='gray', alpha=0.95))

    fig.subplots_adjust(left=0.18, right=0.85, bottom=0.08, top=0.95)

    # Arrows from improvement box to the heatmaps — point at 10% column (improvement is at 10% noise)
    # extent [0,4,1,0]: 10% is rightmost column, center at x=3.5 → axes fraction x ≈ 0.875
    x_10pct = (3.5 - 0) / (4 - 0)  # center of 10% column in axes fraction
    for col_idx in range(3):
        ax_imp = imp_axes[col_idx]
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]
        # Arrow up: to bottom of top heatmap at 10% column
        arrow_up = ConnectionPatch(
            xyA=(0.5, 1.0), xyB=(x_10pct, 0.0),
            coordsA='axes fraction', coordsB='axes fraction',
            axesA=ax_imp, axesB=ax_top,
            arrowstyle='->', color='gray', linewidth=1.2, mutation_scale=8
        )
        fig.add_artist(arrow_up)
        # Arrow down: to top of bottom heatmap at 10% column
        arrow_down = ConnectionPatch(
            xyA=(0.5, 0.0), xyB=(x_10pct, 1.0),
            coordsA='axes fraction', coordsB='axes fraction',
            axesA=ax_imp, axesB=ax_bot,
            arrowstyle='->', color='gray', linewidth=1.2, mutation_scale=8
        )
        fig.add_artist(arrow_down)
    fig.text(0.5, 0.02, 'Noise Probability (%)', ha='center', fontsize=9)

    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'State Fidelity $F$', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    out_pdf = out_dir / 'fidelity_heatmap.pdf'
    out_png = out_dir / 'fidelity_heatmap.png'
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == '__main__':
    main()
