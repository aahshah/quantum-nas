"""
Publication-Quality Figures for Quantum NAS Paper
Generates clear, professional figures that explain the dual contribution
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def create_all_figures(results):
    """
    Create all publication figures
    results = {
        'Random': {'acc': 0.56, 'depth': 13},
        'GNN': {'acc': 0.58, 'depth': 11},
        'Trans_Scalar': {'acc': 0.68, 'depth': 12},
        'Ours': {'acc': 0.70, 'depth': 8}
    }
    """
    
    # Figure 1: Main Results - Dual Comparison
    create_dual_comparison(results)
    
    # Figure 2: Pareto Front - Energy vs Accuracy
    create_pareto_front(results)
    
    # Figure 3: Ablation Studies
    create_ablation_study(results)
    
    # Figure 4: System Overview Diagram
    create_system_diagram()
    
    print("\n✅ All figures created successfully!")
    print("\nGenerated figures:")
    print("  1. figure1_main_results.png - Dual bar chart (accuracy + energy)")
    print("  2. figure2_pareto_front.png - Scatter plot showing Pareto dominance")
    print("  3. figure3_ablation_study.png - Ablation analysis")
    print("  4. figure4_system_overview.png - System architecture diagram")

def create_dual_comparison(results):
    """Figure 1: Side-by-side comparison of accuracy and energy"""
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    methods = list(results.keys())
    accs = [results[m]['acc'] for m in methods]
    depths = [results[m]['depth'] for m in methods]
    
    colors = ['#808080', '#3498db', '#f39c12', '#e74c3c']  # Gray, Blue, Orange, Red
    
    # Subplot 1: Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(range(len(methods)), accs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(['Random', 'GNN+\nScalar', 'Trans+\nScalar', 'Ours\n(Trans+MO)'], rotation=0)
    ax1.set_ylim([0, 1.0])
    ax1.set_title('(a) Classification Accuracy', fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Random Guess')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars1, accs)):
        height = bar.get_height()
        label = f'{acc:.1%}'
        if i == len(accs) - 1:  # Highlight "Ours"
            label = f'{acc:.1%}\n★'
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                    label, ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontsize=11)
    
    # Subplot 2: Energy (Circuit Depth)
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(len(methods)), depths, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Circuit Depth (Energy Proxy)', fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(['Random', 'GNN+\nScalar', 'Trans+\nScalar', 'Ours\n(Trans+MO)'], rotation=0)
    ax2.set_ylim([0, max(depths) * 1.2])
    ax2.set_title('(b) Energy Consumption', fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels and energy savings annotation
    for i, (bar, depth) in enumerate(zip(bars2, depths)):
        height = bar.get_height()
        label = f'{depth}'
        if i == len(depths) - 1:  # Highlight "Ours"
            label = f'{depth}\n★'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    label, ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    label, ha='center', va='bottom', fontsize=11)
    
    # Add energy savings annotation
    if depths[3] < depths[0]:  # Ours vs Random
        savings = (depths[0] - depths[3]) / depths[0] * 100
        ax2.annotate(f'{savings:.0f}% Energy\nSavings', 
                    xy=(3, depths[3]), xytext=(2.5, depths[0] * 0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                    fontsize=11, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Main Results: Accuracy and Energy Efficiency', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure1_main_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created figure1_main_results.png")

def create_pareto_front(results):
    """Figure 2: Pareto front showing energy-accuracy tradeoff"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = list(results.keys())
    accs = [results[m]['acc'] for m in methods]
    depths = [results[m]['depth'] for m in methods]
    
    colors = ['#808080', '#3498db', '#f39c12', '#e74c3c']
    markers = ['o', 'd', 's', '*']
    sizes = [150, 150, 150, 400]
    labels = ['Random', 'GNN+Scalar', 'Trans+Scalar', 'Ours (Trans+MO)']
    
    # Plot points
    for i, (depth, acc, color, marker, size, label) in enumerate(zip(depths, accs, colors, markers, sizes, labels)):
        ax.scatter(depth, acc, s=size, c=color, marker=marker, 
                  edgecolors='black', linewidth=2, label=label, zorder=10, alpha=0.9)
    
    # Add labels for each point
    for i, (depth, acc, label) in enumerate(zip(depths, accs, labels)):
        offset_x = 0.5 if i < 3 else -0.8
        offset_y = 0.02 if i != 1 else -0.03
        ax.annotate(f'{acc:.0%}', 
                   xy=(depth, acc), xytext=(depth + offset_x, acc + offset_y),
                   fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    
    # Highlight Pareto dominance
    ax.annotate('Pareto Optimal:\nHigh Accuracy\n+ Low Energy', 
               xy=(depths[3], accs[3]), xytext=(depths[3] + 2, accs[3] - 0.08),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='red'),
               fontsize=12, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                        edgecolor='red', linewidth=2, alpha=0.9))
    
    # Draw dominated region
    if depths[3] < max(depths) and accs[3] > min(accs):
        rect = mpatches.Rectangle((depths[3], min(accs) - 0.05), 
                                  max(depths) - depths[3], accs[3] - min(accs) + 0.05,
                                  linewidth=2, linestyle='--', edgecolor='red', 
                                  facecolor='red', alpha=0.1, label='Dominated Region')
        ax.add_patch(rect)
    
    ax.set_xlabel('Circuit Depth (Lower = More Energy-Efficient)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (Higher = Better)', fontsize=13, fontweight='bold')
    ax.set_title('Energy-Accuracy Tradeoff: Pareto Front Analysis', fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([min(depths) - 2, max(depths) + 2])
    ax.set_ylim([min(accs) - 0.05, max(accs) + 0.08])
    
    plt.tight_layout()
    plt.savefig('figure2_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created figure2_pareto_front.png")

def create_ablation_study(results):
    """Figure 3: Ablation study showing contribution of each component"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ablation 1: Predictor Comparison (GNN vs Transformer)
    predictors = ['GNN', 'Transformer']
    pred_accs = [results['GNN']['acc'], results['Trans_Scalar']['acc']]
    colors1 = ['#3498db', '#f39c12']
    
    bars1 = ax1.bar(predictors, pred_accs, color=colors1, edgecolor='black', linewidth=1.5, alpha=0.8, width=0.6)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax1.set_title('(a) Predictor Ablation:\nGNN vs Graph Transformer', fontweight='bold', fontsize=13)
    ax1.set_ylim([0.5, 0.75])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, acc in zip(bars1, pred_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    if pred_accs[1] > pred_accs[0]:
        gain = (pred_accs[1] - pred_accs[0]) * 100
        ax1.annotate(f'+{gain:.1f}%', 
                    xy=(1, pred_accs[1]), xytext=(0.5, pred_accs[1] + 0.03),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                    fontsize=11, fontweight='bold', color='green')
    
    # Ablation 2: Search Strategy (Scalar vs Multi-Objective)
    strategies = ['Scalar\nSearch', 'Multi-Obj\nSearch']
    strat_accs = [results['Trans_Scalar']['acc'], results['Ours']['acc']]
    strat_depths = [results['Trans_Scalar']['depth'], results['Ours']['depth']]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, strat_accs, width, label='Accuracy', 
                     color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2_twin = ax2.twinx()
    bars2b = ax2_twin.bar(x + width/2, strat_depths, width, label='Depth (Energy)', 
                          color='#e67e22', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax2.set_ylabel('Accuracy', fontweight='bold', fontsize=12, color='#2ecc71')
    ax2_twin.set_ylabel('Circuit Depth', fontweight='bold', fontsize=12, color='#e67e22')
    ax2.set_title('(b) Search Strategy Ablation:\nScalar vs Multi-Objective', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.set_ylim([0.6, 0.75])
    ax2_twin.set_ylim([0, 15])
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax2_twin.tick_params(axis='y', labelcolor='#e67e22')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, acc in zip(bars2a, strat_accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, depth in zip(bars2b, strat_depths):
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                     f'{depth}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.suptitle('Ablation Studies: Component Contributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure3_ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created figure3_ablation_study.png")

def create_system_diagram():
    """Figure 4: System overview diagram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Define boxes
    boxes = [
        {'xy': (0.05, 0.7), 'width': 0.15, 'height': 0.2, 'text': 'Input:\nMNIST\nDataset', 'color': '#ecf0f1'},
        {'xy': (0.25, 0.7), 'width': 0.18, 'height': 0.2, 'text': 'Bipartite\nGraph\nEncoding', 'color': '#3498db'},
        {'xy': (0.48, 0.7), 'width': 0.20, 'height': 0.2, 'text': 'Graph\nTransformer\nPredictor', 'color': '#e74c3c'},
        {'xy': (0.73, 0.7), 'width': 0.22, 'height': 0.2, 'text': 'Multi-Objective\nSearch\n(NSGA-II)', 'color': '#f39c12'},
        {'xy': (0.40, 0.35), 'width': 0.20, 'height': 0.2, 'text': 'Optimal\nQuantum\nCircuits', 'color': '#2ecc71'},
    ]
    
    # Draw boxes
    for box in boxes:
        rect = mpatches.FancyBboxPatch(box['xy'], box['width'], box['height'],
                                       boxstyle="round,pad=0.01", 
                                       edgecolor='black', facecolor=box['color'],
                                       linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
               box['text'], ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw arrows
    arrows = [
        {'start': (0.20, 0.8), 'end': (0.25, 0.8), 'label': ''},
        {'start': (0.43, 0.8), 'end': (0.48, 0.8), 'label': 'Gate-Qubit\nGraph'},
        {'start': (0.68, 0.8), 'end': (0.73, 0.8), 'label': 'Acc + Energy\nPredictions'},
        {'start': (0.84, 0.7), 'end': (0.60, 0.55), 'label': 'Pareto Front'},
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        if arrow['label']:
            mid_x = (arrow['start'][0] + arrow['end'][0]) / 2
            mid_y = (arrow['start'][1] + arrow['end'][1]) / 2 + 0.05
            ax.text(mid_x, mid_y, arrow['label'], ha='center', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add title and annotations
    ax.text(0.5, 0.95, 'System Overview: Energy-Efficient Quantum NAS', 
           ha='center', fontsize=16, fontweight='bold')
    
    # Add key features
    features = [
        '✓ Novel: Graph Transformer for quantum circuits',
        '✓ Energy-Aware: Multi-objective optimization',
        '✓ Validated: Real PennyLane training on MNIST'
    ]
    for i, feature in enumerate(features):
        ax.text(0.05, 0.25 - i*0.08, feature, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('figure4_system_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created figure4_system_overview.png")

# Example usage
if __name__ == "__main__":
    # Example results (replace with actual results from verification)
    example_results = {
        'Random': {'acc': 0.56, 'depth': 13},
        'GNN': {'acc': 0.58, 'depth': 11},
        'Trans_Scalar': {'acc': 0.68, 'depth': 12},
        'Ours': {'acc': 0.70, 'depth': 8}
    }
    
    create_all_figures(example_results)
