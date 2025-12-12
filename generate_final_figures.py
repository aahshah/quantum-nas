"""
Generate publication figures with error bars from multiple trials
"""
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

def create_figures_with_error_bars():
    """Create figures showing mean ± std from multiple trials"""
    
    # EXAMPLE DATA - Replace with your actual 10-trial results
    results = {
        'Random': {'acc': 55, 'acc_std': 5, 'depth': 13, 'depth_std': 2},
        'GNN+Scalar': {'acc': 58, 'acc_std': 6, 'depth': 11, 'depth_std': 2},
        'Trans+Scalar': {'acc': 68, 'acc_std': 7, 'depth': 10, 'depth_std': 2},
        'Grid': {'acc': 62, 'acc_std': 6, 'depth': 12, 'depth_std': 2},
        'Bayesian': {'acc': 64, 'acc_std': 6, 'depth': 11, 'depth_std': 2},
        'Ours': {'acc': 65, 'acc_std': 8, 'depth': 8, 'depth_std': 2}
    }
    
    # Figure 1: Main Results with Error Bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(results.keys())
    accs = [results[m]['acc'] for m in methods]
    acc_stds = [results[m]['acc_std'] for m in methods]
    depths = [results[m]['depth'] for m in methods]
    depth_stds = [results[m]['depth_std'] for m in methods]
    
    colors = ['#808080', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e74c3c']
    x = np.arange(len(methods))
    
    # Accuracy subplot
    bars1 = ax1.bar(x, accs, yerr=acc_stds, capsize=5, color=colors, 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_title('(a) Classification Accuracy\n(Mean ± Std, n=10 trials)', 
                  fontweight='bold', fontsize=13)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 100])
    
    # Add value labels
    for i, (bar, acc, std) in enumerate(zip(bars1, accs, acc_stds)):
        label = f'{acc}±{std}%'
        if i == len(accs) - 1:  # Highlight "Ours"
            label += '\n★'
            ax1.text(bar.get_x() + bar.get_width()/2., acc + std + 3,
                    label, ha='center', va='bottom', fontsize=11, 
                    fontweight='bold', color='red')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., acc + std + 2,
                    label, ha='center', va='bottom', fontsize=10)
    
    # Depth subplot
    bars2 = ax2.bar(x, depths, yerr=depth_stds, capsize=5, color=colors,
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Circuit Depth (Energy Proxy)', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_title('(b) Energy Consumption\n(Mean ± Std, n=10 trials)', 
                  fontweight='bold', fontsize=13)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, depth, std) in enumerate(zip(bars2, depths, depth_stds)):
        label = f'{depth}±{std}'
        if i == len(depths) - 1:  # Highlight "Ours"
            label += '\n★'
            ax2.text(bar.get_x() + bar.get_width()/2., depth + std + 0.5,
                    label, ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color='red')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., depth + std + 0.3,
                    label, ha='center', va='bottom', fontsize=10)
    
    # Energy savings annotation
    if depths[-1] < depths[0]:
        savings = (depths[0] - depths[-1]) / depths[0] * 100
        ax2.annotate(f'{savings:.0f}% Energy\nSavings', 
                    xy=(len(methods)-1, depths[-1]), 
                    xytext=(len(methods)-2, depths[0] * 0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                    fontsize=11, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Main Results: Accuracy and Energy Efficiency (10 Trials)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figure1_main_results_with_error_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created figure1_main_results_with_error_bars.png")
    
    # Figure 2: Scatter plot with error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (method, data) in enumerate(results.items()):
        ax.errorbar(data['depth'], data['acc'], 
                   xerr=data['depth_std'], yerr=data['acc_std'],
                   fmt='o', markersize=12 if method == 'Ours' else 10,
                   color=colors[i], label=method, capsize=5,
                   linewidth=2, markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_xlabel('Circuit Depth (Lower = More Energy-Efficient)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (Higher = Better)', fontsize=13, fontweight='bold')
    ax.set_title('Energy-Accuracy Tradeoff\n(Mean ± Std, n=10 trials)', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('figure2_tradeoff_with_error_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created figure2_tradeoff_with_error_bars.png")
    
    print("\n✅ All figures created successfully!")
    print("\nGenerated figures:")
    print("  1. figure1_main_results_with_error_bars.png")
    print("  2. figure2_tradeoff_with_error_bars.png")

if __name__ == "__main__":
    create_figures_with_error_bars()
