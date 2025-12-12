"""
Generate figures specifically for the Predictor-Based Validation Results
Highlighting the Efficiency Win (Depth 2 vs Depth 6)
"""
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

def create_validation_figures():
    # Data from predictor_validation_output.txt
    methods = ['Random', 'Trans+Scalar', 'Grid', 'Bayesian', 'Ours']
    accuracy = [83.4, 95.6, 96.6, 98.2, 98.5]
    depth =    [6,    6,    6,    6,    2]
    
    colors = ['#95a5a6', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71']
    markers = ['o', 's', '^', 'D', '*']
    
    # --- Figure 1: The "Efficiency Gap" (Bar Chart) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Plot Accuracy (Left Axis)
    ax1 = ax
    rects1 = ax1.bar(x - width/2, accuracy, width, label='Predicted Accuracy', color='#34495e', alpha=0.9)
    ax1.set_ylabel('Predicted Accuracy (%)', fontweight='bold', color='#34495e')
    ax1.set_ylim(80, 100)
    ax1.tick_params(axis='y', labelcolor='#34495e')
    
    # Plot Depth (Right Axis)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, depth, width, label='Circuit Depth (Lower is Better)', color='#2ecc71', alpha=0.9)
    ax2.set_ylabel('Circuit Depth (Lower is Better)', fontweight='bold', color='#27ae60')
    ax2.set_ylim(0, 8)
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    
    # Labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontweight='bold')
    ax1.set_title('Comparison with Baselines: Accuracy vs. Efficiency', fontweight='bold', pad=20)
    
    # Annotations
    # Highlight Bayesian vs Ours
    # Bayesian bar
    ax1.annotate('High Accuracy,\nHigh Cost', 
                 xy=(3 + width/2, 6), xytext=(3, 7.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 ha='center', fontsize=10)
                 
    # Ours bar
    ax2.annotate('SAME Accuracy,\n66% Less Cost!', 
                 xy=(4 + width/2, 2), xytext=(4, 4.5),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                 ha='center', fontsize=11, fontweight='bold', color='red')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figure_validation_efficiency_gap.png', bbox_inches='tight')
    print("✓ Created figure_validation_efficiency_gap.png")

    # --- Figure 2: Pareto Front (The "Kill Shot") ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot points
    for i, method in enumerate(methods):
        s = 300 if method == 'Ours' else 150
        ax.scatter(depth[i], accuracy[i], c=colors[i], s=s, label=method, marker=markers[i], edgecolors='black', zorder=10)
        
        # Add labels
        offset_y = 0.5
        if method == 'Bayesian': offset_y = -1.5
        if method == 'Grid': offset_y = -1.5
        
        ax.text(depth[i], accuracy[i] + offset_y, method, ha='center', fontweight='bold', fontsize=11)

    # Draw Pareto Front area
    # Ours is at (2, 98.5)
    # Bayesian is at (6, 98.2)
    
    # Shaded region for "Dominated by Ours"
    # Rectangle from (2, 0) to (8, 98.5)
    import matplotlib.patches as patches
    rect = patches.Rectangle((2, 80), 6, 18.5, linewidth=0, edgecolor='none', facecolor='#2ecc71', alpha=0.1)
    ax.add_patch(rect)
    ax.text(5, 85, 'Dominated by Ours\n(Higher Energy, Lower/Same Acc)', ha='center', color='#27ae60', fontsize=12, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2.2, 98.5), xytext=(5.8, 98.2),
                arrowprops=dict(arrowstyle="<|-", color='red', lw=2, mutation_scale=20))
    ax.text(4, 99, '3x More Efficient!', ha='center', color='red', fontweight='bold', fontsize=12, rotation=0)

    ax.set_xlabel('Circuit Depth (Energy Cost) $\leftarrow$ Lower is Better', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Accuracy (%) $\leftarrow$ Higher is Better', fontweight='bold', fontsize=12)
    ax.set_title('Pareto Efficiency Analysis', fontweight='bold', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(1, 8)
    ax.set_ylim(80, 100)
    
    plt.tight_layout()
    plt.savefig('figure_validation_pareto.png', bbox_inches='tight')
    print("✓ Created figure_validation_pareto.png")

if __name__ == "__main__":
    create_validation_figures()
