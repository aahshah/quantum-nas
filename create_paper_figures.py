"""
Create publication-quality figures for NeurIPS paper
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from verification results
methods = ['Random', 'GNN', 'Ours']
accuracies = [0.56, 0.58, 0.70]
depths = [13, 11, 8]
energy_relative = [100, 85, 62]  # Relative to Random baseline

# Figure 1: Dual Comparison (Accuracy + Energy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Accuracy
colors = ['#808080', '#FFA500', '#FF0000']
bars1 = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.set_title('Classification Accuracy', fontsize=16, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Subplot 2: Circuit Depth (Energy Proxy)
bars2 = ax2.bar(methods, depths, color=colors, edgecolor='black', linewidth=2)
ax2.set_ylabel('Circuit Depth (Energy Proxy)', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 15])
ax2.set_title('Energy Consumption', fontsize=16, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, depth in zip(bars2, depths):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{depth}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add energy savings annotation
ax2.annotate('38% Energy\nSavings', xy=(2, 8), xytext=(1.5, 12),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('figure1_dual_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Created figure1_dual_comparison.png")

# Figure 2: Accuracy vs Energy Tradeoff (Scatter)
fig, ax = plt.subplots(figsize=(8, 6))

# Plot points
markers = ['o', 'd', '*']
sizes = [150, 150, 300]
for i, (method, acc, depth) in enumerate(zip(methods, accuracies, depths)):
    ax.scatter(depth, acc, s=sizes[i], c=colors[i], marker=markers[i], 
              edgecolors='black', linewidth=2, label=method, zorder=10)

# Add labels
for method, acc, depth in zip(methods, accuracies, depths):
    offset_x = 0.3 if method != 'Ours' else -0.5
    offset_y = 0.02
    ax.annotate(f'{method}\n({acc:.0%}, d={depth})', 
               xy=(depth, acc), xytext=(depth + offset_x, acc + offset_y),
               fontsize=10, ha='center', bbox=dict(boxstyle='round', 
               facecolor='white', edgecolor='black', alpha=0.8))

# Highlight Pareto dominance
ax.annotate('Pareto Optimal:\nHigh Acc + Low Energy', 
           xy=(8, 0.70), xytext=(10, 0.65),
           arrowprops=dict(arrowstyle='->', lw=2, color='red'),
           fontsize=12, fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.set_xlabel('Circuit Depth (Lower = More Energy-Efficient)', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (Higher = Better)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy vs Energy Tradeoff', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([6, 15])
ax.set_ylim([0.5, 0.75])

plt.tight_layout()
plt.savefig('figure2_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Created figure2_tradeoff.png")

# Figure 3: Energy Savings Breakdown
fig, ax = plt.subplots(figsize=(8, 6))

# Stacked bar showing energy breakdown
baseline_energy = 100
savings = [0, 15, 38]  # Percentage savings

bars = ax.barh(methods, energy_relative, color=colors, edgecolor='black', linewidth=2)

# Add percentage labels
for i, (bar, energy, saving) in enumerate(zip(bars, energy_relative, savings)):
    width = bar.get_width()
    label = f'{energy}%' if saving == 0 else f'{energy}% (-{saving}%)'
    ax.text(width + 2, bar.get_y() + bar.get_height()/2., label,
           ha='left', va='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Relative Energy Consumption (%)', fontsize=14, fontweight='bold')
ax.set_title('Energy Efficiency Comparison', fontsize=16, fontweight='bold')
ax.set_xlim([0, 120])
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add reference line at 100%
ax.axvline(x=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.text(100, -0.5, 'Baseline', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_energy_savings.png', dpi=300, bbox_inches='tight')
print("✓ Created figure3_energy_savings.png")

print("\n✅ All figures created successfully!")
print("\nFigures for your paper:")
print("1. figure1_dual_comparison.png - Side-by-side accuracy + energy bars")
print("2. figure2_tradeoff.png - Scatter plot showing Pareto dominance")
print("3. figure3_energy_savings.png - Energy savings breakdown")
