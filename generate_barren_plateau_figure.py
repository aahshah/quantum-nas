"""
Generate barren plateau figure from gradient_scaling_results.json.
Run from project root: python generate_barren_plateau_figure.py

If JSON doesn't exist, run experiments/gradient_scaling_standalone.py first.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    json_path = Path('experiments/gradient_scaling_results.json')
    if not json_path.exists():
        print("Run experiments/gradient_scaling_standalone.py first to create the data.")
        return

    with open(json_path) as f:
        data = json.load(f)

    qubit_range = data['qubit_range']
    results = data['results']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(qubit_range, results['Global'], 'r-o', label='Standard (Global Cost)')
    ax.semilogy(qubit_range, results['Local'], 'b-s', label='Standard (Local Cost)')
    ax.semilogy(qubit_range, results['NAS'], 'g-^', linewidth=3, label='MO-QAS')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Gradient Variance (Log Scale)')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()
    plt.savefig('barren_plateau_proof.png', dpi=300)
    print("Saved barren_plateau_proof.png")

if __name__ == "__main__":
    main()
