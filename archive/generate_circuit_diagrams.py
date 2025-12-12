"""
Generate Circuit Diagrams for Paper
Visualizes the structural difference between Baseline (Depth 6) and Ours (Depth 2).
"""
import pennylane as qml
import matplotlib.pyplot as plt

def draw_circuits():
    # Define device
    dev = qml.device("default.qubit", wires=4)
    
    # --- 1. Baseline (Depth 6, Linear Entanglement) ---
    @qml.qnode(dev)
    def baseline_circuit():
        for l in range(6): # Depth 6
            # Rotations
            for i in range(4):
                qml.RX(0.1, wires=i)
                qml.RZ(0.1, wires=i)
            # Linear Entanglement
            for i in range(3):
                qml.CNOT(wires=[i, i+1])
        return qml.expval(qml.PauliZ(0))

    # --- 2. Ours (Depth 2, Full Entanglement) ---
    @qml.qnode(dev)
    def ours_circuit():
        for l in range(2): # Depth 2
            # Rotations
            for i in range(4):
                qml.RX(0.1, wires=i)
                qml.RZ(0.1, wires=i)
            # Full Entanglement
            for i in range(4):
                for j in range(i+1, 4):
                    qml.CNOT(wires=[i, j])
        return qml.expval(qml.PauliZ(0))

    # Draw Baseline
    fig, ax = qml.draw_mpl(baseline_circuit, style='pennylane')()
    plt.title("Baseline Architecture (Depth 6)\nLinear Entanglement", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('circuit_baseline_depth6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created circuit_baseline_depth6.png")

    # Draw Ours
    fig, ax = qml.draw_mpl(ours_circuit, style='pennylane')()
    plt.title("Ours: MO-QAS Architecture (Depth 2)\nFull Entanglement", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('circuit_ours_depth2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created circuit_ours_depth2.png")

if __name__ == "__main__":
    draw_circuits()
