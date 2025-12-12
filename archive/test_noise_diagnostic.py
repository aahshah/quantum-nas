"""
Diagnostic: Check if noise is actually affecting circuit outputs
"""
import pennylane as qml
import numpy as np

# Simple test circuit
dev = qml.device("default.mixed", wires=2)

@qml.qnode(dev)
def test_circuit(noise_prob):
    # Simple circuit
    qml.RY(np.pi/4, wires=0)
    qml.RY(np.pi/4, wires=1)
    
    # Apply noise
    if noise_prob > 0:
        qml.DepolarizingChannel(noise_prob, wires=0)
        qml.DepolarizingChannel(noise_prob, wires=1)
    
    qml.CNOT(wires=[0, 1])
    
    if noise_prob > 0:
        qml.DepolarizingChannel(noise_prob, wires=0)
        qml.DepolarizingChannel(noise_prob, wires=1)
    
    return qml.expval(qml.PauliZ(0))

print("Testing if noise affects expectation values:")
for noise in [0.0, 0.02, 0.05, 0.10]:
    result = test_circuit(noise)
    print(f"  Noise {noise*100:.0f}%: Expectation = {result:.4f}, Sign = {np.sign(result)}")
