import numpy as np
import sys
sys.path.insert(0, '.')
from quantum_nas import Architecture, QuantumLayer, ClassicalLayer, HardwareSpec, QuantumSimulator

print("Creating simple architecture...")
arch = Architecture()
arch.add_quantum_layer(QuantumLayer(num_qubits=4, depth=2, rotation_gates=['RX', 'RY'], entanglement='linear'))
arch.add_classical_layer(ClassicalLayer(filters=16))
arch.interface = 'expectation'

print("Creating simulator...")
hw = HardwareSpec()
sim = QuantumSimulator(hw)

print("Creating dummy data...")
X_train = np.random.randn(20, 4)
y_train = np.random.randint(0, 2, 20)
X_test = np.random.randn(10, 4)
y_test = np.random.randint(0, 2, 10)

print("Starting evaluation (this should take ~30 seconds)...")
result = sim.evaluate(arch, X_train, y_train, X_test, y_test, n_epochs=5, quick_mode=True)

print(f"\nResult: {result}")
print(f"Accuracy: {result['accuracy']:.3f}")
print(f"Energy: {result['energy']:.3f}")
print("\nTest completed successfully!")
