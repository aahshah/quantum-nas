# Multi-Objective Quantum Architecture Search (MO-QAS)

## Project Overview
This project implements a **Multi-Objective Quantum Architecture Search** framework designed to find quantum circuits that are both **expressive (accurate)** and **noise-resilient (efficient)**.

Unlike standard approaches that only optimize for accuracy on ideal simulators (often resulting in deep, unstable circuits), MO-QAS explicitly optimizes for circuit depth as a second objective. This allows it to discover "knee-point" architectures that perform significantly better on noisy NISQ hardware.

## Key Features
- **NSGA-II Search**: Evolutionary algorithm for multi-objective optimization.
- **Graph Transformer Predictor**: A neural predictor that estimates circuit accuracy from its graph representation, avoiding expensive simulations during search.
- **Multi-Noise Validation**: Robustness verification across Depolarizing, T1, and T2 noise models.

## Getting Started

### 1. Prerequisites
- Python 3.8+
- PyTorch
- PennyLane
- Matplotlib, Seaborn, Scikit-learn

### 2. Core File Structure
To understand and expand this project, focus on these files:

**The Core Engine:**
- `quantum_nas.py`: **Main Library**. Contains all core classes:
    - `HybridArchitecture`: Defines the circuit structure.
    - `ArchitectureSampler`: Generates random circuits for the search space.
    - `BipartiteGraphBuilder`: Converts circuits to graphs for the predictor.
    - `GraphTransformerPredictor`: The AI model that predicts circuit performance.
    - `HardwareSpec`: Defines hardware constraints (e.g., IBM Quantum, IonQ).

**Running Experiments:**
- `benchmark.py`: **Main Entry Point**. Runs the full NSGA-II search and compares it against random and single-objective baselines.
- `multi_noise_validation_verified.py`: **Validation Script**. Tests the discovered architectures against various noise models to prove their robustness.

**Data & Artifacts:**
- `dataset_1000.pkl`: Validated dataset of 1,000 architectures used to train the predictor.
- `quantum_cache.pkl`: Result cache to speed up repeated runs.

### 3. How to Run
**To run a new search:**
```bash
python benchmark.py
```

**To validate noise resilience:**
```bash
python multi_noise_validation_verified.py
```

## Expanding the Project
Comparison of files needed vs. artifacts.

**To add new objectives (e.g., Energy, Gate Cost):**
1.  Modify `HardwareSpec` in `quantum_nas.py` to define the new cost metric.
2.  Update `GraphTransformerPredictor` in `quantum_nas.py` to output the new metric.
3.  Update the NSGA-II loop in `benchmark.py` to optimize for this third objective.

**To add new hardware backends:**
1.  Add a new entry to the `HARDWARE_SPECS` dictionary in `quantum_nas.py`.

## Clean Up
A large number of temporary validation scripts (`verification_*.py`, `debug_*.py`) and output logs (`*.txt`) were generated during the research phase. These have been moved to the `archive/` directory to keep the workspace clean.
