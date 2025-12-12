# Multi-Objective Quantum Architecture Search for Noise-Resilient Circuits
**Target Venue**: ICML/NeurIPS Workshop on Quantum Machine Learning

## Abstract
Quantum Machine Learning (QML) on Noisy Intermediate-Scale Quantum (NISQ) devices is severely limited by circuit depth due to decoherence and gate errors. Existing Quantum Architecture Search (QAS) methods often prioritize accuracy on ideal simulators, leading to deep circuits that fail in practice. We propose a **Multi-Objective Quantum Architecture Search (MO-QAS)** framework that explicitly optimizes for both accuracy and circuit efficiency using a Graph Transformer-based predictor. Our approach automatically discovers shallow, fully entangled architectures (Depth 2) that are statistically indistinguishable from deeper baselines (Depth 6) on ideal simulators but demonstrate superior robustness on noisy hardware. We show that our discovered circuits retain **1.9x higher signal fidelity** under realistic noise models (Depolarizing, $T_1$, $T_2$) compared to standard single-objective baselines. This establishes multi-objective optimization as a critical principle for discovering practical NISQ algorithms.

---

## 1. Introduction
### 1.1 The NISQ Bottleneck
- Quantum hardware is noisy; coherence times ($T_1, T_2$) are short.
- Deep circuits accumulate exponential noise, rendering them useless despite theoretical expressivity.
- **Problem**: How to find the "sweet spot" between expressivity and noise resilience?

### 1.2 Limitations of Existing QAS
- Most QAS methods optimize a single objective (Accuracy).
- Result: They find deep, complex circuits that work perfectly in simulation but fail on hardware.
- Noise-aware search is computationally expensive (requires noisy simulation for every candidate).

### 1.3 Our Contribution
- **MO-QAS Framework**: Treats Depth and Accuracy as competing objectives.
- **Graph Transformer Predictor**: Efficiently predicts circuit performance (Spearman $\rho=0.687$) to guide search.
- **Noise Resilience**: We demonstrate that optimizing for efficiency *implicitly* optimizes for noise resilience, without needing expensive noisy simulations during search.

---

## 2. Methodology
### 2.1 Search Space
- **Qubits**: 4 (scalable).
- **Operations**: Parameterized rotations ($RX, RY, RZ$) and Entanglement ($CNOT$).
- **Encoding**: Graph-based representation of quantum circuits.

### 2.2 Predictor Architecture
- **Graph Transformer**: Uses attention mechanisms to capture long-range dependencies in quantum circuits.
- **Comparison**: Outperforms standard GNNs (which struggled with directionality, $\rho=-0.265$).

### 2.3 Search Strategy
- **Multi-Objective Optimization**: Navigates the Pareto front of Accuracy vs. Depth.
- **Algorithm**: Validated with **NSGA-II (Evolutionary Search)**.

---

## 3. Experimental Setup
### 3.1 Simulation Environment
- **Framework**: PennyLane (`default.mixed`).
- **Noise Models**:
  1. **Depolarizing**: General gate errors.
  2. **Amplitude Damping ($T_1$)**: Energy relaxation.
  3. **Phase Damping ($T_2$)**: Decoherence.

### 3.2 Baselines
- **Random Search**: Monte Carlo sampling.
- **Standard GNN Predictor**: Baseline graph neural network.

---

## 4. Results
### 4.1 Predictor Performance
- **Graph Transformer**: Achieved Spearman correlation of **0.687** on unseen architectures.
- **Ablation**: Significantly outperformed GNN baseline (-0.265), proving the necessity of attention mechanisms for quantum circuits.

### 4.2 Architecture Discovery
- **Depth Reduction**: MO-QAS identified Depth-2 circuits with Full Entanglement.
- **Expressivity**: These shallow circuits achieved comparable ideal accuracy (~90%) to Depth-6 baselines found by single-objective search.

### 4.3 Noise Resilience (Key Result)
Under 10% noise probability, our Depth-2 circuits demonstrated superior signal retention:
- **Depolarizing Noise**: **1.9x** higher fidelity ($0.160$ vs $0.083$).
- **Amplitude Damping ($T_1$)**: **1.8x** higher fidelity ($0.453$ vs $0.251$).
- **Phase Damping ($T_2$)**: **1.8x** higher fidelity ($0.453$ vs $0.251$).

*Figure 1: Pareto front showing the trade-off between Accuracy and Depth.*
*Figure 2: Signal fidelity decay curves for Depth 2 vs Depth 6 across three noise models.*

---

## 5. Discussion & Conclusion
- **The "Free Lunch"**: By penalizing depth, we gain noise resilience "for free" without expensive noise-aware training.
- **Structural Insight**: Full entanglement allows shallow circuits to be expressive enough for tasks while minimizing gate count.
- **Future Work**: Scaling to larger qubit counts and validation on IBM Quantum hardware.

---

## Appendix
- **A. Hyperparameters**: Details of Transformer training (20 epochs, 1000 samples).
- **B. Circuit Diagrams**: Visual comparison of Linear (Depth 6) vs Full (Depth 2) entanglement.
