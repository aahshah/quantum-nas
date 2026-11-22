# MO-QAS: Multi-Objective Quantum Architecture Search

Code for **"Multi-Objective Quantum Architecture Search for Noise-Resilient Circuits with Barren Plateau Mitigation"**, accepted at QNLPAI 2026 (Springer post-proceedings).

Aahan Shah, Armaan Sharma, Hana Sultana · Texas Research Association, UT Austin

## The problem

Most quantum architecture search (QAS) optimizes for accuracy on an ideal simulator. That favors deep, heavily entangled circuits, which run into two problems on real hardware:

1. **Gate noise grows with depth**, so circuits that look good in simulation degrade on NISQ devices.
2. **Barren plateaus** make gradients vanish exponentially with qubit count, so the circuit can't be trained at all. Hardware noise can cause barren plateaus on its own, even in circuits that were trainable in simulation.

## What MO-QAS does

MO-QAS treats trainability as an explicit objective instead of hoping it shows up. NSGA-II searches over three objectives at once:

| Objective | Direction | Signal |
|---|---|---|
| Accuracy | maximize | predicted task accuracy |
| Depth | minimize | estimated circuit depth |
| Trainability | maximize | −log V̂, where V̂ is the estimated gradient variance deficit |

Two design choices make this work:

- **Trainability-relevant choices are search variables.** Cost-function locality (local vs. global) and parameter initialization are searched alongside gates and entanglement topology, instead of being fixed by hand.
- **A Graph Transformer surrogate** encodes each circuit as a graph and predicts all three objectives, so the evolutionary search never calls a simulator in its inner loop.

## Results

| | Result |
|---|---|
| Surrogate rank correlation (Spearman ρ) | depth 0.961 · accuracy 0.740 · trainability 0.652 |
| Simulator accuracy across the Pareto front | 0.86 to 0.98 |
| Fidelity under 10% depolarizing noise | depth-2 circuits keep 2.55× the fidelity of depth-6 circuits |

Noise robustness was also checked under amplitude damping (T1) and phase damping (T2), across three random seeds.

## Installation

```bash
git clone https://github.com/aahshah/quantum-nas.git
cd quantum-nas
pip install -r requirements.txt
```

Requires Python 3.8+. Main dependencies: PyTorch, PennyLane, NumPy, SciPy, scikit-learn, Matplotlib.

## Reproducing the paper

Run the full pipeline: build the dataset, train the surrogate, run the three-objective NSGA-II search, validate under noise, then compute gradients and make the figures.

```bash
python run_moqas_pipeline.py
```

For a fast smoke test with a small dataset and a short search:

```bash
python run_moqas_pipeline.py --quick
```

Useful flags: `--device cuda` trains the surrogate on GPU. `--skip-noise`, `--skip-gradient` and `--skip-figures` skip individual stages (noise validation is the slowest). Pareto-front results are saved to `experiments/moqas_pareto_results.json`.

## Repository structure

```
quantum_nas.py                     Core library: circuit definition, sampler, graph builder,
                                   Graph Transformer surrogate, hardware specs
run_moqas_pipeline.py              End-to-end paper pipeline (entry point)
run_nsga2_three_objectives.py      Three-objective NSGA-II search
barren_plateau_analyzer.py         Gradient-variance / trainability estimation
multi_noise_validation_verified.py Noise validation (depolarizing, T1, T2)
figures/                           Scripts that produce the paper's figures
experiments/                       Barren plateau, gradient-scaling and Pareto/noise experiments + results
multi_noise_results_*.json         Noise validation results (seeds 42, 123, 999)
benchmark.py                       Earlier two-objective (accuracy, depth) search, kept as a baseline
dataset_1000.pkl                   1,000 simulator-labeled architectures used to train the surrogate
docs/                              Notes on the barren plateau integration and quick starts
archive/                           Earlier experiments and drafts
```

## Extending

- **New objective** (e.g. gate cost, energy): add the metric to `HardwareSpec` in `quantum_nas.py`, add an output head to `GraphTransformerPredictor`, and add it to the NSGA-II objective list.
- **New hardware backend**: add an entry to `HARDWARE_SPECS` in `quantum_nas.py`.

## Citation

```bibtex
@inproceedings{shah2026moqas,
  title     = {Multi-Objective Quantum Architecture Search for Noise-Resilient Circuits with Barren Plateau Mitigation},
  author    = {Shah, Aahan and Sharma, Armaan and Sultana, Hana},
  booktitle = {Proceedings of QNLPAI 2026},
  publisher = {Springer},
  year      = {2026}
}
```

