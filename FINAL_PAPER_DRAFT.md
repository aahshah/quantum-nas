# Multi-Objective Quantum Architecture Search for Noise-Resilient Circuits

**Abstract**
Quantum Machine Learning (QML) on Noisy Intermediate-Scale Quantum (NISQ) devices is severely limited by circuit depth due to decoherence and gate errors. Existing Quantum Architecture Search (QAS) methods often prioritize accuracy on ideal simulators, leading to deep circuits that fail in practice. We propose a Multi-Objective Quantum Architecture Search (MO-QAS) framework that explicitly optimizes for accuracy, depth, and trainability. Search returns shallow **knee-point** candidates (e.g. depth 2, RX+RZ, **data re-uploading**); our **noise benchmarks** compare that template against a deeper baseline using **linear** CNOT ladders only (depth 6 vs depth 2). In PennyLane mixed-state simulations, this shallow template retains **higher state fidelity** than the depth-6 baseline under depolarizing, amplitude-damping ($T_1$), and phase-damping ($T_2$) channels; on a **4-qubit** instance at **10%** noise probability, improvements are approximately **2.55×** (depolarizing), **1.59×** (amplitude damping), and **2.48×** (phase damping) relative to the baseline fidelities reported in our experiments. This establishes multi-objective optimization as a practical principle for discovering noise-aware NISQ circuit templates.

---

## 1. Introduction

**The NISQ Bottleneck**
Quantum computing is currently in the Noisy Intermediate-Scale Quantum (NISQ) era, where hardware is severely limited by decoherence and gate errors. Coherence times ($T_1, T_2$) are short, typically in the range of 10-100 $\mu s$. While variational quantum algorithms (VQAs) offer a promising path to useful applications, their performance is often bottlenecked by circuit depth. Deeper circuits theoretically offer greater expressivity but, in practice, accumulate exponential noise, rendering their output indistinguishable from random noise. Finding the optimal trade-off between expressivity and noise resilience is a critical, open challenge.

**The Gap in Existing Methods**
Existing Quantum Architecture Search (QAS) approaches typically focus on maximizing accuracy on ideal simulators, treating circuit complexity as a secondary constraint or a tie-breaker. Few methods explicitly optimize for noise resilience, and those that do often require expensive noisy simulations during the search process, making them computationally intractable for large search spaces. There is a lack of frameworks that can efficiently identify robust architectures without the heavy overhead of full noise simulation. Furthermore, most studies validate on a single noise model, failing to capture the complexity of real hardware errors. To our knowledge, this is the first work to empirically validate that multi-objective optimization automatically discovers noise-resilient circuits across multiple realistic noise channels, demonstrating a generalizable principle for NISQ algorithm design.

**Our Contribution**
We present a Multi-Objective Quantum Architecture Search (MO-QAS) framework that treats accuracy, depth, and gradient variance (trainability proxy) as competing objectives. NSGA-II on our encoding yields a **knee-point** shallow circuit used to **motivate** the noise study; we then evaluate **explicit** depth-6 vs depth-2 templates in PennyLane (same linear connectivity family, no all-to-all entanglement in the reported circuits). We validate on a **4-qubit** benchmark at fixed noise levels, showing higher state fidelity for the depth-2 template across depolarizing, amplitude damping, and phase damping. We complement this with a separate **gradient-variance scaling** experiment using fixed circuit families (not NSGA-II chromosomes) to illustrate barren-plateau-style scaling.

---

## 2. Related work

**Noise-aware and learned circuit design**
Mitigating NISQ errors has been studied at the pulse, compilation, and circuit levels. *QuantumNAS* (Wang et al., HPCA 2022 / arXiv:2107.10845) is a representative **noise-adaptive** framework: it couples a *SuperCircuit* with iterative subcircuit sampling, evolutionary **co-search** of circuit structure and qubit mapping under device noise models, and pruning or finetuning, with strong empirical results on both QML and VQE across real backends. That work illustrates the scale and detail readers expect in a top-tier architecture paper—**broad related work**, a **factored method** (several subsystems with figures), **heavy experiments** (many tasks, baselines, and ablations), and **clear positioning** vs differentiable NAS and hardware-aware design. Our goal is **not** to match QuantumNAS’s empirical surface area (dozens of benchmarks and machines), but to keep a **similar architectural narrative depth**: motivation → related work → formal method → experiments → analysis, at **conference length** rather than a short workshop note.

**Quantum architecture search (QAS)**
QAS and differentiable NAS (e.g. DQAS-style approaches) overwhelmingly optimize a **single** proxy—typically validation accuracy—sometimes with noise-free or lightly calibrated simulation. Few QAS papers treat **depth, trainability, and accuracy** as explicit **trade-offs** on a Pareto front, or separate **search-time surrogate cost** from **post-hoc** noisy validation. We use a **graph-based surrogate** and **NSGA-II** so inner-loop fitness is cheap, then validate selected templates under **multiple** channels in PennyLane—closer in *spirit* to noise-aware design than to one-shot differentiable search, but **smaller in scope** than QuantumNAS’s full co-search and hardware loop.

**Multi-objective evolutionary optimization**
NSGA-II and related algorithms are standard for **multi-objective** engineering design. Applying them to QAS with a learned surrogate connects MOO to **NISQ-relevant** objectives (accuracy, depth, trainability proxy). Positioning relative to single-objective QAS and to QuantumNAS-style noise adaptation clarifies our **scientific contribution**: a **principled Pareto / knee** view plus **fidelity** evidence, rather than the largest cross-device benchmark suite.

**Barren plateaus and trainability**
Trainability and gradient variance are widely used **surrogates** for optimization hardness in variational circuits. We do not claim new theory; our gradient-scaling experiment is a **control** motivating the third MO-QAS objective, analogous to how a systems paper includes a microbenchmark to justify one design choice.

**Length and scope (authorial target)**
A QuantumNAS-length submission implies roughly **9–12 double-column pages** of main text (excluding references), **8–15 figures/tables**, and **≥1–2 pages** of related work and positioning. Your current draft is **structurally** on track but **shorter**: to approach that scale, expand **§2** with 8–12 additional citations (QAS surveys, noise-aware compilation, hardware mapping, barren-plateau reviews), add **Method** subsections (graph encoding, transformer architecture, loss functions, NSGA formulation, complexity), add **Experiments** (Pareto plot figure, surrogate error vs simulator, optional ablation on removing one objective, wall-clock or query cost), and move long JSON paths to an **Appendix** or supplementary repo README.

---

## 3. Background

**Circuit Depth & Decoherence**
Quantum circuits are sequences of gates applied to qubits. Each gate takes a finite amount of time, and during this time, qubits interact with their environment, leading to decoherence. The probability of a successful computation decreases exponentially with circuit depth ($D$) and error rate ($\epsilon$): $P_{success} \propto (1-\epsilon)^D$. Therefore, minimizing depth is not just an optimization goal but a physical necessity.

**NSGA-II Optimization**
Non-dominated Sorting Genetic Algorithm II (NSGA-II) is a popular multi-objective evolutionary algorithm. It maintains a population of candidate architectures and evolves them to find the Pareto frontier—the set of optimal trade-offs between conflicting objectives (here: accuracy, depth, and gradient variance). We use NSGA-II to explore the architecture space defined by our sampler.

**Noise Models**
We consider three standard noise channels:
1.  **Depolarizing Channel**: Models general symmetric errors where a qubit is replaced by the maximally mixed state with probability $p$.
2.  **Amplitude Damping ($T_1$)**: Models energy relaxation where a qubit decays from $|1\rangle$ to $|0\rangle$.
3.  **Phase Damping ($T_2$)**: Models the loss of quantum information without energy loss (pure decoherence).

---

## 4. Method

**Search Framework**
We seek architectures that score well on **three** objectives: maximize **classification accuracy**, minimize **circuit depth**, and maximize **trainability** (used as a proxy for gradient signal via $\hat{V}=1-\texttt{trainability}$ in the MO-QAS objective $-\log \hat{V}$). We use **NSGA-II** over an `ArchitectureSampler` encoding. Architectures are tallied once with the project simulator to build a training set; a **Graph Transformer** surrogate (`GraphTransformerPredictor`) is then trained to predict accuracy, depth, and trainability so NSGA-II fitness evaluations use **cheap forward passes only**. Full-scale search is `run_moqas_pipeline.py` (dataset size 120, **20** individuals, **15** generations, **60** surrogate epochs); the **saved** `experiments/moqas_pareto_results.json` in this draft was produced with the repository’s **`--quick`** benchmark (**48** architectures, **12** individuals, **8** generations, **25** epochs) for fast iteration. The exported **knee** has logical depth **2**, RX+RZ, data re-uploading, and **4** qubits in that run (connectivity in the chromosome may be `circular` or `linear` depending on the individual); the **noise study below** uses **linear** CNOT ladders for both templates so depth—not topology mismatch—is the controlled variable.

**Noise benchmarks (explicit templates; not full entanglement)**
To isolate **structural** depth vs noise, we compare two fixed templates evaluated in PennyLane `default.mixed`. Parameters are held fixed ($\pi/4$ rotations as implemented) so differences reflect depth and gate pattern, not training:
1.  **Baseline (Depth 6)**: Encode inputs with $RY(\pi x_i)$ per qubit; six layers of **RX** rotations followed by a **linear** CNOT ladder; noise inserted as in code (after one-qubit ops and on CNOT wires).
2.  **Knee-style (Depth 2)**: **Data re-uploading**—each of two layers repeats the $RY$ encoding, applies **RX** and **RZ** per qubit, then a **linear** CNOT ladder, with the same noise placement policy.

Both use **linear** connectivity only; we do **not** claim all-to-all entanglement for these circuits.

**Evaluation Metric: Signal Fidelity**
We measure performance using **Quantum State Fidelity**, defined as:
$$ F(\rho_{ideal}, \rho_{noisy}) = \left( \text{Tr} \sqrt{\sqrt{\rho_{ideal}} \rho_{noisy} \sqrt{\rho_{ideal}}} \right)^2 $$
Fidelity ($F \in [0, 1]$) directly measures how close the noisy quantum state $\rho_{noisy}$ is to the ideal state $\rho_{ideal}$. Unlike task-specific accuracy, fidelity is a fundamental property of the circuit's robustness, independent of the specific problem being solved.

---

## 5. Experiments

*Canonical LaTeX Experiments (embedded PDFs, code-aligned):* `latex/experiments_section_moqas_paper.tex`. *From repo-root* `main.tex`*:* `\input{latex/experiments_section_moqas_paper}`. *Shim:* `latex/experiments_section_arxiv.tex` *only* `\input`s that file. *Preamble:* `\usepackage{amsmath,booktabs,graphicx,hyperref}`*; set* `\renewcommand{\FigPath}{figures/}` *or* `\graphicspath{{figures/}}` *if paths break. Figure labels:* `fig:gradient_scaling`*,* `fig:noise_sweep`*,* `fig:fidelity_heatmap` *— define in that snippet. Add* `\cite{deb2002fast}` *in the Method section if you cite NSGA-II.*

We organize empirical work into three parts: **(i)** MO-QAS with a surrogate-trained NSGA-II pass; **(ii)** controlled **state-fidelity** comparisons under multiple noise channels; **(iii)** a **gradient-variance scaling** study on fixed circuit families. Scripts write JSON under `experiments/` and the repo root so numbers and figures are reproducible without manual transcription.

**Claims vs. evidence (read this once).** §5.1 is **surrogate-only** search: every fitness eval is a neural forward pass, not a noisy-device simulation. §5.2–5.3 are **separate** PennyLane studies on **fixed** circuits; we do **not** re-evaluate every Pareto genome under every channel. The noise study uses **linear** depth-6 vs. depth-2 templates on purpose; MO-QAS JSON may still say `circular` for some chromosomes. That split is intentional so you do not over-claim “hardware NAS” while still showing **cheap MOO + physics-grounded** fidelity.

### Overview of the three figures

The experiments rely on **three figures** (PDF/PNG under `figures/`). Previously the text cited them only by file name or “Figure 1–2–style”; here is what each one **shows**:

1. **Figure 1** — outputs `figures/gradient_scaling.pdf` / `.png`; script `figures/figure1_gradient_scaling.py`.  
   **Axes:** number of qubits \(n \in \{4,6,8,10,12\}\) on the horizontal axis; **mean gradient variance** \(\mathrm{Var}[\nabla_\theta C]\) on the **log** vertical axis.  
   **Series:** three *fixed* circuit families from `experiments/gradient_scaling_standalone.py`: global cost, depth 10, all-to-all CNOTs; local cost, depth 10, all-to-all; local cost, depth 6, **linear** CNOTs (“MO-QAS-style”). Points are measurements; dashed lines are log–log fits \(\mathrm{Var}\propto n^{\alpha}\).  
   **Takeaway:** shallow linear templates keep larger gradient variance than the deep global-cost curve over this range of \(n\).

2. **Figure 2** — outputs `figures/noise_sweep.pdf` / `.png`; script `figures/figure2_noise_sweep.py`.  
   **Layout:** **three panels** — depolarizing, amplitude damping (\(T_1\)), phase damping (\(T_2\)).  
   **Axes:** noise probability (\%) horizontally; **state fidelity** \(F\in[0,1]\) vertically.  
   **Curves:** depth-6 baseline vs. depth-2 knee-style template, from the **4-qubit** block of `multi_noise_results_seed42.json`.  
   **Takeaway:** the depth-2 curve stays above the depth-6 curve as noise increases (depolarizing panel may annotate the gain at 10%).

3. **Figure 3** — outputs `figures/fidelity_heatmap.pdf` / `.png`; script `figures/figure3_fidelity_heatmap.py`.  
   **Content:** the **same** JSON as Figure 2, shown as **heatmaps**: rows = depth-6 vs. depth-2, columns = noise levels 0%, 2%, 5%, 10%, with one block per channel.  
   **Takeaway:** compact view of the full sweep behind Table 1 and Figure 2.

§5.4 lists the script → data map; §5.5 lists copy-paste commands and limitations.

### 5.1 MO-QAS: dataset, surrogate, and NSGA-II

**Procedure.** We sample diverse architectures from `ArchitectureSimulator` / `ArchitectureSampler`, evaluate each on the digits classification task with the project’s quantum simulator, and save tuples to `paper_dataset.pkl`. A **Graph Transformer** regressor is trained to predict **accuracy**, **effective depth** (head output used as a depth surrogate), and **trainability**; NSGA-II then evolves a population **using only the surrogate** (no simulator in the inner loop). Objectives are: **maximize** accuracy; **minimize** predicted depth (implemented as maximizing the negated depth signal used in code); **maximize** $-\log \hat{V}$ with $\hat{V}=1-\texttt{trainability}+\varepsilon$. Non-dominated sorting and crowding distance follow standard NSGA-II; we export the **Pareto front** and a **knee** selection for discussion.

**Hyperparameters (saved artifact).** The file `experiments/moqas_pareto_results.json` (mirrored to `experiments/nsga2_pareto_results.json` for older scripts) was generated by `python run_moqas_pipeline.py --quick` with: **48** dataset circuits, batch size **8**, **25** surrogate epochs, NSGA-II population **12**, **8** generations. Regenerating with `python run_moqas_pipeline.py` (no flag) uses **120** circuits, **60** epochs, population **20**, **15** generations.

**Knee selection (matches code).** On the final non-dominated front, min–max normalize **accuracy** and **\(-\log\hat{V}\)** within the front, and **invert depth** so lower predicted depth maps to a larger normalized value. The **knee** is the front member that **maximizes the minimum** of those three normalized scores (`_find_knee_moqas` in `quantum_nas.py`)—a balanced compromise, not a hand-picked point.

**Outcome (quick run).** The exported knee attains surrogate-reported **accuracy $\approx 0.906$**, **predicted depth $\approx 11.1$** (scaled head units in the log file—not identical to logical layer count **2** in the decoded architecture), **$\hat{V}\approx 0.338$**, and decoded structure **logical depth 2**, **4 qubits**, **RX+RZ**, **data re-uploading**. The same JSON lists **7** non-dominated points on the surrogate objectives; several also use logical depth **2** with 4–7 qubits and mixed `linear` / `circular` entanglement labels from the sampler. **These labels describe the search chromosome;** the fidelity experiments in §5.2 deliberately fix **linear** ladders for both depth-6 and depth-2 templates so conclusions isolate **depth and re-uploading**, not entanglement-pattern differences.

**Surrogate caveat.** The Graph Transformer is trained on dataset labels from the project simulator (not on the mixed-state noise models in §5.2). Pareto/knee numbers are **design-time** under predicted accuracy / depth / trainability; §5.2 tests **hand-specified** templates under noise.

### 5.2 Noise robustness: state fidelity vs channel

**Experimental setup.** Fidelity simulations use **PennyLane** `default.mixed` (density matrices). We sweep noise probability $p\in\{0,0.02,0.05,0.10\}$ for **depolarizing**, **amplitude damping** ($T_1$), and **phase damping** ($T_2$), following the circuit templates in §4. Circuit **parameters are fixed** in code (not variationally optimized), so contrasts reflect structure under identical noise insertion policy. The driver `multi_noise_validation_verified.py` writes `multi_noise_results_seed42.json`; Figures **2** and **3** above are produced by `figures/figure2_noise_sweep.py` and `figures/figure3_fidelity_heatmap.py`, reading the **4-qubit** block (5-qubit rows are retained for comparison).

**Metric and seeds.** Fidelity is the standard mixed-state fidelity $F(\rho_{\mathrm{ideal}},\rho_{\mathrm{noisy}})\in[0,1]$ (same expression as in §4). The reported table uses **seed 42**; `python multi_noise_validation_verified.py verify` aggregates seeds **42, 123, 456** into `multi_noise_results_3seeds.json` for variance at $p=10\%$.

**Table 0: Experimental configuration (fidelity study)**

| Aspect | Value |
|--------|-------|
| Backend | PennyLane `default.mixed` |
| Qubits (main table / figures) | **4** (5-qubit rows also in JSON) |
| Noise levels | 0%, 2%, 5%, **10%** |
| Channels | Depolarizing, amplitude damping, phase damping |
| Seed (main table) | **42** |
| Multi-seed aggregation | `python multi_noise_validation_verified.py verify` → `multi_noise_results_3seeds.json` (seeds **42, 123, 456**) |
| Parameters | **Fixed** as implemented |

**Results.** At **10%** noise on **4 qubits**, the depth- **2** template achieves **higher** state fidelity than the depth- **6** baseline for **all three** channels.

**Table 1: State fidelity at 10% noise (4 qubits, seed 42)**  
*Source: `multi_noise_results_seed42.json` (last index per channel in the `"4"` block).*

| Noise model | Depth 6 (baseline) | Depth 2 (knee-style) | Ratio (D2 / D6) |
| :--- | :--- | :--- | :--- |
| **Depolarizing** | 0.0833 | 0.2121 | **2.55×** |
| **Amplitude damping ($T_1$)** | 0.2514 | 0.3986 | **1.59×** |
| **Phase damping ($T_2$)** | 0.2514 | 0.6223 | **2.48×** |

**Interpretation.** As in the **Overview of the three figures**, the noise-sweep plots fidelity vs.\ $p$ in three panels; the heatmap encodes the same grid as colored cells. The depth-6 baseline **degrades faster** as $p$ increases. **Gain ratios are channel-dependent** (e.g. amplitude damping shows a more modest multiplier here than depolarizing or phase damping).

### 5.3 Gradient variance scaling (control study)

**Goal.** Independently of the evolving NSGA-II population, we quantify how **mean gradient variance** scales with **qubit count** for three **fixed** families in `experiments/gradient_scaling_standalone.py`: (**Global**) global Pauli-string cost, **depth 10**, **all-to-all** CNOTs; (**Local**) single-qubit cost, **depth 10**, **all-to-all** CNOTs; (**NAS**) single-qubit cost, **depth 6**, **linear** CNOTs (“shallow ladder” regime). Gradients use PennyLane **parameter-shift** on `default.qubit`; variances are averaged over random initializations. Output `experiments/gradient_scaling_results.json` stores per-$n$ variances for $n\in\{4,6,8,10,12\}$ and least-squares **log–log** fits $\mathrm{Var}\propto n^{\alpha}$.

**Protocol detail.** For each $(n,\text{family})$, the script uses parameter-shift gradients on `default.qubit`, **40** random Gaussian inits (scale 0.1, subject to filtering in code), then averages gradient variance. Exponents $\alpha$ come from linear regression on $(\log n,\log \mathrm{Var})$.

**Fitted exponents $\alpha$ (current JSON):** **Global** $\approx -4.09$, **Local** $\approx -3.59$, **NAS** $\approx -1.56$. More negative $\alpha$ indicates faster variance decay with $n$ (stronger barren-plateau-style scaling in this benchmark). **Figure 1** (see overview) is built by `figures/figure1_gradient_scaling.py` → `figures/gradient_scaling.pdf`/`.png`. These curves **do not** plot NSGA-II individuals; they **sanity-check** that **local + linear + moderate depth** maintains larger gradient variance than deep global-cost circuits as $n$ grows, consistent with using **trainability / gradient proxy** as a search objective. The “NAS” curve is a **fixed** template, not an evolved architecture archive.

### 5.4 Figure and data map

| Figure (output) | Primary script | Primary data |
|-----------------|----------------|--------------|
| Gradient scaling | `figures/figure1_gradient_scaling.py` | `experiments/gradient_scaling_results.json` |
| Noise sweep | `figures/figure2_noise_sweep.py` | `multi_noise_results_seed42.json` |
| Fidelity heatmap | `figures/figure3_fidelity_heatmap.py` | `multi_noise_results_seed42.json` |
| (Search log / Pareto) | `run_moqas_pipeline.py` | `experiments/moqas_pareto_results.json` |

### 5.5 Scope, limitations, reproduction

**Scope.** Small qubit counts (noise JSON: 4–5 qubits; gradient script: up to 12), digits task for surrogate data—appropriate for a **methods + proof-of-concept**, not a QuantumNAS-scale cross-device study.

**Limitations.** (1) Simulation-only fidelity (not live calibration). (2) Surrogate objectives ≠ noisy fidelity. (3) Fixed parameters in the noise sweep favor **structural** comparison over training under noise. (4) Main tables are single-seed unless you cite `multi_noise_results_3seeds.json`.

**Commands (repo root).**  
1. Full quick pipeline (+ noise JSON + gradient JSON + three figures): `python run_moqas_pipeline.py --quick`  
2. Noise only: `python multi_noise_validation_verified.py 42`  
3. Multi-seed: `python multi_noise_validation_verified.py verify`  
4. Gradient JSON only: `python experiments/gradient_scaling_standalone.py`  
5. Figures only (after JSON exists): `python figures/figure1_gradient_scaling.py`, then `figure2_noise_sweep.py`, `figure3_fidelity_heatmap.py`  
Pipeline flags: `--skip-noise`, `--skip-gradient`, `--skip-figures` for partial runs.

---

## 6. Analysis

**Why depth matters**
Deeper circuits apply more noisy gates; under channel models that insert noise per layer, fidelity typically drops faster for the depth-6 **linear** ladder than for the depth-2 template with re-uploading and RX+RZ layers. The exact margins depend on the channel (Table 1).

**How this connects to MO-QAS**
NSGA-II returns a **Pareto** set and a **knee** candidate that favors shallow depth and higher trainability scores in our simulator. The noise study does **not** re-evaluate every Pareto member; it tests two **explicit** templates aligned with that knee (depth 2, linear, RX+RZ, re-uploading vs depth-6 baseline). Claiming that “MO-QAS always finds full entanglement” would be **incorrect** for our reported circuits: both benchmarks use **linear** CNOTs only.

**Reproducibility**
Primary numbers use **seed 42**. Running `python multi_noise_validation_verified.py verify` aggregates seeds 42, 123, 456 for 4-qubit fidelities at 10% and writes `multi_noise_results_3seeds.json` (improvement statistics for depolarizing and per-channel summaries). With **fixed** circuit parameters as in code, cross-seed variation can be **small**; the structural comparison remains depth-6 vs depth-2 under identical noise placement.

---

## 7. Discussion

**Practical implications**
Linear connectivity matches many superconducting layouts; our depth-2 template uses **fewer** entangling layers than depth 6, which can help under gate-level noise. Hardware deployment would require mapping and calibration beyond this simulator study.

**Limitations**
-   **Simulation Only**: While our noise models are realistic, we have not yet validated these results on physical hardware.
-   **Scale**: Our experiments were limited to 4 qubits. Scaling to 8-16 qubits is necessary to confirm these trends hold for larger systems.
-   **Proxy Metric**: Signal fidelity is a strong proxy, but it does not guarantee performance on a specific downstream task (e.g., VQE energy estimation).

**Future Work**
We plan to validate these architectures on IBM Quantum hardware to measure real-world performance. We also aim to extend our search space to larger qubit counts and incorporate hardware-specific connectivity constraints directly into the search objectives.

---

## 8. Conclusion

We combined **surrogate-based NSGA-II** with three objectives to obtain a **Pareto** set and **knee** shallow candidates, then validated **explicit** depth-6 vs depth-2 **linear** templates in PennyLane under three noise channels. On **4 qubits** at **10%** noise, state fidelity improved by roughly **2.55×** (depolarizing), **1.59×** (amplitude damping), and **2.48×** (phase damping) relative to the depth-6 baseline in our saved run. A separate **gradient-scaling** experiment (`gradient_scaling_standalone.py`) compares three fixed templates—global vs local cost with depth-10 **full** entanglement vs depth-6 **linear** local cost—and reports log–log scaling exponents in `gradient_scaling_results.json` (approximately **−4.1**, **−3.6**, **−1.6** for Global, Local, and NAS-style circuits in the current file). Together, this supports **noise-aware, multi-objective** circuit design as a practical NISQ principle—stated precisely and tied to reproducible scripts and JSON outputs.

---

## 9. References

1.  Wang, H., et al. "QuantumNAS: Noise-adaptive search for robust quantum circuits." *Proc. HPCA*, 2022; arXiv:2107.10845.
2.  Cerezo, M., et al. "Variational quantum algorithms." *Nature Reviews Physics* 3.9 (2021): 625-644.
3.  Deb, K., et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." *IEEE Transactions on Evolutionary Computation* 6.2 (2002): 182-197.
4.  Preskill, J. "Quantum computing in the NISQ era and beyond." *Quantum* 2 (2018): 79.
5.  Zhang, Y., et al. "Differentiable quantum architecture search." *arXiv preprint arXiv:2010.08561* (2020).

---

## Appendix

**A. Noise Model Details**
-   **Depolarizing**: $\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$
-   **Amplitude Damping**: Defined by Kraus operators $K_0 = |0\rangle\langle 0| + \sqrt{1-p}|1\rangle\langle 1|$, $K_1 = \sqrt{p}|0\rangle\langle 1|$.
-   **Phase Damping**: Defined by Kraus operators $K_0 = |0\rangle\langle 0| + \sqrt{1-p}|1\rangle\langle 1|$, $K_1 = \sqrt{p}|1\rangle\langle 1|$.

**B. Experimental Details**
-   **Fidelity / noise**: `multi_noise_validation_verified.py` → `multi_noise_results_seed*.json`, `multi_noise_results_3seeds.json` (`… verify`)
-   **MO-QAS / surrogate Pareto**: `run_moqas_pipeline.py` → `experiments/moqas_pareto_results.json` (copy also `experiments/nsga2_pareto_results.json`); legacy standalone `run_nsga2_three_objectives.py` if used separately
-   **Gradient scaling figure**: `experiments/gradient_scaling_standalone.py` → `experiments/gradient_scaling_results.json`; `figures/figure1_gradient_scaling.py` → `figures/gradient_scaling.png`
-   **Simulator**: PennyLane `default.mixed` (fidelity); `default.qubit` (gradient script)

**C. Additional Figures**
Repository outputs include `figures/noise_sweep.png`, `figures/fidelity_heatmap.png`, `figures/gradient_scaling.png`, and optional archive circuit diagrams under `archive/`.
