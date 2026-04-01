"""
Run NSGA-II with three objectives (simulator-in-the-loop):
  f1 = task accuracy (maximize)
  f2 = lower circuit depth (maximize -depth)
  f3 = higher gradient variance (maximize)

Uses QuantumCircuitSimulator directly — no Graph Transformer surrogate.

For Methods-aligned MO-QAS (surrogate + NSGA-II with V_hat from trainability), run:
  python run_moqas_pipeline.py
which writes experiments/moqas_pareto_results.json.

Outputs Pareto front and knee-point solution.
"""
import json
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Tuple

from quantum_nas import (
    HardwareSpec, HARDWARE_SPECS, ArchitectureSampler,
    QuantumCircuitSimulator, HybridArchitecture
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Search config
POP_SIZE = 20
GENERATIONS = 15
N_GRADIENT_SAMPLES = 20  # Fewer for speed (barren plateau is expensive)


def evaluate_arch(arch: HybridArchitecture, sim: QuantumCircuitSimulator,
                  X_train, y_train, X_test, y_test) -> Dict:
    """Evaluate architecture; returns accuracy, depth, gradient_variance."""
    result = sim.evaluate(arch, X_train, y_train, X_test, y_test,
                          n_epochs=5, quick_mode=True, use_cache=True)
    depth = result['circuit_depth']
    accuracy = result['accuracy']
    grad_var = result.get('gradient_variance', 1e-8)
    # Use trainability as proxy if gradient_variance not available
    if grad_var <= 0 or grad_var == 1e-8:
        grad_var = max(1e-10, result.get('trainability', 0.5) * 1e-4)
    return {
        'accuracy': accuracy,
        'depth': depth,
        'gradient_variance': grad_var,
        'arch': arch
    }


def dominates(p: Dict, q: Dict) -> bool:
    """p dominates q: p is better or equal on all objectives, strictly better on at least one."""
    better_acc = p['accuracy'] >= q['accuracy']
    better_depth = p['depth'] <= q['depth']  # lower depth is better
    better_gv = p['gradient_variance'] >= q['gradient_variance']
    strict = (p['accuracy'] > q['accuracy'] or
              p['depth'] < q['depth'] or
              p['gradient_variance'] > q['gradient_variance'])
    return better_acc and better_depth and better_gv and strict


def fast_non_dominated_sort(scores: List[Dict]) -> List[List[int]]:
    """NSGA-II non-dominated sorting."""
    n = len(scores)
    fronts = [[]]
    S = [[] for _ in range(n)]
    n_dominated = [0] * n

    for p in range(n):
        for q in range(n):
            if dominates(scores[p], scores[q]):
                S[p].append(q)
            elif dominates(scores[q], scores[p]):
                n_dominated[p] += 1
        if n_dominated[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dominated[q] -= 1
                if n_dominated[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts


def crowding_distance(scores: List[Dict], front: List[int]) -> None:
    """Compute crowding distance for front members."""
    for i in front:
        scores[i]['crowding_dist'] = 0.0
    for obj in ['accuracy', 'depth', 'gradient_variance']:
        front_sorted = sorted(front, key=lambda i: scores[i][obj])
        obj_range = scores[front_sorted[-1]][obj] - scores[front_sorted[0]][obj]
        if obj_range == 0:
            continue
        scores[front_sorted[0]]['crowding_dist'] = float('inf')
        scores[front_sorted[-1]]['crowding_dist'] = float('inf')
        for j in range(1, len(front_sorted) - 1):
            idx = front_sorted[j]
            dist = (scores[front_sorted[j + 1]][obj] - scores[front_sorted[j - 1]][obj]) / obj_range
            scores[idx]['crowding_dist'] += dist


def tournament_select(scores: List[Dict], fronts: List[List[int]], exclude: int = -1) -> int:
    """Binary tournament selection."""
    candidates = [i for i in range(len(scores)) if i != exclude]
    a, b = random.sample(candidates, min(2, len(candidates)))
    rank_a = next((r for r, f in enumerate(fronts) if a in f), 999)
    rank_b = next((r for r, f in enumerate(fronts) if b in f), 999)
    if rank_a < rank_b:
        return a
    if rank_b < rank_a:
        return b
    return a if scores[a]['crowding_dist'] >= scores[b]['crowding_dist'] else b


def find_knee_point(scores: List[Dict], front: List[int]) -> int:
    """Find knee point: solution with best trade-off (max normalized distance from worst)."""
    if len(front) <= 1:
        return front[0]
    sub = [scores[i] for i in front]
    acc_vals = [s['accuracy'] for s in sub]
    dep_vals = [s['depth'] for s in sub]
    gv_vals = [s['gradient_variance'] for s in sub]
    a_min, a_max = min(acc_vals), max(acc_vals)
    d_min, d_max = min(dep_vals), max(dep_vals)
    g_min, g_max = min(gv_vals), max(gv_vals)
    a_range = a_max - a_min or 1
    d_range = d_max - d_min or 1
    g_range = g_max - g_min or 1e-10
    best_idx = 0
    best_score = -float('inf')
    for j, i in enumerate(front):
        s = scores[i]
        # Normalize: accuracy and grad_var higher is better; depth lower is better
        na = (s['accuracy'] - a_min) / a_range
        nd = (d_max - s['depth']) / d_range  # invert depth
        ng = (s['gradient_variance'] - g_min) / g_range
        # Knee: maximize minimum normalized distance (balanced)
        knee_score = min(na, nd, ng)
        if knee_score > best_score:
            best_score = knee_score
            best_idx = j
    return front[best_idx]


def arch_to_dict(arch: HybridArchitecture) -> Dict:
    """Serialize architecture for JSON."""
    d = arch.to_dict()
    return {
        'depth': arch.total_depth,
        'n_qubits': arch.total_qubits,
        'entanglement': [ql['entanglement'] for ql in d['quantum_layers']],
        'rotation_gates': [ql['rotation_gates'] for ql in d['quantum_layers']],
        'layer_depths': [ql['depth'] for ql in d['quantum_layers']],
        'parametrization': [ql['parametrization'] for ql in d['quantum_layers']],
    }


def main():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("NSGA-II: 3 Objectives (Accuracy, -Depth, -log Gradient Variance)")
    print("=" * 70)

    hardware = HARDWARE_SPECS['ibm_quantum']
    sampler = ArchitectureSampler(hardware)

    # Enable barren plateau analysis for gradient variance
    try:
        from barren_plateau_analyzer import BarrenPlateauAnalyzer
        bp_analyzer = BarrenPlateauAnalyzer(n_samples=N_GRADIENT_SAMPLES)
        sim = QuantumCircuitSimulator(hardware, analyze_barren_plateau=True, bp_analyzer=bp_analyzer)
        print("Using BarrenPlateauAnalyzer for gradient variance")
    except ImportError:
        sim = QuantumCircuitSimulator(hardware, analyze_barren_plateau=False)
        print("BarrenPlateauAnalyzer not found; using trainability as proxy for gradient variance")

    digits = load_digits()
    X = digits.data / 16.0
    y = (digits.target > 4).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    population = [sampler.sample() for _ in range(POP_SIZE)]
    history = []

    for gen in range(GENERATIONS):
        print(f"\n--- Generation {gen + 1}/{GENERATIONS} ---")
        scores = []
        for i, arch in enumerate(population):
            ev = evaluate_arch(arch, sim, X_train, y_train, X_test, y_test)
            ev['crowding_dist'] = 0.0
            scores.append(ev)
            if (i + 1) % 5 == 0:
                print(f"  Evaluated {i + 1}/{len(population)}")

        fronts = fast_non_dominated_sort(scores)
        for f in fronts:
            crowding_distance(scores, f)

        # Log front
        front0 = fronts[0] if fronts else []
        if front0:
            best_acc = max(scores[i]['accuracy'] for i in front0)
            min_dep = min(scores[i]['depth'] for i in front0)
            max_gv = max(scores[i]['gradient_variance'] for i in front0)
            print(f"  Pareto front: |F0|={len(front0)}, best_acc={best_acc:.3f}, "
                  f"min_depth={min_dep}, max_grad_var={max_gv:.2e}")

        history.append({
            'generation': gen,
            'pareto_front': [
                {
                    'accuracy': scores[i]['accuracy'],
                    'depth': scores[i]['depth'],
                    'gradient_variance': scores[i]['gradient_variance'],
                    'arch_idx': i
                }
                for i in front0
            ]
        })

        if gen == GENERATIONS - 1:
            break

        # Selection + mutation
        offspring = []
        while len(offspring) < POP_SIZE:
            p1 = tournament_select(scores, fronts)
            p2 = tournament_select(scores, fronts, exclude=p1)
            parent = population[p1] if random.random() < 0.5 else population[p2]
            child = sampler.mutate(parent, 0.3)
            offspring.append(child)
        population = offspring

    # Final Pareto front
    final_scores = []
    for arch in population:
        ev = evaluate_arch(arch, sim, X_train, y_train, X_test, y_test)
        ev['arch'] = arch
        ev['crowding_dist'] = 0.0
        final_scores.append(ev)
    fronts = fast_non_dominated_sort(final_scores)
    crowding_distance(final_scores, fronts[0])
    pareto_indices = fronts[0]
    pareto_solutions = [final_scores[i] for i in pareto_indices]

    knee_idx = find_knee_point(final_scores, pareto_indices)
    knee = final_scores[knee_idx]

    # Output
    out = {
        'config': {'pop_size': POP_SIZE, 'generations': GENERATIONS, 'seed': SEED},
        'knee_point': {
            'accuracy': knee['accuracy'],
            'depth': knee['depth'],
            'gradient_variance': knee['gradient_variance'],
            'architecture': arch_to_dict(knee['arch'])
        },
        'pareto_front': [
            {
                'accuracy': s['accuracy'],
                'depth': s['depth'],
                'gradient_variance': s['gradient_variance'],
                'architecture': arch_to_dict(s['arch'])
            }
            for s in pareto_solutions
        ]
    }

    out_path = Path('experiments/nsga2_pareto_results.json')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("KNEE-POINT ARCHITECTURE")
    print("=" * 70)
    k = knee
    print(f"  Accuracy:           {k['accuracy']:.4f}")
    print(f"  Depth:              {k['depth']}")
    print(f"  Gradient Variance:  {k['gradient_variance']:.6e}")
    print(f"  Entanglement:       {arch_to_dict(k['arch'])['entanglement']}")
    print(f"  Rotation Gates:     {arch_to_dict(k['arch'])['rotation_gates']}")
    print(f"  Layer Depths:       {arch_to_dict(k['arch'])['layer_depths']}")
    print(f"  Parametrization:    {arch_to_dict(k['arch'])['parametrization']}")

    print("\n" + "=" * 70)
    print("PARETO FRONT (top 5)")
    print("=" * 70)
    for i, s in enumerate(pareto_solutions[:5]):
        a = arch_to_dict(s['arch'])
        print(f"  [{i+1}] acc={s['accuracy']:.3f}, depth={s['depth']}, grad_var={s['gradient_variance']:.2e} | "
              f"ent={a['entanglement']}, gates={a['rotation_gates']}")


if __name__ == '__main__':
    main()
