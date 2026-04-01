#!/usr/bin/env python3
"""
Full MO-QAS paper pipeline (Methods-aligned):
  1) Generate / load architecture dataset (simulator labels)
  2) Train Graph Transformer surrogate (quantum_nas.QuantumNASSystem)
  3) NSGA-II on surrogate: max accuracy, max (-depth_est), max (-log V_hat), V_hat=1-trainability
  4) Save experiments/moqas_pareto_results.json
  5) Run noise validation + gradient JSON + figure scripts (optional)

Usage (from repo root):
  python run_moqas_pipeline.py --quick
  python run_moqas_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _run_py(script: Path, *, cwd: Path) -> int:
    print(f"  -> python {script.relative_to(cwd)}")
    return subprocess.call([sys.executable, str(script)], cwd=str(cwd))


def main():
    ap = argparse.ArgumentParser(description="MO-QAS surrogate search + paper artefacts")
    ap.add_argument("--quick", action="store_true", help="tiny dataset and search")
    ap.add_argument("--device", default="cpu", help="cuda or cpu for surrogate")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--skip-gradient", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    if args.quick:
        n_samples, epochs, pop_size, generations = 48, 25, 12, 8
    else:
        n_samples, epochs, pop_size, generations = 120, 60, 20, 15

    dataset_path = ROOT / "paper_dataset.pkl"
    out_json = ROOT / "experiments" / "moqas_pareto_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    from quantum_nas import QuantumNASSystem, NSGA2Search

    print("=== [1/5] QuantumNASSystem (ibm_quantum) ===")
    system = QuantumNASSystem(hardware="ibm_quantum", device=args.device)
    system.load_data()

    print(f"=== [2/5] dataset {dataset_path} (n={n_samples}) ===")
    system.generate_dataset(n_samples=n_samples, save_path=str(dataset_path))

    print(f"=== [3/5] train_predictor epochs={epochs} ===")
    system.train_predictor(str(dataset_path), epochs=epochs, batch_size=8)

    print(f"=== [4/5] NSGA-II MO-QAS surrogate pop={pop_size} gen={generations} ===")
    searcher = NSGA2Search(
        system.predictor,
        system.graph_builder,
        system.sampler,
        system.hardware_spec,
        use_qfi=False,
        moqas_objectives=True,
    )
    best_arch, _, history = searcher.search(
        system.X_train,
        system.y_train,
        system.X_test,
        system.y_test,
        pop_size=pop_size,
        generations=generations,
    )
    export = searcher._moqas_pareto_export
    if export is None:
        print("ERROR: moqas export missing")
        return 1
    export["config"] = {
        "pop_size": pop_size,
        "generations": generations,
        "dataset_samples": n_samples,
        "predictor_epochs": epochs,
    }
    with open(out_json, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Saved {out_json}")
    legacy = ROOT / "experiments" / "nsga2_pareto_results.json"
    shutil.copy(out_json, legacy)
    print(f"Copied -> {legacy} (compat for older figure/docs)")
    print(f"Best arch total_depth={best_arch.total_depth} qubits={best_arch.total_qubits}")

    if not args.skip_noise:
        print("=== [5a] multi_noise_validation_verified.py 42 ===")
        subprocess.call([sys.executable, str(ROOT / "multi_noise_validation_verified.py"), "42"], cwd=str(ROOT))

    grad_script = ROOT / "experiments" / "gradient_scaling_standalone.py"
    if not args.skip_gradient and grad_script.exists():
        print("=== [5b] gradient_scaling_standalone.py ===")
        subprocess.call([sys.executable, str(grad_script)], cwd=str(ROOT))

    if not args.skip_figures:
        print("=== [5c] figure scripts ===")
        for name in (
            "figures/figure1_gradient_scaling.py",
            "figures/figure2_noise_sweep.py",
            "figures/figure3_fidelity_heatmap.py",
        ):
            p = ROOT / name
            if p.exists():
                code = _run_py(p, cwd=ROOT)
                if code != 0:
                    print(f"  (warning) {name} exited {code}")
            else:
                print(f"  (skip) missing {name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
