#!/usr/bin/env python3
"""
A->B bridge: evaluate multiple Pareto representatives under noise.

Reads experiments/moqas_pareto_results.json, selects three representative
decoded architecture summaries (knee, high-accuracy non-knee unique, and
lowest-depth remaining unique), then computes state fidelity on default.mixed
at p=0.10 for the three channels used in Experiment (B).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pennylane as qml


ROOT = Path(__file__).resolve().parents[1]
MOQAS_JSON = ROOT / "experiments" / "moqas_pareto_results.json"
OUT_JSON = ROOT / "experiments" / "bridge_pareto_noise_results.json"
MAX_QUBITS_FOR_MIXED = 10


def _apply_noise(noise_type: str, p: float, wires: List[int]) -> None:
    if p <= 0:
        return
    for w in wires:
        if noise_type == "depolarizing":
            qml.DepolarizingChannel(p, wires=w)
        elif noise_type == "amplitude_damping":
            qml.AmplitudeDamping(p, wires=w)
        elif noise_type == "phase_damping":
            qml.PhaseDamping(p, wires=w)
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")


def _ring_edges(n_qubits: int) -> List[List[int]]:
    if n_qubits <= 1:
        return []
    edges = [[i, i + 1] for i in range(n_qubits - 1)]
    edges.append([n_qubits - 1, 0])
    return edges


def _linear_edges(n_qubits: int) -> List[List[int]]:
    if n_qubits <= 1:
        return []
    return [[i, i + 1] for i in range(n_qubits - 1)]


def _apply_rotation(gate: str, angle: float, wire: int) -> None:
    g = gate.upper()
    if g == "RX":
        qml.RX(angle, wires=wire)
    elif g == "RY":
        qml.RY(angle, wires=wire)
    elif g == "RZ":
        qml.RZ(angle, wires=wire)
    else:
        raise ValueError(f"Unsupported rotation gate: {gate}")


def _arch_state(
    n_qubits: int,
    depth: int,
    entanglement: str,
    rotation_gates: List[str],
    parametrization: str,
    noise_type: str,
    p: float,
):
    dev = qml.device("default.mixed", wires=n_qubits)
    inputs = np.full(n_qubits, 0.5)
    angle = np.pi / 4
    edges = _ring_edges(n_qubits) if entanglement == "circular" else _linear_edges(n_qubits)
    reupload = "data-reupload" in parametrization.lower()

    @qml.qnode(dev)
    def circuit():
        if not reupload:
            for i in range(n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
        for layer in range(depth):
            if reupload:
                for i in range(n_qubits):
                    qml.RY(inputs[i] * np.pi, wires=i)
            _apply_noise(noise_type, p, list(range(n_qubits)))
            for i in range(n_qubits):
                for g in rotation_gates:
                    _apply_rotation(g, angle, i)
            for c, t in edges:
                qml.CNOT(wires=[c, t])
                _apply_noise(noise_type, p, [c, t])
        return qml.density_matrix(wires=list(range(n_qubits)))

    return circuit()


def _baseline_depth6_linear_state(n_qubits: int, noise_type: str, p: float):
    dev = qml.device("default.mixed", wires=n_qubits)
    inputs = np.full(n_qubits, 0.5)
    params = np.full((6, n_qubits), np.pi / 4)
    edges = _linear_edges(n_qubits)

    @qml.qnode(dev)
    def circuit():
        for i in range(n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)
        for layer in range(6):
            _apply_noise(noise_type, p, list(range(n_qubits)))
            for i in range(n_qubits):
                qml.RX(params[layer, i], wires=i)
            for c, t in edges:
                qml.CNOT(wires=[c, t])
                _apply_noise(noise_type, p, [c, t])
        return qml.density_matrix(wires=list(range(n_qubits)))

    return circuit()


def _arch_signature(arch: Dict) -> str:
    return json.dumps(
        {
            "depth": int(arch["depth"]),
            "n_qubits": int(arch["n_qubits"]),
            "entanglement": list(arch["entanglement"]),
            "rotation_gates": list(arch["rotation_gates"]),
            "layer_depths": list(arch["layer_depths"]),
            "parametrization": list(arch["parametrization"]),
        },
        sort_keys=True,
    )


def _validate_arch(arch: Dict) -> None:
    rots = [str(g).upper() for g in arch["rotation_gates"][0]]
    param = str(arch["parametrization"][0])
    ent = str(arch["entanglement"][0]).lower()
    if not rots:
        raise ValueError("Empty rotation_gates list")
    supported = {"RX", "RY", "RZ"}
    if any(g not in supported for g in rots):
        raise ValueError(f"Unexpected rotation_gates: {rots}")
    if not param:
        raise ValueError("Empty parametrization")
    if ent not in {"linear", "circular"}:
        raise ValueError(f"Unsupported entanglement: {ent}")


def _select_representatives(data: Dict) -> List[Dict]:
    pareto_all = data["pareto_front"]
    pareto = [p for p in pareto_all if int(p["architecture"]["n_qubits"]) <= MAX_QUBITS_FOR_MIXED]
    if not pareto:
        pareto = pareto_all
    knee_arch = data["knee_point"]["architecture"]
    knee_sig = _arch_signature(knee_arch)
    knee_n = int(knee_arch["n_qubits"])

    by_sig: Dict[str, Dict] = {}
    for p in pareto:
        arch = p["architecture"]
        sig = _arch_signature(arch)
        if sig not in by_sig or float(p["accuracy"]) > float(by_sig[sig]["accuracy"]):
            by_sig[sig] = p

    reps: List[Dict] = []
    if knee_n <= MAX_QUBITS_FOR_MIXED:
        reps.append(
            {
                "label": "knee",
                "score": {
                    "accuracy": float(data["knee_point"]["accuracy"]),
                    "depth": float(data["knee_point"]["depth"]),
                    "neg_log_grad_var": float(data["knee_point"]["neg_log_grad_var"]),
                },
                "architecture": knee_arch,
                "signature": knee_sig,
            }
        )
    else:
        # If knee is too large for practical mixed-state evaluation, anchor on best
        # accuracy architecture within the feasible subset.
        best_acc = max(pareto, key=lambda x: float(x["accuracy"]))
        reps.append(
            {
                "label": "best_accuracy_feasible",
                "score": {
                    "accuracy": float(best_acc["accuracy"]),
                    "depth": float(best_acc["depth"]),
                    "neg_log_grad_var": float(best_acc["neg_log_grad_var"]),
                },
                "architecture": best_acc["architecture"],
                "signature": _arch_signature(best_acc["architecture"]),
            }
        )

    others = [v for k, v in by_sig.items() if k != knee_sig]
    if others:
        best_acc = max(others, key=lambda x: float(x["accuracy"]))
        reps.append(
            {
                "label": "high_accuracy_non_knee_unique",
                "score": {
                    "accuracy": float(best_acc["accuracy"]),
                    "depth": float(best_acc["depth"]),
                    "neg_log_grad_var": float(best_acc["neg_log_grad_var"]),
                },
                "architecture": best_acc["architecture"],
                "signature": _arch_signature(best_acc["architecture"]),
            }
        )

    used = {r["signature"] for r in reps}
    remaining = [v for v in by_sig.values() if _arch_signature(v["architecture"]) not in used]
    if remaining:
        low_depth = min(remaining, key=lambda x: float(x["depth"]))
        reps.append(
            {
                "label": "low_depth_remaining_unique",
                "score": {
                    "accuracy": float(low_depth["accuracy"]),
                    "depth": float(low_depth["depth"]),
                    "neg_log_grad_var": float(low_depth["neg_log_grad_var"]),
                },
                "architecture": low_depth["architecture"],
                "signature": _arch_signature(low_depth["architecture"]),
            }
        )

    if len(reps) < 3:
        # Fall back to best remaining Pareto points (can share decoded summary).
        existing = {(r["score"]["accuracy"], r["score"]["depth"], r["signature"]) for r in reps}
        extra_candidates = sorted(
            pareto,
            key=lambda x: (float(x["accuracy"]), -float(x["depth"])),
            reverse=True,
        )
        for c in extra_candidates:
            sig = _arch_signature(c["architecture"])
            marker = (float(c["accuracy"]), float(c["depth"]), sig)
            if marker in existing:
                continue
            reps.append(
                {
                    "label": "pareto_additional",
                    "score": {
                        "accuracy": float(c["accuracy"]),
                        "depth": float(c["depth"]),
                        "neg_log_grad_var": float(c["neg_log_grad_var"]),
                    },
                    "architecture": c["architecture"],
                    "signature": sig,
                }
            )
            existing.add(marker)
            if len(reps) >= 3:
                break

    return reps[:3]


def main() -> int:
    with MOQAS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    p = 0.10
    channels = ["depolarizing", "amplitude_damping", "phase_damping"]
    reps = _select_representatives(data)
    rep_results = []
    for rep in reps:
        arch = rep["architecture"]
        _validate_arch(arch)
        n_qubits = int(arch["n_qubits"])
        depth = int(arch["depth"])
        ent = str(arch["entanglement"][0]).lower()
        rotation_gates = [str(g).upper() for g in arch["rotation_gates"][0]]
        parametrization = str(arch["parametrization"][0])

        channel_results: Dict[str, Dict[str, float]] = {}
        for ch in channels:
            ideal_arch = _arch_state(
                n_qubits, depth, ent, rotation_gates, parametrization, ch, 0.0
            )
            noisy_arch = _arch_state(
                n_qubits, depth, ent, rotation_gates, parametrization, ch, p
            )
            fid_arch = float(qml.math.fidelity(ideal_arch, noisy_arch))

            ideal_base = _baseline_depth6_linear_state(n_qubits, ch, 0.0)
            noisy_base = _baseline_depth6_linear_state(n_qubits, ch, p)
            fid_base = float(qml.math.fidelity(ideal_base, noisy_base))

            channel_results[ch] = {
                "architecture_fidelity": fid_arch,
                "baseline_depth6_linear_fidelity": fid_base,
                "ratio_over_baseline": (fid_arch / fid_base) if fid_base > 0 else float("nan"),
            }

        rep_results.append(
            {
                "label": rep["label"],
                "score": rep["score"],
                "architecture": arch,
                "channels": channel_results,
            }
        )

    payload = {
        "source_moqas_json": str(MOQAS_JSON.relative_to(ROOT)).replace("\\", "/"),
        "source_moqas_config": data.get("config", {}),
        "noise_probability": p,
        "backend": "default.mixed",
        "representatives": rep_results,
        "notes": [
            "Representative architectures are decoded directly from moqas_pareto_results.json.",
            "Depth-6 baseline uses the linear template from multi_noise_validation_verified.py.",
            "Selection policy: knee + high-accuracy non-knee unique + lowest-depth remaining unique.",
            f"Mixed-state bridge is capped at n <= {MAX_QUBITS_FOR_MIXED} for tractable simulation time.",
        ],
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUT_JSON}")
    for rep in rep_results:
        arch = rep["architecture"]
        print(
            f"\n[{rep['label']}] n={arch['n_qubits']} depth={arch['depth']} "
            f"ent={arch['entanglement'][0]}"
        )
        for ch in channels:
            r = rep["channels"][ch]
            print(
                f"{ch:18s}  arch={r['architecture_fidelity']:.4f}  "
                f"baseline={r['baseline_depth6_linear_fidelity']:.4f}  "
                f"ratio={r['ratio_over_baseline']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
