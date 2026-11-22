#!/usr/bin/env python3
"""
Minimal A->C bridge: gradient-variance for decoded MO-QAS knee architecture.

Computes the same statistic used in experiments/gradient_scaling_standalone.py
for the decoded knee architecture at n=4 (or whatever n is in moqas JSON),
using local cost <Z0> and parameter-shift gradients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pennylane as qml


ROOT = Path(__file__).resolve().parents[1]
MOQAS_JSON = ROOT / "experiments" / "moqas_pareto_results.json"
SCALING_JSON = ROOT / "experiments" / "gradient_scaling_results.json"
OUT_JSON = ROOT / "experiments" / "bridge_knee_gradient_result.json"
BRIDGE_NOISE_JSON = ROOT / "experiments" / "bridge_pareto_noise_results.json"


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


def _create_knee_qnode(
    n_qubits: int,
    depth: int,
    entanglement: str,
    rotation_gates: List[str],
    parametrization: str,
):
    dev = qml.device("default.qubit", wires=n_qubits)
    edges = _ring_edges(n_qubits) if entanglement == "circular" else _linear_edges(n_qubits)
    reupload = "data-reupload" in parametrization.lower()
    n_rot = len(rotation_gates)
    n_params = depth * n_qubits * max(1, n_rot)

    def circuit(params):
        params = params.reshape(depth, n_qubits, max(1, n_rot))
        if not reupload:
            for q in range(n_qubits):
                qml.RY(np.pi * 0.5, wires=q)
        for layer in range(depth):
            if reupload:
                for q in range(n_qubits):
                    qml.RY(np.pi * 0.5, wires=q)
            for q in range(n_qubits):
                for g_idx, gate in enumerate(rotation_gates):
                    _apply_rotation(gate, params[layer, q, g_idx], q)
            for c, t in edges:
                qml.CNOT(wires=[c, t])
        return qml.expval(qml.PauliZ(0))

    qnode = qml.QNode(circuit, dev, diff_method="parameter-shift")
    return qnode, n_params


def _compute_gradient_variance(qnode, n_params: int, n_samples: int = 40) -> float:
    gradients = []
    for _ in range(n_samples):
        params = qml.numpy.random.randn(n_params) * 0.1
        try:
            grad = qml.grad(qnode)(params)
            grad = np.array(grad).flatten()
            if len(grad) == n_params and np.all(np.isfinite(grad)) and np.any(np.abs(grad) > 1e-12):
                gradients.append(grad)
        except Exception:
            continue
    if len(gradients) < 5:
        return float(np.var([np.mean(np.abs(g)) for g in gradients])) if gradients else 1e-10
    return float(np.mean(np.var(gradients, axis=0)))


def main() -> int:
    with MOQAS_JSON.open("r", encoding="utf-8") as f:
        moqas = json.load(f)

    arch = moqas["knee_point"]["architecture"]
    n_qubits = int(arch["n_qubits"])
    depth = int(arch["depth"])
    ent = str(arch["entanglement"][0]).lower()
    rotation_gates = [str(g).upper() for g in arch["rotation_gates"][0]]
    parametrization = str(arch["parametrization"][0])
    if ent not in {"linear", "circular"}:
        raise ValueError(f"Unsupported entanglement: {ent}")
    supported = {"RX", "RY", "RZ"}
    if not rotation_gates or any(g not in supported for g in rotation_gates):
        raise ValueError(f"Unsupported rotation set: {rotation_gates}")

    qnode, n_params = _create_knee_qnode(n_qubits, depth, ent, rotation_gates, parametrization)
    var_knee = _compute_gradient_variance(qnode, n_params, n_samples=40)

    extra = None
    if BRIDGE_NOISE_JSON.exists():
        with BRIDGE_NOISE_JSON.open("r", encoding="utf-8") as f:
            bridge_noise = json.load(f)
        reps = bridge_noise.get("representatives", [])
        target = None
        for r in reps:
            if r.get("label") == "low_depth_remaining_unique":
                target = r
                break
        if target is not None:
            arch2 = target["architecture"]
            n2 = int(arch2["n_qubits"])
            d2 = int(arch2["depth"])
            ent2 = str(arch2["entanglement"][0]).lower()
            rg2 = [str(g).upper() for g in arch2["rotation_gates"][0]]
            p2 = str(arch2["parametrization"][0])
            if ent2 in {"linear", "circular"} and rg2 and all(g in {"RX", "RY", "RZ"} for g in rg2):
                qnode2, n_params2 = _create_knee_qnode(n2, d2, ent2, rg2, p2)
                var2 = _compute_gradient_variance(qnode2, n_params2, n_samples=40)
                extra = {
                    "label": "low_depth_remaining_unique",
                    "architecture": arch2,
                    "gradient_variance": var2,
                }

    nas_ref = None
    if SCALING_JSON.exists():
        with SCALING_JSON.open("r", encoding="utf-8") as f:
            scaling = json.load(f)
        n_range = scaling.get("qubit_range", [])
        nas_vals = scaling.get("results", {}).get("NAS", [])
        if n_qubits in n_range:
            idx = n_range.index(n_qubits)
            if idx < len(nas_vals):
                nas_ref = float(nas_vals[idx])

    payload = {
        "source_moqas_json": str(MOQAS_JSON.relative_to(ROOT)).replace("\\", "/"),
        "knee_architecture": arch,
        "gradient_statistic": "mean coordinate-wise variance over parameter-shift gradients",
        "n_samples": 40,
        "result": {"knee_gradient_variance": var_knee},
        "additional_representative": extra,
        "comparison": {
            "gradient_scaling_reference_json": (
                str(SCALING_JSON.relative_to(ROOT)).replace("\\", "/") if SCALING_JSON.exists() else None
            ),
            "nas_template_variance_at_same_n": nas_ref,
            "ratio_knee_over_nas_template": (var_knee / nas_ref) if nas_ref and nas_ref > 0 else None,
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUT_JSON}")
    if nas_ref is not None and nas_ref > 0:
        print(f"knee var={var_knee:.6e}  NAS-template var@same-n={nas_ref:.6e}  ratio={var_knee/nas_ref:.3f}")
    else:
        print(f"knee var={var_knee:.6e}")
    if extra is not None:
        print(f"{extra['label']} var={extra['gradient_variance']:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
