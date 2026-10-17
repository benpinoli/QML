"""Classical optimization of the ideal unitary-level QFT recovery loss."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from qftlab.circuits.qft import PARAMETER_NAMES, build_qft_ansatz
from qftlab.objectives import process_fidelity
from qiskit.quantum_info import Operator


def parameters_to_unitary(values: np.ndarray) -> np.ndarray:
    mapping = dict(zip(PARAMETER_NAMES, np.asarray(values, dtype=float)))
    return np.asarray(Operator(build_qft_ansatz(mapping)).data, dtype=np.complex128)


def optimize_unitary(target: np.ndarray, initial: np.ndarray, *, max_iterations: int,
                     ftol: float, gtol: float, max_line_search_steps: int = 30,
                     callback: Callable[[int, np.ndarray, float], None] | None = None):
    def objective(values: np.ndarray) -> float:
        return 1.0 - process_fidelity(target, parameters_to_unitary(values))

    step = 0

    def on_iteration(values: np.ndarray) -> None:
        nonlocal step
        step += 1
        if callback:
            callback(step, values.copy(), float(objective(values)))

    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="L-BFGS-B",
        callback=on_iteration,
        options={"maxiter": int(max_iterations), "ftol": float(ftol), "gtol": float(gtol),
                 "maxls": int(max_line_search_steps), "maxfun": max(1000, int(max_iterations) * 50)},
    )
    return result
