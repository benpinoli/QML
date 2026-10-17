"""Phase-insensitive operation and state metrics."""

from __future__ import annotations

import numpy as np


def process_fidelity(target: np.ndarray, predicted: np.ndarray) -> float:
    """Unitary process fidelity, invariant to a physically irrelevant global phase."""
    dimension = target.shape[0]
    return float(np.abs(np.trace(target.conj().T @ predicted)) ** 2 / dimension**2)


def state_fidelities(target_states: np.ndarray, predicted_states: np.ndarray) -> np.ndarray:
    overlaps = np.sum(target_states.conj() * predicted_states, axis=-1)
    return np.abs(overlaps) ** 2
