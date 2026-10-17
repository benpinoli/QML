"""Held-out and frequency task measurements."""

from __future__ import annotations

import numpy as np

from qftlab.objectives import process_fidelity, state_fidelities


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def evaluate_qft(target_unitary: np.ndarray, predicted_unitary: np.ndarray,
                 eval_states: np.ndarray, target_states: np.ndarray,
                 frequency_inputs: np.ndarray, frequency_targets: np.ndarray) -> dict[str, float | np.ndarray]:
    outputs = eval_states @ predicted_unitary.T
    state_fid = state_fidelities(target_states, outputs)
    output_tv = 0.5 * np.sum(np.abs(np.abs(outputs) ** 2 - np.abs(target_states) ** 2), axis=1)
    probs = np.abs(frequency_inputs @ predicted_unitary.T) ** 2
    predicted_bins = np.argmax(probs, axis=1)
    target_probs = np.asarray(frequency_targets, dtype=float)
    eps = np.finfo(float).eps
    cross_entropy = -np.mean(np.sum(target_probs * np.log(np.maximum(probs, eps)), axis=1))
    return {
        "process_fidelity": process_fidelity(target_unitary, predicted_unitary),
        "heldout_state_fidelity_mean": float(np.mean(state_fid)),
        "heldout_state_fidelity_min": float(np.min(state_fid)),
        "heldout_output_total_variation_mean": float(np.mean(output_tv)),
        "frequency_accuracy": float(np.mean(predicted_bins == np.argmax(target_probs, axis=1))),
        "frequency_cross_entropy": float(cross_entropy),
        "frequency_total_variation_mean": float(np.mean(0.5 * np.sum(np.abs(probs - target_probs), axis=1))),
        "frequency_probabilities": probs,
        "frequency_predictions": predicted_bins,
        "heldout_state_fidelities": state_fid,
    }
