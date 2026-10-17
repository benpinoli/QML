"""Seeded state and frequency datasets for the QFT recovery baseline."""

from __future__ import annotations

import numpy as np


def _haar_like_states(rng: np.random.Generator, count: int) -> np.ndarray:
    states = rng.normal(size=(count, 4)) + 1j * rng.normal(size=(count, 4))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return states.astype(np.complex128)


def make_qft_dataset(seed: int, train_size: int, validation_size: int, test_size: int,
                     unitary: np.ndarray) -> dict[str, np.ndarray]:
    """Return fixed random pure-state examples and ideal QFT outputs per split."""
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for split, count in (("train", train_size), ("validation", validation_size), ("test", test_size)):
        inputs = _haar_like_states(rng, count)
        result[f"{split}_inputs"] = inputs
        result[f"{split}_targets"] = inputs @ unitary.T
    return result


def make_frequency_dataset(unitary: np.ndarray) -> dict[str, np.ndarray]:
    """Create four phase-ramp states that the positive-sign QFT maps to bins 0..3."""
    indices = np.arange(4)
    frequencies = np.arange(4)
    states = np.exp(-2j * np.pi * frequencies[:, None] * indices[None, :] / 4) / 2
    outputs = states @ unitary.T
    return {
        "frequencies": frequencies.astype(np.int64),
        "inputs": states.astype(np.complex128),
        "target_states": outputs.astype(np.complex128),
        "target_probabilities": np.abs(outputs) ** 2,
    }
