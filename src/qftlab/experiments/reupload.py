"""Two-qubit QFT-layout model with repeated angle-encoding uploads."""

from __future__ import annotations

import numpy as np


def _u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Qiskit's U(theta, phi, lambda) matrix."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.asarray([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
    ], dtype=np.complex128)


def predict_reuploaded(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ideal Z1 expectations for a (uploads, 7) parameter array.

    Every layer applies RY(2*pi*x) to q0, U(theta,phi,lambda) to q1,
    controlled phase from q1 to q0, and another U to q0. The final bit-reversal
    SWAP and Z1 readout match the existing two-qubit QFT-layout experiment.
    For one upload, this is exactly the current ansatz and data encoding.
    """
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 7 or parameters.shape[0] < 1:
        raise ValueError("parameters must have shape (uploads>=1, 7)")
    x = np.asarray(x, dtype=float).reshape(-1)
    # Qiskit little-endian basis order: [|q1 q0>] = [00, 01, 10, 11].
    state = np.zeros((x.size, 4), dtype=np.complex128)
    state[:, 0] = 1.0
    z1 = np.asarray([1.0, 1.0, -1.0, -1.0])
    for layer in parameters:
        c, s = np.cos(np.pi * x), np.sin(np.pi * x)
        state_2q = state.reshape(-1, 2, 2)
        old0, old1 = state_2q[:, :, 0].copy(), state_2q[:, :, 1].copy()
        state_2q[:, :, 0] = c[:, None] * old0 - s[:, None] * old1
        state_2q[:, :, 1] = s[:, None] * old0 + c[:, None] * old1

        u1 = _u_matrix(*layer[:3])
        state_2q = np.einsum("ab,nbc->nac", u1, state.reshape(-1, 2, 2))
        state = state_2q.reshape(-1, 4)

        state[:, 3] *= np.exp(1j * layer[3])

        u0 = _u_matrix(*layer[4:])
        state_2q = np.einsum("nqb,ab->nqa", state.reshape(-1, 2, 2), u0)
        state = state_2q.reshape(-1, 4)

    state = state.reshape(-1, 2, 2).transpose(0, 2, 1).reshape(-1, 4)
    return np.real(np.sum(np.abs(state) ** 2 * z1[None, :], axis=1))


def spectrum_amplitudes(values: np.ndarray) -> np.ndarray:
    """Return one-sided Fourier amplitudes for a uniformly sampled period."""
    values = np.asarray(values, dtype=float)
    coefficients = np.fft.rfft(values) / len(values)
    amplitudes = 2 * np.abs(coefficients)
    amplitudes[0] = np.abs(coefficients[0])
    if len(values) % 2 == 0:
        amplitudes[-1] = np.abs(coefficients[-1])
    return amplitudes
