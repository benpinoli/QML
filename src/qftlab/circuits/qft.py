"""Two-qubit QFT reference and trainable circuit with the same gate layout."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator

PARAMETER_NAMES = (
    "u1_theta",
    "u1_phi",
    "u1_lambda",
    "controlled_phase",
    "u0_theta",
    "u0_phi",
    "u0_lambda",
)
QFT_PARAMETERS = np.asarray(
    [np.pi / 2, 0.0, np.pi, np.pi / 2, np.pi / 2, 0.0, np.pi], dtype=float
)


def build_reference_qft() -> QuantumCircuit:
    """Return the positive-sign, two-qubit QFT with final bit-reversal swap.

    Qiskit wire 0 is the least-significant computational-basis bit. The gate
    order is H(1), CP(+pi/2; 1->0), H(0), SWAP(0, 1).
    """
    circuit = QuantumCircuit(2, name="QFT2")
    circuit.h(1)
    circuit.cp(np.pi / 2, 1, 0)
    circuit.h(0)
    circuit.swap(0, 1)
    return circuit


def build_qft_ansatz(parameters: Mapping[str, float] | None = None) -> QuantumCircuit:
    """Build the QFT-layout ansatz with trainable U gates and controlled phase."""
    values = ParameterVector("theta", len(PARAMETER_NAMES))
    circuit = QuantumCircuit(2, name="QFT2_ansatz")
    circuit.u(values[0], values[1], values[2], 1)
    circuit.cp(values[3], 1, 0)
    circuit.u(values[4], values[5], values[6], 0)
    circuit.swap(0, 1)
    if parameters is not None:
        bind = {value: float(parameters[name]) for value, name in zip(values, PARAMETER_NAMES)}
        circuit = circuit.assign_parameters(bind, inplace=False)
    return circuit


def qft_unitary(parameters: Mapping[str, float] | None = None) -> np.ndarray:
    """Return the 4x4 unitary; ``None`` selects the reference QFT angles."""
    if parameters is None:
        parameters = dict(zip(PARAMETER_NAMES, QFT_PARAMETERS))
    return np.asarray(Operator(build_qft_ansatz(parameters)).data, dtype=np.complex128)


def reference_unitary() -> np.ndarray:
    return np.asarray(Operator(build_reference_qft()).data, dtype=np.complex128)
