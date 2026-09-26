"""
Quantum Noise Modeling Module for QubitLab.

Provides Qiskit Aer noise models (Bit-Flip, Phase-Flip, Depolarizing)
and analytical density matrix channel transformations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
from src.states import bloch_from_density_matrix, density_matrix_from_bloch


# Pauli matrices
PAULI_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass(frozen=True)
class NoiseChannelInfo:
    """Metadata and analytical transformation details for a noise channel."""
    name: str
    symbol: str
    probability: float
    description: str
    kraus_operators: List[np.ndarray]
    bloch_shrinkage: Tuple[float, float, float]


def create_noise_model(
    noise_type: str = "bit_flip",
    probability: float = 0.0,
    target_instructions: Optional[List[str]] = None
) -> NoiseModel:
    """
    Creates a Qiskit Aer NoiseModel for the specified error type and probability.
    
    Parameters:
        noise_type: 'bit_flip', 'phase_flip', or 'depolarizing'.
        probability: Physical error probability p in [0.0, 0.5].
        target_instructions: Gates to attach noise to (defaults to ['id']).
    """
    p = float(np.clip(probability, 0.0, 1.0))
    noise_model = NoiseModel()
    
    if p <= 1e-12:
        return noise_model

    targets = target_instructions or ["id"]

    if noise_type == "bit_flip":
        # X error with probability p, I with 1 - p
        error = pauli_error([("X", p), ("I", 1.0 - p)])
        noise_model.add_all_qubit_quantum_error(error, targets)
    elif noise_type == "phase_flip":
        # Z error with probability p, I with 1 - p
        error = pauli_error([("Z", p), ("I", 1.0 - p)])
        noise_model.add_all_qubit_quantum_error(error, targets)
    elif noise_type == "depolarizing":
        # Depolarizing error with parameter p
        # In Qiskit, depolarizing_error(p, 1) means:
        # (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)
        error = depolarizing_error(p, 1)
        noise_model.add_all_qubit_quantum_error(error, targets)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}. Choose 'bit_flip', 'phase_flip', or 'depolarizing'.")

    return noise_model


def apply_analytical_noise_channel(
    rho_in: np.ndarray,
    noise_type: str = "bit_flip",
    p: float = 0.0
) -> Tuple[np.ndarray, Tuple[float, float, float], float]:
    """
    Analytically applies a quantum noise channel to a single-qubit density matrix rho_in.
    
    Returns:
        rho_out: Post-channel density matrix.
        bloch_out: (rx, ry, rz) Bloch vector components.
        purity_out: Tr(rho_out^2).
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 1e-12:
        rx, ry, rz, purity = bloch_from_density_matrix(rho_in)
        return rho_in.copy(), (rx, ry, rz), purity

    if noise_type == "bit_flip":
        # E(rho) = (1 - p)*rho + p*(X rho X)
        rho_out = (1.0 - p) * rho_in + p * (PAULI_X @ rho_in @ PAULI_X)
    elif noise_type == "phase_flip":
        # E(rho) = (1 - p)*rho + p*(Z rho Z)
        rho_out = (1.0 - p) * rho_in + p * (PAULI_Z @ rho_in @ PAULI_Z)
    elif noise_type == "depolarizing":
        # E(rho) = (1 - p)*rho + (p/3)*(X rho X + Y rho Y + Z rho Z)
        rho_out = (1.0 - p) * rho_in + (p / 3.0) * (
            PAULI_X @ rho_in @ PAULI_X +
            PAULI_Y @ rho_in @ PAULI_Y +
            PAULI_Z @ rho_in @ PAULI_Z
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    rx, ry, rz, purity = bloch_from_density_matrix(rho_out)
    return rho_out, (rx, ry, rz), purity


def get_noise_channel_explanation(noise_type: str, p: float) -> str:
    """Provides a concise, scientifically accurate description of the chosen noise channel."""
    percent = p * 100.0
    if noise_type == "bit_flip":
        return (
            f"**Bit-Flip Channel (Pauli X)** with probability $p = {percent:.1f}\\%$:\n"
            f"Acts like a classical bit-flip error. With probability $p$, an $X$ gate is applied, "
            f"interchanging $|0\\rangle \\leftrightarrow |1\\rangle$. On the Bloch sphere, the $x$-axis is invariant, "
            f"while the $y$ and $z$ coordinates are contracted by $(1 - 2p)$."
        )
    elif noise_type == "phase_flip":
        return (
            f"**Phase-Flip Channel (Pauli Z)** with probability $p = {percent:.1f}\\%$:\n"
            f"A uniquely quantum error that alters relative superposition phase without changing computational basis probabilities. "
            f"With probability $p$, a $Z$ gate is applied ($|0\\rangle \\to |0\\rangle, |1\\rangle \\to -|1\\rangle$). "
            f"On the Bloch sphere, the $z$-axis is invariant, while equatorial components ($x, y$) contract by $(1 - 2p)$."
        )
    elif noise_type == "depolarizing":
        return (
            f"**Depolarizing Channel** with probability $p = {percent:.1f}\\%$:\n"
            f"Symmetric decoherence: with probability $1 - p$ the state is untouched, and with probability $p$ "
            f"it undergoes an equal chance of $X$, $Y$, or $Z$ errors. "
            f"On the Bloch sphere, the vector isotropically contracts toward the center (the maximally mixed state $I/2$)."
        )
    return ""
