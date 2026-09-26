"""
Quantum State Preparation and Representation Module for QubitLab.

Provides pure state vectors, density matrices, Bloch sphere vector coordinates,
and quantum circuit state preparation primitives.
"""

from __future__ import annotations
import cmath
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
from qiskit import QuantumCircuit


@dataclass(frozen=True)
class QuantumStateInfo:
    """Encapsulates properties of a single-qubit quantum state."""
    name: str
    label_latex: str
    theta: float
    phi: float
    alpha: complex
    beta: complex
    bloch_vector: Tuple[float, float, float]
    purity: float
    is_pure: bool
    prob_0: float
    prob_1: float
    statevector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None


# Canonical single-qubit basis states
CANONICAL_STATES: Dict[str, Tuple[float, float, str]] = {
    "|0>": (0.0, 0.0, r"|0\rangle"),
    "|1>": (np.pi, 0.0, r"|1\rangle"),
    "|+>": (np.pi / 2, 0.0, r"|+\rangle"),
    "|->": (np.pi / 2, np.pi, r"|-\rangle"),
    "|i>": (np.pi / 2, np.pi / 2, r"|i\rangle"),
    "|-i>": (np.pi / 2, 3 * np.pi / 2, r"|-i\rangle"),
}


def bloch_to_amplitudes(theta: float, phi: float) -> Tuple[complex, complex]:
    """
    Computes state amplitudes alpha and beta from Bloch sphere angles theta and phi:
    |psi> = cos(theta / 2)|0> + e^(i * phi) * sin(theta / 2)|1>
    """
    alpha = complex(np.cos(theta / 2.0), 0.0)
    beta = np.sin(theta / 2.0) * cmath.exp(1j * phi)
    return alpha, beta


def amplitudes_to_bloch(alpha: complex, beta: complex) -> Tuple[float, float]:
    """
    Converts normalized amplitudes alpha and beta to Bloch angles (theta, phi).
    Accounts for global phase by fixing alpha to be real and non-negative.
    """
    norm = np.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    if norm < 1e-12:
        raise ValueError("State vector cannot be null.")
    a = alpha / norm
    b = beta / norm

    # Remove global phase so alpha is real and >= 0
    phase_alpha = cmath.phase(a)
    a = a * cmath.exp(-1j * phase_alpha)
    b = b * cmath.exp(-1j * phase_alpha)

    # Real part of a is cos(theta / 2)
    r_a = np.clip(a.real, -1.0, 1.0)
    theta = 2.0 * np.arccos(r_a)

    # Phase of b is phi
    if abs(np.sin(theta / 2.0)) < 1e-8:
        phi = 0.0
    else:
        phi = float(cmath.phase(b)) % (2.0 * np.pi)

    return float(theta), float(phi)


def density_matrix_from_bloch(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Constructs a 2x2 density matrix from Bloch vector components (rx, ry, rz):
    rho = 0.5 * (I + rx*X + ry*Y + rz*Z)
    """
    return 0.5 * np.array([
        [1.0 + rz, rx - 1j * ry],
        [rx + 1j * ry, 1.0 - rz]
    ], dtype=complex)


def bloch_from_density_matrix(rho: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Extracts Bloch vector components (rx, ry, rz) and purity Tr(rho^2)
    from a 2x2 density matrix rho.
    """
    rx = float(2.0 * np.real(rho[0, 1]))
    ry = float(2.0 * np.imag(rho[1, 0]))
    rz = float(np.real(rho[0, 0] - rho[1, 1]))
    purity = float(np.real(np.trace(rho @ rho)))
    return rx, ry, rz, purity


def get_quantum_state(
    state_choice: str = "|0>",
    theta: Optional[float] = None,
    phi: Optional[float] = None
) -> QuantumStateInfo:
    """
    Creates a QuantumStateInfo instance based on a canonical name or explicit Bloch angles.
    """
    if state_choice in CANONICAL_STATES and theta is None and phi is None:
        th, ph, latex_str = CANONICAL_STATES[state_choice]
        name = state_choice
    elif theta is not None and phi is not None:
        th = float(theta) % (2.0 * np.pi)
        if th > np.pi:
            # Map theta to [0, pi]
            th = 2.0 * np.pi - th
            phi = (phi + np.pi) % (2.0 * np.pi)
        ph = float(phi) % (2.0 * np.pi)
        name = f"Custom (θ={th:.2f}, φ={ph:.2f})"
        latex_str = rf"|\psi(\theta={th:.2f}, \phi={ph:.2f})\rangle"
    else:
        th, ph, latex_str = CANONICAL_STATES["|0>"]
        name = "|0>"

    alpha, beta = bloch_to_amplitudes(th, ph)
    statevec = np.array([alpha, beta], dtype=complex)
    rho = np.outer(statevec, np.conj(statevec))

    rx = float(np.sin(th) * np.cos(ph))
    ry = float(np.sin(th) * np.sin(ph))
    rz = float(np.cos(th))

    p0 = float(abs(alpha) ** 2)
    p1 = float(abs(beta) ** 2)

    return QuantumStateInfo(
        name=name,
        label_latex=latex_str,
        theta=th,
        phi=ph,
        alpha=alpha,
        beta=beta,
        bloch_vector=(rx, ry, rz),
        purity=1.0,
        is_pure=True,
        prob_0=p0,
        prob_1=p1,
        statevector=statevec,
        density_matrix=rho,
    )


def prepare_state_on_circuit(
    qc: QuantumCircuit,
    qubit_index: int = 0,
    state_info: Optional[QuantumStateInfo] = None
) -> QuantumCircuit:
    """
    Appends standard single-qubit gates or a unitary rotation to prepare
    the requested quantum state from |0>.
    """
    if state_info is None or state_info.name == "|0>":
        return qc

    name = state_info.name
    if name == "|1>":
        qc.x(qubit_index)
    elif name == "|+>":
        qc.h(qubit_index)
    elif name == "|->":
        qc.x(qubit_index)
        qc.h(qubit_index)
    elif name == "|i>":
        qc.h(qubit_index)
        qc.s(qubit_index)
    elif name == "|-i>":
        qc.h(qubit_index)
        qc.sdg(qubit_index)
    else:
        # Custom state: use standard rotation U(theta, phi, 0)
        # U(theta, phi, 0)|0> = cos(theta/2)|0> + e^(i*phi)sin(theta/2)|1>
        qc.u(state_info.theta, state_info.phi, 0.0, qubit_index)

    return qc
