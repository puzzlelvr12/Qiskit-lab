"""
Unit tests for circuit construction and quantum state preparation in QubitLab.
"""

import numpy as np
import pytest
from qiskit_aer import AerSimulator
from src.states import (
    get_quantum_state,
    bloch_to_amplitudes,
    amplitudes_to_bloch,
    bloch_from_density_matrix,
    density_matrix_from_bloch,
    prepare_state_on_circuit,
)
from src.circuits import (
    build_unprotected_circuit,
    build_encoder_circuit,
    build_syndrome_extraction_circuit,
    build_full_qec_circuit,
    draw_circuit_mpl,
    draw_circuit_text,
)


def test_canonical_states():
    """Verify standard canonical basis states properties."""
    s0 = get_quantum_state("|0>")
    assert np.isclose(s0.prob_0, 1.0)
    assert np.isclose(s0.prob_1, 0.0)
    assert np.isclose(s0.purity, 1.0)
    assert s0.bloch_vector == (0.0, 0.0, 1.0)

    s1 = get_quantum_state("|1>")
    assert np.isclose(s1.prob_0, 0.0)
    assert np.isclose(s1.prob_1, 1.0)
    assert np.isclose(s1.bloch_vector[2], -1.0)

    sp = get_quantum_state("|+>")
    assert np.isclose(sp.prob_0, 0.5)
    assert np.isclose(sp.prob_1, 0.5)
    assert np.isclose(sp.bloch_vector[0], 1.0)
    assert np.isclose(sp.bloch_vector[1], 0.0)
    assert np.isclose(sp.bloch_vector[2], 0.0)


def test_custom_state_roundtrip():
    """Verify custom state Bloch angle conversions."""
    theta = 1.23
    phi = 2.45
    alpha, beta = bloch_to_amplitudes(theta, phi)
    th_recovered, ph_recovered = amplitudes_to_bloch(alpha, beta)
    assert np.isclose(theta, th_recovered, atol=1e-5)
    assert np.isclose(phi, ph_recovered, atol=1e-5)


def test_density_matrix_and_purity():
    """Verify mixed and pure density matrix Bloch conversions."""
    # Pure state along +z
    rho_pure = density_matrix_from_bloch(0.0, 0.0, 1.0)
    rx, ry, rz, purity = bloch_from_density_matrix(rho_pure)
    assert np.isclose(purity, 1.0)
    assert np.isclose(rz, 1.0)

    # Maximally mixed state
    rho_mixed = density_matrix_from_bloch(0.0, 0.0, 0.0)
    rx, ry, rz, purity_mixed = bloch_from_density_matrix(rho_mixed)
    assert np.isclose(purity_mixed, 0.5)
    assert np.isclose(rx, 0.0) and np.isclose(ry, 0.0) and np.isclose(rz, 0.0)


def test_unprotected_circuit_execution():
    """Verify unprotected single-qubit circuit runs on AerSimulator."""
    s0 = get_quantum_state("|0>")
    qc = build_unprotected_circuit(state_info=s0)
    sim = AerSimulator()
    result = sim.run(qc, shots=500).result()
    counts = result.get_counts()
    assert counts == {"0": 500}

    s1 = get_quantum_state("|1>")
    qc1 = build_unprotected_circuit(state_info=s1)
    result1 = sim.run(qc1, shots=500).result()
    counts1 = result1.get_counts()
    assert counts1 == {"1": 500}


def test_encoder_and_syndrome_circuit_registers():
    """Verify register dimensions and structure for QEC circuits."""
    s = get_quantum_state("|0>")
    enc = build_encoder_circuit(state_info=s)
    assert enc.num_qubits == 3

    syn_qc = build_syndrome_extraction_circuit(state_info=s)
    assert syn_qc.num_qubits == 5  # 3 data + 2 ancillas
    assert syn_qc.num_clbits == 2  # 2 syndrome bits

    full_qc = build_full_qec_circuit(state_info=s)
    assert full_qc.num_qubits == 5
    assert full_qc.num_clbits == 3  # 2 syndrome + 1 output


def test_circuit_drawers():
    """Verify circuit drawer exports textual and matplotlib objects."""
    s = get_quantum_state("|0>")
    qc = build_encoder_circuit(state_info=s)
    text = draw_circuit_text(qc)
    assert isinstance(text, str)
    assert "data" in text

    fig = draw_circuit_mpl(qc)
    assert fig is not None
