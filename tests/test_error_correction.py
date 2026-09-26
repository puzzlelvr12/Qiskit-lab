"""
Unit tests for 3-qubit repetition code error correction mechanics in QubitLab.
"""

import pytest
from qiskit_aer import AerSimulator
from src.states import get_quantum_state
from src.error_correction import run_deterministic_qec_walkthrough, SYNDROME_TABLE
from src.circuits import build_full_qec_circuit


def test_noiseless_qec():
    """Verify that ideal transmission succeeds with null syndrome (0, 0)."""
    s0 = get_quantum_state("|0>")
    result = run_deterministic_qec_walkthrough(state_info=s0, error_mode="none")
    assert result.success is True
    assert result.syndrome_bits == (0, 0)
    assert result.diagnosed_qubit == -1


@pytest.mark.parametrize("error_mode, expected_syndrome, expected_qubit", [
    ("flip_q0", (1, 0), 0),
    ("flip_q1", (1, 1), 1),
    ("flip_q2", (0, 1), 2),
])
def test_single_x_error_correction(error_mode, expected_syndrome, expected_qubit):
    """Verify that any single bit-flip error on q0, q1, or q2 is uniquely diagnosed and corrected."""
    s0 = get_quantum_state("|0>")
    res0 = run_deterministic_qec_walkthrough(state_info=s0, error_mode=error_mode)
    assert res0.success is True
    assert res0.syndrome_bits == expected_syndrome
    assert res0.diagnosed_qubit == expected_qubit

    s1 = get_quantum_state("|1>")
    res1 = run_deterministic_qec_walkthrough(state_info=s1, error_mode=error_mode)
    assert res1.success is True
    assert res1.syndrome_bits == expected_syndrome
    assert res1.diagnosed_qubit == expected_qubit


def test_two_x_errors_failure():
    """Verify that two bit-flip errors fool the syndrome lookup into a logical error."""
    s0 = get_quantum_state("|0>")
    res = run_deterministic_qec_walkthrough(state_info=s0, error_mode="flip_two")
    assert res.success is False
    assert res.diagnosed_qubit == 2  # Diagnoses the wrong qubit
    assert "Logical Error" in res.explanation


def test_phase_flip_limitation():
    """Verify that a Pauli Z error passes undetected through the bit-flip repetition code."""
    sp = get_quantum_state("|+>")
    res = run_deterministic_qec_walkthrough(state_info=sp, error_mode="phase_q0")
    assert res.success is False
    assert res.syndrome_bits == (0, 0)  # Undetected!
    assert "Silent Phase Corruption" in res.explanation


def test_circuit_level_conditional_correction_q0():
    """Verify Qiskit dynamic circuit with AerSimulator for error on q0."""
    s0 = get_quantum_state("|0>")
    qc = build_full_qec_circuit(state_info=s0, deterministic_errors={0: "X"})
    sim = AerSimulator()
    result = sim.run(qc, shots=200).result()
    counts = result.get_counts()
    
    # Format 'out syn'. Qubit 0 flipped: syn[0]=1, syn[1]=0 -> bitstring '01'
    # out must be '0' (success)
    # The output key should be '0 01'
    assert "0 01" in counts
    assert counts.get("1 01", 0) == 0


def test_circuit_level_conditional_correction_q1():
    """Verify Qiskit dynamic circuit with AerSimulator for error on q1."""
    s0 = get_quantum_state("|0>")
    qc = build_full_qec_circuit(state_info=s0, deterministic_errors={1: "X"})
    sim = AerSimulator()
    result = sim.run(qc, shots=200).result()
    counts = result.get_counts()
    # syn[0]=1, syn[1]=1 -> '11'
    # out should be '0'
    assert "0 11" in counts
    assert counts.get("1 11", 0) == 0


def test_circuit_level_conditional_correction_q2():
    """Verify Qiskit dynamic circuit with AerSimulator for error on q2."""
    s0 = get_quantum_state("|0>")
    qc = build_full_qec_circuit(state_info=s0, deterministic_errors={2: "X"})
    sim = AerSimulator()
    result = sim.run(qc, shots=200).result()
    counts = result.get_counts()
    # syn[0]=0, syn[1]=1 -> '10'
    # out should be '0'
    assert "0 10" in counts
    assert counts.get("1 10", 0) == 0


def test_circuit_level_two_errors_logical_flip():
    """Verify Qiskit dynamic circuit confirms logical flip when two errors occur."""
    s0 = get_quantum_state("|0>")
    qc = build_full_qec_circuit(state_info=s0, deterministic_errors={0: "X", 1: "X"})
    sim = AerSimulator()
    result = sim.run(qc, shots=200).result()
    counts = result.get_counts()
    # When q0 and q1 flip, syn is '10' (syn[0]=0, syn[1]=1), correction flips q2, out flips to '1'!
    assert "1 10" in counts
    assert counts.get("0 10", 0) == 0
