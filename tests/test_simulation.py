"""
Unit tests for simulation engine and statistical validation in QubitLab.
"""

import numpy as np
import pytest
from src.states import get_quantum_state
from src.simulation import (
    run_unprotected_simulation,
    run_qec_simulation,
    run_side_by_side_experiment,
    run_comparison_sweep,
    simulate_single_trial,
)
from src.analysis import (
    theoretical_unprotected_error,
    theoretical_logical_error_repetition_3,
    compute_standard_error,
    analyze_experiment_point,
)


def test_theoretical_formulas():
    """Verify analytical formulas for 3-qubit repetition code."""
    # At p=0, PL=0
    assert theoretical_logical_error_repetition_3(0.0) == 0.0
    assert theoretical_unprotected_error(0.0) == 0.0

    # At p=0.1: PL = 3(0.01) - 2(0.001) = 0.028
    assert np.isclose(theoretical_logical_error_repetition_3(0.1), 0.028)

    # At pseudothreshold p=0.5: PL = 3(0.25) - 2(0.125) = 0.75 - 0.25 = 0.50
    assert np.isclose(theoretical_logical_error_repetition_3(0.5), 0.50)


def test_unprotected_simulation_noise():
    """Verify unprotected simulation agrees with physical noise rate."""
    s0 = get_quantum_state("|0>")
    p = 0.15
    shots = 3000
    counts, err_rate = run_unprotected_simulation(state_info=s0, p=p, shots=shots, seed=42)
    # 3 standard errors
    se = np.sqrt(p * (1 - p) / shots)
    assert abs(err_rate - p) <= 3.5 * se


def test_qec_simulation_noise_reduction():
    """Verify QEC simulation significantly suppresses errors at low p."""
    s0 = get_quantum_state("|0>")
    p = 0.10
    shots = 4000
    _, syn_counts, logical_err = run_qec_simulation(state_info=s0, p=p, shots=shots, seed=42)
    
    # At p=0.10, unprotected is ~10%, logical should be ~2.8%
    pl_theory = theoretical_logical_error_repetition_3(p)
    se = np.sqrt(pl_theory * (1 - pl_theory) / shots)
    assert abs(logical_err - pl_theory) <= 3.5 * se
    # Logical error should be strictly less than physical noise
    assert logical_err < p


def test_side_by_side_experiment():
    """Verify side-by-side experiment structure and metrics."""
    s0 = get_quantum_state("|0>")
    exp = run_side_by_side_experiment(state_info=s0, p=0.10, shots=2000, seed=123)
    assert exp.shots == 2000
    assert exp.physical_p == 0.10
    assert exp.logical_error_rate < exp.unprotected_error_rate
    assert sum(exp.syndrome_distribution.values()) == 2000


def test_comparison_sweep_data():
    """Verify multi-probability sweep returns valid DataFrame."""
    s0 = get_quantum_state("|0>")
    probs = [0.0, 0.05, 0.10, 0.20]
    df = run_comparison_sweep(state_info=s0, probabilities=probs, shots=1000, seed=99)
    assert len(df) == len(probs)
    assert "logical_simulated" in df.columns
    assert "unprotected_simulated" in df.columns
    assert "logical_theory" in df.columns


def test_single_trial_trace():
    """Verify single trial stochastic trace generation."""
    s0 = get_quantum_state("|0>")
    
    # 0 errors
    t0 = simulate_single_trial(s0, force_errors=[])
    assert t0.success is True
    assert t0.syndrome_str == "00"

    # 1 error on q1
    t1 = simulate_single_trial(s0, force_errors=[1])
    assert t1.success is True
    assert t1.syndrome_str == "11"
    assert "qubit 1" in t1.diagnosed_error.lower()

    # 2 errors on q0 and q1
    t2 = simulate_single_trial(s0, force_errors=[0, 1])
    assert t2.success is False
    assert t2.syndrome_str == "01"


def test_statistical_analysis_validation():
    """Verify analytical validation function."""
    stat = analyze_experiment_point(p=0.10, simulated_unprotected=0.102, simulated_logical=0.027, shots=2000)
    assert stat.is_within_expected_variance is True
    assert np.isclose(stat.theoretical_logical, 0.028)
