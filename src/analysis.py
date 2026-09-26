"""
Statistical Analysis and Theoretical Modeling Module for QubitLab.

Calculates theoretical logical error probabilities, standard errors,
confidence intervals, and validates simulated results against theory.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class StatisticalComparison:
    """Contains statistical comparisons between simulated and theoretical results."""
    physical_p: float
    simulated_unprotected: float
    theoretical_unprotected: float
    simulated_logical: float
    theoretical_logical: float
    shots: int
    unprotected_std_error: float
    logical_std_error: float
    unprotected_ci: Tuple[float, float]
    logical_ci: Tuple[float, float]
    is_within_expected_variance: bool
    summary: str


def theoretical_unprotected_error(p: float) -> float:
    """
    Theoretical bit-flip error probability for an unprotected single qubit:
    P_unprot = p
    """
    return float(np.clip(p, 0.0, 1.0))


def theoretical_logical_error_repetition_3(p: float) -> float:
    """
    Theoretical logical bit-flip failure probability for the 3-qubit repetition code.
    
    The code fails if and only if at least 2 of the 3 physical qubits suffer a bit flip:
    P_L = C(3, 2) * p^2 * (1 - p) + C(3, 3) * p^3
        = 3 * p^2 * (1 - p) + p^3
        = 3*p^2 - 3*p^3 + p^3
        = 3*p^2 - 2*p^3
    """
    p_clamped = float(np.clip(p, 0.0, 1.0))
    return float(3.0 * (p_clamped ** 2) - 2.0 * (p_clamped ** 3))


def compute_standard_error(rate: float, shots: int) -> float:
    """
    Standard error of a Bernoulli proportion:
    SE = sqrt(p * (1 - p) / N)
    """
    if shots <= 0:
        return 0.0
    p = float(np.clip(rate, 0.0, 1.0))
    return float(np.sqrt((p * (1.0 - p)) / float(shots)))


def compute_confidence_interval(rate: float, shots: int, z_score: float = 1.96) -> Tuple[float, float]:
    """
    Calculates the 95% Wald confidence interval for a proportion.
    """
    se = compute_standard_error(rate, shots)
    low = max(0.0, rate - z_score * se)
    high = min(1.0, rate + z_score * se)
    return float(low), float(high)


def analyze_experiment_point(
    p: float,
    simulated_unprotected: float,
    simulated_logical: float,
    shots: int
) -> StatisticalComparison:
    """
    Performs rigorous statistical comparison of a simulated experiment point against theory.
    """
    p_theory_unprot = theoretical_unprotected_error(p)
    p_theory_logical = theoretical_logical_error_repetition_3(p)

    se_unprot = compute_standard_error(simulated_unprotected, shots)
    se_logical = compute_standard_error(simulated_logical, shots)

    ci_unprot = compute_confidence_interval(simulated_unprotected, shots)
    ci_logical = compute_confidence_interval(simulated_logical, shots)

    # Check if theoretical value falls within ~3 standard errors (99.7% confidence)
    margin_unprot = max(3.0 * se_unprot, 0.015)
    margin_logical = max(3.0 * se_logical, 0.015)

    within_unprot = abs(simulated_unprotected - p_theory_unprot) <= margin_unprot
    within_logical = abs(simulated_logical - p_theory_logical) <= margin_logical
    is_valid = within_unprot and within_logical

    summary = (
        f"At p={p:.2f}, simulated logical error is {simulated_logical:.4f} "
        f"vs theory {p_theory_logical:.4f} (SE: {se_logical:.4f}). "
        f"Unprotected simulated is {simulated_unprotected:.4f} vs theory {p_theory_unprot:.4f}."
    )

    return StatisticalComparison(
        physical_p=p,
        simulated_unprotected=simulated_unprotected,
        theoretical_unprotected=p_theory_unprot,
        simulated_logical=simulated_logical,
        theoretical_logical=p_theory_logical,
        shots=shots,
        unprotected_std_error=se_unprot,
        logical_std_error=se_logical,
        unprotected_ci=ci_unprot,
        logical_ci=ci_logical,
        is_within_expected_variance=is_valid,
        summary=summary,
    )
