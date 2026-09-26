"""
Quantum Simulation Engine for QubitLab.

Provides Monte Carlo shot-based simulation using Qiskit Aer:
1. Unprotected baseline simulation under noise
2. Protected 3-qubit repetition QEC simulation
3. Multi-probability sweeps for benchmarking
4. Single-trial stochastic walkthrough inspector
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from src.states import QuantumStateInfo, get_quantum_state
from src.circuits import build_unprotected_circuit, build_full_qec_circuit
from src.noise import create_noise_model
from src.error_correction import SYNDROME_TABLE


@dataclass
class SingleExperimentResult:
    """Results from running a batch of shots for a single configuration."""
    shots: int
    physical_p: float
    noise_type: str
    input_state: str
    target_bit: int  # Expected computational basis bit for state |0> (0) or |1> (1)
    unprotected_error_rate: float
    logical_error_rate: float
    unprotected_counts: Dict[str, int]
    qec_counts: Dict[str, int]
    syndrome_distribution: Dict[str, int]


@dataclass
class SingleTrialTrace:
    """Step-by-step trace of a single stochastic physical trial."""
    trial_id: int
    input_state: str
    encoded_state: str
    physical_p: float
    physical_errors_injected: List[int]  # List of qubit indices that suffered bit-flip
    corrupted_physical_state: str
    syndrome_bits: Tuple[int, int]
    syndrome_str: str
    diagnosed_error: str
    corrective_action: str
    corrected_physical_state: str
    decoded_bit: int
    expected_bit: int
    success: bool
    summary: str


def run_unprotected_simulation(
    state_info: QuantumStateInfo,
    noise_type: str = "bit_flip",
    p: float = 0.0,
    shots: int = 1000,
    seed: Optional[int] = None
) -> Tuple[Dict[str, int], float]:
    """
    Simulates the unprotected single-qubit transmission circuit under noise.
    
    Returns:
        (counts, observed_error_rate)
    """
    qc = build_unprotected_circuit(state_info=state_info, channel_noise_placeholder=True)
    noise_model = create_noise_model(noise_type=noise_type, probability=p, target_instructions=["id"])
    
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Determine expected readout:
    # For |0>, expected is '0'. For |1>, expected is '1'.
    # For superposition states, error rate is measured relative to majority basis.
    expected_bit = "0" if state_info.prob_0 >= state_info.prob_1 else "1"
    incorrect_bit = "1" if expected_bit == "0" else "0"
    errors = counts.get(incorrect_bit, 0)
    error_rate = float(errors / shots)

    return counts, error_rate


def run_qec_simulation(
    state_info: QuantumStateInfo,
    noise_type: str = "bit_flip",
    p: float = 0.0,
    shots: int = 1000,
    seed: Optional[int] = None
) -> Tuple[Dict[str, int], Dict[str, int], float]:
    """
    Simulates the 3-qubit repetition code error-corrected circuit under noise.
    
    Returns:
        (raw_counts, syndrome_counts, observed_logical_error_rate)
    """
    qc = build_full_qec_circuit(state_info=state_info, channel_noise_placeholder=True)
    noise_model = create_noise_model(noise_type=noise_type, probability=p, target_instructions=["id"])

    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    result = sim.run(qc, shots=shots).result()
    raw_counts = result.get_counts()

    # Format of count keys in Qiskit with registers out[1] and syn[2]:
    # Keys are separated by space: "out syn" e.g. "0 00", "1 11"
    expected_bit = "0" if state_info.prob_0 >= state_info.prob_1 else "1"
    incorrect_bit = "1" if expected_bit == "0" else "0"

    logical_errors = 0
    syndrome_counts: Dict[str, int] = {"00": 0, "01": 0, "10": 0, "11": 0}

    for bitstring, count in raw_counts.items():
        parts = bitstring.split()
        if len(parts) == 2:
            out_bit, syn_str = parts
        else:
            # Fallback if no space: out is first char, syn is rest
            out_bit = bitstring[0]
            syn_str = bitstring[1:]

        if out_bit == incorrect_bit:
            logical_errors += count

        syndrome_counts[syn_str] = syndrome_counts.get(syn_str, 0) + count

    logical_error_rate = float(logical_errors / shots)
    return raw_counts, syndrome_counts, logical_error_rate


def run_side_by_side_experiment(
    state_info: QuantumStateInfo,
    noise_type: str = "bit_flip",
    p: float = 0.1,
    shots: int = 2000,
    seed: Optional[int] = None
) -> SingleExperimentResult:
    """
    Runs both the unprotected baseline and the protected 3-qubit QEC circuit
    with identical physical noise probability p for direct side-by-side comparison.
    """
    unprot_counts, unprot_err = run_unprotected_simulation(
        state_info=state_info,
        noise_type=noise_type,
        p=p,
        shots=shots,
        seed=seed
    )

    qec_counts, syn_counts, qec_err = run_qec_simulation(
        state_info=state_info,
        noise_type=noise_type,
        p=p,
        shots=shots,
        seed=seed
    )

    target_bit = 0 if state_info.prob_0 >= state_info.prob_1 else 1

    return SingleExperimentResult(
        shots=shots,
        physical_p=p,
        noise_type=noise_type,
        input_state=state_info.name,
        target_bit=target_bit,
        unprotected_error_rate=unprot_err,
        logical_error_rate=qec_err,
        unprotected_counts=unprot_counts,
        qec_counts=qec_counts,
        syndrome_distribution=syn_counts
    )


def run_comparison_sweep(
    state_info: QuantumStateInfo,
    probabilities: Optional[List[float]] = None,
    noise_type: str = "bit_flip",
    shots: int = 2000,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Sweeps physical error probabilities across [0.0, 0.5] and gathers both
    simulated and theoretical error rates for unprotected vs protected systems.
    """
    if probabilities is None:
        probabilities = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    records = []
    for p in probabilities:
        p_val = float(p)
        exp = run_side_by_side_experiment(
            state_info=state_info,
            noise_type=noise_type,
            p=p_val,
            shots=shots,
            seed=seed
        )

        # Theoretical calculations for bit-flip noise on |0>
        # Unprotected theory = p
        # Protected 3-qubit repetition theory = 3p^2 - 2p^3
        p_unprot_theory = p_val
        p_logical_theory = 3.0 * (p_val ** 2) - 2.0 * (p_val ** 3)

        records.append({
            "physical_p": p_val,
            "unprotected_simulated": exp.unprotected_error_rate,
            "logical_simulated": exp.logical_error_rate,
            "unprotected_theory": p_unprot_theory,
            "logical_theory": p_logical_theory,
            "improvement_factor": (
                exp.unprotected_error_rate / exp.logical_error_rate
                if exp.logical_error_rate > 0 else (1.0 if exp.unprotected_error_rate == 0 else 999.0)
            ),
            "shots": shots,
        })

    return pd.DataFrame(records)


def simulate_single_trial(
    state_info: QuantumStateInfo,
    p: float = 0.15,
    force_errors: Optional[List[int]] = None,
    rng: Optional[np.random.Generator] = None
) -> SingleTrialTrace:
    """
    Simulates a single stochastic trial of the 3-qubit repetition code.
    Visualizes the exact lifecycle:
    Input -> Encoded -> Noise Occurred -> Syndrome -> Corrected -> Decoded -> Verification.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Determine input basis bit
    expected_bit = 0 if state_info.prob_0 >= state_info.prob_1 else 1
    input_str = f"|{expected_bit}⟩"
    encoded_str = f"|{expected_bit}{expected_bit}{expected_bit}⟩"

    # Determine physical errors
    if force_errors is not None:
        flipped_qubits = list(force_errors)
    else:
        # Sample independent Bernoulli trials with probability p for each of the 3 qubits
        flips = rng.random(3) < p
        flipped_qubits = [i for i, flipped in enumerate(flips) if flipped]

    # Build corrupted string
    physical_bits = [expected_bit, expected_bit, expected_bit]
    for q_idx in flipped_qubits:
        physical_bits[q_idx] = 1 - physical_bits[q_idx]

    corrupted_str = f"|{''.join(map(str, physical_bits))}⟩"

    # Parity syndrome calculation:
    # Ancilla 0 measures parity of data 0 and data 1: s0 = q0 ^ q1
    # Ancilla 1 measures parity of data 1 and data 2: s1 = q1 ^ q2
    s0 = physical_bits[0] ^ physical_bits[1]
    s1 = physical_bits[1] ^ physical_bits[2]
    syndrome_key = (s0, s1)
    syndrome_str = f"{s0}{s1}"

    syndrome_data = SYNDROME_TABLE.get(syndrome_key, {
        "diagnosed_error": "Unknown error",
        "action": "None",
        "target_qubit": "None",
        "corrected_qubit_index": -1
    })

    diagnosed = syndrome_data["diagnosed_error"]
    action = syndrome_data["action"]
    qubit_to_fix = syndrome_data["corrected_qubit_index"]

    # Apply correction
    corrected_bits = list(physical_bits)
    if qubit_to_fix >= 0:
        corrected_bits[qubit_to_fix] = 1 - corrected_bits[qubit_to_fix]

    corrected_str = f"|{''.join(map(str, corrected_bits))}⟩"

    # Decoding: CNOT(q0, q2), CNOT(q0, q1), then measure q0
    decoded_bit = corrected_bits[0]
    success = (decoded_bit == expected_bit)

    if success:
        if len(flipped_qubits) == 0:
            summary = "Ideal transmission: No physical errors occurred."
        elif len(flipped_qubits) == 1:
            summary = f"Single error on qubit {flipped_qubits[0]} successfully detected and corrected!"
        else:
            summary = "Transmission preserved."
    else:
        summary = (
            f"Logical Error! {len(flipped_qubits)} physical errors occurred on qubits {flipped_qubits}. "
            f"The code diagnosed qubit {qubit_to_fix}, causing an uncorrectable logical flip."
        )

    return SingleTrialTrace(
        trial_id=1,
        input_state=input_str,
        encoded_state=encoded_str,
        physical_p=p,
        physical_errors_injected=flipped_qubits,
        corrupted_physical_state=corrupted_str,
        syndrome_bits=syndrome_key,
        syndrome_str=syndrome_str,
        diagnosed_error=diagnosed,
        corrective_action=action,
        corrected_physical_state=corrected_str,
        decoded_bit=decoded_bit,
        expected_bit=expected_bit,
        success=success,
        summary=summary
    )
