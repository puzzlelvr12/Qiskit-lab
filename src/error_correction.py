"""
Quantum Error Correction Engine for QubitLab.

Implements the 3-qubit bit-flip repetition code, syndrome extraction,
syndrome lookup decoding, deterministic pedagogical step-by-step walks,
two-error failure dynamics, and phase-flip limitation demonstrations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from src.states import QuantumStateInfo, get_quantum_state
from src.circuits import build_full_qec_circuit


# Canonical syndrome mapping for the 3-qubit repetition code:
# Ancilla 0 measures parity q0 ^ q1
# Ancilla 1 measures parity q1 ^ q2
# Syndrome bitstring is formatted as 's1 s0' (s1=q1^q2, s0=q0^q1)
SYNDROME_TABLE: Dict[Tuple[int, int], Dict[str, str]] = {
    (0, 0): {
        "diagnosed_error": "No bit-flip error detected",
        "action": "Identity (do nothing)",
        "target_qubit": "None",
        "corrected_qubit_index": -1,
        "explanation": "Both parity checks (q0 ⊕ q1 and q1 ⊕ q2) are even (0). All physical qubits agree."
    },
    (1, 0): {
        "diagnosed_error": "Bit-flip error on physical qubit 0",
        "action": "Apply Pauli X to qubit 0",
        "target_qubit": "Qubit 0",
        "corrected_qubit_index": 0,
        "explanation": "q0 ⊕ q1 = 1 (disagree) and q1 ⊕ q2 = 0 (agree). Since q1 and q2 match, q0 must be the outlier."
    },
    (1, 1): {
        "diagnosed_error": "Bit-flip error on physical qubit 1",
        "action": "Apply Pauli X to qubit 1",
        "target_qubit": "Qubit 1",
        "corrected_qubit_index": 1,
        "explanation": "q0 ⊕ q1 = 1 (disagree) and q1 ⊕ q2 = 1 (disagree). The middle qubit q1 disagrees with both neighbors."
    },
    (0, 1): {
        "diagnosed_error": "Bit-flip error on physical qubit 2",
        "action": "Apply Pauli X to qubit 2",
        "target_qubit": "Qubit 2",
        "corrected_qubit_index": 2,
        "explanation": "q0 ⊕ q1 = 0 (agree) and q1 ⊕ q2 = 1 (disagree). Since q0 and q1 match, q2 must be the outlier."
    },
}


@dataclass
class StepByStepQECResult:
    """Detailed trace of an error correction walkthrough."""
    initial_state: QuantumStateInfo
    injected_errors: Dict[int, str]
    error_label: str
    syndrome_bits: Tuple[int, int]  # (s0, s1)
    syndrome_str: str              # "s0s1"
    diagnosed_qubit: int           # 0, 1, 2, or -1
    diagnosed_action: str
    correction_applied: str
    final_logical_state: str       # |0>, |1>, |+>, etc.
    success: bool
    explanation: str
    state_descriptions: List[Dict[str, str]]


def run_deterministic_qec_walkthrough(
    state_info: QuantumStateInfo,
    error_mode: str = "none"
) -> StepByStepQECResult:
    """
    Executes a step-by-step pedagogical walkthrough of the 3-qubit repetition code.
    
    Supported error modes:
    - 'none': No errors occur
    - 'flip_q0': X error on physical qubit 0
    - 'flip_q1': X error on physical qubit 1
    - 'flip_q2': X error on physical qubit 2
    - 'flip_two': X error on physical qubits 0 and 1 (demonstrates 2-error code breakdown)
    - 'phase_q0': Z error on physical qubit 0 (demonstrates repetition code phase vulnerability)
    """
    errors: Dict[int, str] = {}
    if error_mode == "flip_q0":
        errors = {0: "X"}
        error_label = "Single X error on physical Qubit 0"
    elif error_mode == "flip_q1":
        errors = {1: "X"}
        error_label = "Single X error on physical Qubit 1"
    elif error_mode == "flip_q2":
        errors = {2: "X"}
        error_label = "Single X error on physical Qubit 2"
    elif error_mode == "flip_two":
        errors = {0: "X", 1: "X"}
        error_label = "Two X errors on physical Qubits 0 and 1"
    elif error_mode == "phase_q0":
        errors = {0: "Z"}
        error_label = "Phase-flip (Z) error on physical Qubit 0"
    else:
        error_mode = "none"
        errors = {}
        error_label = "No noise (Ideal channel)"

    steps_trace: List[Dict[str, str]] = []

    # Step 1: Input state
    steps_trace.append({
        "step": "1. Input State Preparation",
        "description": f"Initialized logical state on physical qubit 0: {state_info.name} "
                       f"= {state_info.alpha.real:.3f}|0⟩ + {state_info.beta.real:.3f}|1⟩. "
                       f"Qubits 1 and 2 are in ground state |0⟩."
    })

    # Step 2: Encoding
    steps_trace.append({
        "step": "2. Repetition Code Encoding",
        "description": f"Applied CNOT(q0, q1) and CNOT(q0, q2). "
                       f"State is entangled into code space: |ψ_L⟩ = "
                       f"{state_info.alpha.real:.3f}|000⟩ + {state_info.beta.real:.3f}|111⟩."
    })

    # Step 3: Channel Error
    if errors:
        err_desc = ", ".join([f"{op} on physical qubit {q}" for q, op in errors.items()])
        steps_trace.append({
            "step": "3. Channel Noise Injected",
            "description": f"Noise occurred: {err_desc}."
        })
    else:
        steps_trace.append({
            "step": "3. Channel Transmission",
            "description": "State passed through channel without error."
        })

    # Step 4: Parity / Syndrome Calculation
    # Physical qubits bit-values for bit flips:
    # If state started as |0>, encoded is |000>
    # Determine the parity bits:
    q0_flipped = (errors.get(0) == "X")
    q1_flipped = (errors.get(1) == "X")
    q2_flipped = (errors.get(2) == "X")
    z0_flipped = (errors.get(0) == "Z")

    s0 = int(q0_flipped ^ q1_flipped)  # Parity of q0 and q1
    s1 = int(q1_flipped ^ q2_flipped)  # Parity of q1 and q2

    syndrome_key = (s0, s1)
    syndrome_info = SYNDROME_TABLE[syndrome_key]

    steps_trace.append({
        "step": "4. Syndrome Extraction (Ancilla Parity Checks)",
        "description": f"Ancilla a0 measured q0 ⊕ q1 = {s0}. "
                       f"Ancilla a1 measured q1 ⊕ q2 = {s1}. "
                       f"Syndrome pattern: (s0={s0}, s1={s1}). "
                       f"{syndrome_info['explanation']}"
    })

    # Step 5: Diagnosis and Correction
    diagnosed_qubit = syndrome_info["corrected_qubit_index"]
    correction_action = syndrome_info["action"]

    steps_trace.append({
        "step": "5. Classical Decoding & Correction",
        "description": f"Syndrome lookup identifies: {syndrome_info['diagnosed_error']}. "
                       f"Corrective action: {correction_action}."
    })

    # Step 6: Decoding and Final Result
    # Evaluate what the final state actually is:
    if error_mode in ["none", "flip_q0", "flip_q1", "flip_q2"]:
        success = True
        final_state_desc = state_info.name
        explanation = (
            f"SUCCESS! The single {error_label.lower()} was accurately isolated by the syndrome "
            f"measurement and corrected without ever measuring or destroying the quantum superposition."
        )
    elif error_mode == "flip_two":
        # Two bit flips: q0 and q1 flipped -> s0 = 1^1 = 0, s1 = 1^0 = 1 -> syndrome (0, 1)
        # Syndrome table diagnoses qubit 2 as the error!
        # It applies X to qubit 2. Now ALL THREE qubits are flipped (|111> instead of |000>)!
        # When decoded, logical qubit 0 has suffered an UNCORRECTABLE LOGICAL BIT FLIP.
        success = False
        final_state_desc = "LOGICAL ERROR (|0⟩ ↔ |1⟩ inverted)"
        explanation = (
            "FAILURE (Logical Error)! Two physical qubits (q0 and q1) flipped. "
            "Because this code assumes errors are rare and single, the syndrome (0, 1) made it appear "
            "that only qubit 2 was inverted. The correction flipped qubit 2, resulting in all three qubits "
            "being flipped. When decoded, a logical NOT error occurred. "
            "This proves why the 3-qubit repetition code can only tolerate 1 error."
        )
    elif error_mode == "phase_q0":
        # Phase flip (Z):
        # Parity checks measure bit-parity: Z preserves computational basis states |0> and |1>
        # Hence syndrome is (0, 0) -> NO ERROR DETECTED!
        # If the input was |+> = (|0> + |1>)/sqrt(2), Z flips it to |-> = (|0> - |1>)/sqrt(2).
        success = False
        final_state_desc = "PHASE ERROR (Relative phase flipped)"
        explanation = (
            "FAILURE (Silent Phase Corruption)! A Pauli Z error was injected into physical qubit 0. "
            "Because the bit-flip repetition code only checks bit parities (X-basis syndromes), "
            "the Z error left computational bit parities unchanged, producing syndrome (0, 0). "
            "The error passed completely undetected! This demonstrates the fundamental limitation of "
            "classical-style repetition codes: quantum error correction requires codes that protect "
            "against BOTH bit-flip (X) and phase-flip (Z) errors simultaneously (such as Shor's 9-qubit code "
            "or the 7-qubit Steane code)."
        )
    else:
        success = True
        final_state_desc = state_info.name
        explanation = "Walkthrough completed."

    steps_trace.append({
        "step": "6. Logical Decoding & Final Fidelity",
        "description": f"Decoded logical state: {final_state_desc}. "
                       f"Result: {'PASS (High Fidelity)' if success else 'FAIL (Logical Error)'}."
    })

    return StepByStepQECResult(
        initial_state=state_info,
        injected_errors=errors,
        error_label=error_label,
        syndrome_bits=(s0, s1),
        syndrome_str=f"{s0}{s1}",
        diagnosed_qubit=diagnosed_qubit,
        diagnosed_action=correction_action,
        correction_applied=correction_action,
        final_logical_state=final_state_desc,
        success=success,
        explanation=explanation,
        state_descriptions=steps_trace,
    )
