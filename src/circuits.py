"""
Quantum Circuit Generation Module for QubitLab.

Constructs Qiskit quantum circuits for:
1. Unprotected single-qubit transmission baseline
2. 3-qubit repetition code encoder
3. Ancilla-based syndrome extraction
4. Full autonomous QEC pipeline with conditional feed-forward correction and decoding
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from src.states import QuantumStateInfo, get_quantum_state, prepare_state_on_circuit


def build_unprotected_circuit(
    state_info: Optional[QuantumStateInfo] = None,
    custom_gates: Optional[List[str]] = None,
    channel_noise_placeholder: bool = True
) -> QuantumCircuit:
    """
    Builds the baseline unprotected single-qubit transmission circuit.
    
    Architecture:
    |0> -> [State Prep] -> [User Gates] -> [Channel 'id' Noise Slot] -> [Measure] -> c
    """
    if state_info is None:
        state_info = get_quantum_state("|0>")

    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name="c")
    qc = QuantumCircuit(qr, cr, name="Unprotected_Channel")

    # 1. State preparation
    prepare_state_on_circuit(qc, qubit_index=0, state_info=state_info)
    qc.barrier()

    # 2. Optional user gates
    if custom_gates:
        for gate in custom_gates:
            g = gate.upper().strip()
            if g == "X":
                qc.x(0)
            elif g == "H":
                qc.h(0)
            elif g == "Z":
                qc.z(0)
            elif g == "Y":
                qc.y(0)
            elif g == "S":
                qc.s(0)
        qc.barrier()

    # 3. Channel noise placeholder (Noise models attach to 'id')
    if channel_noise_placeholder:
        qc.id(0)
        qc.barrier()

    # 4. Measurement
    qc.measure(qr[0], cr[0])
    return qc


def build_encoder_circuit(
    state_info: Optional[QuantumStateInfo] = None
) -> QuantumCircuit:
    """
    Builds the 3-qubit repetition code encoder.
    Maps: (alpha|0> + beta|1>) (x) |00> -> alpha|000> + beta|111>
    """
    if state_info is None:
        state_info = get_quantum_state("|0>")

    q = QuantumRegister(3, name="data")
    qc = QuantumCircuit(q, name="Repetition_Encoder")

    # State prep on physical qubit 0
    prepare_state_on_circuit(qc, qubit_index=0, state_info=state_info)
    qc.barrier(label="Prep")

    # Repetition encoding via entangling CNOTs
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])
    qc.barrier(label="Encoded")

    return qc


def build_syndrome_extraction_circuit(
    state_info: Optional[QuantumStateInfo] = None,
    deterministic_errors: Optional[Dict[int, str]] = None
) -> QuantumCircuit:
    """
    Constructs the encoding + channel error + syndrome extraction circuit.
    Uses 2 ancilla qubits to measure parities (q0 (xor) q1) and (q1 (xor) q2)
    without collapsing the logical data superposition.
    """
    if state_info is None:
        state_info = get_quantum_state("|0>")

    q = QuantumRegister(3, name="data")
    a = QuantumRegister(2, name="ancilla")
    syn = ClassicalRegister(2, name="syn")
    qc = QuantumCircuit(q, a, syn, name="Syndrome_Extraction")

    # 1. Prepare and encode
    prepare_state_on_circuit(qc, qubit_index=0, state_info=state_info)
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])
    qc.barrier(label="Encoded")

    # 2. Inject deterministic channel errors if specified
    if deterministic_errors:
        for qubit_idx, error_op in deterministic_errors.items():
            if 0 <= qubit_idx < 3:
                op = error_op.upper()
                if op == "X":
                    qc.x(q[qubit_idx])
                elif op == "Z":
                    qc.z(q[qubit_idx])
                elif op == "Y":
                    qc.y(q[qubit_idx])
        qc.barrier(label="Noise")

    # 3. Syndrome extraction parities
    # Ancilla 0 measures parity of data 0 and data 1: s0 = q0 (xor) q1
    qc.cx(q[0], a[0])
    qc.cx(q[1], a[0])

    # Ancilla 1 measures parity of data 1 and data 2: s1 = q1 (xor) q2
    qc.cx(q[1], a[1])
    qc.cx(q[2], a[1])

    qc.barrier(label="Parity")

    # Measure ancillas into classical syndrome register
    qc.measure(a[0], syn[0])
    qc.measure(a[1], syn[1])

    return qc


def build_full_qec_circuit(
    state_info: Optional[QuantumStateInfo] = None,
    deterministic_errors: Optional[Dict[int, str]] = None,
    channel_noise_placeholder: bool = True
) -> QuantumCircuit:
    """
    Constructs the complete autonomous 3-qubit bit-flip error correction circuit.
    
    Circuit Structure:
    1. Prepare single-qubit state |psi> on data[0]
    2. Encode into logical state |psi_L> = alpha|000> + beta|111>
    3. Channel transmission (stochastic via 'id' gates or deterministic manual injections)
    4. Syndrome extraction via 2 ancilla qubits:
         syn[0] = data[0] ^ data[1]
         syn[1] = data[1] ^ data[2]
    5. Conditional feed-forward correction:
         syn = 01 (int 1) -> Flip data[0]
         syn = 11 (int 3) -> Flip data[1]
         syn = 10 (int 2) -> Flip data[2]
         syn = 00 (int 0) -> No correction
    6. Decode: disentangle data[1] and data[2] back to |00>
    7. Readout: measure data[0] into logical output bit out[0]
    """
    if state_info is None:
        state_info = get_quantum_state("|0>")

    q = QuantumRegister(3, name="data")
    a = QuantumRegister(2, name="ancilla")
    syn = ClassicalRegister(2, name="syn")
    out = ClassicalRegister(1, name="out")
    qc = QuantumCircuit(q, a, syn, out, name="Full_QEC_Pipeline")

    # 1. State preparation
    prepare_state_on_circuit(qc, qubit_index=0, state_info=state_info)
    qc.barrier(label="Prep")

    # 2. Encode
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])
    qc.barrier(label="Encoded")

    # 3. Channel stage
    if deterministic_errors:
        for qubit_idx, error_op in deterministic_errors.items():
            if 0 <= qubit_idx < 3:
                op = error_op.upper()
                if op == "X":
                    qc.x(q[qubit_idx])
                elif op == "Z":
                    qc.z(q[qubit_idx])
                elif op == "Y":
                    qc.y(q[qubit_idx])
    elif channel_noise_placeholder:
        qc.id(q[0])
        qc.id(q[1])
        qc.id(q[2])
    qc.barrier(label="Channel")

    # 4. Syndrome extraction
    qc.cx(q[0], a[0])
    qc.cx(q[1], a[0])
    qc.cx(q[1], a[1])
    qc.cx(q[2], a[1])

    qc.measure(a[0], syn[0])
    qc.measure(a[1], syn[1])
    qc.barrier(label="Syndrome")

    # 5. Conditional Feed-Forward Correction
    # Qiskit registers: syn[0] is bit 0 (val 1), syn[1] is bit 1 (val 2)
    # syn integer = syn[0] + 2*syn[1]
    with qc.if_test((syn, 1)):  # syn[0]=1, syn[1]=0 -> data[0] flipped
        qc.x(q[0])
    with qc.if_test((syn, 3)):  # syn[0]=1, syn[1]=1 -> data[1] flipped
        qc.x(q[1])
    with qc.if_test((syn, 2)):  # syn[0]=0, syn[1]=1 -> data[2] flipped
        qc.x(q[2])
    qc.barrier(label="Corrected")

    # 6. Decoding
    qc.cx(q[0], q[2])
    qc.cx(q[0], q[1])
    qc.barrier(label="Decoded")

    # 7. Readout logical qubit
    qc.measure(q[0], out[0])
    return qc


def draw_circuit_mpl(qc: QuantumCircuit, scale: float = 1.0) -> plt.Figure:
    """
    Renders a clean, publication-grade Matplotlib figure of the circuit diagram.
    """
    fig = qc.draw(
        output="mpl",
        style="iqp",
        scale=scale,
        plot_barriers=True,
        justify="left"
    )
    return fig


def draw_circuit_text(qc: QuantumCircuit) -> str:
    """Returns the formatted ASCII/Unicode text representation of the circuit."""
    return str(qc.draw(output="text"))
