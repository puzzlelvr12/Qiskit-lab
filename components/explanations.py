"""
Pedagogical Explanations and Educational Modules for QubitLab.

Provides modular interactive expandable explanations and conceptual
mini-experiments for quantum noise and error correction.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import streamlit as st


EXPLANATIONS = {
    "logical_vs_physical": {
        "title": "What is a Logical Qubit vs. a Physical Qubit?",
        "content": """
**Physical Qubit:** A tangible, hardware-level quantum two-level system—such as a superconducting transmon circuit, a trapped ion, or a semiconductor quantum dot. Physical qubits are fragile and suffer from thermal noise, control imperfections, and environmental decoherence.

**Logical Qubit:** A protected, virtual quantum state encoded across multiple entangled physical qubits. By spreading quantum information non-locally across an entangled multi-qubit code space, no single localized physical error can destroy the underlying quantum information.
        """
    },
    "syndrome_measurement": {
        "title": "Why can we measure the syndrome without destroying the logical state?",
        "content": """
In quantum mechanics, measuring an unknown superposition $\\alpha|0\\rangle + \\beta|1\\rangle$ directly collapses it into $|0\\rangle$ or $|1\\rangle$, destroying the delicate superposition amplitudes $\\alpha$ and $\\beta$.

**The Quantum Error Correction Solution: Parity Measurement (Syndrome Extraction)**
Instead of measuring the individual data qubits, we entangle them with independent helper qubits called **ancillas** using CNOT gates, measuring only the **relative parity** (e.g., $q_0 \\oplus q_1$ and $q_1 \\oplus q_2$).

Both basis codewords share identical parities:
- $|0_L\\rangle = |000\\rangle \\implies q_0 \\oplus q_1 = 0,\\; q_1 \\oplus q_2 = 0$
- $|1_L\\rangle = |111\\rangle \\implies q_0 \\oplus q_1 = 0,\\; q_1 \\oplus q_2 = 0$

Because both components of $\\alpha|000\\rangle + \\beta|111\\rangle$ yield the same parity eigenvalue $+1$, measuring the parity projects the system into an eigenstate of the parity operator **without extracting any information about whether the state is $|0_L\\rangle$ or $|1_L\\rangle$**. Hence, $\\alpha$ and $\\beta$ remain completely untouched!
        """
    },
    "protected_curve_shape": {
        "title": "Why does the protected curve suppress noise quadratically (P_L ≈ 3p²)?",
        "content": """
In an unprotected system, an error on the single physical qubit immediately causes failure: $P_{\\text{unprot}} = p$ (a linear relationship).

In the 3-qubit repetition code, every single isolated error is detected and corrected. The code **only fails when two or more qubits flip independently**:
$$P_L = 3p^2(1-p) + p^3 = 3p^2 - 2p^3$$

For realistic physical error rates ($p \\ll 1$), higher-order terms like $p^3$ are negligible, so $P_L \\approx 3p^2$.
- If physical noise is $p = 5\\% = 0.05$:
  - Unprotected error rate: **5.0%**
  - Logical error rate: $3(0.05)^2 - 2(0.05)^3 = 0.0075 - 0.00025 = \\mathbf{0.725\\%}$ (nearly a **7× reduction!**)
- If physical noise is $p = 1\\% = 0.01$:
  - Unprotected error: **1.0%**
  - Logical error: $3(0.0001) = \\mathbf{0.03\\%}$ (a **33× reduction!**)
        """
    },
    "two_errors_breakdown": {
        "title": "Why do two bit flips cause the repetition code to fail?",
        "content": """
The 3-qubit code operates on the principle of **majority voting**:
- Codewords are $|000\\rangle$ and $|111\\rangle$.
- Hamming distance between codewords is $d = 3$.
- Code capacity: can correct up to $t = \\lfloor(d-1)/2\\rfloor = 1$ error.

Suppose the transmitted state is $|000\\rangle$, but **two errors** occur on physical qubits 0 and 1, producing $|110\\rangle$.
1. Ancilla 0 measures parity $q_0 \\oplus q_1 = 1 \\oplus 1 = 0$ (they match!).
2. Ancilla 1 measures parity $q_1 \\oplus q_2 = 1 \\oplus 0 = 1$ (they disagree!).
3. Extracted syndrome: `01`.

The classical decoder inspects syndrome `01`, which maps to: *"Qubit 2 disagreed with Qubit 1; flip Qubit 2!"*
The decoder flips qubit 2, converting $|110\\rangle \\to |111\\rangle$.
When decoded, the system reads $|1\\rangle$ instead of the original $|0\\rangle$. A **logical NOT error** has occurred!
        """
    },
    "no_cloning_theorem": {
        "title": "Does Quantum Error Correction violate the No-Cloning Theorem?",
        "content": """
**The No-Cloning Theorem** states that an unknown quantum state $|\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$ cannot be copied into $|\psi\\rangle \\otimes |\\psi\\rangle$.

In classical computing, error correction duplicates bits: $0 \\to 000$ and $1 \\to 111$.
If we attempted to clone quantum states:
$$|\\psi\\rangle \\otimes |\\psi\\rangle \\otimes |\\psi\\rangle = (\\alpha|0\\rangle + \\beta|1\\rangle)^{\\otimes 3} = \\alpha^3|000\\rangle + \\alpha^2\\beta|001\\rangle + \\dots + \\beta^3|111\\rangle$$

**What QEC actually does:**
The repetition code does **NOT** clone $|\psi\\rangle$. Instead, it creates an **entangled state**:
$$|\\psi_L\\rangle = \\alpha|000\\rangle + \\beta|111\\rangle$$
Notice that neither physical qubit 1 nor physical qubit 2 contains the state $|\psi\\rangle$ individually! The quantum information exists strictly in the **non-local entanglement** among all three qubits. No-cloning is fully respected.
        """
    },
    "phase_flip_limitation": {
        "title": "The Phase-Flip (Z) Blind Spot: Why classical repetition is insufficient",
        "content": """
In classical information theory, bit flips ($0 \\leftrightarrow 1$) are the only possible error.
In quantum mechanics, relative phase matters:
$$|+\\rangle = \\frac{|0\\rangle + |1\\rangle}{\\sqrt{2}}, \\quad |-\\rangle = \\frac{|0\\rangle - |1\\rangle}{\\sqrt{2}}$$

A **Pauli Z error** leaves $|0\\rangle$ and $|1\\rangle$ alone, but introduces a phase flip: $Z|0\\rangle = |0\\rangle$ and $Z|1\\rangle = -|1\\rangle$.
If we encode $|+\\rangle$ into the 3-qubit bit-flip code:
$$|+_L\\rangle = \\frac{|000\\rangle + |111\\rangle}{\\sqrt{2}}$$
If a $Z$ error hits physical qubit 0:
$$Z_0 |+_L\\rangle = \\frac{|000\\rangle - |111\\rangle}{\\sqrt{2}} = |-_L\\rangle$$

When we measure bit parities $q_0 \\oplus q_1$ and $q_1 \\oplus q_2$, both $|000\\rangle$ and $|111\\rangle$ yield even parity ($0$).
**The syndrome is (0, 0)! The error is invisible to the bit-flip code!**

To correct both bit-flips and phase-flips simultaneously, true quantum codes like **Shor's 9-qubit code**, the **Steane 7-qubit code**, or the **surface code** concatenate bit-flip and phase-flip protections.
        """
    },
    "fault_tolerant_reality": {
        "title": "Why Real Fault-Tolerant Quantum Computing is Harder",
        "content": """
In our educational playground, we made a standard pedagogical assumption:
*The gates, ancilla measurements, and correction steps are ideal; noise only occurs during memory transmission.*

In real hardware (NISQ processors):
1. **Gate Imperfections:** CNOT gates themselves have error rates between $0.1\\%$ and $1.0\\%$.
2. **Measurement Errors:** Reading ancilla qubits can misreport the syndrome.
3. **Error Propagation:** A faulty CNOT during syndrome extraction can spread a single physical error into multiple data qubits.

**Fault tolerance** requires designing circuits where a single component failure anywhere (in a gate, wire, or measurement) can never propagate into an uncorrectable logical error. This requires topological surface codes with thousands of physical qubits per logical qubit.
        """
    }
}


def render_explanation(key: str, expanded: bool = False):
    """Renders a collapsible educational explanation in Streamlit."""
    info = EXPLANATIONS.get(key)
    if not info:
        return
    with st.expander(f"📘 {info['title']}", expanded=expanded):
        st.markdown(info["content"])
