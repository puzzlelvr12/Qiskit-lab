"""
QubitLab: An Interactive Quantum Noise & Error Correction Playground
Main Streamlit Application.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

from src.states import (
    get_quantum_state,
    CANONICAL_STATES,
    bloch_from_density_matrix,
)
from src.noise import (
    apply_analytical_noise_channel,
    get_noise_channel_explanation,
)
from src.circuits import (
    build_unprotected_circuit,
    build_encoder_circuit,
    build_syndrome_extraction_circuit,
    build_full_qec_circuit,
    draw_circuit_mpl,
    draw_circuit_text,
)
from src.error_correction import (
    run_deterministic_qec_walkthrough,
    SYNDROME_TABLE,
)
from src.simulation import (
    run_side_by_side_experiment,
    run_comparison_sweep,
    simulate_single_trial,
)
from src.analysis import (
    theoretical_logical_error_repetition_3,
    theoretical_unprotected_error,
    analyze_experiment_point,
)
from components.visualizations import (
    plot_bloch_sphere_3d,
    plot_measurement_distribution,
    plot_interactive_benchmark,
    plot_syndrome_bar_chart,
)
from components.explanations import render_explanation


# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="QubitLab | Quantum Noise & Error Correction Playground",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Scientific CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f8fafc;
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .pipeline-step {
        background: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.92rem;
    }
    .pipeline-step-err {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.92rem;
    }
    .pipeline-step-success {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.92rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def sidebar_state_selector():
    """Sidebar quantum state configuration widget."""
    st.sidebar.markdown("### ⚛️ Quantum State")
    state_options = list(CANONICAL_STATES.keys()) + ["Custom (θ, φ)"]
    selected_option = st.sidebar.selectbox(
        "Choose Input State |ψ⟩:",
        options=state_options,
        index=0,
        help="Select a standard computational/superposition state or configure Bloch angles."
    )

    if selected_option == "Custom (θ, φ)":
        col_th, col_ph = st.sidebar.columns(2)
        with col_th:
            theta = st.slider("Polar θ (rad)", 0.0, float(np.pi), float(np.pi / 2), 0.05)
        with col_ph:
            phi = st.slider("Azimuthal φ (rad)", 0.0, float(2 * np.pi), 0.0, 0.05)
        state_info = get_quantum_state(theta=theta, phi=phi)
    else:
        state_info = get_quantum_state(selected_option)

    # State parameters summary in sidebar
    st.sidebar.markdown(f"**Selected:** `{state_info.name}`")
    st.sidebar.caption(
        f"α = {state_info.alpha.real:.3f} + {state_info.alpha.imag:.3f}i\n\n"
        f"β = {state_info.beta.real:.3f} + {state_info.beta.imag:.3f}i\n\n"
        f"P(|0⟩) = {state_info.prob_0 * 100:.1f}%, P(|1⟩) = {state_info.prob_1 * 100:.1f}%"
    )
    st.sidebar.divider()
    return state_info


def main_app():
    # Header Banner
    st.title("QubitLab")
    st.markdown(
        "##### **An Interactive Quantum Noise & Error Correction Playground**  \n"
        "Explore what actually happens to quantum information when noise occurs, "
        "and observe how simple quantum error correction protects quantum states."
    )
    st.divider()

    state_info = sidebar_state_selector()

    # Main Navigation Tabs
    tab_playground, tab_noise, tab_walkthrough, tab_compare, tab_benchmark, tab_trials, tab_challenges, tab_learn = st.tabs([
        "🔬 State Playground",
        "⚡ Noise Explorer",
        "🛠️ Interactive QEC Walkthrough",
        "⚖️ QEC ON vs OFF",
        "📈 Theory vs. Simulation",
        "🎲 Single-Trial Inspector",
        "🎯 Predict & Verify",
        "📚 Knowledge Base"
    ])

    # -------------------------------------------------------------
    # TAB 1: STATE PLAYGROUND
    # -------------------------------------------------------------
    with tab_playground:
        st.subheader("1. Quantum State Preparation & Single-Qubit Circuit")
        st.markdown(
            "Configure a quantum state and inspect its 3D Bloch sphere representation, "
            "computational basis measurement statistics, and the Qiskit preparation circuit."
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🌐 3D Bloch Sphere")
            fig_bloch = plot_bloch_sphere_3d(
                bloch_vector=state_info.bloch_vector,
                state_label=state_info.name,
                purity=state_info.purity
            )
            st.plotly_chart(fig_bloch, use_container_width=True)

        with col_right:
            st.markdown("#### 📊 Basis Measurement Probabilities")
            counts_ideal = {
                "0": int(state_info.prob_0 * 1000),
                "1": int(state_info.prob_1 * 1000)
            }
            fig_meas = plot_measurement_distribution(
                counts=counts_ideal,
                title=f"Theoretical Probabilities for {state_info.name}",
                expected_bit="0" if state_info.prob_0 >= state_info.prob_1 else "1"
            )
            st.plotly_chart(fig_meas, use_container_width=True)

        st.markdown("#### 🔌 Qiskit Circuit: State Preparation & Gate Operations")
        gate_col, circ_col = st.columns([1, 2])
        with gate_col:
            st.markdown("**Apply Additional Single-Qubit Gates:**")
            add_x = st.checkbox("Apply Pauli X (Bit Flip)", False)
            add_h = st.checkbox("Apply Hadamard H (Superposition)", False)
            add_z = st.checkbox("Apply Pauli Z (Phase Flip)", False)

            custom_gates = []
            if add_x: custom_gates.append("X")
            if add_h: custom_gates.append("H")
            if add_z: custom_gates.append("Z")

        with circ_col:
            qc_unprot = build_unprotected_circuit(
                state_info=state_info,
                custom_gates=custom_gates,
                channel_noise_placeholder=False
            )
            try:
                fig_circ = draw_circuit_mpl(qc_unprot, scale=1.1)
                st.pyplot(fig_circ)
            except Exception:
                st.code(draw_circuit_text(qc_unprot))

        render_explanation("logical_vs_physical")

    # -------------------------------------------------------------
    # TAB 2: NOISE EXPLORER
    # -------------------------------------------------------------
    with tab_noise:
        st.subheader("2. Quantum Noise Explorer: State Decoherence")
        st.markdown(
            "Observe what happens to a pure quantum state when physical noise is introduced into the quantum channel. "
            "Notice how noise causes the state vector to shrink **inside** the Bloch sphere, forming a **mixed state**."
        )

        n_col1, n_col2 = st.columns([1, 1])

        with n_col1:
            st.markdown("#### ⚙️ Channel Noise Settings")
            noise_channel = st.selectbox(
                "Select Noise Model:",
                options=["bit_flip", "phase_flip", "depolarizing"],
                format_func=lambda x: {
                    "bit_flip": "Bit-Flip Noise (Pauli X)",
                    "phase_flip": "Phase-Flip Noise (Pauli Z)",
                    "depolarizing": "Depolarizing Noise (Symmetric Decoherence)"
                }[x]
            )
            p_noise = st.slider(
                "Physical Error Probability (p):",
                min_value=0.0,
                max_value=0.50,
                value=0.15,
                step=0.01,
                format="%.2f"
            )
            st.info(get_noise_channel_explanation(noise_channel, p_noise))

        # Analytical noise transformation on density matrix
        rho_in = state_info.density_matrix
        rho_out, post_bloch, post_purity = apply_analytical_noise_channel(
            rho_in=rho_in,
            noise_type=noise_channel,
            p=p_noise
        )

        with n_col2:
            st.markdown("#### 🌐 Post-Noise Bloch Sphere (Mixed State)")
            fig_post_bloch = plot_bloch_sphere_3d(
                bloch_vector=post_bloch,
                state_label=f"Decohered (p={p_noise:.2f})",
                purity=post_purity
            )
            st.plotly_chart(fig_post_bloch, use_container_width=True)

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Physical Error Rate (p)", f"{p_noise * 100:.1f}%")
        m2.metric("State Purity Tr(ρ²)", f"{post_purity:.4f}", delta=f"{post_purity - 1.0:.4f}" if p_noise > 0 else "0.0")
        m3.metric("Bloch Radius |r|", f"{np.sqrt(sum(x**2 for x in post_bloch)):.4f}")
        m4.metric("State Class", "Mixed State" if post_purity < 0.999 else "Pure State")

        render_explanation("phase_flip_limitation")

    # -------------------------------------------------------------
    # TAB 3: INTERACTIVE QEC WALKTHROUGH
    # -------------------------------------------------------------
    with tab_walkthrough:
        st.subheader("3. Interactive Error Injection: Step-by-Step Walkthrough")
        st.markdown(
            "Deterministically inject specific errors into physical qubits and follow the "
            "exact sequence: **Encoding → Noise → Ancilla Syndrome Extraction → Diagnosis → Correction → Decoded State**."
        )

        step_col_ctl, step_col_vis = st.columns([1, 2])

        with step_col_ctl:
            st.markdown("#### 🎯 Inject Channel Errors")
            error_choice = st.radio(
                "Choose Error Scenario:",
                options=[
                    "none",
                    "flip_q0",
                    "flip_q1",
                    "flip_q2",
                    "flip_two",
                    "phase_q0"
                ],
                format_func=lambda x: {
                    "none": "No Error (Ideal Channel)",
                    "flip_q0": "Single Bit-Flip on Qubit 0",
                    "flip_q1": "Single Bit-Flip on Qubit 1",
                    "flip_q2": "Single Bit-Flip on Qubit 2",
                    "flip_two": "Two Bit-Flips (Qubits 0 & 1) [Code Failure Case]",
                    "phase_q0": "Phase-Flip (Z) on Qubit 0 [Code Limitation Case]"
                }[x]
            )

            walkthrough_result = run_deterministic_qec_walkthrough(state_info, error_choice)

            # Syndrome lookup status card
            s_s0, s_s1 = walkthrough_result.syndrome_bits
            st.markdown("---")
            st.markdown(f"**Extracted Syndrome:** `(s0={s_s0}, s1={s_s1})`")
            st.markdown(f"**Diagnosed Error:** {walkthrough_result.diagnosed_action}")
            st.markdown(f"**Final Verdict:** `{'PASS (Recovered)' if walkthrough_result.success else 'FAIL (Logical Error)'}`")

        with step_col_vis:
            st.markdown("#### 📋 Step-by-Step Pipeline Execution")
            for step in walkthrough_result.state_descriptions:
                step_title = step["step"]
                desc = step["description"]
                if "Fail" in desc or "Error Injected" in step_title or "FAIL" in desc:
                    css_class = "pipeline-step-err"
                elif "PASS" in desc or "Fidelity" in step_title:
                    css_class = "pipeline-step-success"
                else:
                    css_class = "pipeline-step"
                st.markdown(f'<div class="{css_class}"><b>{step_title}</b><br>{desc}</div>', unsafe_allow_html=True)

            if walkthrough_result.success:
                st.success(walkthrough_result.explanation)
            else:
                st.error(walkthrough_result.explanation)

        st.markdown("#### 🔌 Qiskit 3-Qubit Error Correction Circuit")
        det_errors_map = walkthrough_result.injected_errors
        qc_full = build_full_qec_circuit(state_info=state_info, deterministic_errors=det_errors_map)
        try:
            fig_qec_circ = draw_circuit_mpl(qc_full, scale=0.95)
            st.pyplot(fig_qec_circ)
        except Exception:
            st.code(draw_circuit_text(qc_full))

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            render_explanation("syndrome_measurement")
        with col_exp2:
            render_explanation("two_errors_breakdown")

    # -------------------------------------------------------------
    # TAB 4: QEC ON VS OFF (SIDE-BY-SIDE)
    # -------------------------------------------------------------
    with tab_compare:
        st.subheader("4. Error Correction ON vs. OFF: Head-to-Head Comparison")
        st.markdown(
            "Run an active Monte Carlo experiment comparing the unprotected single-qubit transmission "
            "against the protected 3-qubit repetition code at the exact same physical noise probability."
        )

        cmp_col1, cmp_col2, cmp_col3 = st.columns(3)
        with cmp_col1:
            comp_p = st.slider("Physical Error Probability p:", 0.0, 0.40, 0.10, 0.01, key="comp_p")
        with cmp_col2:
            comp_shots = st.selectbox("Monte Carlo Shots:", [500, 1000, 2000, 5000], index=2, key="comp_shots")
        with cmp_col3:
            comp_run = st.button("🚀 Run Live Experiment", use_container_width=True)

        if comp_run or "last_side_by_side" not in st.session_state:
            with st.spinner("Executing quantum circuits on AerSimulator..."):
                exp_res = run_side_by_side_experiment(
                    state_info=state_info,
                    noise_type="bit_flip",
                    p=comp_p,
                    shots=comp_shots
                )
                st.session_state["last_side_by_side"] = exp_res
        else:
            exp_res = st.session_state["last_side_by_side"]

        # Results Summary Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Physical Noise (p)", f"{exp_res.physical_p * 100:.1f}%")
        c2.metric("Unprotected Error Rate", f"{exp_res.unprotected_error_rate * 100:.2f}%")
        c3.metric("Logical QEC Error Rate", f"{exp_res.logical_error_rate * 100:.2f}%")
        advantage = (
            exp_res.unprotected_error_rate / exp_res.logical_error_rate
            if exp_res.logical_error_rate > 0 else (1.0 if exp_res.unprotected_error_rate == 0 else 999.0)
        )
        c4.metric("Error Reduction Advantage", f"{advantage:.2f}x" if advantage < 100 else ">100x")

        # Visual charts
        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown("#### 📊 Unprotected Channel Measurement")
            fig_unp = plot_measurement_distribution(
                counts=exp_res.unprotected_counts,
                title="Unprotected Readout (Error Correction OFF)",
                expected_bit=str(exp_res.target_bit)
            )
            st.plotly_chart(fig_unp, use_container_width=True)

        with ch_col2:
            st.markdown("#### 📊 Extracted Ancilla Syndromes (Error Correction ON)")
            fig_syn = plot_syndrome_bar_chart(exp_res.syndrome_distribution)
            st.plotly_chart(fig_syn, use_container_width=True)

        render_explanation("protected_curve_shape")

    # -------------------------------------------------------------
    # TAB 5: THEORY VS SIMULATION BENCHMARK
    # -------------------------------------------------------------
    with tab_benchmark:
        st.subheader("5. Theory vs. Simulation: Multi-Point Error Sweep")
        st.markdown(
            "Sweep physical noise probability $p \\in [0.0, 0.50]$ to compare simulated Monte Carlo "
            "frequencies directly with mathematical error correction theory: "
            "$$P_{\\text{unprot}} = p, \\quad P_L = 3p^2 - 2p^3$$"
        )

        b_col_ctl, b_col_info = st.columns([1, 2])
        with b_col_ctl:
            sweep_shots = st.select_slider(
                "Simulation Shots per Point:",
                options=[500, 1000, 2000, 5000],
                value=2000
            )
            run_sweep_btn = st.button("📊 Execute Full Parameter Sweep", use_container_width=True)

        # Cache or load pre-calculated / current benchmark
        benchmark_csv_path = os.path.join("results", "experiment_results.csv")
        if run_sweep_btn:
            with st.spinner("Running Monte Carlo sweep across 13 probability points on AerSimulator..."):
                probs = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                df_sweep = run_comparison_sweep(
                    state_info=state_info,
                    probabilities=probs,
                    shots=sweep_shots
                )
                st.session_state["benchmark_df"] = df_sweep
        elif "benchmark_df" in st.session_state:
            df_sweep = st.session_state["benchmark_df"]
        elif os.path.exists(benchmark_csv_path):
            df_sweep = pd.read_csv(benchmark_csv_path)
        else:
            probs = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
            df_sweep = run_comparison_sweep(state_info=state_info, probabilities=probs, shots=1000)

        # Interactive Plotly Curve
        st.plotly_chart(plot_interactive_benchmark(df_sweep), use_container_width=True)

        # Data table and CSV export
        with st.expander("📄 View Numerical Benchmark Table & Download CSV", expanded=False):
            st.dataframe(df_sweep.style.format({
                "physical_p": "{:.3f}",
                "unprotected_simulated": "{:.4f}",
                "logical_simulated": "{:.4f}",
                "unprotected_theory": "{:.4f}",
                "logical_theory": "{:.4f}",
                "improvement_factor": "{:.2f}x"
            }), use_container_width=True)
            csv_data = df_sweep.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Experiment Data as CSV",
                data=csv_data,
                file_name="qubitlab_benchmark_results.csv",
                mime="text/csv"
            )

        render_explanation("no_cloning_theorem")

    # -------------------------------------------------------------
    # TAB 6: SINGLE-TRIAL INSPECTOR
    # -------------------------------------------------------------
    with tab_trials:
        st.subheader("6. Single-Trial Inspector: Inspect Individual Quantum Shots")
        st.markdown(
            "Inspect an individual simulated quantum shot in slow motion. "
            "Experience how independent physical bit flips occur stochastically and how ancilla parity checks pinpoint errors."
        )

        tr_col1, tr_col2 = st.columns([1, 2])
        with tr_col1:
            p_trial = st.slider("Physical Error Probability (p):", 0.01, 0.50, 0.15, 0.01, key="p_trial")
            sample_btn = st.button("🎲 Sample Random Shot", use_container_width=True)

        if sample_btn or "last_trial_trace" not in st.session_state:
            trial_trace = simulate_single_trial(state_info=state_info, p=p_trial)
            st.session_state["last_trial_trace"] = trial_trace
        else:
            trial_trace = st.session_state["last_trial_trace"]

        with tr_col2:
            st.markdown(f"#### 🧬 Trial Lifecycle Trace (Physical p = {trial_trace.physical_p:.2f})")
            st.markdown(f"""
            - **Original State:** `{trial_trace.input_state}`
            - **Encoded Logical State:** `{trial_trace.encoded_state}`
            - **Physical Noise Injected:** `{len(trial_trace.physical_errors_injected)} error(s)` on physical qubit(s) `{trial_trace.physical_errors_injected}`
            - **Corrupted Physical State:** `{trial_trace.corrupted_physical_state}`
            - **Measured Syndrome:** `(s0={trial_trace.syndrome_bits[0]}, s1={trial_trace.syndrome_bits[1]})`
            - **Diagnosed Error:** `{trial_trace.diagnosed_error}`
            - **Corrective Action Applied:** `{trial_trace.corrective_action}`
            - **Corrected Physical State:** `{trial_trace.corrected_physical_state}`
            - **Decoded Output Bit:** `|{trial_trace.decoded_bit}⟩` (Expected: `|{trial_trace.expected_bit}⟩`)
            """)

            if trial_trace.success:
                st.success(f"**SUCCESS:** {trial_trace.summary}")
            else:
                st.error(f"**FAILURE:** {trial_trace.summary}")

    # -------------------------------------------------------------
    # TAB 7: PREDICT & VERIFY (MINI-CHALLENGES)
    # -------------------------------------------------------------
    with tab_challenges:
        st.subheader("7. Predict the Result: Quantum Error Correction Experiments")
        st.markdown(
            "Test your intuition about how quantum repetition codes respond to errors. "
            "Select your prediction, execute the experiment, and examine the underlying quantum mechanics."
        )

        ch_option = st.selectbox(
            "Select an Experiment Scenario:",
            options=[
                "challenge_1",
                "challenge_2",
                "challenge_3"
            ],
            format_func=lambda x: {
                "challenge_1": "Experiment 1: Single X Error on Qubit 2",
                "challenge_2": "Experiment 2: Two X Errors on Qubits 0 and 1",
                "challenge_3": "Experiment 3: Phase-Flip (Z) Error on Qubit 0"
            }[x]
        )

        if ch_option == "challenge_1":
            st.markdown("**Question:** *A logical state $|0_L\\rangle = |000\\rangle$ is encoded. An X error strikes physical Qubit 2 ($|001\\rangle$). Will the 3-qubit repetition code recover the original state?*")
            pred1 = st.radio("Your Prediction:", ["Yes, it will recover the original state.", "No, it will suffer a logical error.", "The state will be destroyed."], key="pred1")
            if st.button("Run Experiment 1", key="btn_ch1"):
                res1 = run_deterministic_qec_walkthrough(get_quantum_state("|0>"), error_mode="flip_q2")
                if "Yes" in pred1:
                    st.success("🎯 **Correct!** " + res1.explanation)
                else:
                    st.info("💡 " + res1.explanation)

        elif ch_option == "challenge_2":
            st.markdown("**Question:** *A logical state $|0_L\\rangle = |000\\rangle$ experiences TWO bit flips on Qubits 0 and 1 ($|110\\rangle$). What will happen after syndrome extraction and correction?*")
            pred2 = st.radio("Your Prediction:", ["The code detects 2 errors and corrects both.", "The code misidentifies Qubit 2, resulting in a logical error.", "The code flags an alarm and halts."], key="pred2")
            if st.button("Run Experiment 2", key="btn_ch2"):
                res2 = run_deterministic_qec_walkthrough(get_quantum_state("|0>"), error_mode="flip_two")
                if "misidentifies" in pred2:
                    st.success("🎯 **Correct!** " + res2.explanation)
                else:
                    st.info("💡 " + res2.explanation)

        elif ch_option == "challenge_3":
            st.markdown("**Question:** *We prepare $|+\\rangle$ and encode it into $|+_L\\rangle = \\frac{|000\\rangle + |111\\rangle}{\\sqrt{2}}$. A Pauli Z error occurs on Qubit 0. What will the parity syndrome be?*")
            pred3 = st.radio("Your Prediction:", ["Syndrome is (1, 0) and identifies Z.", "Syndrome is (0, 0) and the error passes undetected.", "The ancilla qubits collapse into |1>."], key="pred3")
            if st.button("Run Experiment 3", key="btn_ch3"):
                res3 = run_deterministic_qec_walkthrough(get_quantum_state("|+>"), error_mode="phase_q0")
                if "undetected" in pred3:
                    st.success("🎯 **Correct!** " + res3.explanation)
                else:
                    st.info("💡 " + res3.explanation)

    # -------------------------------------------------------------
    # TAB 8: KNOWLEDGE BASE
    # -------------------------------------------------------------
    with tab_learn:
        st.subheader("8. Scientific Reference & Theoretical Principles")
        st.markdown(
            "Detailed explanations of the core quantum information and error correction principles "
            "implemented inside QubitLab."
        )

        render_explanation("logical_vs_physical", expanded=True)
        render_explanation("syndrome_measurement", expanded=True)
        render_explanation("protected_curve_shape", expanded=True)
        render_explanation("two_errors_breakdown", expanded=True)
        render_explanation("no_cloning_theorem", expanded=True)
        render_explanation("phase_flip_limitation", expanded=True)
        render_explanation("fault_tolerant_reality", expanded=True)


if __name__ == "__main__":
    main_app()
