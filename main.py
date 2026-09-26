"""
QubitLab: Main Entrypoint & Command-Line Benchmark Runner

Runs quantum noise simulations, computes theoretical models, and exports
experiment results to CSV and high-resolution plots.
"""

from __future__ import annotations
import argparse
import os
import sys

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.states import get_quantum_state
from src.simulation import run_comparison_sweep, simulate_single_trial
from src.error_correction import run_deterministic_qec_walkthrough
from src.analysis import theoretical_logical_error_repetition_3, compute_standard_error


def run_cli_demo():
    """Runs a terminal walkthrough of single error correction and failure modes."""
    print("=" * 70)
    print(" QubitLab: 3-Qubit Quantum Repetition Code Walkthrough Demo")
    print("=" * 70)

    s0 = get_quantum_state("|0>")
    print(f"\n1. Initial State: {s0.name} (|ψ⟩ = {s0.alpha.real:.1f}|0⟩ + {s0.beta.real:.1f}|1⟩)")

    print("\n--- Scenario A: Single Bit-Flip Error on Qubit 1 ---")
    res_a = run_deterministic_qec_walkthrough(s0, error_mode="flip_q1")
    for step in res_a.state_descriptions:
        print(f"  {step['step']}: {step['description']}")
    print(f"  Result: {res_a.explanation}")

    print("\n--- Scenario B: Two Bit-Flip Errors on Qubits 0 & 1 ---")
    res_b = run_deterministic_qec_walkthrough(s0, error_mode="flip_two")
    for step in res_b.state_descriptions:
        print(f"  {step['step']}: {step['description']}")
    print(f"  Result: {res_b.explanation}")

    print("\n--- Scenario C: Phase-Flip (Z) Error Limitation ---")
    sp = get_quantum_state("|+>")
    res_c = run_deterministic_qec_walkthrough(sp, error_mode="phase_q0")
    for step in res_c.state_descriptions:
        print(f"  {step['step']}: {step['description']}")
    print(f"  Result: {res_c.explanation}")
    print("=" * 70)


def run_benchmark(shots: int = 5000, output_dir: str = "results"):
    """
    Executes a comprehensive sweep across physical bit-flip noise probabilities,
    saves the benchmark data to CSV, and generates an error-rate comparison figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[QubitLab] Starting Qiskit Aer Monte Carlo simulation ({shots} shots/point)...")

    probabilities = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    s0 = get_quantum_state("|0>")
    df = run_comparison_sweep(
        state_info=s0,
        probabilities=probabilities,
        noise_type="bit_flip",
        shots=shots,
        seed=42
    )

    # Compute standard errors for error bars
    df["unprot_se"] = [compute_standard_error(r, shots) for r in df["unprotected_simulated"]]
    df["logical_se"] = [compute_standard_error(r, shots) for r in df["logical_simulated"]]

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'p (Physical)':<12} {'Unprot Sim':<14} {'Unprot Theory':<14} {'QEC Sim':<14} {'QEC Theory':<14} {'Advantage':<10}")
    print("-" * 80)
    for _, row in df.iterrows():
        adv = f"{row['improvement_factor']:.2f}x" if row['improvement_factor'] < 100 else ">100x"
        print(f"{row['physical_p']:<12.3f} {row['unprotected_simulated']:<14.4f} {row['unprotected_theory']:<14.4f} "
              f"{row['logical_simulated']:<14.4f} {row['logical_theory']:<14.4f} {adv:<10}")
    print("=" * 80)

    # Save to CSV
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Experiment results written to: {csv_path}")

    # Generate Matplotlib publication figure
    plot_path = os.path.join(output_dir, "error_rates.png")
    generate_benchmark_plot(df, plot_path)
    print(f"[Saved] Error rates plot generated at: {plot_path}\n")


def generate_benchmark_plot(df: pd.DataFrame, save_path: str):
    """Creates a high-resolution comparison plot comparing theory and simulation."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    p_dense = np.linspace(0.0, 0.5, 200)
    pl_theory_dense = 3.0 * (p_dense ** 2) - 2.0 * (p_dense ** 3)

    # Theoretical curves
    ax.plot(p_dense, p_dense, label="Unprotected Theory ($P_{unprot} = p$)", color="#d9534f", linestyle="--", linewidth=1.8, alpha=0.85)
    ax.plot(p_dense, pl_theory_dense, label="3-Qubit QEC Theory ($P_L = 3p^2 - 2p^3$)", color="#0275d8", linestyle="-", linewidth=2.2)

    # Simulation data points with error bars
    ax.errorbar(
        df["physical_p"],
        df["unprotected_simulated"],
        yerr=df["unprot_se"],
        fmt="o",
        color="#d9534f",
        ecolor="#d9534f",
        elinewidth=1.2,
        capsize=3,
        markersize=6,
        label=f"Unprotected Sim (N={df['shots'].iloc[0]})",
        alpha=0.9
    )

    ax.errorbar(
        df["physical_p"],
        df["logical_simulated"],
        yerr=df["logical_se"],
        fmt="s",
        color="#0275d8",
        ecolor="#0275d8",
        elinewidth=1.2,
        capsize=3,
        markersize=6,
        label=f"3-Qubit QEC Sim (N={df['shots'].iloc[0]})",
        alpha=0.9
    )

    # Threshold marker
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.6)
    ax.text(0.485, 0.25, "Pseudothreshold $p = 0.50$", rotation=90, verticalalignment="center", fontsize=9, color="#555")

    # Annotations
    ax.annotate(
        "Quadratic Suppression\n($P_L \\approx 3p^2$ for small $p$)",
        xy=(0.10, 0.028),
        xytext=(0.18, 0.08),
        arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
        fontsize=9,
        fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eef", ec="#0275d8", lw=0.8)
    )

    ax.set_title("Quantum Error Correction Performance: Unprotected vs. 3-Qubit Repetition Code", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Physical Bit-Flip Error Probability ($p$)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Observed / Predicted Error Probability", fontsize=11, fontweight="medium")
    ax.set_xlim(-0.01, 0.52)
    ax.set_ylim(-0.01, 0.55)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="QubitLab: Interactive Quantum Noise & Error Correction Playground")
    parser.add_argument("--demo", action="store_true", help="Run terminal walkthrough of error correction scenarios")
    parser.add_argument("--benchmark", action="store_true", help="Run full simulation benchmark and save results")
    parser.add_argument("--shots", type=int, default=5000, help="Number of Monte Carlo shots per probability point (default: 5000)")
    args = parser.parse_args()

    if args.demo:
        run_cli_demo()
    else:
        # Default behavior: run benchmark and demo
        run_cli_demo()
        run_benchmark(shots=args.shots)


if __name__ == "__main__":
    main()
