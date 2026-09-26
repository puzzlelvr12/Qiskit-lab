"""
Interactive Visualizations Module for QubitLab.

Provides interactive Plotly visualizations:
1. 3D Bloch Sphere supporting pure and mixed states with purity indicators
2. Measurement probability distributions
3. Interactive multi-point benchmark curves (Theory vs Simulation)
4. Syndrome extraction distribution charts
5. Pedagogical HTML pipeline visual cards
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_bloch_sphere_3d(
    bloch_vector: Tuple[float, float, float],
    state_label: str = "|ψ⟩",
    purity: float = 1.0,
    height: int = 420
) -> go.Figure:
    """
    Renders a 3D interactive Bloch sphere in Plotly.
    Accurately supports mixed states: if purity < 1, the vector sits inside
    the sphere and an informational annotation indicates state mixture.
    """
    rx, ry, rz = bloch_vector
    vec_len = np.sqrt(rx**2 + ry**2 + rz**2)
    is_mixed = purity < 0.999 or vec_len < 0.999

    # Generate wireframe grid for unit sphere
    phi = np.linspace(0, 2 * np.pi, 30)
    theta = np.linspace(0, np.pi, 20)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    xs = np.sin(theta_grid) * np.cos(phi_grid)
    ys = np.sin(theta_grid) * np.sin(phi_grid)
    zs = np.cos(theta_grid)

    fig = go.Figure()

    # 1. Translucent Bloch sphere surface
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.15,
        colorscale=[[0, "#3b82f6"], [1, "#1d4ed8"]],
        showscale=False,
        hoverinfo="skip"
    ))

    # 2. Equator ring (xy plane, z=0)
    t = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter3d(
        x=np.cos(t), y=np.sin(t), z=np.zeros_like(t),
        mode="lines",
        line=dict(color="#94a3b8", width=2, dash="dot"),
        name="Equator",
        hoverinfo="skip"
    ))

    # 3. Coordinate axes
    axis_coords = [
        ([-1.2, 1.2], [0, 0], [0, 0], "X Axis"),
        ([0, 0], [-1.2, 1.2], [0, 0], "Y Axis"),
        ([0, 0], [0, 0], [-1.2, 1.2], "Z Axis"),
    ]
    for x_c, y_c, z_c, a_name in axis_coords:
        fig.add_trace(go.Scatter3d(
            x=x_c, y=y_c, z=z_c,
            mode="lines",
            line=dict(color="#cbd5e1", width=2),
            showlegend=False,
            hoverinfo="skip"
        ))

    # 4. Standard basis reference points
    ref_points = [
        (0, 0, 1.05, "|0⟩ (Z+)"),
        (0, 0, -1.05, "|1⟩ (Z-)"),
        (1.05, 0, 0, "|+⟩ (X+)"),
        (-1.05, 0, 0, "|-⟩ (X-)"),
        (0, 1.05, 0, "|i⟩ (Y+)"),
        (0, -1.05, 0, "|-i⟩ (Y-)"),
    ]
    for px, py, pz, lbl in ref_points:
        fig.add_trace(go.Scatter3d(
            x=[px], y=[py], z=[pz],
            mode="text",
            text=[lbl],
            textposition="middle center",
            textfont=dict(size=11, color="#475569"),
            showlegend=False,
            hoverinfo="skip"
        ))

    # 5. Bloch vector line
    vec_color = "#dc2626" if is_mixed else "#2563eb"
    fig.add_trace(go.Scatter3d(
        x=[0, rx], y=[0, ry], z=[0, rz],
        mode="lines",
        line=dict(color=vec_color, width=6),
        name=f"Bloch Vector ({state_label})",
        hoverinfo="skip"
    ))

    # 6. Bloch vector tip marker
    tip_hover = (
        f"<b>State:</b> {state_label}<br>"
        f"<b>Vector:</b> ({rx:.3f}, {ry:.3f}, {rz:.3f})<br>"
        f"<b>Radius |r|:</b> {vec_len:.3f}<br>"
        f"<b>Purity Tr(ρ²):</b> {purity:.3f}<br>"
        f"<b>Class:</b> {'Mixed State (Decohered)' if is_mixed else 'Pure State'}"
    )
    fig.add_trace(go.Scatter3d(
        x=[rx], y=[ry], z=[rz],
        mode="markers+text",
        marker=dict(size=8, color=vec_color, symbol="diamond"),
        text=[f" {state_label}"],
        textposition="top right",
        textfont=dict(size=12, color=vec_color),
        name="State Vector Tip",
        hovertext=[tip_hover],
        hoverinfo="text"
    ))

    # Scene layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        height=height,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="X"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Y"),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Z"),
            camera=dict(eye=dict(x=1.45, y=1.35, z=1.15)),
            aspectmode="cube"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_measurement_distribution(
    counts: Dict[str, int],
    title: str = "Measurement Outcome Distribution",
    expected_bit: Optional[str] = None
) -> go.Figure:
    """
    Renders an interactive bar chart of measurement counts.
    """
    total = sum(counts.values()) if counts else 1
    # Sort keys for consistent presentation ('0' before '1')
    sorted_keys = sorted(counts.keys())
    values = [counts[k] for k in sorted_keys]
    percents = [(v / total) * 100.0 for v in values]

    colors = []
    for k in sorted_keys:
        if expected_bit is not None:
            colors.append("#10b981" if k == expected_bit else "#ef4444")
        else:
            colors.append("#3b82f6")

    hover_texts = [
        f"Outcome |{k}⟩<br>Count: {v}<br>Frequency: {p:.1f}%"
        for k, v, p in zip(sorted_keys, values, percents)
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=[f"|{k}⟩" for k in sorted_keys],
            y=values,
            text=[f"{p:.1f}%" for p in percents],
            textposition="auto",
            marker=dict(color=colors, line=dict(color="#1e293b", width=1)),
            hovertext=hover_texts,
            hoverinfo="text"
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1e293b")),
        xaxis=dict(title="Basis State"),
        yaxis=dict(title="Measured Counts"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_interactive_benchmark(df: pd.DataFrame) -> go.Figure:
    """
    Renders an interactive Plotly plot comparing simulated vs theoretical error rates
    across physical bit-flip error probabilities p in [0.0, 0.5].
    """
    fig = go.Figure()

    p_dense = np.linspace(0.0, 0.50, 150)
    pl_theory_dense = 3.0 * (p_dense ** 2) - 2.0 * (p_dense ** 3)

    # 1. Unprotected Theory line
    fig.add_trace(go.Scatter(
        x=p_dense, y=p_dense,
        mode="lines",
        line=dict(color="#ef4444", width=2, dash="dash"),
        name="Unprotected Theory (P = p)",
        hoverinfo="skip"
    ))

    # 2. 3-Qubit Repetition Code Theory curve
    fig.add_trace(go.Scatter(
        x=p_dense, y=pl_theory_dense,
        mode="lines",
        line=dict(color="#2563eb", width=3),
        name="3-Qubit QEC Theory (P_L = 3p² - 2p³)",
        hoverinfo="skip"
    ))

    # 3. Unprotected Simulation Data Points
    fig.add_trace(go.Scatter(
        x=df["physical_p"],
        y=df["unprotected_simulated"],
        mode="markers",
        marker=dict(color="#ef4444", size=8, symbol="circle"),
        name=f"Unprotected Sim (N={df['shots'].iloc[0]})",
        hovertemplate="<b>Unprotected Simulated</b><br>p: %{x:.3f}<br>Error: %{y:.4f}<extra></extra>"
    ))

    # 4. QEC Logical Simulation Data Points
    fig.add_trace(go.Scatter(
        x=df["physical_p"],
        y=df["logical_simulated"],
        mode="markers",
        marker=dict(color="#2563eb", size=9, symbol="square"),
        name=f"3-Qubit QEC Sim (N={df['shots'].iloc[0]})",
        hovertemplate="<b>Logical QEC Simulated</b><br>p: %{x:.3f}<br>Logical Error: %{y:.4f}<extra></extra>"
    ))

    # Pseudothreshold vertical line at p = 0.50
    fig.add_vline(
        x=0.50,
        line_width=1.5,
        line_dash="dot",
        line_color="#64748b",
        annotation_text="Pseudothreshold p = 0.50",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#64748b")
    )

    fig.update_layout(
        title=dict(
            text="Physical Error Probability vs. Logical Error Probability",
            font=dict(size=15, color="#0f172a", family="Inter, sans-serif")
        ),
        xaxis=dict(
            title="Physical Bit-Flip Error Probability (p)",
            range=[-0.01, 0.52],
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1"
        ),
        yaxis=dict(
            title="Observed / Predicted Error Probability",
            range=[-0.01, 0.55],
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=50, r=20, t=70, b=50),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_syndrome_bar_chart(syndrome_counts: Dict[str, int]) -> go.Figure:
    """
    Renders an interactive distribution of extracted syndrome measurements.
    """
    syndrome_labels = {
        "00": "00: No Error",
        "01": "01: Q2 Flipped",
        "10": "10: Q0 Flipped",
        "11": "11: Q1 Flipped",
    }
    categories = ["00", "10", "11", "01"]
    labels = [syndrome_labels.get(k, k) for k in categories]
    counts = [syndrome_counts.get(k, 0) for k in categories]
    total = sum(counts) if sum(counts) > 0 else 1
    percents = [(c / total) * 100.0 for c in counts]

    colors = ["#10b981", "#3b82f6", "#6366f1", "#8b5cf6"]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=counts,
            text=[f"{p:.1f}% ({c})" for p, c in zip(percents, counts)],
            textposition="auto",
            marker=dict(color=colors, line=dict(color="#1e293b", width=1)),
            hoverinfo="x+y"
        )
    ])

    fig.update_layout(
        title=dict(text="Syndrome Measurement Distribution", font=dict(size=14, color="#1e293b")),
        xaxis=dict(title="Extracted Syndrome (s0 s1)"),
        yaxis=dict(title="Occurrences"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
