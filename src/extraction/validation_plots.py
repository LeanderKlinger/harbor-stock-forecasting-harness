# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false
"""Plotly visualizations for the validation harness. Separated to isolate plotly type stub gaps."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _apply_theme(fig: go.Figure) -> None:
    """Clean minimal theme matching visualization.py conventions."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", color="#1a1a1a", size=12),
        margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.06)",
            linecolor="rgba(0,0,0,0.12)",
            zerolinecolor="rgba(0,0,0,0.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.06)",
            linecolor="rgba(0,0,0,0.12)",
            zerolinecolor="rgba(0,0,0,0.12)",
        ),
    )


def plot_distribution(
    scores: list[float],
    diagnostics: dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    """Score distribution histogram with neutral zone and effective range markers."""
    arr = np.array(scores)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=arr,
            nbinsx=40,
            name="Score Distribution",
            marker_color="rgba(59, 130, 246, 0.6)",
            marker_line=dict(color="rgba(59, 130, 246, 0.8)", width=1),
        )
    )

    ncr = diagnostics["neutral_collapse_rate"]
    p5 = diagnostics["effective_range_low"]
    p95 = diagnostics["effective_range_high"]

    fig.update_layout(
        title=f"{title}<br><sub>Neutral collapse: {ncr:.1%} | Effective range: [{p5:.2f}, {p95:.2f}]</sub>",
        xaxis_title="Sentiment Score",
        yaxis_title="Count",
        showlegend=False,
        xaxis=dict(range=[-1.05, 1.05]),
    )

    # Neutral zone shading
    fig.add_vrect(
        x0=-0.1,
        x1=0.1,
        fillcolor="rgba(0,0,0,0.05)",
        line_width=0,
        annotation_text="neutral",
        annotation_position="top",
    )

    # Effective range boundaries
    fig.add_vline(x=p5, line_dash="dash", line_color="rgba(0,0,0,0.3)")
    fig.add_vline(x=p95, line_dash="dash", line_color="rgba(0,0,0,0.3)")

    _apply_theme(fig)

    if ncr > 0.7:
        fig.add_annotation(
            x=0,
            y=1.05,
            xref="paper",
            yref="paper",
            text="WARNING: Neutral collapse rate > 70%",
            showarrow=False,
            font=dict(color="#dc2626", size=14),
        )

    fig.write_image(str(output_path), width=900, height=500, scale=2)


def plot_bland_altman(
    scores_a: list[float],
    scores_b: list[float],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    """Bland-Altman plot: mean vs difference with limits of agreement."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    means = (a + b) / 2
    diffs = a - b

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=means,
            y=diffs,
            mode="markers",
            marker=dict(color="rgba(59, 130, 246, 0.5)", size=5),
            name="Articles",
        )
    )

    fig.add_hline(
        y=mean_diff,
        line_dash="solid",
        line_color="#1a1a1a",
        annotation_text=f"Mean: {mean_diff:.3f}",
    )
    fig.add_hline(
        y=upper_loa,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"+1.96\u03c3: {upper_loa:.3f}",
    )
    fig.add_hline(
        y=lower_loa,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"-1.96\u03c3: {lower_loa:.3f}",
    )

    fig.update_layout(
        title=f"Bland-Altman: {label_a} vs {label_b}",
        xaxis_title="Mean Score",
        yaxis_title=f"Difference ({label_a} \u2212 {label_b})",
    )

    _apply_theme(fig)
    fig.write_image(str(output_path), width=900, height=600, scale=2)


def plot_scatter(
    scores_a: list[float],
    scores_b: list[float],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    """Scatter plot with identity line."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scores_a,
            y=scores_b,
            mode="markers",
            marker=dict(color="rgba(59, 130, 246, 0.5)", size=5),
            name="Articles",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[-1, 1],
            y=[-1, 1],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.3)", dash="dash"),
            name="y = x",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Score Agreement: {label_a} vs {label_b}",
        xaxis_title=label_a,
        yaxis_title=label_b,
        xaxis=dict(range=[-1.05, 1.05]),
        yaxis=dict(range=[-1.05, 1.05]),
    )

    _apply_theme(fig)
    fig.write_image(str(output_path), width=700, height=700, scale=2)


def _discretize(scores: list[float]) -> list[str]:
    """Discretize scores into negative/neutral/positive."""
    labels: list[str] = []
    for s in scores:
        if s < -0.1:
            labels.append("negative")
        elif s > 0.1:
            labels.append("positive")
        else:
            labels.append("neutral")
    return labels


def plot_confusion_matrix(
    scores_a: list[float],
    scores_b: list[float],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    """3x3 confusion matrix heatmap for direction agreement."""
    cats = ["negative", "neutral", "positive"]
    labels_a = _discretize(scores_a)
    labels_b = _discretize(scores_b)

    matrix = np.zeros((3, 3), dtype=int)
    for la, lb in zip(labels_a, labels_b):
        i = cats.index(la)
        j = cats.index(lb)
        matrix[i, j] += 1

    text = [[str(matrix[i, j]) for j in range(3)] for i in range(3)]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=cats,
            y=cats,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )

    fig.update_layout(
        title=f"Direction Agreement: {label_a} vs {label_b}",
        xaxis_title=label_b,
        yaxis_title=label_a,
    )

    _apply_theme(fig)
    fig.write_image(str(output_path), width=600, height=600, scale=2)


def plot_violins(
    all_scores: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Side-by-side violin plots for 3+ experiments."""
    fig = go.Figure()

    colors = [
        "rgba(59, 130, 246, 0.6)",
        "rgba(239, 68, 68, 0.6)",
        "rgba(34, 197, 94, 0.6)",
        "rgba(168, 85, 247, 0.6)",
        "rgba(249, 115, 22, 0.6)",
        "rgba(20, 184, 166, 0.6)",
    ]

    for i, (label, scores) in enumerate(all_scores.items()):
        fig.add_trace(
            go.Violin(
                y=scores,
                name=label,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                line_color="rgba(0,0,0,0.5)",
            )
        )

    fig.update_layout(
        title="Score Distributions Comparison",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.05, 1.05]),
        showlegend=True,
    )

    _apply_theme(fig)
    fig.write_image(
        str(output_path), width=max(600, 200 * len(all_scores)), height=600, scale=2
    )
