# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false
# (plotly ships no type stubs — same isolation pragma as src/extraction/validation_plots.py)
"""Seed-variance figures for the ICAIF paper (data/system/figures/).

Regenerates the poster's seed-variance chart (data/system/figures/seed_variance.png,
built from April data) from the canonical `big_eval_canon` runs at the headline
environment, for both rebalance horizons plus a fused two-panel comparison:

  seed_variance.png         — h=20 single panel
  seed_variance_h1.png      — h=1 single panel (sign flips -> y=0 line)
  seed_variance_h1_h20.png  — h=1 | h=20 side by side, shared y-axis

Left block per panel: deterministic models as single diamond markers. Right
block: neural models as 5-seed box plots with all seed points overlaid.
Whiskers span MIN-MAX (fences precomputed) so no seed point ever detaches
from its box. Horizontal dashed reference line at the Buy & Hold Sortino.

Run: uv run python scripts/paper/generate_seed_variance_figure.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots  # pyright: ignore[reportUnknownVariableType]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import RUNS_INDEX_PATH

Run = dict[str, Any]

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "STONKICAIF"
SINGLE_OUTPUTS = {
    20: OUTPUT_DIR / "seed_variance.png",
    1: OUTPUT_DIR / "seed_variance_h1.png",
}
FUSED_OUTPUT = OUTPUT_DIR / "seed_variance_h1_h20.png"
SQUISHED_OUTPUT = OUTPUT_DIR / "seed_variance_h1_h20_squished.png"

CANON_EXPERIMENT = "big_eval_canon"

# Headline environment (canon cell the paper quotes throughout)
HEADLINE_COST = 0.001
HEADLINE_CONVICTION = "volatility_scaled"
HEADLINE_MIN_PREDICTED_RETURN = 0.01
HEADLINE_EMA_SPANS = [5, 10, 20]
HORIZONS = (1, 20)

# ARIMA appears at h=20 only — its h=1 auto-cadence cell is excluded from canon
# (deterministic LU failure), so h=1 panels render 4 diamonds.
DETERMINISTIC_MODELS = ["HoltsLES", "ARIMA", "Momentum", "BuyAndHold", "EqualWeight"]
NEURAL_MODELS = ["MLP", "xLSTM", "GRU", "Mamba", "Jamba"]
SEEDS_PER_CELL = 5

# Sanity gate: expected (min, max) Sortino per model per horizon in the
# headline env. Guards against silently plotting the wrong slice of the index.
EXPECTED_RANGES: dict[int, dict[str, tuple[float, float]]] = {
    20: {
        "HoltsLES": (1.55, 1.63),
        "ARIMA": (1.34, 1.42),
        "Momentum": (1.24, 1.32),
        "BuyAndHold": (1.16, 1.24),
        "EqualWeight": (1.13, 1.21),
        "MLP": (0.42, 1.28),
        "xLSTM": (0.73, 1.33),
        "GRU": (0.92, 1.60),
        "Mamba": (0.45, 1.10),
        "Jamba": (0.63, 1.15),
    },
    1: {
        "HoltsLES": (0.18, 0.26),
        "Momentum": (1.26, 1.34),
        "BuyAndHold": (1.16, 1.24),
        "EqualWeight": (1.09, 1.17),
        "MLP": (-0.74, 1.26),
        "xLSTM": (0.07, 1.14),
        "GRU": (0.28, 1.21),
        "Mamba": (-0.14, 0.72),
        "Jamba": (0.05, 0.75),
    },
}

# Muted pastel palette carried over from the poster figure — color follows the
# model identity (not its rank), so models keep their poster hues even though
# the descending-mean order changed under canon data.
MODEL_COLORS = {
    "HoltsLES": "#E07B54",
    "ARIMA": "#C08CA8",
    "Momentum": "#F0B429",
    "BuyAndHold": "#8A9BB8",
    "EqualWeight": "#B0B7BF",
    "MLP": "#AECBBA",
    "xLSTM": "#A99FD1",
    "GRU": "#7FBCB4",
    "Mamba": "#EBD3B3",
    "Jamba": "#DBA976",
}

MARKER_OUTLINE_COLOR = "#4A5568"
REFERENCE_LINE_COLOR = "#8A9BB8"
ZERO_LINE_COLOR = "#9AA1A9"
SEPARATOR_COLOR = "#C4CAD1"
BLOCK_LABEL_COLOR = "#8A8F98"
AXIS_TEXT_COLOR = "#2D3239"
GRID_COLOR = "#EDEFF2"

DIAMOND_SIZE = 22
POINT_SIZE = 13
BOX_WIDTH = 0.5
# B&H label ends just left of the block separator — the only spot above the
# reference line that is empty in every panel (deterministic diamonds crowd
# the line on the left; neural whiskers cross it on the right)
BH_LABEL_GAP = 0.15
# deterministic pseudo-jitter for the 5 overlaid seed points (sorted order)
POINT_JITTER = (-0.09, 0.05, -0.04, 0.09, 0.0)

FIG_WIDTH_PX = 1800
FIG_HEIGHT_PX = 900
EXPORT_SCALE = 2  # -> 3600x1800 (~300+ DPI at paper-column width)
# 20 (not 24) since ARIMA makes 5 long deterministic labels share the block
SINGLE_TICK_FONT_SIZE = 20
FUSED_Y_RANGE = (-0.8, 2.0)
FUSED_TICK_FONT_SIZE = 17
FUSED_TICK_ANGLE = 40
# fused-panel framing: thin mirrored border around each panel + a wider gutter
# so the h=1 / h=20 grouping reads without the titles
FUSED_HORIZONTAL_SPACING = 0.055
PANEL_BORDER_COLOR = "#B7BDC6"
PANEL_BORDER_WIDTH = 1.5
# squished fused variant for full-width (figure*) placement: 1800x550 @ 2x ->
# 3600x1100, ~3.3:1 banner aspect; sparser y ticks + bold text for legibility
SQUISHED_HEIGHT_PX = 550
SQUISHED_Y_DTICK = 0.5


def load_headline_runs() -> dict[int, dict[str, list[float]]]:
    """Sortino values per horizon per model at the headline env.

    BuyAndHold is horizon-agnostic — its (unique) value is injected into every
    horizon's dict after an explicit uniqueness check.
    """
    with open(RUNS_INDEX_PATH) as f:
        index = json.load(f)
    canon = [r for r in index if r.get("experiment_name") == CANON_EXPERIMENT]
    if not canon:
        raise ValueError(f"no '{CANON_EXPERIMENT}' runs in {RUNS_INDEX_PATH}")

    by_horizon: dict[int, dict[str, list[float]]] = {
        h: defaultdict(list) for h in HORIZONS
    }
    buy_and_hold: set[float] = set()
    for r in canon:
        c: dict[str, Any] = r["config_summary"]
        if (
            c.get("transaction_cost_pct") != HEADLINE_COST
            or c.get("conviction_mode") != HEADLINE_CONVICTION
        ):
            continue
        sortino = r["metrics"]["sortino_ratio"]
        # BuyAndHold ignores horizon/threshold/EMA knobs — same value in every cell
        if r["model_type"] == "BuyAndHold":
            buy_and_hold.add(round(sortino, 6))
            continue
        if (
            c.get("min_predicted_return") == HEADLINE_MIN_PREDICTED_RETURN
            and c.get("sentiment_ema_spans") == HEADLINE_EMA_SPANS
            and c.get("rebalance_horizon_days") in by_horizon
        ):
            by_horizon[c["rebalance_horizon_days"]][r["model_type"]].append(sortino)

    if len(buy_and_hold) != 1:
        raise ValueError(
            f"BuyAndHold Sortino not unique across cells: {sorted(buy_and_hold)}"
        )
    bh_value = buy_and_hold.pop()
    for h in HORIZONS:
        by_horizon[h]["BuyAndHold"] = [bh_value]
    return {h: dict(by_horizon[h]) for h in HORIZONS}


def verify(by_model: dict[str, list[float]], horizon: int) -> None:
    expected = EXPECTED_RANGES[horizon]
    unexpected = set(by_model) - set(expected)
    if unexpected:
        raise ValueError(
            f"h={horizon}: unexpected models in headline slice: {sorted(unexpected)}"
        )
    for model, (lo, hi) in expected.items():
        values = by_model.get(model, [])
        expected_n = SEEDS_PER_CELL if model in NEURAL_MODELS else 1
        if len(values) != expected_n:
            raise ValueError(
                f"h={horizon} {model}: expected {expected_n} runs, got {len(values)}"
            )
        if not (lo <= min(values) and max(values) <= hi):
            raise ValueError(
                f"h={horizon} {model}: Sortino {min(values):.3f}..{max(values):.3f} "
                f"outside expected [{lo}, {hi}] — wrong data slice?"
            )


def model_orders(by_model: dict[str, list[float]]) -> tuple[list[str], list[str]]:
    present = [m for m in DETERMINISTIC_MODELS if m in by_model]
    det = sorted(present, key=lambda m: -by_model[m][0])
    neural = sorted(NEURAL_MODELS, key=lambda m: -statistics.mean(by_model[m]))
    return det, neural


def add_panel_traces(
    fig: go.Figure,
    by_model: dict[str, list[float]],
    det_order: list[str],
    neural_order: list[str],
    col: int | None = None,
) -> None:
    """Diamonds, min-max boxes, seed-point overlays, and the block separator."""
    row = 1 if col is not None else None
    for i, model in enumerate(det_order):
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=by_model[model],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=DIAMOND_SIZE,
                    color=MODEL_COLORS[model],
                    line=dict(color=MARKER_OUTLINE_COLOR, width=2),
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    for j, model in enumerate(neural_order):
        pos = len(det_order) + 1 + j
        values = sorted(by_model[model])
        q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
        # precomputed stats: whiskers pinned to min/max so no point detaches
        fig.add_trace(
            go.Box(
                x=[pos],
                q1=[q1],
                median=[median],
                q3=[q3],
                lowerfence=[values[0]],
                upperfence=[values[-1]],
                fillcolor=MODEL_COLORS[model],
                line=dict(color=MODEL_COLORS[model], width=2),
                width=BOX_WIDTH,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[pos + off for off in POINT_JITTER[: len(values)]],
                y=values,
                mode="markers",
                marker=dict(
                    size=POINT_SIZE,
                    color=MODEL_COLORS[model],
                    line=dict(color=MARKER_OUTLINE_COLOR, width=1.5),
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    fig.add_vline(
        x=len(det_order),
        line=dict(color=SEPARATOR_COLOR, width=2, dash="dot"),
        row=row,  # pyright: ignore[reportArgumentType]
        col=col,  # pyright: ignore[reportArgumentType]
    )


def panel_axis_config(
    det_order: list[str], neural_order: list[str], tick_font_size: int, tick_angle: int
) -> dict[str, Any]:
    tick_positions = list(range(len(det_order))) + [
        len(det_order) + 1 + j for j in range(len(neural_order))
    ]
    return dict(
        tickvals=tick_positions,
        ticktext=det_order + neural_order,
        tickfont=dict(size=tick_font_size, color=AXIS_TEXT_COLOR, weight="bold"),
        tickangle=tick_angle,
        range=[-0.7, len(det_order) + len(neural_order) + 0.7],
        showgrid=False,
        zeroline=False,
    )


def build_single_figure(by_model: dict[str, list[float]], horizon: int) -> go.Figure:
    det_order, neural_order = model_orders(by_model)
    bh_value = by_model["BuyAndHold"][0]

    fig = go.Figure()
    add_panel_traces(fig, by_model, det_order, neural_order)

    if min(min(v) for v in by_model.values()) < 0:
        # sign flips are the figure's argument — make the zero crossing visible
        fig.add_hline(y=0, line=dict(color=ZERO_LINE_COLOR, width=1.5))
    fig.add_hline(
        y=bh_value, line=dict(color=REFERENCE_LINE_COLOR, width=2, dash="dash")
    )
    fig.add_annotation(
        x=len(det_order) - BH_LABEL_GAP,
        y=bh_value,
        text=f"Buy & Hold ({bh_value:.2f})",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=20, color=REFERENCE_LINE_COLOR, weight="bold"),
    )

    for center_x, label in [
        ((len(det_order) - 1) / 2, "Deterministic"),
        (len(det_order) + 1 + (len(neural_order) - 1) / 2, "Neural (5 seeds)"),
    ]:
        fig.add_annotation(
            x=center_x,
            y=-0.16,
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=22, color=BLOCK_LABEL_COLOR, weight="bold"),
        )

    fig.update_layout(
        width=FIG_WIDTH_PX,
        height=FIG_HEIGHT_PX,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=90, r=40, t=40, b=120),
        xaxis=panel_axis_config(
            det_order, neural_order, tick_font_size=SINGLE_TICK_FONT_SIZE, tick_angle=0
        ),
        yaxis=dict(
            title=dict(
                text="Sortino Ratio",
                font=dict(size=28, color=AXIS_TEXT_COLOR, weight="bold"),
            ),
            tickfont=dict(size=22, color=AXIS_TEXT_COLOR, weight="bold"),
            gridcolor=GRID_COLOR,
            zeroline=False,
        ),
    )
    return fig


def build_fused_figure(
    runs_by_horizon: dict[int, dict[str, list[float]]],
    height_px: int = FIG_HEIGHT_PX,
    compact: bool = False,
) -> go.Figure:
    """Two-panel h=1 | h=20 figure; `compact` = y-squished banner variant for
    full-width figure* placement (sparser y ticks against the reduced height)."""
    bh_value = runs_by_horizon[HORIZONS[0]]["BuyAndHold"][0]
    first_det_order, _ = model_orders(runs_by_horizon[min(HORIZONS)])
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=FUSED_HORIZONTAL_SPACING,
        subplot_titles=[f"h = {h}" for h in sorted(HORIZONS)],
    )
    # only the subplot titles exist as annotations at this point
    fig.update_annotations(font=dict(size=26, color=AXIS_TEXT_COLOR, weight="bold"))

    for col, horizon in enumerate(sorted(HORIZONS), start=1):
        by_model = runs_by_horizon[horizon]
        det_order, neural_order = model_orders(by_model)
        add_panel_traces(fig, by_model, det_order, neural_order, col=col)
        axis = panel_axis_config(
            det_order,
            neural_order,
            tick_font_size=FUSED_TICK_FONT_SIZE,
            tick_angle=FUSED_TICK_ANGLE,
        )
        fig.update_xaxes(row=1, col=col, **axis)

    fig.add_hline(y=0, line=dict(color=ZERO_LINE_COLOR, width=1.5))
    fig.add_hline(
        y=bh_value, line=dict(color=REFERENCE_LINE_COLOR, width=2, dash="dash")
    )
    fig.add_annotation(
        x=len(first_det_order) - BH_LABEL_GAP,
        y=bh_value,
        text=f"Buy & Hold ({bh_value:.2f})",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=18, color=REFERENCE_LINE_COLOR, weight="bold"),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        showline=True,
        linecolor=PANEL_BORDER_COLOR,
        linewidth=PANEL_BORDER_WIDTH,
        mirror=True,
    )
    fig.update_yaxes(
        range=list(FUSED_Y_RANGE),
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont=dict(size=20, color=AXIS_TEXT_COLOR, weight="bold"),
        showline=True,
        linecolor=PANEL_BORDER_COLOR,
        linewidth=PANEL_BORDER_WIDTH,
        mirror=True,
    )
    if compact:
        fig.update_yaxes(dtick=SQUISHED_Y_DTICK)
    fig.update_yaxes(
        row=1,
        col=1,
        title=dict(
            text="Sortino Ratio",
            font=dict(size=26, color=AXIS_TEXT_COLOR, weight="bold"),
        ),
    )
    fig.update_layout(
        width=FIG_WIDTH_PX,
        height=height_px,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=90, r=40, t=60, b=110),
    )
    return fig


if __name__ == "__main__":
    runs = load_headline_runs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for h in sorted(HORIZONS):
        verify(runs[h], h)
        print(f"--- h={h} ---")
        for name in DETERMINISTIC_MODELS + NEURAL_MODELS:
            if name not in runs[h]:
                print(f"  {name:>12}: (excluded from canon at h={h})")
                continue
            vals = sorted(runs[h][name])
            print(f"  {name:>12}: {', '.join(f'{v:.3f}' for v in vals)}")
        single = build_single_figure(runs[h], h)
        single.write_image(str(SINGLE_OUTPUTS[h]), scale=EXPORT_SCALE)
        print(f"wrote {SINGLE_OUTPUTS[h]}")
    fused = build_fused_figure(runs)
    fused.write_image(str(FUSED_OUTPUT), scale=EXPORT_SCALE)
    print(f"wrote {FUSED_OUTPUT}")
    squished = build_fused_figure(runs, height_px=SQUISHED_HEIGHT_PX, compact=True)
    squished.write_image(str(SQUISHED_OUTPUT), scale=EXPORT_SCALE)
    print(f"wrote {SQUISHED_OUTPUT}")
