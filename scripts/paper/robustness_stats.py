"""Reproducible source of the robustness figures reported in the paper (Section 4.5).

Two experiments, each scoped to its own records in the run index:
  1. Input ablation (13_feature_set_ablation): neural models at three input
     widths (full / numeric / close_only) under the headline environment.
  2. Sequence-length sweep (seqlen_sweep): neural models at 60 / 120 / 250
     input days under the headline environment.

Run: uv run python scripts/paper/robustness_stats.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import binomtest  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

Run = dict[str, Any]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import RUNS_INDEX_PATH  # noqa: E402  (needs the sys.path bootstrap above)

ABLATION_EXPERIMENT = "13_feature_set_ablation"
SWEEP_EXPERIMENT = "seqlen_sweep"
NEURAL_MODELS = {"MLP", "GRU", "Mamba", "Jamba", "xLSTM"}
# Contrasts fixed before the ablation ran: two-sided sign tests over seed-matched deltas.
ABLATION_CONTRASTS = [("full", "numeric"), ("numeric", "close_only")]
SIGNIFICANCE_LEVEL = 0.05
PARITY_ARM = "close_only"
NOTABLE_ABLATION_CELL = ("MLP", 20)
BASELINE_SEQUENCE_LENGTH = 60
LONG_CONTEXT_EXCEPTION = ("Mamba", 5, 250)


def load_experiment(name: str) -> list[Run]:
    with open(RUNS_INDEX_PATH) as f:
        index = json.load(f)
    runs = [
        r
        for r in index
        if r.get("experiment_name") == name and r["model_type"] in NEURAL_MODELS
    ]
    if not runs:
        raise ValueError(f"no neural '{name}' runs in {RUNS_INDEX_PATH}")
    return runs


def sign_test(
    by_unit: dict[tuple[Any, ...], dict[str, Any]],
    units: list[tuple[Any, ...]],
    arm_a: str,
    arm_b: str,
    metric: str,
    lower_is_better: bool,
) -> str:
    """Counts the seed-matched units where arm_b beats arm_a; ties are dropped,
    as a sign test cannot assign them a direction."""
    b_better = 0
    a_better = 0
    ties = 0
    for unit in units:
        a = by_unit[unit + (arm_a,)][metric]
        b = by_unit[unit + (arm_b,)][metric]
        if a == b:
            ties += 1
        elif (b < a) == lower_is_better:
            b_better += 1
        else:
            a_better += 1
    p_value = float(binomtest(b_better, b_better + a_better, 0.5).pvalue)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    verdict = "significant" if p_value < SIGNIFICANCE_LEVEL else "not significant"
    return (
        f"{arm_b} better in {b_better}/{b_better + a_better + ties} "
        f"({a_better} worse, {ties} tied)  p={p_value:.2g}  {verdict}"
    )


def ablation(runs: list[Run]) -> None:
    print("=" * 72)
    print(f"1. INPUT ABLATION ({ABLATION_EXPERIMENT})")
    print("=" * 72)
    by_unit: dict[tuple[Any, ...], dict[str, Any]] = {}
    cells: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for r in runs:
        c = r["config_summary"]
        unit = (r["model_type"], c["rebalance_horizon_days"], r["seed"])
        by_unit[unit + (c["feature_set"],)] = r["metrics"]
        cells[(r["model_type"], c["rebalance_horizon_days"], c["feature_set"])].append(
            r["metrics"]["sortino_ratio"]
        )
    units = sorted({key[:3] for key in by_unit})
    print(f"seed-matched units (model x horizon x seed): {len(units)}")
    for arm_a, arm_b in ABLATION_CONTRASTS:
        print(f"  {arm_a} -> {arm_b}")
        print(
            f"    RMSE: {sign_test(by_unit, units, arm_a, arm_b, 'rmse', lower_is_better=True)}"
        )
        print(
            f"    DA:   {sign_test(by_unit, units, arm_a, arm_b, 'directional_accuracy', lower_is_better=False)}"
        )
    model, horizon = NOTABLE_ABLATION_CELL
    transitions: list[str] = []
    for feature_set in ("full", "numeric"):
        transitions.append(
            f"{feature_set} {statistics.mean(cells[(model, horizon, feature_set)]):.2f}"
        )
    print(f"  {model} h={horizon} Sortino (5-seed mean): " + " -> ".join(transitions))
    parity: list[tuple[float, tuple[Any, ...]]] = []
    for key, values in cells.items():
        if key[2] == PARITY_ARM:
            parity.append((statistics.mean(values), key))
    best_mean, best_key = max(parity)
    print(
        f"  best {PARITY_ARM} cell: {best_key[0]} h={best_key[1]} "
        f"{best_mean:.2f} ±{statistics.stdev(cells[best_key]):.2f}"
    )


def sweep(runs: list[Run]) -> None:
    print("=" * 72)
    print(f"2. SEQUENCE-LENGTH SWEEP ({SWEEP_EXPERIMENT})")
    print("=" * 72)
    cells: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    per_seed: dict[tuple[Any, ...], float] = {}
    durations: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in runs:
        c = r["config_summary"]
        key = (r["model_type"], c["rebalance_horizon_days"], c["sequence_length"])
        cells[key].append(r["metrics"]["sortino_ratio"])
        per_seed[key + (r["seed"],)] = r["metrics"]["sortino_ratio"]
        durations[(r["model_type"], c["sequence_length"])].append(
            r["training_duration_s"]
        )

    negative = 0
    comparisons = 0
    for model, horizon, length in sorted(cells):
        if length == BASELINE_SEQUENCE_LENGTH:
            continue
        comparisons += 1
        longer = statistics.mean(cells[(model, horizon, length)])
        baseline = statistics.mean(cells[(model, horizon, BASELINE_SEQUENCE_LENGTH)])
        negative += longer < baseline
    print(
        f"  longer context worse than seq={BASELINE_SEQUENCE_LENGTH} "
        f"(5-seed mean): {negative}/{comparisons}"
    )

    best_mean, best_key = max((statistics.mean(v), k) for k, v in cells.items())
    print(
        f"  best cell: {best_key[0]} h={best_key[1]} seq={best_key[2]} "
        f"{best_mean:.2f} ±{statistics.stdev(cells[best_key]):.2f}"
    )

    model, horizon, length = LONG_CONTEXT_EXCEPTION
    deltas: list[float] = []
    for unit in sorted(per_seed):
        if unit[:3] == (model, horizon, length):
            seed = unit[3]
            deltas.append(
                per_seed[unit]
                - per_seed[(model, horizon, BASELINE_SEQUENCE_LENGTH, seed)]
            )
    improved = sum(1 for d in deltas if d > 0)
    print(
        f"  {model} h={horizon} seq={length} vs seq={BASELINE_SEQUENCE_LENGTH}: "
        f"improved on {improved}/{len(deltas)} seeds, mean {statistics.mean(deltas):+.2f}"
    )

    # The sweep shares no fits with any cache (shortest fit runs for minutes), so
    # every duration is a fresh fit. Medians resist the co-scheduling outliers of
    # a shared machine; ratios remain specific to that deployment's hardware.
    longest = max(length for _, length in durations)
    print(
        f"  median training time, seq={longest} vs seq={BASELINE_SEQUENCE_LENGTH} "
        f"(single deployment):"
    )
    for model in sorted(NEURAL_MODELS):
        short = statistics.median(durations[(model, BASELINE_SEQUENCE_LENGTH)])
        long = statistics.median(durations[(model, longest)])
        print(f"    {model:>6}: {short:7.0f}s -> {long:7.0f}s  ({long / short:.1f}x)")


if __name__ == "__main__":
    ablation(load_experiment(ABLATION_EXPERIMENT))
    sweep(load_experiment(SWEEP_EXPERIMENT))
