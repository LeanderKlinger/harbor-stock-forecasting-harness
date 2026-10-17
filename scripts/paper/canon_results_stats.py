"""Reproducible source of every run-index statistic reported in the paper.

Computed from `big_eval_canon` ONLY, the canonical experiment. No other
experiment contributes a number here; mixing scopes is the practice the paper
critiques. REPORTED_FIGURES.md lists the resulting figures.

Run: uv run python scripts/paper/canon_results_stats.py

Sections printed:
  1. Seed dispersion per cell (RMSE CV vs Sortino range vs DA range)
  2. Per-model DA ranges + EqualWeight per-horizon DA (the always-up floor)
  3. Best single-seed neural run (the counterfactual exhibit, Section 4)
  4. Sortino variance decomposition (neural block)
  5. Best config per model (Table 2)
  6. Saidd-mirror RMSE decomposition
  7. Anchors vs best neural per shared environment (selection-free comparison)
  8. Sortino by model x horizon at the headline environment (Table 3)
  9. Seed spread per neural model at the headline environment
 10. Win-rate exhibit against the EqualWeight passive floor
 11. Cost sensitivity at h=1 (0% vs 0.1% transaction cost)
 12. NVDA gain over the test window (the bull run in the limitations)
"""

import json
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

import polars as pl

Run = dict[str, Any]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import RUNS_INDEX_PATH, STOCK_DATA_DIR  # noqa: E402  (needs the sys.path bootstrap above)

CANON_EXPERIMENT = "big_eval_canon"
HEADLINE_COST = 0.001
ZERO_COST = 0.0
NEURAL_MODELS = {"MLP", "GRU", "Mamba", "Jamba", "xLSTM", "XLSTM"}
SEEDS_PER_CELL = 5
# The headline environment (Table 3): conviction mode, min-return threshold, EMA spans.
HEADLINE_ENV = ("volatility_scaled", 0.01, (5, 10, 20))
HORIZONS = (1, 5, 20)
TABLE3_ORDER = [
    "BuyAndHold",
    "EqualWeight",
    "Momentum",
    "HoltsLES",
    "ARIMA",
    "xLSTM",
    "GRU",
    "Mamba",
    "Jamba",
    "MLP",
]
HORIZON_INDEPENDENT_MODEL = "BuyAndHold"
TEST_START = date(2020, 1, 1)
TEST_END = date(2023, 12, 31)
BULL_RUN_STOCK = "NVDA"
WIN_RATE_EXHIBIT_MODEL = "GRU"
PASSIVE_FLOOR_MODEL = "EqualWeight"
COST_SENSITIVITY_HORIZON = 1
COST_SENSITIVITY_MODELS = ("xLSTM", "Momentum")


def load_canon() -> list[Run]:
    with open(RUNS_INDEX_PATH) as f:
        index = json.load(f)
    canon = [r for r in index if r.get("experiment_name") == CANON_EXPERIMENT]
    if not canon:
        raise ValueError(f"no '{CANON_EXPERIMENT}' runs in {RUNS_INDEX_PATH}")
    return canon


def at_cost(runs: list[Run], cost: float) -> list[Run]:
    return [r for r in runs if r["config_summary"].get("transaction_cost_pct") == cost]


def cell_key(run: Run) -> tuple[Any, ...]:
    c: dict[str, Any] = run["config_summary"]
    return (
        run["model_type"],
        c["rebalance_horizon_days"],
        c.get("conviction_mode"),
        c.get("min_predicted_return"),
        tuple(c.get("sentiment_ema_spans", [])),
    )


def seed_dispersion(runs: list[Run]) -> None:
    print("=" * 72)
    print(f"1. SEED DISPERSION per neural cell (canon @ {HEADLINE_COST:.1%} cost)")
    print("=" * 72)
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        if r["model_type"] in NEURAL_MODELS:
            cells[cell_key(r)].append(r["metrics"])
    rmse_cvs: list[float] = []
    sortino_ranges: list[float] = []
    da_ranges: list[float] = []
    flagship_lines: list[str] = []
    for key, metrics in cells.items():
        if len(metrics) != SEEDS_PER_CELL:
            print(f"  SKIP {key}: {len(metrics)} runs (expected {SEEDS_PER_CELL})")
            continue
        rmse = [m["rmse"] for m in metrics]
        sortino = [m["sortino_ratio"] for m in metrics]
        da = [m["directional_accuracy"] for m in metrics]
        cv = statistics.pstdev(rmse) / statistics.mean(rmse) * 100
        rmse_cvs.append(cv)
        sortino_ranges.append(max(sortino) - min(sortino))
        da_ranges.append(max(da) - min(da))
        if key[0] == "MLP" and key[1] == 1 and key[4] == (5, 10, 20):
            flagship_lines.append(
                f"  FLAGSHIP MLP h=1 {key[2]} mpr={key[3]}: "
                f"rmse CV {cv:.1f}% | sortino {min(sortino):+.2f}..{max(sortino):+.2f} "
                f"| DA range {(max(da) - min(da)) * 100:.1f}pp"
            )
    print(f"clean {SEEDS_PER_CELL}-seed cells: {len(rmse_cvs)}")
    print(
        f"RMSE CV across seeds:  median {statistics.median(rmse_cvs):.1f}%  max {max(rmse_cvs):.1f}%"
    )
    print(
        f"Sortino seed range:    median {statistics.median(sortino_ranges):.2f}  "
        f"mean {statistics.mean(sortino_ranges):.2f}"
    )
    print(f"DA seed range:         median {statistics.median(da_ranges) * 100:.1f}pp")
    # per-horizon Sortino seed-range breakdown (cells regrouped by horizon)
    by_horizon: dict[int, list[float]] = defaultdict(list)
    for key, metrics in cells.items():
        if len(metrics) != SEEDS_PER_CELL:
            continue
        sortino = [m["sortino_ratio"] for m in metrics]
        by_horizon[key[1]].append(max(sortino) - min(sortino))
    for h in sorted(by_horizon):
        v = by_horizon[h]
        print(
            f"  Sortino seed range h={h:>2}: median {statistics.median(v):.2f}  "
            f"mean {statistics.mean(v):.2f}  max {max(v):.2f}  (n={len(v)} cells)"
        )
    for line in flagship_lines:
        print(line)


def da_ranges_per_model(runs: list[Run]) -> None:
    print("=" * 72)
    print(f"2. DIRECTIONAL ACCURACY per model (canon @ {HEADLINE_COST:.1%} cost)")
    print("=" * 72)
    by_model: dict[str, list[float]] = defaultdict(list)
    ew_by_horizon: dict[int, set[float]] = defaultdict(set)
    for r in runs:
        da = r["metrics"]["directional_accuracy"]
        by_model[r["model_type"]].append(da)
        if r["model_type"] == "EqualWeight":
            ew_by_horizon[r["config_summary"]["rebalance_horizon_days"]].add(
                round(da, 4)
            )
    for model in sorted(by_model):
        v = by_model[model]
        print(f"  {model:>12}: {min(v) * 100:5.1f}..{max(v) * 100:5.1f}%  (n={len(v)})")
    neural = [x for m in NEURAL_MODELS for x in by_model.get(m, [])]
    print(f"  NEURAL overall: {min(neural) * 100:.2f}..{max(neural) * 100:.2f}%")
    print("  EqualWeight per horizon (always-up DA floor):")
    for h in sorted(ew_by_horizon):
        print(f"    h={h}: {sorted(ew_by_horizon[h])}")


def best_neural_run(runs: list[Run]) -> None:
    print("=" * 72)
    print(
        f"3. BEST SINGLE-SEED NEURAL RUN by Sortino (canon @ {HEADLINE_COST:.1%} cost)"
    )
    print("=" * 72)
    neural_runs = [r for r in runs if r["model_type"] in NEURAL_MODELS]
    top = sorted(neural_runs, key=lambda r: -r["metrics"]["sortino_ratio"])[:5]
    for r in top:
        m, c = r["metrics"], r["config_summary"]
        print(
            f"  {r['model_type']:>6} h={c['rebalance_horizon_days']:>2} "
            f"{c.get('conviction_mode'):<17} mpr={c.get('min_predicted_return')} "
            f"ema={c.get('sentiment_ema_spans')} seed={r['seed']}: "
            f"sortino {m['sortino_ratio']:.2f} | ann {m['annualized_return'] * 100:.1f}% "
            f"| total {m['total_return'] * 100:.0f}% | maxDD {m['maximum_drawdown'] * 100:.1f}%"
        )


def sortino_variance_decomposition(runs: list[Run]) -> None:
    """Descriptive main-effect SS shares over the balanced 600-run neural block.

    Same construction as Saidd's Table-14 decomposition (labeled-factor sum of
    squares); interactions land in the residual — which therefore CONTAINS the
    within-cell seed instability. The seed main effect here is NOT the
    instability number; the within-cell seed spreads (section 1) are.
    """
    print("=" * 72)
    print(
        f"4. SORTINO VARIANCE DECOMPOSITION, neural block (canon @ {HEADLINE_COST:.1%})"
    )
    print("=" * 72)
    neural = [r for r in runs if r["model_type"] in NEURAL_MODELS]
    ys = [r["metrics"]["sortino_ratio"] for r in neural]
    grand = statistics.mean(ys)
    ss_total = sum((y - grand) ** 2 for y in ys)
    print(f"runs: {len(neural)} | grand mean {grand:.3f} | total SS {ss_total:.1f}")

    def share(name: str, keyfn: "Callable[[Run], Any]") -> float:
        groups: dict[Any, list[float]] = defaultdict(list)
        for r in neural:
            groups[keyfn(r)].append(r["metrics"]["sortino_ratio"])
        ss = sum(len(v) * (statistics.mean(v) - grand) ** 2 for v in groups.values())
        print(f"  {name:>10}: {ss / ss_total * 100:5.1f}%  ({len(groups)} levels)")
        return ss / ss_total

    total = 0.0
    total += share("horizon", lambda r: r["config_summary"]["rebalance_horizon_days"])
    total += share("model", lambda r: r["model_type"])
    total += share("seed", lambda r: r["seed"])
    total += share("conviction", lambda r: r["config_summary"].get("conviction_mode"))
    total += share(
        "threshold", lambda r: r["config_summary"].get("min_predicted_return")
    )
    total += share(
        "ema_spans", lambda r: tuple(r["config_summary"].get("sentiment_ema_spans", []))
    )
    print(
        f"  main effects sum {total * 100:.1f}% -> residual (interactions incl. within-cell seed) {(1 - total) * 100:.1f}%"
    )


def best_config_per_model(runs: list[Run]) -> None:
    """Table 2: each model's best cell (by mean Sortino) with full row data."""
    print("=" * 72)
    print(f"5. BEST CONFIG PER MODEL (canon @ {HEADLINE_COST:.1%}) — leaderboard feed")
    print("=" * 72)
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        cells[cell_key(r)].append(r["metrics"])
    best: dict[str, tuple[float, tuple[Any, ...], list[dict[str, Any]]]] = {}
    for key, metrics in cells.items():
        model = key[0]
        mean_sortino = statistics.mean(m["sortino_ratio"] for m in metrics)
        if model not in best or mean_sortino > best[model][0]:
            best[model] = (mean_sortino, key, metrics)
    for model, (mean_sortino, key, metrics) in sorted(
        best.items(), key=lambda kv: -kv[1][0]
    ):
        _, h, conv, thr, ema = key
        n = len(metrics)
        # sample std (n-1): the findings file's ± convention — printed tables must match it
        std = statistics.stdev([m["sortino_ratio"] for m in metrics]) if n > 1 else 0.0
        ann = statistics.mean(m["annualized_return"] for m in metrics)
        mdd = statistics.mean(m["maximum_drawdown"] for m in metrics)
        print(
            f"  {model:>12}: h={h:>2} {str(conv):<17} thr={thr} ema={list(ema)} n={n} | "
            f"sortino {mean_sortino:.2f}"
            + (f" ±{std:.2f}" if n > 1 else "      ")
            + f" | ann {ann * 100:6.2f}% | maxDD {mdd * 100:4.1f}%"
        )


HEADLINE_EMA_SPANS = [5, 10, 20]


def saidd_mirror_decomposition(runs: list[Run]) -> None:
    """Saidd Table-14 construction on OUR data: two-factor (model x seed) SS
    decomposition of forecast RMSE, per horizon at the headline env. This is
    the ONLY like-for-like comparison against his seed share (<=0.04%) —
    never juxtapose his share with our within-cell CV (different statistics).
    RMSE is prediction-level, so portfolio-knob duplicates are deduped (and
    asserted identical) down to unique (model, horizon, seed) units.
    """
    print("=" * 72)
    print("6. SAIDD-MIRROR: model x seed SS decomposition of RMSE (headline env)")
    print("=" * 72)
    unique: dict[tuple[Any, ...], float] = {}
    for r in runs:
        c = r["config_summary"]
        if (
            r["model_type"] not in NEURAL_MODELS
            or c.get("sentiment_ema_spans") != HEADLINE_EMA_SPANS
        ):
            continue
        key = (r["model_type"], c["rebalance_horizon_days"], r["seed"])
        rmse = r["metrics"]["rmse"]
        if key in unique and abs(unique[key] - rmse) > 1e-12:
            raise ValueError(f"RMSE differs across portfolio knobs for {key}")
        unique[key] = rmse
    for h in sorted({k[1] for k in unique}):
        vals = {(k[0], k[2]): v for k, v in unique.items() if k[1] == h}
        ys = list(vals.values())
        grand = statistics.mean(ys)
        ss_total = sum((y - grand) ** 2 for y in ys)

        def main_effect(pick: Callable[[tuple[Any, ...]], Any]) -> float:
            groups: dict[Any, list[float]] = defaultdict(list)
            for k, v in vals.items():
                groups[pick(k)].append(v)
            return (
                sum(len(v) * (statistics.mean(v) - grand) ** 2 for v in groups.values())
                / ss_total
            )

        m = main_effect(lambda k: k[0])
        s = main_effect(lambda k: k[1])
        print(
            f"  h={h:>2}: Model {m * 100:5.1f}%  Seed {s * 100:5.2f}%  Residual {(1 - m - s) * 100:5.1f}%"
        )


STATISTICAL_ANCHORS = {"HoltsLES", "ARIMA"}
MARKET_ANCHORS = {"Momentum", "EqualWeight"}
HORIZON_INDEPENDENT_ANCHOR = "BuyAndHold"
PARAMETER_FREE_ANCHOR = "Momentum"


def anchors_vs_neural_per_environment(runs: list[Run]) -> None:
    """Selection-free counterpart to Table 2.

    Every model is compared inside every shared environment (horizon x
    conviction x threshold x EMA arm), so no per-model configuration is picked
    on the test window. Conservative toward neural: each environment's neural
    side is the BEST of the five architectures. BuyAndHold is horizon-
    independent (single buy), so its one value joins every environment.
    Momentum has no fitted parameters, hence nothing to refit: its count
    isolates the frozen-vs-updating asymmetry the reviewers raised.
    """
    print("=" * 72)
    print(
        f"7. ANCHORS vs BEST NEURAL per shared environment (canon @ {HEADLINE_COST:.1%})"
    )
    print("=" * 72)
    cells: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    buy_and_hold: list[float] = []
    for r in runs:
        sortino = r["metrics"]["sortino_ratio"]
        if r["model_type"] == HORIZON_INDEPENDENT_ANCHOR:
            buy_and_hold.append(sortino)
            continue
        key = cell_key(r)
        cells[(key[1:], r["model_type"])].append(sortino)
    if not buy_and_hold:
        raise ValueError(f"no {HORIZON_INDEPENDENT_ANCHOR} runs in canon")
    buy_and_hold_sortino = statistics.mean(buy_and_hold)

    environments = sorted({environment for environment, _ in cells})
    anchor_wins = 0
    parameter_free_wins = 0
    statistical_wins = 0
    for environment in environments:
        neural_means: list[tuple[float, str]] = []
        for model in NEURAL_MODELS:
            if (environment, model) in cells:
                neural_means.append(
                    (statistics.mean(cells[(environment, model)]), model)
                )
        if not neural_means:
            raise ValueError(f"no neural cells in environment {environment}")
        best_neural, best_neural_model = max(neural_means)

        statistical_means: list[float] = []
        for model in STATISTICAL_ANCHORS:
            if (environment, model) in cells:
                statistical_means.append(statistics.mean(cells[(environment, model)]))
        if not statistical_means:
            raise ValueError(f"no statistical anchor in environment {environment}")
        best_statistical = max(statistical_means)

        market_means: list[float] = []
        for model in MARKET_ANCHORS:
            if (environment, model) not in cells:
                raise ValueError(f"{model} missing in environment {environment}")
            market_means.append(statistics.mean(cells[(environment, model)]))
        parameter_free = statistics.mean(cells[(environment, PARAMETER_FREE_ANCHOR)])

        best_anchor = max(best_statistical, buy_and_hold_sortino, *market_means)
        anchor_wins += best_anchor > best_neural
        parameter_free_wins += parameter_free > best_neural
        statistical_wins += best_statistical > best_neural
        print(
            f"  {str(environment):<44} best neural {best_neural_model:>5} {best_neural:5.2f} | "
            f"stat {best_statistical:5.2f} | {PARAMETER_FREE_ANCHOR} {parameter_free:5.2f}"
        )
    total = len(environments)
    print(f"  environments: {total} (BuyAndHold {buy_and_hold_sortino:.2f} in each)")
    print(f"  best anchor > best neural:       {anchor_wins}/{total}")
    print(
        f"  {PARAMETER_FREE_ANCHOR} alone > best neural:    {parameter_free_wins}/{total}"
    )
    print(f"  statistical anchor > best neural: {statistical_wins}/{total}")


def cell_summary(model: str, values: list[float]) -> str:
    """Mean ± sample std across seeds for neural models; deterministic models run
    once (BuyAndHold's stored duplicates are identical), so they print the value."""
    if model in NEURAL_MODELS:
        return f"{statistics.mean(values):.2f} ±{statistics.stdev(values):.2f}"
    return f"{statistics.mean(values):.2f}"


def headline_cells(runs: list[Run]) -> dict[tuple[str, int], list[float]]:
    """Sortino per (model, horizon) at the headline environment. BuyAndHold is
    horizon-independent (single buy, held throughout), so it fills every horizon."""
    cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in runs:
        sortino = r["metrics"]["sortino_ratio"]
        if r["model_type"] == HORIZON_INDEPENDENT_MODEL:
            for h in HORIZONS:
                cells[(HORIZON_INDEPENDENT_MODEL, h)].append(sortino)
            continue
        key = cell_key(r)
        if key[2:] == HEADLINE_ENV:
            cells[(r["model_type"], key[1])].append(sortino)
    return cells


def headline_matrix(runs: list[Run]) -> None:
    print("=" * 72)
    print(
        f"8. SORTINO BY MODEL x HORIZON, headline environment (Table 3, {HEADLINE_COST:.1%})"
    )
    print("=" * 72)
    cells = headline_cells(runs)
    print(f"  {'model':>12} " + " ".join(f"{'h=' + str(h):>12}" for h in HORIZONS))
    for model in TABLE3_ORDER:
        row: list[str] = []
        for h in HORIZONS:
            values = cells.get((model, h), [])
            row.append(f"{cell_summary(model, values) if values else '---':>12}")
        print(f"  {model:>12} " + " ".join(row))


def headline_seed_spreads(runs: list[Run]) -> None:
    print("=" * 72)
    print("9. SEED SPREAD (max - min Sortino) per neural model, headline environment")
    print("=" * 72)
    cells = headline_cells(runs)
    for model in sorted(NEURAL_MODELS):
        spreads: list[str] = []
        for h in HORIZONS:
            values = cells.get((model, h), [])
            if values:
                spreads.append(f"h={h}: {max(values) - min(values):.2f}")
        if spreads:
            print(f"  {model:>6}: " + "  ".join(spreads))


def win_rate_exhibit(runs: list[Run]) -> None:
    """The contextualization exhibit: the GRU cell with the highest win rate,
    against the EqualWeight passive floor at the same horizon."""
    print("=" * 72)
    print(f"10. WIN-RATE EXHIBIT vs {PASSIVE_FLOOR_MODEL} floor ({HEADLINE_COST:.1%})")
    print("=" * 72)
    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        cells[cell_key(r)].append(r["metrics"])
    candidates: list[tuple[float, tuple[Any, ...]]] = []
    for key, metrics in cells.items():
        if key[0] == WIN_RATE_EXHIBIT_MODEL:
            candidates.append((statistics.mean(m["win_ratio"] for m in metrics), key))
    if not candidates:
        raise ValueError(f"no {WIN_RATE_EXHIBIT_MODEL} cells in canon")
    _, top_key = max(candidates)
    top = cells[top_key]
    win = [m["win_ratio"] * 100 for m in top]
    accuracy = [m["directional_accuracy"] * 100 for m in top]
    floor_key = (PASSIVE_FLOOR_MODEL, top_key[1], *HEADLINE_ENV)
    if floor_key not in cells:
        raise ValueError(f"no {PASSIVE_FLOOR_MODEL} cell at horizon {top_key[1]}")
    floor = cells[floor_key][0]
    floor_win = floor["win_ratio"] * 100
    floor_accuracy = floor["directional_accuracy"] * 100
    print(f"  top {WIN_RATE_EXHIBIT_MODEL} win-rate cell: {top_key}")
    print(
        f"    win rate {cell_summary(WIN_RATE_EXHIBIT_MODEL, win)}%   "
        f"DA {cell_summary(WIN_RATE_EXHIBIT_MODEL, accuracy)}%"
    )
    print(
        f"  {PASSIVE_FLOOR_MODEL} h={top_key[1]}: win rate {floor_win:.2f}%   DA {floor_accuracy:.2f}%"
    )
    print(
        f"  below the floor by {floor_win - statistics.mean(win):.2f}pp (win rate) and "
        f"{floor_accuracy - statistics.mean(accuracy):.2f}pp (DA)"
    )


def cost_sensitivity(all_runs: list[Run]) -> None:
    print("=" * 72)
    print(f"11. COST SENSITIVITY at h={COST_SENSITIVITY_HORIZON}, headline environment")
    print("=" * 72)
    for model in COST_SENSITIVITY_MODELS:
        line: list[str] = []
        for cost in (ZERO_COST, HEADLINE_COST):
            values = headline_cells(at_cost(all_runs, cost)).get(
                (model, COST_SENSITIVITY_HORIZON), []
            )
            if not values:
                raise ValueError(
                    f"no {model} h={COST_SENSITIVITY_HORIZON} cell at cost {cost}"
                )
            line.append(f"{cost:.1%} cost: {cell_summary(model, values)}")
        print(f"  {model:>8}: " + "  ->  ".join(line))


def bull_run_gain() -> None:
    print("=" * 72)
    print(
        f"12. {BULL_RUN_STOCK} GAIN over the test window ({TEST_START} to {TEST_END})"
    )
    print("=" * 72)
    prices = (
        pl.read_csv(STOCK_DATA_DIR / f"{BULL_RUN_STOCK}.csv")
        .with_columns(pl.col("date").str.to_date())
        .filter((pl.col("date") >= TEST_START) & (pl.col("date") <= TEST_END))
        .sort("date")
    )
    if prices.is_empty():
        raise ValueError(f"no {BULL_RUN_STOCK} prices inside the test window")
    first_close: float = prices["close"][0]
    last_close: float = prices["close"][-1]
    print(
        f"  close {prices['date'][0]}: {first_close:.2f}  ->  {prices['date'][-1]}: "
        f"{last_close:.2f}  =  {(last_close / first_close - 1) * 100:+.1f}%"
    )


if __name__ == "__main__":
    all_canon_runs = load_canon()
    canon_runs = at_cost(all_canon_runs, HEADLINE_COST)
    print(f"{CANON_EXPERIMENT}: {len(canon_runs)} runs at cost {HEADLINE_COST}")
    seed_dispersion(canon_runs)
    da_ranges_per_model(canon_runs)
    best_neural_run(canon_runs)
    sortino_variance_decomposition(canon_runs)
    best_config_per_model(canon_runs)
    saidd_mirror_decomposition(canon_runs)
    anchors_vs_neural_per_environment(canon_runs)
    headline_matrix(canon_runs)
    headline_seed_spreads(canon_runs)
    win_rate_exhibit(canon_runs)
    cost_sensitivity(all_canon_runs)
    bull_run_gain()
