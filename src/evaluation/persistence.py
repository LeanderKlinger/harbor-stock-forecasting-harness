# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
import fcntl
import hashlib
import json
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.models.config import Config
from src.models.evaluation import PipelineEvaluation
from src.models.input import PipelineInput
from src.models.output import PipelineOutput, PortfolioSnapshot
from src.paths import (
    EXECUTION_OUTPUT_ROOT,
    INPUT_CACHE_DIR,
    RUNS_INDEX_PATH,
    STOCK_DATA_DIR,
)

STOCK_PRICES_DIR = EXECUTION_OUTPUT_ROOT / ".stock_prices"

# Advisory lock on runs_index.json mutations. POSIX (macOS + Linux) only.
# A SEPARATE sentinel file is locked, not runs_index.json itself — the
# atomic tmp+rename pattern moves the index's inode out from under any held
# lock, so locking the index file directly is silently useless.
_INDEX_LOCK_PATH = RUNS_INDEX_PATH.with_suffix(".lock")


@contextmanager
def locked_index() -> Generator[None]:
    """Hold an exclusive flock on the index sentinel for the duration of a
    read-modify-write cycle on runs_index.json.

    Coordinates the pipeline runner's writer thread with the FastAPI server's
    delete handler. Does NOT coordinate with `python -m src.experiments
    reindex` — don't run that concurrently with either.
    """
    _INDEX_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_LOCK_PATH.touch(exist_ok=True)
    with open(_INDEX_LOCK_PATH, "r+") as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)


def _held_intervals_from_snapshots(
    stock_id: str,
    timestamps: list[datetime],
    prices: list[float],
    snapshots: list[PortfolioSnapshot],
) -> tuple[
    list[tuple[datetime, datetime, bool, bool]], list[tuple[datetime, datetime]]
]:
    """Derive held/not-held intervals from daily portfolio snapshots.

    Returns (held, not_held) where held entries are (start, end, profitable, closed).
    """
    held_days: set[datetime] = set()
    for snap in snapshots:
        qty = snap.positions.get(stock_id, 0.0)
        if qty > 0:
            held_days.add(snap.timestamp)

    held_intervals: list[tuple[datetime, datetime, bool, bool]] = []
    not_held_intervals: list[tuple[datetime, datetime]] = []

    price_by_ts: dict[datetime, float] = dict(zip(timestamps, prices))

    run_start: datetime = timestamps[0]
    run_held: bool = timestamps[0] in held_days
    prev_ts: datetime = timestamps[0]

    for ts in timestamps[1:]:
        currently_held = ts in held_days
        if currently_held != run_held:
            if run_held:
                start_price = price_by_ts[run_start]
                end_price = price_by_ts[prev_ts]
                held_intervals.append(
                    (run_start, prev_ts, end_price > start_price, True)
                )
            else:
                not_held_intervals.append((run_start, prev_ts))
            run_start = ts
            run_held = currently_held
        prev_ts = ts

    if run_held:
        start_price = price_by_ts[run_start]
        end_price = price_by_ts[timestamps[-1]]
        is_closed = timestamps[-1] not in held_days
        held_intervals.append(
            (run_start, timestamps[-1], end_price > start_price, is_closed)
        )
    else:
        not_held_intervals.append((run_start, timestamps[-1]))

    return held_intervals, not_held_intervals


class Persistence:
    """Stateless run persistence. All methods take inputs as arguments, store nothing."""

    def persist_run(
        self,
        pipeline_input: PipelineInput,
        output: PipelineOutput,
        evaluation: PipelineEvaluation,
    ) -> Path:
        """Save all run artifacts to a per-run subfolder. Returns the output directory."""
        run_dir = _make_run_dir(output)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pipeline input/output JSON (canonical)
        _write_pipeline_input_cached(run_dir, pipeline_input)
        (run_dir / "pipeline_output.json").write_text(output.model_dump_json(indent=2))

        # 2. Metrics markdown
        (run_dir / "metrics.md").write_text(_format_metrics(output, evaluation))

        # 3. Frontend-optimized run detail
        run_detail = build_run_detail(output, evaluation)
        (run_dir / "run_detail.json").write_text(json.dumps(run_detail, indent=2))

        # 4. Ensure shared stock price files exist
        ensure_stock_prices(output.config.stock_ids)

        # 5. Update runs index
        _append_to_runs_index(run_dir, output, evaluation)

        return run_dir


# ---------------------------------------------------------------------------
# Frontend-optimized run detail
# ---------------------------------------------------------------------------


def build_run_detail(
    output: PipelineOutput,
    evaluation: PipelineEvaluation,
) -> dict[str, Any]:
    """Build the frontend-optimized run_detail.json content.

    Contains everything the React frontend needs for the detail page:
    metrics, config, equity curve, per-stock chart data (trades + held intervals).
    Close prices are NOT included — those come from shared .stock_prices/ files.
    """
    config = output.config

    # Equity curve from snapshots
    equity_curve = [
        {
            "timestamp": s.timestamp.strftime("%Y-%m-%d"),
            "value": round(s.portfolio_value, 2),
        }
        for s in output.snapshot_records
    ]

    # Per-stock chart data
    stock_charts: dict[str, Any] = {}
    for stock_id in output.stocks:
        stock_trades = [t for t in output.trade_records if t.stock_id == stock_id]
        held, not_held = _compute_held_intervals(
            stock_id, output.snapshot_records, config
        )

        stock_charts[stock_id] = {
            "trades": [
                {
                    "timestamp": t.timestamp.strftime("%Y-%m-%d"),
                    "action": t.action.name,
                    "quantity": round(t.quantity, 4),
                    "price": round(t.execution_price, 2),
                }
                for t in stock_trades
            ],
            "held_intervals": [
                {
                    "start": h["start"],
                    "end": h["end"],
                    "profitable": h["profitable"],
                    "closed": h["closed"],
                }
                for h in held
            ],
            "not_held_intervals": [
                {"start": nh["start"], "end": nh["end"]} for nh in not_held
            ],
        }

    # Threshold data for vol-scaled conviction runs
    threshold_data = None
    if config.conviction_mode == "volatility_scaled":
        threshold_data = [
            {
                "timestamp": s.timestamp.strftime("%Y-%m-%d"),
                "value": s.threshold,
            }
            for s in output.snapshot_records
            if s.threshold is not None
        ]

    return {
        "run_id": output.run_id,
        "model_type": output.model_type,
        "model_name": output.model_name,
        "run_timestamp": output.run_timestamp.isoformat(),
        "training_duration_s": round(output.training_duration_s, 1),
        "backtest_duration_s": round(output.backtest_duration_s, 1),
        "config": {
            "test_start": config.test_start.strftime("%Y-%m-%d"),
            "test_end": config.test_end.strftime("%Y-%m-%d"),
            "training_start": config.training_start.strftime("%Y-%m-%d"),
            "training_end": config.training_end.strftime("%Y-%m-%d"),
            "market_condition": config.market_condition,
            "headline_only": config.headline_only,
            "rebalance_horizon_days": config.rebalance_horizon_days,
            "sentiment_ema_spans": config.sentiment_ema_spans,
            "transaction_cost_pct": config.transaction_cost_pct,
            "risk_free_rate": config.risk_free_rate,
            "stock_ids": config.stock_ids,
            "top_k": config.top_k,
            "min_predicted_return": config.min_predicted_return,
            "conviction_mode": config.conviction_mode,
            "early_stopping_patience": config.early_stopping_patience,
            "sequence_length": config.sequence_length,
            "retraining_cadence_days": config.retraining_cadence_days,
            "feature_set": config.feature_set,
        },
        "metrics": _metrics_dict(evaluation),
        "equity_curve": equity_curve,
        "stock_charts": stock_charts,
        "threshold_data": threshold_data,
    }


def _compute_held_intervals(
    stock_id: str,
    snapshots: list[PortfolioSnapshot],
    config: Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute held/not-held intervals for a stock using snapshot positions.

    Returns serializable dicts (not tuples) for JSON output.
    Needs timestamps from snapshots within the test period as the price axis.
    """
    # Use snapshot timestamps as the time axis for interval detection
    test_snapshots = [
        s for s in snapshots if config.test_start <= s.timestamp <= config.test_end
    ]
    if not test_snapshots:
        return [], []

    timestamps = [s.timestamp for s in test_snapshots]
    # Use portfolio value as a dummy price axis — the profitability calculation
    # compares start vs end price within each held interval. For the frontend,
    # the actual close prices come from shared stock price files. Here we just
    # need a price series that covers the same timestamps to detect intervals.
    # We'll use 1.0 as a placeholder and compute profitability from actual
    # close prices if available, but the interval boundaries are what matter.
    #
    # Actually — the held_intervals_from_snapshots function needs real prices
    # to determine is_profitable. We need the close prices here.
    # Load them from the curated stock_data CSV.
    prices = _load_close_prices_for_stock(stock_id, timestamps)

    held_raw, not_held_raw = _held_intervals_from_snapshots(
        stock_id, timestamps, prices, test_snapshots
    )

    held = [
        {
            "start": h[0].strftime("%Y-%m-%d"),
            "end": h[1].strftime("%Y-%m-%d"),
            "profitable": h[2],
            "closed": h[3],
        }
        for h in held_raw
    ]
    not_held = [
        {
            "start": nh[0].strftime("%Y-%m-%d"),
            "end": nh[1].strftime("%Y-%m-%d"),
        }
        for nh in not_held_raw
    ]
    return held, not_held


def _load_close_prices_for_stock(
    stock_id: str,
    timestamps: list[datetime],
) -> list[float]:
    """Load close prices from data/stock_data CSV for the given timestamps."""
    csv_path = STOCK_DATA_DIR / f"{stock_id}.csv"
    if not csv_path.exists():
        return [1.0] * len(timestamps)

    df = pl.read_csv(csv_path, try_parse_dates=True)
    df = df.select(["date", "close"]).drop_nulls().sort("date")

    price_lookup: dict[str, float] = {}
    for row in df.iter_rows():
        date_val = row[0]
        if hasattr(date_val, "strftime"):
            key = date_val.strftime("%Y-%m-%d")
        else:
            key = str(date_val)[:10]
        price_lookup[key] = float(row[1])

    return [price_lookup.get(ts.strftime("%Y-%m-%d"), 1.0) for ts in timestamps]


# ---------------------------------------------------------------------------
# Shared stock price files
# ---------------------------------------------------------------------------


def ensure_stock_prices(stock_ids: list[str]) -> None:
    """Ensure .stock_prices/{STOCK}.json exists for each stock.

    Reads from data/stock_data/{STOCK}.csv. Idempotent — skips if file exists.
    Contains full available price history so the frontend can slice to any test period.
    """
    STOCK_PRICES_DIR.mkdir(parents=True, exist_ok=True)

    for stock_id in stock_ids:
        out_path = STOCK_PRICES_DIR / f"{stock_id}.json"
        if out_path.exists():
            continue

        csv_path = STOCK_DATA_DIR / f"{stock_id}.csv"
        if not csv_path.exists():
            continue

        df = pl.read_csv(csv_path, try_parse_dates=True)
        df = df.select(["date", "close"]).drop_nulls().sort("date")

        prices = []
        for row in df.iter_rows():
            date_val = row[0]
            if hasattr(date_val, "strftime"):
                ts_str = date_val.strftime("%Y-%m-%d")
            else:
                ts_str = str(date_val)[:10]
            prices.append({"timestamp": ts_str, "close": round(float(row[1]), 2)})

        out_path.write_text(json.dumps(prices))


# ---------------------------------------------------------------------------
# Private helpers (module-level, stateless)
# ---------------------------------------------------------------------------


def _write_pipeline_input_cached(run_dir: Path, pipeline_input: PipelineInput) -> None:
    """Write pipeline_input.json with stock_data deduplicated into a shared cache."""
    dump = pipeline_input.model_dump(mode="json")
    stock_data_json = json.dumps(dump["stock_data"], sort_keys=True)
    stock_data_hash = hashlib.sha256(stock_data_json.encode()).hexdigest()[:16]

    INPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = INPUT_CACHE_DIR / f"{stock_data_hash}.json"
    if not cache_path.exists():
        cache_path.write_text(stock_data_json)

    ref = {
        "config": dump["config"],
        "initial_capital": dump["initial_capital"],
        "stock_data_ref": stock_data_hash,
    }
    (run_dir / "pipeline_input.json").write_text(json.dumps(ref, indent=2))


def _make_run_dir(output: PipelineOutput) -> Path:
    model_dir = EXECUTION_OUTPUT_ROOT / output.model_type
    date_str = output.run_timestamp.strftime("%Y-%m-%d")
    base = f"{date_str}_{output.config.execution_name}"

    counter = 0
    if model_dir.exists():
        prefix = f"{base}_"
        for d in model_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                suffix = d.name[len(prefix) :]
                if suffix.isdigit():
                    counter = max(counter, int(suffix))

    return model_dir / f"{base}_{counter + 1}"


def build_index_entry(
    run_dir: Path,
    output: PipelineOutput,
    evaluation: PipelineEvaluation,
) -> dict[str, Any]:
    config = output.config
    rel_dir = f"{run_dir.parent.name}/{run_dir.name}"
    return {
        "run_id": output.run_id,
        "run_dir": rel_dir,
        "model_type": output.model_type,
        "model_name": output.model_name,
        "execution_name": config.execution_name,
        "description": config.description,
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "timestamp": output.run_timestamp.isoformat(),
        "training_duration_s": round(output.training_duration_s, 1),
        "backtest_duration_s": round(output.backtest_duration_s, 1),
        "config_summary": {
            "test_start": config.test_start.strftime("%Y-%m-%d"),
            "test_end": config.test_end.strftime("%Y-%m-%d"),
            "market_condition": config.market_condition,
            "headline_only": config.headline_only,
            "rebalance_horizon_days": config.rebalance_horizon_days,
            "stock_ids": config.stock_ids,
            "sentiment_ema_spans": config.sentiment_ema_spans,
            "transaction_cost_pct": config.transaction_cost_pct,
            "top_k": config.top_k,
            "min_predicted_return": config.min_predicted_return,
            "conviction_mode": config.conviction_mode,
            "early_stopping_patience": config.early_stopping_patience,
            "retraining_cadence_days": config.retraining_cadence_days,
            "sequence_length": config.sequence_length,
            "feature_set": config.feature_set,
        },
        "metrics": _metrics_dict(evaluation),
    }


def _metrics_dict(evaluation: PipelineEvaluation) -> dict[str, Any]:
    """Shared metrics dict used by both index entries and run_detail.json."""
    return {
        "rmse": evaluation.rmse,
        "mae": evaluation.mae,
        "mape": evaluation.mape,
        "directional_accuracy": evaluation.directional_accuracy,
        "tp": evaluation.tp,
        "fp": evaluation.fp,
        "tn": evaluation.tn,
        "fn": evaluation.fn,
        "sharpe_ratio": evaluation.sharpe_ratio,
        "sortino_ratio": evaluation.sortino_ratio,
        "total_return": evaluation.total_return,
        "annualized_return": evaluation.annualized_return,
        "maximum_drawdown": evaluation.maximum_drawdown,
        "cvar": evaluation.CVar,
        "volatility": evaluation.volatility,
        "downside_volatility": evaluation.downside_volatility,
        "win_ratio": evaluation.win_ratio,
        "total_trades": int(evaluation.total_trades),
        "avg_trade_return": evaluation.avg_trade_return,
        "total_transaction_cost": evaluation.total_transaction_cost,
    }


def _append_to_runs_index(
    run_dir: Path,
    output: PipelineOutput,
    evaluation: PipelineEvaluation,
) -> None:
    entry = build_index_entry(run_dir, output, evaluation)

    with locked_index():
        index: list[dict[str, Any]] = []
        if RUNS_INDEX_PATH.exists():
            index = json.loads(RUNS_INDEX_PATH.read_text())

        index.append(entry)

        tmp = RUNS_INDEX_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.rename(RUNS_INDEX_PATH)


def delete_runs(run_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Bulk delete. Returns (deleted_run_dirs, not_found_run_ids, failed_run_ids).

    Order matters: rmtree FIRST, then drop the index entry. If rmtree raises,
    the entry stays in the index and the run_id lands in `failed` — surfaces
    via the API for the UI to display. Avoids the orphan-on-disk-without-
    index-entry failure mode.
    """
    deleted: list[str] = []
    not_found: list[str] = []
    failed: list[str] = []

    with locked_index():
        if not RUNS_INDEX_PATH.exists():
            return [], run_ids, []

        index: list[dict[str, Any]] = json.loads(RUNS_INDEX_PATH.read_text())
        target_ids = set(run_ids)
        keep: list[dict[str, Any]] = []
        to_delete: list[dict[str, Any]] = []

        for entry in index:
            if entry.get("run_id") in target_ids:
                to_delete.append(entry)
            else:
                keep.append(entry)

        present_ids = {e["run_id"] for e in to_delete}
        not_found = [rid for rid in run_ids if rid not in present_ids]

        root = EXECUTION_OUTPUT_ROOT.resolve()
        for entry in to_delete:
            run_dir_str = entry["run_dir"]
            run_dir = EXECUTION_OUTPUT_ROOT / run_dir_str
            # Path-traversal guard: the resolved path MUST live inside the
            # execution root. A hand-edited index entry with `../...` would
            # otherwise rmtree arbitrary directories.
            try:
                resolved = run_dir.resolve()
                if not resolved.is_relative_to(root):
                    raise OSError(f"run_dir escapes execution root: {run_dir_str}")
                if resolved.exists():
                    shutil.rmtree(resolved)
                deleted.append(run_dir_str)
            except OSError:
                failed.append(entry["run_id"])
                keep.append(entry)  # entry stays in index — UI sees it failed

        tmp = RUNS_INDEX_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(keep, indent=2))
        tmp.rename(RUNS_INDEX_PATH)

    return deleted, not_found, failed


def _format_metrics(output: PipelineOutput, evaluation: PipelineEvaluation) -> str:
    e = evaluation
    return f"""# Run Metrics: {output.model_name}

- **Run ID:** {output.run_id}
- **Execution:** {output.config.execution_name}
- **Timestamp:** {output.run_timestamp:%Y-%m-%d %H:%M}
- **Test Period:** {output.config.test_start:%Y-%m-%d} to {output.config.test_end:%Y-%m-%d}
- **Stocks:** {", ".join(output.stocks)}

## Error Metrics
| Metric | Value |
|--------|-------|
| RMSE | {e.rmse:.6f} |
| MAE | {e.mae:.6f} |
| MAPE | {e.mape:.4%} |

## Directional Accuracy
| Metric | Value |
|--------|-------|
| Accuracy | {e.directional_accuracy:.4%} |
| TP / FP | {e.tp} / {e.fp} |
| TN / FN | {e.tn} / {e.fn} |

## Returns & Risk
| Metric | Value |
|--------|-------|
| Total Return | {e.total_return:.4%} |
| Annualized Return | {e.annualized_return:.4%} |
| Sharpe Ratio | {e.sharpe_ratio:.4f} |
| Sortino Ratio | {e.sortino_ratio:.4f} |
| Max Drawdown | {e.maximum_drawdown:.4%} |
| CVaR (95%) | {e.CVar:.6f} |
| Volatility | {e.volatility:.4f} |
| Downside Volatility | {e.downside_volatility:.4f} |

## Trade Statistics
| Metric | Value |
|--------|-------|
| Total Trades | {int(e.total_trades)} |
| Win Ratio | {e.win_ratio:.4%} |
| Avg Trade Return | {e.avg_trade_return:.4%} |
| Transaction Costs | {e.total_transaction_cost:.4f} |
"""
