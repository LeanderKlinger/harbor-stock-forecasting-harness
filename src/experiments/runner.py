"""
Experiment grid runner.

Parses a YAML experiment config into typed Pydantic models, computes the
cartesian product of grid axes, runs each (config, model, seed) combination
through the full pipeline, and produces a comparison report at the end.

Supports parallel execution via --workers N. Each outer worker runs one
(model, config, seed) combo. Inner per-stock parallelism (in TrainingModule)
coexists — the runner computes inner_workers = cores // outer_workers.

Usage:
    python -m src.experiments run experiments/my_grid.yaml
    python -m src.experiments run experiments/my_grid.yaml --workers 60 --device cpu
"""

import dataclasses
import itertools
import multiprocessing
import os
import queue
import random
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from src.evaluation.evaluation_module import EvaluationModule
from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.evaluation import PipelineEvaluation
from src.models.input import PipelineInput
from src.models.output import PipelineOutput
from src.training.strategies.arima import ARIMAStrategy
from src.training.strategies.baselines import (
    BuyAndHoldStrategy,
    EqualWeightStrategy,
    MomentumStrategy,
)
from src.training.strategies.gru import GRUStrategy
from src.training.strategies.holts import HoltsStrategy
from src.training.strategies.jamba_strategy import JambaStrategy
from src.training.strategies.mamba_strategy import MambaStrategy
from src.training.strategies.mlp import MLPStrategy
from src.training.strategies.xlstm import XLSTMStrategy
from src.training.training_module import TrainingModule

MODEL_REGISTRY: dict[str, type] = {
    "xLSTM": XLSTMStrategy,
    "GRU": GRUStrategy,
    "MLP": MLPStrategy,
    "Mamba": MambaStrategy,
    "Jamba": JambaStrategy,
    "ARIMA": ARIMAStrategy,
    "HoltsLES": HoltsStrategy,
    "BuyAndHold": BuyAndHoldStrategy,
    "EqualWeight": EqualWeightStrategy,
    "Momentum": MomentumStrategy,
}

_TORCH_MODELS = {"xLSTM", "GRU", "MLP", "Mamba", "Jamba"}

# Baselines that need top_k = stock count (they hold all stocks, not top_k)
_ALL_STOCKS_BASELINES = {"BuyAndHold", "EqualWeight"}

# Defaults for non-neural models — visible at module level, not buried in functions
DEFAULT_ARIMA_ORDER: tuple[int, int, int] = (5, 1, 2)
DEFAULT_MOMENTUM_LOOKBACK: int = 20


def _ts() -> str:
    return datetime.now().strftime("[%H:%M:%S]")


def _log(msg: str) -> None:
    # flush=True is the reliable fix for PYTHONUNBUFFERED-via-uv on macOS.
    # Use tqdm.write() instead of this when inside an active tqdm bar.
    print(f"{_ts()} {msg}", flush=True)


def _format_run_cfg(spec: "RunSpec") -> str:
    """Compact one-line config summary for ERROR / WARN log lines — only the grid-varying knobs.

    Stock list is omitted (too long for a log line); look it up in the YAML by run_num.
    """
    cfg = spec.config
    return (
        f"{spec.model_entry.type} seed={spec.seed} "
        f"horizon={cfg.rebalance_horizon_days} "
        f"top_k={cfg.top_k} "
        f"cadence={cfg.retraining_cadence_days!r} "
        f"conviction={cfg.conviction_mode} "
        f"features={cfg.feature_set}"
    )


# Disable the live bar when stderr isn't a terminal (e.g. piped to tee + log file
# during remote runs). Without this, tqdm's per-update bar redraws end up
# concatenated with \r separators in the log, making the file unreadable.
# tqdm.write() lines still flow through cleanly either way.
_TQDM_DISABLE = not sys.stderr.isatty()


def _make_run_bar(*, total: int | None = None, iterable: Any = None) -> Any:
    return tqdm(
        iterable,
        total=total,
        desc="runs",
        unit="run",
        dynamic_ncols=True,
        disable=_TQDM_DISABLE,
    )


# ---------------------------------------------------------------------------
# YAML schema as Pydantic models
# ---------------------------------------------------------------------------


class BaseConfig(BaseModel):
    """Maps to the `base:` section of the YAML."""

    training_start: datetime
    training_end: datetime
    test_start: datetime
    test_end: datetime
    headline_only: bool = True
    stock_ids: list[str]
    transaction_cost_pct: float = 0.0
    min_predicted_return: float = 0.0
    conviction_mode: str = "fixed"
    top_k: int = 4
    sequence_length: int = 60
    sentiment_ema_spans: list[int] = [5, 10, 20]
    retraining_cadence_days: int | Literal["auto"] | None = None
    feature_set: Literal["full", "numeric", "close_only"] = "full"


class ModelDefaults(BaseModel):
    """Maps to the `model_defaults:` section — shared neural model kwargs."""

    hidden_size: int = 64
    num_layers: int = 2
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout: float = 0.2
    early_stopping_patience: int | None = None


class ModelEntry(BaseModel):
    """A single model in the grid. `type` is required, rest are overrides."""

    type: str
    # Per-model overrides (merged on top of model_defaults)
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    dropout: float | None = None
    # Non-neural model params
    lookback: int | None = None
    order: list[int] | None = None


class GridConfig(BaseModel):
    """Maps to the `grid:` section.

    Only `model` is a typed field. Everything else is YAML-driven:
    any extra field listed under `grid:` becomes a grid axis.
    List values are crossed (cartesian product), scalars become
    single-value axes.
    """

    model_config = ConfigDict(extra="allow")
    model: list[ModelEntry]

    def config_axes(self) -> dict[str, list[Any]]:
        """Extract grid axes as {key: [values]} for cartesian product."""
        axes: dict[str, list[Any]] = {}
        for key, val in self.model_dump(exclude={"model"}).items():
            if isinstance(val, list):
                axes[key] = val
            else:
                axes[key] = [val]
        return axes


class ExperimentSpec(BaseModel):
    """Top-level YAML schema."""

    name: str
    description: str = ""
    base: BaseConfig
    model_defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    grid: GridConfig
    seeds: list[int] = [42]


# ---------------------------------------------------------------------------
# RunSpec — immutable work unit for one (config, model, seed) combination
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """Everything a worker needs to execute one run. Immutable, self-contained.

    Each RunSpec has its own Config copy — no shared mutable state between runs.
    The extraction_cache_key maps to the correct PipelineInput for this config.
    """

    config: Config
    model_entry: ModelEntry
    model_defaults: ModelDefaults
    seed: int
    run_num: int
    total_runs: int
    device: str | None
    is_deterministic: bool
    extraction_cache_key: str
    max_inner_workers: int | None
    no_cache: bool


# Timing record: (model_type, rebalance_horizon_days, compute_seconds)
TimingRecord = tuple[str, int, float]


def _print_timing_summary(records: list[TimingRecord]) -> None:
    """Print a table of mean compute time per (model_type, horizon)."""
    if not records:
        return

    # Group by (model_type, horizon) → list of elapsed times
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for model_type, horizon, elapsed in records:
        grouped[(model_type, horizon)].append(elapsed)

    # Discover all unique horizons and model types (sorted)
    all_horizons = sorted(set(h for _, h in grouped))
    all_models = sorted(set(m for m, _ in grouped))

    # Build the table
    # Header: Model, H=1, H=5, H=20, ..., Avg
    horizon_labels = [f"H={h}" for h in all_horizons]
    header = (
        f"{'Model':<14}" + "".join(f"{h:>10}" for h in horizon_labels) + f"{'Avg':>10}"
    )
    separator = "─" * len(header)

    print(f"\n{separator}")
    print("Timing Summary (mean compute seconds per run)")
    print(separator)
    print(header)
    print(separator)

    for model in all_models:
        row = f"{model:<14}"
        all_times: list[float] = []
        for horizon in all_horizons:
            times = grouped.get((model, horizon))
            if times:
                mean_t = statistics.mean(times)
                all_times.extend(times)
                row += f"{mean_t:>9.1f}s"
            else:
                row += f"{'—':>10}"
        if all_times:
            row += f"{statistics.mean(all_times):>9.1f}s"
        else:
            row += f"{'—':>10}"
        print(row)

    print(separator)


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


def _instantiate_model(
    entry: ModelEntry,
    defaults: ModelDefaults,
    config: Config,
) -> Any:
    """Create a model instance from the registry with merged kwargs."""
    model_type = entry.type
    cls = MODEL_REGISTRY[model_type]

    if model_type in _TORCH_MODELS:
        kwargs: dict[str, Any] = {"config": config}
        # Start with defaults
        kwargs["hidden_size"] = defaults.hidden_size
        kwargs["num_layers"] = defaults.num_layers
        kwargs["epochs"] = defaults.epochs
        kwargs["batch_size"] = defaults.batch_size
        kwargs["learning_rate"] = defaults.learning_rate
        kwargs["dropout"] = defaults.dropout
        # Apply per-model overrides (only non-None)
        if entry.hidden_size is not None:
            kwargs["hidden_size"] = entry.hidden_size
        if entry.num_layers is not None:
            kwargs["num_layers"] = entry.num_layers
        if entry.epochs is not None:
            kwargs["epochs"] = entry.epochs
        if entry.batch_size is not None:
            kwargs["batch_size"] = entry.batch_size
        if entry.learning_rate is not None:
            kwargs["learning_rate"] = entry.learning_rate
        if entry.dropout is not None:
            kwargs["dropout"] = entry.dropout
        # Model-specific neural kwargs
        if entry.num_attention_heads is not None:
            kwargs["num_attention_heads"] = entry.num_attention_heads
        return cls(**kwargs)

    if model_type == "ARIMA":
        order = tuple(entry.order) if entry.order else DEFAULT_ARIMA_ORDER
        return cls(order=order)  # pyright: ignore[reportCallIssue]
    if model_type == "HoltsLES":
        return cls()
    if model_type == "Momentum":
        return cls(lookback=entry.lookback or DEFAULT_MOMENTUM_LOOKBACK)
    # BuyAndHold, EqualWeight
    return cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(spec: ExperimentSpec, overrides: dict[str, Any]) -> Config:
    """Build a Config from the spec's base section + grid overrides."""
    base_dict = spec.base.model_dump()
    base_dict.update(overrides)
    base_dict["early_stopping_patience"] = spec.model_defaults.early_stopping_patience
    return Config(**base_dict)


def _extraction_cache_key(config: Config) -> str:
    """Cache key for extraction-affecting config fields only.

    Extraction depends on: which stocks, date range, sentiment mode/model, and EMA spans.
    Grid axes like rebalance_horizon_days, transaction_cost_pct, etc. do NOT affect
    extraction and must not appear here — otherwise identical extractions get duplicated.
    """
    key_fields = {
        "stock_ids": tuple(sorted(config.stock_ids)),
        "headline_only": config.headline_only,
        "semantic_model": config.semantic_model,
        "sentiment_ema_spans": tuple(config.sentiment_ema_spans),
        "training_start": config.training_start.isoformat(),
        "test_end": config.test_end.isoformat(),
    }
    return str(sorted(key_fields.items()))


def _seed_all_rngs(seed: int) -> None:
    """Seed all RNG sources for reproducibility.

    torch.manual_seed covers PyTorch (weight init, DataLoader shuffle).
    numpy/random are defensive — not currently used in the pipeline, but
    protects against statsmodels or future code introducing numpy randomness.
    """
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    np.random.seed(seed)  # pyright: ignore[reportUnknownMemberType]
    random.seed(seed)


def _is_known_infeasible(model_type: str, config: Config) -> bool:
    """ARIMA under "auto" cadence at h=1 refits daily and deterministically hits
    LU decomposition failures (canonical exclusion) — each attempt
    burns 30-90 min before dying. Skip up front with a WARN instead of launching
    guaranteed failures, so grids can include ARIMA at all horizons naturally.
    Escape hatch: an explicit `retraining_cadence_days: 1` still runs (only the
    "auto" shorthand is refused — "auto" promises a *sensible* per-rebalance
    refit, and daily ARIMA re-estimation is known not to be one).
    """
    return (
        model_type == "ARIMA"
        and config.retraining_cadence_days == "auto"
        and config.rebalance_horizon_days == 1
    )


def _count_runs(spec: ExperimentSpec, config_combos: list[dict[str, Any]]) -> int:
    """Compute total runs: deterministic models need only 1 seed; known-infeasible
    (model, config) cells are skipped and must not inflate the progress total."""
    total = 0
    for config_overrides in config_combos:
        config = _build_config(spec, config_overrides)
        for entry in spec.grid.model:
            if _is_known_infeasible(entry.type, config):
                continue
            if entry.type in _TORCH_MODELS:
                total += len(spec.seeds)
            else:
                total += 1
    return total


def _build_run_specs(
    spec: ExperimentSpec,
    config_combos: list[dict[str, Any]],
    total_runs: int,
    device: str | None,
    max_inner_workers: int | None = None,
    no_cache: bool = False,
) -> tuple[list[RunSpec], dict[str, Config]]:
    """Build all RunSpecs upfront + collect extraction configs.

    Returns:
        run_specs: Ordered list of all runs to execute.
        extraction_configs: Map of extraction_cache_key -> Config suitable
            for ExtractionModule. One entry per unique extraction config
            (typically 2 for the big grid — one per EMA setting).
    """
    extraction_configs: dict[str, Config] = {}
    run_specs: list[RunSpec] = []
    run_num = 0

    for config_overrides in config_combos:
        base_config = _build_config(spec, config_overrides)
        cache_key = _extraction_cache_key(base_config)

        if cache_key not in extraction_configs:
            extraction_configs[cache_key] = base_config

        for model_entry in spec.grid.model:
            model_type = model_entry.type
            if _is_known_infeasible(model_type, base_config):
                _log(
                    f"  [skip] WARN {model_type} horizon={base_config.rebalance_horizon_days} "
                    f"cadence={base_config.retraining_cadence_days!r}: known-infeasible cell (deterministic LU failure) — excluded up front. "
                    "Set retraining_cadence_days: 1 explicitly to force."
                )
                continue
            is_deterministic = model_type not in _TORCH_MODELS
            seeds = [spec.seeds[0]] if is_deterministic else spec.seeds

            for seed in seeds:
                run_num += 1

                # Build the run description
                desc = f"{spec.description} | {model_type} seed={seed}"
                if config_overrides:
                    override_str = " ".join(
                        f"{k}={v}" for k, v in config_overrides.items()
                    )
                    desc += f" | {override_str}"

                # Build run-specific config as an independent copy
                run_updates: dict[str, Any] = {
                    "experiment_name": spec.name,
                    "execution_name": spec.name,
                    "seed": seed,
                    "description": desc,
                }

                # Baseline overrides: hold all stocks, BuyAndHold never rebalances
                if model_type in _ALL_STOCKS_BASELINES:
                    run_updates["top_k"] = len(base_config.stock_ids)
                    if model_type == "BuyAndHold":
                        n_test_days = (
                            base_config.test_end - base_config.test_start
                        ).days
                        run_updates["rebalance_horizon_days"] = n_test_days

                run_config = base_config.model_copy(update=run_updates)

                run_specs.append(
                    RunSpec(
                        config=run_config,
                        model_entry=model_entry,
                        model_defaults=spec.model_defaults,
                        seed=seed,
                        run_num=run_num,
                        total_runs=total_runs,
                        device=device,
                        is_deterministic=is_deterministic,
                        extraction_cache_key=cache_key,
                        max_inner_workers=max_inner_workers,
                        no_cache=no_cache,
                    )
                )

    return run_specs, extraction_configs


# ---------------------------------------------------------------------------
# Worker function (used by both sequential and parallel paths)
# ---------------------------------------------------------------------------


def execute_run(
    run_spec: RunSpec,
    pipeline_input: PipelineInput,
) -> tuple[PipelineOutput, PipelineEvaluation]:
    """Execute one (model, config, seed) combo. Pure function — no IO, no persistence.

    Seeds RNGs, instantiates model, trains, backtests, evaluates.
    Suitable for both sequential and parallel execution.
    """
    _seed_all_rngs(run_spec.seed)

    model = _instantiate_model(
        run_spec.model_entry, run_spec.model_defaults, run_spec.config
    )
    training = TrainingModule(
        model,
        run_spec.config,
        device_override=run_spec.device,
        max_inner_workers=run_spec.max_inner_workers,
        no_cache=run_spec.no_cache,
    )
    training.train(pipeline_input)
    output = training.backtest(pipeline_input)
    evaluation = EvaluationModule().evaluate(output)
    return output, evaluation


# ---------------------------------------------------------------------------
# Parallel execution infrastructure
# ---------------------------------------------------------------------------

# Module-level global for worker processes — set by _init_pool_worker
_worker_extraction_cache: dict[str, PipelineInput] = {}


def _init_pool_worker(extraction_cache: dict[str, PipelineInput]) -> None:
    """Pool initializer: sets up shared data in each worker process.

    Also limits PyTorch thread count to 1 — tiny models (hidden=64) are
    faster single-threaded, and N outer workers with default threads
    causes catastrophic OpenMP oversubscription.
    """
    global _worker_extraction_cache
    _worker_extraction_cache = extraction_cache
    torch.set_num_threads(1)  # pyright: ignore[reportUnknownMemberType]


def _execute_run_in_pool(
    run_spec: RunSpec,
) -> tuple[PipelineOutput, PipelineEvaluation]:
    """Thin wrapper for ProcessPoolExecutor. Looks up PipelineInput from
    the module global set by _init_pool_worker."""
    pipeline_input = _worker_extraction_cache[run_spec.extraction_cache_key]
    return execute_run(run_spec, pipeline_input)


# Type alias for the persistence queue items
_PersistItem = tuple[PipelineInput, PipelineOutput, PipelineEvaluation]


def _persistence_worker(
    persist_queue: queue.Queue[_PersistItem | None],
    eval_module: EvaluationModule,
) -> None:
    """Writer thread: sequentially persists completed runs. Consumes from
    queue until it receives None (sentinel). Runs in the main process —
    no races on runs_index.json."""
    while True:
        item = persist_queue.get()
        if item is None:
            break
        pipeline_input, output, evaluation = item
        eval_module.persist(pipeline_input, output, evaluation)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_experiment(
    yaml_path: Path,
    device: str | None = None,
    workers: int = 1,
    no_cache: bool = False,
) -> list[str]:
    """
    Execute a full experiment grid from a YAML config file.

    Three phases:
      1. EXTRACT: Build PipelineInputs for each unique extraction config.
      2. EXECUTE: Run each (model, config, seed) combo, persist results.
      3. FINALIZE: Regenerate leaderboards, generate comparison report.

    Args:
        workers: Number of outer parallel workers. 1 = sequential (no pool overhead).
        no_cache: Disable the per-stock training cache (force fresh fits).
    """
    with open(yaml_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    spec = ExperimentSpec.model_validate(raw)

    # Compute cartesian product of config overrides
    config_axes = spec.grid.config_axes()
    if config_axes:
        axis_keys = list(config_axes.keys())
        axis_values = [config_axes[k] for k in axis_keys]
        config_combos: list[dict[str, Any]] = [
            dict(zip(axis_keys, combo)) for combo in itertools.product(*axis_values)
        ]
    else:
        config_combos = [{}]

    total_runs = _count_runs(spec, config_combos)

    # Compute inner worker count for per-stock parallelism coexistence
    cores = os.cpu_count() or 1
    if workers > 1:
        inner_workers: int | None = max(1, cores // workers)
    else:
        inner_workers = None  # sequential: use all cores for inner parallelism

    # Build all RunSpecs upfront + identify unique extraction configs
    run_specs, extraction_configs = _build_run_specs(
        spec,
        config_combos,
        total_runs,
        device,
        max_inner_workers=inner_workers,
        no_cache=no_cache,
    )

    _log(f"{'=' * 60}")
    _log(f"Experiment: {spec.name}")
    _log(f"Description: {spec.description}")
    _log(f"Models: {[m.type for m in spec.grid.model]}")
    _log(f"Config combos: {len(config_combos)}")
    _log(f"Unique extractions: {len(extraction_configs)}")
    _log(f"Seeds: {spec.seeds}")
    _log(f"Total runs: {total_runs}")
    _log(f"Workers: {workers} outer × {inner_workers or cores} inner ({cores} cores)")
    if device:
        _log(f"Device override: {device}")
    _log(f"{'=' * 60}")

    # --- Phase 1: EXTRACT ---
    _log("[extraction] Phase start")
    t_extract_start = time.perf_counter()
    extraction_cache: dict[str, PipelineInput] = {}
    for cache_key, config in extraction_configs.items():
        _log(f"[extraction] Building pipeline input ({cache_key[:60]}...)")
        extraction = ExtractionModule(config)
        extraction_cache[cache_key] = extraction.build_pipeline_input()
    _log(f"[extraction] Phase done in {time.perf_counter() - t_extract_start:.1f}s")

    # --- Phase 2: EXECUTE + PERSIST ---
    _log(f"[execute] Phase start ({total_runs} runs, {workers} outer workers)")
    eval_module = EvaluationModule()
    t_experiment_start = time.perf_counter()

    if workers <= 1:
        run_ids, _, failed_runs, timing_records = _run_sequential(
            run_specs, extraction_cache, eval_module
        )
    else:
        run_ids, _, failed_runs, timing_records = _run_parallel(
            run_specs, extraction_cache, eval_module, workers, device
        )

    experiment_elapsed = time.perf_counter() - t_experiment_start
    _log(f"[execute] Phase done in {experiment_elapsed / 3600:.2f}h")

    # --- Phase 3: FINALIZE ---
    _log("[finalize] Phase start")
    _print_timing_summary(timing_records)

    _log(f"{'=' * 60}")
    _log(f"Experiment '{spec.name}' complete")
    _log(f"  {len(run_ids)} succeeded, {failed_runs} failed")
    _log(f"  Total wall time: {experiment_elapsed / 3600:.2f} hours")
    _log(f"{'=' * 60}")

    return run_ids


# ---------------------------------------------------------------------------
# Sequential execution (--workers 1)
# ---------------------------------------------------------------------------


def _run_sequential(
    run_specs: list[RunSpec],
    extraction_cache: dict[str, PipelineInput],
    eval_module: EvaluationModule,
) -> tuple[list[str], set[str], int, list[TimingRecord]]:
    """Execute all runs sequentially. No pool overhead."""
    run_ids: list[str] = []
    model_types_seen: set[str] = set()
    timing_records: list[TimingRecord] = []
    failed = 0

    bar = _make_run_bar(iterable=run_specs)
    for run_spec in bar:
        pipeline_input = extraction_cache[run_spec.extraction_cache_key]
        det_tag = " (deterministic)" if run_spec.is_deterministic else ""
        bar.set_postfix(model=run_spec.model_entry.type, seed=run_spec.seed)  # pyright: ignore[reportUnknownMemberType]
        tqdm.write(
            f"{_ts()} [{run_spec.run_num}/{run_spec.total_runs}] "
            f"{run_spec.model_entry.type} seed={run_spec.seed}{det_tag}"
        )

        try:
            t0 = time.perf_counter()
            output, evaluation = execute_run(run_spec, pipeline_input)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            tqdm.write(f"{_ts()}   ERROR {_format_run_cfg(run_spec)}: {e}")
            tqdm.write(traceback.format_exc())
            failed += 1
            continue

        run_dir = eval_module.persist(pipeline_input, output, evaluation)

        compute_s = output.training_duration_s + output.backtest_duration_s
        timing_records.append(
            (
                run_spec.model_entry.type,
                run_spec.config.rebalance_horizon_days,
                compute_s,
            )
        )

        model_types_seen.add(run_spec.model_entry.type)
        run_ids.append(output.run_id)
        bar.set_postfix(  # pyright: ignore[reportUnknownMemberType]
            model=run_spec.model_entry.type,
            seed=run_spec.seed,
            Sortino=f"{evaluation.sortino_ratio:.3f}",
            elapsed=f"{elapsed:.0f}s",
        )
        tqdm.write(
            f"{_ts()}   Sortino={evaluation.sortino_ratio:.4f} "
            f"Return={evaluation.total_return:.2%} "
            f"({elapsed:.1f}s) -> {run_dir.name}"
        )

    bar.close()
    return run_ids, model_types_seen, failed, timing_records


# ---------------------------------------------------------------------------
# Parallel execution (--workers N, N > 1)
# ---------------------------------------------------------------------------


def _run_parallel(
    run_specs: list[RunSpec],
    extraction_cache: dict[str, PipelineInput],
    eval_module: EvaluationModule,
    workers: int,
    device: str | None = None,
) -> tuple[list[str], set[str], int, list[TimingRecord]]:
    """Execute runs in parallel via ProcessPoolExecutor.

    A writer thread persists results as they complete — no races on
    runs_index.json, and results are safe on disk even if the process
    is interrupted.
    """
    run_ids: list[str] = []
    model_types_seen: set[str] = set()
    timing_records: list[TimingRecord] = []
    failed = 0
    completed = 0
    total = len(run_specs)

    mp_ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=_init_pool_worker,
        initargs=(extraction_cache,),
    )

    # Start writer thread AFTER pool creation (safe — no active threads during fork)
    persist_queue: queue.Queue[_PersistItem | None] = queue.Queue()
    writer = threading.Thread(
        target=_persistence_worker,
        args=(persist_queue, eval_module),
        daemon=True,
    )
    writer.start()

    t_start = time.perf_counter()

    bar = _make_run_bar(total=total)

    with pool:
        futures = {pool.submit(_execute_run_in_pool, spec): spec for spec in run_specs}

        for future in as_completed(futures):
            spec = futures[future]
            completed += 1

            try:
                output, evaluation = future.result()
            except Exception as e:
                failed += 1
                bar.update(1)
                bar.set_postfix(failed=failed, last=spec.model_entry.type)  # pyright: ignore[reportUnknownMemberType]
                tqdm.write(
                    f"{_ts()}   [{completed}/{total}] ERROR {_format_run_cfg(spec)}: {e}"
                )
                # tqdm.write — not traceback.print_exc() — to avoid corrupting the bar
                tqdm.write(traceback.format_exc())
                continue

            # Hand off to writer thread for persistence
            pipeline_input = extraction_cache[spec.extraction_cache_key]
            persist_queue.put((pipeline_input, output, evaluation))

            compute_s = output.training_duration_s + output.backtest_duration_s
            timing_records.append(
                (
                    spec.model_entry.type,
                    spec.config.rebalance_horizon_days,
                    compute_s,
                )
            )

            model_types_seen.add(spec.model_entry.type)
            run_ids.append(output.run_id)

            elapsed_total = time.perf_counter() - t_start
            # avg_wall already reflects parallelism (it's wall time / completions,
            # not CPU time). Don't divide by workers again.
            throughput = completed / elapsed_total  # runs per second
            remaining_est = (total - completed) / throughput if throughput > 0 else 0

            bar.update(1)
            bar.set_postfix(  # pyright: ignore[reportUnknownMemberType]
                last=spec.model_entry.type,
                Sortino=f"{evaluation.sortino_ratio:.3f}",
                ETA=f"{remaining_est / 3600:.1f}h",
            )
            tqdm.write(
                f"{_ts()}   [done {completed}/{total}] {spec.model_entry.type} "
                f"seed={spec.seed} | "
                f"Sortino={evaluation.sortino_ratio:.4f} "
                f"Return={evaluation.total_return:.2%} | "
                f"{compute_s:.1f}s | "
                f"{elapsed_total / 3600:.1f}h elapsed ~{remaining_est / 3600:.1f}h remaining"
            )

    bar.close()

    # Drain the writer thread
    persist_queue.put(None)
    writer.join()

    return run_ids, model_types_seen, failed, timing_records
