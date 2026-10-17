import copy
import math
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch

from src.models.config import Config
from src.models.input import PipelineInput, StockDataPairings
from src.models.output import (
    Direction,
    PipelineOutput,
    PredictionRecord,
    PortfolioSnapshot,
    TradeRecord,
)
from src.training.portfolio import PortfolioState
from src.training.protocols import PredictionModel
from src.training.training_cache import TrainingCacheKey, cache_path_for

# Minimum stocks to justify process pool spawn overhead
_MIN_STOCKS_FOR_PARALLEL = 3


def _fit_stock(
    model: PredictionModel,
    training_data: list[StockDataPairings],
    seed: int,
    cache_key: TrainingCacheKey | None,
) -> tuple[PredictionModel, bool]:
    """Returns (fitted model, cache_hit). Hit reporting must come from here —
    checking cache files post-hoc always sees the files this run just wrote."""
    # set_num_threads(1) avoids OpenMP oversubscription when this runs in a process pool.
    torch.set_num_threads(1)  # pyright: ignore[reportUnknownMemberType]
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    if cache_key is not None and hasattr(model, "restore_fit"):
        if model.restore_fit(cache_path_for(cache_key)):  # type: ignore[union-attr]
            return model, True
    model.fit(training_data)
    if cache_key is not None and hasattr(model, "cache_fit"):
        model.cache_fit(cache_path_for(cache_key))  # type: ignore[union-attr]
    return model, False


class TrainingModule:
    """
    Walk-forward backtesting engine with multi-stock rebalancing.

    Usage:
        module = TrainingModule(model, config)
        module.train(pipeline_input)    # fit one model clone per stock
        output = module.backtest(pipeline_input)  # walk-forward with rebalancing
    """

    def __init__(
        self,
        model: PredictionModel,
        config: Config,
        vol_window: int = 20,
        device_override: str | None = None,
        max_inner_workers: int | None = None,
        no_cache: bool = False,
    ):
        self.model = model
        self.config = config
        self.vol_window = vol_window
        self.device_override = device_override
        self.max_inner_workers = max_inner_workers
        self.no_cache = no_cache
        self._models: dict[str, PredictionModel] = {}

        # Volatility-scaled conviction tracking
        self._daily_returns: list[float] = []
        self._rolling_vols: list[float] = []

        # Timing
        self.training_duration_s: float = 0.0
        self.backtest_duration_s: float = 0.0

    def train(self, pipeline_input: PipelineInput) -> None:
        """
        Fit one model clone per stock on training data.

        Deepcopies self.model for each stock sequentially (preserves
        torch.manual_seed determinism), then dispatches fit() in parallel
        via ProcessPoolExecutor for neural models with enough stocks.
        """
        t0 = time.perf_counter()

        # Phase 1: prepare clones + training data + cache key sequentially
        # (deepcopy order determines random init — must be deterministic)
        work_items: list[
            tuple[
                str, PredictionModel, list[StockDataPairings], TrainingCacheKey | None
            ]
        ] = []
        # Stat models (no _model attribute) get seed=None so all seeds share one cache entry —
        # they're deterministic, no point keying on the runner's nominal seed.
        is_neural = hasattr(self.model, "_model")
        for stock_data in pipeline_input.stock_data:
            stock_id = stock_data.stock_id
            all_pairings = sorted(stock_data.data, key=lambda x: x.timestamp)
            training_data, _ = self._split_data(all_pairings)

            if len(training_data) < 30:
                raise ValueError(
                    f"Insufficient training data for {stock_id}: "
                    f"{len(training_data)} days. Need at least 30."
                )

            model_clone = copy.deepcopy(self.model)

            # Device override: explicit device request (e.g. --device cuda)
            # skips the force-CPU path so the model trains on the requested device.
            # Without override: force CPU for parallel workers — MPS can't be
            # pickled and serializes through a single command queue anyway.
            if hasattr(model_clone, "_device"):
                if self.device_override:
                    model_clone._device = torch.device(self.device_override)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    model_clone._device = torch.device("cpu")  # pyright: ignore[reportAttributeAccessIssue]

            cache_key: TrainingCacheKey | None = None
            if not self.no_cache and hasattr(model_clone, "restore_fit"):
                cache_key = TrainingCacheKey(
                    training_signature=model_clone.training_signature,
                    stock_id=stock_id,
                    seed=self.config.seed if is_neural else None,
                    rebalance_horizon_days=self.config.rebalance_horizon_days,
                    training_start=self.config.training_start.isoformat(),
                    training_end=self.config.training_end.isoformat(),
                    headline_only=self.config.headline_only,
                    semantic_model=self.config.semantic_model,
                    sentiment_ema_spans=tuple(self.config.sentiment_ema_spans),
                    retraining_cadence_days=self.config.retraining_cadence_days,
                    early_stopping_patience=self.config.early_stopping_patience,
                )
            work_items.append((stock_id, model_clone, training_data, cache_key))

        # Phase 2: fit in parallel or sequentially
        # Skip process parallelism when using GPU — CUDA contexts per
        # subprocess are wasteful for small models, and the GPU already
        # parallelizes at the kernel level.
        n_stocks = len(work_items)
        on_gpu = self.device_override in ("cuda", "mps")
        use_parallel = (
            self.model.parallelizable
            and n_stocks >= _MIN_STOCKS_FOR_PARALLEL
            and not on_gpu  # GPU runs sequentially per stock
        )

        # Derive per-stock seeds: base_seed + stock_index. Each stock gets a
        # unique seed so DataLoader shuffle patterns differ across stocks.
        if self.config.seed is None:
            raise ValueError("config.seed must be set for reproducible training")
        base_seed: int = self.config.seed
        stock_seeds = {
            stock_id: base_seed + i for i, (stock_id, _, _, _) in enumerate(work_items)
        }

        cache_hits: dict[str, bool] = {}
        if use_parallel:
            cores = os.cpu_count() or 1
            if self.max_inner_workers is not None:
                max_workers = min(self.max_inner_workers, n_stocks)
            else:
                max_workers = min(cores, n_stocks)
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _fit_stock, clone, data, stock_seeds[stock_id], cache_key
                    ): stock_id
                    for stock_id, clone, data, cache_key in work_items
                }
                for future in futures:
                    stock_id = futures[future]
                    self._models[stock_id], cache_hits[stock_id] = future.result()
        else:
            for stock_id, model_clone, training_data, cache_key in work_items:
                self._models[stock_id], cache_hits[stock_id] = _fit_stock(
                    model_clone, training_data, stock_seeds[stock_id], cache_key
                )

        # Hit/miss summary from actual restore_fit outcomes
        if not self.no_cache:
            keyed = [sid for sid, _, _, ck in work_items if ck is not None]
            if keyed:
                hits = sum(1 for sid in keyed if cache_hits[sid])
                print(
                    f"[cache] {hits}/{len(keyed)} hits — {self.model.model_type}",
                    flush=True,
                )

        self.training_duration_s = time.perf_counter() - t0

    def backtest(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """
        Walk-forward backtest with periodic rebalancing.

        Walks every trading day for daily snapshots.
        Predicts and rebalances every rebalance_horizon_days.
        """
        t0 = time.perf_counter()
        if not self._models:
            raise RuntimeError("No trained models. Call train() first.")

        # Build per-stock test data
        test_data_by_stock: dict[str, list[StockDataPairings]] = {}
        for stock_data in pipeline_input.stock_data:
            all_pairings = sorted(stock_data.data, key=lambda x: x.timestamp)
            _, test_data = self._split_data(all_pairings)
            test_data_by_stock[stock_data.stock_id] = test_data

        # Verify alignment: all stocks must have same test timeline
        stock_ids = list(test_data_by_stock.keys())
        reference_timeline = test_data_by_stock[stock_ids[0]]
        n_days = len(reference_timeline)
        for stock_id in stock_ids[1:]:
            other = test_data_by_stock[stock_id]
            assert len(other) == n_days, (
                f"Test data length mismatch: {stock_ids[0]} has {n_days} days, "
                f"{stock_id} has {len(other)} days"
            )

        prediction_records: list[PredictionRecord] = []
        trade_records: list[TradeRecord] = []
        snapshot_records: list[PortfolioSnapshot] = []

        portfolio = PortfolioState(cash=pipeline_input.initial_capital)
        horizon = self.config.rebalance_horizon_days

        # Resolve cadence: "auto" → horizon (refit every rebalance, matches HoltsLES default behaviour).
        cadence_raw = self.config.retraining_cadence_days
        cadence: int | None
        if cadence_raw == "auto":
            cadence = horizon
        else:
            cadence = cadence_raw

        # Reset volatility tracking for this backtest
        self._daily_returns = []
        self._rolling_vols = []

        # Track last refit step per stock for retraining_cadence_days check
        last_refit_day: dict[str, int] = {sid: 0 for sid in stock_ids}

        for i in range(n_days):
            timestamp = reference_timeline[i].timestamp

            open_prices: dict[str, float] = {}
            close_prices: dict[str, float] = {}
            for stock_id in stock_ids:
                open_prices[stock_id] = test_data_by_stock[stock_id][
                    i
                ].numeric_data.open
                close_prices[stock_id] = test_data_by_stock[stock_id][
                    i
                ].numeric_data.close

            # Track portfolio daily return for volatility-scaled conviction
            if snapshot_records:
                prev_value = snapshot_records[-1].portfolio_value
                curr_value = portfolio.total_value(close_prices)
                if prev_value > 0:
                    self._daily_returns.append(curr_value / prev_value - 1)

            # Rebalance day: predict all stocks, select top k, rebalance
            rebalance_threshold: float | None = None
            if i % horizon == 0:
                predicted_returns: dict[str, float] = {}

                for stock_id, model in self._models.items():
                    predicted_return = model.predict(horizon)
                    predicted_returns[stock_id] = predicted_return

                    # Record prediction only when actual return is verifiable
                    future_idx = i + horizon
                    if future_idx < n_days:
                        today_close = test_data_by_stock[stock_id][i].numeric_data.close
                        future_close = test_data_by_stock[stock_id][
                            future_idx
                        ].numeric_data.close
                        actual_return = (future_close - today_close) / today_close

                        prediction_records.append(
                            PredictionRecord(
                                timestamp=timestamp,
                                stock_id=stock_id,
                                predicted_return=predicted_return,
                                actual_return=actual_return,
                                predicted_direction=self._return_to_direction(
                                    predicted_return
                                ),
                                actual_direction=self._return_to_direction(
                                    actual_return
                                ),
                                horizon_days=horizon,
                            )
                        )

                rebalance_threshold = self._compute_threshold()
                selected = self.select_top_k_equal_holdings(
                    predicted_returns, rebalance_threshold
                )
                target_allocations = dict(selected)

                trades = portfolio.rebalance(
                    target_allocations=target_allocations,
                    prices=open_prices,
                    transaction_cost_pct=self.config.transaction_cost_pct,
                    timestamp=timestamp,
                )
                trade_records.extend(trades)

            # Daily snapshot (always — evaluation needs daily returns)
            snapshot = portfolio.snapshot(timestamp, close_prices)
            snapshot.threshold = rebalance_threshold
            snapshot_records.append(snapshot)

            # Update all models with today's observation; refit periodically if cadence is set.
            # Track last_refit_day per stock so cadence works regardless of horizon stepping
            # (e.g. cadence=252, horizon=5: i never lands on multiples of 252, modulo would miss).
            for stock_id, model in self._models.items():
                today_pairing = test_data_by_stock[stock_id][i]
                model.update(today_pairing)
                if cadence is not None and (i - last_refit_day[stock_id]) >= cadence:
                    model.refit()
                    last_refit_day[stock_id] = i

        # Final value at last day's close
        final_close_prices: dict[str, float] = {}
        for stock_id in stock_ids:
            final_close_prices[stock_id] = test_data_by_stock[stock_id][
                -1
            ].numeric_data.close
        final_value = portfolio.total_value(final_close_prices)

        self.backtest_duration_s = time.perf_counter() - t0

        return PipelineOutput(
            model_type=self.model.model_type,
            model_name=self.model.name,
            run_id=str(uuid.uuid4()),
            run_timestamp=datetime.now(),
            feature_flags={
                "semantic": self.model.uses_semantic,
                "retraining": self.config.retraining_cadence_days is not None,
            },
            stocks=[sd.stock_id for sd in pipeline_input.stock_data],
            config=self.config,
            initial_capital=pipeline_input.initial_capital,
            final_capital=final_value,
            training_duration_s=self.training_duration_s,
            backtest_duration_s=self.backtest_duration_s,
            trade_records=trade_records,
            prediction_records=prediction_records,
            snapshot_records=snapshot_records,
        )

    def _compute_threshold(self) -> float:
        """Compute the effective conviction threshold for stock selection."""
        base = self.config.min_predicted_return

        if self.config.conviction_mode == "fixed" or base == 0.0:
            return base

        # Volatility-scaled: threshold rises when vol is above historical mean
        if len(self._daily_returns) < self.vol_window:
            return base

        recent = self._daily_returns[-self.vol_window :]
        current_vol = _std(recent)
        self._rolling_vols.append(current_vol)

        mean_vol = sum(self._rolling_vols) / len(self._rolling_vols)

        if mean_vol < 1e-10:
            return base

        return base * (current_vol / mean_vol)

    def select_top_k_equal_holdings(
        self, predictions: dict[str, float], threshold: float = 0.0
    ) -> list[tuple[str, float]]:
        """
        Select top k stocks by predicted return with equal weighting.

        Only stocks above the conviction threshold are selected.
        Returns list of (stock_id, weight) sorted by predicted_return descending.
        If none above threshold, returns empty list (portfolio goes to cash).
        """
        above_threshold = [
            (stock_id, returns)
            for stock_id, returns in predictions.items()
            if returns > threshold
        ]
        above_threshold.sort(key=lambda x: x[1], reverse=True)
        selected = above_threshold[: self.config.top_k]

        if not selected:
            return []

        weight = 1.0 / len(selected)
        return [(stock_id, weight) for stock_id, _ in selected]

    def _split_data(
        self, data: list[StockDataPairings]
    ) -> tuple[list[StockDataPairings], list[StockDataPairings]]:
        """Split data into training and test sets based on config dates."""
        training = [
            p
            for p in data
            if self.config.training_start <= p.timestamp <= self.config.training_end
        ]
        test = [
            p
            for p in data
            if self.config.test_start <= p.timestamp <= self.config.test_end
        ]
        return training, test

    def _return_to_direction(self, ret: float) -> Direction:
        """Convert return to direction enum."""
        if ret > 0:
            return Direction.UP
        elif ret < 0:
            return Direction.DOWN
        return Direction.NEUTRAL


def _std(values: list[float]) -> float:
    """Population standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)
