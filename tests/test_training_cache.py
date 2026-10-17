"""Tests for the training cache: cache_fit/restore_fit round-trip + retraining cadence."""

import warnings
from datetime import datetime
from pathlib import Path

import pytest
import torch

from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.input import StockDataPairings
from src.training.strategies.arima import ARIMAStrategy
from src.training.strategies.gru import GRUStrategy
from src.training.strategies.holts import HoltsStrategy
from src.training.training_cache import TrainingCacheKey, cache_path_for
from src.training.training_module import TrainingModule

warnings.filterwarnings("ignore")


# Long training window — ARIMA/Holts need plenty of data to converge.
TRAINING_END = datetime(2019, 12, 31)


@pytest.fixture(scope="module")
def cache_config() -> Config:
    return Config(
        training_start=datetime(2011, 1, 1),
        training_end=TRAINING_END,
        test_start=datetime(2020, 1, 1),
        test_end=datetime(2020, 3, 31),  # ~62 trading days for the cadence tests
        stock_ids=["NVDA"],
        headline_only=True,
        sequence_length=60,
        rebalance_horizon_days=5,
        sentiment_ema_spans=[5, 10, 20],
        seed=42,
        description="cache test",
    )


@pytest.fixture(scope="module")
def training_pairings(cache_config: Config) -> list[StockDataPairings]:
    pipeline_input = ExtractionModule(cache_config).build_pipeline_input()
    return [p for p in pipeline_input.stock_data[0].data if p.timestamp <= TRAINING_END]


class TestCacheRoundTrip:
    """fit → cache_fit → restore_fit on a fresh instance → predict matches exactly."""

    def test_arima(
        self, training_pairings: list[StockDataPairings], tmp_path: Path
    ) -> None:
        s1 = ARIMAStrategy()
        s1.fit(training_pairings)
        pred1 = s1.predict(horizon_days=5)

        cache_path = tmp_path / "arima.bin"
        s1.cache_fit(cache_path)
        assert cache_path.exists()

        s2 = ARIMAStrategy()
        assert s2.restore_fit(cache_path)
        assert s2.predict(horizon_days=5) == pytest.approx(pred1, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]

    def test_holts(
        self, training_pairings: list[StockDataPairings], tmp_path: Path
    ) -> None:
        s1 = HoltsStrategy()
        s1.fit(training_pairings)
        pred1 = s1.predict(horizon_days=5)

        cache_path = tmp_path / "holts.bin"
        s1.cache_fit(cache_path)

        s2 = HoltsStrategy()
        assert s2.restore_fit(cache_path)
        assert s2.predict(horizon_days=5) == pytest.approx(pred1, abs=1e-9)  # pyright: ignore[reportUnknownMemberType]

    def test_gru(
        self,
        cache_config: Config,
        training_pairings: list[StockDataPairings],
        tmp_path: Path,
    ) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        s1 = GRUStrategy(cache_config, hidden_size=32, num_layers=1, epochs=3)
        s1.fit(training_pairings)
        pred1 = s1.predict(horizon_days=5)

        cache_path = tmp_path / "gru.bin"
        s1.cache_fit(cache_path)

        s2 = GRUStrategy(cache_config, hidden_size=32, num_layers=1, epochs=3)
        # _history is needed for predict() — restore_fit re-populates it.
        assert s2.restore_fit(cache_path)
        # Float32 weights, identical input → match to high precision regardless of device.
        assert s2.predict(horizon_days=5) == pytest.approx(pred1, abs=1e-5)  # pyright: ignore[reportUnknownMemberType]

    def test_restore_missing_returns_false(
        self, cache_config: Config, tmp_path: Path
    ) -> None:
        """A non-existent cache path returns False rather than raising."""
        s = GRUStrategy(cache_config, hidden_size=32, num_layers=1, epochs=1)
        assert not s.restore_fit(tmp_path / "does_not_exist.bin")

    def test_restore_corrupt_returns_false(
        self, cache_config: Config, tmp_path: Path
    ) -> None:
        """A corrupt cache file returns False — corrupted slot self-heals on next miss."""
        bad = tmp_path / "corrupt.bin"
        bad.write_bytes(b"not a torch payload")
        s = GRUStrategy(cache_config, hidden_size=32, num_layers=1, epochs=1)
        assert not s.restore_fit(bad)


class TestRetrainingCadence:
    """retraining_cadence_days fires at days-since-last-refit boundaries, regardless of horizon."""

    def _count_refits(self, cache_config: Config, cadence: int) -> int:
        """Run a backtest with cadence set, count ARIMA.refit() calls."""
        config = cache_config.model_copy(update={"retraining_cadence_days": cadence})
        pipeline_input = ExtractionModule(config).build_pipeline_input()

        refit_calls: list[int] = []
        original = ARIMAStrategy.refit

        def counting_refit(self: ARIMAStrategy) -> None:
            refit_calls.append(len(refit_calls))
            original(self)

        ARIMAStrategy.refit = counting_refit  # type: ignore[method-assign]
        try:
            module = TrainingModule(ARIMAStrategy(), config)
            module.train(pipeline_input)
            module.backtest(pipeline_input)
        finally:
            ARIMAStrategy.refit = original  # type: ignore[method-assign]
        return len(refit_calls)

    def test_cadence_20_fires_three_times_in_3mo_window(
        self, cache_config: Config
    ) -> None:
        """~62 trading days, cadence=20 → fires at i=20, 40, 60 → 3 refits."""
        assert self._count_refits(cache_config, cadence=20) == 3

    def test_cadence_none_never_fires(self, cache_config: Config) -> None:
        """retraining_cadence_days=None → no refits during backtest."""
        # cache_config already has retraining_cadence_days=None by default
        pipeline_input = ExtractionModule(cache_config).build_pipeline_input()

        refit_calls: list[int] = []
        original = ARIMAStrategy.refit

        def counting_refit(self: ARIMAStrategy) -> None:
            refit_calls.append(len(refit_calls))
            original(self)

        ARIMAStrategy.refit = counting_refit  # type: ignore[method-assign]
        try:
            module = TrainingModule(ARIMAStrategy(), cache_config)
            module.train(pipeline_input)
            module.backtest(pipeline_input)
        finally:
            ARIMAStrategy.refit = original  # type: ignore[method-assign]
        assert refit_calls == []


class TestCacheKeyHash:
    """Stable hashes for TrainingCacheKey + cache_path_for invariants."""

    def _key(self, **overrides: object) -> TrainingCacheKey:
        defaults: dict[str, object] = {
            "training_signature": "ARIMA|order=(5, 1, 2)",
            "stock_id": "NVDA",
            "seed": None,
            "rebalance_horizon_days": 5,
            "training_start": "2011-01-01T00:00:00",
            "training_end": "2019-12-31T00:00:00",
            "headline_only": True,
            "semantic_model": "claude-haiku-4-5",
            "sentiment_ema_spans": (5, 10, 20),
            "retraining_cadence_days": None,
            "early_stopping_patience": None,
        }
        defaults.update(overrides)
        return TrainingCacheKey(**defaults)  # type: ignore[arg-type]

    def test_same_fields_same_hash(self) -> None:
        assert self._key().to_hash() == self._key().to_hash()

    def test_different_horizon_different_hash(self) -> None:
        assert self._key().to_hash() != self._key(rebalance_horizon_days=20).to_hash()

    def test_different_signature_different_hash(self) -> None:
        a = self._key().to_hash()
        b = self._key(training_signature="ARIMA|order=(2, 1, 2)").to_hash()
        assert a != b

    def test_path_uses_hash(self) -> None:
        key = self._key()
        path = cache_path_for(key)
        assert path.name == f"{key.to_hash()}.bin"
