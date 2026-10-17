"""Smoke tests for the feature_set input-width ablation knob."""

import warnings
from datetime import datetime
from typing import Literal

import pytest

from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.input import StockDataPairings
from src.training.strategies.gru import GRUStrategy
from src.training.training_cache import TrainingCacheKey

warnings.filterwarnings("ignore")

FULL_WIDTH = 16
NUMERIC_WIDTH = 7
CLOSE_ONLY_WIDTH = 1


def _config(feature_set: Literal["full", "numeric", "close_only"]) -> Config:
    return Config(
        training_start=datetime(2020, 1, 1),
        training_end=datetime(2022, 12, 31),
        test_start=datetime(2023, 1, 1),
        test_end=datetime(2023, 3, 31),
        rebalance_horizon_days=5,
        sequence_length=30,
        stock_ids=["AMD"],
        feature_set=feature_set,
    )


@pytest.fixture(scope="module")
def pairing() -> StockDataPairings:
    """One real enriched AMD pairing via the extraction module."""
    extraction = ExtractionModule(_config("full"))
    pipeline_input = extraction.build_pipeline_input()
    return pipeline_input.stock_data[0].data[-1]


class TestFeatureWidths:
    def test_full_is_sixteen(self, pairing: StockDataPairings) -> None:
        m = GRUStrategy(_config("full"), hidden_size=32, num_layers=1, epochs=1)
        assert len(m._extract_features(pairing)) == FULL_WIDTH  # pyright: ignore[reportPrivateUsage]

    def test_numeric_is_seven(self, pairing: StockDataPairings) -> None:
        m = GRUStrategy(_config("numeric"), hidden_size=32, num_layers=1, epochs=1)
        assert len(m._extract_features(pairing)) == NUMERIC_WIDTH  # pyright: ignore[reportPrivateUsage]

    def test_close_only_is_one(self, pairing: StockDataPairings) -> None:
        m = GRUStrategy(_config("close_only"), hidden_size=32, num_layers=1, epochs=1)
        features = m._extract_features(pairing)  # pyright: ignore[reportPrivateUsage]
        assert len(features) == CLOSE_ONLY_WIDTH
        assert features[0] == pairing.numeric_data.close


class TestTrainingSignature:
    def test_marker_emitted_for_every_mode(self) -> None:
        # Uniform convention: every knob always in the signature, including the
        # default ("full") — the pre-flag cache was greenfielded deliberately.
        for feature_set in ("full", "numeric", "close_only"):
            m = GRUStrategy(
                _config(feature_set), hidden_size=32, num_layers=1, epochs=1
            )
            assert f"features={feature_set}" in m.training_signature

    def test_signatures_distinct_across_modes(self) -> None:
        signatures: set[str] = set()
        for feature_set in ("full", "numeric", "close_only"):
            m = GRUStrategy(
                _config(feature_set), hidden_size=32, num_layers=1, epochs=1
            )
            signatures.add(m.training_signature)
        assert len(signatures) == 3

    def test_cache_keys_distinct_across_modes(self) -> None:
        hashes: set[str] = set()
        for feature_set in ("full", "numeric", "close_only"):
            m = GRUStrategy(
                _config(feature_set), hidden_size=32, num_layers=1, epochs=1
            )
            key = TrainingCacheKey(
                training_signature=m.training_signature,
                stock_id="AMD",
                seed=42,
                rebalance_horizon_days=5,
                training_start="2020-01-01",
                training_end="2022-12-31",
                headline_only=True,
                semantic_model="claude-haiku-4-5",
                sentiment_ema_spans=(5, 10, 20),
                retraining_cadence_days=None,
                early_stopping_patience=None,
            )
            hashes.add(key.to_hash())
        assert len(hashes) == 3


class TestUsesSemantic:
    def test_full_uses_semantic(self) -> None:
        m = GRUStrategy(_config("full"), hidden_size=32, num_layers=1, epochs=1)
        assert m.uses_semantic is True

    def test_numeric_does_not(self) -> None:
        m = GRUStrategy(_config("numeric"), hidden_size=32, num_layers=1, epochs=1)
        assert m.uses_semantic is False

    def test_close_only_does_not(self) -> None:
        m = GRUStrategy(_config("close_only"), hidden_size=32, num_layers=1, epochs=1)
        assert m.uses_semantic is False
