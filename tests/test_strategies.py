"""Smoke tests for all model zoo strategies."""

import warnings
from datetime import datetime

import pytest
import torch

from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.input import StockDataPairings
from src.training.strategies.arima import ARIMAStrategy
from src.training.strategies.gru import GRUStrategy
from src.training.strategies.holts import HoltsStrategy
from src.training.strategies.jamba_strategy import JambaStrategy
from src.training.strategies.mamba_strategy import MambaStrategy
from src.training.strategies.mlp import MLPStrategy
from src.training.strategies.xlstm import XLSTMStrategy

warnings.filterwarnings("ignore")


@pytest.fixture
def model_config() -> Config:
    """Config for model smoke tests — small sequence, 5-day horizon."""
    return Config(
        training_start=datetime(2020, 1, 1),
        training_end=datetime(2022, 12, 31),
        test_start=datetime(2023, 1, 1),
        test_end=datetime(2023, 3, 31),
        rebalance_horizon_days=5,
        sequence_length=30,
        stock_ids=["AMD"],
    )


@pytest.fixture
def training_data(model_config: Config) -> list[StockDataPairings]:
    """Real AMD training data via extraction module."""
    extraction = ExtractionModule(model_config)
    pipeline_input = extraction.build_pipeline_input()
    return [
        p
        for p in pipeline_input.stock_data[0].data
        if p.timestamp <= model_config.training_end
    ]


class TestProtocolCompliance:
    """All strategies must expose model_type, name, and uses_semantic."""

    def test_arima(self) -> None:
        m = ARIMAStrategy()
        assert m.model_type == "ARIMA"
        assert "ARIMA" in m.name
        assert m.uses_semantic is False

    def test_holts(self) -> None:
        m = HoltsStrategy()
        assert m.model_type == "HoltsLES"
        assert "HoltsLES" in m.name
        assert m.uses_semantic is False

    def test_xlstm(self, model_config: Config) -> None:
        m = XLSTMStrategy(model_config, hidden_size=32, num_layers=1, epochs=1)
        assert m.model_type == "xLSTM"
        assert m.uses_semantic is True

    def test_gru(self, model_config: Config) -> None:
        m = GRUStrategy(model_config, hidden_size=32, num_layers=1, epochs=1)
        assert m.model_type == "GRU"
        assert m.uses_semantic is True

    def test_mlp(self, model_config: Config) -> None:
        m = MLPStrategy(model_config, hidden_size=64, epochs=1)
        assert m.model_type == "MLP"
        assert m.uses_semantic is True

    def test_mamba(self, model_config: Config) -> None:
        m = MambaStrategy(model_config, hidden_size=32, num_layers=1, epochs=1)
        assert m.model_type == "Mamba"
        assert m.uses_semantic is True

    def test_jamba(self, model_config: Config) -> None:
        m = JambaStrategy(
            model_config, hidden_size=32, num_layers=2, num_attention_heads=4, epochs=1
        )
        assert m.model_type == "Jamba"
        assert m.uses_semantic is True


class TestStatisticalModels:
    """Smoke: fit, update, predict for numeric-only models."""

    def test_arima_fit_update_predict(
        self, training_data: list[StockDataPairings]
    ) -> None:
        model = ARIMAStrategy()
        model.fit(training_data[:500])
        for p in training_data[500:510]:
            model.update(p)
        assert isinstance(model.predict(horizon_days=5), float)

    def test_holts_fit_update_predict(
        self, training_data: list[StockDataPairings]
    ) -> None:
        model = HoltsStrategy()
        model.fit(training_data[:500])
        for p in training_data[500:510]:
            model.update(p)
        assert isinstance(model.predict(horizon_days=5), float)


class TestNeuralModels:
    """Smoke: fit and predict for all TorchStrategy subclasses."""

    def test_xlstm(
        self, model_config: Config, training_data: list[StockDataPairings]
    ) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        model = XLSTMStrategy(model_config, hidden_size=32, num_layers=1, epochs=2)
        model.fit(training_data)
        assert isinstance(model.predict(horizon_days=5), float)

    def test_gru(
        self, model_config: Config, training_data: list[StockDataPairings]
    ) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        model = GRUStrategy(model_config, hidden_size=32, num_layers=1, epochs=2)
        model.fit(training_data)
        assert isinstance(model.predict(horizon_days=5), float)

    def test_mlp(
        self, model_config: Config, training_data: list[StockDataPairings]
    ) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        model = MLPStrategy(model_config, hidden_size=64, epochs=2)
        model.fit(training_data)
        assert isinstance(model.predict(horizon_days=5), float)

    def test_mamba(
        self, model_config: Config, training_data: list[StockDataPairings]
    ) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        model = MambaStrategy(model_config, hidden_size=32, num_layers=1, epochs=2)
        model.fit(training_data)
        assert isinstance(model.predict(horizon_days=5), float)

    def test_jamba(
        self, model_config: Config, training_data: list[StockDataPairings]
    ) -> None:
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        model = JambaStrategy(
            model_config, hidden_size=32, num_layers=2, num_attention_heads=4, epochs=2
        )
        model.fit(training_data)
        assert isinstance(model.predict(horizon_days=5), float)

    def test_horizon_mismatch_raises(
        self, model_config: Config, training_data: list[StockDataPairings]
    ) -> None:
        """Predicting with wrong horizon should fail — config has horizon=5."""
        torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
        model = GRUStrategy(model_config, hidden_size=32, num_layers=1, epochs=1)
        model.fit(training_data)
        with pytest.raises(AssertionError, match="horizon"):
            model.predict(horizon_days=1)
