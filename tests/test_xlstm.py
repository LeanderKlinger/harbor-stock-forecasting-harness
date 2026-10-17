"""Smoke tests for xLSTM strategy with real data extraction."""

import warnings
from datetime import datetime

import pytest

from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.output import PipelineOutput
from src.paths import ARTICLES_DIR
from src.training.strategies.xlstm import XLSTMStrategy
from src.training.training_module import TrainingModule

warnings.filterwarnings("ignore")


@pytest.fixture
def xlstm_config() -> Config:
    """Config with enough data for xLSTM sequence length."""
    return Config(
        training_start=datetime(2020, 1, 1),
        training_end=datetime(2022, 12, 31),
        test_start=datetime(2023, 1, 1),
        test_end=datetime(2023, 3, 31),
        rebalance_horizon_days=1,
        sequence_length=30,
        stock_ids=["AMD"],
        seed=42,
    )


class TestXLSTMSmoke:
    """Basic smoke tests for xLSTM - does it run without crashing?"""

    def test_xlstm_strategy_runs(self, xlstm_config: Config) -> None:
        """XLSTMStrategy should fit and predict without error."""
        extraction = ExtractionModule(xlstm_config)
        pipeline_input = extraction.build_pipeline_input()

        data = pipeline_input.stock_data[0].data

        model = XLSTMStrategy(xlstm_config, hidden_size=32, num_layers=1, epochs=2)
        model.fit(data)
        prediction = model.predict(horizon_days=1)

        assert isinstance(prediction, float)

    def test_xlstm_training_module_runs(self, xlstm_config: Config) -> None:
        """TrainingModule with xLSTM should produce PipelineOutput."""
        extraction = ExtractionModule(xlstm_config)
        pipeline_input = extraction.build_pipeline_input()

        model = XLSTMStrategy(xlstm_config, hidden_size=32, num_layers=1, epochs=2)
        training = TrainingModule(model, xlstm_config)
        training.train(pipeline_input)
        result = training.backtest(pipeline_input)

        assert isinstance(result, PipelineOutput)
        assert result.final_capital > 0
        assert "xLSTM" in result.model_name

    def test_xlstm_with_semantic(self, xlstm_config: Config) -> None:
        """xLSTM should work with semantic features when available."""
        articles_path = ARTICLES_DIR / "AMD.csv"
        if not articles_path.exists():
            pytest.skip("AMD articles not available")

        semantic_config = Config(
            training_start=xlstm_config.training_start,
            training_end=xlstm_config.training_end,
            test_start=xlstm_config.test_start,
            test_end=xlstm_config.test_end,
            rebalance_horizon_days=xlstm_config.rebalance_horizon_days,
            sequence_length=xlstm_config.sequence_length,
            semantic_model="claude-haiku-4-5",
            stock_ids=["AMD"],
            seed=xlstm_config.seed,
        )

        extraction = ExtractionModule(semantic_config)
        pipeline_input = extraction.build_pipeline_input()

        has_semantic = any(
            p.semantic_data is not None for p in pipeline_input.stock_data[0].data
        )

        if not has_semantic:
            pytest.skip("No semantic data in checkpoint")

        model = XLSTMStrategy(semantic_config, hidden_size=32, num_layers=1, epochs=2)
        training = TrainingModule(model, semantic_config)
        training.train(pipeline_input)
        result = training.backtest(pipeline_input)

        assert result.final_capital > 0
