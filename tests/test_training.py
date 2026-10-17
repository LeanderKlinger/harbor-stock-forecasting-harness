"""Smoke tests for training module."""

import warnings

from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.output import PipelineOutput
from src.training.strategies.arima import ARIMAStrategy
from src.training.training_module import TrainingModule


# Suppress statsmodels convergence warnings in tests
warnings.filterwarnings("ignore")


class TestTrainingSmoke:
    """Basic smoke tests - does it run without crashing?"""

    def test_arima_strategy_runs(self, fast_config: Config) -> None:
        """ARIMAStrategy should fit and predict without error."""
        module = ExtractionModule(fast_config)
        pipeline_input = module.build_pipeline_input()

        # Get some training data (now using pairings)
        all_pairings = pipeline_input.stock_data[0].data
        training_data = [
            p for p in all_pairings if p.timestamp <= fast_config.training_end
        ]

        strategy = ARIMAStrategy()
        strategy.fit(training_data)
        prediction = strategy.predict(horizon_days=1)

        assert isinstance(prediction, float)

    def test_training_module_runs(self, fast_config: Config) -> None:
        """TrainingModule should produce PipelineOutput."""
        module = ExtractionModule(fast_config)
        pipeline_input = module.build_pipeline_input()

        model = ARIMAStrategy()
        training = TrainingModule(model, fast_config)
        training.train(pipeline_input)
        result = training.backtest(pipeline_input)

        assert isinstance(result, PipelineOutput)
        assert result.final_capital > 0

    def test_training_with_retraining_cadence_runs(self, fast_config: Config) -> None:
        """TrainingModule with retraining_cadence_days set should complete and flag retraining."""
        cadence_config = fast_config.model_copy(update={"retraining_cadence_days": 5})
        module = ExtractionModule(cadence_config)
        pipeline_input = module.build_pipeline_input()

        model = ARIMAStrategy()
        training = TrainingModule(model, cadence_config)
        training.train(pipeline_input)
        result = training.backtest(pipeline_input)

        assert isinstance(result, PipelineOutput)
        assert result.feature_flags.get("retraining") is True
