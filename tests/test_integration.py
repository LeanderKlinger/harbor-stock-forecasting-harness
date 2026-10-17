"""Integration tests - full pipeline end-to-end."""

import warnings
from pathlib import Path

from src.evaluation.evaluation_module import EvaluationModule
from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.evaluation import PipelineEvaluation
from src.models.output import PipelineOutput
from src.training.strategies.arima import ARIMAStrategy
from src.training.training_module import TrainingModule


warnings.filterwarnings("ignore")


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_extraction_to_evaluation(self, fast_config: Config) -> None:
        """Full pipeline: extraction → training → evaluation."""
        # 1. Extraction
        extraction = ExtractionModule(fast_config)
        pipeline_input = extraction.build_pipeline_input()

        # 2. Training
        model = ARIMAStrategy()
        training = TrainingModule(model, fast_config)
        training.train(pipeline_input)
        pipeline_output = training.backtest(pipeline_input)

        # 3. Evaluation
        evaluation = EvaluationModule().evaluate(pipeline_output)

        # Sanity checks - not testing specific values, just that it's reasonable
        assert isinstance(pipeline_output, PipelineOutput)
        assert isinstance(evaluation, PipelineEvaluation)

        # Output should have records
        assert len(pipeline_output.prediction_records) > 0
        assert len(pipeline_output.snapshot_records) > 0

        # Evaluation metrics should be numbers (not NaN)
        assert evaluation.rmse == evaluation.rmse  # NaN != NaN
        assert evaluation.total_return == evaluation.total_return
        assert evaluation.sharpe_ratio == evaluation.sharpe_ratio

    def test_pipeline_with_transaction_costs(self, fast_config: Config) -> None:
        """Pipeline should work with transaction costs enabled."""
        # Modify config to add fees
        config_with_fees = Config(
            training_start=fast_config.training_start,
            training_end=fast_config.training_end,
            test_start=fast_config.test_start,
            test_end=fast_config.test_end,
            rebalance_horizon_days=fast_config.rebalance_horizon_days,
            stock_ids=fast_config.stock_ids,
            transaction_cost_pct=0.001,  # 0.1%
            seed=fast_config.seed,
        )

        extraction = ExtractionModule(config_with_fees)
        pipeline_input = extraction.build_pipeline_input()

        model = ARIMAStrategy()
        training = TrainingModule(model, config_with_fees)
        training.train(pipeline_input)
        pipeline_output = training.backtest(pipeline_input)

        evaluation = EvaluationModule().evaluate(pipeline_output)

        # Transaction costs should be tracked
        assert evaluation.total_transaction_cost >= 0


class TestPersistence:
    """Smoke test: persist_run creates expected output artifacts."""

    def test_persist_creates_all_artifacts(
        self, fast_config: Config, tmp_path: Path
    ) -> None:
        """persist_run should produce pipeline_output.json, pipeline_input.json, run_detail.json, and metrics.md."""
        extraction = ExtractionModule(fast_config)
        pipeline_input = extraction.build_pipeline_input()

        model = ARIMAStrategy()
        training = TrainingModule(model, fast_config)
        training.train(pipeline_input)
        output = training.backtest(pipeline_input)
        eval_module = EvaluationModule()
        evaluation = eval_module.evaluate(output)

        import src.evaluation.persistence as persistence_mod

        original_root = persistence_mod.EXECUTION_OUTPUT_ROOT
        persistence_mod.EXECUTION_OUTPUT_ROOT = tmp_path

        try:
            run_dir = eval_module.persist(pipeline_input, output, evaluation)

            assert run_dir.exists()
            assert (run_dir / "pipeline_output.json").exists()
            assert (run_dir / "pipeline_input.json").exists()
            assert (run_dir / "metrics.md").exists()
            assert (run_dir / "run_detail.json").exists()

            metrics_text = (run_dir / "metrics.md").read_text()
            assert "MAPE" in metrics_text
            assert "execution_name" not in run_dir.name or "default" in run_dir.name
        finally:
            persistence_mod.EXECUTION_OUTPUT_ROOT = original_root
