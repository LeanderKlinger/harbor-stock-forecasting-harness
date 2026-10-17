"""Smoke tests for extraction module."""

from datetime import datetime

import pytest

from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config
from src.models.input import PipelineInput

from tests.conftest import TEST_STOCK_ID


class TestExtractionSmoke:
    """Basic smoke tests - does it run without crashing?"""

    def test_numeric_extraction_runs(self, config: Config) -> None:
        """NumericExtraction should complete without error."""
        module = ExtractionModule(config)
        result = module.numeric_extractor.extract(TEST_STOCK_ID)

        assert len(result) > 0

    def test_build_pipeline_input_runs(self, config: Config) -> None:
        """build_pipeline_input should produce valid PipelineInput."""
        module = ExtractionModule(config)
        result = module.build_pipeline_input()

        assert isinstance(result, PipelineInput)
        assert len(result.stock_data) == 1
        assert len(result.stock_data[0].data) > 0

    def test_nonexistent_stock_handled(self) -> None:
        """Nonexistent stock should raise, not crash mysteriously."""
        bad_config = Config(
            training_start=datetime(2019, 1, 1),
            training_end=datetime(2022, 12, 31),
            test_start=datetime(2023, 1, 1),
            test_end=datetime(2023, 12, 28),
            rebalance_horizon_days=1,
            stock_ids=["THIS_STOCK_DOES_NOT_EXIST_XYZ"],
        )
        module = ExtractionModule(bad_config)

        with pytest.raises(ValueError):
            module.build_pipeline_input()
