import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from src.extraction.extraction_module import ExtractionModule
from src.models.config import Config

# Configure logging before other imports
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Full date range
TRAINING_START = datetime(2019, 1, 1)
TRAINING_END = datetime(2022, 12, 31)
TEST_START = datetime(2023, 1, 1)
TEST_END = datetime(2023, 12, 28)

# Fast tests: just January 2023
FAST_TEST_END = datetime(2023, 1, 31)

DEFAULT_INITIAL_CAPITAL = 100_000.0

config = Config(
    training_start=TRAINING_START,
    training_end=TRAINING_END,
    test_start=TEST_START,
    test_end=TEST_END,
    rebalance_horizon_days=1,
    headline_only=True,
    semantic_model="claude-haiku-4-5",
    stock_ids=["COP", "COST", "MRK", "MS", "NVDA"],
)

extraction_module = ExtractionModule(config)

# Just run extraction for the scores
result = extraction_module.build_pipeline_input(100_000)
