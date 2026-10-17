"""Shared fixtures and constants for tests."""

from datetime import datetime

import pytest

from src.models.config import Config


# =============================================================================
# Test Constants
# =============================================================================

TEST_STOCK_ID = "AMD"

# Full date range
TRAINING_START = datetime(2019, 1, 1)
TRAINING_END = datetime(2022, 12, 31)
TEST_START = datetime(2023, 1, 1)
TEST_END = datetime(2023, 12, 28)

# Fast tests: just January 2023
FAST_TEST_END = datetime(2023, 1, 31)

DEFAULT_INITIAL_CAPITAL = 100_000.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> Config:
    """Standard test configuration."""
    return Config(
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        test_start=TEST_START,
        test_end=TEST_END,
        rebalance_horizon_days=1,
        stock_ids=[TEST_STOCK_ID],
        seed=42,
    )


@pytest.fixture
def fast_config() -> Config:
    """Fast test configuration (shorter test period)."""
    return Config(
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        test_start=TEST_START,
        test_end=FAST_TEST_END,
        rebalance_horizon_days=1,
        stock_ids=[TEST_STOCK_ID],
        seed=42,
    )
