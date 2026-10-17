from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Config(BaseModel):
    # Time context
    training_start: datetime
    training_end: datetime
    test_start: datetime
    test_end: datetime
    trading_days: int = Field(default=252)

    # Market environment
    market_condition: str = "idyllic"  # Informational — describes experiment setup
    transaction_cost_pct: float = Field(default=0.0, ge=0)  # e.g., 0.001 = 0.1%

    # Risk parameters
    risk_free_rate: float = Field(default=0.02)  # annual, for Sharpe/Sortino
    var_confidence: float = Field(default=0.95)  # for VaR/CVaR

    # Model parameters (shared across all neural models in a run)
    sequence_length: int = Field(default=60, ge=1)  # Lookback window for neural models
    rebalance_horizon_days: int = Field(
        default=1, ge=1
    )  # Prediction horizon == rebalance cadence
    # Input width for neural models: full = 16 (7 numeric + 9 semantic),
    # numeric = 7, close_only = 1 (input parity with HoltsLES/ARIMA).
    # Training knob, not extraction — semantic extraction runs regardless.
    feature_set: Literal["full", "numeric", "close_only"] = "full"

    # Portfolio selection
    top_k: int = Field(default=4, ge=1)  # Number of stocks to hold
    min_predicted_return: float = Field(default=0.0, ge=0)  # Conviction threshold
    conviction_mode: str = "fixed"  # "fixed" or "volatility_scaled"

    # Training
    early_stopping_patience: int | None = None  # None = no early stopping
    # None = frozen weights; int = refit every N test days; "auto" = refit every rebalance (= rebalance_horizon_days).
    retraining_cadence_days: int | Literal["auto"] | None = None

    # Execution
    execution_name: str = "default"
    description: str = ""  # Human-readable note for what this run tests, must be set
    experiment_name: str | None = None  # Set by experiment runner, None for manual runs
    seed: int | None = None  # Set by experiment runner, None for manual runs

    # Semantic extraction
    semantic_model: str = "claude-haiku-4-5"  # Model for sentiment scoring
    headline_only: bool = False  # True = headline-only sentiment, False = full article
    sentiment_ema_spans: list[int] = [5, 10, 20]  # EMA spans in trading days

    # Stock universe
    stock_ids: list[str]
