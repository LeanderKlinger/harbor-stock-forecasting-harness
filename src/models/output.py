from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict
from src.models.config import Config


class Direction(Enum):
    UP = 1
    DOWN = -1
    NEUTRAL = 0


class Action(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


# RMSE, acc
class PredictionRecord(BaseModel):
    timestamp: datetime
    stock_id: str

    # RMSE
    predicted_return: float  # in % for horizon
    actual_return: float  # in % for horizon

    # Directional accuracy
    predicted_direction: Direction  # again for horizon
    actual_direction: Direction  # again for horizon

    horizon_days: int = Field(default=1, ge=1)

    # Optional
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


# Transaction analysis
class TradeRecord(BaseModel):
    timestamp: datetime
    stock_id: str
    action: Action
    quantity: float
    execution_price: float  # for idyllic == open
    transaction_cost: float = Field(default=0.0, ge=0)


# Sharpe, Sortino, MDD, CVar, Total Return
class PortfolioSnapshot(BaseModel):
    timestamp: datetime

    # return-based
    portfolio_value: float = Field(gt=0)
    daily_return: float

    # position-tracking
    positions: Dict[str, float]
    free_cash: float = Field(ge=0)

    # running metrics
    cumulative_return: float
    peak_value: float

    # conviction threshold (only set on rebalance days when vol-scaled)
    threshold: Optional[float] = None


class PipelineOutput(BaseModel):
    # Descriptive
    model_type: str  # Clean name for grouping (e.g. "xLSTM", "ARIMA")
    model_name: str  # Full parametric name (e.g. "xLSTM(seq=60, h=64)")
    run_id: str
    run_timestamp: datetime
    feature_flags: Dict[str, bool] = Field(default_factory=dict)

    # Config
    stocks: List[str]
    config: Config

    # Financial shorthands
    initial_capital: float
    final_capital: float

    # Timing (seconds)
    training_duration_s: float = 0.0
    backtest_duration_s: float = 0.0

    # Records
    trade_records: List[TradeRecord]
    prediction_records: List[PredictionRecord]
    snapshot_records: List[PortfolioSnapshot]
