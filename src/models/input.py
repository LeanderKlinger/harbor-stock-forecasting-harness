from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
from src.models.config import Config


class Sectors(Enum):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    FINANCIALS = "financials"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    MATERIALS = "materials"


STOCK_SECTOR_MAP: dict[str, Sectors] = {
    "AMD": Sectors.TECHNOLOGY,
    "NVDA": Sectors.TECHNOLOGY,
    "AAPL": Sectors.TECHNOLOGY,
    "MSFT": Sectors.TECHNOLOGY,
    "TMO": Sectors.HEALTHCARE,
    "MRK": Sectors.HEALTHCARE,
    "JNJ": Sectors.HEALTHCARE,
    "TJX": Sectors.CONSUMER_DISCRETIONARY,
    "NKE": Sectors.CONSUMER_DISCRETIONARY,
    "DE": Sectors.INDUSTRIALS,
    "COST": Sectors.CONSUMER_STAPLES,
    "WMT": Sectors.CONSUMER_STAPLES,
    "COP": Sectors.ENERGY,
    "VLO": Sectors.ENERGY,
    "MS": Sectors.FINANCIALS,
    "JPM": Sectors.FINANCIALS,
    "T": Sectors.COMMUNICATION_SERVICES,
    "GOOG": Sectors.COMMUNICATION_SERVICES,
    "NEE": Sectors.UTILITIES,
    "AMT": Sectors.REAL_ESTATE,
    "LIN": Sectors.MATERIALS,
}


class Category(Enum):
    COMPANY_SPECIFIC = 1
    SECTORAL = 2
    GENERAL = 3


class NumericData(BaseModel):
    timestamp: datetime

    # OHLCV
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)

    # Returns
    log_return: float  # log(close_t / close_{t-1})
    annualized_return_1w: float  # 5-day return × (252/5)
    annualized_return_2w: float  # 10-day return × (252/10)
    annualized_return_1m: float  # 21-day return × (252/21)

    # Volatility
    rolling_volatility_1w: float  # std(log_return) over 5 days

    # Momentum
    macd: float  # EMA_5 - EMA_21 (short-term vs long-term trend)


class SemanticScore(BaseModel):
    timestamp: datetime
    category: Optional[
        Category
    ]  # Questionable if even necessary here... For now always is going to be company specific

    headline: str
    summary: str
    full_text: str
    score: float = Field(ge=-1, le=1)


class SemanticData(BaseModel):
    assigned_timestamp: datetime  # May differ from article publication

    news_count: int
    average_sentiment: float
    sentiment_variance: float
    weighted_sentiment: float  # Legacy, kept for backward compat — use EMAs instead
    raw: List[SemanticScore]

    # Temporal features (computed by enrich_temporal_features after alignment)
    has_news: bool = True
    days_since_news: int = 0
    sentiment_emas: List[
        float
    ] = []  # One EMA per span in config.sentiment_ema_spans, α = 2/(span+1)
    sentiment_momentum: float = (
        0.0  # average_sentiment - middle EMA on news days, else 0.0
    )


class StockDataPairings(BaseModel):
    stock_id: str
    timestamp: datetime

    numeric_data: NumericData
    semantic_data: Optional[
        SemanticData
    ]  # Articles published during non-trading days will be associated to the next trading day


class StockData(BaseModel):
    stock_id: str
    category: Sectors

    start_timestamp_numeric: datetime
    end_timestamp_numeric: datetime
    start_timestamp_semantic: datetime
    end_timestamp_semantic: datetime

    data: List[StockDataPairings]


class PipelineInput(BaseModel):
    config: Config
    stock_data: List[StockData]

    initial_capital: float
