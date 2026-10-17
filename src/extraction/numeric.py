import logging
from pathlib import Path

import polars as pl

from src.models.config import Config
from src.models.input import NumericData

logger = logging.getLogger(__name__)


class NumericExtraction:
    def __init__(self, data_dir: Path, config: Config):
        self.data_dir = data_dir
        self.config = config

    def extract(self, stock_id: str) -> list[NumericData]:
        """
        Extract NumericData for a stock from its CSV.
        Returns list sorted chronologically (oldest first).
        """
        csv_path = self.data_dir / f"{stock_id}.csv"
        if not csv_path.exists():
            logger.warning(f"{csv_path.stem} doesn't exist: skipping {stock_id}")
            return []

        df = self._load_from_csv(csv_path)
        if df.height < 21:
            logger.warning(
                f"{csv_path.stem} doesn't have more than 21 records: skipping {stock_id}"
            )
            return []

        df_indicators = self._calculate_indicators(df)

        # Drop rows without enough history for EMA_21
        df_indicators = df_indicators.slice(21)

        results: list[NumericData] = []
        for row in df_indicators.iter_rows(named=True):
            results.append(
                NumericData(
                    timestamp=row["date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=int(row["volume"]),
                    log_return=row["log_return"],
                    annualized_return_1w=row["ann_ret_1w"],
                    annualized_return_2w=row["ann_ret_2w"],
                    annualized_return_1m=row["ann_ret_1m"],
                    rolling_volatility_1w=row["rolling_vol_1w"],
                    macd=row["macd"],
                )
            )
        return results

    def _load_from_csv(self, csv_path: Path) -> pl.DataFrame:
        """Load CSV and sort chronologically."""
        df = pl.read_csv(csv_path, try_parse_dates=True)
        df = df.select(["date", "open", "high", "low", "close", "volume"])

        na_count = df.null_count().sum_horizontal().item()
        if na_count > 0:
            logger.warning(f"{csv_path.stem}: dropping rows with {na_count} NA values")

        df = df.drop_nulls()

        # Filter out rows with invalid OHLCV values (must be > 0)
        initial_count = df.height
        df = df.filter(
            (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
            & (pl.col("volume") >= 0)
        )
        dropped = initial_count - df.height
        if dropped > 0:
            logger.warning(
                f"{csv_path.stem}: dropped {dropped} rows with invalid OHLCV values"
            )

        return df.sort("date")

    def _calculate_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate all derived metrics."""
        trading_days = self.config.trading_days

        return df.with_columns(
            # Log returns
            (pl.col("close") / pl.col("close").shift(1))
            .log()
            .fill_null(0)
            .alias("log_return"),
            # Annualized returns
            self._annualized_return(5, trading_days).alias("ann_ret_1w"),
            self._annualized_return(10, trading_days).alias("ann_ret_2w"),
            self._annualized_return(21, trading_days).alias("ann_ret_1m"),
            # Rolling volatility (std of log returns over 5 days)
            (pl.col("close") / pl.col("close").shift(1))
            .log()
            .rolling_std(5)
            .fill_null(0)
            .alias("rolling_vol_1w"),
            # MACD: EMA_5 - EMA_21
            (
                pl.col("close").ewm_mean(span=5) - pl.col("close").ewm_mean(span=21)
            ).alias("macd"),
        )

    def _annualized_return(self, lookback: int, trading_days: int) -> pl.Expr:
        """Calculate annualized return over lookback period."""
        period_return = (pl.col("close") - pl.col("close").shift(lookback)) / pl.col(
            "close"
        ).shift(lookback)
        return (period_return * (trading_days / lookback)).fill_null(0)
