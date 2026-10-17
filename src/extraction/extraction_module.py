import statistics
from datetime import datetime
from logging import getLogger

from src.extraction.numeric import NumericExtraction
from src.extraction.semantic import SemanticExtraction
from src.models.config import Config
from src.models.input import (
    NumericData,
    PipelineInput,
    SemanticData,
    SemanticScore,
    STOCK_SECTOR_MAP,
    StockData,
    StockDataPairings,
)
from src.paths import ARTICLES_DIR, PARSING_CHECKPOINTS_DIR, STOCK_DATA_DIR

logger = getLogger(__name__)


def enrich_temporal_features(
    pairings: list[StockDataPairings],
    ema_spans: list[int],
) -> list[StockDataPairings]:
    """
    Walk pairings chronologically and compute temporal sentiment features.

    After enrichment, every pairing has a non-None semantic_data with:
    - has_news / days_since_news
    - sentiment_emas (one per span, α = 2/(span+1))
    - sentiment_momentum (average_sentiment - middle EMA on news days, else 0)

    No-news days get a synthetic SemanticData with decayed EMAs.
    """
    if not ema_spans:
        return pairings

    alphas = [2.0 / (span + 1) for span in ema_spans]
    emas = [0.0] * len(ema_spans)
    days_since_news = 0
    middle_idx = len(ema_spans) // 2

    for pairing in pairings:
        s = pairing.semantic_data
        has_news = s is not None and s.news_count > 0

        if has_news:
            assert s is not None  # for pyright
            raw = s.average_sentiment
            days_since_news = 0
        else:
            raw = 0.0
            days_since_news += 1

        # Update EMAs
        for i, alpha in enumerate(alphas):
            emas[i] = alpha * raw + (1 - alpha) * emas[i]

        momentum = (
            (s.average_sentiment - emas[middle_idx])
            if has_news and s is not None
            else 0.0
        )

        if s is not None:
            s.has_news = has_news
            s.days_since_news = days_since_news
            s.sentiment_emas = list(emas)
            s.sentiment_momentum = momentum
        else:
            pairing.semantic_data = SemanticData(
                assigned_timestamp=pairing.timestamp,
                news_count=0,
                average_sentiment=0.0,
                sentiment_variance=0.0,
                weighted_sentiment=0.0,
                raw=[],
                has_news=False,
                days_since_news=days_since_news,
                sentiment_emas=list(emas),
                sentiment_momentum=0.0,
            )

    return pairings


class ExtractionModule:
    def __init__(self, config: Config, parsing_tag: str = ""):
        self.config = config
        self.articles_dir = ARTICLES_DIR
        self.numeric_extractor = NumericExtraction(STOCK_DATA_DIR, config)
        self.semantic_extractor = SemanticExtraction(
            PARSING_CHECKPOINTS_DIR, config, parsing_tag
        )

    def _align_semantic_to_trading_days(
        self,
        semantic_scores: list[SemanticScore],
        trading_days: list[datetime],
    ) -> dict[datetime, SemanticData]:
        """
        Align articles to next trading day.

        Articles published on non-trading days (weekends, holidays)
        get assigned to the next available trading day.
        """
        sorted_trading_days = sorted(trading_days)

        day_to_articles: dict[datetime, list[SemanticScore]] = {}
        skipped_count = 0

        for score in semantic_scores:
            article_date = score.timestamp.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            # Find next trading day >= article_date
            assigned = None
            for td in sorted_trading_days:
                if td >= article_date:
                    assigned = td
                    break

            if assigned is None:
                skipped_count += 1
                continue  # Article after all trading days

            if assigned not in day_to_articles:
                day_to_articles[assigned] = []
            day_to_articles[assigned].append(score)

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} articles outside trading day range")

        # Aggregate into SemanticData
        result: dict[datetime, SemanticData] = {}
        for day, articles in day_to_articles.items():
            scores = [a.score for a in articles]
            result[day] = SemanticData(
                assigned_timestamp=day,
                news_count=len(articles),
                average_sentiment=statistics.mean(scores),
                sentiment_variance=statistics.variance(scores)
                if len(scores) > 1
                else 0.0,
                weighted_sentiment=statistics.mean(scores),
                raw=articles,
            )

        return result

    def _extract_stock(
        self,
        stock_id: str,
    ) -> tuple[
        list[StockDataPairings], dict[datetime, SemanticData], list[NumericData]
    ]:
        """Extract numeric + semantic data for one stock. Returns (pairings, semantic_by_day, filtered_numeric)."""
        numeric_data = self.numeric_extractor.extract(stock_id=stock_id)
        if not numeric_data:
            raise ValueError(f"No numeric data found for stock {stock_id}")

        filtered = [
            d
            for d in numeric_data
            if self.config.training_start <= d.timestamp <= self.config.test_end
        ]
        if not filtered:
            raise ValueError(
                f"No data for {stock_id} in range "
                f"{self.config.training_start} to {self.config.test_end}"
            )

        # Semantic extraction (if articles CSV exists)
        articles_path = self.articles_dir / f"{stock_id}.csv"
        semantic_by_day: dict[datetime, SemanticData] = {}
        if articles_path.exists():
            semantic_scores = self.semantic_extractor.process_articles(
                csv_path=articles_path,
                stock_id=stock_id,
            )
            trading_days = [d.timestamp for d in filtered]
            semantic_by_day = self._align_semantic_to_trading_days(
                semantic_scores, trading_days
            )

        # Build pairings
        pairings: list[StockDataPairings] = []
        for d in filtered:
            pairings.append(
                StockDataPairings(
                    stock_id=stock_id,
                    timestamp=d.timestamp,
                    numeric_data=d,
                    semantic_data=semantic_by_day.get(d.timestamp),
                )
            )

        # Enrich temporal features if we had articles
        if articles_path.exists():
            enrich_temporal_features(pairings, self.config.sentiment_ema_spans)

        return pairings, semantic_by_day, filtered

    def build_pipeline_input(
        self,
        initial_capital: float = 100_000.0,
    ) -> PipelineInput:
        """
        Build PipelineInput for all stocks in config.stock_ids.

        For each stock:
        - Extracts numeric data, filters to config date range
        - If articles CSV exists at ARTICLES_DIR/{STOCK}.csv, extracts + aligns semantic data
        - Enriches temporal features
        """
        all_stock_data: list[StockData] = []

        for stock_id in self.config.stock_ids:
            pairings, semantic_by_day, filtered = self._extract_stock(stock_id)

            # Determine semantic date range
            semantic_start = filtered[0].timestamp
            semantic_end = filtered[-1].timestamp
            if semantic_by_day:
                semantic_timestamps = list(semantic_by_day.keys())
                semantic_start = min(semantic_timestamps)
                semantic_end = max(semantic_timestamps)

            sector = STOCK_SECTOR_MAP.get(stock_id)
            if sector is None:
                raise ValueError(
                    f"No sector mapping for {stock_id}. "
                    f"Add it to STOCK_SECTOR_MAP in input.py."
                )

            all_stock_data.append(
                StockData(
                    stock_id=stock_id,
                    category=sector,
                    start_timestamp_numeric=filtered[0].timestamp,
                    end_timestamp_numeric=filtered[-1].timestamp,
                    start_timestamp_semantic=semantic_start,
                    end_timestamp_semantic=semantic_end,
                    data=pairings,
                )
            )

        return PipelineInput(
            config=self.config,
            stock_data=all_stock_data,
            initial_capital=initial_capital,
        )
