import asyncio
import json
from datetime import datetime
from collections.abc import AsyncGenerator
from logging import getLogger
from pathlib import Path
from typing import TypedDict

import anthropic
import openai
import polars as pl
from anthropic.types.beta import BetaTextBlock
from dotenv import load_dotenv
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from src.models.config import Config
from src.models.input import SemanticScore

load_dotenv()
logger = getLogger(__name__)


# MARK - Support Classes
class SingleArticleOutput(BaseModel):
    """Output for a single article in a batch."""

    article_index: int
    summary: str = ""
    sentiment_score: float


class SingleHeadlineOutput(BaseModel):
    """Output for a single headline in a batch (no summary)."""

    article_index: int
    sentiment_score: float


class BatchedModelOutput(BaseModel):
    """Output for a batch of articles."""

    results: list[SingleArticleOutput]


class BatchedHeadlineOutput(BaseModel):
    """Output for a batch of headlines (no summary)."""

    results: list[SingleHeadlineOutput]


class ArticleInput(TypedDict):
    """Input article for batch processing."""

    index: int
    headline: str
    body: str
    date_str: str


class ArticleResult(TypedDict):
    index: int
    timestamp: str
    headline: str
    full_text: str  # Headline in headline mode, first 500 chars of body in article mode
    score: float
    summary: str


class CheckpointData(BaseModel):
    stock_id: str
    model: str
    processed_count: int  # Legacy field, kept for compatibility
    completed_indices: list[int]
    results: list[ArticleResult]
    last_updated: str


# MARK - Prompt
SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analyzer for stock price prediction research. Your task is to assess how a news article is likely to impact the stock price of the company/companies mentioned.

## Scoring Guidelines (-1.0 to 1.0)

### Strong Negative (-1.0 to -0.7)
- Earnings miss, revenue decline, lowered guidance
- Executive misconduct, fraud allegations, regulatory violations
- Major product failures, recalls, safety issues
- Significant layoffs, bankruptcy concerns
- Loss of major contracts/partnerships

### Moderate Negative (-0.7 to -0.3)
- Mixed earnings with concerning trends
- Increased competition eroding market share
- Analyst downgrades with specific concerns
- Supply chain or operational challenges
- Sector headwinds affecting the company

### Slight Negative (-0.3 to -0.1)
- Minor operational setbacks
- Cautious analyst commentary
- Broader market weakness mentioned alongside company

### Neutral (-0.1 to 0.1)
- Purely informational (ETF composition, index membership)
- Historical analysis without forward implications
- Balanced coverage with no clear directional signal
- News already fully priced in

### Slight Positive (0.1 to 0.3)
- Routine product updates
- Minor analyst mentions without strong conviction
- General positive market sentiment

### Moderate Positive (0.3 to 0.7)
- Earnings beat, revenue growth, raised guidance
- New product launches with clear market demand
- Strategic partnerships or acquisitions
- Analyst upgrades with price target increases
- Market share gains

### Strong Positive (0.7 to 1.0)
- Significant earnings surprise (>10% beat)
- Transformative deals, major contract wins
- Breakthrough products or regulatory approvals
- Entry into high-growth markets with clear advantage

## Important Rules
1. **Time horizon**: Assess likely impact over the NEXT 1-5 TRADING DAYS
2. **Stock-specific**: Score based on impact to the PRIMARY company mentioned, not general market
3. **Already priced in**: If news is old or widely known, bias toward neutral
4. **Mixed signals**: When positive and negative factors coexist, weight by magnitude and certainty
5. **Avoid extremes**: Reserve ±0.8 to ±1.0 for unambiguous, material news events"""

HEADLINE_ADDENDUM = """

## Headline-Only Mode
You are scoring based on the headline alone — no article body is available. Keep these guidelines in mind:
- The headline IS all the information you have. Do not default to neutral just because context is limited.
- Headlines with explicit buy/sell recommendations, analyst actions, or earnings language carry clear signal — score them accordingly.
- Buy-recommendation questions ("Is X a Buy?", "Should You Buy X?") in financial media typically imply "yes" — treat as slightly positive (0.1 to 0.2). But negatively-framed questions ("Why Is X Falling?", "Is X Overvalued?") carry the negative signal in the framing — score based on the sentiment of the content words, not the question structure.
- Factual product/business news (launches, expansions, new GPUs) is real signal — score based on likely market impact, same as you would with a full article.
- Many headlines in financial datasets are genuinely irrelevant to the target stock (sector roundups, index movers lists, unrelated companies). These SHOULD be scored neutral — do not force a directional score when there is no signal."""

BATCH_SYSTEM_PROMPT = (
    SENTIMENT_SYSTEM_PROMPT
    + """

## Batch Processing Instructions
You will receive multiple articles to analyze. Return results for ALL articles in the batch.
Each result must include the article_index matching the input index."""
)

BATCH_HEADLINE_SYSTEM_PROMPT = (
    SENTIMENT_SYSTEM_PROMPT
    + HEADLINE_ADDENDUM
    + """

## Batch Processing Instructions
You will receive multiple articles to analyze. Return results for ALL articles in the batch.
Each result must include the article_index matching the input index."""
)


def _build_batch_user_prompt(
    articles: list[ArticleInput], stock: str, headline_only: bool = False
) -> str:
    """Build user prompt for batch analysis."""
    parts = [f"Analyze these financial news articles for the stock {stock}:\n"]
    for article in articles:
        if headline_only:
            parts.append(f"""
---
ARTICLE INDEX: {article["index"]}
HEADLINE: {article["headline"]}
---""")
        else:
            parts.append(f"""
---
ARTICLE INDEX: {article["index"]}
HEADLINE: {article["headline"]}
BODY:
{article["body"]}
---""")
    return "\n".join(parts)


# MARK - Implementation
class SemanticExtraction:
    def __init__(self, checkpoint_dir: Path, config: Config, parsing_tag: str):
        self.anthropic_client = anthropic.AsyncAnthropic()
        self.openai_client = openai.AsyncOpenAI()
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.model = config.semantic_model
        self.headline_only = config.headline_only
        self.system_prompt = BATCH_SYSTEM_PROMPT
        self.parsing_tag = parsing_tag

    async def _analyze_batch(
        self,
        articles: list[ArticleInput],
        stock: str,
    ) -> tuple[list[SingleArticleOutput], int, int]:
        """Route batch to appropriate provider. Returns (results, input_tokens, output_tokens)."""
        if self.model.startswith("claude"):
            return await self._analyze_batch_anthropic(articles, stock)
        elif self.model.startswith("gpt"):
            return await self._analyze_batch_openai(articles, stock)
        else:
            raise ValueError(f"Unknown model provider for: {self.model}")

    def _build_output_schema(self) -> dict[str, object]:
        """Build JSON schema for structured output, conditional on headline_only."""
        item_properties: dict[str, object] = {
            "article_index": {
                "type": "integer",
                "description": "Index of the article being analyzed",
            },
            "sentiment_score": {
                "type": "number",
                "description": "Score from -1.0 to 1.0",
            },
        }
        required = ["article_index", "sentiment_score"]

        if not self.headline_only:
            item_properties["summary"] = {
                "type": "string",
                "description": "Concise summary of key points, max 100 words",
            }
            required.append("summary")

        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["results"],
            "additionalProperties": False,
        }

    async def _analyze_batch_anthropic(
        self,
        articles: list[ArticleInput],
        stock: str,
    ) -> tuple[list[SingleArticleOutput], int, int]:
        """Analyze batch of articles using Anthropic Claude models."""
        user_prompt = _build_batch_user_prompt(articles, stock, self.headline_only)

        # noinspection PyTypeChecker
        message = await self.anthropic_client.beta.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0,
            betas=["structured-outputs-2025-11-13"],
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {
                        "type": "ephemeral"
                    },  # This caches the system instructions
                }
            ],
            output_format={
                "type": "json_schema",
                "schema": self._build_output_schema(),
            },
            messages=[{"role": "user", "content": user_prompt}],
        )

        if not isinstance(message.content[0], BetaTextBlock):
            raise ValueError(f"Expected text block, got {type(message.content[0])}")

        parsed = BatchedModelOutput.model_validate_json(message.content[0].text)
        return parsed.results, message.usage.input_tokens, message.usage.output_tokens

    async def _analyze_batch_openai(
        self,
        articles: list[ArticleInput],
        stock: str,
    ) -> tuple[list[SingleArticleOutput], int, int]:
        """Analyze batch of articles using OpenAI GPT models."""
        user_prompt = _build_batch_user_prompt(articles, stock, self.headline_only)

        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": self.system_prompt,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user_prompt,
        }

        response_format = (
            BatchedHeadlineOutput if self.headline_only else BatchedModelOutput
        )

        response = await self.openai_client.chat.completions.parse(
            model=self.model,
            max_completion_tokens=2000,
            temperature=0,
            response_format=response_format,
            messages=[system_message, user_message],
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            refusal = response.choices[0].message.refusal
            if refusal:
                raise ValueError(f"OpenAI refused request: {refusal}")
            raise ValueError("OpenAI returned empty response")

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        # Convert headline outputs to SingleArticleOutput (summary defaults to "")
        results: list[SingleArticleOutput] = []
        for r in parsed.results:
            if isinstance(r, SingleHeadlineOutput):
                results.append(
                    SingleArticleOutput(
                        article_index=r.article_index,
                        sentiment_score=r.sentiment_score,
                    )
                )
            else:
                results.append(r)

        return results, input_tokens, output_tokens

    def _get_checkpoint_path(self, stock_id: str) -> Path:
        """Get checkpoint file path for a stock, model, and extraction mode."""
        stock_dir = self.checkpoint_dir / stock_id
        stock_dir.mkdir(parents=True, exist_ok=True)
        safe_model = self.model.replace("/", "-").replace(":", "-")
        mode = "headlines" if self.headline_only else "articles"
        parsing_tag = f"_{self.parsing_tag}" if self.parsing_tag else ""

        return stock_dir / f"checkpoint_{safe_model}_{mode}{parsing_tag}.json"

    def _load_checkpoint(self, stock_id: str) -> CheckpointData | None:
        """Load existing checkpoint if available, handling legacy format."""
        checkpoint_path = self._get_checkpoint_path(stock_id)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)

            # Handle legacy checkpoints without completed_indices
            if "completed_indices" not in data:
                logger.warning(
                    f"Legacy checkpoint detected for {stock_id}. "
                    f"Please run migrate_checkpoint.py to convert to new format."
                )
                # For legacy checkpoints, assume first N articles were processed sequentially
                # This is the old (broken) behavior but maintains backward compatibility
                data["completed_indices"] = list(range(data.get("processed_count", 0)))

            return CheckpointData.model_validate(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Checkpoint corrupted for {stock_id}, starting fresh: {e}")
            return None

    def _save_checkpoint(
        self,
        stock_id: str,
        results: list[ArticleResult],
        completed_indices: list[int],
    ) -> None:
        """Save checkpoint to disk."""
        checkpoint_path = self._get_checkpoint_path(stock_id)
        checkpoint = CheckpointData(
            stock_id=stock_id,
            model=self.model,
            processed_count=len(results),
            completed_indices=completed_indices,
            results=results,
            last_updated=datetime.now().isoformat(),
        )

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.model_dump(), f, indent=2)

    def _results_to_semantic_scores(
        self, results: list[ArticleResult]
    ) -> list[SemanticScore]:
        """Convert checkpoint results to SemanticScore objects."""
        semantic_scores: list[SemanticScore] = []
        for r in results:
            try:
                timestamp = datetime.fromisoformat(r["timestamp"].replace(" ", "T"))
            except (ValueError, AttributeError):
                logger.warning(
                    f"Using current timestamp for article: {r['headline'][:50]}..."
                )
                timestamp = datetime.now()

            semantic_scores.append(
                SemanticScore(
                    timestamp=timestamp,
                    headline=r["headline"],
                    summary=r["summary"],
                    full_text=r["full_text"],
                    score=r["score"],
                    category=None,
                )
            )

        return semantic_scores

    async def _process_batch_with_retry(
        self,
        articles: list[ArticleInput],
        stock_id: str,
        max_retries: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[tuple[ArticleInput, SingleArticleOutput]], int, int]:
        """Process a single batch with retry logic and semaphore for rate limiting."""
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    results, in_tok, out_tok = await self._analyze_batch(
                        articles, stock_id
                    )

                    # Match results back to articles by index
                    result_map = {r.article_index: r for r in results}
                    matched: list[tuple[ArticleInput, SingleArticleOutput]] = []
                    for article in articles:
                        if article["index"] in result_map:
                            matched.append((article, result_map[article["index"]]))
                        else:
                            logger.warning(
                                f"Missing result for article {article['index']}"
                            )

                    return matched, in_tok, out_tok

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(f"Batch failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Batch failed after {max_retries} retries: {e}")
                        raise

        return [], 0, 0  # Should not reach here

    def process_articles(
        self,
        csv_path: Path,
        stock_id: str,
        articles_per_batch: int = 5,
        max_concurrent: int = 10,
        checkpoint_every: int = 250,
        max_retries: int = 3,
        reparse: bool = False,
    ) -> list[SemanticScore]:
        """
        Process articles with batching and concurrency.

        Uses self.model and self.headline_only from config.

        Args:
            csv_path: Path to CSV with columns Date, Article_title, Article, etc.
            stock_id: Stock identifier (e.g., "AMD")
            articles_per_batch: Number of articles per API call
            max_concurrent: Maximum concurrent API calls
            checkpoint_every: Save checkpoint every N articles
            max_retries: Max retries for failed batches
            reparse: If True, delete existing checkpoint and start fresh

        Returns:
            List of SemanticScore objects
        """
        return asyncio.run(
            self._process_articles_async(
                csv_path=csv_path,
                stock_id=stock_id,
                articles_per_batch=articles_per_batch,
                max_concurrent=max_concurrent,
                checkpoint_every=checkpoint_every,
                max_retries=max_retries,
                reparse=reparse,
            )
        )

    async def _process_articles_async(
        self,
        csv_path: Path,
        stock_id: str,
        articles_per_batch: int,
        max_concurrent: int,
        checkpoint_every: int,
        max_retries: int,
        reparse: bool,
    ) -> list[SemanticScore]:
        """Async implementation of article processing."""
        df = pl.read_csv(csv_path, null_values=["nan", "NaN", ""])
        if not self.headline_only:
            df = df.filter(pl.col("Article").is_not_null() & (pl.col("Article") != ""))

        total_articles = df.height
        logger.info(f"Starting processing for {stock_id}: {total_articles} articles")

        # Handle reparse flag
        checkpoint_path = self._get_checkpoint_path(stock_id)
        if reparse and checkpoint_path.exists():
            logger.info(f"Reparse requested, deleting checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()

        # Load checkpoint
        checkpoint = self._load_checkpoint(stock_id)
        results: list[ArticleResult] = []
        completed_indices: list[int] = []

        if checkpoint is not None:
            results = checkpoint.results
            completed_indices = checkpoint.completed_indices

            if len(completed_indices) >= total_articles:
                logger.info(
                    f"Using cached results for {stock_id}: {len(results)} articles"
                )
                # Sort by index before returning
                results_sorted = sorted(results, key=lambda r: r["index"])
                return self._results_to_semantic_scores(results_sorted)

            logger.info(
                f"Resuming from checkpoint: {len(completed_indices)}/{total_articles} completed"
            )

        # Cost tracking
        cost_map = {
            "claude-sonnet-4-5": (3.0, 15.0),
            "claude-haiku-4-5": (0.80, 4.0),
            "gpt-5.2": (1.75, 14.0),
            "gpt-5-mini": (0.25, 2),
        }
        input_cost, output_cost = cost_map.get(self.model, (0.0, 0.0))
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        # Build article inputs (only for unprocessed articles)
        rows = df.to_dicts()
        article_inputs: list[ArticleInput] = []
        for i in range(total_articles):
            # Skip already completed articles
            if i in completed_indices:
                continue

            row = rows[i]
            headline = row.get("Article_title", "")

            if self.headline_only:
                if not headline:
                    continue
                body = ""
            else:
                body = row.get("Article", "")
                if not body:
                    continue

            article_inputs.append(
                ArticleInput(
                    index=i,
                    headline=headline,
                    body=body,
                    date_str=row.get("Date", ""),
                )
            )

        # Create batches
        batches: list[list[ArticleInput]] = []
        for i in range(0, len(article_inputs), articles_per_batch):
            batches.append(article_inputs[i : i + articles_per_batch])

        logger.info(
            f"Processing {len(article_inputs)} articles in {len(batches)} batches "
            f"({articles_per_batch}/batch, {max_concurrent} concurrent)"
        )

        semaphore = asyncio.Semaphore(max_concurrent)
        processed_since_checkpoint = 0

        # Process batches with progress bar
        pbar = tqdm(
            self._process_batches(batches, stock_id, max_retries, semaphore),
            total=len(batches),
            desc=f"Processing {stock_id} ({'headlines' if self.headline_only else 'articles'})",
        )
        async for batch_result in pbar:
            matched, in_tok, out_tok = batch_result
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_cost = (
                total_input_tokens / 1e6 * input_cost
                + total_output_tokens / 1e6 * output_cost
            )
            pbar.set_postfix(cost=f"${total_cost:.3f}")  # pyright: ignore [reportUnknownMemberType]

            for article, output in matched:
                full_text = (
                    article["headline"] if self.headline_only else article["body"][:500]
                )
                result_dict: ArticleResult = {
                    "index": article["index"],
                    "timestamp": article["date_str"],
                    "headline": article["headline"],
                    "full_text": full_text,
                    "score": output.sentiment_score,
                    "summary": output.summary,
                }
                results.append(result_dict)

                # Check for duplicates before adding
                idx = article["index"]
                if idx in completed_indices:
                    logger.warning(f"Duplicate index {idx} in results, skipping add")
                else:
                    completed_indices.append(idx)

                processed_since_checkpoint += 1

            # Checkpoint periodically
            if processed_since_checkpoint >= checkpoint_every:
                self._save_checkpoint(stock_id, results, completed_indices)
                processed_since_checkpoint = 0
                logger.info(
                    f"Checkpoint: {len(completed_indices)}/{total_articles}, Cost: ${total_cost:.4f}"
                )

        # Final save
        self._save_checkpoint(stock_id, results, completed_indices)
        total_cost = (
            total_input_tokens / 1e6 * input_cost
            + total_output_tokens / 1e6 * output_cost
        )
        logger.info(
            f"Processing complete. Total: {len(results)}, Cost: ${total_cost:.4f}"
        )

        # Sort results by index to ensure chronological order
        results_sorted = sorted(results, key=lambda r: r["index"])
        return self._results_to_semantic_scores(results_sorted)

    async def _process_batches(
        self,
        batches: list[list[ArticleInput]],
        stock_id: str,
        max_retries: int,
        semaphore: asyncio.Semaphore,
    ) -> AsyncGenerator[
        tuple[list[tuple[ArticleInput, SingleArticleOutput]], int, int], None
    ]:
        """Yield batch results as they complete."""
        tasks = [
            self._process_batch_with_retry(batch, stock_id, max_retries, semaphore)
            for batch in batches
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                yield result
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                empty_result: list[tuple[ArticleInput, SingleArticleOutput]] = []
                yield empty_result, 0, 0
