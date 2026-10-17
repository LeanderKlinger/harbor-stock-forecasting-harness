"""
Sentiment extraction quality validation harness.

Reusable experimentation framework for comparing sentiment extraction quality
across models, providers, batch sizes, prompt variations, and headline-only vs
full-article modes.

Usage:
    python src/extraction/validation_harness.py run \
        --model claude-haiku-4-5 --batch-size 3 --tag baseline

    python src/extraction/validation_harness.py compare \
        --experiments <id1> <id2> --name "headline_vs_article"

    python src/extraction/validation_harness.py list
"""

import argparse
import hashlib
import json
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np

from src.extraction.semantic import (
    BATCH_HEADLINE_SYSTEM_PROMPT,
    BATCH_SYSTEM_PROMPT,
    ArticleResult,
    CheckpointData,
    SemanticExtraction,
)
from src.extraction.validation_plots import (
    plot_bland_altman,
    plot_confusion_matrix,
    plot_distribution,
    plot_scatter,
    plot_violins,
)
from src.models.config import Config
from src.paths import VALIDATION_DIR

logger = getLogger(__name__)

CHECKPOINTS_DIR = VALIDATION_DIR / "checkpoints"
EXPERIMENTS_DIR = VALIDATION_DIR / "experiments"
COMPARISONS_DIR = VALIDATION_DIR / "comparisons"
EXPERIMENTS_INDEX_PATH = VALIDATION_DIR / "experiments_index.json"

DEFAULT_CSV = Path("data/amd_testing_articles.csv")
DEFAULT_STOCK = "AMD"


# MARK - Validation Subclass
class ValidationSemanticExtraction(SemanticExtraction):
    """Extends checkpoint key with batch_size and prompt_hash for experiment isolation."""

    def __init__(
        self,
        checkpoint_dir: Path,
        config: Config,
        articles_per_batch: int,
        prompt_hash: str,
        system_prompt_override: str | None = None,
    ):
        super().__init__(checkpoint_dir, config, parsing_tag="")
        self.articles_per_batch = articles_per_batch
        self.prompt_hash = prompt_hash
        if system_prompt_override is not None:
            self.system_prompt = system_prompt_override

    def _get_checkpoint_path(self, stock_id: str) -> Path:
        stock_dir = self.checkpoint_dir / stock_id
        stock_dir.mkdir(parents=True, exist_ok=True)
        safe_model = self.model.replace("/", "-").replace(":", "-")
        mode = "headlines" if self.headline_only else "articles"
        return (
            stock_dir
            / f"checkpoint_{safe_model}_{mode}_b{self.articles_per_batch}_p{self.prompt_hash[:8]}.json"
        )

    def get_checkpoint_path(self, stock_id: str) -> Path:
        """Public accessor for checkpoint path."""
        return self._get_checkpoint_path(stock_id)

    def load_checkpoint(self, stock_id: str) -> CheckpointData | None:
        """Public accessor for loading checkpoint."""
        return self._load_checkpoint(stock_id)


# MARK - Experiment Index
def _load_index() -> list[dict[str, Any]]:
    if not EXPERIMENTS_INDEX_PATH.exists():
        return []
    with open(EXPERIMENTS_INDEX_PATH) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def _save_index(index: list[dict[str, Any]]) -> None:
    EXPERIMENTS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def _load_experiment_config(experiment_id: str) -> dict[str, Any]:
    config_path = EXPERIMENTS_DIR / experiment_id / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_id}")
    with open(config_path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def _load_experiment_scores(experiment_id: str) -> list[ArticleResult]:
    scores_path = EXPERIMENTS_DIR / experiment_id / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found for experiment: {experiment_id}")
    with open(scores_path) as f:
        result: list[ArticleResult] = json.load(f)
        return result


# MARK - Metrics
def _compute_diagnostics(scores: list[float]) -> dict[str, float]:
    """Compute per-experiment diagnostic metrics."""
    arr = np.array(scores)
    neutral_count = int(np.sum(np.abs(arr) <= 0.1))
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))

    return {
        "count": len(scores),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "neutral_collapse_rate": neutral_count / len(scores) if scores else 0.0,
        "effective_range_low": p5,
        "effective_range_high": p95,
    }


def _compute_pairwise_metrics(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """Compute MAD and direction flip rate between two score lists."""
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)

    mad = float(np.mean(np.abs(arr_a - arr_b)))

    dir_a = np.sign(arr_a) * (np.abs(arr_a) > 0.1).astype(int)
    dir_b = np.sign(arr_b) * (np.abs(arr_b) > 0.1).astype(int)

    both_non_neutral = (dir_a != 0) & (dir_b != 0)
    if np.sum(both_non_neutral) > 0:
        flips = np.sum((dir_a[both_non_neutral] != dir_b[both_non_neutral]))
        flip_rate = float(flips / np.sum(both_non_neutral))
    else:
        flip_rate = 0.0

    return {"mad": mad, "direction_flip_rate": flip_rate}


def _compute_headline_vs_article_metrics(
    headline_scores: list[float],
    article_scores: list[float],
) -> dict[str, float]:
    """Compute amplification index and conditional bias for headline-vs-article pairs."""
    h = np.array(headline_scores)
    a = np.array(article_scores)

    amplification_index = float(np.mean(np.abs(h) - np.abs(a)))

    diff = h - a
    pos_mask = a > 0.1
    neg_mask = a < -0.1

    conditional_bias_positive = (
        float(np.mean(diff[pos_mask])) if np.any(pos_mask) else 0.0
    )
    conditional_bias_negative = (
        float(np.mean(diff[neg_mask])) if np.any(neg_mask) else 0.0
    )

    return {
        "amplification_index": amplification_index,
        "conditional_bias_positive": conditional_bias_positive,
        "conditional_bias_negative": conditional_bias_negative,
    }


def _make_experiment_label(config: dict[str, Any]) -> str:
    """Short human-readable label for an experiment."""
    model_short = (
        config["model"].split("-")[1] if "-" in config["model"] else config["model"]
    )
    mode = "headlines" if config["headline_only"] else "articles"
    label = f"{model_short}/b{config['articles_per_batch']}/{mode}"
    if config.get("tag"):
        label += f"/{config['tag']}"
    return label


# MARK - Core Functions
def run_experiment(
    csv_path: Path | None = None,
    stock_id: str = DEFAULT_STOCK,
    model: str = "claude-haiku-4-5",
    headline_only: bool = False,
    articles_per_batch: int = 3,
    prompt_version: str = "default",
    system_prompt_override: str | None = None,
    tag: str | None = None,
) -> str:
    """
    Run one experiment. Returns experiment_id.

    Scores are cached in validation checkpoints. Re-running the same config
    with a different tag is instant.
    """
    default_prompt = (
        BATCH_HEADLINE_SYSTEM_PROMPT if headline_only else BATCH_SYSTEM_PROMPT
    )
    prompt_text = system_prompt_override or default_prompt
    prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:8]
    csv_path = csv_path or DEFAULT_CSV

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Build experiment ID
    now = datetime.now()
    timestamp_str = now.strftime("%y%m%d_%H%M%S")
    model_short = model.split("-")[1] if "-" in model else model
    mode_str = "headlines" if headline_only else "articles"
    parts = [timestamp_str, model_short, f"b{articles_per_batch}", mode_str]
    if tag:
        parts.append(tag)
    experiment_id = "_".join(parts)

    experiment_dir = EXPERIMENTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = Config(
        training_start=datetime(2020, 1, 1),
        training_end=datetime(2022, 12, 31),
        test_start=datetime(2023, 1, 1),
        test_end=datetime(2023, 12, 31),
        stock_ids=[stock_id],
        semantic_model=model,
        headline_only=headline_only,
    )

    extractor = ValidationSemanticExtraction(
        checkpoint_dir=CHECKPOINTS_DIR,
        config=experiment_config,
        articles_per_batch=articles_per_batch,
        prompt_hash=prompt_hash,
        system_prompt_override=system_prompt_override,
    )

    # Check if checkpoint already exists (= cached, no API calls needed)
    checkpoint_path = extractor.get_checkpoint_path(stock_id)
    from_cache = checkpoint_path.exists()

    extractor.process_articles(
        csv_path=csv_path,
        stock_id=stock_id,
        articles_per_batch=articles_per_batch,
    )

    checkpoint = extractor.load_checkpoint(stock_id)
    if checkpoint is None:
        raise RuntimeError("Checkpoint should exist after processing")

    results = sorted(checkpoint.results, key=lambda r: r["index"])

    with open(experiment_dir / "scores.json", "w") as f:
        json.dump(results, f, indent=2)

    experiment_config_dict: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": now.isoformat(),
        "model": model,
        "provider": "anthropic" if model.startswith("claude") else "openai",
        "headline_only": headline_only,
        "articles_per_batch": articles_per_batch,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "system_prompt_text": prompt_text,
        "stock_id": stock_id,
        "csv_path": str(csv_path),
        "article_count": len(results),
        "tag": tag,
        "temperature": 0,
        "checkpoint_key": checkpoint_path.name,
        "from_cache": from_cache,
    }

    with open(experiment_dir / "config.json", "w") as f:
        json.dump(experiment_config_dict, f, indent=2)

    score_values = [r["score"] for r in results]
    diagnostics = _compute_diagnostics(score_values)

    with open(experiment_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    plot_distribution(
        score_values,
        diagnostics,
        title=f"{model_short} / {mode_str} / b{articles_per_batch}",
        output_path=experiment_dir / "distribution.png",
    )

    index = _load_index()
    index.append(
        {
            "experiment_id": experiment_id,
            "timestamp": now.isoformat(),
            "model": model,
            "headline_only": headline_only,
            "articles_per_batch": articles_per_batch,
            "stock_id": stock_id,
            "article_count": len(results),
            "tag": tag,
            "prompt_version": prompt_version,
            "from_cache": from_cache,
            "neutral_collapse_rate": diagnostics["neutral_collapse_rate"],
        }
    )
    _save_index(index)

    ncr = diagnostics["neutral_collapse_rate"]
    warn = " WARNING: neutral collapse > 70%!" if ncr > 0.7 else ""
    print(
        f"Experiment {experiment_id}: {len(results)} articles, "
        f"neutral collapse {ncr:.1%}, "
        f"range [{diagnostics['effective_range_low']:.2f}, {diagnostics['effective_range_high']:.2f}]"
        f"{warn}"
    )

    return experiment_id


def compare_experiments(
    experiment_ids: list[str],
    comparison_name: str | None = None,
) -> Path:
    """
    Compare N experiments. Returns comparison output dir.

    For 2 experiments: pairwise metrics + Bland-Altman + scatter + confusion matrix.
    If headline-vs-article pair detected: amplification index + conditional bias.
    For 3+: adds violin plot + pairwise summary table.
    """
    if len(experiment_ids) < 2:
        raise ValueError("Need at least 2 experiments to compare")

    configs: list[dict[str, Any]] = []
    all_scores: dict[str, list[ArticleResult]] = {}
    for eid in experiment_ids:
        configs.append(_load_experiment_config(eid))
        all_scores[eid] = _load_experiment_scores(eid)

    # Validate: same stock, same article count
    stock_ids = {c["stock_id"] for c in configs}
    if len(stock_ids) > 1:
        raise ValueError(f"Cannot compare across stocks: {stock_ids}")

    counts = {c["experiment_id"]: c["article_count"] for c in configs}
    unique_counts = set(counts.values())
    if len(unique_counts) > 1:
        raise ValueError(
            f"Article count mismatch — cannot compare experiments with different "
            f"row counts (likely different modes on a CSV with null articles): {counts}"
        )

    now = datetime.now()
    timestamp_str = now.strftime("%y%m%d_%H%M%S")
    if comparison_name:
        comp_id = f"{timestamp_str}_{comparison_name}"
    else:
        comp_id = f"{timestamp_str}_{'_vs_'.join(experiment_ids[:3])}"
    comp_dir = COMPARISONS_DIR / comp_id
    comp_dir.mkdir(parents=True, exist_ok=True)

    labels = {eid: _make_experiment_label(c) for eid, c in zip(experiment_ids, configs)}

    score_arrays: dict[str, list[float]] = {}
    for eid in experiment_ids:
        sorted_results = sorted(all_scores[eid], key=lambda r: r["index"])
        score_arrays[eid] = [r["score"] for r in sorted_results]

    summary_parts: list[str] = []
    summary_parts.append(f"# Comparison: {comp_id}\n")
    summary_parts.append(f"**Date**: {now.isoformat()}\n")
    summary_parts.append(f"**Experiments**: {len(experiment_ids)}\n")
    for eid, cfg in zip(experiment_ids, configs):
        summary_parts.append(
            f"- `{eid}` \u2192 {labels[eid]} ({cfg['article_count']} articles)\n"
        )
    summary_parts.append("")

    pairwise_table: list[dict[str, Any]] = []

    for i in range(len(experiment_ids)):
        for j in range(i + 1, len(experiment_ids)):
            eid_a = experiment_ids[i]
            eid_b = experiment_ids[j]
            label_a = labels[eid_a]
            label_b = labels[eid_b]
            sa = score_arrays[eid_a]
            sb = score_arrays[eid_b]

            metrics = _compute_pairwise_metrics(sa, sb)
            pairwise_table.append(
                {
                    "pair": f"{label_a} vs {label_b}",
                    "mad": metrics["mad"],
                    "direction_flip_rate": metrics["direction_flip_rate"],
                }
            )

            if len(experiment_ids) == 2:
                _generate_pairwise_outputs(
                    sa,
                    sb,
                    label_a,
                    label_b,
                    eid_a,
                    eid_b,
                    configs[i],
                    configs[j],
                    all_scores,
                    metrics,
                    summary_parts,
                    comp_dir,
                )

    # Multi-experiment: violin + pairwise summary table + per-pair plots
    if len(experiment_ids) >= 3:
        labeled_scores = {labels[eid]: score_arrays[eid] for eid in experiment_ids}
        plot_violins(labeled_scores, comp_dir / "violins.png")

        summary_parts.append("## Pairwise Summary\n")
        summary_parts.append("| Pair | MAD | Direction Flip Rate |")
        summary_parts.append("|------|-----|---------------------|")
        for row in pairwise_table:
            summary_parts.append(
                f"| {row['pair']} | {row['mad']:.4f} | {row['direction_flip_rate']:.1%} |"
            )
        summary_parts.append("")

        pairwise_dir = comp_dir / "pairwise"
        pairwise_dir.mkdir(exist_ok=True)
        for i in range(len(experiment_ids)):
            for j in range(i + 1, len(experiment_ids)):
                eid_a = experiment_ids[i]
                eid_b = experiment_ids[j]
                sa = score_arrays[eid_a]
                sb = score_arrays[eid_b]
                la = labels[eid_a]
                lb = labels[eid_b]
                safe_name = f"{eid_a}_vs_{eid_b}"
                plot_bland_altman(
                    sa, sb, la, lb, pairwise_dir / f"bland_altman_{safe_name}.png"
                )
                plot_scatter(sa, sb, la, lb, pairwise_dir / f"scatter_{safe_name}.png")
                plot_confusion_matrix(
                    sa, sb, la, lb, pairwise_dir / f"confusion_{safe_name}.png"
                )

    with open(comp_dir / "summary.md", "w") as f:
        f.write("\n".join(summary_parts))

    comp_config: dict[str, Any] = {
        "comparison_id": comp_id,
        "timestamp": now.isoformat(),
        "experiment_ids": experiment_ids,
        "labels": labels,
        "pairwise_metrics": pairwise_table,
    }
    with open(comp_dir / "config.json", "w") as f:
        json.dump(comp_config, f, indent=2)

    print(f"Comparison saved: {comp_dir}")
    return comp_dir


def _generate_pairwise_outputs(
    sa: list[float],
    sb: list[float],
    label_a: str,
    label_b: str,
    eid_a: str,
    eid_b: str,
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
    all_scores: dict[str, list[ArticleResult]],
    metrics: dict[str, float],
    summary_parts: list[str],
    comp_dir: Path,
) -> None:
    """Generate plots, metrics, and most-divergent table for a 2-experiment comparison."""
    plot_bland_altman(sa, sb, label_a, label_b, comp_dir / "bland_altman.png")
    plot_scatter(sa, sb, label_a, label_b, comp_dir / "scatter.png")
    plot_confusion_matrix(sa, sb, label_a, label_b, comp_dir / "confusion_matrix.png")

    summary_parts.append("## Pairwise Metrics\n")
    summary_parts.append(f"- **MAD**: {metrics['mad']:.4f}")
    summary_parts.append(
        f"- **Direction Flip Rate**: {metrics['direction_flip_rate']:.1%}\n"
    )

    # Headline vs article specific metrics
    is_headline_vs_article = (
        cfg_a["headline_only"] != cfg_b["headline_only"]
        and cfg_a["model"] == cfg_b["model"]
    )
    if is_headline_vs_article:
        if cfg_a["headline_only"]:
            h_scores, a_scores = sa, sb
        else:
            h_scores, a_scores = sb, sa

        hva_metrics = _compute_headline_vs_article_metrics(h_scores, a_scores)
        summary_parts.append("## Headline vs Article Analysis\n")
        amp = hva_metrics["amplification_index"]
        summary_parts.append(
            f"- **Amplification Index**: {amp:.4f} "
            f"({'headlines more extreme' if amp > 0 else 'articles more extreme'})"
        )
        summary_parts.append(
            f"- **Conditional Bias (article > 0)**: "
            f"{hva_metrics['conditional_bias_positive']:.4f}"
        )
        summary_parts.append(
            f"- **Conditional Bias (article < 0)**: "
            f"{hva_metrics['conditional_bias_negative']:.4f}\n"
        )

    # Most divergent articles
    sorted_a = sorted(all_scores[eid_a], key=lambda r: r["index"])
    sorted_b = sorted(all_scores[eid_b], key=lambda r: r["index"])

    divergent: list[dict[str, Any]] = []
    for ra, rb in zip(sorted_a, sorted_b):
        delta = abs(ra["score"] - rb["score"])
        divergent.append(
            {
                "index": ra["index"],
                "headline": ra["headline"],
                f"score_{label_a}": ra["score"],
                f"score_{label_b}": rb["score"],
                "delta": delta,
            }
        )

    divergent.sort(key=lambda d: d["delta"], reverse=True)
    top_10 = divergent[:10]

    summary_parts.append("## Most Divergent Articles (Top 10)\n")
    summary_parts.append(f"| # | Headline | {label_a} | {label_b} | Delta |")
    summary_parts.append("|---|----------|-------|-------|-------|")
    for rank, d in enumerate(top_10, 1):
        headline = (
            d["headline"][:60] + "..." if len(d["headline"]) > 60 else d["headline"]
        )
        summary_parts.append(
            f"| {rank} | {headline} | {d[f'score_{label_a}']:.2f} | "
            f"{d[f'score_{label_b}']:.2f} | {d['delta']:.2f} |"
        )
    summary_parts.append("")

    with open(comp_dir / "most_divergent.csv", "w") as f:
        headers = list(top_10[0].keys()) if top_10 else []
        f.write(",".join(headers) + "\n")
        for row in divergent:
            vals = [str(row[h]) for h in headers]
            f.write(",".join(vals) + "\n")


def list_experiments(
    model: str | None = None,
    headline_only: bool | None = None,
    stock_id: str | None = None,
) -> list[dict[str, Any]]:
    """List experiments, optionally filtered."""
    index = _load_index()

    filtered: list[dict[str, Any]] = []
    for entry in index:
        if model is not None and entry["model"] != model:
            continue
        if headline_only is not None and entry["headline_only"] != headline_only:
            continue
        if stock_id is not None and entry["stock_id"] != stock_id:
            continue
        filtered.append(entry)

    return filtered


# MARK - CLI
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Sentiment extraction quality validation harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--model", default="claude-haiku-4-5")
    run_parser.add_argument("--batch-size", type=int, default=3)
    run_parser.add_argument("--headline-only", action="store_true")
    run_parser.add_argument("--tag", default=None)
    run_parser.add_argument("--csv", default=None, help="Path to articles CSV")
    run_parser.add_argument("--stock", default=DEFAULT_STOCK)
    run_parser.add_argument("--prompt-version", default="default")

    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument(
        "--experiments", nargs="+", required=True, help="Experiment IDs to compare"
    )
    compare_parser.add_argument("--name", default=None, help="Comparison name")

    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--model", default=None)
    list_parser.add_argument("--headline-only", action="store_true")
    list_parser.add_argument("--articles-only", action="store_true")
    list_parser.add_argument("--stock", default=None)

    args = parser.parse_args()

    if args.command == "run":
        csv_path = Path(args.csv) if args.csv else None
        run_experiment(
            csv_path=csv_path,
            stock_id=args.stock,
            model=args.model,
            headline_only=args.headline_only,
            articles_per_batch=args.batch_size,
            prompt_version=args.prompt_version,
            tag=args.tag,
        )

    elif args.command == "compare":
        compare_experiments(
            experiment_ids=args.experiments,
            comparison_name=args.name,
        )

    elif args.command == "list":
        if args.headline_only and args.articles_only:
            parser.error("Cannot specify both --headline-only and --articles-only")
        headline_filter: bool | None = None
        if args.headline_only:
            headline_filter = True
        elif args.articles_only:
            headline_filter = False
        results = list_experiments(
            model=args.model,
            headline_only=headline_filter,
            stock_id=args.stock,
        )
        if not results:
            print("No experiments found.")
            return

        print(
            f"{'ID':<50} {'Model':<20} {'Mode':<12} {'Articles':>8} {'NCR':>6} {'Tag'}"
        )
        print("-" * 110)
        for r in results:
            mode = "headlines" if r["headline_only"] else "articles"
            ncr = r.get("neutral_collapse_rate", 0)
            tag = r.get("tag", "") or ""
            print(
                f"{r['experiment_id']:<50} {r['model']:<20} {mode:<12} "
                f"{r['article_count']:>8} {ncr:>5.1%} {tag}"
            )


if __name__ == "__main__":
    _cli()
