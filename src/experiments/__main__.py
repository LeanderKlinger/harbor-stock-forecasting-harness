"""
CLI entry point for experiments.

Usage:
    python -m src.experiments run experiments/my_grid.yaml [--workers 4] [--device cpu|cuda|mps]
    python -m src.experiments reindex [--force]
"""

import argparse
import os
import sys
from pathlib import Path

from src.experiments.reindex import run_reindex
from src.experiments.runner import run_experiment

_VALID_DEVICES = {"cpu", "cuda", "mps"}


def _resolve_workers(raw: str, device: str | None) -> int:
    """Resolve --workers value. 'auto' computes based on cores and device."""
    if raw != "auto":
        try:
            n = int(raw)
        except ValueError:
            print(
                f"Error: --workers must be a positive integer or 'auto', got: {raw!r}"
            )
            sys.exit(1)
        if n < 1:
            print("Error: --workers must be >= 1")
            sys.exit(1)
        return n

    cores = os.cpu_count() or 1
    if device in ("cuda", "mps"):
        return min(8, cores)
    return max(1, cores // 2)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.experiments")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run an experiment grid")
    run_parser.add_argument("yaml_path", type=Path, help="Path to YAML config")
    run_parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=sorted(_VALID_DEVICES),
        help="Device override for training (default: auto-detect per model)",
    )
    run_parser.add_argument(
        "--workers",
        type=str,
        default="1",
        help="Number of outer parallel workers (default: 1 = sequential, 'auto' = cores/2)",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the per-stock training cache (force fresh fits on every run)",
    )

    reindex_parser = sub.add_parser(
        "reindex", help="Rebuild runs_index.json and generate run_detail.json files"
    )
    reindex_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all run_detail.json and stock price files even if they exist",
    )

    args = parser.parse_args()

    if args.command == "run":
        yaml_path: Path = args.yaml_path
        if not yaml_path.exists():
            print(f"Error: {yaml_path} not found")
            sys.exit(1)

        workers = _resolve_workers(args.workers, args.device)
        run_ids = run_experiment(
            yaml_path,
            device=args.device,
            workers=workers,
            no_cache=args.no_cache,
        )
        print(f"\nCompleted {len(run_ids)} runs")

    elif args.command == "reindex":
        run_reindex(force=args.force)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
