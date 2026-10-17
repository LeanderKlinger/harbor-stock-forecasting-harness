"""
Reindex all runs under data/system/execution_output/.

Rebuilds runs_index.json from pipeline_output.json files on disk,
generates run_detail.json for each run, and ensures shared stock price files exist.

Usage:
    python -m src.experiments reindex [--force]
"""

import json
import time
from pathlib import Path
from typing import Any

from src.evaluation.evaluation_module import EvaluationModule
from src.evaluation.persistence import (
    STOCK_PRICES_DIR,
    build_index_entry,
    build_run_detail,
    ensure_stock_prices,
)
from src.models.output import PipelineOutput
from src.paths import EXECUTION_OUTPUT_ROOT, RUNS_INDEX_PATH


def _find_run_dirs() -> list[Path]:
    """Find all run directories that contain a pipeline_output.json."""
    run_dirs: list[Path] = []
    for model_dir in EXECUTION_OUTPUT_ROOT.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        if model_dir.name == "comparisons":
            continue
        for run_dir in model_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "pipeline_output.json").exists():
                run_dirs.append(run_dir)
    return sorted(run_dirs)


def run_reindex(force: bool = False) -> None:
    """Reindex all runs: rebuild runs_index.json + generate run_detail.json files."""
    t_start = time.perf_counter()

    run_dirs = _find_run_dirs()
    print(f"Found {len(run_dirs)} run directories")

    if not run_dirs:
        print("Nothing to reindex.")
        return

    eval_module = EvaluationModule()
    all_stock_ids: set[str] = set()
    index_entries: list[dict[str, Any]] = []
    details_written = 0
    errors: list[str] = []

    for i, run_dir in enumerate(run_dirs):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(run_dirs)}...")

        try:
            output = PipelineOutput.model_validate_json(
                (run_dir / "pipeline_output.json").read_text()
            )
            evaluation = eval_module.evaluate(output)

            # Write run_detail.json if missing or forced
            detail_path = run_dir / "run_detail.json"
            if force or not detail_path.exists():
                run_detail = build_run_detail(output, evaluation)
                detail_path.write_text(json.dumps(run_detail, indent=2))
                details_written += 1

            entry = build_index_entry(run_dir, output, evaluation)
            index_entries.append(entry)

            stock_ids = entry.get("config_summary", {}).get("stock_ids", [])
            all_stock_ids.update(stock_ids)

        except Exception as e:
            errors.append(f"{run_dir.name}: {e}")

    # Sort index by timestamp (newest first)
    index_entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    # Write rebuilt runs_index.json (atomic rename)
    tmp = RUNS_INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(index_entries, indent=2))
    tmp.rename(RUNS_INDEX_PATH)
    print(f"Wrote runs_index.json with {len(index_entries)} entries")

    # Ensure shared stock price files
    if all_stock_ids:
        print(f"Ensuring stock price files for {len(all_stock_ids)} stocks...")
        if force:
            if STOCK_PRICES_DIR.exists():
                for f in STOCK_PRICES_DIR.iterdir():
                    f.unlink()
        ensure_stock_prices(sorted(all_stock_ids))

    elapsed = time.perf_counter() - t_start

    print(f"\nReindex complete in {elapsed:.1f}s")
    print(f"  {len(index_entries)} runs indexed")
    print(f"  {details_written} run_detail.json files written")
    if errors:
        print(f"  {len(errors)} errors:")
        for err in errors[:20]:
            print(f"    - {err}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
