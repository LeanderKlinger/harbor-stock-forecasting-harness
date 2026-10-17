#!/usr/bin/env bash
# One-time setup for a fresh clone. Idempotent — safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  (see https://docs.astral.sh/uv/)"
  exit 1
fi

echo "==> Installing Python dependencies (uv sync)..."
uv sync

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> Created .env — add your ANTHROPIC_API_KEY to it (required for sentiment extraction)."
else
  echo "==> .env already exists, leaving it untouched."
fi

echo "==> Sanity-checking the pipeline (test collection)..."
uv run pytest tests/ --collect-only -q >/dev/null
echo "    OK"

if command -v bun >/dev/null 2>&1; then
  echo "==> Installing dashboard dependencies (bun install)..."
  (cd frontend && bun install)
else
  echo "==> bun not found — skipping dashboard install (optional; https://bun.sh)"
fi

echo
echo "Setup complete. Next steps:"
echo "  uv run python -m src.experiments run experiments/06_smoke_test.yaml   # pipeline smoke test (minutes)"
echo "  ./dev.sh                                                              # results dashboard"
echo "  REPORTED_FIGURES.md / METRICS.md                                      # verify the paper without running anything"
