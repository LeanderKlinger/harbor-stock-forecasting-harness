#!/usr/bin/env bash
# Full-stack quality gate: backend (lint + format-check + type-check + tests) + frontend (tsc + eslint).
# Run continuously during implementation AND before committing any feature
# (quality gate).
#
# Usage:
#   ./scripts/check.sh           — full gate: ruff + pyright + pytest + frontend (slow)
#   ./scripts/check.sh --quick   — fast tier: ruff + pyright + frontend (skip pytest)
#
# Frontend check runs in both modes — tsc + eslint are dirt cheap, no reason to skip.

set -euo pipefail

cd "$(dirname "$0")/.."

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
  QUICK=1
fi

echo "==> ruff check"
uv run ruff check .

echo "==> ruff format --check"
uv run ruff format --check .

echo "==> pyright"
uv run pyright src/ tests/

if [[ "$QUICK" -eq 0 ]]; then
  echo "==> pytest"
  uv run pytest tests/
fi

echo "==> frontend (tsc --noEmit + eslint)"
( cd frontend && bun run check )

echo "==> all clean"
