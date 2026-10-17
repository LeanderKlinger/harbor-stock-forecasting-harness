#!/usr/bin/env bash
# Start the FastAPI server (port 3001) and the Vite dev server (port 5173).
# Frontend hits FastAPI directly via CORS — no proxy.

set -euo pipefail

cleanup() {
  echo "Shutting down..."
  kill "$API_PID" "$VITE_PID" 2>/dev/null || true
  wait "$API_PID" "$VITE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI on :3001"
(cd .. && uv run uvicorn src.server:app --port 3001 --reload --reload-dir src) &
API_PID=$!

echo "Starting Vite dev server on :5173"
bun run dev &
VITE_PID=$!

wait
