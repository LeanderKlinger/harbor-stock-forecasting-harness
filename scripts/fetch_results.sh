#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/fetch_results.sh user@host

if [ $# -lt 1 ]; then
  echo "Usage: ./scripts/fetch_results.sh user@host"
  exit 1
fi

TARGET=$1
REMOTE_DIR="~/harbor"

echo "Fetching results from ${TARGET}..."
# Exclude the remote index: it only knows the remote's runs, so letting rsync
# copy it over the local one clobbers every locally-indexed run. The local
# index is rebuilt from disk below instead.
rsync -avz \
  --exclude='runs_index.json' \
  --exclude='runs_index.lock' \
  "${TARGET}:${REMOTE_DIR}/data/system/execution_output/" ./data/system/execution_output/

# Experiment logs: deploy.sh tees each run's log straight into this dir on the
# remote, so pulling it here is the whole archiving story — no moving around.
rsync -avz \
  "${TARGET}:${REMOTE_DIR}/data/experiment_log_archive/" ./data/experiment_log_archive/ 2>/dev/null \
  || echo "  (no experiment_log_archive on remote yet — skipped)"

# Fold the fetched runs into the local index (full rebuild from on-disk
# pipeline_output.json files). NOT safe concurrently with a local runner or
# the FastAPI server mid-write — reindex is not covered by runs_index.lock.
echo "Reindexing..."
uv run python -m src.experiments reindex
echo "Done. Runs index: data/system/execution_output/runs_index.json"
