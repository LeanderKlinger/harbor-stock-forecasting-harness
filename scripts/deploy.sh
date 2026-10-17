#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/deploy.sh user@host yaml_path [--stage] [--kill-existing] [runner args...]
# Example: ./scripts/deploy.sh root@1.2.3.4 experiments/05_big_eval_grid.yaml --workers 60 --device cuda

if [ $# -lt 2 ]; then
  echo "Usage: ./scripts/deploy.sh user@host yaml_path [--stage] [--kill-existing] [runner args...]"
  echo "Example: ./scripts/deploy.sh root@1.2.3.4 experiments/05_big_eval_grid.yaml --workers 60 --device cuda"
  exit 1
fi

TARGET=$1; shift
YAML=$1; shift

STAGE=false
KILL_EXISTING=false
RUNNER_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --stage)         STAGE=true ;;
    --kill-existing) KILL_EXISTING=true ;;
    *)               RUNNER_ARGS+=("$arg") ;;
  esac
done

SESSION="harbor-$(date +%Y%m%d)"
REMOTE_DIR="~/harbor"
# tmux (and other Homebrew bins) live in /opt/homebrew/bin on Apple Silicon and
# /usr/local/bin on Intel — neither is on the non-login SSH PATH, so every remote
# tmux call must prepend them. Harmless on Linux/CUDA hosts (paths just don't exist).
REMOTE_PATH_FIX="export PATH=/opt/homebrew/bin:/usr/local/bin:\$PATH"

echo "=== Deploying to ${TARGET} ==="

# Step 1: Sync source code + lock file (fast, small)
echo "[$(date +%H:%M:%S)] [1/5] Syncing source code..."
rsync -avz --delete \
  --exclude='.venv' \
  --exclude='data/' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.claude' \
  --exclude='frontend/' \
  --exclude='notebooks/' \
  ./ "${TARGET}:${REMOTE_DIR}/"

# Step 2: Sync only data files needed for the experiment stocks
echo "[$(date +%H:%M:%S)] [2/5] Syncing data..."
STOCKS=$(uv run python -c "import yaml; print(' '.join(yaml.safe_load(open('${YAML}'))['base']['stock_ids']))")
echo "       Stocks: ${STOCKS}"

# Experiment logs land directly in data/experiment_log_archive/ so that
# fetch_results.sh picks them up with the same rsync that pulls run outputs —
# no post-hoc archiving. data/ is excluded from step 1's --delete, so past
# logs survive redeploys.
EXP_NAME=$(uv run python -c "import yaml; print(yaml.safe_load(open('${YAML}'))['name'])")
LOG_PATH="data/experiment_log_archive/$(date +%Y-%m-%d)_${EXP_NAME}_experiment.log"

ssh "${TARGET}" "mkdir -p ${REMOTE_DIR}/data/stock_data ${REMOTE_DIR}/data/articles ${REMOTE_DIR}/data/system/parsing_checkpoints ${REMOTE_DIR}/data/experiment_log_archive"

for STOCK in ${STOCKS}; do
  # Case-insensitive lookup — macOS is case-insensitive but Linux isn't
  HIST_FILE=$(find data/stock_data -maxdepth 1 -iname "${STOCK}.csv" | head -1)
  if [ -n "${HIST_FILE}" ]; then
    rsync -az "${HIST_FILE}" "${TARGET}:${REMOTE_DIR}/data/stock_data/"
  else
    echo "  WARNING: No CSV found for ${STOCK} in data/stock_data/"
  fi
  ART_FILE=$(find data/articles -maxdepth 1 -iname "${STOCK}.csv" | head -1)
  if [ -n "${ART_FILE}" ]; then
    rsync -az "${ART_FILE}" "${TARGET}:${REMOTE_DIR}/data/articles/"
  fi
done
rsync -avz data/system/parsing_checkpoints/ "${TARGET}:${REMOTE_DIR}/data/system/parsing_checkpoints/"

# Step 3: Create stub .env + install deps (under explicit bash, via stdin heredoc).
# Heredoc-to-stdin is robust against ssh's arg-concatenation (the old `bash -c '...'`
# pattern caused "bash: -c: option requires an argument" warnings on multi-line content
# because ssh joins args with spaces and the remote re-tokenizes).
echo "[$(date +%H:%M:%S)] [3/5] Setting up environment..."
# uv itself is once-only provisioning — installed by scripts/setup_remote.sh,
# not here. ~/.local/bin covers curl-installed uv (Linux hosts), the brew dirs
# cover brew-installed uv on a bootstrapped macOS host.
ssh "${TARGET}" bash <<EOF
set -euo pipefail
cd ${REMOTE_DIR}
printf 'ANTHROPIC_API_KEY=""\nOPENAI_API_KEY=""\n' > .env
export PATH="\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found on remote. Run ./scripts/setup_remote.sh ${TARGET} first."
  exit 1
fi
uv sync
EOF

# Step 4: tmux session management
echo "[$(date +%H:%M:%S)] [4/5] Checking for existing tmux session '${SESSION}'..."
if ssh "${TARGET}" "${REMOTE_PATH_FIX} && tmux has-session -t ${SESSION} 2>/dev/null"; then
  if [ "${KILL_EXISTING}" = true ]; then
    echo "  Killing existing session '${SESSION}'..."
    ssh "${TARGET}" "${REMOTE_PATH_FIX} && tmux kill-session -t ${SESSION}"
  else
    echo "ERROR: tmux session '${SESSION}' already exists on ${TARGET}."
    echo "  Reattach: ssh ${TARGET} -t tmux attach -t ${SESSION}"
    echo "  Kill it:  ssh ${TARGET} bash -c 'tmux kill-session -t ${SESSION}'"
    echo "  Or re-run with --kill-existing"
    exit 1
  fi
fi

# Step 5: stage or launch
# Shell-escape each runner arg so single-quoted bash -c on the remote can't be
# broken by quotes/spaces in args. printf '%q' produces strings safe to embed.
# Guard against empty array: bash `set -u` errors on ${ARR[@]} when ARR=().
QUOTED_ARGS=""
if [ "${#RUNNER_ARGS[@]}" -gt 0 ]; then
  QUOTED_ARGS=$(printf '%q ' "${RUNNER_ARGS[@]}")
fi

if [ "${STAGE}" = true ]; then
  echo "[$(date +%H:%M:%S)] [5/5] --stage set, skipping run launch."
  echo "  To launch manually:"
  echo "    ssh ${TARGET} -t tmux new-session -s ${SESSION}"
  echo "    cd ${REMOTE_DIR} && export PATH=\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
  echo "    PYTHONUNBUFFERED=1 uv run python -m src.experiments run ${YAML} ${QUOTED_ARGS}2>&1 | tee ${LOG_PATH}"
  echo "=== Stage complete ==="
  exit 0
fi

echo "[$(date +%H:%M:%S)] [5/5] Launching experiment in tmux session '${SESSION}'..."

# The experiment must run under bash inside the tmux pane (not the user's login shell).
# We pass `bash -c '<remote-cmd>'` as tmux's command so set -euo pipefail, export, and
# the pipe-into-tee all behave consistently regardless of the remote shell.
# Locally-interpolated values (REMOTE_DIR, YAML, QUOTED_ARGS) are substituted before SSH;
# $HOME and $PATH are escaped with \$ so they expand at run time on the remote.
# KEEPAWAKE resolves ON THE REMOTE at run time: `caffeinate -is` on macOS keeps
# the box from sleeping for the duration of the run; empty (no-op) elsewhere.
# Command substitution instead of a quoted variable because REMOTE_CMD is
# embedded in bash -c '...' — literal single quotes would terminate it.
KEEPAWAKE="\$(command -v caffeinate >/dev/null 2>&1 && echo caffeinate -is)"
# ulimit: macOS defaults to a 256 fd soft limit; cache-hit runs completing
# near-simultaneously at the queue tail exhaust it (EMFILE failures).
# Hard limit is unlimited, so raising is safe.
REMOTE_CMD="cd ${REMOTE_DIR} && export PATH=\$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH && ulimit -n 65536 && PYTHONUNBUFFERED=1 ${KEEPAWAKE} uv run python -m src.experiments run ${YAML} ${QUOTED_ARGS}2>&1 | tee ${LOG_PATH}"

ssh "${TARGET}" "${REMOTE_PATH_FIX} &&" tmux new-session -d -s "${SESSION}" "bash -c '${REMOTE_CMD}'"

echo ""
echo "  Attach:  ssh ${TARGET} -t tmux attach -t ${SESSION}"
echo "  Log:     ssh ${TARGET} 'tail -f ${REMOTE_DIR}/${LOG_PATH}'"
echo "  Results: ./scripts/fetch_results.sh ${TARGET}"
echo "=== Deploy complete ==="
