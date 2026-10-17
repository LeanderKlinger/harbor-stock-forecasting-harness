#!/usr/bin/env bash
set -euo pipefail
# Usage: ./scripts/setup_remote.sh <user@host | ssh-alias>
# Example: ./scripts/setup_remote.sh my-remote-host
#
# One-time, IDEMPOTENT bootstrap of a remote host so deploy.sh works and the
# interactive SSH shell is usable. Safe to re-run. Fixes the two recurring pains:
#
#   1. PATH in non-login SSH shells. `ssh host 'tmux ...'` runs a NON-LOGIN,
#      NON-INTERACTIVE zsh, which sources ONLY ~/.zshenv (never ~/.zprofile).
#      Homebrew's installer writes its PATH line to ~/.zprofile, so bare ssh
#      commands can't find brew-installed binaries (tmux, etc.). We add the
#      brew shellenv to ~/.zshenv so EVERY shell invocation sees it.
#
#   2. terminfo for the local terminal. Modern terminals (Ghostty ->
#      TERM=xterm-ghostty) ship their own terminfo. If the remote's terminfo
#      DB lacks that entry, the line editor and tmux can't drive the terminal:
#      backspace inserts spaces, arrow keys emit garbage, `tmux attach` errors
#      with "missing or unsuitable terminal". We compile the local terminfo
#      entry onto the remote (~/.terminfo, no root needed).

TARGET="${1:?usage: ./scripts/setup_remote.sh <user@host | ssh-alias> [term-name]}"

echo "=== setup_remote.sh -> ${TARGET} ==="

# --- Step 1: push the local terminal's terminfo entry to the remote ----------
# Runs from LOCAL (where the terminfo entry exists) and compiles it remotely.
# tic lives in /usr/bin (on the bare non-login PATH), so no PATH fix needed here.
# Optional 2nd arg overrides $TERM — for shells not running under the target
# terminal (automation, tmux with TERM=screen-256color, etc.).
LOCAL_TERM="${2:-${TERM:-}}"
if [[ -n "${LOCAL_TERM}" ]] && infocmp -x "${LOCAL_TERM}" >/dev/null 2>&1; then
  echo "[1/2] Installing terminfo '${LOCAL_TERM}' on remote..."
  if infocmp -x "${LOCAL_TERM}" | ssh "${TARGET}" -- tic -x -; then
    echo "      OK: '${LOCAL_TERM}' compiled into remote ~/.terminfo"
  else
    echo "      WARN: remote tic failed for '${LOCAL_TERM}'. Fallback: connect with"
    echo "            TERM=xterm-256color ssh ${TARGET}   (or add 'SetEnv TERM=xterm-256color' to ~/.ssh/config)"
  fi
else
  echo "[1/2] Skipping terminfo (local TERM='${LOCAL_TERM}' has no infocmp entry)"
fi

# --- Step 2: remote bootstrap (brew + packages + PATH propagation) -----------
# Heredoc-to-stdin (not `bash -c '...'`) so multi-line content survives ssh's
# arg re-tokenization. Same pattern deploy.sh uses for its env setup.
echo "[2/2] Bootstrapping remote (Homebrew + packages + shell env)..."
ssh "${TARGET}" bash <<'REMOTE'
set -euo pipefail

# --- Homebrew (install if absent) ---
if ! command -v brew >/dev/null 2>&1 \
   && [[ ! -x /opt/homebrew/bin/brew ]] && [[ ! -x /usr/local/bin/brew ]]; then
  echo "      Installing Homebrew (non-interactive)..."
  NONINTERACTIVE=1 /bin/bash -c \
    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Resolve brew prefix (Apple Silicon vs Intel) and load its env for THIS shell.
if   [[ -x /opt/homebrew/bin/brew ]]; then BREW=/opt/homebrew/bin/brew
elif [[ -x /usr/local/bin/brew ]];   then BREW=/usr/local/bin/brew
else BREW="$(command -v brew)"; fi
eval "$("${BREW}" shellenv)"

# --- Packages deploy.sh needs on the remote (extend this list as needed) ---
# uv lives here too (once-only provisioning; deploy.sh only runs `uv sync`).
# Brew-installed uv rides the same shellenv PATH line as tmux.
PACKAGES=(tmux uv)
for pkg in "${PACKAGES[@]}"; do
  if brew list --formula "${pkg}" >/dev/null 2>&1; then
    echo "      ${pkg}: present"
  else
    echo "      ${pkg}: installing..."
    brew install "${pkg}"
  fi
done

# --- PATH propagation to non-login/non-interactive shells ---
# ~/.zshenv is the ONLY startup file sourced by `ssh host 'cmd'`. Put brew there
# so deploy.sh's bare `ssh host 'tmux ...'` resolves. Also keep it in ~/.zprofile
# so login shells re-prepend after macOS /etc/zprofile path_helper reorders PATH.
SHELLENV_LINE="eval \"\$(${BREW} shellenv)\""
MARKER="# harbor: brew PATH (setup_remote.sh)"
for rc in "${HOME}/.zshenv" "${HOME}/.zprofile"; do
  touch "${rc}"
  if grep -qF "${MARKER}" "${rc}"; then
    echo "      $(basename "${rc}"): brew env already present"
  else
    printf '\n%s\n%s\n' "${MARKER}" "${SHELLENV_LINE}" >> "${rc}"
    echo "      $(basename "${rc}"): brew env added"
  fi
done

echo "      Remote ready: $(brew --version | head -1); $(tmux -V)"
REMOTE

echo ""
echo "Verify:  ssh ${TARGET} 'tmux -V'          # should print a version, not 'command not found'"
echo "         ssh ${TARGET} -t tmux ls         # should list sessions with a usable terminal"
echo "=== setup_remote.sh complete ==="
