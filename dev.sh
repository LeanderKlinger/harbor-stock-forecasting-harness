#!/usr/bin/env bash
# Top-level dev launcher — delegates to frontend/dev.sh.
# Starts both the Vite dev server and the static data server.

set -euo pipefail

cd "$(dirname "$0")/frontend"
exec ./dev.sh "$@"
