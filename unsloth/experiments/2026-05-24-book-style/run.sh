#!/usr/bin/env bash
# Thin wrapper: sets PYTHONPATH for the torchvision stub, then runs lora_test.py
# via uv inside the parent project's venv.
#
# Usage:
#   ./run.sh prepare [--canon-db PATH]
#   ./run.sh train   [--epochs N] [--lr 1e-4]
#   ./run.sh eval    [--epoch best]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
export PYTHONPATH="$ROOT/stubs:${PYTHONPATH:-}"
exec uv run --project "$ROOT" python "$DIR/lora_test.py" "$@"
