#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100-80gb}"
exec "$SCRIPT_DIR/run_branch_eval_matched.sh" "$@"
