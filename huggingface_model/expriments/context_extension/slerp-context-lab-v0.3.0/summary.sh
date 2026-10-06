#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
cd "$SLERP_PACKAGE_ROOT"
[[ -x "$SLERP_PYTHON" ]] || { echo 'Run bash setup.sh first.' >&2; exit 1; }
exec "$SLERP_PYTHON" -m slerp_context.summary.cli "$@"
