#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
cd "$SLERP_PACKAGE_ROOT"
[[ -x "$SLERP_PYTHON" ]] || { echo 'Run bash setup.sh first.' >&2; exit 1; }
if [[ "${1:-}" == "test" ]]; then
  shift
  exec "$SLERP_PYTHON" -m pytest -q "$@"
elif [[ "${1:-}" == "verify" ]]; then
  exec "$SLERP_PYTHON" scripts/verify_package.py
else
  exec "$SLERP_PYTHON" -m slerp_context.cli "$@"
fi

