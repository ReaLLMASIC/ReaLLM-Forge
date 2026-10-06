#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ $# -ge 2 ]] || { echo 'Usage: bash scripts/archive_run.sh RUN_DIR EXTERNAL_DESTINATION [MB_PER_SECOND]' >&2; exit 1; }
# Deliberately manual. No background sync, no data/model-cache archive, no local deletion.
bash run.sh archive --run "$1" --destination "$2" --mbps "${3:-20}"

