#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Explicit space-saving action. This removes resume state, not learned weights.
for method in local nlerp slerp; do
  bash run.sh finalize --run "runs/gate-$method-seed17"
done
