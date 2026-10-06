#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
config="${1:-configs/hunyuan-slerp.json}"
# Stops at the first OOM or <2 GiB headroom result. No disk offload is configured.
for length in 2048 4096 8192; do
  bash run.sh profile --config "$config" --length "$length" --steps 3 --device cuda --out "reports/profiles/$(basename "$config" .json)-$length.json"
done
