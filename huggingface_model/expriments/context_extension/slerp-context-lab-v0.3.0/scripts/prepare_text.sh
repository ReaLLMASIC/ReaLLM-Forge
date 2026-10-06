#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
model="${1:-hunyuan}"
for split in train validation test; do
  tokens=25000000
  [[ "$split" == train ]] || tokens=2000000
  bash run.sh prepare-data --config "configs/$model-slerp.json" --out "data/pg19-$model" --split "$split" --tokens "$tokens"
done
