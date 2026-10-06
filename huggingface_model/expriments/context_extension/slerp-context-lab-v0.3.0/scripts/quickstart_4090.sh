#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bash run.sh verify
bash run.sh doctor --device cuda
bash run.sh download --config configs/hunyuan-native.json
bash run.sh parity --config configs/hunyuan-local.json --length 512 --device cuda --out reports/hunyuan-parity.json
for length in 2048 4096; do
  bash run.sh profile --config configs/hunyuan-slerp.json --length "$length" --steps 3 --device cuda --out "reports/profile-hunyuan-slerp-${length}.json"
done
bash scripts/screen_candidates.sh hunyuan
echo 'Read reports/screen/assessment.json, then run: bash scripts/learning_gate.sh'
