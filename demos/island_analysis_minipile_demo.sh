#!/bin/bash
# Run island analysis on a minipile checkpoint and generate summary + plotly dashboard.

set -euo pipefail

SKIP_TRAINING="${1:-yes}"
OUT_DIR="out_minipile_island_demo"
ANALYSIS_DIR="${OUT_DIR}/island_analysis"
RUN_ROUTING_AUGMENT="${RUN_ROUTING_AUGMENT:-no}"
RUN_TRADEOFF_SEARCH="${RUN_TRADEOFF_SEARCH:-no}"

pushd data/minipile >/dev/null
if [[ ! -f "train.bin" ]] || [[ ! -f "val.bin" ]] || [[ ! -f "meta.pkl" ]]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
fi
popd >/dev/null

if [[ "${SKIP_TRAINING}" == "no" ]]; then
  rm -rf "${OUT_DIR}"
  python3 train.py \
    --dataset minipile \
    --out_dir "${OUT_DIR}" \
    --max_iters 1000 \
    --eval_interval 100 \
    --log_interval 10 \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256
fi

if [[ ! -f "${OUT_DIR}/ckpt.pt" ]]; then
  echo "Missing checkpoint: ${OUT_DIR}/ckpt.pt" >&2
  echo "Run with: bash demos/island_analysis_minipile_demo.sh no" >&2
  exit 1
fi

python3 analysis/checkpoint_analysis/analyze_dot_islands_ckpt.py \
  "${OUT_DIR}" \
  --pattern "transformer\\.(wte|h\\.[0-9]+\\.(attn|mlp))" \
  --metric cosine \
  --thresholds 0.2,0.3,0.4,0.5 \
  --min_island_size 4 \
  --top_providers 8 \
  --out_dir "${ANALYSIS_DIR}"

echo "Done. See:"
echo "  - ${ANALYSIS_DIR}/islands_detailed.json"
echo "  - ${ANALYSIS_DIR}/islands_summary.csv"
echo "  - ${ANALYSIS_DIR}/islands_dashboard.html"


if [[ "${RUN_ROUTING_AUGMENT}" == "yes" ]]; then
  ROUTING_DIR="${OUT_DIR}/island_routing"
  python3 analysis/checkpoint_analysis/augment_ckpt_with_island_routing.py \
    "${OUT_DIR}" \
    --island_json "${ANALYSIS_DIR}/islands_detailed.json" \
    --threshold 0.3 \
    --provider_mode top \
    --out_dir "${ROUTING_DIR}"

  echo "Routing augmentation + speed comparison artifacts:"
  echo "  - ${ROUTING_DIR}/island_routing.pt"
  echo "  - ${ROUTING_DIR}/island_routing_speed.csv"
  echo "  - ${ROUTING_DIR}/island_routing_speed.json"
  echo "  - ${ROUTING_DIR}/island_routing_speed.html"
fi


if [[ "${RUN_TRADEOFF_SEARCH}" == "yes" ]]; then
  SEARCH_DIR="${OUT_DIR}/island_tradeoff_search"
  python3 analysis/checkpoint_analysis/search_island_tradeoff.py \
    "${OUT_DIR}" \
    --island_json "${ANALYSIS_DIR}/islands_detailed.json" \
    --loss_tolerance_pct 2.0 \
    --eval_dataset minipile \
    --eval_iters 50 \
    --device cpu \
    --dtype float32 \
    --out_dir "${SEARCH_DIR}"

  echo "Tradeoff search artifacts:"
  echo "  - ${SEARCH_DIR}/search_log.yaml"
  echo "  - ${SEARCH_DIR}/search_results.json"
  echo "(Optional) TUI viewer: python3 analysis/checkpoint_analysis/view_island_tradeoff_log.py ${SEARCH_DIR}/search_log.yaml"
fi
