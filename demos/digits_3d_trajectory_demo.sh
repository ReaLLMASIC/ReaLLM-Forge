#!/usr/bin/env bash
# Train a width-3 model and export every saved embedding snapshot for Three.js.
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
MAX_ITERS="${MAX_ITERS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
OUT_DIR="${OUT_DIR:-out/digits_3d}"
DATA_DIR="data/digits_3d"
VIEW_DIR="report/threejs/digits-3d"
WTE_FIXED_NORM="${WTE_FIXED_NORM:-true}"
WTE_FIXED_NORM_VALUE="${WTE_FIXED_NORM_VALUE:-}"
WTE_WEIGHT_TYING="${WTE_WEIGHT_TYING:-true}"
NUM_DIGITS="${NUM_DIGITS:-10}"
NUM_LETTERS="${NUM_LETTERS:-10}"
EMBEDDING_DIM="${EMBEDDING_DIM:-3}"
TRAJECTORY_FILE="${TRAJECTORY_FILE:-${VIEW_DIR}/token_trajectories.json}"
DROPOUT_PERCENT="${DROPOUT_PERCENT:-}"
DROPOUT_COUNT="${DROPOUT_COUNT:-1}"
OPTIMIZER_MODE="${OPTIMIZER_MODE:-adam}"
ADAM_WEIGHT_DECAY="${ADAM_WEIGHT_DECAY:-0.1}"

case "${WTE_FIXED_NORM}" in
  true|1|yes) WTE_NORM_ARGS=(--wte_fixed_norm) ;;
  false|0|no) WTE_NORM_ARGS=(--no-wte_fixed_norm) ;;
  *) echo "WTE_FIXED_NORM must be true or false" >&2; exit 2 ;;
esac
if [ -n "${WTE_FIXED_NORM_VALUE}" ]; then
  WTE_NORM_ARGS+=(--wte_fixed_norm_value "${WTE_FIXED_NORM_VALUE}")
fi
case "${WTE_WEIGHT_TYING}" in
  true|1|yes) WTE_TYING_ARGS=(--wte_weight_tying) ;;
  false|0|no) WTE_TYING_ARGS=(--no-wte_weight_tying) ;;
  *) echo "WTE_WEIGHT_TYING must be true or false" >&2; exit 2 ;;
esac
case "${OPTIMIZER_MODE}" in
  full_muon) OPTIMIZER_ARGS=(--optimizer muon --muon_include_all_weights --muon_min_ndim 2 --weight_decay 0) ;;
  adam) OPTIMIZER_ARGS=(--optimizer adam --weight_decay "${ADAM_WEIGHT_DECAY}") ;;
  adagrad|sgd|rmsprop) OPTIMIZER_ARGS=(--optimizer "${OPTIMIZER_MODE}" --weight_decay 0) ;;
  *) echo "OPTIMIZER_MODE must be full_muon, adam, adagrad, sgd, or rmsprop" >&2; exit 2 ;;
esac

AFFECTED_TOKENS="$(python3 -c 'import runpy,sys; s=runpy.run_path("data/digits_3d/prepare.py")["TRAINED_SYMBOLS"][:int(sys.argv[1])]; n=int(sys.argv[2]); print(s[-n:] if n else "")' "${NUM_DIGITS}" "${DROPOUT_COUNT}")"

TRAIN_ARGS=(--dataset digits_3d --out_dir "${OUT_DIR}" --device "${DEVICE}" --dtype "${DTYPE}"
  --block_size 10 --batch_size 64 --n_layer 1 --n_head 1 --n_embd "${EMBEDDING_DIM}"
  "${WTE_NORM_ARGS[@]}" "${WTE_TYING_ARGS[@]}" --dropout 0.0 --eval_interval "${SAVE_INTERVAL}"
  "${OPTIMIZER_ARGS[@]}"
  --eval_iters 20 --save_major_ckpt_interval "${SAVE_INTERVAL}" --always_save_checkpoint
  --only_save_checkpoint_at_end
  --learning_rate 3e-3 --min_lr 3e-4 --warmup_iters 20 --decay_lr --no-compile)

prepare_data() {
  local active="$1" extra=()
  if [ "${active}" = false ]; then extra=(--dropout-count "${DROPOUT_COUNT}"); fi
  python3 "${DATA_DIR}/prepare.py" --num-digits "${NUM_DIGITS}" --num-letters "${NUM_LETTERS}" "${extra[@]}"
}
FIRST_PHASE=true
train_until() {
  local iteration="$1" resume=()
  if [ "${FIRST_PHASE}" = false ]; then resume=(--init_from resume); fi
  python3 train.py "${TRAIN_ARGS[@]}" --max_iters "${iteration}" "${resume[@]}"
  FIRST_PHASE=false
}

SCHEDULE_MODE="${SCHEDULE_MODE:-${DROPOUT_PERCENT:+drop}}"
TRANSITION_EXPORT_ARGS=()
case "${SCHEDULE_MODE}" in
  "") prepare_data true; train_until "${MAX_ITERS}" ;;
  drop|add)
    if [ -z "${DROPOUT_PERCENT}" ] || [ "${DROPOUT_PERCENT}" -le 0 ] || [ "${DROPOUT_PERCENT}" -ge 100 ]; then
      echo "DROPOUT_PERCENT must be between 1 and 99 for ${SCHEDULE_MODE}" >&2; exit 2
    fi
    TRANSITION_ITERATION=$((MAX_ITERS * DROPOUT_PERCENT / 100))
    if [ "${SCHEDULE_MODE}" = drop ]; then prepare_data true; else prepare_data false; fi
    train_until "${TRANSITION_ITERATION}"
    if [ "${SCHEDULE_MODE}" = drop ]; then prepare_data false; else prepare_data true; fi
    train_until "${MAX_ITERS}"
    TRANSITION_EXPORT_ARGS=(--dropout-iteration "${TRANSITION_ITERATION}" --transition-mode "${SCHEDULE_MODE}" --transition-iterations "${TRANSITION_ITERATION}" --affected-tokens "${AFFECTED_TOKENS}")
    ;;
  duty_cycle)
    DUTY_CYCLE_PERCENT="${DUTY_CYCLE_PERCENT:-50}"; DUTY_PERIOD_PERCENT="${DUTY_PERIOD_PERCENT:-10}"
    if [ "${DUTY_CYCLE_PERCENT}" -lt 20 ] || [ "${DUTY_CYCLE_PERCENT}" -gt 80 ]; then echo "DUTY_CYCLE_PERCENT must be between 20 and 80" >&2; exit 2; fi
    PERIOD_ITERS=$((MAX_ITERS * DUTY_PERIOD_PERCENT / 100)); [ "${PERIOD_ITERS}" -gt 0 ] || { echo "Duty period is too short" >&2; exit 2; }
    TRANSITIONS=(); cycle_start=0
    while [ "${cycle_start}" -lt "${MAX_ITERS}" ]; do
      off_at=$((cycle_start + PERIOD_ITERS * DUTY_CYCLE_PERCENT / 100)); cycle_end=$((cycle_start + PERIOD_ITERS))
      [ "${off_at}" -gt "${MAX_ITERS}" ] && off_at="${MAX_ITERS}"; [ "${cycle_end}" -gt "${MAX_ITERS}" ] && cycle_end="${MAX_ITERS}"
      prepare_data true; train_until "${off_at}"
      if [ "${off_at}" -lt "${MAX_ITERS}" ]; then TRANSITIONS+=("${off_at}"); prepare_data false; train_until "${cycle_end}"; fi
      if [ "${cycle_end}" -lt "${MAX_ITERS}" ]; then TRANSITIONS+=("${cycle_end}"); fi
      cycle_start="${cycle_end}"
    done
    TRANSITION_EXPORT_ARGS=(--transition-mode duty_cycle --transition-iterations "${TRANSITIONS[@]}" --affected-tokens "${AFFECTED_TOKENS}" --duty-cycle "${DUTY_CYCLE_PERCENT}")
    ;;
  *) echo "SCHEDULE_MODE must be empty, drop, add, or duty_cycle" >&2; exit 2 ;;
esac

python3 analysis/export_3d_token_trajectories.py --checkpoint-dir "${OUT_DIR}" --meta "${DATA_DIR}/meta.pkl" --output "${TRAJECTORY_FILE}" "${TRANSITION_EXPORT_ARGS[@]}"

cat <<EOF
Done. Serve the repository (fetch does not work from file://), then open:
  python3 -m http.server 8000
  http://localhost:8000/${VIEW_DIR}/viewer.html?data=${TRAJECTORY_FILE#${VIEW_DIR}/}
The ${NUM_DIGITS} digit-like symbols are trained; ${NUM_LETTERS} letters are vocabulary-only controls.
Embedding dimension: ${EMBEDDING_DIM} (2D is shown in the XY plane; dimensions above 3 are globally PCA-projected for viewing).
WTE/LM-head weight tying: ${WTE_WEIGHT_TYING}.
Optimizer: ${OPTIMIZER_MODE}.
EOF
