#!/usr/bin/env bash
# Run every embedding-radius mode across several trained/held-out vocab sizes.
set -euo pipefail

DIGIT_COUNTS="${DIGIT_COUNTS:-10}"
LETTER_COUNTS="${LETTER_COUNTS:-10}"
EMBEDDING_DIMS="${EMBEDDING_DIMS:-2 3}"
WTE_TYING_MODES="${WTE_TYING_MODES:-tied untied}"
OPTIMIZER_MODES="${OPTIMIZER_MODES:-full_muon adam adagrad sgd rmsprop}"
ADAM_WEIGHT_DECAYS="${ADAM_WEIGHT_DECAYS:-0.0 0.01 0.05 0.1 0.5}"
RADIUS_MODES="${RADIUS_MODES:-free sqrt_dim 1}"
TRANSITION_PERCENTAGES="${TRANSITION_PERCENTAGES:-20 40 60 80}"
DUTY_CYCLES="${DUTY_CYCLES:-20 40 60 80}"
DUTY_PERIOD_PERCENT="${DUTY_PERIOD_PERCENT:-10}"
DROPOUT_COUNTS="${DROPOUT_COUNTS:-1}"
SWEEP_MAX_ITERS="${SWEEP_MAX_ITERS:-30000}"
SWEEP_SAVE_INTERVAL="${SWEEP_SAVE_INTERVAL:-100}"
RUNS_DIR="report/threejs/digits-3d/runs"

mkdir -p "${RUNS_DIR}"

SCHEDULES=()
for percent in ${TRANSITION_PERCENTAGES}; do
  SCHEDULES+=("drop:${percent}" "add:${percent}")
done
if ! [[ "${DUTY_PERIOD_PERCENT}" =~ ^[0-9]+$ ]] || [ "${DUTY_PERIOD_PERCENT}" -gt 100 ]; then
  echo "DUTY_PERIOD_PERCENT must be an integer from 0 to 100 (0 disables duty-cycle runs)" >&2
  exit 2
fi
if [ "${DUTY_PERIOD_PERCENT}" -gt 0 ]; then
  for duty in ${DUTY_CYCLES}; do SCHEDULES+=("duty_cycle:${duty}"); done
else
  echo "DUTY_PERIOD_PERCENT=0; skipping duty-cycle runs"
fi

for embedding_dim in ${EMBEDDING_DIMS}; do
  for num_digits in ${DIGIT_COUNTS}; do
    for num_letters in ${LETTER_COUNTS}; do
      for radius_mode in ${RADIUS_MODES}; do
        for tying_mode in ${WTE_TYING_MODES}; do
          for optimizer_mode in ${OPTIMIZER_MODES}; do
            weight_decays="0"
            [ "${optimizer_mode}" = adam ] && weight_decays="${ADAM_WEIGHT_DECAYS}"
            for weight_decay in ${weight_decays}; do
            for dropout_count in ${DROPOUT_COUNTS}; do
            for schedule in "${SCHEDULES[@]}"; do
              schedule_mode="${schedule%%:*}"; schedule_value="${schedule#*:}"
              case "${radius_mode}" in
                free) fixed=false; radius=""; radius_name=free ;;
                sqrt_dim) fixed=true; radius=""; radius_name=sqrt-dim ;;
                *)
                  if ! [[ "${radius_mode}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || ! awk -v radius="${radius_mode}" 'BEGIN { exit !(radius > 0) }'; then
                    echo "RADIUS_MODES values must be free, sqrt_dim, or positive numbers" >&2; exit 2
                  fi
                  fixed=true; radius="${radius_mode}"; radius_name="${radius_mode}"
                  ;;
              esac
              case "${tying_mode}" in
                tied) weight_tying=true ;;
                untied) weight_tying=false ;;
                *) echo "Unknown WTE tying mode: ${tying_mode}" >&2; exit 2 ;;
              esac
              if [ "${schedule_mode}" = duty_cycle ]; then
                schedule_name="duty-${schedule_value}pct"
                schedule_args=(SCHEDULE_MODE=duty_cycle DUTY_CYCLE_PERCENT="${schedule_value}" DUTY_PERIOD_PERCENT="${DUTY_PERIOD_PERCENT}")
              else
                schedule_name="${schedule_mode}-at-${schedule_value}pct"
                schedule_args=(SCHEDULE_MODE="${schedule_mode}" DROPOUT_PERCENT="${schedule_value}")
              fi
              name="dim-${embedding_dim}_digits-${num_digits}_letters-${num_letters}_radius-${radius_name}_${tying_mode}_${optimizer_mode}_wd-${weight_decay}_drop-${dropout_count}_${schedule_name}"
              echo "=== ${name} ==="
              env NUM_DIGITS="${num_digits}" NUM_LETTERS="${num_letters}" EMBEDDING_DIM="${embedding_dim}" \
                WTE_FIXED_NORM="${fixed}" WTE_FIXED_NORM_VALUE="${radius}" WTE_WEIGHT_TYING="${weight_tying}" \
                OPTIMIZER_MODE="${optimizer_mode}" ADAM_WEIGHT_DECAY="${weight_decay}" \
                DROPOUT_COUNT="${dropout_count}" MAX_ITERS="${SWEEP_MAX_ITERS}" SAVE_INTERVAL="${SWEEP_SAVE_INTERVAL}" \
                OUT_DIR="out/digits_3d_sweep/${name}" TRAJECTORY_FILE="${RUNS_DIR}/${name}.json" \
                "${schedule_args[@]}" bash demos/digits_3d_trajectory_demo.sh
              python3 analysis/update_3d_sweep_manifest.py --runs-dir "${RUNS_DIR}"
            done
          done
          done
        done
      done
    done
  done
done
done

cat <<EOF
Sweep complete. Serve the repository with:
  python3 -m http.server 8000

Sweep selector:
  http://localhost:8000/report/threejs/digits-3d/index.html
EOF
