#!/usr/bin/env bash
# Keep generated datasets, checkpoints and reports outside this Git checkout.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-help}"
CJK_EVAL_ROOT="${CJK_EVAL_ROOT:-${ROOT_DIR}/../cjk-evaluation}"

if [[ "${MODE}" == help || "${MODE}" == --help ]]; then
  cat <<'EOF'
Usage: CJK_RUN_ROOT=/absolute/path/to/original-run bash demos/cjk_full_checkpoint_eval.sh MODE

Modes (run individually, in order):
  prepare-flores  Download pinned FLORES, convert IPA, audit OOV, encode frozen IDs.
  prepare-ted     Verify all 12 final checkpoints and the original TED train/val files.
  prepare-all     Verify TED and attach the prepared FLORES jobs to a new manifest.
  smoke           Evaluate up to 2049 source tokens per TED split on CPU; no geometry.
  run             Evaluate all ready jobs on CUDA, including static token geometry.
  report          Rebuild tables and figures from existing evaluation results.

Environment:
  CJK_RUN_ROOT          Original run containing final_resume/<run>/ckpt.pt (required).
  CJK_DATA_ROOT         Directory containing the original 12 TED tokenized datasets.
  CJK_EVAL_ROOT         Artifact directory OUTSIDE Git (default: sibling cjk-evaluation).
  CJK_TED_REVISION      Original TED Hub commit (required for prepare-ted/prepare-all).
  CJK_FLORES_REVISION   FLORES revision, resolved to an immutable SHA (default: main).
  CJK_FLORES_RECORDS    Optional offline JSONL with id, split and texts.{ja,ko,zh_cn}.
  CJK_EVAL_DEVICE       Full evaluation device (default: cuda:0).
  CJK_EVAL_DTYPE        Full evaluation dtype (default: bfloat16).
  CJK_EVAL_BATCH_SIZE   Full evaluation batch size (default: 4).
  CJK_EVAL_STRIDE       Window advance (default: 512; context is checkpoint block_size).

Use a fresh artifact directory for a new manifest or changed evaluation settings.
Completed jobs resume only with matching artifacts, implementation and settings.
EOF
  exit 0
fi

: "${CJK_RUN_ROOT:?Set CJK_RUN_ROOT to the original experiment run directory}"
CJK_DATA_ROOT="${CJK_DATA_ROOT:-${CJK_RUN_ROOT}/ReaLLM-Forge/data/cjk_ipa_charbpe}"
EVALUATOR="${ROOT_DIR}/benchmarks/evaluate_cjk_checkpoints.py"

case "${MODE}" in
  prepare-flores)
    offline_args=()
    [[ -z "${CJK_FLORES_RECORDS:-}" ]] || offline_args=(--records-jsonl "${CJK_FLORES_RECORDS}")
    "${PYTHON_BIN}" "${ROOT_DIR}/data/flores200-res/prepare_cjk_eval.py" \
      --data-root "${CJK_DATA_ROOT}" --output-dir "${CJK_EVAL_ROOT}/flores" \
      --revision "${CJK_FLORES_REVISION:-main}" "${offline_args[@]}"
    ;;
  prepare-ted|prepare-all)
    : "${CJK_TED_REVISION:?Set CJK_TED_REVISION to the original TED Hub commit}"
    flores_args=()
    if [[ "${MODE}" == prepare-all ]]; then
      [[ -f "${CJK_EVAL_ROOT}/flores/manifest.json" ]] || { echo 'Prepare FLORES first.' >&2; exit 1; }
      flores_args=(--flores-manifest "${CJK_EVAL_ROOT}/flores/manifest.json")
    fi
    "${PYTHON_BIN}" "${EVALUATOR}" prepare --run-root "${CJK_RUN_ROOT}" \
      --data-root "${CJK_DATA_ROOT}" --ted-revision "${CJK_TED_REVISION}" \
      "${flores_args[@]}" --output "${CJK_EVAL_ROOT}/manifest.json"
    ;;
  smoke)
    "${PYTHON_BIN}" "${EVALUATOR}" run --manifest "${CJK_EVAL_ROOT}/manifest.json" \
      --output "${CJK_EVAL_ROOT}/smoke" --device cpu --dtype float32 \
      --max-tokens 2049 --batch-size 2 --corpus ted --no-geometry
    ;;
  run)
    "${PYTHON_BIN}" "${EVALUATOR}" run --manifest "${CJK_EVAL_ROOT}/manifest.json" \
      --output "${CJK_EVAL_ROOT}/results" --device "${CJK_EVAL_DEVICE:-cuda:0}" \
      --dtype "${CJK_EVAL_DTYPE:-bfloat16}" --batch-size "${CJK_EVAL_BATCH_SIZE:-4}" \
      --stride "${CJK_EVAL_STRIDE:-512}"
    ;;
  report)
    "${PYTHON_BIN}" "${EVALUATOR}" report --output "${CJK_EVAL_ROOT}/results"
    ;;
  *) echo "Unknown mode: ${MODE}. Use --help." >&2; exit 2 ;;
esac
