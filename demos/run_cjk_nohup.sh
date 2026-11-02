#!/usr/bin/env bash
# Launch a CJK pipeline in the background and record its terminal status.
set -Eeuo pipefail

SCRIPT_FILE="$(realpath "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "${SCRIPT_FILE}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash demos/run_cjk_nohup.sh [charbpe|char_byte_fallback|byte] [train|all|prepare|download|download-train]

Defaults: charbpe train (reuse already prepared data).
On a fresh machine, use all to download xinyixuu/ted-2020 and prepare the data.
Use charbpe download-train or char_byte_fallback download-train to fetch the
published tokenized data and skip preprocessing. download fetches data only.
JA_TEXT / KO_TEXT / ZH_CN_TEXT optionally override the downloaded files.
PYTHON_BIN, CJK_DATA_ROOT, CJK_OUT_ROOT, CJK_LOG_ROOT and training overrides
are passed to the pipeline. Default output/log roots have a _12l_3epochs suffix.

Files in CJK_LOG_ROOT:
  nohup.log     Combined stdout/stderr.
  pid           Background supervisor PID.
  status        RUNNING while active, 0 on success, 1 on failure.
  exit_code     Original pipeline exit code (written when finished).
  started_at / finished_at   UTC timestamps.

Each launch requires a log root with no previous .nohup-run directory.
EOF
}

write_value() {
  local path=$1 value=$2
  printf '%s\n' "${value}" >"${path}.tmp.$$"
  mv -f "${path}.tmp.$$" "${path}"
}

worker() {
  local pipeline=$1 mode=$2 log_root=$3 child_pid=""
  finish() {
    local code=$?
    trap - EXIT
    write_value "${log_root}/exit_code" "${code}"
    write_value "${log_root}/finished_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    if (( code == 0 )); then
      write_value "${log_root}/status" 0
    else
      write_value "${log_root}/status" 1
    fi
    exit "${code}"
  }
  stop() {
    local code=$1
    trap - TERM INT
    if [[ -n "${child_pid}" ]]; then
      kill -TERM -- "-${child_pid}" 2>/dev/null || true
      wait "${child_pid}" 2>/dev/null || true
    fi
    exit "${code}"
  }
  trap finish EXIT
  trap 'stop 143' TERM
  trap 'stop 130' INT
  write_value "${log_root}/pid" "$$"
  write_value "${log_root}/started_at" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  cd "${ROOT_DIR}"
  # A separate session lets TERM stop the pipeline and its training children.
  setsid bash "${pipeline}" "${mode}" &
  child_pid=$!
  local code=0
  wait "${child_pid}" || code=$?
  exit "${code}"
}

main() {
  if [[ "${1:-}" == --worker ]]; then
    worker "$2" "$3" "$4"
    return
  fi
  local comparison="${1:-charbpe}" mode="${2:-train}" suffix pipeline
  case "${comparison}" in
    -h|--help) usage; return ;;
    charbpe)
      suffix=cjk_ipa_charbpe
      pipeline="${ROOT_DIR}/demos/cjk_ipa_charbpe_compare.sh"
      export COMPARISON_TOKENIZER=char_bpe
      ;;
    char_byte_fallback|byte)
      suffix="cjk_ipa_${comparison}"
      pipeline="${ROOT_DIR}/demos/cjk_ipa_char_byte_fallback_compare.sh"
      export COMPARISON_TOKENIZER="${comparison}"
      ;;
    *) usage >&2; exit 2 ;;
  esac
  case "${mode}" in train|all|prepare|download|download-train) ;; *) usage >&2; exit 2 ;; esac
  command -v nohup >/dev/null
  command -v setsid >/dev/null
  export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
  export PYTHONUNBUFFERED=1
  export MPLBACKEND=Agg
  export CJK_DATA_ROOT="$(realpath -m "${CJK_DATA_ROOT:-${ROOT_DIR}/data/${suffix}}")"
  export CJK_OUT_ROOT="$(realpath -m "${CJK_OUT_ROOT:-${ROOT_DIR}/out/${suffix}_12l_3epochs}")"
  export CJK_LOG_ROOT="$(realpath -m "${CJK_LOG_ROOT:-${ROOT_DIR}/logs/${suffix}_12l_3epochs}")"
  mkdir -p "${CJK_LOG_ROOT}"
  if ! mkdir "${CJK_LOG_ROOT}/.nohup-run"; then
    printf 'This log root already has a launch. Choose a fresh CJK_LOG_ROOT: %s\n' "${CJK_LOG_ROOT}" >&2
    exit 1
  fi
  write_value "${CJK_LOG_ROOT}/status" RUNNING
  nohup bash "${SCRIPT_FILE}" --worker "${pipeline}" "${mode}" "${CJK_LOG_ROOT}" \
    >"${CJK_LOG_ROOT}/nohup.log" 2>&1 </dev/null &
  local pid=$!
  write_value "${CJK_LOG_ROOT}/pid" "${pid}"
  printf 'Started PID %s\nLog: %s/nohup.log\nStatus: %s/status\nOutput: %s\n' \
    "${pid}" "${CJK_LOG_ROOT}" "${CJK_LOG_ROOT}" "${CJK_OUT_ROOT}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
