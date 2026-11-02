#!/usr/bin/env bash
# Foreground runner; use nohup externally to detach it.
set -Eeuo pipefail

finish() {
  local raw_status=$? status=0
  trap - EXIT
  if (( raw_status != 0 )); then
    status=1
  fi
  if [[ -n "${CJK_EXIT_CODE_FILE:-}" ]]; then
    if ! printf '%s\n' "${status}" >"${CJK_EXIT_CODE_FILE}"; then
      status=1
    fi
  fi
  printf 'CJK experiment finished: exit_code=%s (original=%s)\n' "${status}" "${raw_status}" >&2
  exit "${status}"
}
trap finish EXIT
trap 'exit 1' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-byte-fallback}"
MODE="${2:-all}"
if (( $# > 2 )); then
  printf 'Usage: bash demos/run_cjk_experiment.sh [byte-fallback|charbpe] [prepare|train|all|download|download-train|--help]\n' >&2
  exit 1
fi
case "${EXPERIMENT}" in
  byte-fallback) ENTRY_POINT="${SCRIPT_DIR}/cjk_ipa_char_byte_fallback_compare.sh" ;;
  charbpe) ENTRY_POINT="${SCRIPT_DIR}/cjk_ipa_charbpe_compare.sh" ;;
  *) printf 'Unknown experiment: %s\n' "${EXPERIMENT}" >&2; exit 1 ;;
esac

# The entry point downloads xinyixuu/ted-2020 unless local paths are supplied.

# A unique status path per job avoids stale results and concurrent writers.
if [[ -n "${CJK_EXIT_CODE_FILE:-}" ]]; then
  mkdir -p "$(dirname "${CJK_EXIT_CODE_FILE}")"
  printf 'running\n' >"${CJK_EXIT_CODE_FILE}"
fi
bash "${ENTRY_POINT}" "${MODE}"
