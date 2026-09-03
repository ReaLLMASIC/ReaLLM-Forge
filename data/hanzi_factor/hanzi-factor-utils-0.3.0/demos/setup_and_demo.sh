#!/usr/bin/env bash
# One-command setup and end-to-end demonstration for hanzi-factor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK_DIR="${1:-${REPO_ROOT}/out/hanzi_factor_demo}"

if [[ "${WORK_DIR}" != /* ]]; then
  WORK_DIR="${REPO_ROOT}/${WORK_DIR}"
fi

VENV_DIR="${VENV_DIR:-${WORK_DIR}/.venv}"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
CCD_FILE="${DATA_DIR}/ccd.json"
SAMPLE_FILE="${REPO_ROOT}/demos/sample_chinese.txt"

step() {
  printf '\n==> %s\n' "$1"
}

step "Creating the isolated demo workspace"
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"

step "Installing hanzi-factor with phrase-aware normalization support"
"${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}[normalize]"

if [[ ! -s "${CCD_FILE}" ]]; then
  step "Fetching and verifying the pinned CCD decomposition catalogue"
  "${VENV_PYTHON}" "${REPO_ROOT}/scripts/fetch_ccd.py" "${CCD_FILE}"
else
  step "Reusing the existing verified CCD catalogue"
  printf '%s\n' "${CCD_FILE}"
fi

step "Normalizing the sample document to Simplified Chinese"
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/normalize_chinese.py" \
  "${SAMPLE_FILE}" \
  --to simplified \
  --variant generic \
  --output "${RESULTS_DIR}/sample.simplified.txt"

step "Normalizing the same document to Taiwan Traditional Chinese"
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/normalize_chinese.py" \
  "${SAMPLE_FILE}" \
  --to traditional \
  --variant taiwan-phrases \
  --output "${RESULTS_DIR}/sample.traditional-tw.txt"

step "Replacing Simplified Han characters with fully expanded prefix IDS"
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/text_to_ids.py" \
  "${RESULTS_DIR}/sample.simplified.txt" \
  --ccd "${CCD_FILE}" \
  --format expanded \
  --on-uncovered escape \
  --report "${RESULTS_DIR}/sample.simplified.ids.report.json" \
  --output "${RESULTS_DIR}/sample.simplified.ids.txt"

step "Replacing Traditional Han characters with fully expanded prefix IDS"
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/text_to_ids.py" \
  "${RESULTS_DIR}/sample.traditional-tw.txt" \
  --ccd "${CCD_FILE}" \
  --format expanded \
  --on-uncovered escape \
  --report "${RESULTS_DIR}/sample.traditional-tw.ids.report.json" \
  --output "${RESULTS_DIR}/sample.traditional-tw.ids.txt"

step "Restoring both IDS streams to ordinary text"
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/ids_to_text.py" \
  "${RESULTS_DIR}/sample.simplified.ids.txt" \
  --ccd "${CCD_FILE}" \
  --report "${RESULTS_DIR}/sample.simplified.restored.report.json" \
  --output "${RESULTS_DIR}/sample.simplified.restored.txt"
"${VENV_PYTHON}" "${REPO_ROOT}/scripts/ids_to_text.py" \
  "${RESULTS_DIR}/sample.traditional-tw.ids.txt" \
  --ccd "${CCD_FILE}" \
  --report "${RESULTS_DIR}/sample.traditional-tw.restored.report.json" \
  --output "${RESULTS_DIR}/sample.traditional-tw.restored.txt"

if ! cmp -s \
  "${RESULTS_DIR}/sample.simplified.txt" \
  "${RESULTS_DIR}/sample.simplified.restored.txt"; then
  printf 'ERROR: Simplified text -> IDS -> text was not exact.\n' >&2
  exit 1
fi
if ! cmp -s \
  "${RESULTS_DIR}/sample.traditional-tw.txt" \
  "${RESULTS_DIR}/sample.traditional-tw.restored.txt"; then
  printf 'ERROR: Traditional text -> IDS -> text was not exact.\n' >&2
  exit 1
fi
printf 'Both document round trips are byte-identical.\n'

step "Running the structural and binary round-trip example"
"${VENV_PYTHON}" "${REPO_ROOT}/examples/roundtrip.py" 汉 国 语 清 森 \
  > "${RESULTS_DIR}/roundtrip.txt"

step "Running the automated test suite"
(
  cd "${REPO_ROOT}"
  "${VENV_PYTHON}" -m unittest discover -s tests -q
)

step "Demo output preview"
printf '\n--- Simplified ---\n'
sed -n '1,12p' "${RESULTS_DIR}/sample.simplified.txt"
printf '\n--- Taiwan Traditional ---\n'
sed -n '1,12p' "${RESULTS_DIR}/sample.traditional-tw.txt"
printf '\n--- Expanded IDS (Simplified, first 4 lines) ---\n'
sed -n '1,4p' "${RESULTS_DIR}/sample.simplified.ids.txt"

step "Complete"
printf 'Results: %s\n' "${RESULTS_DIR}"
printf '%s\n' \
  "  sample.simplified.txt" \
  "  sample.traditional-tw.txt" \
  "  sample.simplified.ids.txt" \
  "  sample.traditional-tw.ids.txt" \
  "  sample.simplified.restored.txt" \
  "  sample.traditional-tw.restored.txt" \
  "  sample.simplified.ids.report.json" \
  "  sample.traditional-tw.ids.report.json" \
  "  sample.simplified.restored.report.json" \
  "  sample.traditional-tw.restored.report.json" \
  "  roundtrip.txt"
