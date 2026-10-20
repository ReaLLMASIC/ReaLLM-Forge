#!/usr/bin/env bash
# Run from any directory. This creates no Git repository and changes no parent one.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE=cpu
DEV=0
for arg in "$@"; do
  case "$arg" in
    --cpu) MODE=cpu ;;
    --cuda) MODE=cuda ;;
    --dev) DEV=1 ;;
    --help|-h)
      cat <<'HELP'
Usage: bash setup.sh [--cpu|--cuda] [--dev]
Default: create .venv and install CPU PyTorch plus the project.
CUDA: retain CUDA-enabled PyTorch already in .venv, or set TORCH_INDEX_URL
to an official CUDA wheel index selected for your driver at pytorch.org.
Optional: PYTHON=python3.12, VENV_DIR=/absolute/path/to/venv.
HELP
      exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
"${PYTHON:-python3}" -m venv "$VENV_DIR"
PIP_PY="$VENV_DIR/bin/python"
"$PIP_PY" -m pip install --upgrade pip
if [[ "$MODE" == cpu ]]; then
  "$PIP_PY" -m pip install --upgrade 'torch>=2.6,<3' \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
elif [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  "$PIP_PY" -m pip install --upgrade 'torch>=2.6,<3' --index-url "$TORCH_INDEX_URL"
elif ! "$PIP_PY" -c 'import torch; assert torch.version.cuda is not None' 2>/dev/null; then
  echo 'CUDA setup needs existing CUDA PyTorch or an explicit TORCH_INDEX_URL.' >&2
  echo 'Choose your CUDA build at https://pytorch.org/get-started/locally/ then rerun.' >&2
  exit 2
fi
"$PIP_PY" -m pip install -e "$PROJECT_DIR"
if [[ "$DEV" == 1 ]]; then
  "$PIP_PY" -m pip install -r "$PROJECT_DIR/requirements-dev.txt"
fi
"$PIP_PY" -c 'import torch, transformers, lane_lab; print("PyTorch", torch.__version__, "| Transformers", transformers.__version__, "| CUDA available:", torch.cuda.is_available())'
echo "Ready. Activate with: source \"$VENV_DIR/bin/activate\""
echo "Then: cd \"$PROJECT_DIR\""
echo '      python -m lane_lab serve --runs examples/checkpoints --port 8765'
