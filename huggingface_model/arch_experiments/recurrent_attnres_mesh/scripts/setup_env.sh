#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

"$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Override TORCH_INDEX_URL/TORCH_VERSION for a different official wheel.
"$VENV_DIR/bin/python" -m pip install \
  --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION"
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"

"$VENV_DIR/bin/python" -c \
  'import accelerate, datasets, torch, transformers; print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}"); print(f"transformers={transformers.__version__} datasets={datasets.__version__} accelerate={accelerate.__version__}")'

echo "Environment ready: $VENV_DIR"
echo "Run: $PROJECT_ROOT/scripts/check_install.sh"
