#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

"$PYTHON_BIN" -c \
  'import torch, transformers; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}"); print(f"transformers={transformers.__version__}")'
"$PYTHON_BIN" "$MESH_SCRIPT" --mode inspect --hardware-profile "$HARDWARE_PROFILE"
"$PYTHON_BIN" "$MESH_SCRIPT" --mode smoke --hardware-profile portable
