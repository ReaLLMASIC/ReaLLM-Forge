#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

exec "$PYTHON_BIN" "$MESH_SCRIPT" \
  --mode smoke \
  --hardware-profile portable \
  --no-strict-hardware-profile \
  --no-require-fused-sdpa \
  --sdpa-backend auto \
  --no-compile \
  "$@"
