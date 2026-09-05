#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

exec "$PYTHON_BIN" "$MESH_SCRIPT" \
  --mode inspect \
  --hardware-profile "$HARDWARE_PROFILE" \
  "$@"
