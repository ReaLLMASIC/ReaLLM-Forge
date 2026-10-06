#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MESH_SCRIPT="$PROJECT_ROOT/recurrent_attnres_mesh.py"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Keep the established A100 configuration as the default. A trailing CLI
# --hardware-profile selection takes precedence over the environment so output
# names and profile-aware launcher defaults match the model's final argparse
# value. As in argparse, the last occurrence wins.
resolve_launcher_hardware_profile() {
  local selected_profile="${HARDWARE_PROFILE:-a100-80gb}"
  local expect_profile_value=0
  local argument profile_value

  for argument in "$@"; do
    if (( expect_profile_value )); then
      if [[ -z "$argument" || "$argument" == --* ]]; then
        echo "Missing value after --hardware-profile." >&2
        exit 2
      fi
      selected_profile="$argument"
      expect_profile_value=0
      continue
    fi
    case "$argument" in
      --hardware-profile)
        expect_profile_value=1
        ;;
      --hardware-profile=*)
        profile_value="${argument#--hardware-profile=}"
        if [[ -z "$profile_value" ]]; then
          echo "--hardware-profile= requires a non-empty value." >&2
          exit 2
        fi
        selected_profile="$profile_value"
        ;;
    esac
  done
  if (( expect_profile_value )); then
    echo "Missing value after --hardware-profile." >&2
    exit 2
  fi
  HARDWARE_PROFILE="$selected_profile"
}

resolve_launcher_hardware_profile "$@"
case "$HARDWARE_PROFILE" in
  a100-80gb|h100-sxm-80gb|h100-pcie-80gb|rtx4090-24gb|cuda-generic|portable)
    ;;
  *)
    echo "Unknown HARDWARE_PROFILE='$HARDWARE_PROFILE'." >&2
    echo "Choose: a100-80gb, h100-sxm-80gb, h100-pcie-80gb, rtx4090-24gb, cuda-generic, or portable." >&2
    exit 2
    ;;
esac

run_stamp() {
  date -u +%Y%m%dT%H%M%SZ
}

require_accelerate() {
  if ! "$PYTHON_BIN" -c 'import accelerate' >/dev/null 2>&1; then
    echo "Accelerate was not found. Run scripts/setup_env.sh first." >&2
    exit 1
  fi
}

require_single_gpu_process() {
  local world_size="${WORLD_SIZE:-1}"
  if [[ ! "$world_size" =~ ^[0-9]+$ ]] || [[ "$world_size" -ne 1 ]]; then
    echo "These launchers require WORLD_SIZE=1 (one process on one GPU)." >&2
    exit 1
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
  elif [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "CUDA_VISIBLE_DEVICES is empty; select exactly one GPU." >&2
    exit 1
  elif [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to exactly one GPU, not '$CUDA_VISIBLE_DEVICES'." >&2
    exit 1
  fi
}
