#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
exec "${LAB_PYTHON:-$root/venv/bin/python}" -m hf_search.launch "$@"
