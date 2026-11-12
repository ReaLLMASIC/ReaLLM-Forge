#!/usr/bin/env bash
# Sets up the live H10 forecasts on a Jetson Orin: PyTorch for the Jetson's GPU, transformers, the model
# folders, then a timing check. Run it from the release folder, as yourself (no sudo):
#
#   ./tools/orin_setup.sh --models ~/Downloads/H10_hub_upload.zip   (or any zip / folder holding ecg/ (or ecg_only/), ecg_acc/, ecg_rr/)
#   ./tools/orin_setup.sh                                         (models already in polar_dashboard/forecast/models)
#
# What it does:
#   1. Checks this is a Jetson and reads its JetPack (L4T) and CUDA versions.
#   2. Makes a Python environment just for the forecasts, shared by every release folder (torch is ~1 GB):
#      ~/.local/share/bio-dash/forecast-venv (BIODASH_FORECAST_VENV or --venv picks another). The
#      dashboards use it by themselves when it's there; BIODASH_FORECAST_PYTHON still overrides.
#   3. Installs PyTorch built for the Jetson from NVIDIA's Jetson AI Lab index (pypi.jetson-ai-lab.io,
#      jp6/cu126 for JetPack 6), never the PyPI build (that one doesn't use the Orin's GPU), then
#      transformers and safetensors from PyPI.
#   4. Copies the model folders into polar_dashboard/forecast/models (--models).
#   5. Starts the forecasts at 4 samples instead of 8 (the Orin is slower than the A100 the 8 was timed
#      on), unless forecast settings were saved already; the page's settings change it later.
#   6. Times both models with forecast_service.py --bench at 4 and 8 samples.
#
# Options: --models <zip|folder>   --venv <dir>   --torch <version>   --index <url>   --no-bench
#          --replace      take the whole model folders from --models (default: only the weights, and only
#                         when they're the model this release's config.json describes)
#          --own-torch    install PyTorch into the environment even when the system already has a working one
#          --not-jetson   (testing this script elsewhere: PyPI's CPU torch, bench on the CPU)
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
REL="$(pwd)"
VENV="${BIODASH_FORECAST_VENV:-$HOME/.local/share/bio-dash/forecast-venv}"
MODELS="" TORCH_VER="" INDEX="" BENCH=1 JETSON=1 REPLACE=0 OWN_TORCH=0
while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --torch) TORCH_VER="$2"; shift 2 ;;
        --index) INDEX="$2"; shift 2 ;;
        --no-bench) BENCH=0; shift ;;
        --replace) REPLACE=1; shift ;;
        --own-torch) OWN_TORCH=1; shift ;;
        --not-jetson) JETSON=0; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
done
say() { echo "▶ $*"; }
die() { echo "✖ $*" >&2; exit 1; }
[ -f polar_dashboard/forecast_service.py ] || die "run this from a release folder with polar_dashboard/forecast_service.py (V5.5 or V6)"

# ---------------------------------------------------------------- 1. which Jetson software
if [ "$JETSON" = 1 ]; then
    [ "$(uname -m)" = aarch64 ] && [ -f /etc/nv_tegra_release ] || \
        die "this doesn't look like a Jetson (no /etc/nv_tegra_release). Elsewhere: install torch from pytorch.org; --not-jetson tests this script with a CPU torch."
    L4T=$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release)
    L4T_MAJOR=${L4T%%.*}
    CUDA=""
    if [ -f /usr/local/cuda/version.json ]; then
        CUDA=$(python3 -c 'import json; print(json.load(open("/usr/local/cuda/version.json"))["cuda"]["version"])' 2>/dev/null || true)
    fi
    [ -n "$CUDA" ] || CUDA=$( (nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
    say "Jetson: L4T R${L4T:-?}, CUDA ${CUDA:-not found}"
    if [ -z "$INDEX" ]; then
        case "$L4T_MAJOR" in
            36) case "$CUDA" in
                    12.8*) CU=cu128 ;;
                    12.9*) CU=cu129 ;;
                    12.6*) CU=cu126 ;;
                    *) CU=cu126; echo "  (CUDA ${CUDA:-?}: JetPack 6 ships 12.6, using its wheels)" ;;
                esac
                INDEX="https://pypi.jetson-ai-lab.io/jp6/$CU/+simple/" ;;
            35) die "JetPack 5 (L4T R35): the Jetson AI Lab index only builds for JetPack 6. Flash JetPack 6 (recommended), or install NVIDIA's JetPack 5 torch wheel by hand (NVIDIA's \"Installing PyTorch for Jetson Platform\" guide) and rerun with --index <url>." ;;
            *) die "L4T R${L4T:-?} isn't one this script knows. Find the right PyTorch at https://pypi.jetson-ai-lab.io/ and rerun with --index <its +simple/ url>." ;;
        esac
    fi
fi

# ---------------------------------------------------------------- 2. the forecast environment
PY=python3
if [ "$JETSON" = 1 ]; then
    command -v python3.10 >/dev/null && PY=python3.10    # the Jetson wheels are built for Python 3.10 (JetPack 6)
    "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' || \
        die "the Jetson PyTorch wheels need Python 3.10 (JetPack 6's own). Found $("$PY" --version)."
fi
"$PY" -c 'import venv, ensurepip' 2>/dev/null || die "Python's venv module is missing: sudo apt install python3-venv"
# A working GPU PyTorch already installed for this Python (e.g. by ReaLLM-Forge's
# orin_dev_setup_scripts/00-setup-jetson-pytorch.sh, with cuDSS / cuSPARSELt system-wide) is used as it is:
# the environment sees it (--system-site-packages) instead of installing its own. --own-torch / --torch skip this.
SYSTORCH=0
if [ "$JETSON" = 1 ] && [ "$OWN_TORCH" = 0 ] && [ -z "$TORCH_VER" ] && \
   "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    SYSTORCH=1
    say "Using the PyTorch already installed for $PY: $("$PY" -c 'import torch; print(torch.__version__)')"
fi
if [ ! -x "$VENV/bin/python" ]; then
    say "Making $VENV"
    mkdir -p "$(dirname "$VENV")"
    if [ "$SYSTORCH" = 1 ]; then "$PY" -m venv --system-site-packages "$VENV"; else "$PY" -m venv "$VENV"; fi
elif [ "$SYSTORCH" = 1 ]; then           # made earlier with its own torch: switch it to the system one
    sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' "$VENV/pyvenv.cfg"
    "$VENV/bin/python" -m pip uninstall -y -q torch >/dev/null 2>&1 || true
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install -q --upgrade pip

# ---------------------------------------------------------------- 3. PyTorch, transformers
TORCH_SPEC="torch${TORCH_VER:+==$TORCH_VER}"
if [ "$SYSTORCH" = 1 ]; then
    :                                    # the system's PyTorch (above)
elif [ "$JETSON" = 1 ]; then
    # torch's own requirements from PyPI first, then torch ONLY from the Jetson index (--no-deps): with both
    # indexes at once pip would take whichever torch has the higher version, usually PyPI's, which can't
    # use the Orin's GPU.
    say "PyTorch for the Jetson from $INDEX"
    "$VPY" -m pip install -q filelock typing-extensions sympy networkx jinja2 fsspec "numpy<3"
    "$VPY" -m pip install --no-deps --index-url "$INDEX" "$TORCH_SPEC"
else
    say "PyTorch (CPU) from PyPI (--not-jetson)"
    "$VPY" -m pip install -q "$TORCH_SPEC" --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
        "$VPY" -m pip install -q "$TORCH_SPEC"
fi
say "transformers 4.x, safetensors, Pillow"
# transformers 4.x, as the models were tested (4.53.3). 5.x needs PyTorch 2.6+ and isn't what they were tested
# with. Pillow 9.1+ in the environment itself: with the system's PyTorch the environment also sees the system's
# packages, and JetPack 6's (Ubuntu 22.04's) Pillow 9.0 lacks Image.Resampling, which transformers imports, so
# every model load failed ("Could not import module 'PreTrainedModel'"). An environment made earlier with 5.x
# is moved back to 4.x by this line.
"$VPY" -m pip install -q "transformers>=4.40,<5" safetensors "pillow>=9.1"
say "Checking transformers"
"$VPY" -c 'import transformers; from transformers.modeling_utils import PreTrainedModel; print("transformers", transformers.__version__, "OK")' || \
    die "transformers doesn't import (above). Paste the error into an issue; the forecasts can't load a model until it does."

say "Checking PyTorch"
torch_check() {
    "$VPY" - <<'EOF' 2>&1
import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
x = torch.randn(64, 64, device="cuda" if torch.cuda.is_available() else "cpu")
with torch.autocast(x.device.type, dtype=torch.bfloat16):
    (x @ x).sum().item()
print("bf16 matmul OK")
EOF
}
# The Jetson AI Lab torch builds (2.8+) link NVIDIA's cuDSS and cuSPARSELt, which JetPack 6 doesn't ship.
# Fetch NVIDIA's aarch64 (Jetson) builds of the one that's missing into the forecast environment and load
# them at Python start-up (a .pth file): no sudo, nothing changed outside $VENV.
NV=https://developer.download.nvidia.com/compute
NVLIB_URLS="libcudss=$NV/cudss/redist/libcudss/linux-aarch64/libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz
libcusparseLt=$NV/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz"
fetch_nvidia_lib() {     # $1: libcudss | libcusparseLt
    url=$(printf '%s\n' "$NVLIB_URLS" | sed -n "s/^$1=//p")
    dest="$VENV/lib/nvidia-jetson"; tmp=$(mktemp -d)
    say "Fetching NVIDIA's $1 for the Jetson ($(basename "$url"))"
    "$VPY" - "$url" "$tmp/a.tar.xz" <<'EOF' || die "couldn't download $url"
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
EOF
    mkdir -p "$dest"
    tar -xJf "$tmp/a.tar.xz" -C "$tmp"
    find "$tmp" -path "*/lib/$1*.so*" -exec cp -P {} "$dest/" \;
    rm -rf "$tmp"
    ls "$dest/$1.so.0" >/dev/null 2>&1 || die "$1.so.0 wasn't in $url"
    site=$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    cat > "$site/biodash_nvidia_libs.py" <<'EOF'
"""Loads the NVIDIA libraries tools/orin_setup.sh put in <venv>/lib/nvidia-jetson (cuDSS, cuSPARSELt), so the
Jetson torch build finds them without LD_LIBRARY_PATH. Run at start-up by biodash_nvidia_libs.pth."""
import ctypes, os, sys
for _name in ("libcusparseLt.so.0", "libcudss.so.0"):
    _p = os.path.join(sys.prefix, "lib", "nvidia-jetson", _name)
    if os.path.exists(_p):
        try:
            ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        except OSError as _e:
            print(f"biodash_nvidia_libs: couldn't load {_p}: {_e}", file=sys.stderr)
EOF
    echo "import biodash_nvidia_libs" > "$site/biodash_nvidia_libs.pth"
}
tries=0
until OUT=$(torch_check); do
    missing=$(echo "$OUT" | grep -o "libcudss\|libcusparseLt" | head -1 || true)
    tries=$((tries + 1))
    if [ "$JETSON" = 1 ] && [ -n "$missing" ] && [ "$tries" -le 2 ]; then
        fetch_nvidia_lib "$missing"
        continue
    fi
    echo "$OUT" | tail -5
    if echo "$OUT" | grep -q "libcusparseLt\|libcudss\|libcudnn"; then
        echo "  PyTorch needs an NVIDIA library that isn't installed (named above). Install it for"
        echo "  aarch64-jetson (NVIDIA's \"Installing PyTorch for Jetson Platform\" guide lists them), then rerun."
    fi
    die "PyTorch doesn't work in $VENV"
done
echo "$OUT" | sed 's/^/  /'
if [ "$JETSON" = 1 ] && ! echo "$OUT" | grep -q "CUDA available: True"; then
    die "PyTorch installed but doesn't see the GPU: check that $INDEX matches this JetPack (--index to pick another)"
fi

# ---------------------------------------------------------------- 4. the models
DEST="$REL/polar_dashboard/forecast/models"
if [ -n "$MODELS" ]; then
    SRC="$MODELS" TMP=""
    if [ -f "$MODELS" ]; then
        TMP=$(mktemp -d); "$VPY" -m zipfile -e "$MODELS" "$TMP"; SRC="$TMP"
    fi
    [ -d "$SRC" ] || die "--models: $MODELS isn't a zip or a folder"
    for m in ecg ecg_acc ecg_rr; do
        # the folder named ecg / ecg_acc / ecg_rr that has a config.json (anywhere inside, e.g. out/hf_models/best/ecg)
        found=$(find "$SRC" -type f -name config.json -path "*/$m/config.json" | head -1)
        # the Hugging Face upload (H10_hub_upload*.zip) calls the ECG-only model ecg_only/
        [ -z "$found" ] && [ "$m" = ecg ] && found=$(find "$SRC" -type f -name config.json -path "*/ecg_only/config.json" | head -1)
        if [ -z "$found" ]; then echo "  (no $m/ model in $MODELS)"; continue; fi
        src=$(dirname "$found")
        if [ -f "$DEST/$m/config.json" ] && [ "$REPLACE" = 0 ]; then
            # The release has this model's settings already (they're in git): bring only the weights, and only
            # when they're the same model (the release's config.json has to describe them).
            if ! "$VPY" -c '
import json, os, sys
a, b = (json.load(open(os.path.join(d, "config.json"))) for d in sys.argv[1:3])
keep = lambda c: {k: v for k, v in c.items() if k not in ("transformers_version", "torch_dtype", "dtype")}
sys.exit(keep(a) != keep(b))' "$src" "$DEST/$m"; then
                echo "  ✖ $m in $MODELS isn't the model this release is set up for: skipped (--replace takes it anyway)"
                continue
            fi
            for w in pytorch_model.bin model.safetensors; do
                if [ -f "$src/$w" ]; then cp "$src/$w" "$DEST/$m/"; fi
            done
            say "model $m: weights -> $DEST/$m"
        else
            mkdir -p "$DEST/$m"
            for f in "$src"/*; do
                case "$(basename "$f")" in
                    pytorch_model.bin|model.safetensors|config.json|columns.json|RUN.txt|training_state.json|*.py) cp "$f" "$DEST/$m/" ;;
                esac
            done
            say "model $m -> $DEST/$m"
        fi
    done
    [ -n "$TMP" ] && rm -rf "$TMP"
fi
for m in ecg ecg_acc ecg_rr; do
    if [ -f "$DEST/$m/pytorch_model.bin" ] || [ -f "$DEST/$m/model.safetensors" ]; then echo "  $m: weights OK"
    else echo "  $m: no weights yet (copy the model folder in, or rerun with --models)"; fi
done

# ---------------------------------------------------------------- 5. start at 4 samples on the Orin
SETTINGS="$REL/polar_dashboard/forecast_settings.json"
if [ "$JETSON" = 1 ] && [ ! -f "$SETTINGS" ]; then
    echo '{"ecg": false, "acc": false, "rr": false, "horizon_s": 1.0, "refresh_s": 0.5, "samples": 4}' > "$SETTINGS"
    say "Forecasts start at 4 samples (change it on the page)"
fi

# ---------------------------------------------------------------- 6. timing
if [ "$BENCH" = 1 ]; then
    DEV=cuda; [ "$JETSON" = 1 ] || DEV=cpu
    for n in 4 8; do
        say "Timing the models at $n samples (forecast_service.py --bench)"
        (cd "$REL/polar_dashboard" && "$VPY" forecast_service.py --bench 10 --samples "$n" --device "$DEV") 2>&1 \
            | grep -E "loaded|per forecast|all overlays|no weights|couldn't|Error" || true
    done
fi

echo
if [ "$VENV" = "$HOME/.local/share/bio-dash/forecast-venv" ] || [ "$VENV" = "${BIODASH_FORECAST_VENV:-}" ]; then
    say "Done. The dashboards use $VPY for the forecasts from their next start."
else
    say "Done. This environment isn't the default one: start the dashboards with BIODASH_FORECAST_PYTHON=$VPY"
fi
if [ "$JETSON" = 1 ]; then
    echo "  For the fastest forecasts, run the Orin in its top power mode (a system setting, so it's left to you):"
    echo "    sudo nvpmodel -q    lists the modes (MAXN on an AGX Orin); sudo nvpmodel -m <mode> && sudo jetson_clocks"
fi
