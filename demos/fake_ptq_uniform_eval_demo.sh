#!/bin/bash
# demos/fake_ptq_uniform_eval_demo.sh
#
# Runs the fake PTQ pipeline across a sweep of uniform bit-widths and records
# validation loss for each configuration. The script prepares the Shakespeare
# dataset, trains a compact reference model (if necessary), quantizes the
# checkpoint across a configurable bit-width range, evaluates each checkpoint
# with `train.py --eval_only`, and plots the resulting validation loss curve.

set -euo pipefail

EVAL_DATASET_DIR="data/shakespeare_char"
OUT_DIR="out_fake_ptq_shakespeare"
SWEEP_ROOT="${OUT_DIR}_uniform_sweep"
EVAL_ITERS=200
BATCH_SIZE=64
BLOCK_SIZE=256

BIT_START=8
BIT_STOP=3
BIT_STEP=-1

usage() {
  cat <<'EOF'
Usage: demos/fake_ptq_uniform_eval_demo.sh [--bit-start N] [--bit-stop N] [--bit-step N]

  --bit-start  Starting bit-width for the sweep (default: 8)
  --bit-stop   Final bit-width for the sweep (default: 3)
  --bit-step   Step increment for the sweep (default: -1)
  --help       Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bit-start)
      BIT_START="$2"
      shift 2
      ;;
    --bit-stop)
      BIT_STOP="$2"
      shift 2
      ;;
    --bit-step)
      BIT_STEP="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! mapfile -t BITS < <(seq "$BIT_START" "$BIT_STEP" "$BIT_STOP"); then
  echo "Failed to generate bit-width sweep with start=$BIT_START, step=$BIT_STEP, stop=$BIT_STOP" >&2
  exit 1
fi

if [ "${#BITS[@]}" -eq 0 ]; then
  echo "Bit-width sweep is empty; adjust --bit-start/--bit-stop/--bit-step" >&2
  exit 1
fi

echo "Sweeping uniform weight bit-widths: ${BITS[*]}"

mkdir -p "$EVAL_DATASET_DIR"

echo "=== Step 1: Prepare the shakespeare_char dataset ==="
pushd "$EVAL_DATASET_DIR" > /dev/null
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
else
  echo "Found existing tokenized dataset artifacts."
fi
popd > /dev/null

mkdir -p "$SWEEP_ROOT"

echo "=== Step 2: Train a reference model on shakespeare_char (if needed) ==="
if [ ! -f "$OUT_DIR/ckpt.pt" ]; then
  python3 train.py \
    --dataset shakespeare_char \
    --out_dir "$OUT_DIR" \
    --n_layer 5 \
    --n_head 3 \
    --n_embd 384 \
    --use_rotary_embeddings \
    --no-use_abs_pos_embeddings \
    --block_size "$BLOCK_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --eval_iters "$EVAL_ITERS" \
    --attention_variant infinite \
    --n_qk_head_dim 120 \
    --n_v_head_dim 120 \
    --use_concat_heads \
    --use_peri_ln \
    --use_qk_norm \
    --use_qk_norm_scale \
    --max_iters 800 \
    --eval_interval 800 \
    --compile
else
  echo "Found existing checkpoint at $OUT_DIR/ckpt.pt; skipping training."
fi

echo "=== Step 3: Evaluate the baseline (fp32) checkpoint ==="
python3 sample.py \
  --out_dir "$OUT_DIR" \
  --eval_only \
  --eval_dataset shakespeare_char

step=4
for bit in "${BITS[@]}"; do
  QUANT_OUT_DIR="${SWEEP_ROOT}/${bit}bit"
  mkdir -p "$QUANT_OUT_DIR"

  echo "=== Step ${step}: Quantize to ${bit}-bit weights ==="
  if [ ! -f "$QUANT_OUT_DIR/ckpt.pt" ]; then
    python3 quantizations/ptq/fake_quantize_ckpt.py "$OUT_DIR" \
      --out_dir "$QUANT_OUT_DIR" \
      --num_bits "$bit"
  else
    echo "Found existing ${bit}-bit checkpoint at $QUANT_OUT_DIR/ckpt.pt; skipping quantization."
  fi

  step=$((step + 1))

  echo "=== Step ${step}: Evaluate the ${bit}-bit checkpoint ==="
  python3 sample.py \
    --out_dir "$QUANT_OUT_DIR" \
    --eval_only \
    --eval_dataset shakespeare_char

  step=$((step + 1))
done

python3 - "$OUT_DIR" "$SWEEP_ROOT" "${BITS[@]}" <<'PY'
import json
import os
import sys

out_dir = os.path.abspath(sys.argv[1])
sweep_root = os.path.abspath(sys.argv[2])
sweep_bits = [int(arg) for arg in sys.argv[3:]]

if not sweep_bits:
    raise SystemExit("No bit-width sweep values provided to plotting helper")

entries = [("fp32", 32, os.path.join(out_dir, "eval_loss.txt"))]
for bit in sweep_bits:
    entries.append((f"{bit}-bit", bit, os.path.join(sweep_root, f"{bit}bit", "eval_loss.txt")))

results = []
for label, bit, path in entries:
    if not os.path.exists(path):
        raise SystemExit(f"Missing evaluation summary at {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    val = data.get("val")
    if val is None:
        raise SystemExit(f"No 'val' key found in {path}")
    results.append((bit, float(val), label))

results.sort(key=lambda item: item[0], reverse=True)

csv_path = os.path.join(sweep_root, "uniform_quantization_eval.csv")
with open(csv_path, "w", encoding="utf-8") as csv_file:
    csv_file.write("bits,label,val_loss\n")
    for bit, loss, label in results:
        csv_file.write(f"{bit},{label},{loss:.8f}\n")
print(f"Wrote summary CSV to {csv_path}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; skipping plot generation.")
else:
    baseline = next((item for item in results if item[2].lower() == "fp32"), None)
    quantized = [item for item in results if item is not baseline]

    bits = [item[0] for item in quantized]
    losses = [item[1] for item in quantized]

    plt.figure(figsize=(8, 4.5))
    line_handle = None
    if bits:
        (line_handle,) = plt.plot(bits, losses, marker="o", label="Quantized checkpoints")
    baseline_handle = None
    if baseline is not None:
        baseline_handle = plt.axhline(
            baseline[1], linestyle="--", color="tab:orange", label=f"{baseline[2]} loss"
        )
    plt.gca().invert_xaxis()
    if bits:
        plt.xticks(bits, [item[2] for item in quantized])
    plt.xlabel("Uniform weight bit-width")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs quantization bit-width")
    plt.grid(True, linestyle="--", alpha=0.4)
    legend_handles = []
    legend_labels = []
    if line_handle is not None:
        legend_handles.append(line_handle)
        legend_labels.append("Quantized checkpoints")
    if baseline_handle is not None:
        legend_handles.append(baseline_handle)
        legend_labels.append(f"{baseline[2]} loss")
    if legend_handles:
        plt.legend(legend_handles, legend_labels)
    plt.tight_layout()

    plot_path = os.path.join(sweep_root, "uniform_quantization_eval.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Wrote plot to {plot_path}")
PY

echo "Sweep complete. Evaluation summaries live in $SWEEP_ROOT."
