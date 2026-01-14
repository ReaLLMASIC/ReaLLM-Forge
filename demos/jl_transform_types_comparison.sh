#!/bin/bash
# demos/jl_transform_types_comparison.sh

embd_dim=350

echo "1. Prepare minipile dataset"


pushd data/minipile
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin, val.bin, and meta.pkl already found for minipile"
fi
popd

echo "2. Train original minipile model"
out_mini="out_minipile"

python3 train.py \
  --dataset minipile \
  --attention_variant infinite \
  --n_head 3 \
  --n_qk_head_dim 120 \
  --n_v_head_dim 120 \
  --use_concat_heads \
  --out_dir "$out_mini" \
  --tensorboard_run_name original_minipile \
  --max_iters 5000 \
  --compile

echo "3. Apply Gaussian JL Transform with multiple std devs"
STD_DEVS=(0.8 1.0 1.2)

for std in "${STD_DEVS[@]}"; do
  proj="proj_${embd_dim}_gauss${std}.pt"
  out_dir="${out_mini}_jl_gauss${std}"

  python3 initializations/jl_transform_ckpt.py \
    "$out_mini" \
    --out_embd "${embd_dim}" \
    --proj_out "$proj" \
    --out_dir "$out_dir" \
    --jl_type gaussian \
    --gaussian_std "$std"

  python3 train.py \
    --dataset minipile \
    --init_from resume \
    --out_dir "$out_dir" \
    --tensorboard_run_name "minipile_jl_gauss${std}" \
    --max_iters 5000 \
    --compile
done

echo "4. Apply other JL transform types"
TYPES=("sign" "sparse" "srht" "qr")

for jl_type in "${TYPES[@]}"; do
  proj="proj_${embd_dim}_${jl_type}.pt"
  out_dir="${out_mini}_jl_${jl_type}"

  python3 initializations/jl_transform_ckpt.py \
    "$out_mini" \
    --out_embd "${embd_dim}" \
    --proj_out "$proj" \
    --out_dir "$out_dir" \
    --jl_type "$jl_type"

  python3 train.py \
    --dataset minipile \
    --init_from resume \
    --out_dir "$out_dir" \
    --tensorboard_run_name "minipile_jl_${jl_type}" \
    --max_iters 5000 \
    --compile
done

