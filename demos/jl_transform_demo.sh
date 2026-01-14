#!/bin/bash
# demos/jl_transform_demo.sh

embd_dim=350

echo "1. Prepare datasets"

# obtain and tokenize shakespeare char
bash data/shakespeare_char/get_dataset.sh

# obtain and tokenize minipile
pushd data/minipile
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for minipile"
fi
popd

echo "2. Train models of same embedding dim"
out_mini="out_minipile"
out_shakes="out_shakespeare"

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

python3 train.py \
  --dataset shakespeare_char \
  --attention_variant infinite \
  --n_head 3 \
  --n_qk_head_dim 120 \
  --n_v_head_dim 120 \
  --use_concat_heads \
  --out_dir "$out_shakes" \
  --tensorboard_run_name original_shakespeare \
  --max_iters 2000 \
  --compile

echo "3. Transform Shakespeare model with Gaussian-based JL Transform"
# default out dir name is <input_name>_jl
# and gaussian jl_type
# python jl_transform_ckpt.py ckpt --out_embd 360 # easiset
python3 initializations/jl_transform_ckpt.py \
  ${out_shakes} \
  --out_dir "${out_shakes}_jl" \
  --out_embd "${embd_dim}" \
  --proj_out proj_"${embd_dim}".pt \
  --jl_type gaussian \
  --gaussian_std 1.0 # For experiments with different std devs, see jl_transform_types_comparison.sh

echo "4. Transform minipile model with saved projection matrix used with minipile model"
python3 initializations/jl_transform_ckpt.py \
  "$out_mini" \
  --out_embd "${embd_dim}" \
  --proj_in proj_"${embd_dim}".pt \
  --out_dir "${out_mini}_jl"


echo "5. Demonstrate earlier convergence with transformed models"
python3 train.py \
  --dataset shakespeare_char \
  --init_from resume \
  --out_dir "${out_shakes}_jl" \
  --tensorboard_run_name transformed_shakespeare \
  --max_iters 2000 \
  --compile

python3 train.py \
  --dataset minipile \
  --init_from resume \
  --out_dir "${out_mini}_jl" \
  --tensorboard_run_name transformed_minipile \
  --max_iters 5000 \
  --compile
