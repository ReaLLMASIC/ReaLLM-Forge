#!/bin/bash
# demos/ptq_demo.sh

# 1. Prepare minipile dataset
pushd data/minipile
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for minipile"
fi
popd

# 2. Train a larger model on minipile
out_dir="out_ptq_demo"
run_name_before="ptq_fp32"
python3 train.py \
  --dataset minipile \
  --out_dir "$out_dir" \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --block_size 256 \
  --max_iters 10000 \
  --log_interval 10 \
  --tensorboard_run_name "$run_name_before"

# 3. Compute model stats before quantization
python3 train.py \
  --dataset minipile \
  --out_dir "$out_dir" \
  --eval_only \
  --compute_model_stats \
  --print_model_stats_table "${run_name_before}.csv" \
  --tensorboard_run_name "$run_name_before"

# 4. Report validation loss before quantization
python3 sample.py \
  --out_dir "$out_dir" \
  --eval_only \
  --eval_dataset minipile

# 5. Sample from the original model
python3 sample.py \
  --out_dir "$out_dir" \
  --num_samples 1 \
  --max_new_tokens 50 \
  --start "Hello" \
  --sample_file before_ptq.txt

# 6. Apply fake PTQ (8-bit uniform)
python3 quantizations/ptq/fake_quantize_ckpt.py "$out_dir" --num_bits 8 --out_dir "${out_dir}_ptq"

# 7. Compute model stats after quantization
run_name_after="ptq_int8"
python3 train.py \
  --dataset minipile \
  --out_dir "${out_dir}_ptq" \
  --eval_only \
  --compute_model_stats \
  --print_model_stats_table "${run_name_after}.csv" \
  --tensorboard_run_name "$run_name_after"

# 8. Report validation loss after quantization
python3 sample.py \
  --out_dir "${out_dir}_ptq" \
  --eval_only \
  --eval_dataset minipile

# 9. Sample from the quantized model
python3 sample.py \
  --out_dir "${out_dir}_ptq" \
  --num_samples 1 \
  --max_new_tokens 50 \
  --start "Hello" \
  --sample_file after_ptq.txt
