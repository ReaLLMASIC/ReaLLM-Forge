#/bin/bash

# head to repo root
cd ../

block_size="2048"
eval_iters="10"
few_shot_examples="10"
timestamp="$(date +%F_%T)"
notes="run_hellaswag_benchmark"
run_name="${block_size}${eval_iters}_${notes}"

output_dir="results/${timestamp}_${notes}_hellaswag"
if [ ! -d "${output_dir}" ]; then
  mkdir -p "${output_dir}"
fi

python3 sample.py \
  --init_from "gpt2" \
  --hellaswag_benchmark \
  --device "cpu" \
  --few_shot_examples "$few_shot_examples" \
  --eval_iters "$eval_iters" \
  --block_size "$block_size"