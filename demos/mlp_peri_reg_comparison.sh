#!/bin/bash
# demos/mlp_peri_reg_comparison.sh

# obtain and tokenize minipile
pushd data/minipile
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for minipile"
fi
popd

python3 optimization_and_search/run_experiments.py -c explorations/mlp_cproj_comparison.yaml

# model stat printed outputs
default="default_settings.csv"
post_act_l2="only_post_activation_norm.csv"
cproj_scale="only_cproj_scale.csv"
both="c-proj-scale_and_post-act-norm.csv"

# Create Tables
metric="abs_max"
python3 view_model_stats.py "$default" "$post_act_l2" --stats "$metric"
python3 view_model_stats.py "$default" "$cproj_scale" --stats "$metric"
python3 view_model_stats.py "$default" "$both" --stats "$metric"
