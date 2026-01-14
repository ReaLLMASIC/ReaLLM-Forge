#!/bin/bash

ts="$(date +'%Y%m%d_%H%M%S')"
log="logs/run_${ts}.log"

python run_exp.py \
    --user xinting \
    --key ~/.ssh/id_rsa \
    --hosts ../host_configs/internal_hosts.yaml \
    --resume_ckpt /home/xinting/Evo_GPT/optimization_and_search/nsga_search/ckpts/infi_med/1012_0723_ckpt_gen16.json \
    --pop_size 24 \
    --max_layers 24 \
    --min_layers 2 \
    --offspring 12 \
    --generations 34 \
    --exp_name infi_med_resume \
    --conda_env reallmforge \
    --max_iters 10000 \
    2>&1 | tee -a "$log"
