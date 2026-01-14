#!/bin/bash

dataset="minipile"
# obtain and tokenize minipile
pushd data/"$dataset"
if [ ! -f "train.bin" ] || [ ! -f "val.bin" ] || [ ! -f "meta.pkl" ]; then
  bash get_dataset.sh
  python3 prepare.py -t input.txt --method tiktoken
else
  echo "train.bin val.bin and meta.pkl already found for ${dataset}"
fi
popd

# Train a fully quantized asic model
## using a linear quantization scheduler, increasing to full quantization
## after 10000 iterations
python3 train.py \
    --out_dir asic_quant \
    --use_edgellm_asic \
    --mlp_variant edgellm_asic_mlp \
    --attention_variant edgellm_asic_attn \
    --max_iters 20000 \
    --full_quant_iteration 10000 \
    --dataset "$dataset" \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 256 \
    --batch_size 64 \
    --no-bias \
    --dtype bfloat16 \
    --use_gradual_activation \
    --activation_start gelu \
    --activation_end relu \
    --activation_transition_start_iter 0 \
    --activation_transition_end_iter 20000 \
    --quantization_warmup_iters 0 \
    --no-use_pre_ln \
    --use_flash_norm \
    --quantize_attn_act \
    --quantize_mlp_act \
    --quantize_asic_prenorm \
    --quantize_asic_attn_softmax_denom \
    --quantize_asic_attn_softmax_denom_bits 16 \
    --quantize_asic_attn_softmax_numerator \
    --quantize_asic_attn_softmax_numerator_bits 8 \
    --linear_variant_attn quantized_linear \
    --linear_variant_mlp quantized_linear \
    --quantize_linear_method symmetric_quant \
    --activations_quant_method symmetric_quant \
    --dropout 0 \
    --grad_clip 1.0 \
    --beta1 0.95 \
    --beta2 0.95 \
    --weight_decay 0.05 \
    --learning_rate 0.75e-3 \
    --quant_scheduler linear \
    --max_sample_tokens 100 \
    --sample_each_eval

# Test the model's inference capabilities when holding the scales and zero points static
python3 sample.py \
    --out_dir asic_quant \
    --eval_only \
    --eval_dataset="$dataset" \
