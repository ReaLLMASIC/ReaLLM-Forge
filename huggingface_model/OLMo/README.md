# OLMo Fine-Tuning with Custom Attention

This script fine-tunes the pre-trained OLMo model using a custom attention variant.  
It supports all of the variants:

- **model_variant**: `softmax` (default), `relu`, `relu_norm`, `softplus`, `softplus_norm`,  
  `sigmoid`, `sigmoid_norm`, `gradual_relu`, `gradual_softplus`, `gradual_sigmoid`  
- **obo_variant**: `none` (default), `obo`, `learned_obo_per_layer`, `learned_obo_per_layer_per_head`  
- **clip**: turn on “clipped” scaling of attention outputs  
- **gate**: turn on per-head gating

By default it fine-tunes on Wikitext-2 (train+validation) and reports evaluation every `--eval_steps`.

## Usage

```bash
python3 OLMo_train.py \
  --model_variant <variant> \
  --obo_variant <variant> \
  [--clip] \
  [--gate] \
  --lr <learning rate> \
  --weight_decay <weight_decay> \
  --max_steps <max training steps> \
  --eval_steps <steps before evaluation> \
  --save_steps <steps before saving model checkpoint>
```

## Example Invocations

# 1) Default softmax (no clipping, no gating, no off-by-one)
python3 OLMo_train.py

# 2) Clipped softmax (uses best zeta=1.0, gamma=-0.03 values from Qualcomm paper: https://arxiv.org/pdf/2306.12929)
python3 OLMo_train.py \
  --clip \
  --model_variant softmax \

# 3) Can combine gating and clipping with other softmax variants
python3 OLMo_train.py \
  --model_variant softplus \
  --gate \

# 4) Can combine learned off-by-one with other normalized softmax variants
python3 OLMo_train.py \
  --model_variant relu_norm \
  --obo_variant learned_obo_per_layer_per_head \

# 4) Gradual Softmax Variants (will linearly convert from Softmax/LOBO Softmax to Softmax Variant over max_steps)
python3 OLMo_train.py \
  --model_variant gradual_relu \
  --max_steps 10000 \

# 5) Fully-featured: clipped + gated + learned per-head off-by-one, normalized softplus
python3 OLMo_train.py \
  --model_variant softplus_norm \
  --clip \
  --gate \
  --obo_variant learned_obo_per_layer_per_head \
  --lr 5e-6 \
  --weight_decay 0.05 \
  --max_steps 20000 \
  --eval_steps 1000 \
  --save_steps 2000 \