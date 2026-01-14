# Learned Confidence Residual Scaling

This feature lets each transformer block learn a scalar "confidence" for its output before adding it back to the residual stream. A learned vector takes a dot product with the attention or MLP output, optionally adds a constant, and the result scales the activation.

## Configuration

Enable the scalers and choose their initialization in `gpt_conf.py` or via `train_args.py`:

- `--use_attn_resid_scaling`, `--use_mlp_resid_scaling`
- `--attn_confidence_variant`, `--mlp_confidence_variant` (`zeros`, `ones`, `gaussian`)
- `--use_attn_resid_const`, `--attn_resid_const`, `--learn_attn_resid_const`
- `--use_mlp_resid_const`, `--mlp_resid_const`, `--learn_mlp_resid_const`

The scaling occurs after the pre-LN (`peri_ln`) and before the residual addition.

## Exploration

`explorations/learned_confidence_resid_scaling.yaml` compares training with and without this method and sweeps initialization and constant options.
