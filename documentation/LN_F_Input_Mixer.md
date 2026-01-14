# Mixing Block Outputs Before Final Layer Norm

The `use_ln_f_input_mixer` option blends hidden states from each transformer block
(including the initial embedding) into a single representation prior to the final
layer normalization (`ln_f`). The `ln_f_input_mixer_variant` setting controls how
the outputs are combined:

- `linear` – learned linear combination (default)
- `router_top1` – router selects a single block output per token
- `router_topk` – router mixes the top-k outputs with softmax weights
- `decoder` – a full-attention decoder layer attends over all block outputs

## Usage

Enable the feature in code:

```python
config.use_ln_f_input_mixer = True
```

or from the command line:

```bash
python train.py ... --use_ln_f_input_mixer
python train.py ... --use_ln_f_input_mixer --ln_f_input_mixer_variant router_top1
```
For `router_topk`, control the number of mixed routes with `--ln_f_mixer_top_k`.

Weights for the linear mixer are initialized to focus on the last block's output,
preserving the standard behavior when the option is disabled.

