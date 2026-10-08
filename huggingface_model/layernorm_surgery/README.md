# LayerNorm inference sweep

This directory measures the accuracy/cost effect of pruning LayerNorm or RMSNorm gains in Hugging Face causal language models.

## Threshold, fold, quantize, benchmark

Install `lm-eval`, then run, for example:

```bash
python -m huggingface_model.layernorm_surgery.sweep \
  --model google/gemma-3-270m --targets final \
  --threshold-start 0 --threshold-stop 0.20 --threshold-step 0.02 \
  --bits 2,3,4,5,6,7,8 --tasks hellaswag,arc_easy,piqa \
  --output-dir runs/gemma-ln-sweep
```

`--targets final` (the default) selects the last discovered normalization. Exact comma-separated module names and `all` are also supported. Each threshold zeros effective gains satisfying `abs(gain) < threshold`. Gemma RMSNorm stores the multiplier minus one, so these tools translate its parameter to/from the effective gain rather than pruning the raw stored value.

For the final norm, the operation order is deliberately:

1. threshold the per-channel gain;
2. select the LM-head columns whose folded gains remain non-zero;
3. fold those gains into those LM-head columns;
4. symmetrically fake-quantize **only that remaining LM-head matrix**.

Neither normalization gains nor any other model tensors are quantized. Non-zero channel indices are placed first by a recorded permutation and zero columns are omitted from the reduced representation. The ordinary evaluator receives a dense equivalent so no custom CUDA kernel is required.

For a vocabulary of `V` and `Z` zeroed final channels, the reduced final dot product shaves **`V × Z` multiplications and stored scalar parameters**, a fraction `Z / hidden_size` of the head. The sweep reports these estimates as `shaved_parameters` and `shaved_fraction` for every configuration. Internal norms are pruned but do not themselves reduce the final LM-head dot product; only the final norm is folded into that head.

Outputs are `settings.csv`, tidy `results.csv`, and an interactive Plotly `results.html`. The HTML uses the Plotly CDN, so opening it offline requires replacing the script URL with a local Plotly bundle.

> This is inference-time structured channel pruning/fake quantization. The zero columns are mathematically skippable only with a reduced/grouped dot-product kernel; the dense benchmark path estimates, rather than realizes, its latency and memory savings.
