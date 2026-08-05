# Attention Residuals

Set `attention_residual_variant: full` (or pass
`--attention_residual_variant full`) to replace the running residual sum with
token-local attention over depth.

For each Transformer block, the implementation:

1. mixes the embedding and earlier sublayer outputs for the attention input;
2. appends the raw self-attention output to depth memory;
3. computes a separate mixture for the MLP input; and
4. appends the raw MLP output.

One final mixture is passed to `ln_f`. Each destination owns a learned
pseudo-query. Queries are initialized to zero, so every mixture initially is an
equal-weight average. Keys are parameter-free RMS-normalized source vectors,
values are the raw vectors, and softmax is only over depth. Consequently this
feature does not replace causal token-to-token self-attention.

Full Attention Residuals store the embedding and all `2 * n_layer` sublayer
outputs, and perform quadratic work in the number of sublayers. The current
implementation supports sequential attention-then-MLP blocks without post-LN;
the usual PreNorm configuration is supported. Use `standard` (the default) for
the existing additive residual architecture and checkpoint compatibility.
