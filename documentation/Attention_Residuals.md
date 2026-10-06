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

## Comparison with Kimi K3

The `full` variant implements the Full Attention Residuals equations described
in the Kimi K3 report, but it is **not an exact reproduction of Kimi K3's
deployed residual topology**. The following pieces agree:

- routing is independently computed for every token and destination;
- the embedding and earlier transformation outputs are used as values;
- keys are RMS-normalized while values remain unnormalized;
- a learned, destination-specific pseudo-query scores the keys; and
- a separate final depth mixture is computed before the output norm.

The important differences are:

1. **Kimi K3 uses Block Attention Residuals.** Its released configuration uses
   a 12-layer residual block size. Completed blocks are represented by sums,
   while the current block contributes its running partial sum. K3 therefore
   routes over block-level states rather than retaining every transformation
   output. The K3 report describes eight 12-layer blocks (with a partial final
   block) and nine routing sources when the embedding is counted. This
   repository currently offers only the full, per-sublayer form.
2. **The source granularity differs.** Here, attention and MLP outputs are each
   stored as independent sources, producing `2 * n_layer + 1` destinations.
   K3's block implementation keeps ordinary additive accumulation within each
   12-layer group and exposes completed block sums plus the current partial sum
   to both the attention and MLP routing points.
3. **K3's released scorer has an affine RMSNorm.** Its effective pseudo-query
   is the elementwise product of a destination-specific RMSNorm scale and a
   bias-free scalar projection. This implementation uses parameter-free
   RMSNorm and one query vector. The two parameterizations can express the same
   score, but do not have identical parameters or optimization dynamics.
4. **Initialization is different.** This implementation zero-initializes each
   query, deliberately starting with uniform depth weights. K3's released code
   applies its normal linear-layer initialization to the scalar projection, so
   K3 does not explicitly enforce an initially uniform mixture. The K3 report's
   equations do not prescribe an initialization.
5. **Routing precision is different.** K3's released implementation converts
   values, normalization, scores, softmax, and the weighted reduction to FP32,
   then casts the result back to the activation dtype. This implementation
   does not explicitly promote the routing calculation to FP32.

Consequently, no term is missing from the documented **Full Attention
Residuals equation**, but block aggregation, K3's scorer parameterization and
initialization, and its explicit FP32 routing path are missing for strict Kimi
K3 parity. Adding a `block` variant should be preferred over changing `full`,
so existing experiments and checkpoints retain their present semantics.

## ReLU2Max routing

Set `attention_residual_weight_variant: relu2max` (or pass
`--attention_residual_weight_variant relu2max`) to replace the depth softmax
with normalized squared-ReLU weights. For depth scores `s`, the implementation
uses

```text
c_i = s_i - mean_depth(s)
t_i = relu(c_i + shift)^2
w_i = t_i / sum_depth(t)
```

The default `shift` is `1.0` and can be changed with
`--attention_residual_relu2max_shift`. Centering preserves invariance to a
constant score offset. The shift is important because pseudo-queries are
zero-initialized: an unshifted squared ReLU would produce all-zero weights and
zero query gradients, whereas the shifted form begins as the same uniform
mixture as softmax and remains trainable. Normalizing by the sum, rather than
using the fixed divisor of the token-attention `ReLU2Max` variation, keeps the
depth result a convex combination and prevents its scale from growing with the
number of retained residual sources. If every squared-ReLU term is zero, the
implementation safely falls back to uniform weights.

References:

- [Kimi K3 technical report, section 2.2](https://arxiv.org/abs/2607.24653)
- [Released Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- [Released Kimi K3 text-model implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)
