This analysis motivated the new ablations. Bias and causal length scaling are now implemented; see the root README and CONTEXT_ATTENTION.md for current commands and definitions.

# Investigating the ReLU² / softmax crossover

Prepared 2026-09-23. This is an implementation audit and a measurement plan, not a diagnosis proved on your trained checkpoints. The available evidence is your observation that ReLU² wins at contexts 256 and 512 and ties at 1024. No corresponding loss tables or trained weights were supplied for this analysis.

The audited release is `relu2-50m-muon-output-norm.zip`, version 1, whose ZIP SHA256 is `304b08ac877bab70b9e985482c70d2598c5d8f4ae6d0063d81ae412e6d21128e`. Its internal root is `relu2-50m-muon-output-norm-kda/`. Its default contexts are 512/1024/2048, so a result containing 256 may come from an earlier or customized sweep; check its saved `run.json` before assuming every setting below applies.

## What the implementation establishes

In `hf_model/modeling_comparison.py`, Q and K are L2-normalized per head after RoPE. Each layer has one learned scalar g, initialized to 20. The 384-wide, three-head model has head dimension 128. For a query i and an allowed key j:

    s_ij = g * dot(unit_q_i, unit_k_j)
    w_ij = max(0, s_ij)^2 / 256
    attention_i = sum_j w_ij * v_j

`configs/sweep_50m.json` sets `relu2max_divide_by_sequence_length` to false. The fused kernel confirms the same operation: it squares positive scores, multiplies by `INV_DIVISOR`, casts the weights to the input dtype, and accumulates weighted values. There is no row-sum normalization. The SDPA arm applies softmax to the same scaled cosine similarities. Softmax here is QK-normalized softmax, rather than a vanilla Transformer with unnormalized Q/K.

Two different consequences follow.

1. **Amplitude can depend on the number of visible keys.** Even weak positive scores add weight. Softmax fixes each row's total weight to one; ReLU² does not. This can change attention's contribution relative to the residual and FFN. QK normalization bounds individual scores but does not bound their aggregate.
2. **The learned scale cannot sharpen pure ReLU² attention.** For positive g, `relu(g*x)^2 = g^2 * relu(x)^2`. Changing g rescales a row without changing the relative weights or which keys are active. In contrast, changing g in softmax changes relative weights by `exp(g*(x-y))`. ReLU² can still learn sharper Q/K directions and more negative irrelevant similarities; it lacks this particular temperature mechanism. A scale crossing zero changes the active sign set, which is why the probe records its sign.

These properties suggest an amplitude problem and/or increasing interference from weakly selected keys at longer contexts. Neither proves that either problem is occurring in your trained models. A stable magnitude with useful distributed attention could be entirely healthy.

## Quantifying the length dependence under a null model

Assume independent, isotropic unit Q/K vectors, head dimension d, fixed positive g, and no learned correlation with V. By symmetry, `E[max(0, dot(q,k))^2] = 1/(2*d)`. Thus, for a row with n allowed keys and divisor C:

    E[row weight sum] = n * g^2 / (2*d*C)

Using g=20, d=128, C=256:

| Visible keys in one query row | Expected ReLU² weight sum | Softmax weight sum |
|---:|---:|---:|
| 256 | 1.5625 | 1 |
| 512 | 3.1250 | 1 |
| 1024 | 6.2500 | 1 |
| 2048 | 12.5000 | 1 |
| 4096 | 25.0000 | 1 |

These are theoretical values, not observations of your activations. In a causal block, the average row has approximately half as many visible keys as the last row. Learned g, Q/K anisotropy, position effects, and correlations invalidate a literal prediction for a trained network.

Do not equate this linear growth in weight sum with linear growth in output RMS. With independent zero-mean values, output variance scales with `sum_j w_ij^2`, giving roughly square-root growth of output RMS with n when per-key statistics stay fixed. Aligned or correlated value components can grow faster. This is why the probe records both row mass and weight energy as well as actual branch RMS.

## Why the short-context win might disappear

At shorter contexts, zeroing negative matches, allowing variable total attention strength, and avoiding competition through a softmax denominator may be helpful inductive or optimization biases. This is a hypothesis about the reported short-context advantage, not a measured explanation. At longer contexts, weak positive matches can accumulate; amplitude and retrieval specificity need separate checks.

There are also experiment-specific alternatives:

- **Softmax may simply benefit more from additional context.** ReLU² need not be deteriorating. Evaluate the same final checkpoints at shared short context and at native context on the same target tokens.
- **Initialization may favor a particular context.** The fixed g=20 happens to be close to the original QKNorm paper's proposed `log2(L^2-L)` initializer at L=1024. That rule gives about 16, 18, 20, and 22 at lengths 256, 512, 1024, and 2048. This is a reason to inspect learned scales and independently tune softmax temperature; it is not evidence that this rule is optimal for this dataset or that using 20 is a bug. Changing g in ReLU² instead changes amplitude quadratically.
- **Document packing introduces distractors.** `experiment/prepare.py` inserts EOS between documents, but attention is allowed across EOS. Longer blocks expose more unrelated text. Stratifying losses around document boundaries, then testing the same document mask for both attention mechanisms, can isolate this effect.
- **The 100M-token regime is an early-training comparison.** The supplied recipe has about 3052 optimizer steps and one seed, with the same token batch and validation cadence across lengths. Check final matched-budget loss and learning curves, not only the minimum validation loss. Best native and best common-context losses can come from different saved steps.
- **Capped output norm is not attention normalization.** It caps the full 384-channel projected branch vector before multiplying by learned channel gains. It does not alter relative attention weights, and gains can move its output outside the original radius. It can control amplitude while leaving retrieval interference untouched. The existing sweep also changes attention and FFN output norms together; an attention-only ablation is needed to isolate an attention-specific effect.

## Read-only checkpoint probe

`probe_context.py` runs each checkpoint through its original attention forward path and reconstructs a few causal attention rows in FP32. It does not train, change parameters, load optimizer state, overwrite model files, or edit the experiment. CUDA runs require the fused custom path and use the selected SDPA backend. Use the same Python environment that can already evaluate the training model.

Extract this folder beside the experiment, or pass its absolute script path. From the experiment root, replace `PATH_TO_RUN` and `PATH_TO_PREPARED_DATA` below with the actual paths:

```bash
python /path/to/relu2-context-diagnosis/probe_context.py \
  --project . \
  --checkpoint PATH_TO_RUN \
  --data PATH_TO_PREPARED_DATA \
  --contexts 256 512 1024 2048 \
  --tokens 32768 \
  --device cuda:0 \
  --output diagnostics/relu2_checkpoint
```

`--checkpoint` accepts a run directory with `latest.json`, a particular checkpoint directory, or `final/`. It resolves the checkpoint once at startup. Prefer completed, matched-budget runs for comparisons. Repeat for softmax and, if useful, linear16, choosing a fresh output directory for each. Repeat separately for capped/uncapped checkpoints. Do not run on a saturated GPU concurrently with the training throughput measurements.

Outputs are saved incrementally after each context in `diagnostics.json`:

- Validation loss and loss by position bucket.
- Per-layer/head sampled row weight sum, squared-weight sum, active-key count, effective-key count, largest weight share, recent-512 weight share, and fraction of scores above 16.
- The actual learned QK scale, including its sign.
- Actual attention/FFN branch RMS before and after the optional output norm, and fraction above the cap, by position bucket.
- Checkpoint identity, training budget/context when available, source hashes, and data manifest identity.

Effective-key count is `(sum w)^2 / sum(w^2)`: it is one for a single selected key and n for uniform weights over n keys. It measures concentration, not whether the selected keys are relevant. Rows with all-zero weights have zero mass, zero effective support, and zero reported shares. Reconstructed diagnostics use the model's Q/K but are not bitwise captures of the fused kernel's internal weights.

Every evaluation context uses exactly the same prefix of shifted validation targets. The grouping and available context change. The position buckets within different block lengths do not necessarily contain identical target subsets. The default 32768 targets are a quick diagnostic sample, smaller than the training sweep's 131072; use `--tokens 131072` for the same evaluation count as the default sweep. Extending evaluation beyond a checkpoint's training length is an extrapolation test, not a result from a model trained at that length.

The saved JSON can be shared with the sweep's `live/sweep_results.csv`, `live/validation.csv`, and `live/output_norm_diagnostics.csv`. These files are enough to assess the leading hypotheses without sharing model weights.

| Observation | Interpretation to test next |
|---|---|
| Row mass/energy and actual branch RMS grow strongly with position | Causal length scaling and attention-only output normalization |
| RMS is controlled but effective support grows and largest weight share falls | Inspect key relevance; try a learned subtractive score bias |
| ReLU² stays ahead at shared short evaluation context but ties at native context | Softmax may be extracting more benefit from distant context |
| Cap rescues the long-context result | Supports an amplitude/optimization explanation; separate attention from FFN caps |
| Both methods gain little from longer evaluation context | Data distribution, training budget, and long-range task coverage need examination |

## Recommended next experiments

Start with one change that preserves the current operation for every query with at most 512 visible keys:

    n_i = number of causally visible, unmasked keys for query i
    denominator_i = 256 * max(1, n_i / 512)^alpha
    w_ij = relu(s_ij)^2 / denominator_i

Test alpha=0 (original), 0.5, and 1. Alpha=0.5 compensates independent-value RMS growth under the null model; alpha=1 compensates coherent weight-mass growth. Neither is guaranteed to be optimal. Count actual allowed keys, including the KV-cache offset; using the full tensor length for every row would change early prefixes and train/decode scaling. Do not simply enable the existing divide-by-sequence-length option: it multiplies the divisor by the whole key length, changes short-context magnitudes drastically, and does not implement the proposed causal-row rule.

This factor can multiply each row's accumulated output. Alpha=0.5 uses reciprocal square root and alpha=1 uses reciprocal. Both are compatible with a fused Triton epilogue without materializing attention matrices. Their backward must apply the same row factor. Performance needs measurement; no new kernel or speedup is claimed in this package.

If diagnostics instead point to poor selectivity, test:

    w_ij = relu(s_ij - tau_head)^2 / denominator_i

Use a learned, initially zero subtractive bias per head, kept in auxiliary AdamW with the other scalar parameters. This lets the model learn to discard more weak positive matches while retaining subtraction/ReLU/square arithmetic. Avoid imposing a large threshold before checking whether any heads already attend to very few keys. Plain row-sum normalization is also a useful amplitude control, but it leaves the relative weights unchanged, so it cannot directly remove distractors from a frozen attention pattern.

A more substantial option is an input-dependent output gate or mostly local ReLU² with periodic global attention. The GAU/FLASH family provides evidence for gated squared-ReLU attention at long contexts. A gate can suppress an unhelpful mixed output but cannot by itself reconstruct relevant information already mixed with distractors. A local-only model must be tested for loss of distant retrieval; periodic global layers avoid making that trade silently.

For an efficient training screen, keep width 384, three heads, Muon recipe, data prefix, optimizer tokens/update, and one chosen output-norm setting fixed. Compare softmax, original ReLU², alpha=0.5, and alpha=1 at context 2048 with 100M training tokens each. Reuse existing control runs only when their settings and budgets match exactly. Expand promising choices to 1024/4096 and repeat the best comparison across at least three seeds. Retain 512 as a short-context control. Treat new attention rules as new experiment arms, not an exact resume of old runs.

## Research evidence and its limits

1. [A Study on ReLU and Softmax in Transformer (ReLUFormer)](https://arxiv.org/abs/2302.06461), sections 4.1–4.2: analyzes sequence-length-dependent ReLU attention variance and uses a square-root length factor, with per-token lengths for causal attention. It also needs additional regularization. This is ReLU, translation, and a different architecture, so it motivates an ablation rather than proving the right normalization for your ReLU² model.
2. [Replacing softmax with ReLU in Vision Transformers](https://arxiv.org/abs/2309.08586): reports that dividing by sequence length mitigates degradation in vision models. Supports trying the alpha=1 alternative; not direct evidence for causal language modeling.
3. [Sparse Attention with Linear Units (ReLA)](https://aclanthology.org/2021.emnlp-main.523/): studies output normalization/gating, sparse attention, and heads that can assign zero total attention. Supports the relevance of normalization and optional attention strength, not a diagnosis of this sweep.
4. [Query-Key Normalization for Transformers](https://aclanthology.org/2020.findings-emnlp.379/), equation 3: proposes length-dependent initialization of the learned softmax QK scale. The initializer came from translation experiments, not the current small causal language model.
5. [Transformer Quality in Linear Time](https://arxiv.org/abs/2202.10447), equations 3–5, Tables 1–2 and 13: uses squared-ReLU in GAU/FLASH-Quad with gates and additional architectural choices. Its language-model experiments include context 8192. This argues against treating 1024 as a universal ReLU² ceiling; its training scale and architecture differ substantially from this 100M-token sweep. FLASH here is the GAU architecture, not FlashAttention/SDPA.

## Validation of this diagnostic package

`check_probe.py` verified causal masking, row mass, effective support, zero-weight rows, softmax normalization, and the linear16 tail. Twelve tiny real-model/context evaluations cover the three attention variants with capped/uncapped outputs at two lengths; probe losses agree with the existing reference evaluator, parameters remain byte-for-byte equal, and hooks are removed. A save/load CLI check verifies JSON output and refusal to overwrite previous diagnostics. See `validation.log`.

These are CPU functional checks using PyTorch 2.6.0+cpu and Transformers 4.51.3. CUDA execution and your trained-model results remain untested here. No performance measurements or new training results are included.
