# ReaLLM-Forge search and flexible attention review

Reviewed 2026-10-03. Repository `master` was verified at commit
[`6a42c661d62aafcd10a525f20e736298f0f8a56c`](https://github.com/ReaLLMASIC/ReaLLM-Forge/commit/6a42c661d62aafcd10a525f20e736298f0f8a56c).
Findings below describe that source snapshot, not measured training results.

## Which search code does what

| Source | Role | Reusable concept |
|---|---|---|
| [`hp_searches/README.md`](https://github.com/ReaLLMASIC/ReaLLM-Forge/blob/6a42c661d62aafcd10a525f20e736298f0f8a56c/hp_searches/README.md) and its shell/YAML examples | Experiment entry points | A baseline plus named growth directions |
| [`hyperparam_search.py`](https://github.com/ReaLLMASIC/ReaLLM-Forge/blob/6a42c661d62aafcd10a525f20e736298f0f8a56c/hyperparam_search.py) | Greedy coordinate growth | Evaluate candidates, select the best marginal improvement per extra cost, persist search state |
| [`optimization_and_search/run_experiments.py`](https://github.com/ReaLLMASIC/ReaLLM-Forge/blob/6a42c661d62aafcd10a525f20e736298f0f8a56c/optimization_and_search/run_experiments.py) | Exhaustive JSON/YAML configuration expansion | Scalars, lists, ranges, conditional options, nested parameter groups, named groups |
| [`optimization_and_search/run_vizier.py`](https://github.com/ReaLLMASIC/ReaLLM-Forge/blob/6a42c661d62aafcd10a525f20e736298f0f8a56c/optimization_and_search/run_vizier.py) | Vizier suggestions | A future search engine behind the same trial interface |
| [`view_hp_log.py`](https://github.com/ReaLLMASIC/ReaLLM-Forge/blob/6a42c661d62aafcd10a525f20e736298f0f8a56c/view_hp_log.py) | Search monitoring | Persisted status and a live viewer independent of the trainer |

The hp-search shell scripts call `hyperparam_search.py`, not `run_vizier.py`.
The existing root search imports `train.Trainer` and its argparse objects in process
by default; `--spawn_subprocess` instead calls the root `train.py` with
`sys.executable`. Neither path is a Hugging Face trainer adapter.

Its important arguments are:

```bash
python hyperparam_search.py \
  --orig_settings hp_searches/grow_inf_per_layer.yaml \
  --param_names n_layer n_head_layerlist n_embd mlp_size_layerlist \
                n_qk_head_dim_layerlist n_v_head_dim_layerlist \
  --increments 1 1 16 16 16 16 \
  --iterations 1 --random_iterations 1 --num_iterations 10000 \
  --nlayer_dup_mode dup_each --results_file sweep_log.yaml
```

`iterations` is the number of increment multiples tried for a coordinate;
`num_iterations` is the number of outer growth rounds; `random_iterations` is
the number of candidate seeds. `n_layer` has special handling: it always grows
by one and duplicates layer-list configuration entries. It does not copy trained
weights into a grown model. `dup_each` tests insertion after every existing layer.
The implemented override flag is `--override_cfg`, although the README example
uses `--override`.

The default quality score is `exp(-best_validation_loss)`. The candidate score
is averaged over seeds, then the source defines `avg_loss = -log(mean_score)`;
that is not the arithmetic mean validation loss. Costs can be parameter count,
allocated/reserved/process GPU memory, or average iteration latency. The chosen
candidate maximizes the positive ratio of objective improvement to cost change.

## Issues to fix while adapting the search

1. **Use named, versioned JSON trial results.** The subprocess parser in
   `hyperparam_search.py` lines 132–160 expects the earlier comma-column layout.
   Current `train.py` lines 2421–2461 inserts bits-per-byte as column 1. The parser
   reads that column as integer iteration and reads token count as parameter
   count. A typical decimal bits-per-byte value therefore raises before a result
   is returned. Do not build the HF adapter around this positional text file.
2. **Separate score improvement from cost eligibility.** At lines 793–798, a
   negative quality improvement divided by a negative cost change produces
   positive efficiency; a worse but slightly faster candidate can win.
   Improvements that also reduce cost produce a negative ratio and are rejected.
   Prefer a loss/cost Pareto frontier. If retaining greedy ratios, handle
   dominance explicitly, require a meaningful positive quality improvement,
   and use a cost-noise floor before division.
3. **Keep token budget equal.** `max_iters_increase` changes the baseline config
   at lines 907–934 but does not remeasure its stored baseline score before
   comparing the next round. Changing batch/context also changes token count
   unless the controller fixes tokens per optimizer step. An HF search should
   derive steps from the agreed token budget and remeasure every candidate in
   a new cohort if that budget changes.
4. **Match baseline seeds.** The initial baseline is run once, while candidates
   can average multiple seeds. Use the same seed set for baseline and candidates;
   report arithmetic mean loss, spread, and paired deltas.
5. **Classify failures and isolate CUDA workers.** The greedy search catches
   arbitrary exceptions and returns from a candidate while its saved status
   remains running. The exhaustive runner appends log rows even after a failed
   subprocess and treats logged names as completed on future invocations.
   Record explicit `failed`, `oom`, `unsupported_backend`, and `complete` states;
   only complete equal-budget jobs enter ranking. Use fresh subprocesses and
   preserve resumable optimizer state for interrupted work.
6. **Do not silently change a discrete search space.** The Vizier adapter turns
   numeric lists into min/max intervals and ignores range `step`. Thus a list of
   fast head dimensions can produce unsupported dimensions between them. Use
   categorical/discrete choices and a backend-aware validity predicate.

These are static source findings; they do not establish whether a particular
previous user run encountered each issue.

## InfiniteHeadAttention means flexible dimensions and head merging

The actual file is
[`variations/attention_variations.py`](https://github.com/ReaLLMASIC/ReaLLM-Forge/blob/6a42c661d62aafcd10a525f20e736298f0f8a56c/variations/attention_variations.py#L981-L1315).
Its `InfiniteHeadAttention` is not recurrent Infini-attention or a compressive
memory mechanism. It allows independent head widths and either concatenating
or summing head outputs.

Let `d=d_model`, `H=n_head`, `G=n_kv_group`, `Dq=n_qk_head_dim`, and
`Dv=n_v_head_dim`. Using PyTorch Linear's `(output,input)` weight order:

| Projection/tensor | Shape |
|---|---|
| Q projection weight | `(H*Dq, d)` |
| K projection weight | `(G*Dq, d)` |
| V projection weight | `(G*Dv, d)` |
| Q activations | `(B,H,T,Dq)` |
| K activations/cache | `(B,G,T,Dq)` |
| V activations/cache | `(B,G,T,Dv)` |
| Attention output before head merge | `(B,H,T,Dv)` |
| Concatenated output projection weight | `(d,H*Dv)` |
| Summed-head output projection weight | `(d,Dv)` |

None of these shapes requires `d % H == 0`. For example, `d=512`, `H=3`,
`Dq=128`, `Dv=64` is valid: Q/K are 384-wide, V is 192-wide, and a concat
projection maps 192 back to 512. V width does not affect the QK dot-product
scale or RoPE dimension.

The source additionally handles uneven GQA groups with a deterministic head-to-KV
mapping and `index_select`. That relaxes `H % G == 0`, but expanding K/V costs
memory and should not be represented as a native grouped fused kernel.
For the first adaptation use `G=H`; add explicit GQA support later.

`use_concat_heads=True` flattens all `H*Dv` channels before output projection.
With `False,n_cproj=1`, it sums the heads and projects `Dv→d`. With multiple
output projections, it applies each projection to the same head sum and sums
their outputs. Without an intervening nonlinearity, multiple such projections
are equivalent in the forward function to one projection with their weights
and biases summed; they still change optimization and parameter accounting.
Treat concat-versus-sum as its own ablation. More summed heads change output
magnitude; a head-count scale such as `1/sqrt(H)` is a useful separate experiment,
not something to introduce invisibly while claiming exact upstream semantics.

## Mathematical contract for a Hugging Face attention adapter

Preserve existing lab QK normalization, learned scale, positional encoding,
causal/cache masking, and divisor. Apply RoPE to Q and K using the QK rotary
width; normalize each Q/K head in its final feature dimension, with the same
epsilon placement as the established implementation. Store these choices in
the HF config and checkpoint. RoPE's rotated dimension must be even and no
larger than `Dq`.

After normalization, the intended score is `s = g * dot(q,k)` when the learned
QK scale is enabled. Softmax and both ReLU alternatives must receive exactly
that same score. With SDPA, either pass `q*g` and `scale=1.0`, or pre-scale Q by
`g*sqrt(Dq)` when using SDPA's default scale. Do not detach `g` or use `.item()`;
its gradient must flow. Without the learned scale use the chosen explicit
`1/sqrt(Dq)` convention.

The upstream `InfiniteHeadAttention` flash branch uses the expected
`g*sqrt(Dq)` prescaling. The separate upstream `CausalSelfAttention` flash branch
currently sets `head_dim=sqrt(Dq)` then multiplies by `sqrt(head_dim)`; copying
that code would introduce an extra `Dq^(-1/4)` relative to the desired learned
scale. The HF lab should keep its own tested scale semantics.

The alternatives are unnormalized weights:

```
relu2(s) = max(s,0)^2
linear16(s) = 0                      for s <= 0
              s^2                    for 0 < s <= 16
              32*s - 256             for s > 16
weights = activation(s) / divisor
```

Apply the same causal/padding mask after the activation or otherwise guarantee
masked weights are exactly zero. The linear16 slope is 32 on its upper branch;
at 16 its value and derivative match the quadratic. Its threshold is fixed at
compile time. No normalization to a sum of one should be inserted.

The repository's public `hf_model` snapshot still creates the full score matrix
and accelerates only its elementwise ReLU² operation. Use the lab's complete
fused forward/backward implementation instead. Its independent-width extension
must size Q/K reductions with `Dq` and output/V reductions with `Dv`, including
backward gradients and cache decoding. Padding both to a common width is a
valid compatibility bridge but can waste compute, so label that backend
separately from a native independent-width kernel and benchmark both.

## Optimizer contract

Use module roles and tensor identities instead of loose name-substring rules:

| Parameters | Optimizer | Weight decay |
|---|---|---:|
| Decoder Q/K/V/output and FFN matrix weights | Muon | **0.0** |
| WTE and LM-head matrices | AdamW | **0.1** default, configurable |
| RMSNorm gains, learned QK scalars, biases and other vector/scalar parameters | AdamW | **0.0** |

Tied WTE/LM-head weights appear exactly once. Give Muon and auxiliary AdamW
separate learning rates. The established lab starts at Muon 0.02 and auxiliary
AdamW 0.0006; keep those as explicit starting values rather than asserting they
are optimal for every width and head geometry. Save an optimizer-routing report
and assert that every trainable parameter belongs to one group exactly once.

Upstream `train_variations/optimizer_variants.py` lines 1556–1581 assigns the
same global learning rate and weight decay to both Muon and auxiliary AdamW.
It cannot express the requested split without changing the optimizer setup.
Keep the lab's explicit split optimizer instead of porting that default.

## Recommended adaptation boundary

Use one trial contract under multiple search strategies:

1. Validate an immutable HF config and search cohort (dataset/tokenizer hashes,
   token budget, validation positions, seed set, math/backend policy).
2. Estimate disk use, then run a disposable GPU fit probe that includes optimizer
   state and backward. Record unsupported shapes separately from true CUDA OOM.
3. Run one HF model trial through the lab trainer, with structured progress and
   resumable checkpoints. HF save/load/generation remain the model interface;
   a custom Trainer optimizer hook can consume the same routing if required.
4. Produce a versioned result object containing budget completion, final and
   best scheduled validation loss, training tokens/s, allocation/reservation
   peaks, inference prefill/decode metrics, backend actually used, and errors.
5. Let finite grid/random search, a corrected greedy grower, or later Vizier
   consume those results without changing model mathematics or budgets.

A finite grid/random implementation is a useful first delivery. It should be
described as that, rather than as a completed port of greedy growth, per-layer
growth, or Vizier. Add a compatibility translator only for supported legacy
keys; reject unknown upstream options rather than silently dropping them.

## Highest-value next features

1. **A fair result contract and capability checks.** Equal tokens and validation
   cadence; final loss as well as best loss; explicit failed/unsupported states;
   native/common-context validation; true backend labels. These prevent a fast
   fallback or a partial run from looking like an architectural win.
2. **Resource-aware discrete spaces.** Jointly search H, Dq, Dv, FFN width, and
   depth with real parameter counts, KV-cache bytes, measured 4090 memory, and
   the 30 GB disk retention policy. Start with Dq/Dv in 32/64/128 and even RoPE
   widths; tile boundaries can produce throughput cliffs, not smooth trends.
3. **Pareto plots rather than one ratio.** Validation loss versus training time,
   inference latency, parameter count, and VRAM. Report `Δloss/Δcost` only where
   both the improvement and cost difference exceed measurement noise.
4. **Paired multi-seed confirmation and separate test data.** Search on validation,
   then rerun finalists with multiple matched seeds and evaluate a held-out test
   set once. Greedy search can otherwise select noise over many comparisons.
5. **Hardware precision diagnostics.** Per-head QK scale, positive-score fraction,
   linear16-tail occupancy, attention row mass, activation/gradient percentiles,
   and quantization sweeps. These explain why the tail helps, and estimate actual
   precision requirements instead of relying on random-input kernel timings.
6. **Budgeted multi-fidelity search.** A 25M/50M/100M-token promotion ladder with
   comparisons only within the same rung. Keep the LR schedule contract explicit
   so a paused 100M-token run is not confused with a separately trained 25M run.
7. **Native unequal-width kernels and grouped KV.** Benchmark Dq/Dv independently;
   avoid physical GQA repeats; add a cache-aware decode benchmark. Compile time
   and autotune cache sizes should be tracked when searching many shapes.
8. **Per-layer growth and hybrid attention.** Port the layer-list grammar after
   the uniform-geometry backend is validated; track compile cost and shape
   diversity. Compare concat/sum merges and capped output normalization as
   explicit experimental axes. Keep recurrent memory attention a separate model
   family with its own state/masking/evaluation tests.

For the initial 100M-token experiments, use the same seed, token stream,
validation token positions, evaluation cadence, effective tokens per optimizer
step, QK scale initialization, divisor, and optimizer recipe across attention
variants. Changing H/Dq/Dv changes both capacity and parameter count, so include
counts and cost in every graph rather than calling such comparisons matched-size.
