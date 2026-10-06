# Summarization protocol

## Purpose and sources

Test whether compressed memory preserves a whole document's main findings,
explanations and qualifications after raw tokens exceed working/native attention
limits. Generated-summary quality is the endpoint. Lower training loss or merely
running beyond 8K is insufficient. A compact memory is lossy for arbitrary inputs.

Pinned SmolLM2 revision: `a10cc1512eabd3dde888204e902eca88bddb4951`.
Pinned GovReport revision: `4e21184e01ae8017e2c036e180fe5e541fef60a0`.

Primary references:

- [SmolLM2 architecture/config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/config.json)
- [GovReport HF corpus](https://huggingface.co/datasets/ccdv/govreport-summarization)
- [GovReport original project](https://gov-report-data.github.io/)
- [HF streaming datasets](https://huggingface.co/docs/datasets/stream)
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)
- [Gist Tokens](https://arxiv.org/abs/2304.08467)
- [Canonical BERTScore implementation](https://github.com/Tiiiger/bert_score)

## Whole data and honest context limits

GovReport configuration `document` maps `report` to `summary`. Token-length bins
are half-open: [512,4096), [4096,7168), [7168,8193), [8193,16384), [16384,32768).
The pilot selects the first 32 train, three validation and four test documents
per bin, scanning at most 20,000 rows per split. This is deterministic length
stratification, not an unbiased corpus sample. Manifest counts expose incomplete
quotas. Exact duplicate sources across prepared splits are skipped; near-duplicate
detection is not implemented. Whole documents and references are never clipped.

HF streaming reads Parquet ranges. Only accepted rows, text and IDs are persisted,
with a 250 MB selected-data cap across splits. Network transfer and HF cache are
distinct from this cap; all active files remain subject to the project disk guard.
Choose a new data directory for a different tokenizer or sample. Completed files
are immutable. After interruption, inspect and deliberately remove/move the
incomplete `.pending` split, then prepare only missing splits with `--splits`.

Training selects whole sources <=16,384 and whole reference targets <=1,024 tokens,
then appends EOS. Larger targets are skipped, not truncated. Native warmup selects
sources <=6,144 and checks prompt + full target against the native limit. Rejection
counts are saved. Evaluation retains every selected full reference; its 512-token
generation cap may reduce coverage of long summaries. Output-limit hits and both
text lengths are logged. Change the answer budget uniformly in a new study if it
becomes a dominant limitation.

`source_over_native` means source >8,192, independent of prompt length.
`full_request_over_native` includes prompt and reserved output. A 7K–8K source can
fit the source limit while the full request does not. `source_over_working` means
source >2,048. Native evaluation skips requests that cannot fit; it neither clips
the source nor increases positions. Hunyuan's 262K native context means the same
32K sweep would measure working-window compression only.

## Memory implementations

Native and text baselines use the actual HF backbone/cache with manual greedy
decoding and final-position-only vocabulary projection. Text-summary approaches
first summarize passages and then generate the answer; no reference text enters
the generator. All intermediate prefill/decoding costs are counted.

Local/NLERP/SLERP/delta use 2K KV, 256-token eviction chunks and rebased local RoPE.
Local-only can indirectly carry history in recent contextualized KV; it is not
fresh tail truncation. NLERP/SLERP share learned writer/reader/slot architecture,
64 slots per KV head per layer, and differ principally in interpolation geometry.
That pair is the closest controlled geometry comparison.

RMT uses a causal read–process–write bottleneck: 64 memory hidden vectors precede
each 256-token segment; 64 learned write positions follow it. Only the resulting
write hidden states pass between segments. Source tokens cannot attend to future
writes. Generation uses a live HF cache until the segment boundary, writes memory,
then resets cache and positions. Train/inference follow the same protocol. This
is RMT-inspired soft memory, **not an implementation of the Gist Tokens paper**.

Delta maintains a value-by-key matrix per KV head/layer. Key and value dimensions
equal the backbone head dimension. Each evicted chunk first applies a learned
decay per head, then corrective token writes `S <- S + beta (v - S k) k^T`.
Keys are projected and unit-normalized. A block triangular solve computes this
tokenwise recurrence exactly, with forward/gradient equivalence tests. Queries
retrieve from the matrix through learned output/read gates. This is a plain
associative control, not full Gated DeltaNet or its optimized kernels. Its matrix
size is independent of `slots`.

These methods have different operation counts/capacities. Report extra parameters,
actual state bytes and measured latency; identical `slots=64` is not equal resource
usage. After screening, sweep RMT slot/segment sizes and NLERP/SLERP slot counts
on validation to compare useful quality/resource frontiers. Gist-specific masks,
KV quantization/eviction packages, sparse attention and Mamba are future arms.

## Full-weight supervision, fairness and KL

Shared native warmup: cap 500K processed tokens. Recurrent arms initialize from
those exact backbone weights, then share document order/seed and a 2M cap. Every
stage stops after three passes if earlier. One document at a time; accumulation
targets 8,192 prompt-plus-target tokens. CE is mean per-document target loss,
averaged over documents in an update. Whole-document processing can overshoot a
cap by one example. Source, total, supervised and optional teacher/student KL
tokens are logged separately. Small pilots do not establish convergence.

Backbone/memory LR: 1e-5/1e-4, FP32 AdamW weights/moments/gradients, BF16 CUDA
matmuls, gradient clipping, warmup/cosine schedule, checkpointing and chunked
supervised vocabulary projections. Temporal gradients span the entire source.
Bounded inference state does not bound training activations.

Text baselines use shared warmup weights; recurrent arms receive extra full-weight
adaptation. Their comparison is a system comparison, not solely the causal effect
of memory. The recurrent local arm gets the same additional data budget as other
learned arms. Extra control: evaluate `native_full`, `head`, `tail`, `rolling` or
`map_reduce` with each arm's checkpoint. Baseline generation uses its backbone
without the memory path. Metadata preserves checkpoint identities. Multiple
training seeds are separate replications, not independent copies of documents.

`distill_weight=0` by default. Setting it to, for example, 0.02 in a new study adds
exact vocabulary `KL(frozen full-source teacher || compressed student)` every
fourth document at the first 16 answer prediction positions. Both see the same
source and short gold-answer prefix. There are no rewards, policy ratios or RL
advantages. The teacher is the initial backbone for that stage, also on resume.
The entire teacher input must fit native context; longer examples log a skipped
distillation term instead of receiving a truncated teacher. Profile the specific
KL study because it adds a frozen model and extra computation. The old retrieval
`model.kl_weight` is rejected here to avoid applying the wrong objective.

Exact resume uses unchanged study, data and method:

```bash
bash summary.sh train --method slerp --out runs/summary-smol-seed17-slerp \
  --resume --device cuda
```

Use a fresh run for changed settings. Summary initialization currently accepts a
native warmup checkpoint, not an arbitrary recurrent-stage checkpoint. Saves
contain the full backbone/memory, tokenizer/config, optimizer, progress and RNG.
Explicit finalization retains weights but removes previous optimizer resumability.
Reload custom memory checkpoints with this package's
`RecurrentLM.from_pretrained(path)` or the summary CLI. Their container format is
not directly loadable by vanilla HF `AutoModelForCausalLM.from_pretrained(path)`;
the backbone and memory protocol must be restored together.

## Measurement and interpretation

Timing excludes model load and initial source tokenization; all methods reuse the
same prepared IDs. It includes prompt assembly, intermediate generation and
tokenization, memory processing and final decoding. Time to the first final token
includes all earlier passes. Early EOS can cause different answer lengths; report
lengths and cap hits with speed. Scoring runs outside generation timing.

GPU peaks include loaded parameters/workspaces. RSS is sampled every 20 ms and can
miss brief peaks. These are distinct measurements; do not simply add them. CUDA
cache is cleared between documents. `max_call_cache_bytes` means largest returned
**persistent** state across calls, not peak activations/transient workspaces or
total inference memory. Energy is not measured.

Default eval weights stay FP32, BF16 CUDA compute, for every method. Optional
`eval_weight_dtype=bfloat16` is a separate precision study; memory geometry stays
FP32. Paired latency comparisons require matching hardware name, runtime versions
and precision. Hardware load/clocks/power mode still need experimental control.

ROUGE uses `rouge-score==0.1.2`, stemming, and deterministic punctuation/newline
sentence splitting for Lsum. It is not an official leaderboard recipe. The optional
semantic scorer uses pinned MiniLM contextual token vectors in 192-token blocks
and all-token maximum cosine matching. Every summary token is covered, with some
cross-block encoder context lost. There is no IDF or baseline rescaling. It is
**not canonical BERTScore, entailment, or factuality assessment**.

The blind review sheet maps to source text via its hash in prepared JSONL. Check
central findings, relationships across sections, qualifications, contradictions,
unsupported claims and coherence. The separate key maps blind IDs to methods.
Increase documents and training seeds after the pilot. Bootstrap intervals cover
paired document variation in this selected set only. OOM/skips stay in count
tables; paired quality uses successful common documents and can be selective.

Choose on validation; test fixed settings on untouched test documents:

```bash
bash summary.sh evaluate --method slerp --split test --min-source-tokens 8193 \
  --checkpoint runs/summary-smol-seed17-slerp \
  --out reports/test-slerp-above-native.jsonl --device cuda
bash summary.sh evaluate --method rolling --split test --min-source-tokens 8193 \
  --checkpoint runs/summary-smol-seed17-shared \
  --out reports/test-rolling-above-native.jsonl --device cuda
bash summary.sh report reports/test-*-above-native.jsonl \
  --out reports/test-above-native-comparison
```

Prefer useful quality at low measured latency/RAM. A method that runs past 8K but
loses conclusions is not a successful extension. If all compressed methods fail,
check native quality and training sufficiency before compressing further. Only
after quality survives should you lower precision and benchmark on Orin with its
actual runtime/power telemetry. Host timings are not edge latency/energy forecasts.
