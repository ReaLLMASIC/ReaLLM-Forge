# Full TED and FLORES evaluation of final CJK checkpoints

This evaluates the 12 Japanese/Korean/Simplified Chinese × original/IPA ×
Char/CharBPE checkpoints from the CJK experiment. It loads the final `ckpt.pt`
weights and the original `meta.pkl` for each run. It never enters `Trainer`,
constructs an optimizer, restores optimizer state, calls backward, updates
parameters, or writes a checkpoint. Checkpoint/model state fingerprints and
gradient checks enforce this during evaluation.

The feature is based on PR #926. While that PR remains open, this is a dependent
change; a PR against upstream `master` can also show the prerequisite commits.
Keep experimental checkpoints, tokenized corpora and new raw evaluation outputs
outside the checkout. The explicitly bundled offline dashboard is a checked-in
textual snapshot of the original experiment; it does not include model weights
or training corpora.

The [bundled offline dashboard](../report/pr926-dashboard/index.html) preserves
the original 12-run training snapshot and its sampled validation measurements.
Double-click its `index.html` to explore it with the included chart assets and
data. It is separate from this evaluator's new full-corpus outputs.

## Measurement contract

- Teacher forcing, unchanged model output distribution, no sampling or top-k filter.
- Context length from the checkpoint (1024 for the original runs); stride 512 by
  default. The first window scores all its targets; later windows score only the
  new suffix, using the overlap as context. Each predictable source token is
  counted once. The last short window is included; padded labels use `-1` and
  never contribute to any metric.
- Each split is a continuous newline-separated stream. Context resets between
  splits, not at every sentence. The initial source token of each split is
  excluded because these vocabularies have no trained BOS. No BOS/EOS is added.
- CUDA uses the requested autocast precision; logits/probabilities/CE/cosines are
  measured in float32 and sums accumulated in float64. Float32 CPU smoke tests
  are not numerically equivalent to historical bfloat16 CUDA evaluation.
- `loss`: mean cross-entropy in nats/target. `bits_per_byte`: total NLL / ln(2) /
  UTF-8 bytes represented by **scored** targets. A fallback byte counts as one
  raw byte, including partial UTF-8 sequences; decoding with replacement would
  give a wrong denominator. Initial excluded token bytes are excluded too.
- `top1_prob`, `top1_correct`, `target_prob`, `target_rank`,
  `target_left_prob` retain their training definitions. Rank is `1 + count(logit
  > target_logit)`; left probability sums exactly those competitors. Ties are
  excluded. Top-1 uses argmax and therefore need not be correct for every tied
  rank-1 target.
- `target_rank_95` uses the global rank histogram and linear interpolation.
  Historical `left_prob_95` is the 95th percentile of left mass **plus target
  mass**. Historical `ln_f_cosine_95` is actually the **5th** percentile of final
  normalization/target output-vector cosine. These names/definitions are kept.
- Target rank/probability/left mass match `PerTokenMetrics`' FP32 implementation.
  The old bulk trainer computed softmax in autocast dtype and compared rounded
  probabilities for left mass; tiny differences from that path are possible.
- Percentiles are over all scored occurrences, not per-batch quantiles or
  vocabulary-type averages. Exact float quantiles use temporary disk-backed
  arrays; rank histograms and per-token sums use bounded RAM. Saved distributions
  support exact train+val aggregation without rerunning a model.
- The original geometry helper computes token vector lengths/minimum pairwise
  angles once per loaded model. `--no-geometry` is useful for smoke tests.
  Original `training_seen_count` comes from the matching final training snapshot;
  evaluation counts are separate. RankMe/AREQ remain disabled.

Historical random 50-batch validation and the new complete traversal have
different context/coverage distributions. Preserve both results; do not interpret
their difference as further training. Token loss/rank have different prediction
units across tokenizers; use BPB as the primary Char/CharBPE comparison within a
language and representation. Original-text BPB and IPA BPB use different texts.

## 1. Prepare FLORES (networked machine or offline source)

The existing `data/flores200-res/get_dataset.sh` points to
`muhammadravi251001/restructured-flores200`. The new preparation script resolves a
requested revision to an immutable Hub SHA, reads the CJK columns from Parquet,
and records source files, hashes, split labels, row order and sentence IDs. It
does not invent canonical dev/devtest labels when a restructured source uses
different splits. Source IDs are retained when present; otherwise explicit
shard/row IDs are used.

```bash
export CJK_RUN_ROOT=/absolute/path/to/original-run
export CJK_DATA_ROOT="$CJK_RUN_ROOT/ReaLLM-Forge/data/cjk_ipa_charbpe"
export CJK_EVAL_ROOT=/absolute/path/outside-checkout/cjk-evaluation
export CJK_TED_REVISION=d307aa82efc9b649630cbc545c532c0a99600cf1
bash demos/cjk_full_checkpoint_eval.sh prepare-flores
```

The converter matches `demos/cjk_ipa_charbpe_compare.sh`:

| Language | Conversion |
|---|---|
| Japanese | `ja2ipa.py --text_output --text_no_sentence --text_field unspaced_ipa`; no spaCy/MeCab |
| Korean | `espeak2ipa.py --lang ko --mode text --no-wrapper` |
| Simplified Chinese | `zh_to_ipa.py --input_type text --no-wrapper` |

Converter source hashes, exact commands, package versions, stats/diagnostics and the Korean
espeak-ng version are recorded. Inspect diagnostics for passthrough or failed
transcriptions: nonempty output alone does not prove every word was converted.
TED was downloaded as a prepared package; matching the local driver flags does
not by itself prove identical upstream converter dependency versions. The
manifest keeps that distinction explicit.

For offline preparation, supply `CJK_FLORES_RECORDS=/path/to/records.jsonl`:

```json
{"id":"sentence-1","split":"dev","texts":{"ja":"日本語。","ko":"한국어.","zh_cn":"中文。"}}
```

Every row must include all three nonempty single-line translations. Duplicate
IDs within a split and missing/empty/multiline translations fail validation.
Supply source revision/provenance alongside externally prepared records; an
offline file hash identifies the supplied input, not an upstream Hub commit.

Only frozen TED vocabularies are used. CharBPE retains the repository's
longest-string matching with UTF-8 byte fallback, not merge-rank replay. Every
encoded stream must round-trip byte-exactly. No vocabulary/merges are trained.

Plain Char has no unknown token or fallback. Full Char results with any OOV are
marked `unencodable_oov`, with affected character counts and sentence/byte
coverage in `manifest.json`. Full CharBPE remains available. Both models receive
an additional **identical sentence-ID subset** of jointly encodable sentences;
this is labeled `jointly_encodable` and is never substituted for the full result.
If no sentences qualify, the subset is `no_scorable_tokens`.

## 2. Verify the final checkpoint/data inventory

```bash
bash demos/cjk_full_checkpoint_eval.sh prepare-all
```

Preparation checks all 12 recorded final iterations, model/tokenizer vocabulary
sizes, original training-exposure counts, token counts, ID ranges and decoded
UTF-8 byte counts, and hashes all checkpoint/metadata/token files. TED uses the
existing `train.bin` and `val.bin` without re-encoding. Both splits are evaluated
in full, with independent contexts. The combined result uses token/byte-weighted
totals and global quantiles; it is not held-out validation.

If FLORES is not available yet, `prepare-ted` builds a TED-only manifest. Use a
fresh output root to build the later combined manifest; the evaluator refuses
to reuse outputs with changed artifacts or evaluation settings.

## 3. Test and smoke-evaluate

Install the repository's existing PyTorch/model/IPA dependencies first. The new
unit tests use Python unittest, without requiring pytest:

```bash
python3 -m unittest discover -s tests -p test_cjk_full_eval.py -v
bash -n demos/cjk_full_checkpoint_eval.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash demos/cjk_full_checkpoint_eval.sh smoke
```

Smoke mode scores up to 2049 source tokens per TED split for every checkpoint on
CPU with geometry disabled. Results are explicitly marked `partial` and written
under `smoke/`; they never count as complete-corpus measurements. To select one
checkpoint or another limit, use `benchmarks/evaluate_cjk_checkpoints.py run
--help`. Labels/counts/hashes refer to the exact input manifest.

## 4. Run full evaluation and inspect outputs

```bash
bash demos/cjk_full_checkpoint_eval.sh run
bash demos/cjk_full_checkpoint_eval.sh report
```

Full evaluation defaults to CUDA bfloat16 and fails clearly if CUDA is
unavailable; it never silently starts an extremely long CPU job. CPU use requires
explicit `--device cpu --dtype float32`. Batch size, stride and geometry settings
are recorded. There is no compile by default, avoiding extra compile dependence.

Outputs include:

- `source_manifest.json`, `evaluation_config.json`: artifact hashes, software,
  implementation hashes and evaluation settings.
- `<run>/<corpus_split_scope>/result.json`, `.complete`: measured summary and
  output-hash completion marker written only after all planned positions are
  scored. Output hashes are rechecked before reusing completed jobs.
- Per-token CSV: original loss/count/probability/rank/geometry/training-exposure
  fields plus generic `loss`/`eval_count`. FLORES jobs use validation aliases even
  if the source provider calls a split `train`.
- `totals.npz`, disk-backed occurrence distributions: support exact combined
  metrics. These and all other binary artifacts stay outside Git.
- `<run>/ted_combined.json`, `summary.tsv`, `report.md`, comparison PNGs.

Completed jobs can be resumed with identical settings and source hashes.
An incomplete nonempty job directory is never silently overwritten: inspect and
archive it, then use a fresh output directory. Changing a batch size, precision,
stride, manifest or evaluator implementation requires fresh outputs.

For a follow-up PR, include this feature's source/tests/docs, the test output,
small result tables and links to hosted datasets/large reports. State clearly
which corpora finished and which remain unavailable; smoke results cannot
stand in for full TED/FLORES results.
