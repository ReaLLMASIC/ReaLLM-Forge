# Evaluation protocol — v0.2.0

## First establish task competence

Run each unmodified pretrained model on fresh synthetic examples at 4K, 8K, 16K,
32K. The initial screen uses recall, multiple queried values, and two-hop tracing;
the full pilot adds latest-value updates. Native HF generation is the reference
implementation. Thinking is disabled using each tokenizer's own chat template.
Generation is greedy, with a default 128-token output budget. Limit hits are
logged so truncated explanations are visible rather than mistaken for conclusive
retrieval failures. Keep the same budget across compared arms.

Use native-context task competence to choose the backbone before interpreting a
memory failure. A model that fails when all evidence is accessible provides weak
evidence about the new memory mechanism. The release smoke checks are not a full
candidate screen and do not establish long-context accuracy.

## Matched experiments

Within a backbone, preserve the same revision, seed, training episode sequence,
budgets, local window, memory size, tasks and output budget. All arms use greedy decoding with repetition_penalty=1.0. Compare local, NLERP,
SLERP, and optionally EMA/fixed gates. `--disable-memory` is an inference ablation
on the same trained checkpoint, not a separately trained local control.

Random targets are not always the first record. Multi-query order is also random.
Train/test keys have disjoint namespaces; test seeds and recall wording differ.
Some other task wording is shared so the test remains about retention. Distractors
come from several simple background templates, so these are still synthetic tasks.

Test fact counts 1, 8 and 32 with multiple positions. Two-hop needs two facts, so
its one-fact cell is explicitly recorded as skipped in the evaluation metadata.
The nominal position locates a record region, not necessarily the queried record.
Exact token spans and actual evidence distances are recorded for each example.

Each example records total requested length, actual prompt length, generation
length, native context, adaptation length limit, memory-state bytes, time, GPU
peaks, full raw answer, target and tokenized-prompt hash. Length includes the
reference answer in data construction, not the model's generated answer.

Across different tokenizers, token-length-matched inputs are not identical texts;
model ID is included in grouping and no cross-model paired comparison is made.
Within a backbone, hashes verify exact prompt pairing. Do not combine v0.1 and
v0.2 synthetic results: the generators, models and training modes differ.

## Scoring and uncertainty

Exact match strips leading/trailing whitespace, takes the first answer line, and
normalizes whitespace around commas. It does not search output for a correct value.
Raw outputs remain available to inspect formatting errors and alternative metrics.
A separate `value_exact_match` compares the entire ordered list of six-digit codes
in the final answer. It removes a completed thinking prefix and uses a single
explicit `<answer>` block if present, without consulting the reference to select
codes. Extra, duplicated and reordered codes fail. Both scores are retained. The
screen, learning gate and default plots use code-value accuracy; `--metric
exact_match` requests the stricter formatting metric. Do not conflate the two.

The primary memory metric filters for all relevant evidence being evicted from
the local queue at the last prompt token. This flag is also computed for native
baselines as a hypothetical local-window condition; native models retain access.
For two-hop questions every required record must be evicted for this subset.

Summary CSVs include each task/length/fact-count/position cell with bootstrap
intervals. Plots show pooled trends; inspect the cell tables. Paired SLERP-minus-
control differences include only identical prompts within the same backbone and
training seed. Their intervals resample examples, not independent training runs.
Use at least three independently trained seeds for publication and report those
separately. An all-zero bootstrap interval is a failure floor, not evidence of
method equivalence. Duplicate files or checkpoints must not be pooled as if they
were independent samples.

Completed evaluations have `status=complete` in their `.meta.json`. Assessment
rejects an incomplete evaluation. No implicit evaluation resume or overwrite is
performed; after an interruption choose a new output path for a complete rerun.

## Natural-text retention

Prepare bounded document-separated PG-19 splits with the matching tokenizer.
Suffix perplexity evaluates an identical final 512-token target after nested
prefixes. It is token-level PPL, not PG-19's published word-level metric.

```bash
bash run.sh perplexity --run runs/hunyuan-slerp-seed17 \
  --data-dir data/pg19-hunyuan --split test --lengths 2048 4096 8192 \
  --out reports/hunyuan-slerp-ppl.jsonl --device cuda
```

The generated-history command remains an exploratory free-running continuation
check: a model emits its own code, continues past the local window, then is asked
for the code. It logs absent initial facts and repetitions rather than silently
counting them as successful remote recall. It uses a raw continuation protocol;
it is not the primary instruct-model screen and is not combined with random-fact
exact-match results.

## Official benchmark bridge

Use official generation/scoring from https://github.com/NVIDIA/RULER and preserve
its commit, tokenizer revision, task configuration and generation budget. This
package does not install or run the official suite automatically. The prediction
bridge accepts `input` or `prompt`, preserves target fields and adds `pred`.

```bash
bash run.sh predict-jsonl --config configs/hunyuan-native.json \
  --input /path/to/official-task.jsonl --out reports/ruler-native-predictions.jsonl \
  --max-new-tokens 128 --device cuda

bash run.sh predict-jsonl --run runs/hunyuan-slerp-seed17 \
  --input /path/to/official-task.jsonl --out reports/ruler-slerp-predictions.jsonl \
  --max-new-tokens 128 --device cuda
```

Inputs are used as supplied by default. Add `--chat-template` only if they have not
already been formatted, and apply the same choice to all compared arms. Select
the official task's generation budget rather than assuming 128 is universal.
There is no silent prompt truncation. Use the official scorer afterward. LongBench
requires its own task formatting and scorer; the built-in synthetic suite is not
an implementation of either benchmark.

## Decision order

1. Native parity passes and unmodified model solves the target task.
2. Small full-weight learning gate improves held-out evicted recall.
3. Full training profile meets the actual GPU memory requirement.
4. Matched SLERP/local/NLERP comparison, plus short-context retention.
5. Multiple training seeds and official benchmark runs before general claims.
