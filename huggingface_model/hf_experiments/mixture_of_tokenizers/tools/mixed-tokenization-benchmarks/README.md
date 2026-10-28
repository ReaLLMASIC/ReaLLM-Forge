# Language benchmarks for the mixture-of-tokenizers experiment

This add-on evaluates the four existing `checkpoint-best` models: K=10,000 and K=20,000, each with training seeds 1337 and 1338. For each language it scores the matching native route and the shared raw UTF-8 byte route. It never trains or changes a checkpoint. Keep using the Python environment that ran your previous evaluations; do not upgrade it for this add-on.

The byte route is byte-only tokenization **inside the jointly trained mixture model**. These results do not represent a separately trained byte-only language model. Scores measure the model, tokenizer, and training allocation together; K20k also has more parameters and different source-text exposure.

## What is measured

| Language | Language modeling evaluation | Short commonsense task | Public final split | Questions / chance |
| --- | --- | --- | --- | --- |
| Chinese | Prepared FineWeb2 Chinese test documents | XCOPA-zh | test | 500 / 50% |
| Korean | Prepared FineWeb2 Korean test documents | KoBEST-COPA | test | 1,000 / 50% |
| Japanese | Prepared FineWeb2 Japanese test documents | JCommonsenseQA v1.3 | test | 1,118 / 20% |
| English | Prepared FineWeb English test documents | PIQA | validation | 1,817 / 1,838 eligible / 50% |

The default LM evaluation uses the first 1,000 stored test documents in each language, with unchanged text and matching hashes across every condition. These are the experiment's frozen held-out splits, not official FineWeb benchmark splits. It rejects insufficient documents and exact-document overlap with the corresponding training or validation files. The overlap audit is not a near-duplicate or public-benchmark contamination audit.

The primary LM metric is **bits per UTF-8 byte (BPB)**; the report also gives **byte-normalized perplexity = 2^BPB**. Lower is better. Raw token perplexity cannot rank different tokenizers because their prediction units differ. BPB is computed from each tokenizer's canonical sequence, not a likelihood marginalized over alternative tokenizations. Compare tokenizers within a language; UTF-8 byte widths make cross-language BPB rankings misleading.

The primary context policy uses the same Unicode-safe chunks of at most 510 UTF-8 bytes for all routes, with BOS resets. It reports genuine document EOS separately from text BPB. This is the existing matched-text evaluator, not sliding-window full-history perplexity. The wrapper also supports `--context-mode token` for a separately labeled secondary evaluation.

Task scoring uses short, fixed zero-shot continuation prompts and scores answer text. It requires no generation, instruction tuning, answer-letter decoding, or benchmark fine-tuning. It separately tokenizes prompt and answer, sums answer log probabilities, and excludes EOS. These are **custom fixed adapters**, not official leaderboard scores.

Primary task results are raw summed-log-probability accuracy, gold-option log-loss, and gold-versus-best-incorrect log-probability margin. Exact ties get fractional credit. Chance accuracy is 1 / number of choices; chance option log-loss is ln(number of choices). Higher accuracy/margin and lower loss are better. Option log-loss uses a softmax over candidate sequence scores, not token perplexity. The secondary result uses log-probability per answer UTF-8 byte; this normalization is identical across tokenizers, and must not be selected after seeing which metric favors a model.

All choices must fit `BOS + prompt UTF-8 bytes + answer UTF-8 bytes <= 512`. If any choice fails, the entire question is excluded from **all K values, seeds and routes**. Native token counts are checked against that shared byte bound. The report lists coverage and skipped IDs. The prepared final CJK sets all fit; 21 longer PIQA questions are excluded without truncation, leaving 4,435 eligible questions overall.

## Run next

Download `mixed-tokenization-benchmarks.zip` to `~/Downloads`, then run these commands in the environment used for your existing evaluations:

```bash
cd ~/nanogpt_mixture_of_tokenizers/huggingface_model/hf_experiments/mixture_of_tokenizers
unzip -n ~/Downloads/mixed-tokenization-benchmarks.zip -d tools
python tools/mixed-tokenization-benchmarks/run_benchmarks.py --dry-run
python tools/mixed-tokenization-benchmarks/run_benchmarks.py
```

The dry run checks all four completed run identities, checkpoint/tokenizer presence, the 512-token context and document availability. It prints commands without downloading, scoring or writing results. The real run downloads approximately a few MB of pinned public benchmark files, evaluates sequentially on one GPU, creates tables and paired intervals, then prints the path of `language-benchmark-results-<timestamp>-<suffix>.zip`. Upload that ZIP for review. It contains results, provenance, code and summaries, without weights or corpus/benchmark text.

Outputs are under `runs/language-benchmarks-v1/`:

- `results/seed1337/k10000/` and the other three seed/K combinations: `lm-bytes.json` and `tasks.json`.
- `report/report.md`, `report/summary.json`, `report/scores.csv`: all task and LM results, with paired task comparisons.
- `tasks/manifest.json`: source revisions, SHA-256 hashes, licenses and exact prompt versions.

If interrupted, rerun the same command. A completed score file is reused only after its inputs, checkpoint, source code, settings and stored arithmetic pass validation. Changed inputs or a changed environment require a new `--output` directory. A process lock prevents two launchers from using one result directory; after a forced kill, inspect its recorded PID/hostname before removing a stale lock.

If a language has fewer than 1,000 test documents, preflight fails with the available count. Use a smaller fixed value for all conditions, e.g. `--max-documents 500 --output runs/language-benchmarks-v1-500`. It never silently substitutes validation text or scores different sets for different models.

No runtime forecast is claimed for the new tasks. The earlier eight document-scoring loops totaled about 117 minutes on the user's recorded evaluation runs, but task inference, repeated model loading and this new test sample differ. The runner logs elapsed scoring time.

## Progress during future training

Use `--split-set dev --output runs/language-benchmarks-v1-dev` for validation monitoring on the current four-run layout. Chinese dev has 100 questions, Korean 500, Japanese 1,119; English uses the same public PIQA validation set as the final suite, because public test labels are unavailable. It must not be described as an independent English final test after repeated monitoring. For later checkpoints, call `score_tasks.py` / `score_lm.py` directly with their saved model directory and a new output path; the four-run launcher intentionally checks the current completed 100M experiment identities.

Days to a week of training do not guarantee above-chance commonsense accuracy for this small multilingual model. BPB is the main continuous early signal; candidate log-loss and margins can reveal changes before accuracy moves, but can also reflect calibration or length bias. Keep the templates and primary metric fixed. Evaluate periodically, avoid choosing checkpoints using test scores, and reserve the CJK test sets for a final comparison. A claim about learning progress needs multiple training checkpoints; the four current models compare conditions at approximately 100M training targets. Seed-1338 K20k's best checkpoint is slightly before its last update, as recorded in the existing handoff.

The report uses 10,000 paired question bootstrap draws by default, preserving the same sampled question indices across models and both fixed seeds. It reports within-seed comparisons and a descriptive mean across the two fixed seeds. These marginal intervals condition on the trained models; they are not uncertainty over the population of possible training seeds and are not a correction for multiple comparisons. An English task result alone cannot establish the optimal multilingual vocabulary size.

## Already known: validation language modeling

These are the existing verified matched-text validation BPBs, averaged over seeds 1337 and 1338 (1,000 identical documents per language). They are **not new test results**:

| Language | K10k native | K10k byte | K20k native | K20k byte |
| --- | ---: | ---: | ---: | ---: |
| Chinese | 2.2253 | 2.8625 | 2.3034 | 2.9145 |
| Korean | 1.7250 | 2.1414 | 1.7919 | 2.1844 |
| Japanese | 1.5495 | 2.0212 | 1.6146 | 2.0563 |
| English | 1.9721 | 2.1984 | 2.0347 | 2.1943 |

Native routes have lower validation BPB than byte routes in this experiment. K10k leads every native language, while the English byte comparison is small and seed-sensitive. This does not isolate tokenizer quality: shared-byte training is spread across languages, route token budgets imply unequal source exposure, and the K conditions have unequal parameter counts.

## Sources and verification

The downloader pins every source by SHA-256, and Git/Hugging Face sources additionally by commit. It uses standard-library HTTPS downloads of plain data files, without executing remote dataset scripts. Public data files retain their original text; only the documented prompt adapters transform the evaluation inputs. Existing data or normalized artifacts that disagree with the pinned bytes cause an error rather than replacement.

- XCOPA: https://github.com/cambridgeltl/xcopa — CC-BY-4.0; commit `e2e9d7f105a758ee869cd13e0fe251ef93a29840`.
- KoBEST: https://huggingface.co/datasets/skt/kobest_v1 — CC-BY-SA-4.0; commit `a5ea15e3ac77ed694b79f6204eb31889a2ba989f`.
- JGLUE: https://github.com/yahoojapan/JGLUE — CC-BY-SA-4.0; commit `6f071c09316baae89c3d083a90985b4b1cb9968c`, JCommonsenseQA v1.3.
- PIQA: https://yonatanbisk.com/piqa/ and https://yonatanbisk.com/piqa/data/ — AFL-3.0; author-hosted validation examples and labels are pinned by file SHA-256 because the site provides no commit revision.
- Tokenization and perplexity: https://huggingface.co/docs/transformers/perplexity .

`VALIDATION.json` records the checks actually run before delivery. Real checkpoint weights and the user's corpus are absent from this assistant workspace. The package has therefore not produced real task accuracy or new test perplexity here. The next run on the user's machine supplies those measurements.
