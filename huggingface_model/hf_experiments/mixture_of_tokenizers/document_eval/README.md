# Per-document evaluation and paired uncertainty

Version 1.0.0 — October 25, 2026 (America/Los_Angeles).

This add-on is the next evaluation stage for the completed 100M-token mixture-of-tokenizers experiment. It uses the existing seed-1337/1338, K=10,000/20,000 best checkpoints. It adds files only under `document_eval/` and does not replace the experiment's source scripts.

## Run on the existing Linux/RTX 4090 environment

Download `mixed-tokenization-document-evaluation.zip` to `~/Downloads`. Use the same Python environment that ran the previous evaluations; no dependency upgrade is needed.

```bash
cd ~/nanogpt_mixture_of_tokenizers/huggingface_model/hf_experiments/mixture_of_tokenizers
unzip -n ~/Downloads/mixed-tokenization-document-evaluation.zip
python document_eval/run_next.py --dry-run
python document_eval/run_next.py --device cuda
```

The dry run checks the input files and prints commands without loading weights or starting evaluation. The real run prints the full path of `document-evaluation-results-<timestamp>-<id>.zip` when finished. Upload that ZIP for review.

If interrupted, run the same command again. Completed exports are accepted only after verifying their source text, checkpoint/tokenizer/source-file hashes, software versions, reference evaluation and aggregate arithmetic. An unfinished job restarts that one evaluation; there is no within-job checkpoint. A completed launch releases its output lock. If the process is killed without cleanup, inspect `runs/document-evaluation/.run_next.lock` and remove it only after confirming the recorded process stopped.

## What it runs

Eight evaluations, sequentially on one GPU:

| Seed | Existing run root | Vocabulary sizes | Context policies |
|---|---|---|---|
| 1337 | `runs/100m` | 10000 and 20000 | bytes and token |
| 1338 | `runs/100m-seed1338` | 10000 and 20000 | bytes and token |

Every job uses `checkpoint-best`, the first 1,000 validation documents per language, native and shared-byte routes, context 512, the existing FP32 evaluation behavior, and separately reported EOS loss. The completed seed-1338 20k final-checkpoint sensitivity control remains a separate analysis and is not substituted into these primary comparisons.

It then runs 10,000 paired bootstrap replicates, using seed 20261025, and creates an archive containing the eight exports, comparison, accepted reference evaluations, run configurations/summaries, add-on scripts, and checksums. Weights and source-text contents are not included. No training, tokenizer preparation, or benchmark download is run.

## Output layout

```text
runs/document-evaluation/
  seed1337/k10000/{bytes,token}.json
  seed1337/k20000/{bytes,token}.json
  seed1338/k10000/{bytes,token}.json
  seed1338/k20000/{bytes,token}.json
  comparison-<timestamp>-<id>.json
document-evaluation-results-<timestamp>-<id>.zip
```

Each export contains ordinary aggregate scores plus one row per language/document occurrence, with a zero-based index, exact UTF-8 SHA256, byte/character counts and native/byte loss totals. Repeated texts retain separate indices. Text contents are omitted. Empty documents retain EOS accounting and contribute zero text bytes/NLL.

The exporter reuses the installed `evaluate.py` functions (`iter_documents`, `score_text`, `summarize`). It requires source hashes and integer token/window/fallback counters to match the corresponding accepted `evaluation-val-{bytes,token}-1000.json`. Per-language/route BPB must agree within absolute tolerance `1e-5`; mean EOS NLL has a similarly small tolerance. A mismatch stops the workflow before publishing that export. Investigate changed weights, text, code, precision, or environment instead of loosening the threshold silently.

## Statistical meaning

Documents are sampled with replacement separately within each language. The same sampled indices are used for every seed, vocabulary, context mode and route. For each replicate, language BPB is `sum(text_nll) / sum(source_bytes) / ln(2)`. The macro averages the four language BPBs equally. The combined result averages BPB across the two fixed seeds before calculating vocabulary differences and percentage reductions.

The result includes point estimates and percentile 95% intervals by language and macro, by seed and for the two fixed-seed mean. Positive `delta_bpb_20k_minus_10k` and `k10000_reduction_percent` favor 10k. These are document-resampling intervals conditional on the existing models, not confidence intervals for a population of training seeds. They are marginal intervals without a multiple-comparison adjustment. Correlated documents from the same underlying source can make document-level intervals too narrow. They do not isolate tokenization from capacity or training-exposure differences.

## Optional checks and settings

```bash
# CPU correctness tests; no downloads, weights or corpora needed.
python -m unittest discover -s document_eval -p 'test_*.py' -v

# A separate output directory, when deliberately changing the environment.
python document_eval/run_next.py --device cuda --output runs/document-evaluation-rerun

# Change bootstrap replication count; evaluation semantics stay fixed.
python document_eval/run_next.py --device cuda --bootstrap-repetitions 20000
```

Use the ordinary launcher for this matched comparison. Individual `evaluate_documents.py` and `paired_bootstrap.py` commands are available through `--help`, but the launcher supplies the expected paths and checks all run identities first.

## Validation and remaining work

The add-on is tested on CPU with analytic probabilities, Unicode/empty/repeated documents, aggregate reconciliation, paired-bootstrap fixtures, and launcher workflow simulations. The exporter reproduces every original evaluator aggregate exactly on its analytic fixtures in both modes. The user's real checkpoint evaluation and resulting confidence intervals remain to be run on the 4090. No downstream benchmark adapters are included; those are the next implementation stage after this export.
