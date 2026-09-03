# CJK ICL Translation Pipeline

This directory contains an in-context-learning evaluation path for the same CJK
translation task used by `cjk_translation/`, but it evaluates pretrained
checkpoints directly instead of fine-tuned SFT checkpoints.

The prompt format is the same translation prompt used by the SFT pipeline. The
config controls `shot_counts`: `0` is zero-shot, `1` is one-shot, and larger
values prepend that many same-direction training examples before the query.

For IPA, the pipeline uses the native rendered IPA tasks:

```text
source IPA -> target IPA
```

It does not use the IPA source-only orthographic-target mode.

## Quick Start

```bash
python cjk_ICL/icl_pipeline.py --config cjk_ICL/config.example.json
```

Useful smoke run:

```bash
python cjk_ICL/icl_pipeline.py \
  --config cjk_ICL/config.example.json \
  --families tiktoken byte ipa \
  --shot-counts 0 1 \
  --max-examples 12 \
  --force
```

Outputs are written under `cjk_ICL/runs/<model_variant>/shots_<N>/`:

- `<split>_predictions.jsonl`
- `<split>_scores.json`
- `benchmark_easy_predictions.jsonl`
- `benchmark_easy_scores.json`
- `prompt_config.json`

The aggregate files are:

- `cjk_ICL/runs/all_icl_scores.csv`
- `cjk_ICL/runs/all_icl_scores.json`

