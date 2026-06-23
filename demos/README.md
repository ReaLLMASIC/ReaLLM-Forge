# Demos

This folder will hold repeatable demonstrations of features and results.

## Shifted GELU

This shows that the GELU wants to shift:


Call this from either repo root dir, or from demos dir, but ensure the relative
path for ckpt_path is from your present directory.

Example for calling from repo root directory:

```bash
python3 demos/check_ckpt_for_gelu_shift.py \
        --ckpt_path out/ckpt.pt
```

## Optimizer Comparison

`adam_vs_adamw.sh` trains two tiny Shakespeare models, one with Adam and one
with AdamW, then compares their statistics using `view_model_stats.py`.

## lm_head Vocab Magnitude Histogram (TensorBoard Animation)

`lm_head_vocab_histogram_demo.sh` runs a short `shakespeare_char` training job
with `--log_lm_head_vocab_hist` enabled so you can watch the distribution of
per-token `lm_head` vector magnitudes evolve over time in TensorBoard.
It also emits an interactive HTML file with sortable bars (by vocab id asc/desc
or magnitude asc/desc) and hover labels that include both vocab id and rendered
token text.

```bash
bash demos/lm_head_vocab_histogram_demo.sh
```

## Grouped Asymmetric Vector PTQ Comparison

`fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh` runs a
bit-width sweep (default int8 down to int3) on `shakespeare_char` and compares:

1. Original full-vector PTQ (`--granularity vector`, symmetric).
2. Grouped asymmetric vector PTQ (`--granularity vector --quantization asymmetric`)
   while sweeping group-count per vector (default 1 through 10).
3. Grouped symmetric vector PTQ (`--granularity vector --quantization symmetric`)
   over the same group-count sweep.

The demo now gracefully skips requested group-count values that do not evenly
divide the embedding dimension.

Example with default group-count sweep 1..10 for 300-d embeddings:

```bash
bash demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh
```

Example constraining the group-count sweep to 1..10 explicitly:

```bash
bash demos/fake_ptq_asymmetric_grouped_vector_eval_demo_shakespeare_char.sh \
  --group-count-start 1 \
  --group-count-stop 10 \
  --group-count-step 1
```
