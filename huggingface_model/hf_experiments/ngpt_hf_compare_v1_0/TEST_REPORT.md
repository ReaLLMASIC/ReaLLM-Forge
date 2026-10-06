# Release testing report — v1.0

Test date: 2026-09-29. These are actual execution results, not projected results.

## Executed successfully

`python -m pytest -q` reported **20 passed, 1 skipped in 3.43 seconds**.
The skipped item is the Hugging Face integration module; it contains two
parametrized variant cases that require Transformers.

Core coverage includes both variants' causality and prefix invariance, matched
backbone dimensions and initialization directions, absence of biases, untied
embeddings, effective scale initialization, Q/K normalization after correct RoPE,
explicit SDPA attention multipliers, normalized nGPT hidden states, correct
row/column weight projection, preservation of Parameter identities and optimizer
moments, dense-versus-chunked losses and all gradients, activation checkpointing,
BF16 reference-storage CPU forward/backward passes, deterministic sampling,
optimizer/scheduler settings, and exact CPU pause/resume equivalence for both
models.

`bash -n` passed for the installer and both launch scripts. Python source files
passed bytecode compilation. The synthetic smoke command completed 100 optimizer
updates for **each** model and generated all four PNG/SVG plots.

## Actual synthetic result

| Variant | Initial validation loss | Validation loss after 100 updates |
|---|---:|---:|
| GPT | 3.470158 | 0.489217 |
| nGPT | 3.469344 | 1.733932 |

Loss is next-token cross-entropy in nats/token. This is a tiny artificial noisy
cyclic-transition task, vocabulary 32, width 64, two blocks, four heads, context 32,
batch four, 128 targets/update, seed zero, FP32 CPU execution. It uses the PyTorch
core directly (`--core-only`), not the HF adapter. Both models learned; the baseline
was better here. This does **not** establish either model's performance on
OpenWebText or reproduce the paper's speedup claims.

Training-loss example plots use a labeled trailing mean of five minibatches;
validation is not smoothed. All raw per-update values are in CSV. The main-run
plotter defaults to unsmoothed curves. CPU timing values are local observations,
not RTX 4090 estimates.

## Not tested in this environment

There is no CUDA device here. Transformers and Datasets were not installed, and
network/DNS restrictions prevented installing them. Therefore the following remain
unverified by execution: the selected HF dependency combination, downloading and
tokenizing OpenWebText, `AutoModelForCausalLM` integration and HF checkpoint round
trips, GPU numerical behavior, GPU memory use and speed, and a real language-model
training run. The integration tests are included and `setup.sh` runs them after
installing dependencies on the target machine.

The successful core tests are **not** described as an end-to-end Hugging Face or
GPU validation. Installation and a short real-data pilot should pass before a
longer comparison is started.

## Build environment and included evidence

Python 3.13; PyTorch 2.10.0+cpu; NumPy 2.3.5; Matplotlib 3.10.8; pytest 9.0.2.
See each example `run.json` for recorded versions and exact settings.

`example_results/synthetic_cpu/` contains raw CSVs, manifests, summaries, console
output, pytest output and four PNG/SVG plots. No toy model checkpoints or token
binaries are included, keeping the package small. Regenerate them with:

```bash
python compare_ngpt.py smoke --out runs/my_synthetic_smoke
```
