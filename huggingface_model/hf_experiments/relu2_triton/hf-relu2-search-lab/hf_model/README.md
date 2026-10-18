# Hugging Face nanoGPT QK-norm model

This directory provides a `PreTrainedModel`/`PretrainedConfig` implementation
of the architecture used for the repository's QK-normalized attention
experiments. It intentionally ports the relevant path rather than wrapping the
original training model, so checkpoints use normal Hugging Face
`save_pretrained`/`from_pretrained` APIs.

## Exact correspondence

* Q and K are L2-normalized per token and head with `1e-6` added to the norm.
* With QK-norm scaling enabled, the ordinary `1/sqrt(head_dim)` scaling is
  replaced by one learned scalar, initialized to
  `log2(context_length ** 2 - context_length)`.
* RoPE rotates adjacent even/odd channels, uses base 10,000, and supports the
  original partial `rope_length`. It is applied before QK normalization.
* ReLU2Max is exactly `relu(attention_logits) ** 2 / divisor`, optionally also
  divided by key sequence length. It is **not** renormalized to sum to one.
* The model is a pre-norm causal decoder with RMSNorm, GELU MLP, tied token/LM
  head weights, optional GQA, generation cache support, and standard shifted
  causal-language-model loss.

The implementation supports standard softmax as a control. PyTorch's SDPA
primitive computes softmax, so ReLU2Max uses its own fused attention kernel.

### Triton acceleration

On supported NVIDIA GPUs with Triton installed, `relu2max_accelerator="auto"`
now fuses **QK multiplication, scaling, causal/padding masking, ReLU, square,
division, and multiplication by V**. It dispatches before allocating attention
scores. Tiled backward recomputes scores and calculates Q/K/V gradients and the
learned QK-scale gradient without saving quadratic attention matrices.

The fused path supports FP16, BF16, and FP32 on Ampere or newer GPUs, head
dimensions up to 256, transposed model Q/K/V layouts, padding masks, and cached
decoding. It applies when effective attention dropout is zero (including
evaluation). Nonzero training attention dropout and unsupported layouts retain
the existing elementwise Triton/PyTorch path. `relu2max_accelerator="torch"`
forces the reference implementation; `"triton"` requires Triton/CUDA but can
still use the elementwise kernel when full fusion is unsupported.

The mathematical operation is unchanged; fused accumulation and intermediate
rounding can differ from eager low-precision PyTorch. The original
`triton_relu2max(values, divisor)` API remains available for elementwise use and
benchmark comparison. See [performance and validation](RELU2MAX_PERFORMANCE.md)
for the algorithm, limitations, correctness tests, and GPU benchmark commands.
See [RTX 4090 feedback and next steps](RELU2MAX_NEXT_STEPS.md) for the supplied
results, CUDA-graph comparison, alternative frameworks, and HF integration.
The [complete 4090 run and benchmark repairs](RELU2MAX_4090_RESULTS.md) records
the later 4096-token results and updated rerun commands.

## Matched pre-training comparison

From the repository root:

```bash
python scripts/compare_hf_relu2max.py \
  --dataset roneneldan/TinyStories --tokenizer gpt2 \
  --max-steps 1000 --output-dir runs/hf-normalizer-ablation
```

Use `--relu2max-accelerator auto` (the default), `torch`, or `triton` to select
the ReLU2Max execution path.

The direct file command above and the equivalent module form
`python -m scripts.compare_hf_relu2max` are both supported. The launcher
resolves the repository root itself, so it can import `hf_model` regardless of
the caller's current working directory.

The launcher is compatible with both the repository-pinned Transformers 4.44
API (`evaluation_strategy`) and newer releases (`eval_strategy`); it detects
the supported `TrainingArguments` keyword at runtime.

The script downloads/tokenizes the dataset once and runs ReLU2Max then softmax.
It resets Python, NumPy, PyTorch, Trainer, and sampler seeds before each model,
so model initialization, data order, architecture, Muon settings, and schedule
match. It writes each normal Hugging Face checkpoint and a combined
`comparison.json`.

Muon follows the native training setup: matrix-shaped hidden weights use the
quintic Newton--Schulz Muon update, while embeddings, the LM head, and
scalar/vector parameters use auxiliary Adam. Use `--help` for all architecture,
optimizer, precision, and dataset controls.

### A100 80GB pre-training and downstream evaluation

Install `lm-evaluation-harness` (`pip install lm-eval`) and run:

```bash
bash scripts/run_hf_a100_80gb_comparison.sh
```

The preset trains each 124M-parameter variation for approximately 1.31 billion
tokens (10,000 optimizer steps, effective batch 128, sequence length 1,024) in
BF16 with TF32 matrix multiplication. It uses the full TinyStories training
split, appends EOS between documents, and packs tokens into full sequences.
After **each** variation it evaluates the in-memory model on `lambada_openai`,
`hellaswag`, `piqa`, `winogrande`, and `arc_easy`. These cover continuation,
commonsense completion, physical reasoning, pronoun/coreference reasoning, and
elementary multiple-choice reasoning without selecting benchmarks that require
instruction tuning. Results are saved per run and in the combined
`comparison.json`.
Intermediate checkpoints are retained every 1,000 steps (the newest two per
variation). Add `--resume-from-checkpoint` to resume interrupted runs.

The preset is deliberately a shell wrapper: append any comparison-script flags
to override it, for example `--max-steps 20 --lm-eval-limit 20` for an end-to-end
smoke test. Do not use `--lm-eval-limit` for reportable scores. Actual throughput
and maximum batch size depend on the installed PyTorch/Triton versions; lower
`--batch-size` and raise `--gradient-accumulation-steps` proportionally if the
specific A100 environment runs out of memory.

## API use

```python
from hf_model import NanoGPTConfig, NanoGPTForCausalLM

config = NanoGPTConfig(attention_normalizer="relu2max")
model = NanoGPTForCausalLM(config)
model.save_pretrained("my-model")
model = NanoGPTForCausalLM.from_pretrained("my-model")
```
