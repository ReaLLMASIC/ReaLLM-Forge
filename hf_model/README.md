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

The implementation supports standard softmax as a control. ReLU2Max disables
fused scaled-dot-product attention by construction, which is necessary because
PyTorch's fused primitive always computes softmax.

### Triton acceleration

On CUDA systems with Triton installed, `relu2max_accelerator="auto"` fuses the
ReLU, square, and divisor into one elementwise kernel. A custom backward kernel
also fuses the derivative and incoming-gradient multiplication. Masked logits
remain zero because the causal/padding mask is applied before the kernel.
Unsupported devices and dtypes automatically use the ordinary PyTorch path.
Set `relu2max_accelerator="torch"` to force that reference path, or `"triton"`
to require CUDA/Triton and receive an explicit error when unavailable. This
accelerates the ReLU2Max transformation itself; QK and probability-value matrix
multiplications remain PyTorch operations.

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
