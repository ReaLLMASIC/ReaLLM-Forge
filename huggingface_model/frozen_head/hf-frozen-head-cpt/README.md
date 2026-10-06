# Frozen LM head during continual pretraining

Hugging Face Hub API + Transformers + Datasets experiment runner for **one H100, targeting the 80 GB model**. The default uses **Qwen/Qwen2.5-1.5B (base)** and full AdamW training. It tests whether freezing the output head reduces forgetting at useful, comparable levels of new-domain adaptation.

This package contains runnable code and offline correctness tests. **No H100 training result or measured H100 VRAM figure is bundled.** Run the included memory preflight on your hardware before the experiment.

**New: task-specific fine-tuning.** [TASK_FINE_TUNING.md](TASK_FINE_TUNING.md) adds response-only GSM8K math SFT, generated-answer accuracy, and standard HellaSwag / ARC / PIQA / BoolQ retention benchmarks. Start that experiment with `bash run_task_h100.sh configs/h100_math_pilot.json`. The original CPT experiment below remains available.

## Run the two-condition pilot

Use Python 3.10+ with a CUDA-enabled PyTorch installation compatible with your H100 driver. The offline tests were run on Python 3.12, PyTorch 2.8.0 CPU, and Transformers 4.57.6. The package intentionally pins Transformers to that tested API rather than tracking its latest major version.

```bash
cd hf-frozen-head-cpt
python -m venv --system-site-packages .venv
source .venv/bin/activate

# Keep the working CUDA-enabled PyTorch from your H100 image.
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'

# Optional: select one GPU on a multi-GPU machine.
export CUDA_VISIBLE_DEVICES=0

# Measures memory, prepares data, trains A, runs both B conditions, and makes plots.
bash run_h100.sh configs/h100_pilot.json
```

Install a CUDA build of PyTorch using the [official selector](https://pytorch.org/get-started/locally/) if your image does not already have one. `pip install -r requirements.txt` alone is not a guarantee that a CPU-only image gains a working CUDA installation.

The model and datasets are public. If Hugging Face asks for authentication or you use gated replacements, run `hf auth login` in your own environment. This code downloads through the Hugging Face APIs and trains locally; it does not submit cloud jobs or upload your results.

Individual commands:

```bash
python cpt.py preflight --config configs/h100_pilot.json
python cpt.py prepare --config configs/h100_pilot.json
python cpt.py run --config configs/h100_pilot.json
python cpt.py report --config configs/h100_pilot.json --gains 0.025 0.05 0.1 0.2
```

## What the pilot does

1. Pins model and dataset repository revisions through `HfApi`, then uses `AutoTokenizer`, `load_dataset(streaming=True)`, and `AutoModelForCausalLM`.
2. Prepares fixed token caches once. Corpus **A is WikiText-103**, using its official train/validation/test splits. Corpus **B is Python code from CodeParrot-clean**, with entire repositories assigned to train/validation/test by a deterministic hash. Exact duplicate text is removed across selected splits and corpora.
3. Performs **100 A-consolidation optimizer steps**, preserving the checkpoint's original embedding tying, and saves a common A anchor.
4. Forks two conditions from that exact anchor: **full CPT** and **frozen head**. Both use independent copies of the initially identical input/output matrix so that output-head freezing does not freeze input embeddings.
5. Trains each condition for **300 B steps**, using the same seed, batch ordering, optimizer-reset policy, learning rate, and token budget.
6. Produces validation trajectories, head-swap diagnostics, final test scores, and retention–adaptation plots.

At 1 × 1024 tokens × 16 accumulation steps, each optimizer step trains on **16,384 targets**. The pilot uses **1,638,400 A targets once** and **4,915,200 B targets per condition**. Prepared training pools are larger than these budgets. Packing appends EOS and uses one token of overlap so every position in a full block has a next-token target; attention can cross EOS between concatenated documents, consistently in every condition.

**Interpret the A score correctly:** WikiText is a retention/consolidation domain, not a reconstruction of Qwen's complete original pretraining corpus. These runs test A-domain retention after A→B adaptation. They do not by themselves establish preservation of all pretrained knowledge, instruction following, or reasoning. Qwen may also have seen overlapping material during its original training; the constructed splits prevent leakage in this experiment, not unknown historical contamination.

Set `a_steps` to `0` to measure A-proxy retention from the untouched pretrained model; all conditions still start from one saved anchor. For a stricter experiment, use your own model pretrained on a known A corpus and supply its held-out A data.

## H100 memory controls

| Setting | Default | Purpose |
|---|---:|---|
| Model | Qwen2.5-1.5B | Full-parameter optimizer fits with useful headroom |
| Parameters and Adam moments | FP32 | Stable updates; avoids silently reducing optimizer precision |
| Matrix computation | BF16 autocast | Reduces compute/activation memory |
| Attention | PyTorch SDPA | Efficient attention without a separate FlashAttention installation |
| Gradient checkpointing | Non-reentrant | Reduces retained backbone activations |
| Microbatch / context | 1 / 1024 | Bounds per-forward memory |
| Gradient accumulation | 16 | Larger effective batch without larger activations |
| Head loss chunk | 128 positions | Never retains the entire sequence × vocabulary logit tensor |
| Allocation ceiling | min(70 GiB, 90% of detected VRAM) | Leaves space outside the PyTorch allocator |
| Execution | Sequential, one process per condition | Only one trainable model/optimizer on the GPU |

The full-vocabulary cross-entropy is **not approximated**. Output loss is calculated in position chunks and checkpointed for backward recomputation. The head still receives all vocabulary gradients when it is trainable. A frozen head still propagates gradients into the backbone.

Persistent memory is estimated as `4 * all_parameters + 12 * trainable_parameters` bytes: FP32 weights, gradients, and two Adam moments. The guard also adds BF16 weight casts, one original-head copy for diagnostics, and a configurable 10 GiB activation/workspace reserve. Untying adds a second embedding-sized parameter matrix; the estimate counts it.

For an untied ~1.8B-parameter model, persistent tensors alone are roughly **27 GiB**, before activations and workspace. This is an estimate, not a measurement. A 7B full-FP32-AdamW model needs about **104 GiB** for those persistent tensors alone, so this runner rejects it on one 80 GB H100. An 80 GB GPU's advertised capacity also need not equal 80 GiB. The runner uses the capacity reported by CUDA.

`preflight` performs two real optimizer updates on synthetic input, exercising state allocation and gradient accumulation, followed by an original-head evaluation. It writes actual peak allocated/reserved VRAM to `preflight.json`. These throwaway weights are discarded. The complete experiment logs its own peak VRAM too. The allocator ceiling constrains PyTorch allocations, not memory used by other processes or all driver overhead. Use an otherwise available GPU.

To try a larger supported model, copy the config, change `model` and `output`, and run preflight first. Do not change a live run's microbatch, precision, model, optimizer, or tokenization to make an OOM disappear: use a new output directory and compare the conditions under the same revised settings.

Plan for **64 GB or more host RAM** with the default model. CPU copies support loading, checkpoint resume, and head diagnostics. Plan for approximately **100 GB free disk for the pilot** to accommodate checkpoint replacement peaks and Hub caches. The larger study retains many final model checkpoints; budget approximately **600 GB or more**, depending on caches and filesystem behavior.

## Run the study and controls

```bash
bash run_h100.sh configs/h100_study.json
```

The study config runs **6 conditions × 3 learning rates × 3 seeds = 54 B runs**, sequentially. Each seed gets one shared 500-step A anchor. Each B run has 1,500 steps / 24,576,000 training targets. This is a substantial study, not the quick pilot. Start with the pilot and inspect whether B improves and A forgetting is measurable before committing to this sweep.

| Mode | Input embeddings | LM head | Backbone |
|---|---|---|---|
| `full` | Train | Train | Train |
| `freeze_head` | Train | Freeze | Train |
| `freeze_embed` | Freeze | Train | Train |
| `freeze_both` | Freeze | Freeze | Train |
| `slow_head` | Train | Train at 0.1× LR | Train |
| `replay` | Train | Train | Train on 90% B + 10% A |
| `freeze_head_norm` (optional) | Train | Freeze | Train except final normalization |
| `tied_full` (optional) | Train, tied | Train, tied | Train |
| `tied_freeze_both` (optional) | Freeze, tied | Freeze, tied | Train |

All primary controls untie at the same A→B boundary. `tied_*` modes require a genuinely tied anchor and are separate practical controls. Output biases, when present, freeze with the output head. Optimizers exclude frozen parameters entirely, so AdamW weight decay cannot move them. Complete frozen tensors are hashed before/after training and on resume.

The default A-consolidation and B training use AdamW (β₁=0.9, β₂=0.95), cosine decay after 5% warmup, weight decay 0.01 for matrix parameters, and gradient clipping at norm 1.0. Dropout is disabled consistently by default. Change these in a fresh experiment if needed. The same learning-rate schedule applies to the backbone in each paired condition. B optimizers always reset at the domain boundary; the anchor optimizer is not inherited by any B arm.

Replay uses a deterministic schedule and only A-training blocks. Replay runs match the **total** training-token budget, so they see fewer B tokens; both A and B counts are logged. Non-replay conditions see exactly the same ordered B blocks at a given seed. Within a cached corpus, blocks follow seeded permutations without replacement, cycling only after exhaustion. Prepared data never draws validation/test examples into training or replay.

Supported model types are `qwen2`, `llama`, `mistral`, `gpt_neox`, and `gpt2`, with standard linear LM heads. The optimized forward uses their base decoder and final normalization, then applies the head explicitly. Other architectures are rejected because some add logit scaling or softcapping. Do not assume compatibility with Qwen3, Gemma, encoder-decoder models, PEFT adapters, or quantized checkpoints.

## Outputs and interpreting the hypothesis

Every B checkpoint is compared to its **shared A anchor**:

```text
forgetting = A_loss(current) - A_loss(anchor)      # smaller is better
gain       = B_loss(anchor) - B_loss(current)     # larger is better
```

| Output | Meaning |
|---|---|
| `analysis/retention_adaptation.png` and `.svg` | Old/new validation loss trajectories and forgetting versus new-domain gain |
| `analysis/trajectories.csv` | Every validation point, token counts, timing, head interventions, geometry, memory |
| `analysis/final_test.csv` | Actual end-only held-out test scores and changes relative to the anchor |
| `analysis/matched_validation.csv` | Forgetting at common new-domain gains; unreached gains explicitly flagged |
| `analysis/paired_validation.csv` | Paired full-minus-intervention forgetting at common gains; positive favors intervention |
| `seed_*/.../metrics.jsonl` | Full structured validation records, including token accuracy and target margins |
| `seed_*/.../test.json` | Held-out A/B end scores |
| `seed_*/.../memory_estimate.json` | Estimated state memory, parameter counts, and tying state |
| `seed_*/.../frozen_initial.json`, `frozen_final.json` | Exact frozen-tensor hashes |
| `data/manifest.json` | Pinned source revisions, prepared file hashes, tokens, and splitting specification |

Evaluation reports cross-entropy in nats/token, perplexity, token accuracy, target-versus-best-competitor logit margin, and a frequency bucket: tokens with at least 20 A-training occurrences whose B relative frequency is below one tenth of their A relative frequency. Head geometry reports a fixed seeded sample's angular drift, norm ratio, and relative weight drift.

The head-swap probe evaluates a small fixed set of A-validation contexts in four combinations:

| Diagnostic | Output head | Hidden states |
|---|---|---|
| `w0_h0` | Original | Original |
| `wt_h0` | Adapted | Original |
| `w0_ht` | Original | Adapted |
| `wt_ht` | Adapted | Adapted |

Original hidden states and the original output head are cached on CPU. Diagnostics temporarily place the original head on the GPU; they never load a second backbone. These are interventions on co-adapted components, not a proof that loss changes add linearly.

**The strongest evidence is less A forgetting at comparable B improvement**, across seeds and reasonable learning rates, with a benefit beyond freezing only input embeddings. A fixed-budget benefit that disappears at matched B gain suggests slowed adaptation instead. Failure to reach the requested gain is an adaptation limitation, not zero forgetting or missing data to be silently dropped.

Matched points use **linear interpolation at the first temporal validation crossing**. They do not sort nonmonotonic training curves by gain, extrapolate unreachable gains, or claim to evaluate an actual interpolated model. Bootstrap intervals resample paired training seeds and are omitted with fewer than three seeds; three-seed intervals remain weak evidence. Do not use test results to select hyperparameters.

For a confirmatory matched-adaptation experiment, select an LR and achievable gain on validation, then create a new config with `stop_at_validation_gain`, for example `0.05`. Each arm stops at its first evaluated crossing and evaluates its untouched test set. Report actual attained gains too: checkpoint spacing means the match is approximate. Use finer `eval_every` near your target. Distinct test suites for reasoning, factual knowledge, or generation should be added when making claims about those capabilities; token-level losses do not establish them.

## Resume and re-evaluate

Re-run the exact same `run` command. Completed conditions are skipped; incomplete conditions resume their latest committed model, optimizer, RNG state, schedule position, and data position. Checkpoints use a new directory plus an atomic pointer update. Logs are trimmed to the committed step. Interrupted partial directories do not replace the previous checkpoint.

SIGINT/SIGTERM request a checkpoint after the current optimizer step. A hard kill can lose progress since the most recent committed checkpoint. Exact CPU equality after an interrupted/resumed run is tested. GPU kernels may not be bitwise deterministic; set `deterministic: true` in a fresh config if required, accepting that a backend can reject a nondeterministic operation.

For a controlled interruption check:

```bash
python cpt.py anchor --config configs/h100_pilot.json --seed 42
python cpt.py train --config configs/h100_pilot.json --seed 42 --mode full --lr 0.00003 --stop-after 50
# Exit code 75 means intentionally interrupted with a checkpoint.
python cpt.py train --config configs/h100_pilot.json --seed 42 --mode full --lr 0.00003
```

Re-evaluate a final saved Hugging Face model:

```bash
python cpt.py evaluate --config configs/h100_pilot.json --seed 42 --mode freeze_head --lr 0.00003
```

Only the latest checkpoint per condition is retained. Completed B runs remove optimizer state by default but retain model weights, metrics, and the completion marker. Set `keep_completed_optimizer: true` before a new run to retain it. Completed runs cannot simply be extended by changing `b_steps`, since that would change the original LR schedule. Use their final weights as a new explicitly defined experiment instead.

The tokenizer is saved once at `runs/.../data/tokenizer`. For custom downstream evaluation, load the model from the directory named by a run's `latest.json`, and load the tokenizer from that shared tokenizer directory. These are standard `save_pretrained` model files, with `tie_word_embeddings=false` preserved for untied conditions.

## Replace the datasets

Each corpus accepts any Hugging Face dataset with a string text field:

```json
{
  "id": "organization/dataset",
  "config": "optional-subset",
  "revision": "main",
  "text_field": "text",
  "splits": {"train": "train", "validation": "validation", "test": "test"},
  "train_tokens": 20971520,
  "validation_tokens": 131072,
  "test_tokens": 131072
}
```

Omit `splits` to hash-partition one `source_split`. Set `group_field` to an existing document-family/repository ID to keep related records together. Without `group_field`, full text determines the partition. A hash split prevents exact document sharing; it does not guarantee semantic or near-duplicate disjointness. Deduplication here removes exact text only. Never hash individual paragraphs from the same document independently when document IDs are available.

For local JSONL, replace the Hub fields with:

```json
"local_jsonl": {
  "train": "/absolute/path/a_train.jsonl",
  "validation": "/absolute/path/a_val.jsonl",
  "test": "/absolute/path/a_test.jsonl"
}
```

Keep the same `splits` mapping and token budgets. Each line should contain the selected text field. To hash-split one local file, supply `local_jsonl` as a filename string and omit `splits`.

Preparation streams source data, retaining only bounded token pools on disk. Training uses memory-mapped uint32 blocks. The exact-text dedup set is kept in host RAM and scales with selected documents. Start with the supplied budgets rather than tokenizing an entire huge dataset. If a source runs out before a budget, actual block counts are recorded; an empty split is an error. Re-running preparation verifies packed-file hashes and reuses the existing pinned artifacts. Delete/recreate the output only when intentionally starting over.

## Offline correctness tests

```bash
python -m pytest -q tests
```

Tests construct tiny local Hugging Face Qwen2/GPT-2 models and a tokenizer; they need no model or dataset downloads. They compare chunked loss and gradients with the standard HF forward, verify output-only freezing and AdamW exclusion, inspect packing/replay/split behavior, run a complete tiny A→B experiment, verify the initial four-way head swap, and compare interrupted/resumed weights bit-for-bit.

## Files, references, and license

`cpt.py` is the entry point; the other Python files keep data preparation, training, diagnostics, and analysis readable. `configs/h100_pilot.json` is the recommended first run. `configs/h100_study.json` is the larger sweep. `VALIDATION.md` records what was tested here.

- [Qwen2.5-1.5B configuration, including tied embeddings](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json)
- [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
- [Hugging Face dataset streaming](https://huggingface.co/docs/datasets/stream)
- [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
- [CodeParrot-clean dataset](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
- [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763)
- [Spurious Forgetting in Continual Learning of Language Models](https://arxiv.org/abs/2501.13453)

Code: Apache-2.0; see `LICENSE`. Model and dataset terms remain their respective sources' terms. No model weights or source datasets are redistributed in this package.
