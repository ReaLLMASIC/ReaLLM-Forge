# Task fine-tuning and capability retention

The original CPT runner measures language-model loss and output geometry. This extension asks a more concrete question:

**After learning to solve math word problems, does a model with a frozen pretrained LM head retain more of its original capabilities than a model trained normally, at comparable math performance?**

The default is still Qwen2.5-1.5B on one 80 GB H100. All runs use full-parameter AdamW, BF16 compute, checkpointing, and chunked full-vocabulary loss. The memory preflight and optimizer-state guard from the CPT runner apply.

## Metrics already in the CPT package

| Metric | What it measures |
|---|---|
| A/B held-out cross-entropy and perplexity | Retention of the old text distribution versus adaptation to the new one |
| Token accuracy | Correct next-token argmax |
| Target logit margin | Correct target logit minus the strongest competitor |
| A-common/B-rare token loss | Retention of tokens that receive little new-domain supervision |
| Angular / norm / relative head drift | Movement of output vectors |
| Four-way head swaps | Effects of pairing old/new heads with old/new hidden states |
| Forgetting versus new-domain loss gain | Whether an apparent retention benefit is just slower learning |

Those are not task benchmarks like GSM8K or HellaSwag. The new suite adds the following.

## New task and retention benchmarks

| Dataset | Role | Main reported metric |
|---|---|---|
| GSM8K | Math task being learned | Greedy final numerical-answer accuracy, plus strict `####` accuracy |
| WikiText-103 | General language retention | Held-out cross-entropy/perplexity, token accuracy, target margin |
| HellaSwag | Commonsense continuation retention | Harness `acc_norm` |
| ARC Easy | Science-question retention | Harness `acc_norm` |
| ARC Challenge | Harder science-question retention | Harness `acc_norm` |
| PIQA | Physical commonsense retention | Harness `acc_norm` |
| BoolQ | Reading/yes-no reasoning retention | Harness `acc` |

The five standard retention tasks use the **EleutherAI LM Evaluation Harness Python API**, with **zero-shot evaluation, no chat template, batch size 1**, and the same configured maximum sequence length for every model. The task's own harness definition selects its evaluation partition; some public tasks expose validation labels rather than test labels. Their data never enters the math training set. Scores should be compared within this fixed protocol; published model cards may use different shot counts, prompts, or normalization.

GSM8K evaluation in this package uses an explicit fixed plain-text prompt and custom transparent numerical extraction. It is **not the harness's few-shot GSM8K protocol**. We report both strict `####`-marked accuracy and relaxed numerical accuracy so formatting changes remain visible. We also report the valid-format rate and the fraction of generations hitting the token cap. Relaxed scoring uses the marked number if present, otherwise the last number in the continuation; inspect saved predictions because an intermediate number can occasionally be mistaken for a final answer.

The pilot evaluates deterministic subsets, not full benchmark scores. It is useful for debugging and checking the direction of change; modest differences on 200 examples are noisy. The study config removes those example limits. Report actual sample counts, any length-filtered GSM8K examples, and the selected harness protocol with results.

## Quick start

On the H100 machine, use the same CUDA-enabled environment as the CPT package:

```bash
cd hf-frozen-head-cpt
python -m pip install -r requirements-benchmarks.txt
bash run_task_h100.sh configs/h100_math_pilot.json
```

This runs:

1. A two-update synthetic VRAM preflight, discarding those weights.
2. Tokenization and dataset splitting through Hugging Face APIs.
3. Evaluation and saving of the untouched pretrained baseline.
4. **300 optimizer steps of ordinary math SFT** and **300 steps with only the LM head frozen**, starting from that same baseline.
5. Held-out math evaluation, WikiText retention metrics, head diagnostics, and plots.
6. Standard retention benchmarks on the baseline and both final checkpoints, in separate GPU processes.

The pilot uses one seed, one LR (3e-5), 100 math-validation examples, 200 math-test examples, and up to 200 examples per standard retention benchmark. At batch size 1 with accumulation 16, each arm sees **4,800 math training examples**, cycling only if the selected training set is smaller. Supervised-token counts depend on solution length and are recorded. The old-language retention pool is never mixed into the default math SFT.

Task-only run without installing the optional harness:

```bash
python cpt.py preflight --config configs/h100_math_pilot.json
python task_sft.py run --config configs/h100_math_pilot.json
```

Add the standard benchmarks afterward:

```bash
python -m pip install -r requirements-benchmarks.txt
python benchmarks.py suite --config configs/h100_math_pilot.json

# Run full benchmark sets on the SAME saved models, in a separate result directory.
python benchmarks.py suite --config configs/h100_math_pilot.json --limit 0
```

The compatible harness dependency range is resolved by pip together with the existing Transformers pin. Keep the resulting environment unchanged across comparisons: the runner records the resolved harness version and hashes its task-definition code, and rejects an environment/protocol change in an existing benchmark result directory. A separate benchmark environment is also possible as long as it can read the checkpoint, tokenizer, and this package.

No GPU training or full external benchmark run was performed in the authoring environment. The native training/scoring path is covered by offline tests; the optional harness integration follows its documented Python API and needs its dependencies/datasets on your machine.

## Larger study

```bash
bash run_task_h100.sh configs/h100_math_study.json
```

This is **4 conditions × 2 learning rates × 3 seeds = 24 runs**, each with 1,200 optimizer steps. The four conditions are:

- `full`: train the backbone, input embeddings, and output head.
- `freeze_head`: train backbone/input embeddings while freezing the output head.
- `freeze_embed`: freeze input embeddings while training backbone/output head.
- `freeze_both`: freeze input embeddings and output head while training the backbone.

All four untie initially identical input/output weights before SFT. The common baseline remains the original tied pretrained model. Untying preserves its initial predictions but lets the interventions be identified separately. Batches, schedules, and initialization are paired by seed. Prompt tokens and padding receive label `-100`; only solution tokens and the terminating EOS are supervised. Accumulation divides by the **total number of supervised tokens across all microbatches**, so variable solution lengths do not accidentally change the objective.

The study uses the whole selected GSM8K training pool across repeated seeded permutations, and all eligible validation/test examples. Training-example counts and response-token counts reveal repeat exposure. A 1,200-step run at 16 examples/step can traverse the GSM8K training set several times; validation accuracy and format/cap rates should guide the interpretation of late checkpoints. This is a controllable SFT experiment, not an assertion that this schedule is optimal for math.

Optional modes already supported are `slow_head`, `freeze_head_norm`, `replay`, and the tied controls. SFT replay uses A-training microbatches on the configured schedule. Since responses have variable lengths, `replay_fraction=0.1` means **10% of microbatches**, not exactly 10% of supervised tokens. The actual replay/response counts are logged. Compare replay at both total compute and task exposure when drawing conclusions.

## Dataset separation

For GSM8K, a deterministic question hash reserves 10% of the official training split for validation. The official test split remains outside training and checkpoint selection. Exact duplicate test questions are excluded from training/validation. If you supply explicit local validation data, that split is honored instead of hash splitting. Train/validation/test examples are deterministically ordered and optionally capped for the pilot. Model and dataset Hub revisions are recorded.

Examples whose combined prompt and full reference solution exceed the configured sequence length are excluded rather than having their final answer silently truncated. `data/task_manifest.json` reports exclusions and resulting split counts. Thus a length-restricted run must be labeled as such, rather than assumed to cover every official test problem. The supplied default context is 1024 training tokens.

There is **no preliminary WikiText consolidation** in this task experiment. `baseline/model` is the untouched pretrained checkpoint. Retention is measured relative to its original performance, addressing the question of task specialization causing general-capability loss. Historical contamination of the pretrained model's training data cannot be ruled out by these experiment splits.

The prompt/response format is identical in training and evaluation:

```text
Solve the problem. Show your work and end with #### followed by the final numerical answer.
Question: ...
Answer:
```

Use a base model with this plain-text setup. An instruction-tuned model with its native chat template is a useful separate experiment, but needs consistent chat-format training and evaluation; this package does not silently apply a chat template to one side only.

## What counts as evidence?

For a retained accuracy benchmark:

```text
forgetting_pp = 100 × (baseline_accuracy - final_accuracy)
```

For retained language modeling, use `final_loss - baseline_loss`. For the newly trained task, use the increase in held-out math accuracy. Negative forgetting is positive transfer, and should not be clamped to zero.

Evidence favoring frozen heads would be **similar math accuracy with smaller old-task accuracy drops and smaller WikiText loss increases**, repeated across seeds and LRs. If frozen-head training retains more but fails to learn math as well, the result is a stability/plasticity trade-off, not yet a better forgetting mitigation. Inspect each benchmark separately rather than letting a macro average hide a large loss on one skill.

The generated plot shows WikiText forgetting against actual math-validation accuracy. Accuracy is discrete and noisy, so the report selects **actual first-crossing checkpoints** for accuracy targets rather than interpolating an accuracy value. By default targets are +5, +10, and +15 percentage points above each run's common baseline. Report the achieved accuracy and overshoot; a single noisy threshold crossing is exploratory evidence.

For a confirmatory comparison, choose an achievable **absolute validation accuracy** above the baseline, then copy the config to a new output directory and set:

```json
"sft": {
  "stop_at_validation_accuracy": 0.50
}
```

Merge this field into the existing `sft` object. The runner stops at the first evaluated crossing and performs its end-only test evaluation. If the pretrained baseline already meets the target, it rejects that target. A run that never reaches it completes its budget and must be reported as not reaching the target. Use denser evaluations near the target, several seeds, and the untouched test sets for confirmation. Choose LRs using validation rather than selecting from final retention benchmarks after seeing their scores.

The standard retention benchmarks run before/after SFT, not at every training checkpoint. Repeatedly using those final benchmark scores for model selection would turn them into development sets; reserve a separate confirmatory set if you iterate on them heavily.

## Files to inspect and share

| Output | Contents |
|---|---|
| `baseline/validation.json`, `baseline/test.json` | Pretrained math and language scores |
| `task_analysis/task_retention.png` / `.svg` | Math acquisition and language-retention trade-off |
| `task_analysis/final_test.csv` | Math gains, strict accuracy, format rate, cap rate, and old-language loss change |
| `task_analysis/validation_accuracy_crossings.csv` | Actual checkpoints reaching common math targets |
| `benchmarks/limit_200/retention.csv` | Pilot standard-task accuracy changes in percentage points |
| `benchmarks/full/retention.csv` | Full standard-task accuracy changes |
| `benchmarks/.../baseline.json` and `seed_*.json` | Harness configuration, task versions, per-example records, and raw scores |
| `seed_*/.../metrics.jsonl` | Task/language trajectories and head-swap/geometry diagnostics |
| `seed_*/.../test_predictions.jsonl` | Generated answers and numerical scoring decisions |

Benchmark comparisons verify identical per-task document and prompt fingerprints. Where the harness exposes per-example accuracy, the CSV also reports old-correct→new-wrong and old-wrong→new-correct counts. They are paired observations; do not compute an unpaired significance test as if each model saw different questions. Seed variation and benchmark sampling uncertainty are separate sources of uncertainty.

Resume by repeating the same task command; model, optimizer, RNG, data position, and token counters are restored. The original CPT commands and outputs remain available. CPU resume equality and the solution-only objective are tested.

## Sources

- [GSM8K dataset and reference solutions](https://huggingface.co/datasets/openai/gsm8k)
- [EleutherAI harness Python API](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)
- [HellaSwag task definition](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/hellaswag.yaml)
- [ARC dataset](https://huggingface.co/datasets/allenai/ai2_arc)
- [PIQA task definition](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/piqa.yaml)

Code remains Apache-2.0. Models, datasets, and evaluation-harness dependencies retain their own licenses.
