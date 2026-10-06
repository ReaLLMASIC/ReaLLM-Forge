# SLERP Context Lab v0.3.0 — whole-document summarization

A Hugging Face comparison of **summary quality, latency, GPU/RAM use and retained
state beyond a small model's context length**. This adds summarization to the v0.2
full-weight retrieval lab.

Default: `HuggingFaceTB/SmolLM2-360M-Instruct` (native context 8,192 tokens), whole
GovReport documents of 512–32,767 tokens, and a 2,048-token working window. Both
limits are reported separately. Native position limits are never enlarged.

**Every learned arm updates the entire backbone plus memory. No LoRA.** Default
loss is supervised reference-summary cross-entropy. Optional full-source teacher
KL is distillation, not RL.

## Comparison

| Method | Information carried forward | Entire source processed? |
|---|---|---|
| `native_full` | Ordinary HF transformer KV cache | Only when complete request fits 8K |
| `head`, `tail`, `head_tail` | Selected text within 2K | No; tokens seen are logged |
| `rolling` | Running text summary | Yes |
| `map_reduce` | Passage summaries, then hierarchical summaries | Yes |
| `local` | Sliding KV; no separate memory | Yes, sequentially |
| `nlerp`, `slerp` | Learned spherical slots plus sliding KV | Yes |
| `rmt` | Recurrent hidden-state tokens | Yes |
| `delta` | Corrective associative matrix plus sliding KV | Yes |

All use greedy decoding, identical whole documents/references and a 512-token
final budget. Text compression has a 192-token intermediate budget; **every
intermediate pass counts toward latency and token cost**. References never enter
generation. RMT/delta are research implementations, not paper/kernel reproductions.

## Start on Linux RTX 4090

Use Python 3.10–3.12 and a compatible NVIDIA driver. Extract this version separately:

```bash
unzip slerp-context-lab-v0.3.0.zip
cd slerp-context-lab-v0.3.0
bash setup.sh
bash run.sh download --config configs/summary-smol-backbone.json
bash summary.sh prepare --config configs/summary-smol.json
bash summary.sh evaluate --method native_full --max-source-tokens 6144 \
  --limit 5 --out reports/native-summary-screen.jsonl --device cuda
```

Inspect native summaries for coverage and invented claims. Then profile full
training at the maximum configured source/target lengths, including AdamW state:

```bash
bash scripts/summary_profile_4090.sh
```

Profiles must leave at least 2 GiB GPU headroom. If one fails, lower
`train_source_limit` in a new config and profile again in a fresh directory.
Bounded inference state does **not** imply bounded full-backpropagation memory.
Start with 4K/8K training if needed and evaluate length generalization above 8K.

Train a shared summarization warmup and inspect its native outputs:

```bash
bash summary.sh train --config configs/summary-smol-warmup.json \
  --out runs/summary-smol-seed17-shared --device cuda
bash summary.sh evaluate --method native_full --max-source-tokens 6144 \
  --checkpoint runs/summary-smol-seed17-shared \
  --out reports/warm-summary-screen.jsonl --device cuda
```

If native quality is useful, run the matched comparison:

```bash
bash scripts/summary_compare_4090.sh
```

This reuses the warmup, initializes each recurrent arm from the same backbone,
trains sequentially, evaluates validation and writes reports under
`reports/summary-smol-seed17/plots/`. Each arm has a 2M processed-token cap or
three passes, whichever comes first. This is a pilot, not a convergence claim.
Set `METHODS="local slerp rmt"` on both scripts for a smaller first comparison.
If native summaries remain poor, improve the backbone/data/warmup first.

## Results

Outputs include per-document ROUGE-1/2/L/Lsum, repetitions, output-limit hits,
total latency, time to the first final-answer token, all input/output token counts,
GPU allocated/reserved peaks, sampled process RSS and persistent cache/state.
PNG/PDF figures show quality, latency and state versus length and quality versus
latency/GPU memory. CSV/JSON data, paired document-bootstrap intervals and blind
human-review sheets accompany them.

Optional HF contextual token matching covers long reference summaries in blocks:

```bash
bash summary.sh semantic reports/summary-smol-seed17/*.jsonl \
  --out reports/summary-smol-seed17-semantic.jsonl
bash summary.sh report reports/summary-smol-seed17/*.jsonl \
  --semantic-file reports/summary-smol-seed17-semantic.jsonl \
  --out reports/summary-smol-seed17/plots-semantic
```

This custom chunked similarity metric is **not canonical BERTScore or a factuality
score**. Human review must check main findings, cross-section relationships and
qualifications against the source. Choose settings on validation; only then use
`--split test`. See `docs/SUMMARIZATION.md` for exact methodology and commands.

## Resources and edge deployment

Full training uses FP32 weights/AdamW with BF16 CUDA compute, batch one,
accumulation and checkpointing. Actual 4090 fit must pass the supplied profiles.
The inherited internal-storage policy caps the project at 40 GB and keeps 8 GB
free after a planned checkpoint write, within the approximately 50 GB budget.
Active files stay internal; no disk optimizer offload or automatic external sync.

A nominal 360M backbone needs about 1.44 GB FP32 weights + 2.88 GB Adam moments,
before memory modules/metadata. Atomic replacement needs both old and new copies.
Six resumable stages plus dependencies/cache/replacement can approach the cap.
Explicitly discard each completed arm's optimizer after evaluation if needed:

```bash
FINALIZE_AFTER_EVAL=1 bash scripts/summary_compare_4090.sh
bash scripts/archive_run.sh runs/summary-smol-seed17-slerp \
  /media/YOUR_EXTERNAL_DRIVE/context-archives 20
```

Finalization retains weights but removes exact-resume ability. Archiving is manual,
sequential and rate-limited; it never deletes the local run. Drive temperature is
not guaranteed. The archive is for occasional transfers, not active training.

This is a HF/PyTorch research runtime. Orin deployment, quantization, fused kernels
and energy measurement are later steps. `setup.sh` targets the Linux 4090 host,
not Jetson. Small state alone does not establish low latency or energy use.

## Validation and alternatives

See `docs/VALIDATION.md` for CPU tests and real HF integration. No 4090 training,
trained summarization quality, or Orin performance result is claimed.

`configs/summary-arxiv.json` selects paper → abstract summarization with separate
data. `configs/summary-hunyuan.json` supports the earlier 0.5B model; 32K documents
exceed its working window, not its 262K native limit. Alternatives need matching
warmup configs, profiling and native quality checks.

The prior retrieval workflow remains in `docs/RETRIEVAL-v0.2.0.md`.
Commands: `bash summary.sh --help`, `bash run.sh test`, `bash run.sh verify`.
