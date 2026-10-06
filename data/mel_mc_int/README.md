# Mel integer multicontext

For setup, audio-file placement, processing, training, and the browser demo,
follow the [complete audio walkthrough](QUICKSTART.md). It includes both folder
and single-recording training, plus inference with an existing checkpoint.

Each mel band is an aligned categorical stream. The existing transformer sums
per-band learned embeddings and predicts each next-frame band with its own tied
head. This adapter is a **full learned-table baseline**, with averaged channel
cross-entropy; it does not enable SLERP or numerical regression.

## Folder pipeline

Install the repository's training dependencies and FFmpeg/FFprobe. Run:

```bash
bash demos/mel_mc_int_music_pipeline.sh /path/to/recordings /path/to/prompt.wav 10.0
```

Use at least two recordings, each with at least `block_size + 1` encoded frames.
Defaults are 384 bands, 64 states, 48 kHz, 15 ms hop, 60 ms centered Hann window,
8192 FFT, 10–20000 Hz, and 96 dB quantization span. A 380-frame context is 5.7 s.
A longer prompt is exported in full, but only its last context window conditions
sampling. Recordings are mono after the encoder's existing downmix.

Before encoding, the pipeline makes a seeded recording-level train/validation
split. It calibrates one reference on **training recordings only** (the maximum
of their individual 99.5th-percentile mel powers), then freezes that reference
for all recordings and prompts. `MEL_MC_REFERENCE_POWER` supplies an explicit
fixed value instead. Calibration processes one recording at a time in an extra
pass; it is not an exact pooled corpus percentile. The existing encoder still
holds each recording in memory. Global DC removal is disabled with `--keep-dc`.
The defaults are a demo configuration, not a measured 4090 memory guarantee.

The cache is keyed by source path/content, encoder code, and transform settings.
Only selected sources enter the run; unrelated cache files are ignored. Changing
a source, split, or transform invalidates affected cached output. Exact duplicate
source bytes are rejected before splitting. Transcoded copies of the same
recording still need to be deduplicated by the dataset owner.

Prepared datasets are immutable, content-addressed generations under
`data/OUTPUT_ROOT/datasets/ID/`. Their manifests retain recording ranges. The
multicontext loader opts into these ranges when metadata provides them and
samples shared channel indices wholly within one recording. Existing datasets
without range metadata retain their previous sampling path.

The pipeline saves `mel_manifest.json` next to `ckpt.pt` and always saves the
final checkpoint, even when a short run does not improve its first validation
loss. Dataset metadata and order are checked against the checkpoint at inference. Repeated preparation
reuses an unchanged generation after checking its hashes. Old generations and
caches are retained because checkpoints may reference them; delete only those
no longer needed. No redundant concatenated corpus CSV is made by this pipeline.
Preparation uses bounded per-batch memory; binary storage retains the existing
uint16/uint32 training convention (64-state streams use uint16).

## Settings

All paths supplied through arguments or environment overrides resolve relative
to the calling directory, except `MEL_MC_OUTPUT_ROOT`, which is under repo
`data/`. The default work/checkpoint paths are repository-relative.

| Environment setting | Default | Meaning |
|---|---|---|
| `MEL_MC_OUTPUT_ROOT` | `mel_mc_int_music` | Dataset root under `data/` |
| `MEL_MC_WORK_DIR` | `data/mel_mc_int/music_pipeline_out` | Encoder cache/calibration |
| `MEL_MC_OUT_DIR` | `out/mel_mc_int_music` | Checkpoint and saved manifest |
| `MEL_MC_MAX_ITERS` | `1000` | Training iterations |
| `MEL_MC_MAX_NEW_TOKENS` | `200` | Generated mel frames |
| `MEL_MC_DEVICE` / `MEL_MC_DTYPE` | `cuda:0` / `bfloat16` | Train/sample; device also reaches audio tools |
| `MEL_MC_SKIP_ENCODE` | `0` | `1` requires matching, validated cache entries |
| `MEL_MC_SKIP_TRAIN` | `0` | `1` uses the checkpoint's saved manifest and runs inference only |
| `MEL_MC_PREPARE_ONLY` | `0` | Stop after validated dataset creation |
| `MEL_MC_COMPILE` / `MEL_MC_TENSORBOARD` | `1` / `0` | Training compilation/logging |
| `MEL_MC_TRAIN_RATIO` / `MEL_MC_SEED` | `0.9` / `1337` | Recording split; seed also reaches train/sample |
| `MEL_MC_REFERENCE_POWER` | train-calibrated | Explicit positive fixed reference |
| `MEL_MC_TOP_K` / `MEL_MC_TEMPERATURE` | `1` / `0.8` | One deliberate sampling setting; top-k 1 is greedy |
| `MEL_MC_N_LAYER` / `MEL_MC_N_EMBD` | `12` / `500` | Transformer shape |
| `MEL_MC_N_HEAD` / `MEL_MC_N_KV_GROUP` | `12` / head count | Attention heads/groups |
| `MEL_MC_QK_DIM` / `MEL_MC_V_DIM` | `120` / `120` | Per-head dimensions |
| `MEL_MC_MLP_SIZE` | `1536` | MLP width |
| `MEL_MC_BLOCK_SIZE` / `MEL_MC_BATCH_SIZE` | `380` / `12` | Context and batch |
| `MEL_MC_EVAL_INTERVAL` / `MEL_MC_EVAL_ITERS` | `100` / `10` | Validation schedule |
| `MEL_MC_LR` / `MEL_MC_DROPOUT` | `0.001` / `0.0` | Optimization |

Encoder overrides are `MEL_MC_SAMPLE_RATE`, `BANDS`, `LEVELS`, `HOP_MS`, `WIN_MS`,
`N_FFT`, `FMIN`, `FMAX`, and `TOP_DB`, each with the `MEL_MC_` prefix. Their
values are saved and reused for inference. They must satisfy the existing
encoder's supported ranges. Runtime changes to encoder settings do not silently
reinterpret an existing checkpoint. Choose a new output directory to retrain;
a preexisting checkpoint is not overwritten by a fresh training invocation.

## Single-recording preparation

```bash
MEL_MC_DEVICE=cpu bash data/mel_mc_int/run.sh /path/to/audio.wav mel_single
```

This uses an explicit fixed reference (`MEL_MC_REFERENCE_POWER`, default 1.0),
then makes a temporal split with a full-window guard gap. It reports an error
if either side cannot form a batch. This is within-recording evaluation, not
held-out-recording generalization. The single-file helper prepares data only;
train using the returned manifest's ordered `multicontext_datasets`, and pass
that manifest explicitly to inference. The low-level `prepare` subcommand
validates fixed-reference CSVs, but inference requires the encoder settings
saved by the audio pipeline.

The low-level `concat-csv` utility validates every source checksum and full
transform signature, including reference power, before emitting a combined
container. Use `--input_json PATH` (a JSON array of paths) for arbitrary
filenames. Training uses individual recordings directly so no cross-recording
windows are introduced by concatenation.

## Prefix-safe continuation

```bash
MEL_MC_DEVICE=cpu MEL_MC_DTYPE=float32 \
  bash data/mel_mc_int/demo_infer.sh out/mel_mc_int_music /path/to/prompt.wav 4.5 200
# For a separately trained run:
# ... --manifest /absolute/path/to/data/mel_single/manifest.json
```

The wrapper trims decoded audio at the source sample rate **before** resampling,
statistics, or encoding. It uses the saved fixed reference and keeps only
frames whose entire right window is observed. A centered 60 ms window has
approximately 30 ms lookahead relative to its frame-center timestamp; this
latency is recorded, not hidden by a change of codec convention. Lossy source
codec decoding has its own codec framing; prefix invariance is defined on the
decoded waveform. Very short prefixes are rejected. Cutoffs beyond the source
length are clipped, and the available duration is reported in `run.json`.

Output under `OUT_DIR/mel_samples/sample-*/` includes the original decoded
prefix, codec-only reconstruction, reconstructed prompt plus generation,
continuation-only audio, self-describing generated CSV, run settings, and a
static viewer. Frame boundary and inverse-window overlap are labeled. Selecting
a file in the viewer auditions it; the filesystem path must be entered
explicitly. Copied commands quote every argument without shell expansion.

**Migration:** previously prepared PR #910 data used future-dependent,
per-recording normalization. Re-prepare and retrain for this fixed-reference
contract. The new inference wrapper deliberately refuses a legacy manifest
without the saved encoder settings. Retain old data for old checkpoints.

## Validation

```bash
python -m unittest discover -s tests -p test_mel_mc_int.py
python tests/smoke_mel_mc_int.py
```

Regression tests use NumPy and FFmpeg; Node executes the viewer quoting test.
The real smoke test requires the repository's CPU training dependencies and
checks a tiny train/checkpoint reload/sample/decode run, including checkpoint
reuse and execution from outside the repository. A targeted GitHub Actions job
runs both. It does not establish audio quality or full-size 4090 VRAM usage.
Before long GPU runs, use two short real recordings, reduce batch size as needed,
and record measured peak VRAM, throughput, loss, and successful audio output.
