# Mel integer multicontext

`data/mel_mc_int` bridges the self-describing mel CSVs from
`data/mel_spectrogram` with the integer CSV multicontext pattern used by
`data/csv_mc_int`. Each mel band (`mel_000_q`, `mel_001_q`, ...) becomes one
regular nanoGPT multicontext dataset with token IDs equal to the quantized mel
state.

## Defaults

`run.sh` intentionally mirrors the high-fidelity settings in
`data/mel_spectrogram/run.sh`:

- preset `max`
- sample rate `48000`
- `fmin=10`, `fmax=20000`
- `384` mel columns per timestep
- `64` states per column
- `15 ms` timestep and `60 ms` window
- `n_fft=8192`, `top_db=96`
- `reference-mode=file_percentile`

## Build a dataset

```bash
bash data/mel_mc_int/run.sh path/to/audio.wav mel_mc_int
```

This writes the intermediate mel CSV/PNG under `data/mel_mc_int/mel_out/` and
the multicontext dataset under `data/mel_mc_int/` by default. The manifest lists
all mel-band dataset names for training or inference:

```bash
python3 - <<'PY'
import json
m = json.load(open('data/mel_mc_int/manifest.json'))
print(' '.join(m['multicontext_datasets']))
PY
```

## Inference and continuation demo

After training a multicontext model on the generated manifest, run:

```bash
bash data/mel_mc_int/demo_infer.sh out/your_run path/to/audio.wav 4.5 200
```

The demo encodes the selected audio, cuts the prompt at `4.5` seconds, asks
`sample.py` to continue all mel-band contexts, wraps the generated CSV with the
reference mel metadata, reconstructs `generated.wav`, and writes a small
`index.html` viewer. The viewer includes a browser-side file picker and cutoff
field for auditioning/selecting the next source file before re-running the demo
command.

## Whole-folder music pipeline

For a single command that encodes a directory of music files, concatenates the
resulting mel-state CSV rows into one training CSV, prepares multicontext data,
trains, and runs continuation inference, use:

```bash
bash demos/mel_mc_int_music_pipeline.sh path/to/music_dir path/to/prompt.wav 10.0
```

The script writes per-file mel CSVs to
`data/mel_mc_int/music_pipeline_out/encoded/`, concatenates their `mel_*_q`
values into `all_music.max.mel.csv`, prepares `data/mel_mc_int_music/`, trains
into `out/mel_mc_int_music`, and then calls `data/mel_mc_int/demo_infer.sh`.
Set `MEL_MC_SKIP_ENCODE=1` or `MEL_MC_SKIP_TRAIN=1` to reuse existing encoded
CSVs or checkpoints while iterating. The pipeline disables TensorBoard by default
(`MEL_MC_TENSORBOARD=0`) so training does not import TensorFlow/TensorBoard in
environments with incompatible NumPy/TensorFlow wheels; set
`MEL_MC_TENSORBOARD=1` if your environment supports it.

Both concatenation and dataset preparation are streaming operations. They scan
the CSV data in multiple passes for validation and metadata, but never hold the
combined state matrix in memory. Consequently, peak memory is bounded by one
CSV row during concatenation and by `--buffer_rows` per mel band during
preparation (4096 rows by default, approximately 3 MiB for 384 bands of
16-bit states). This is suitable for large music folders; the tradeoff is extra
sequential disk reads. Make sure the output filesystem has room for the combined
CSV plus the train/validation binaries. The folder pipeline also passes inputs
through a newline-delimited list file instead of placing every filename on the
command line, avoiding the operating system's command-argument size limit for
very large collections.
