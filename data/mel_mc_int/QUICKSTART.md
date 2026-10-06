# Audio and music demo: complete walkthrough

This guide takes raw music or speech through preparation, training, continuation
generation, audio reconstruction, and the browser viewer. It targets the repaired
PR #910 pipeline, with **384 mel bands and 64 amplitude states**.

Run the commands from the **ReaLLM-Forge repository root**, in one Bash terminal,
unless a step says otherwise. Replace example source-file paths with your own.

| What you have | Where to start |
|---|---|
| Two or more recordings and no trained checkpoint | Follow steps 1–7. |
| One recording and no trained checkpoint | Follow steps 1–3, then the single-recording route in step 8. |
| A checkpoint trained with the repaired pipeline | Activate its environment, then use step 7 with its checkpoint directory. |

## 1. Install the repaired code and dependencies

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
ffprobe -version
```

Verify the GPU from the activated environment:

```bash
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable in this Python environment'
print('GPU:', torch.cuda.get_device_name(0))
PY
```
After dependencies are installed, check both entry points:

```bash
bash demos/mel_mc_int_music_pipeline.sh --help
bash data/mel_mc_int/demo_infer.sh --help
```

## 2. Put your audio files in two folders

The source files can live anywhere. This walkthrough uses a directory on your
internal drive, outside the Git checkout:

```bash
export AUDIO_DEMO_ROOT="$HOME/reallm-audio-demo"
mkdir -p "$AUDIO_DEMO_ROOT/recordings" "$AUDIO_DEMO_ROOT/prompts"

cp "/path/to/first-recording.wav" "$AUDIO_DEMO_ROOT/recordings/track01.wav"
cp "/path/to/second-recording.flac" "$AUDIO_DEMO_ROOT/recordings/track02.flac"
cp "/path/to/prompt.wav" "$AUDIO_DEMO_ROOT/prompts/prompt.wav"
```

For the one-recording route, copy only your first recording and use it as the
prompt in step 8. Do not copy the same recording twice to satisfy the folder
pipeline's two-file requirement. If you only have one prompt file and an
existing checkpoint, no training-recordings folder is needed for step 7.

| Location | Purpose |
|---|---|
| `$AUDIO_DEMO_ROOT/recordings/` | Recordings to split into training and validation sets |
| `$AUDIO_DEMO_ROOT/prompts/` | Audio used to start a continuation; this folder is not added to training |
| `$AUDIO_DEMO_ROOT/cache/music-v1/` | Encoded CSV cache, calibration, and selected-source list |
| `data/mel_music_pilot_v1/` inside the repository | Prepared binary datasets and manifests |
| `$AUDIO_DEMO_ROOT/runs/music-pilot-001/` | Checkpoint, saved manifest, and demo outputs |

Supported extensions are `.wav`, `.flac`, `.mp3`, `.m4a`, `.aac`, `.ogg`, and
`.opus`. Keep recordings directly inside the folder: discovery is not recursive.
The pipeline decodes, downmixes, and resamples them; manual CSV conversion is not
needed. Use at least two distinct recordings for the folder route. Start with a
few 30–60 second clips to check the workflow before processing a large corpus.

A prompt may also be a training recording if you just want to inspect behavior.
Use an unseen recording when assessing generalization. Keep a separate test
set if you plan to make repeated choices based on validation performance.

## 3. Save a small initial configuration

This configuration keeps the 384-band, 64-state audio format and uses a smaller
transformer for the first run: 4 layers, width 128, context 128, batch 1, and 200
training iterations. It is a workflow check, not a trained music model or a
measured guarantee of GPU memory use. A 128-frame context is 1.92 seconds at the
default 15 ms hop.

Save the settings so you can reuse them in another terminal:

```bash
cat > "$AUDIO_DEMO_ROOT/pilot.env" <<'ENV'
export AUDIO_DEMO_ROOT="${AUDIO_DEMO_ROOT:-$HOME/reallm-audio-demo}"
export MEL_MC_WORK_DIR="$AUDIO_DEMO_ROOT/cache/music-v1"
export MEL_MC_OUTPUT_ROOT=mel_music_pilot_v1
export MEL_MC_OUT_DIR="$AUDIO_DEMO_ROOT/runs/music-pilot-001"

export MEL_MC_DEVICE=cuda:0
export MEL_MC_DTYPE=bfloat16
export MEL_MC_COMPILE=0
export MEL_MC_TENSORBOARD=0

export MEL_MC_SAMPLE_RATE=48000
export MEL_MC_BANDS=384
export MEL_MC_LEVELS=64
export MEL_MC_HOP_MS=15
export MEL_MC_WIN_MS=60
export MEL_MC_N_FFT=8192
export MEL_MC_FMIN=10
export MEL_MC_FMAX=20000
export MEL_MC_TOP_DB=96
unset MEL_MC_REFERENCE_POWER

export MEL_MC_N_LAYER=4
export MEL_MC_N_EMBD=128
export MEL_MC_N_HEAD=4
export MEL_MC_N_KV_GROUP=4
export MEL_MC_QK_DIM=32
export MEL_MC_V_DIM=32
export MEL_MC_MLP_SIZE=512
export MEL_MC_BLOCK_SIZE=128
export MEL_MC_BATCH_SIZE=1
export MEL_MC_MAX_ITERS=200
export MEL_MC_EVAL_INTERVAL=50
export MEL_MC_EVAL_ITERS=5
export MEL_MC_LR=0.001
export MEL_MC_DROPOUT=0.0

export MEL_MC_TRAIN_RATIO=0.9
export MEL_MC_SEED=1337
export MEL_MC_TOP_K=1
export MEL_MC_TEMPERATURE=0.8
export MEL_MC_MAX_NEW_TOKENS=200
export MEL_MC_SKIP_ENCODE=0
export MEL_MC_SKIP_TRAIN=0
export MEL_MC_PREPARE_ONLY=0
ENV

source "$AUDIO_DEMO_ROOT/pilot.env"
```

For CPU, change the two device/dtype lines in that file to `cpu` and `float32`,
then source it again. Compilation is disabled for the first run to simplify
startup; it can be enabled for a later measured GPU experiment.

## 4. Process the recordings without starting training

For the two-or-more-recording route:

```bash
MEL_MC_PREPARE_ONLY=1 bash demos/mel_mc_int_music_pipeline.sh \
  "$AUDIO_DEMO_ROOT/recordings" \
  "$AUDIO_DEMO_ROOT/prompts/prompt.wav" \
  4.5
```

This selects whole recordings for training and validation, calibrates a shared
amplitude reference using training recordings, encodes the audio, checks the
containers, and writes aligned per-band datasets. Every recording must have at
least `block_size + 1` frames. At context 128 and 15 ms hop this is about 1.94
seconds; the suggested 30–60 second clips leave ample room.

The printed manifest path is the result of preparation. Inspect the split and
counts if desired:

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['MEL_MC_WORK_DIR'])
selected = json.loads((root / 'selected_sources.json').read_text())
manifest = json.loads((Path(os.environ['MEL_MC_OUT_DIR']) / 'mel_manifest.json').read_text())
print('Training recordings:', selected['train'])
print('Validation recordings:', selected['val'])
print('Train frames:', manifest['train_rows'], 'Validation frames:', manifest['val_rows'])
print('Bands:', len(manifest['columns']), 'States:', manifest['vocab_size'])
PY
```

`MEL_MC_PREPARE_ONLY=1` above applies to that invocation only. The saved setting
remains `0`. Leave `MEL_MC_SKIP_ENCODE=0`: matching cache entries are reused
automatically, and missing or changed entries can be rebuilt. Setting it to `1`
requires valid matching encoded caches and fails when they are unavailable.

## 5. Train, generate, and reconstruct audio

Run the same wrapper without the preparation-only override:

```bash
bash demos/mel_mc_int_music_pipeline.sh \
  "$AUDIO_DEMO_ROOT/recordings" \
  "$AUDIO_DEMO_ROOT/prompts/prompt.wav" \
  4.5
```

The arguments mean: training folder, prompt audio, and prompt cutoff in seconds.
This reuses the prepared data, trains, saves `ckpt.pt`, reloads it, conditions on
the observed prefix, generates frames, reconstructs audio, and prints the
viewer path. No separate call to the encoder or sampler is required.

With these settings, the last 1.92 seconds of available prompt frames condition
the model. Earlier prompt frames are still included in the exported audio.
`MEL_MC_MAX_NEW_TOKENS=200` means **200 mel frames**, approximately 3 seconds of
new audio at 15 ms per frame. Each frame contains all 384 band states.

The final output contains `ckpt.pt`, `mel_manifest.json`, and one new directory
under `mel_samples/sample-*/`. A completed run is expected to sound rough after
only 200 iterations. Compare codec reconstruction before attributing all audio
artifacts to the transformer.

The wrapper refuses to start fresh training if the output directory already
contains `ckpt.pt`. To generate again, use step 7. To train a new experiment,
choose a new output directory as shown in step 9.

## 6. Open the demo viewer and listen

At completion, the terminal prints `Viewer: /absolute/path/.../index.html`.
Open that HTML file in your browser. Alternatively, serve the sample directory
locally:

```bash
python -m http.server 8000 --bind 127.0.0.1 \
  --directory "$MEL_MC_OUT_DIR/mel_samples"
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/), choose the `sample-*`
directory named in the terminal output, and open `index.html`. This command
stays running; press Ctrl+C to stop the server. On a remote workstation, keep
the server bound to localhost and forward its port with SSH if needed.

| Output file | What to listen to or inspect |
|---|---|
| `original_prefix.wav` | The decoded source prefix before mel reconstruction |
| `codec_prompt.wav` | The prompt reconstructed through the mel codec alone |
| `generated.wav` | Reconstructed prompt plus generated continuation |
| `continuation.wav` | The generated section, trimmed at the reported frame boundary |
| `generated.csv` | Prompt and generated integer band states |
| `generated.mel.csv` | The same states with self-describing decoder metadata |
| `run.json` | Input path, sampling settings, timing, and conditioning information |
| `index.html` | Audio players and the command builder for another run |

The centered 60 ms analysis window has about 30 ms lookahead. Only fully observed
frames are passed to the model. The viewer reports the actual generation frame
boundary; inverse windows overlap around it, so it is not an exact waveform
splice at the requested cutoff.

The viewer is static. Choosing a file auditions it locally; it does not launch
training or generation. Enter the real filesystem path, adjust settings, and
copy the displayed command into a terminal at the repository root to make a
new sample. Open the new output viewer afterward.

## 7. Run the demo on another audio file using the saved checkpoint

No dataset processing or training is needed for another prompt:

```bash
MEL_MC_DEVICE=cuda:0 MEL_MC_DTYPE=bfloat16 \
MEL_MC_TOP_K=1 MEL_MC_TEMPERATURE=0.8 MEL_MC_SEED=1337 \
bash data/mel_mc_int/demo_infer.sh \
  "$MEL_MC_OUT_DIR" \
  "/absolute/path/to/another-song.mp3" \
  4.5 \
  200
```

The positional arguments are:

| Position | Meaning |
|---|---|
| 1 | Directory containing `ckpt.pt` and `mel_manifest.json` |
| 2 | Audio file used as the prompt |
| 3 | Seconds of the original audio available to the model |
| 4 | Number of new frames; optional, default 200 |

Use `cpu`/`float32` for a CPU run. For a checkpoint from a previous session,
replace `$MEL_MC_OUT_DIR` with its actual directory. Encoder settings come from
the saved manifest; do not change the band count or quantizer for inference.
The prepared dataset metadata referenced by the manifest must still exist in
the repository's `data/` directory.

Top-k 1 is greedy; temperature does not change which token wins in that mode.
To explore stochastic continuations, try `MEL_MC_TOP_K=8`, keep temperature at
`0.8`, and vary `MEL_MC_SEED`. Top-k must be between 1 and the saved vocabulary
size, here 64. There is no promise that more randomness improves quality.

For the folder wrapper, `MEL_MC_SKIP_TRAIN=1` also reuses the saved checkpoint,
but it still expects a folder containing supported audio files. The direct
`demo_infer.sh` command above is simpler when all you need is a new prompt.

## 8. Complete alternative when you have only one recording

Use steps 1–3 first. Then use the commands below instead of steps 4–5. With
context 128 and a 90/10 split, use at least a 30-second source. A short file's
validation tail may be too small after the guard gap.

Choose separate output paths and a fixed reference:

```bash
export MEL_MC_WORK_DIR="$AUDIO_DEMO_ROOT/cache/single-v1"
export MEL_MC_OUTPUT_ROOT=mel_single_pilot_v1
export MEL_MC_OUT_DIR="$AUDIO_DEMO_ROOT/runs/single-pilot-001"
export MEL_MC_REFERENCE_POWER=1.0

bash data/mel_mc_int/run.sh \
  "$AUDIO_DEMO_ROOT/recordings/track01.wav" \
  "$MEL_MC_OUTPUT_ROOT"
```

This helper **only prepares data**. It splits the recording temporally with a
full-window gap between training and validation. It uses the supplied fixed
reference rather than calibrating on the entire recording. This measures
within-recording prediction, not generalization to an unseen recording.

Train from that manifest using the same training helper as the folder pipeline:

```bash
python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path('data/mel_mc_int').resolve()))
import pipeline

manifest_path = Path('data') / os.environ['MEL_MC_OUTPUT_ROOT'] / 'manifest.json'
manifest = pipeline.read_manifest(manifest_path)
out = Path(os.environ['MEL_MC_OUT_DIR']).resolve()
if (out / 'ckpt.pt').exists():
    raise SystemExit('Choose a new MEL_MC_OUT_DIR, or run inference with the existing checkpoint.')
out.mkdir(parents=True, exist_ok=True)
pipeline.tools.atomic_json(out / 'mel_manifest.json', manifest)
pipeline.train(manifest, out, os.environ['MEL_MC_DEVICE'], os.environ['MEL_MC_DTYPE'])
PY

bash data/mel_mc_int/demo_infer.sh \
  "$MEL_MC_OUT_DIR" \
  "$AUDIO_DEMO_ROOT/recordings/track01.wav" \
  4.5 \
  200
```

Use step 6 to open the resulting viewer. In a new terminal, source `pilot.env`
and repeat the four `export` assignments at the start of this section to select
the single-recording paths again. The environment file itself still selects the
folder experiment. Source it again before returning to the folder route; it
also clears the single-recording reference override.

## 9. Start another training experiment or a new terminal

For the folder route in a new terminal:

```bash
cd /path/to/ReaLLM-Forge
source .venv-mel/bin/activate
source "$HOME/reallm-audio-demo/pilot.env"
```

Adjust the environment-file path if you chose a different root. To train a new,
longer experiment with the same data and model shape:

```bash
export MEL_MC_OUT_DIR="$AUDIO_DEMO_ROOT/runs/music-pilot-002"
export MEL_MC_MAX_ITERS=5000

bash demos/mel_mc_int_music_pipeline.sh \
  "$AUDIO_DEMO_ROOT/recordings" \
  "$AUDIO_DEMO_ROOT/prompts/prompt.wav" \
  4.5
```

This starts a **new training run** and reuses matching prepared data. It does not
resume `music-pilot-001`. `MEL_MC_SKIP_TRAIN=1` means inference only, not training
resume. The wrappers in this patch do not expose a training-resume option.

Increase context, model size, or batch size only after the small run succeeds.
Use a new `MEL_MC_OUT_DIR` for each new experiment. When changing
`MEL_MC_BLOCK_SIZE`, also choose a **new `MEL_MC_OUTPUT_ROOT`**, so preparation
publishes a manifest with the requested context setting. For example:

```bash
export MEL_MC_BLOCK_SIZE=380
export MEL_MC_OUTPUT_ROOT=mel_music_context380_v1
export MEL_MC_OUT_DIR="$AUDIO_DEMO_ROOT/runs/music-context380-001"
```

A 380-frame context is 5.7 seconds. Every folder recording must then have at least
381 frames, and the validation part of a single recording needs that many usable
frames too. Keep the other small-model settings initially and rerun the relevant
preparation/training route. The larger original script defaults have not been
verified for peak memory on a 24 GB RTX 4090.

## 10. Storage and troubleshooting

Keep active audio, caches, and checkpoints on the internal drive for the first
run. With roughly 50 GB free, start with a few clips and check space before a
large corpus. Use the external USB drive for occasional archives rather than
continuous cache traffic if you want to limit its activity.

```bash
du -sh "$AUDIO_DEMO_ROOT" "data/$MEL_MC_OUTPUT_ROOT"
df -h "$AUDIO_DEMO_ROOT" .
```

At 384 bands, 15 ms hop, and uint16 storage, the prepared binary streams use
approximately 184 MB per hour of source audio. Source files, encoded CSVs,
temporary encoding data, checkpoints, and reconstructed WAVs add to that.
Old cache entries, immutable dataset generations, and sample folders are retained.

Preserve the checkpoint directory and its referenced `data/.../datasets/ID/`
directory together. Inference reads the per-band metadata, so copying only
`ckpt.pt` is insufficient. Keep `mel_manifest.json` beside the checkpoint.
Training logs may also be written to the repository's `csv_logs/` directory.

| Symptom | What to do |
|---|---|
| `pipeline.py` or `--help` entry point is missing | Apply the appropriate implementation patch first. |
| `ModuleNotFoundError` | Activate the environment used for installation; wrappers invoke its `python3`. |
| CUDA is unavailable | Run the Torch check in step 1; use a matching CUDA build/driver, or select CPU and float32. |
| No supported audio files | Put supported files directly inside the input folder and check the path. |
| Folder training needs two recordings | Use two distinct recordings or follow step 8. |
| A recording needs more usable frames | Use a longer recording or reduce context and choose a fresh dataset root. |
| No valid encoded cache | Set `MEL_MC_SKIP_ENCODE=0` and rerun preparation. |
| A checkpoint already exists | Use `demo_infer.sh` for sampling, or choose a new training output directory. |
| Legacy manifest or encoder changed | Use the matching original code/data for that checkpoint, or re-prepare and retrain under the repaired contract. |
| Checkpoint/manifest order or vocabulary mismatch | Restore the manifest and dataset metadata belonging to that checkpoint. |
| CUDA out of memory | Start with batch 1, compilation off, and the small model above; reduce model size or context further if needed. |
| Prompt is too short | Provide enough audio for at least one fully observed analysis frame; a few seconds is practical. |
| Browser file selection does not generate audio | The page auditions files and builds commands; run the command in the terminal. |
| Codec-only audio already sounds poor | Inspect the representation/reconstruction before judging generation quality. |

The earlier patch passed 16 regressions and a real tiny CPU
train/checkpoint/sample/decode run. To rerun those checks, install Node 22 for the
viewer quoting regression and run:

```bash
python -m unittest discover -s tests -p test_mel_mc_int.py
python tests/smoke_mel_mc_int.py
```

These checks establish integration. They do not establish music quality,
long-run convergence, or full-size GPU memory use.
