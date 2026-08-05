# Standalone bounded mel-token roundtrips

This package has two Python scripts and one repeatable demo:

- `audio_to_token_mel.py` decodes any non-DRM audio format supported by the
  installed FFmpeg build and writes a self-describing CSV, a self-describing
  QMel-PNG, or both. It can optionally apply a bounded, ladder-specific
  automatic gain controller before mel analysis.
- `token_mel_to_audio.py` reconstructs a best-effort mono PCM16 WAV from
  **one CSV or one PNG**. It does not read a JSON sidecar.
- `demo.sh` generates both containers and reconstructs each to a distinct WAV.

The CSV has one data row per acoustic timestep. The PNG stores the same
integer state matrix exactly, not a screenshot or an interpolated plot.
JSON is now optional diagnostics only (`--write-json`) and is never needed
for reconstruction.

The forward CPU path uses a memory-mapped FFmpeg decode and batched NumPy
operations, so the input and full floating-point spectrogram do not need to
fit in RAM. A CUDA-enabled PyTorch installation can accelerate the forward
FFT/mel projection and inverse mel/phase iterations.

## Install

Python 3.10 or later:

```bash
python -m pip install numpy pillow
```

Install the `ffmpeg` executable with the operating system package manager.
Pillow is needed only when reading or writing PNG. PyTorch is optional; install
a CUDA build only if GPU acceleration is wanted. Matplotlib, librosa, and
soundfile are not required. Keep the two Python scripts in the same directory;
the inverse imports the shared transform definitions from the forward script.

Verify this five-tier release:

```bash
python3 audio_to_token_mel.py --version
# audio_to_token_mel.py 2.3.0
```

Run the demo with an optional preset and device:

```bash
bash demo.sh recording.wav
bash demo.sh recording.flac ultra cuda
bash demo.sh recording.wav max cpu
bash demo.sh recording.wav medium cuda mel_out \
  --columns-per-timestep 32 --states-per-column 48
bash demo.sh recording.wav medium cpu mel_out \
  --agc --timestep-ms 12.5
bash demo.sh recording.wav high cuda mel_out \
  --agc --columns-per-timestep 64 --states-per-column 128 \
  --timestep-ms 12.5
```

## CLI: audio to standalone CSV or PNG

Write both standalone forms (the default):

```bash
python audio_to_token_mel.py recording.m4a \
  --preset medium \
  --output-format both \
  --output-dir mel_out
```

Write only CSV:

```bash
python audio_to_token_mel.py recording.flac \
  --preset medium \
  --output-format csv \
  --output-dir mel_out
```

Write only QMel-PNG:

```bash
python audio_to_token_mel.py recording.mp3 \
  --preset medium \
  --output-format png \
  --output-dir mel_out
```

Generate all five quality levels:

```bash
python audio_to_token_mel.py interview.mp4 \
  --preset all \
  --output-format both \
  --output-dir mel_out
```

Use CUDA when a CUDA-enabled PyTorch build is installed, or explicitly keep
RAM use small on CPU:

```bash
python audio_to_token_mel.py speech.wav \
  --preset high --device cuda --batch-frames 512

python audio_to_token_mel.py speech.wav \
  --preset high --device cpu --batch-frames 128
```

`Ultra` and especially `max` are compute-heavy on a small CPU. For constrained
RAM, prefer CSV-only output, lower `--batch-frames` to 32 or 64, and lower
`--mel-batch-frames` during inversion. The default 80-million-pixel PNG limit
can represent roughly 160 MB of raw 16-bit raster before Pillow's working
copies, so set a smaller `--png-max-pixels` when memory is tight. CUDA is
recommended for routine ultra/max inversion.

JSON remains available for human-readable histograms and diagnostics:

```bash
python audio_to_token_mel.py speech.wav \
  --preset medium --write-json
```

Use `--force` to replace selected outputs and `--describe-presets` to print
the exact built-in configurations.

### Manual token geometry

Choose a built-in preset for the remaining sample-rate, STFT, mel-range, hop,
and dB defaults, then override the token matrix dimensions:

```bash
python audio_to_token_mel.py speech.wav \
  --preset medium \
  --columns-per-timestep 32 \
  --states-per-column 48 \
  --output-prefix speech.c32.k48 \
  --output-format both \
  --output-dir mel_out
```

Each timestep now contains exactly 32 acoustic values, and each value is a
zero-based integer in `[0,47]`. `--n-mels 32` and `--levels 48` remain exact
aliases. `K` need not be a power of two. Self-describing CSV and PNG outputs
store the actual `C` and `K`, so their normal inverse commands need no matching
flags. The demo automatically adds `.cC.kK` to custom output prefixes so
different geometry experiments do not overwrite one another.

The supported bounds are `1 ≤ C ≤ 4,096`, `2 ≤ K ≤ 65,536`, and
`C×K ≤ 16,777,216`. The last bound caps the per-band diagnostic histogram at
128 MiB (`8CK` bytes). A chosen `C` must also be feasible for the selected
FFT and frequency interval; if a mel band would be empty, increase `--n-fft`,
reduce `C`, or narrow the interval. Overrides cannot be combined with
`--preset all`.

At frame rate `F`, scalar serialization uses `C×F` tokens per second. Separate
band/state embeddings require `C×K` learned vectors, while shared level plus
band embeddings require only `C+K`.

### Direct timestep control

Set the spacing between rows directly with `--timestep-ms`; `--hop-ms` is an
exact alias:

```bash
python audio_to_token_mel.py speech.wav \
  --preset medium \
  --timestep-ms 12.5 \
  --output-format both \
  --output-dir mel_out
```

The selected preset still supplies the sample rate, analysis window, FFT, mel
range, columns, states, and dB range. Because a hop must contain a whole number
of samples, requested time is converted as follows:

```text
H                = round(sample_rate × requested_timestep_ms / 1000)
actual timestep  = 1000H / sample_rate  ms
frame rate F     = sample_rate / H
rows T           = ceil(decoded_sample_count / H)
scalar tokens/s  = C × F
```

The command summary reports `timestep_ms` and `timesteps_per_second` from the
actual sample-rounded hop, not merely the requested decimal. At the built-in
medium sample rate, `--timestep-ms 12.5` is exactly 200 samples, 80 rows/s, and
3,200 scalar tokens/s with the normal 40 columns. Combining it with 32 columns
reduces that to 2,560 scalar tokens/s:

```bash
bash demo.sh speech.wav medium cpu mel_out \
  --columns-per-timestep 32 \
  --states-per-column 32 \
  --timestep-ms 12.5
```

A shorter timestep increases sequence length linearly but does not change
`K`, the per-column vocabulary. A longer timestep reduces sequence length but
also reduces temporal detail and eventually makes phase reconstruction less
stable. The timestep must be positive and no more than half the analysis
window, so adjacent windows retain at least 50% overlap. Timestep and other
transform overrides cannot be combined with `--preset all`; run each desired
preset separately when overriding them.

Self-describing CSV and PNG inputs store the actual integer hop, so their
inverse commands need no timestep option. For a metadata-free input, give the
same base preset and original timestep:

```bash
python token_mel_to_audio.py raw_states.csv \
  -o recovered.wav \
  --preset medium \
  --timestep-ms 12.5
```

When given with a self-describing input, the inverse timestep option is an
assertion and fails if its sample-rounded hop disagrees with the stored one.
The demo adds `.dtMSms` to custom-timestep filenames so they cannot overwrite
the built-in-timestep outputs.

### Optional ladder-specific automatic gain control

`--agc` enables a bounded broadband waveform gain controller before the STFT.
It is off by default, so existing transforms remain unchanged. Each quality
ladder has a profile chosen to trade level consistency against preservation of
natural speech dynamics:

| Quality | Control interval with built-in timestep | Target RMS | Attack | Release | Maximum boost | Maximum attenuation | Gate | Peak ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small, lossy | 20 ms | −18 dBFS | 20 ms | 400 ms | +24 dB | 18 dB | −55 dBFS | −1 dBFS |
| Medium | 10 ms | −20 dBFS | 10 ms | 500 ms | +24 dB | 24 dB | −60 dBFS | −1 dBFS |
| High | 10 ms | −22 dBFS | 10 ms | 650 ms | +18 dB | 24 dB | −65 dBFS | −1 dBFS |
| Ultra | 8 ms | −23 dBFS | 8 ms | 800 ms | +15 dB | 24 dB | −70 dBFS | −1 dBFS |
| Max | 5 ms | −24 dBFS | 5 ms | 1000 ms | +12 dB | 24 dB | −75 dBFS | −1 dBFS |

Use `--describe-agc` to print the profiles. `--preset all --agc` applies the
corresponding profile independently to every ladder:

```bash
python audio_to_token_mel.py interview.wav \
  --preset all \
  --agc \
  --output-format both \
  --output-dir mel_out
```

The controller works on mono decoded PCM after the normal global DC removal.
For each control block it measures RMS and peak, computes a target gain,
limits that gain by the peak ceiling and the profile's boost/attenuation
bounds, suppresses positive boost below the gate, and smooths gain in dB with
separate attack and release constants. It interpolates gain through the block
and retains a hard peak ceiling as a final safety bound. Processing is
streamable with one control block of buffering and requires only CPU-scale
working memory; CUDA still accelerates the mel and inverse operations. When
`--timestep-ms` changes the hop, it also changes the AGC control interval to
the actual sample-rounded timestep.

Expert settings can be overridden when `--agc` is present:

```bash
python audio_to_token_mel.py speech.wav \
  --preset medium \
  --agc \
  --agc-target-dbfs -22 \
  --agc-attack-ms 15 \
  --agc-release-ms 700 \
  --agc-max-gain-db 18 \
  --agc-max-attenuation-db 24 \
  --agc-gate-dbfs -65 \
  --agc-peak-dbfs -1 \
  --output-format both
```

`--agc-max-attenuation-db` is a **nonnegative magnitude**: `24` permits the
controller gain to fall as low as −24 dB. The target must be below the peak
ceiling, the gate must be below the target, release must not be shorter than
attack, and all values must be finite. Expert AGC options without `--agc` are
rejected. The demo forwards these options only to the forward transform and
adds `.agc` to its filenames.

AGC deliberately does **not** change `top_db`, state thresholds, `K`, or mel
band gains from one timestep to the next. A token ID must keep one fixed
meaning throughout a dataset; adapting the quantizer range independently for
each frame would make identical IDs represent different amplitudes and would
damage state coverage and out-of-distribution behavior. AGC instead moves the
broadband waveform toward the fixed quantizer range, while endpoint clipping
still guarantees every cell is in `[0,K-1]`.

The default `file_percentile` reference is still computed per file and remains
future-dependent. For model training and deployment, first apply the chosen
AGC configuration to representative **training data only**, estimate a
corpus-level mel reference after that preprocessing, then freeze it and use
`--reference-mode fixed --reference-power VALUE` everywhere. AGC reduces local
recording-level variation; the frozen reference makes state meanings
comparable across files and prevents validation or deployment data from
changing the transform.

AGC is intentionally lossy. The CSV and PNG record that AGC was applied and
its configuration, but do not store a per-frame gain envelope or add a gain
token column. Reconstructed audio therefore represents the AGC-normalized
signal; the original absolute level and loudness envelope cannot be recovered.
This avoids increasing columns, vocabulary, or token rate. Disable AGC if
preserving the source's natural dynamics is more important than token
occupancy. The inverse reports a warning when it encounters AGC metadata.

The design study included the official
[WebRTC GainController2 source](https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/gain_controller2.cc),
which combines RMS/peak analysis, a noise estimate, headroom, and limiting,
and WebRTC's
[adaptive digital gain controller](https://webrtc.googlesource.com/src/+/1deb4f8adeac3cce9565c4a22ad17fa644893e18/modules/audio_processing/agc2/adaptive_digital_gain_controller.cc),
which bounds gain and gain-change rate. Frame-wise target level, peak
restriction, smoothing, and bounded amplification were also compared with
FFmpeg's official
[Dynamic Audio Normalizer documentation](https://ffmpeg.org/pipermail/ffmpeg-cvslog/2015-July/091728.html)
and
[current implementation](https://ffmpeg.org/doxygen/8.0/af__dynaudnorm_8c_source.html).
This script retains the bounded gain, gate/headroom, smoothing, and limiter
ideas in a simpler deterministic controller for token preprocessing. It does
not implement WebRTC's noise estimator and is not bit-compatible with either
project.

## CLI: one CSV or one PNG back to audio

CSV-only roundtrip:

```bash
python token_mel_to_audio.py \
  mel_out/recording.medium.mel.csv \
  -o recovered_from_csv.wav
```

PNG-only roundtrip:

```bash
python token_mel_to_audio.py \
  mel_out/recording.medium.mel.png \
  -o recovered_from_png.wav
```

For a metadata-free raw CSV, use a preset for the remaining transform
assumptions and state the original manual geometry:

```bash
python token_mel_to_audio.py raw_states.csv \
  -o recovered.wav \
  --preset medium \
  --columns-per-timestep 32 \
  --states-per-column 48
```

These inverse flags are unnecessary for self-describing files. When supplied
with one, they act only as assertions and fail if they disagree with embedded
metadata. A metadata-free PNG must be an integer state raster in QMel's strip
layout (frequency-reversed band rows, time across each strip), rather than a
plotted spectrogram. For `K > 256`, its pixels must be direct state IDs unless
it retains the normal QMel pixel header.

CUDA, if available:

```bash
python token_mel_to_audio.py \
  mel_out/recording.high.mel.png \
  -o recovered.wav \
  --device cuda
```

A faster, lower-quality CPU inversion uses a pseudoinverse and fewer phase
iterations:

```bash
python token_mel_to_audio.py \
  mel_out/recording.medium.mel.csv \
  -o recovered_fast.wav \
  --device cpu \
  --mel-inverse pinv \
  --griffin-lim-iters 32
```

Force a very small phase-recovery working set for a long recording:

```bash
python token_mel_to_audio.py \
  mel_out/interview.high.mel.png \
  -o recovered_low_ram.wav \
  --device cpu \
  --griffin-lim-chunk-frames 512 \
  --griffin-lim-overlap-frames 16
```

The default uses 32 nonnegative multiplicative mel-inversion iterations and
64 Griffin-Lim phase iterations. `--mel-batch-frames` bounds each mel inversion
batch. Long inputs are processed in overlapping 2,048-frame phase chunks with
32 frames of phase handoff and raised-cosine crossfading; the PCM working file
is memory-mapped. Use `--griffin-lim-chunk-frames 0` for one global phase solve
when memory is plentiful, or lower the chunk size to 512 for very small
machines. Keep the chunk size greater than twice the overlap. `--force` is
required to overwrite an existing WAV.

By default, `--peak 0` retains the amplitude implied by the stored reference
power and clips only if PCM full scale is exceeded. Use `--peak 0.95` to opt
into peak normalization.
State 0 is decoded as true zero power by default to prevent a silent token
matrix from becoming broadband hiss. `--floor-mode db-floor` instead decodes
it at the finite dB floor.

## What is stored in each standalone file

### CSV

The first line is the normal column header. The second line is a compact
comment beginning with:

```text
#TOKEN_MEL_CSV_V2
```

It contains the sample rate and sample count, STFT convention, mel filter
configuration, quantizer range, reference power, shape, and CRC. Every
remaining non-comment row is one acoustic timestep. Loaders should ignore
comment lines; for example, NumPy can use `comments="#"`.

When AGC is enabled, the compact metadata also records the controller
algorithm, selected ladder profile, effective control interval, bounds, and
the explicit fact that no reversible gain envelope is stored.

`--include-time-columns` adds `frame_index,time_s` for inspection. Those two
columns and the metadata comment must not be tokenized.

### QMel-PNG

When `K ≤ 256`, the PNG is indexed mode `P` and each acoustic pixel's palette
index is the exact integer state ID; this covers the built-in presets through
`high`. When `K > 256`, it uses mode `I;16`: each state is mapped bijectively
onto a 16-bit grayscale lattice, then checked and decoded back to the exact
integer; this covers built-in `ultra` and `max` and applies equally to custom
overrides. Time is split into horizontal strips for long files, and high
frequency is displayed at the top. The decoder reverses this layout back to
the canonical `timesteps × mel_bands` matrix.

Compact reconstruction metadata is stored twice:

1. compressed `iTXt` under `qmel.v2`; and
2. a complemented, compressed, checksummed pixel header.

The redundant pixel header means reconstruction still works if an otherwise
lossless PNG rewrite strips ancillary text. Header and padding pixels are
container data, not acoustic tokens, and the inverse script removes them.
Indexed PNG handles up to 256 states; 16-bit QMel-PNG handles up to 65,536
states. Both directions default to an 80-million-pixel safety limit; raise
`--png-max-pixels` explicitly on both commands for a larger trusted file.

Do not crop, resize, screenshot, convert to JPEG, or convert/re-quantize a
QMel-PNG. Those operations can destroy state IDs. A legacy plotted RGB
spectrogram is not equivalent to QMel-PNG: axes, color mapping, interpolation,
and unknown duration make its inversion heuristic and underdetermined.

## Exact built-in state spaces

All acoustic columns within a preset have the same integer range and number of
legal states. The dB value is relative to the reference power embedded in the
same CSV or PNG. Enabling AGC changes waveform levels before analysis, not
these column counts, legal ranges, or state counts.

| Quality | Sample rate | Window / FFT | Timestep (hop) | Frames/s | Band | Columns per timestep \(C\) | Integer range in every column | States per cell \(K\) | Dequantized dB range | dB step | Scalar tokens/s \(CF\) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small, lossy | 12 kHz | 40 ms / 512 | 20 ms | 50 | 50–6000 Hz | **20** | **0…15** | **16** | −48…0 | 3.2000 | **1,000** |
| Medium | 16 kHz | 25 ms / 512 | 10 ms | 100 | 50–8000 Hz | **40** | **0…63** | **64** | −64…0 | 1.0159 | **4,000** |
| High | 24 kHz | 25 ms / 1024 | 10 ms | 100 | 20–12000 Hz | **80** | **0…255** | **256** | −80…0 | 0.3137 | **8,000** |
| Ultra | 32 kHz | 32 ms / 4096 | 8 ms | 125 | 20–16000 Hz | **160** | **0…1023** | **1,024** | −96…0 | 0.0938 | **20,000** |
| Max | 48 kHz | 32 ms / 8192 | 5 ms | 200 | 20–24000 Hz | **320** | **0…4095** | **4,096** | −112…0 | 0.02735 | **64,000** |

Thus the direct answer for one data row is:

- small: 20 tokenized columns, each an integer in `[0,15]`, 16 possible
  states per cell;
- medium: 40 tokenized columns, each an integer in `[0,63]`, 64 possible
  states per cell;
- high: 80 tokenized columns, each an integer in `[0,255]`, 256 possible
  states per cell;
- ultra: 160 tokenized columns, each an integer in `[0,1023]`, 1,024 possible
  states per cell;
- max: 320 tokenized columns, each an integer in `[0,4095]`, 4,096 possible
  states per cell.

With inverse default `floor-mode=zero`, reconstructed power is in `[0,r]`,
where `r` is the stored reference power. With `db-floor`, it is in
`[r × 10^(-D/10), r]`, where `D` is 48, 64, 80, 96, or 112 dB.

These are lossy acoustic feature tiers, not lossless audio codecs. All five
downmix to mono, discard phase, integrate FFT bins into mel bands, clamp
dynamic range, and quantize. Ultra and max add bandwidth, mel resolution,
temporal resolution, FFT-grid density, dynamic range, and amplitude precision,
but still cannot reproduce the source waveform exactly. Zero padding makes the
large FFT grids denser; it does not provide the resolving power of an equally
long analysis window.

The state ladder deliberately adds two bits per tier:

```text
small  4 bits → medium 6 bits → high 8 bits
→ ultra 10 bits → max 12 bits
```

The nominal packed payloads are approximately 4, 24, 64, 200, and 768 kb/s.
CSV text is larger.

Small now uses a 40 ms window with a 20 ms hop. This adds overlap and materially
improves phase reconstruction without adding columns, states, frames, or
tokens compared with a 25 ms/20 ms configuration.

## Quantization and the hard OOD bound

For mel power `S`, stored reference `r`, dB floor `-D`, and `K` levels:

```text
d     = 10 log10(max(S, epsilon) / r)
d_bar = clip(d, -D, 0)
q     = round((d_bar + D) / D × (K - 1))
```

Consequently:

```text
q ∈ {0, 1, ..., K - 1}
```

for every finite input. Extreme low or high values saturate at an endpoint
rather than creating an out-of-range token ID. Magnitude must not be
modulo-wrapped: wrapping an over-range loud value to a silence-like ID creates
a discontinuity. Modulo is appropriate only for circular variables such as
phase.

The default `file_percentile` reference is the 99.5th percentile of a bounded,
deterministic sample of mel powers. It handles arbitrary files and stores the
chosen reference inside each output. It is not ideal for generative model
training because utterance-level gain is removed and the transform depends on
future samples.

For train/validation/deployment consistency, estimate a reference from
training data only, freeze it, and use:

```bash
python audio_to_token_mel.py clip.wav \
  --preset medium \
  --reference-mode fixed \
  --reference-power YOUR_TRAIN_REFERENCE
```

If AGC will be enabled, estimate that training reference from audio processed
with the same frozen AGC profile and overrides. Changing AGC after calibration
changes the distribution presented to the quantizer.

Hard clipping guarantees legal IDs, but a legal edge ID can still be rare in
training. Track floor/ceiling rates and state histograms, train with realistic
gain/noise/channel augmentation, and include the rare endpoints deliberately.

## Vocabulary size versus sequence length

There are four different state-count questions:

| Quality | Shared level IDs \(K\) | Separate `(band,state)` IDs \(CK\) | Factorized band + level vectors \(C+K\) | Whole-row combinations \(K^C\) |
|---|---:|---:|---:|---:|
| Small | **16** | 320 | 36 | \(16^{20}=2^{80}\approx1.21\times10^{24}\) |
| Medium | **64** | 2,560 | 104 | \(64^{40}=2^{240}\approx1.77\times10^{72}\) |
| High | **256** | 20,480 | 336 | \(256^{80}=2^{640}\approx4.56\times10^{192}\) |
| Ultra | **1,024** | 163,840 | 1,184 | \(1024^{160}=2^{1600}\approx4.45\times10^{481}\) |
| Max | **4,096** | 1,310,720 | 4,416 | \(4096^{320}=2^{3840}\approx9.02\times10^{1155}\) |

Do not assign one token to a complete row: almost every row would be unseen.
If band position is known from the row structure, the same `K` amplitude token
IDs can be shared across every band. A band-position embedding can be added:

```text
embedding(c, q) = band_embedding(c) + level_embedding(q)
```

This changes the learned vector count from `C×K` to `C+K` without increasing
sequence length. A shared ordinal head, or an embedding generated by a small
function of normalized `q`, can reduce independently learned parameters
further while preserving one scalar token per cell.

Factorizing a large level ID into base-4 digits can also reduce embedding
tables if digits are predicted in parallel:

| Quality | Direct factorized vectors \(C+K\) | Base-4 parallel-factor vectors \(C+4\lceil\log_4K\rceil\) |
|---|---:|---:|
| Small | 36 | **28** |
| Medium | 104 | **52** |
| High | 336 | **96** |
| Ultra | 1,184 | **180** |
| Max | 4,416 | **344** |

Serial digit prediction increases token count by the number of digits, so it
is counterproductive when context length is the binding constraint.

## Coverage available per embedding

For one corpus hour, idealized uniform occupancy gives:

| Quality | Frames/hour | Mean hits per separate `(band,state)` | Mean hits per shared level state |
|---|---:|---:|---:|
| Small | 180,000 | 11,250 | 225,000 |
| Medium | 360,000 | 5,625 | 225,000 |
| High | 360,000 | 1,406 | 112,500 |
| Ultra | 450,000 | 439 | 70,313 |
| Max | 720,000 | 176 | 56,250 |

These are optimistic because speech frames are correlated and uniform dB bins
are not uniformly occupied. The minimum count, number of distinct speakers and
recordings reaching a state, and train/evaluation shift matter more than the
mean. Sharing level states across bands is the largest immediate improvement
for embedding repetition.

Ultra and max are fidelity-first rather than coverage-first tokenizations.
Their independent `(band,state)` inventories are much too large for modest
spoken corpora. Use shared ordered levels, an embedding generated from
normalized state value, or parallel base-4 factors. Ultra needs five base-4
digits per cell and max needs six; predicting those factors serially would
multiply sequence length, so use parallel local heads if factorization is
chosen.

## Lower-state, lower-column candidates

The built-ins are stable baselines. These are useful coverage-first experiments:

| Target | Columns \(C\) | States/cell \(K\) | Timestep | dB range | dB step | Scalar tokens/s | Separate \(CK\) | Factorized \(C+K\) | Packed lower bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggressive small speech | 16 | 8 | 20 ms | 48 | 6.857 | **800** | 128 | 24 | 2.4 kb/s |
| Compact medium | 32 | 32 | 10 ms | 56 | 1.806 | **3,200** | 1,024 | 64 | 16 kb/s |
| Safer high vocabulary | 80 | 128 | 10 ms | 80 | 0.630 | 8,000 | 10,240 | 208 | 56 kb/s |
| Compact high | 64 | 128 | 10 ms | 72 | 0.567 | **6,400** | 8,192 | 192 | 44.8 kb/s |

Example compact-medium command:

```bash
python audio_to_token_mel.py speech.wav \
  --preset medium \
  --columns-per-timestep 32 \
  --states-per-column 32 \
  --top-db 56 \
  --output-prefix speech.c32.k32 \
  --output-format both
```

Example compact-high command:

```bash
python audio_to_token_mel.py speech.wav \
  --preset high \
  --columns-per-timestep 64 \
  --states-per-column 128 \
  --top-db 72 \
  --output-prefix speech.c64.k128 \
  --output-format both
```

Reducing `K` reduces vocabulary and improves state coverage but does not reduce
scalar sequence length. Reducing mel columns or frames per second reduces
sequence length. Choose the smallest candidate that meets held-out phoneme/WER
and reconstruction-listening targets, not visual similarity alone.
When increasing `--timestep-ms` to reduce frames per second, keep the window
at least twice as long as the hop so every centered output sample remains
covered.

For an even smaller sequence, group bands and use product/vector quantization.
For example, eight 64-way group codes per 100 Hz frame produce 800 tokens/s,
five times fewer than medium scalar serialization and ten times fewer than
high. Freeze the codebooks learned on training data, reserve or saturate
fallback codes, and audit dead/rare code usage. A grouped learned decoder is
more complex and can introduce codebook OOD, so it should follow the scalar
baselines rather than replace them blindly.

Long all-floor spans can be represented by a bounded silence/run token outside
the mel matrix. Saturate long durations; never modulo-wrap them.

## Recommended optimization order

1. Start with medium `40×64`, but share level IDs across bands.
2. Evaluate `32×32` at `top_db=56`; it is the best first coverage/quality
   tradeoff for ordinary spoken datasets.
3. If input gain varies substantially, compare the preset's `--agc` profile
   against no AGC on held-out speech. Freeze the chosen AGC configuration.
4. After that choice, freeze one training-corpus reference and measure
   endpoint occupancy on validation microphones, speakers, noise levels, and
   codecs.
5. Fit per-band training quantiles if uniform dB bins leave many rare states.
   Reserve state 0 for silence/floor and the top state for saturation; freeze
   thresholds and reconstruct each state with its training median.
6. Merge adjacent states until every active state clears a chosen minimum
   count across enough distinct recordings, not just enough adjacent frames.
7. If context length still dominates, reduce bands or increase timestep, or
   move to grouped codes. Do not build a whole-frame vocabulary.
8. Treat ultra and max as fidelity/vocoder or analysis tiers. For an
   autoregressive scalar-token LM, require very large datasets or parallel
   factorized/ordinal heads before adopting their 1,024- or 4,096-state cells.

The current script implements optional waveform AGC, uniform dB quantization,
and fixed/file-level references. Per-band quantile fitting and grouped
codebooks are recommended next-stage experiments, not silently applied
transforms.

## Inverse method and unavoidable losses

The inverse performs:

```text
integer states
→ dequantized mel power
→ batched nonnegative linear-frequency power estimate
→ magnitude STFT
→ Griffin-Lim phase estimation
→ overlap-add PCM WAV
```

It uses the forward script's exact centered-frame convention, periodic Hann
window, right-side FFT zero padding, mel normalization, and original decoded
sample count. The CSV/PNG state payload roundtrip is exact and CRC-checked;
audio reconstruction is necessarily approximate because phase, stereo,
within-band detail, unclipped magnitude, and quantization residuals were not
stored.

If audio fidelity is the primary goal, a learned vocoder conditioned on the
same frozen mel representation will usually outperform Griffin-Lim. It does
not change the token-state accounting, but its training distribution must
cover every intended preset and recording domain.

## Verification performed

The delivered scripts were checked on CPU with speech-like audio, silence,
very short audio, custom 32-column/48-state and 53-column/1,000-state settings,
metadata-free CSV/PNG fallback, and the v2.3 AGC/timestep demo path:

- all five presets' CSV and PNG decoded to exactly equal state matrices;
- ultra and max used exact 16-bit grayscale lattices with 1,024 and 4,096
  states per cell, including pixel-lattice and CRC validation;
- fixed-seed CSV and PNG inversions produced byte-identical WAV files;
- the demo accepted `--hop-ms 12.5` as the timestep alias together with
  medium AGC, every expert AGC override, and custom `C=32,K=48`; it created
  collision-safe `.dt12.5ms.agc` names, retained the exact 1,920 samples, and
  reconstructed byte-identical WAVs from CSV and PNG;
- forced multi-chunk CSV and PNG inversions also produced byte-identical WAV
  files with the exact stored sample count;
- the default medium nonnegative inverse achieved about `0.00369` relative
  mel reprojection error on the test clip;
- all-zero states reconstructed to exact digital silence;
- the custom STFT/ISTFT convention reconstructed known-phase audio with
  relative numerical error below `9e-8`;
- PNG inversion survived removal of `iTXt` by using its pixel header;
- 16-bit PNG inversion also survived removal of `iTXt`;
- a deliberately changed state pixel failed the stored CRC check;
- CSV-only and PNG-only forward modes emitted no unwanted JSON or companion
  representation;
- self-describing CSV rows were parsed into compact state storage in bounded
  batches, and long phase recovery used bounded overlapping chunks.

CUDA was not available in the verification environment, so the optional
PyTorch CUDA path was not runtime-tested there. The NumPy CPU path is the
validated fallback.
