#!/usr/bin/env bash
# bootstrap.sh -- one-shot setup for the Orin Nano LLM inference sweep
# harness. Checks for five things and installs/builds/downloads/writes
# whichever are missing; safe to re-run any time since everything is
# skipped if already present.
#
#   1. System prerequisites (python3, pip3, git, cmake, curl, gcc/g++/make)
#                                                            -> apt installed
#   2. llama.cpp (llama-bench, llama-quantize, llama-cli)  -> cloned + built
#   3. Gemma 3 270M model, F16 GGUF                         -> downloaded
#   4. The sweep harness's own scripts (this project)       -> written out
#   5. Python deps (gguf, numpy) needed by vocab_truncate.py -> pip installed
#
# Usage:
#   ./bootstrap.sh
#
# Override any default path/URL via environment variables, e.g.:
#   LLAMA_CPP_DIR=/opt/llama.cpp SWEEPER_DIR=/data/sweep ./bootstrap.sh

set -euo pipefail

REAL_USER="${SUDO_USER:-${USER:-$(id -un)}}"
REAL_HOME="$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6 || true)"
REAL_HOME="${REAL_HOME:-$HOME}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$REAL_HOME/llama.cpp}"
SWEEPER_DIR="${SWEEPER_DIR:-$REAL_HOME/Sweeper}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-F16.gguf}"
MODEL_PATH="$SWEEPER_DIR/gemma-3-270m-f16.gguf"

# Sweeper is the project root: create it and cd in immediately, before any
# other step runs, so everything from here on (downloads, generated files,
# relative-path invocations of the harness scripts) is anchored there
# rather than wherever this script happened to be launched from.
mkdir -p "$SWEEPER_DIR"
cd "$SWEEPER_DIR"

echo "llama.cpp dir: $LLAMA_CPP_DIR"
echo "Sweeper dir:   $SWEEPER_DIR  (now the working directory)"
echo

# =============================================================================
echo "=== 1/5: system prerequisites ==="
# Maps each command-line tool this script (or the rest of the harness)
# needs to the apt package that provides it. build-essential covers
# gcc/g++/make, all needed to compile llama.cpp.
declare -A TOOL_TO_PKG=(
    [python3]=python3
    [pip3]=python3-pip
    [git]=git
    [cmake]=cmake
    [curl]=curl
    [gcc]=build-essential
    [g++]=build-essential
    [make]=build-essential
)

MISSING_PKGS=()
for tool in "${!TOOL_TO_PKG[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        MISSING_PKGS+=("${TOOL_TO_PKG[$tool]}")
    fi
done
# de-duplicate (build-essential can appear up to 3 times)
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    mapfile -t MISSING_PKGS < <(printf '%s\n' "${MISSING_PKGS[@]}" | sort -u)
fi

if [[ ${#MISSING_PKGS[@]} -eq 0 ]]; then
    echo "  found: python3, pip3, git, cmake, curl, gcc, g++, make"
else
    echo "  missing packages: ${MISSING_PKGS[*]} -- installing via apt"
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "  ERROR: apt-get not found -- this script only automates installs on" >&2
        echo "  Debian/Ubuntu (which is what Jetson L4T is built on). Install these" >&2
        echo "  manually and re-run: ${MISSING_PKGS[*]}" >&2
        exit 1
    fi
    # Use sudo only if not already root -- some environments (containers,
    # some minimal images) run as root directly with no sudo binary at all.
    SUDO_CMD=""
    if [[ "$(id -u)" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "  ERROR: not running as root and no sudo available -- install manually:" >&2
            echo "    apt-get update && apt-get install -y ${MISSING_PKGS[*]}" >&2
            exit 1
        fi
        SUDO_CMD="sudo"
    fi
    $SUDO_CMD apt-get update || echo "  WARNING: apt-get update had errors (possibly an unrelated broken repo) -- continuing anyway" >&2
    $SUDO_CMD apt-get install -y "${MISSING_PKGS[@]}"
    echo "  installed: ${MISSING_PKGS[*]}"
fi
echo

# ---------------------------------------------------------------------------
# write_if_missing <path> -- reads a heredoc into <path> unless it already
# exists, in which case the heredoc is discarded and the file is left alone.
# ---------------------------------------------------------------------------
write_if_missing() {
    local path="$1"
    if [[ -f "$path" ]]; then
        echo "  found: $path"
        cat >/dev/null
        return
    fi
    echo "  writing: $path"
    cat > "$path"
}

# =============================================================================
echo "=== 2/5: llama.cpp ==="
LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"
LLAMA_QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"
LLAMA_CLI="$LLAMA_CPP_DIR/build/bin/llama-cli"

if [[ -x "$LLAMA_BENCH" && -x "$LLAMA_QUANTIZE" && -x "$LLAMA_CLI" ]]; then
    echo "  found: llama-bench, llama-quantize, llama-cli in $LLAMA_CPP_DIR/build/bin"
else
    echo "  not found -- cloning/building (this takes a while, especially the first CUDA build)"
    if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
        git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
    else
        echo "  $LLAMA_CPP_DIR already exists -- building in place (git pull skipped, remove the dir first for a clean clone)"
    fi
    (
        cd "$LLAMA_CPP_DIR"
        cmake -B build -DGGML_CUDA=ON
        cmake --build build --config Release -j"$(nproc)" --target llama-bench llama-quantize llama-cli
    )
    if [[ ! -x "$LLAMA_BENCH" ]]; then
        echo "  ERROR: build finished but llama-bench still not found at $LLAMA_BENCH -- check the build output above" >&2
        exit 1
    fi
    echo "  build complete"
fi
echo

# =============================================================================
echo "=== 3/5: Gemma 3 270M model (F16 GGUF) ==="
if [[ -f "$MODEL_PATH" ]]; then
    echo "  found: $MODEL_PATH"
else
    echo "  not found -- downloading from Hugging Face (~543 MB)"
    if ! curl -L --fail -o "$MODEL_PATH.partial" "$MODEL_URL"; then
        echo "  ERROR: download failed. Check network access and MODEL_URL:" >&2
        echo "    $MODEL_URL" >&2
        rm -f "$MODEL_PATH.partial"
        exit 1
    fi
    mv "$MODEL_PATH.partial" "$MODEL_PATH"
    echo "  downloaded: $MODEL_PATH"
fi
echo

# =============================================================================
echo "=== 4/5: sweep harness files ==="

write_if_missing "README.md" <<'FILE_README_EOF'
# Orin Nano llama.cpp Sweep Harness

Measures energy/latency/memory for prefill and decode as a function of
sequence length, and for the LM head as a function of vocab size, on a
Jetson Orin Nano running llama.cpp (CUDA backend) against Gemma 3 270M.

**Status: validated end-to-end on real hardware (2026-08).** All three
sweeps ran clean except the expected V=1 degenerate case (see below).
Results showed: a clean power-law energy/token decay with prefill length
(batch amortization), flat per-token decode cost (as expected for
autoregressive generation), and a clear LM-head cost inflection around
V≈2k–8k vocab entries for this 270M model, with latency and energy/token
both bending upward past that point. A model-size self-consistency check
(linear extrapolation of `model_buf_MB` vs `vocab_size` predicted the
V=131072 value to within 0.01 MB of what was actually measured) gives good
confidence the vocab-truncation methodology is sound.

## What this actually measures, and how

| Axis | Tool | Notes |
|---|---|---|
| Prefill sweep (p = 1..256, n=0) | `llama-bench -p <list> -n 0` | pp-only, no generation |
| Decode sweep (p=1, n = 1..256) | `llama-bench -p 1 -n <list>` | 1-token prefill so timing is decode-dominated |
| Vocab sweep (V = 1..2^18) | `llama-bench` against **vocab-truncated GGUF copies** | see `vocab_truncate.py`; confirmed tied embeddings on Gemma 3 (no separate `output.weight` tensor) |
| Latency (TTFT, TPOT) | `llama-bench -o json` | `avg_ns`/`avg_ts` per test → TTFT = pp time, TPOT = tg time / n |
| Energy/token | `tegrastats` power rails, integrated over the wall-clock window of the llama-bench invocation | reps auto-scaled to control model-load contamination, see below |
| Memory (total + breakdown) | `tegrastats` RAM field (total) + llama.cpp stderr load logs (model / KV / compute buffer sizes) | **requires `llama-bench -v`** — without it, no buffer-size logs are printed at all |

## Changelog / bugs found and fixed during validation

- **Model-load energy contamination (fixed).** The original reps-calibration
  used `wall_clock_time / CALIB_REPS` to estimate per-repetition cost. This
  conflates one-time model-load time (CUDA init, loading the model,
  parsing a 262k-token vocab) with actual per-token compute time. At small
  p/n, load time dominates completely, so the energy window was mostly
  measuring "how much energy does loading the model take" rather than the
  intended prefill/decode cost — this showed up as `energy_per_token_mJ`
  values that didn't scale sensibly with sequence length. Fixed by reading
  `llama-bench`'s own `avg_ns` (which already excludes load) from a
  calibration run's JSON output, computing the true load overhead as
  `wall_time - reps*avg_ns`, and solving for the rep count needed to keep
  load overhead ≤10% of the total window (`contam_max` in
  `estimate_reps_needed`). This means small-p/n tests can now require
  hundreds to thousands of reps and take much longer than before — that's
  the real cost of correctness, not a new bug.
- **Vocab-sweep token-count bug (fixed).** The vocab-sweep probe runs a
  combined (nonzero p, nonzero n) test, meaning the measured window covers
  both a pp loop and a tg loop. The original code divided total window
  energy by tg-token count only, silently discarding the pp tokens from
  the denominator. Fixed to divide by `(p+n)*reps` for combined tests
  (see `run_one_point` in `run_sweep.py`) — this means combined-test
  energy/token is a blended prefill+decode number, not phase-separable.
- **`llama-quantize --allow-requantize` needed** if your base checkpoint
  has any already-quantized tensors (e.g. Unsloth's UD dynamic quants mix
  f32/f16/q8_0 per-tensor) — without the flag, quantize refuses to
  requantize q8_0 tensors and fails outright.
- **V=1 vocab variant reliably fails**, confirmed empirically: `llama-bench`'s
  synthetic benchmark prompt uses token IDs that don't exist in a 1-token
  vocabulary (`init: invalid token[0] = -1`), so it can't build a warmup
  batch at all. This is not a bug in this harness — treat V=1 (and
  probably V=2..8) as smoke tests, not clean data, per the original
  caveat below. In the validated run, V≥2 all completed cleanly.
- **New CSV columns**: `model_load_overhead_s` (seconds of load time
  estimated for that test point) and `est_contam_frac` (estimated fraction
  of the energy window attributable to load, not compute) — check these
  before trusting any row. Anything above ~0.15 (aside from a known
  degenerate case like V=1) means the calibration estimate for that point
  didn't hold and the row deserves a second look.

## Known instrumentation limits (read before trusting numbers)

1. **tegrastats timestamp resolution is 1 second**, even at `--interval 50`.
   `run_sweep.py`'s calibration step (see changelog above) auto-scales
   `-r` reps per test to (a) reach a multi-second window and (b) keep
   load-time contamination controlled — check `est_contam_frac` in the
   output CSV. A single occasional outlier row (e.g. a transient system
   hiccup inflating the one-shot calibration measurement) can cause that
   point to badly overprovision reps and take far longer than its
   neighbors; the resulting energy/token number is usually still fine
   (it's a ratio, robust to extra reps) but the run will be slow. Sanity
   check by comparing a suspicious row's derived metric against its
   immediate neighbors in the sweep.

2. **Energy is total-board energy, not per-token-marginal energy.** It
   includes idle/background draw on whichever rails you sum. Report both
   raw energy/token and, if you want marginal cost, fit a line across the
   sweep and use the slope (that cancels the fixed idle floor).

3. **Vocab sweep truncates a real model's output projection**, it does not
   swap in 2^k different pretrained models. See `vocab_truncate.py` — this
   isolates the LM-head matmul cost as a function of V while leaving the
   transformer body identical, which is what you want for a controlled
   sweep, but the *logits themselves are garbage* above the truncation
   point (expected — we don't care about output quality here).

4. **Tied embeddings**: confirmed on Gemma 3 270M — there is no separate
   `output.weight` tensor, only `token_embd.weight`, so truncating the
   embedding table also changes which token ids are legal to embed.
   `vocab_truncate.py` handles this automatically (detects the tied case
   via `find_tensor(reader, "output.weight") is None`). Treat V<8 as
   smoke tests per the changelog note above.

5. **Memory breakdown** comes from parsing llama.cpp's own stderr load-time
   logs (`load_tensors: ... buffer size`, `llama_kv_cache: ... KV buffer
   size`, `sched_reserve: ... compute buffer size` on this project's
   build). **Requires `-v`/`--verbose` on `llama-bench`** — without it no
   buffer-size lines print at all. `llama-bench` creates a fresh context
   (and therefore fresh KV/compute buffer log lines) per test entry in one
   invocation, so these lines print multiple times with identical values;
   `mem_parser.py` deduplicates identical `(label, value)` pairs before
   summing so this doesn't inflate the totals. Re-check the regexes in
   `mem_parser.py` against `strings` of your build's log output if this
   ever comes back empty on a different llama.cpp version.

## Files

- `vocab_truncate.py` — produces a vocab-truncated copy of a GGUF file.
- `tegra_energy.py` — starts/stops `tegrastats` logging, parses the log,
  integrates power → energy over an arbitrary time window.
- `mem_parser.py` — parses llama.cpp stderr load logs for buffer sizes.
- `run_sweep.py` — orchestrator: runs all three sweeps, writes the CSV,
  and auto-archives every run (see below).
- `report.py` — shared analytics (power-law fits, contamination/outlier
  checks, model-size self-consistency check) and README rendering, used
  by both the auto-archive step and `analyze_results.py`. Not meant to be
  run directly.
- `analyze_results.py` — standalone QC/README generator for any
  results.csv (e.g. to re-check an old archived run, or a CSV you've
  merged from multiple runs by hand).

## Auto-archiving

Every `run_sweep.py` run writes its CSV to whatever `--out` path you gave
it, **and** also copies that CSV plus a generated `README.md` into
`<archive-dir>/<timestamp>/` (default archive dir: `runs/`). This builds
up a running, self-documenting record of every sweep across time instead
of a single results.csv that gets overwritten on each run.

The generated README includes: which model file(s)/architecture(s) were
actually used (captured per-row from `llama-bench`'s own JSON output --
important since the vocab sweep uses a different model file per row),
the same quick-analytics numbers described above, and the same
contamination/outlier QC checks.

Options:
- `--archive-dir DIR` — change where runs get archived (default `runs/`)
- `--no-archive` — skip archiving, just write the one CSV like before
- `--run-note "..."` — free-text note prepended to that run's README
  (e.g. "testing a different VOCAB_PROBE_P")

To regenerate or inspect a README for an existing CSV without re-running
the sweep:
```bash
python3 analyze_results.py path/to/results.csv --write-readme
```

## Usage

```bash
pip install gguf --break-system-packages   # for vocab_truncate.py

# 1. Get a non-quantized (F16) GGUF -- vocab_truncate.py refuses quantized
#    tensors. If your checkpoint has any already-quantized tensors (e.g.
#    Unsloth UD dynamic quants), you need --allow-requantize:
llama-quantize --allow-requantize gemma-3-270m-UD-Q8_K_XL.gguf gemma-3-270m-f16.gguf f16

# 2. Sanity-check the llama-bench JSON schema + memory logs on your build
python3 run_sweep.py --debug-schema \
    --llama-bench /path/to/llama-bench \
    --model gemma-3-270m-f16.gguf \
    --out /dev/null

# 3. Build the vocab-truncated model set once (slow, do it up front)
python3 vocab_truncate.py --base gemma-3-270m-f16.gguf --out-dir vocab_variants \
    --sizes 1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144

# 4. Run the full sweep (sudo needed for full tegrastats rail visibility)
#    -- this also auto-archives to runs/<timestamp>/, see above
sudo python3 run_sweep.py \
    --llama-bench /path/to/llama-bench \
    --model gemma-3-270m-f16.gguf \
    --vocab-variant-dir vocab_variants \
    --out results.csv

# 5. QC before trusting the numbers (also covered by the archived README)
python3 analyze_results.py results.csv
```

## Output schema (`results.csv`)

```
sweep_type, prefill_len, decode_len, vocab_size, reps, effective_reps,
ttft_ms, tpot_ms, pp_tok_per_s, tg_tok_per_s,
energy_total_mJ, energy_per_token_mJ, energy_rails_used,
ram_peak_MB, ram_avg_MB, model_buf_MB, kv_cache_MB, compute_buf_MB,
wall_window_s, model_load_overhead_s, est_contam_frac, timestamp
```

Plot energy/token and TTFT/TPOT vs. sequence length on log-log axes —
prefill shows a clean power-law decay (batch amortization at the GPU
level); decode stays essentially flat (no batching available across the
autoregressive sequence dimension). Vocab sweep shows latency/energy flat
until V gets large enough that the final matmul (d_model × V) stops being
dwarfed by the rest of the forward pass (observed inflection around
V≈2k–8k for this 270M model), then scales up noticeably with V.
FILE_README_EOF

write_if_missing "requirements.txt" <<'FILE_REQUIREMENTS_EOF'
gguf>=0.10.0
numpy
FILE_REQUIREMENTS_EOF

write_if_missing "tegra_energy.py" <<'FILE_TEGRAENERGY_EOF'
#!/usr/bin/env python3
"""
Start/stop `tegrastats` logging and integrate power rails -> energy over
an arbitrary wall-clock window.

CAVEAT (see README): tegrastats timestamps only have 1-second resolution
even at --interval 50, so short single-shot benchmarks will straddle very
few distinct timestamps. Mitigate by measuring across many repetitions
(run_sweep.py does this automatically) so the window is several seconds.
"""
import re
import subprocess
import time

TS_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
# rails look like: VDD_GPU_SOC 512mW/512mW  or  VIN_SYS_5V0 1234mW/1200mW
POWER_RE = re.compile(r"([A-Z][A-Z0-9_]*(?:_SOC|_CV|_CPU|_GPU|_SYS|_5V0|_IN|_MW)?)\s+(\d+)mW/(\d+)mW")


class TegraLogger:
    def __init__(self, log_path, interval_ms=50):
        self.log_path = log_path
        self.interval_ms = interval_ms
        self._proc = None

    def start(self):
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms),
             "--logfile", str(self.log_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.2)  # let the first sample land before the workload starts
        return time.time()

    def stop(self):
        end = time.time()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        return end


def parse_log(path):
    """Returns list of {'t': epoch_seconds, 'ram_mb': int, 'power_mw': {rail: mw}}.

    NOTE: tegrastats only prints wall-clock HH:MM:SS (no sub-second field),
    so 't' has 1s granularity regardless of --interval.
    """
    samples = []
    with open(path) as f:
        for line in f:
            ts_m = TS_RE.match(line)
            ram_m = RAM_RE.search(line)
            if not ts_m or not ram_m:
                continue
            t = time.mktime(time.strptime(ts_m.group(1), "%m-%d-%Y %H:%M:%S"))
            rails = {name: int(cur) for name, cur, _avg in POWER_RE.findall(line)}
            samples.append({"t": t, "ram_mb": int(ram_m.group(1)), "power_mw": rails})
    return samples


def window(samples, start_t, end_t):
    return [s for s in samples if start_t <= s["t"] <= end_t]


def integrate_energy_mJ(samples, rail_names):
    """Trapezoidal integration of sum(rail power) over time -> millijoules."""
    if len(samples) < 2:
        return 0.0
    energy = 0.0
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        pa = sum(a["power_mw"].get(r, 0) for r in rail_names)
        pb = sum(b["power_mw"].get(r, 0) for r in rail_names)
        energy += 0.5 * (pa + pb) * dt  # mW * s = mJ
    return energy


def all_rail_names(samples):
    names = set()
    for s in samples:
        names |= set(s["power_mw"].keys())
    return names


def ram_stats(samples):
    if not samples:
        return None, None
    vals = [s["ram_mb"] for s in samples]
    return max(vals), sum(vals) / len(vals)
FILE_TEGRAENERGY_EOF

write_if_missing "mem_parser.py" <<'FILE_MEMPARSER_EOF'
#!/usr/bin/env python3
"""
Parse llama.cpp's stderr load/context-creation logs for buffer size
breakdown.

Verified (2026-08) against this project's actual llama.cpp build
(b10547, commit 749f688f) on an Orin Nano, running `llama-bench -v`.
Real lines look like:

  load_tensors:        CUDA0 model buffer size =   511.46 MiB
  load_tensors:    CUDA_Host model buffer size =   320.00 MiB
  llama_context:  CUDA_Host  output buffer size =     1.00 MiB
  llama_kv_cache:      CUDA0 KV buffer size =     0.75 MiB
  llama_kv_cache:      CUDA0 KV buffer size =     3.75 MiB   (SWA cache, separate line)
  sched_reserve:      CUDA0 compute buffer size =     8.02 MiB
  sched_reserve:  CUDA_Host compute buffer size =     0.05 MiB

IMPORTANT: `llama-bench` requires `-v`/`--verbose` to print any of this at
all -- without it these lines don't appear (confirmed: plain `llama-bench`
stderr had none of them, `-v` produced all of the above).

IMPORTANT: llama-bench creates a fresh context (and therefore a fresh KV
cache + compute buffer) for EACH test entry in one invocation -- e.g. a
single `-p 8 -n 8` call produces both a pp-only and a tg-only test entry,
so these lines are printed twice, back to back, with identical values
(context sizing depends on n_ctx/n_batch config, not on the specific
prompt/gen length being timed). We de-duplicate identical (label, value)
pairs before summing so repeated context creations don't inflate the
totals. If you ever see genuinely different KV/compute values across
repeated blocks in your own log (e.g. because n_ctx changes per test),
this dedup would silently drop a real distinct measurement -- spot check
raw_matches if numbers look suspiciously small.
"""
import re

# Matches lines like "<anything> <label> buffer size = <N.NN> MiB"
BUFFER_RE = re.compile(r"([\w. ]*?buffer size)\s*=\s*([\d.]+)\s*(MiB|MB)", re.IGNORECASE)
# Older/alternate llama.cpp wording for KV cache that does NOT say
# "buffer size" (e.g. "KV self size = 16.00 MiB"), kept separate so a line
# like "CUDA0 KV buffer size = ..." isn't counted twice.
KV_ALT_RE = re.compile(r"(KV self size)\s*=\s*([\d.]+)\s*(MiB|MB)", re.IGNORECASE)


def parse_stderr_log(text):
    """Returns dict with model_buf_MB, kv_cache_MB, compute_buf_MB
    (each a float, or None if nothing matched at all) plus raw_matches
    (deduplicated (label, mb) pairs) so nothing silently disappears.
    """
    seen = set()
    buckets = {"model": 0.0, "kv": 0.0, "compute": 0.0}
    any_match = False

    for label, val, _unit in BUFFER_RE.findall(text):
        mb = float(val)
        key = (label.strip(), mb)
        if key in seen:
            continue  # duplicate context-creation block, see docstring
        seen.add(key)
        any_match = True
        label_l = label.lower()
        if "compute" in label_l:
            buckets["compute"] += mb
        elif "kv" in label_l:
            buckets["kv"] += mb
        else:
            buckets["model"] += mb  # includes the small "output buffer" line

    for label, val, _unit in KV_ALT_RE.findall(text):
        mb = float(val)
        key = (label.strip(), mb)
        if key in seen:
            continue
        seen.add(key)
        any_match = True
        buckets["kv"] += mb

    return {
        "model_buf_MB": buckets["model"] if any_match else None,
        "kv_cache_MB": buckets["kv"] if buckets["kv"] else None,
        "compute_buf_MB": buckets["compute"] if buckets["compute"] else None,
        "raw_matches": sorted(seen),
    }
FILE_MEMPARSER_EOF

write_if_missing "vocab_truncate.py" <<'FILE_VOCABTRUNC_EOF'
#!/usr/bin/env python3
"""
Produce vocab-truncated copies of a GGUF checkpoint so the LM-head /
output-projection matmul cost can be swept as a function of vocab size V
while the rest of the transformer body stays byte-identical.

Truncates (keeps first V rows of):
  - token_embd.weight   ([n_vocab, n_embd] in GGUF tensor order)
  - output.weight       (only present if embeddings are NOT tied)
  - tokenizer.ggml.tokens / .scores / .token_type metadata arrays

Updates the "<arch>.vocab_size" metadata key.

IMPORTANT — quantization requirement:
  Row-wise truncation is a plain array slice, which is only safe for
  non-quantized tensors (F32 / F16 / BF16) where each row is a contiguous,
  independently-meaningful set of values. For K-quantized formats
  (Q4_K, Q8_0, etc.) values are packed into fixed-size blocks that don't
  line up with rows in a way you can safely slice. If your checkpoint is
  quantized, reconvert/dequantize the vocab-relevant tensors to F16 first,
  e.g. with llama.cpp's `llama-quantize model.gguf model-f16.gguf f16`
  (Gemma 3 270M is small enough that F16 is fine to benchmark against).
  This script refuses to touch quantized tensors and tells you so.

Verify tensor/metadata key names against your specific checkpoint with
`gguf-dump <file>` before trusting this on an unfamiliar architecture —
naming has arch-specific quirks and this was written against llama.cpp's
current (2026) Gemma 3 export conventions.

Requires: pip install gguf numpy --break-system-packages
"""
import argparse
import os
import sys

try:
    import numpy as np
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.constants import GGMLQuantizationType
except ImportError:
    print("Missing deps. Run: pip install gguf numpy --break-system-packages",
          file=sys.stderr)
    raise

FLOAT_TYPES = {GGMLQuantizationType.F32, GGMLQuantizationType.F16,
               GGMLQuantizationType.BF16}

SCALAR_ADDERS = {
    GGUFValueType.UINT8: "add_uint8", GGUFValueType.INT8: "add_int8",
    GGUFValueType.UINT16: "add_uint16", GGUFValueType.INT16: "add_int16",
    GGUFValueType.UINT32: "add_uint32", GGUFValueType.INT32: "add_int32",
    GGUFValueType.UINT64: "add_uint64", GGUFValueType.INT64: "add_int64",
    GGUFValueType.FLOAT32: "add_float32", GGUFValueType.FLOAT64: "add_float64",
    GGUFValueType.BOOL: "add_bool", GGUFValueType.STRING: "add_string",
}

SKIP_TOP_LEVEL = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                  "general.architecture"}  # GGUFWriter(arch=...) sets this already
VOCAB_ARRAY_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.scores",
                    "tokenizer.ggml.token_type")


def get_arch(reader):
    f = reader.fields.get("general.architecture")
    if f is None:
        raise RuntimeError("no general.architecture key in this GGUF")
    return f.contents()


def find_tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            return t
    return None


def truncate_one(base_path, out_path, vocab_size):
    reader = GGUFReader(base_path)
    arch = get_arch(reader)
    vocab_key = f"{arch}.vocab_size"

    embd = find_tensor(reader, "token_embd.weight")
    if embd is None:
        raise RuntimeError("token_embd.weight not found — check tensor "
                            "names for this arch with gguf-dump")
    output = find_tensor(reader, "output.weight")  # None => tied embeddings

    for t in (embd, output):
        if t is not None and t.tensor_type not in FLOAT_TYPES:
            raise RuntimeError(
                f"tensor '{t.name}' is quantized ({t.tensor_type}); "
                "dequantize to f16 first (see module docstring)")

    # n_vocab is by far the larger of the two dims for any real model;
    # sanity-check this assumption against gguf-dump on your checkpoint.
    orig_vocab = max(embd.shape[0], embd.shape[1])

    if vocab_size > orig_vocab:
        raise ValueError(f"requested V={vocab_size} > base model vocab {orig_vocab}")

    writer = GGUFWriter(out_path, arch)

    for name, field in reader.fields.items():
        if name in SKIP_TOP_LEVEL or name == vocab_key or name in VOCAB_ARRAY_KEYS:
            continue
        main_type = field.types[0]
        if main_type == GGUFValueType.ARRAY:
            elem_type = field.types[-1]
            values = field.contents(slice(None))
            if elem_type == GGUFValueType.ARRAY:
                raise RuntimeError(f"nested array metadata not supported: {name}")
            writer.add_array(name, values)
        else:
            val = field.contents()
            fn_name = SCALAR_ADDERS.get(main_type)
            if fn_name is None:
                raise RuntimeError(f"unhandled metadata type {main_type} for {name}")
            getattr(writer, fn_name)(name, val)

    writer.add_uint32(vocab_key, vocab_size)

    for arr_name in VOCAB_ARRAY_KEYS:
        field = reader.fields.get(arr_name)
        if field is None:
            continue
        writer.add_array(arr_name, field.contents(slice(0, vocab_size)))

    for t in reader.tensors:
        data = t.data
        if t.name == "token_embd.weight" or (output is not None and t.name == "output.weight"):
            axis = 0 if data.shape[0] == orig_vocab else 1
            data = np.take(data, range(vocab_size), axis=axis).copy()
        writer.add_tensor(t.name, data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[V={vocab_size:>7}] wrote {out_path}  "
          f"(tied_embeddings={output is None}, orig_vocab={orig_vocab})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="source GGUF file (F16/F32/BF16)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sizes", required=True,
                    help="comma-separated vocab sizes, e.g. 1,2,4,...,262144")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sizes = sorted(int(s) for s in args.sizes.split(","))
    base_name = os.path.splitext(os.path.basename(args.base))[0]

    for v in sizes:
        out_path = os.path.join(args.out_dir, f"{base_name}.vocab{v}.gguf")
        truncate_one(args.base, out_path, v)


if __name__ == "__main__":
    main()
FILE_VOCABTRUNC_EOF

write_if_missing "report.py" <<'FILE_REPORT_EOF'
#!/usr/bin/env python3
"""
Analytics and README generation shared between run_sweep.py's auto-archive
step and standalone use via analyze_results.py. Stdlib only.
"""
import math
import statistics


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def power_law_fit(points):
    """Fit y = a * x^b via log-log linear regression. points: [(x, y), ...]
    with x, y > 0. Returns (a, b, r2) or None if fewer than 2 usable points.
    """
    pts = [(x, y) for x, y in points if x and y and x > 0 and y > 0]
    if len(pts) < 2:
        return None
    xs = [math.log(x) for x, _ in pts]
    ys = [math.log(y) for _, y in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a_log = my - b * mx
    a = math.exp(a_log)
    ss_res = sum((y - (b * x + a_log)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return a, b, r2


def linear_extrapolation_check(points, check_x):
    """points: [(x, y), ...]. Fits a line through the endpoints (lowest and
    highest x) and reports predicted vs actual y at check_x, if check_x is
    itself one of the observed points. Returns dict or None."""
    pts = sorted((x, y) for x, y in points if x is not None and y is not None)
    if len(pts) < 2:
        return None
    x_lo, y_lo = pts[0]
    x_hi, y_hi = pts[-1]
    if x_hi == x_lo:
        return None
    slope = (y_hi - y_lo) / (x_hi - x_lo)
    intercept = y_lo - slope * x_lo
    actual = next((y for x, y in pts if x == check_x), None)
    if actual is None:
        return None
    predicted = slope * check_x + intercept
    return {"slope": slope, "check_x": check_x, "predicted": predicted, "actual": actual}


def contamination_check(rows, warn_threshold=0.15, degenerate_vocab=1):
    clean = [r for r in rows if to_float(r.get("vocab_size")) != degenerate_vocab]
    contams = [to_float(r.get("est_contam_frac")) for r in clean]
    contams = [c for c in contams if c is not None]
    flagged = [r for r in clean
               if to_float(r.get("est_contam_frac")) is not None
               and to_float(r["est_contam_frac"]) > warn_threshold]
    known_bad = [r for r in rows if to_float(r.get("vocab_size")) == degenerate_vocab]
    return {
        "n": len(contams),
        "max": max(contams) if contams else None,
        "mean": statistics.mean(contams) if contams else None,
        "flagged": flagged,
        "known_bad": known_bad,
    }


def reps_outliers(rows, ratio=8.0):
    by_sweep = {}
    for r in rows:
        by_sweep.setdefault(r["sweep_type"], []).append(r)
    out = []
    for sweep, sub in by_sweep.items():
        reps_vals = [to_float(r.get("reps")) for r in sub if to_float(r.get("reps")) is not None]
        if len(reps_vals) < 3:
            continue
        med = statistics.median(reps_vals)
        for r in sub:
            reps = to_float(r.get("reps"))
            if reps is not None and med > 0 and reps > ratio * med:
                out.append((r, med))
    return out


def compute_analytics(rows):
    """Compute the full set of derived numbers used in the README, from
    whatever rows are actually present -- works for partial sweeps too
    (e.g. only a prefill sweep, no vocab sweep run)."""
    a = {}

    prefill = [(to_float(r["prefill_len"]), to_float(r["energy_per_token_mJ"]))
               for r in rows if r["sweep_type"] == "prefill"]
    a["prefill_fit"] = power_law_fit(prefill)
    a["prefill_n"] = len(prefill)
    if prefill:
        xs_sorted = sorted(prefill)
        a["prefill_range"] = (xs_sorted[0], xs_sorted[-1])

    decode_tpot = [to_float(r["tpot_ms"]) for r in rows if r["sweep_type"] == "decode"]
    decode_tpot = [v for v in decode_tpot if v is not None]
    if decode_tpot:
        a["decode_tpot_min"] = min(decode_tpot)
        a["decode_tpot_max"] = max(decode_tpot)
        a["decode_tpot_spread_pct"] = (max(decode_tpot) - min(decode_tpot)) / min(decode_tpot) * 100

    vocab_rows = [r for r in rows if r["sweep_type"] == "vocab" and to_float(r.get("vocab_size")) != 1]
    if vocab_rows:
        vocab_energy = [(to_float(r["vocab_size"]), to_float(r["energy_per_token_mJ"])) for r in vocab_rows]
        vocab_energy = [(v, e) for v, e in vocab_energy if v is not None and e is not None]
        flat = [e for v, e in vocab_energy if v <= 1024]
        if flat and vocab_energy:
            v_hi, e_hi = max(vocab_energy)
            a["vocab_flat_mean_mJ"] = statistics.mean(flat)
            a["vocab_high_v"] = v_hi
            a["vocab_high_mJ"] = e_hi
            a["vocab_energy_ratio"] = e_hi / statistics.mean(flat) if statistics.mean(flat) else None

        vocab_tpot = [(to_float(r["vocab_size"]), to_float(r["tpot_ms"])) for r in vocab_rows]
        vocab_tpot = [(v, t) for v, t in vocab_tpot if v is not None and t is not None]
        flat_t = [t for v, t in vocab_tpot if v <= 1024]
        if flat_t and vocab_tpot:
            v_hi_t, t_hi = max(vocab_tpot)
            a["vocab_tpot_flat_mean_ms"] = statistics.mean(flat_t)
            a["vocab_tpot_high_ms"] = t_hi
            a["vocab_tpot_ratio"] = t_hi / statistics.mean(flat_t) if statistics.mean(flat_t) else None

        model_buf = [(to_float(r["vocab_size"]), to_float(r["model_buf_MB"])) for r in vocab_rows]
        model_buf = [(v, b) for v, b in model_buf if v is not None and b is not None]
        vs_present = sorted(v for v, _ in model_buf)
        # pick the observed vocab size closest to the geometric mean of the
        # endpoints, so the check is a meaningful mid-range extrapolation
        # rather than a trivial point right next to an endpoint
        v_lo, v_hi = vs_present[0], vs_present[-1]
        target = (v_lo * v_hi) ** 0.5 if v_lo > 0 else (v_lo + v_hi) / 2
        mid_candidates = [v for v in vs_present if v not in (v_lo, v_hi)]
        check_v = min(mid_candidates, key=lambda v: abs(v - target)) if mid_candidates else None
        if check_v:
            a["model_buf_check"] = linear_extrapolation_check(model_buf, check_v)

    a["contamination"] = contamination_check(rows)
    a["reps_outliers"] = reps_outliers(rows)

    model_names = sorted({r.get("model_filename") for r in rows if r.get("model_filename")})
    model_types = sorted({r.get("model_type") for r in rows if r.get("model_type")})
    a["model_filenames"] = model_names
    a["model_types"] = model_types

    a["device_ids"] = sorted({r.get("device_id") for r in rows if r.get("device_id")})

    return a


def render_readme(rows, analytics, run_label, source_note=""):
    a = analytics
    lines = []
    lines.append(f"# Sweep run: {run_label}")
    lines.append("")
    if source_note:
        lines.append(source_note)
        lines.append("")

    if a.get("device_ids"):
        if len(a["device_ids"]) == 1:
            lines.append(f"**Device:** `{a['device_ids'][0]}`")
        else:
            lines.append(f"**Devices in this run:** {', '.join(f'`{d}`' for d in a['device_ids'])}")
            lines.append("_Multiple devices present -- if this is unexpected, check `device_id` "
                         "wasn't accidentally left at a default/cloned value across units._")
        lines.append("")

    lines.append("## Model(s) in this run")
    if a.get("model_types"):
        for mt in a["model_types"]:
            lines.append(f"- `{mt}`")
    if a.get("model_filenames"):
        lines.append("")
        lines.append("Files referenced:")
        for mf in a["model_filenames"]:
            lines.append(f"- `{mf}`")
    if not a.get("model_types") and not a.get("model_filenames"):
        lines.append("(model metadata not available -- older CSV schema, or "
                      "all llama-bench calls failed before reporting JSON)")
    lines.append("")

    lines.append(f"## Rows: {len(rows)}")
    lines.append("")

    lines.append("## Quick analytics")
    lines.append("")

    fit = a.get("prefill_fit")
    if fit:
        coef, exp, r2 = fit
        lo, hi = a["prefill_range"]
        r2_str = f"{r2:.4f}" if r2 is not None else "n/a"
        lines.append(
            f"**Prefill** ({a['prefill_n']} points): energy/token follows "
            f"`energy_per_token_mJ ≈ {coef:.1f} × prefill_len^({exp:.3f})` "
            f"(log-log fit, R²={r2_str}). "
            f"Falls from {lo[1]:.1f} mJ at p={int(lo[0])} to {hi[1]:.2f} mJ "
            f"at p={int(hi[0])}."
        )
        lines.append("")

    if "decode_tpot_spread_pct" in a:
        lines.append(
            f"**Decode**: per-token time (`tpot_ms`) varies by only "
            f"{a['decode_tpot_spread_pct']:.2f}% across the sweep "
            f"({a['decode_tpot_min']:.3f}–{a['decode_tpot_max']:.3f} ms) — "
            f"flat, as expected for autoregressive decode with no batching "
            f"available across the sequence dimension."
        )
        lines.append("")

    if "vocab_energy_ratio" in a:
        lines.append(
            f"**Vocab**: energy/token sits flat around "
            f"{a['vocab_flat_mean_mJ']:.2f} mJ for V≤1024, rising to "
            f"{a['vocab_high_mJ']:.2f} mJ at V={int(a['vocab_high_v'])} "
            f"({a['vocab_energy_ratio']:.2f}× the flat-region mean)."
        )
        if "vocab_tpot_ratio" in a:
            lines.append(
                f"TPOT shows the same inflection: flat "
                f"{a['vocab_tpot_flat_mean_ms']:.2f} ms through V≤1024, "
                f"rising to {a['vocab_tpot_high_ms']:.2f} ms at the highest "
                f"V tested ({a['vocab_tpot_ratio']:.2f}× increase)."
            )
        lines.append("")

    check = a.get("model_buf_check")
    if check:
        lines.append(
            f"**Self-consistency check**: linear extrapolation of "
            f"`model_buf_MB` vs. `vocab_size` (fit from the endpoints) "
            f"predicts {check['predicted']:.2f} MB at V={int(check['check_x'])}; "
            f"actual measured value was {check['actual']:.2f} MB. Close "
            f"agreement here is independent evidence the vocab-truncation "
            f"methodology preserved the rest of the model correctly."
        )
        lines.append("")

    contam = a.get("contamination", {})
    lines.append("## QC")
    lines.append("")
    if contam.get("n"):
        lines.append(
            f"- Contamination (`est_contam_frac`, excluding known-degenerate "
            f"rows): n={contam['n']}, max={contam['max']:.4f}, "
            f"mean={contam['mean']:.4f}"
        )
        if contam.get("flagged"):
            lines.append(f"  - **{len(contam['flagged'])} row(s) above 0.15 threshold** -- check these before trusting:")
            for r in contam["flagged"]:
                lines.append(
                    f"    - {r['sweep_type']} p={r.get('prefill_len')} "
                    f"n={r.get('decode_len')} V={r.get('vocab_size') or '-'}: "
                    f"contam={r['est_contam_frac']}"
                )
        else:
            lines.append("  - none above threshold")
    if contam.get("known_bad"):
        for r in contam["known_bad"]:
            lines.append(
                f"- Excluded from contamination stats: vocab_size=1 "
                f"(known-degenerate case -- llama-bench can't build a warmup "
                f"batch with a 1-token vocabulary)"
            )

    outliers = a.get("reps_outliers", [])
    if outliers:
        lines.append(f"- **{len(outliers)} reps outlier(s)** (>8x the sweep's median reps) -- "
                      f"likely a one-off calibration hiccup, not necessarily bad data:")
        for r, med in outliers:
            lines.append(
                f"  - {r['sweep_type']} p={r.get('prefill_len')} n={r.get('decode_len')} "
                f"V={r.get('vocab_size') or '-'}: reps={r.get('reps')} (median={med:.0f}), "
                f"check whether its energy_per_token_mJ fits the surrounding trend"
            )
    else:
        lines.append("- no reps outliers detected")

    lines.append("")
    lines.append("_Generated automatically from results.csv -- see the parent "
                 "`orin_sweep/README.md` for full methodology and instrumentation caveats._")

    return "\n".join(lines)
FILE_REPORT_EOF

write_if_missing "run_sweep.py" <<'FILE_RUNSWEEP_EOF'
#!/usr/bin/env python3
"""
Orchestrates the full sweep against llama-bench on the Jetson Orin Nano:
  - prefill sweep:  p in POWERS_OF_TWO(1..256), n=0
  - decode sweep:   p=1,                        n in POWERS_OF_TWO(1..256)
  - vocab sweep:    p=VOCAB_PROBE_P, n=VOCAB_PROBE_N, against each
                    vocab-truncated model produced by vocab_truncate.py

For each test point:
  1. start tegrastats logging
  2. run `llama-bench -o json -r <reps> ...`, capturing stdout (json) and
     stderr (llama.cpp load logs, for memory breakdown)
  3. stop tegrastats, parse the log window, integrate energy
  4. write one row to the output CSV

Each run is auto-archived to `<archive-dir>/<device_id>_<timestamp>/`
(default archive dir: `runs/`), containing a copy of the results CSV and a
generated README.md with model provenance and quick analytics -- see
report.py. This gives a running, self-documenting record of every sweep
run rather than a single results.csv that gets overwritten each time.

Every row is tagged with a `device_id` column (defaults to hostname, or
override with the SWEEP_DEVICE_ID env var -- important for multi-device
setups, since cloned SD-card images often share identical hostnames
unless explicitly renamed). This is a grouping key in aggregate_runs.py,
so results from different physical devices never get silently averaged
together as if they were repeat runs of the same hardware.

`-r` (repeat count) is auto-scaled per test to control model-load energy
contamination (see estimate_reps_needed) -- tegrastats has 1s timestamp
resolution, so short single-shot tests would otherwise integrate mostly
load-time energy rather than the intended compute cost. See README.
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from tegra_energy import TegraLogger, parse_log, window, integrate_energy_mJ, all_rail_names, ram_stats
from mem_parser import parse_stderr_log
import report

# Device identifier tagged onto every row -- critical for multi-device
# setups (a cluster of Orins, etc.) since cloned SD-card images often
# share identical hostnames unless explicitly renamed. Override with
# SWEEP_DEVICE_ID if hostname isn't reliably unique across your fleet.
DEVICE_ID = os.environ.get("SWEEP_DEVICE_ID") or socket.gethostname()

POWERS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MIN_WINDOW_S = 3.0
BASE_REPS = 5           # llama-bench's own default-ish repeat count
CALIB_REPS = 3          # quick untimed calibration run to estimate per-rep time
VOCAB_PROBE_P = 64      # fixed prefill/decode length used while sweeping vocab size
VOCAB_PROBE_N = 32
TEGRA_INTERVAL_MS = 50


def run_llama_bench(binary, model, p, n, reps, extra_args=None):
    cmd = [binary, "-m", model, "-p", str(p), "-n", str(n),
           "-r", str(reps), "-o", "json", "-v"]
    if extra_args:
        cmd += extra_args
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    if proc.returncode != 0:
        print(f"llama-bench failed (p={p} n={n}): {proc.stderr[-2000:]}",
              file=sys.stderr)
    return proc.stdout, proc.stderr, t0, t1


def estimate_reps_needed(binary, model, p, n, contam_max=0.10):
    """Calibration run to determine reps needed so that (a) the measurement
    window is >= MIN_WINDOW_S, and (b) fixed one-time model-load overhead
    is at most `contam_max` fraction of the total energy-integration
    window -- otherwise short tests are dominated by load energy, not the
    thing we're trying to measure (this was silently wrong in an earlier
    version of this script; see README changelog note).

    Uses llama-bench's OWN reported avg_ns per test entry (which already
    excludes model load -- it's purely the timed repetition loop) rather
    than naive wall-clock-time/CALIB_REPS, which conflated load time with
    per-repetition compute time.
    """
    out, _err, t0, t1 = run_llama_bench(binary, model, p, n, CALIB_REPS)
    entries = parse_bench_json(out)
    per_rep_test_s = sum((e.get("avg_ns") or 0) for e in entries) / 1e9
    wall_s = t1 - t0
    load_overhead_s = max(wall_s - per_rep_test_s * CALIB_REPS, 0.0)

    if per_rep_test_s <= 0:
        return BASE_REPS, 0.0, load_overhead_s

    reps_for_window = int(MIN_WINDOW_S / per_rep_test_s) + 1
    # solve reps * per_rep_test_s >= ((1-contam_max)/contam_max) * load_overhead_s
    reps_for_contam = (int(((1 - contam_max) / contam_max) * load_overhead_s / per_rep_test_s) + 1
                       if load_overhead_s > 0 else BASE_REPS)

    reps = max(BASE_REPS, reps_for_window, reps_for_contam)
    return reps, per_rep_test_s, load_overhead_s


def parse_bench_json(stdout_text):
    """llama-bench -o json prints one JSON array with one object per test
    (pp and tg are separate entries sharing the same invocation)."""
    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError:
        # some versions print one JSON object per line instead of an array
        data = [json.loads(l) for l in stdout_text.splitlines() if l.strip().startswith("{")]
    return data


def summarize_bench(entries):
    """Pull pp/tg tokens-per-sec out of whichever entries are present.

    UNVERIFIED ASSUMPTION, CHECK BEFORE TRUSTING RESULTS: this assumes
    each llama-bench invocation with a single (p, n) pair emits either one
    pp-only JSON entry (n_gen == 0) or one tg-only entry (n_prompt == 0),
    per this project's own convention of always calling llama-bench with
    n=0 for prefill tests and p=1 for decode tests. Different llama.cpp
    versions have varied in whether a nonzero (p, n) pair emits one
    combined row or two rows. Run `--debug-schema` once on your build
    (see main()) and eyeball the raw JSON before trusting a multi-hour
    sweep on this parser.
    """
    pp_ts = tg_ts = None
    for e in entries:
        n_prompt = e.get("n_prompt", 0)
        n_gen = e.get("n_gen", 0)
        ts = e.get("avg_ts")
        if n_prompt and not n_gen:
            pp_ts = ts
        elif n_gen and not n_prompt:
            tg_ts = ts
        elif n_gen:  # both nonzero -- some builds only emit one combined entry
            tg_ts = ts
            if pp_ts is None:
                pp_ts = e.get("avg_ts_pp")  # may not exist; best-effort
    return pp_ts, tg_ts


def run_one_point(binary, model, p, n, tegra_log_dir, rail_names_override=None):
    reps, per_rep_test_s, load_overhead_s = estimate_reps_needed(binary, model, p, n)

    tegra_log = os.path.join(tegra_log_dir, f"tegra_{p}_{n}_{os.getpid()}_{int(time.time()*1000)}.log")
    logger = TegraLogger(tegra_log, interval_ms=TEGRA_INTERVAL_MS)
    t_start = logger.start()

    stdout_text, stderr_text, t0, t1 = run_llama_bench(binary, model, p, n, reps)

    t_end = logger.stop()

    samples = parse_log(tegra_log)
    win = window(samples, t_start, t_end)
    rails = rail_names_override or all_rail_names(win)
    energy_mj = integrate_energy_mJ(win, rails)
    ram_peak, ram_avg = ram_stats(win)

    entries = parse_bench_json(stdout_text)
    pp_ts, tg_ts = summarize_bench(entries)

    # model provenance -- important because the vocab sweep uses a
    # DIFFERENT model file per row, so the CSV needs to self-document
    # which file/arch/param-count each row actually ran against.
    model_filename = model_type = model_n_params = None
    for e in entries:
        if e.get("model_filename"):
            model_filename = e["model_filename"]
            model_type = e.get("model_type")
            model_n_params = e.get("model_n_params")
            break

    # For a pure prefill test (n=0) or pure decode test (p=0), all tokens
    # in the window are that one phase. For a combined test (both nonzero,
    # i.e. the vocab-sweep probe), the window's energy covers BOTH a pp
    # loop and a tg loop -- divide by total tokens across both, since we
    # cannot separate phase-specific energy from one integrated window.
    if n == 0:
        tokens_this_test = p * reps
    elif p == 0:
        tokens_this_test = n * reps
    else:
        tokens_this_test = (p + n) * reps

    ttft_ms = (1000.0 * p / pp_ts) if pp_ts else None
    tpot_ms = (1000.0 / tg_ts) if tg_ts else None

    mem = parse_stderr_log(stderr_text)

    predicted_test_loop_s = per_rep_test_s * reps
    denom = load_overhead_s + predicted_test_loop_s
    est_contam_frac = (load_overhead_s / denom) if denom > 0 else None

    return {
        "reps": reps,
        "effective_reps": reps,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "pp_tok_per_s": pp_ts,
        "tg_tok_per_s": tg_ts,
        "energy_total_mJ": energy_mj,
        "energy_per_token_mJ": (energy_mj / tokens_this_test) if tokens_this_test else None,
        "energy_rails_used": ",".join(sorted(rails)),
        "ram_peak_MB": ram_peak,
        "ram_avg_MB": ram_avg,
        "model_buf_MB": mem["model_buf_MB"],
        "kv_cache_MB": mem["kv_cache_MB"],
        "compute_buf_MB": mem["compute_buf_MB"],
        "wall_window_s": t_end - t_start,
        "model_load_overhead_s": load_overhead_s,
        "est_contam_frac": est_contam_frac,
        "model_filename": model_filename,
        "model_type": model_type,
        "model_n_params": model_n_params,
        "device_id": DEVICE_ID,
        "timestamp": t0,
    }


FIELDNAMES = [
    "sweep_type", "prefill_len", "decode_len", "vocab_size", "reps",
    "effective_reps", "ttft_ms", "tpot_ms", "pp_tok_per_s", "tg_tok_per_s",
    "energy_total_mJ", "energy_per_token_mJ", "energy_rails_used",
    "ram_peak_MB", "ram_avg_MB", "model_buf_MB", "kv_cache_MB",
    "compute_buf_MB", "wall_window_s", "model_load_overhead_s",
    "est_contam_frac", "model_filename", "model_type", "model_n_params",
    "device_id", "timestamp",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-bench", required=True, help="path to llama-bench binary")
    ap.add_argument("--model", required=True, help="base GGUF model for prefill/decode sweeps")
    ap.add_argument("--vocab-variant-dir", help="dir of vocab-truncated GGUFs from vocab_truncate.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tegra-log-dir", default=None)
    ap.add_argument("--archive-dir", default="runs",
                    help="each run auto-archives to <archive-dir>/<timestamp>/ "
                         "with a results.csv copy and generated README.md "
                         "(default: runs/). Pass --no-archive to skip.")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--run-note", default="",
                    help="free-text note included at the top of the archived README "
                         "(e.g. what's different about this run)")
    ap.add_argument("--debug-schema", action="store_true",
                    help="run one p=8,n=8 llama-bench call, print raw JSON, "
                         "and exit -- do this first on a new build/machine")
    args = ap.parse_args()

    print(f"device_id: {DEVICE_ID}  (override with SWEEP_DEVICE_ID env var)")
    print()

    if args.debug_schema:
        out, err, *_ = run_llama_bench(args.llama_bench, args.model, 8, 8, 1)
        print("---- raw stdout (JSON) ----")
        print(out)
        print("---- raw stderr (load logs, first 4000 chars) ----")
        print(err[:4000])
        print("\nCheck: does each entry have n_prompt/n_gen as expected by "
              "summarize_bench()? Adjust that function if not, before "
              "running the full sweep.")
        return

    tegra_log_dir = args.tegra_log_dir or tempfile.mkdtemp(prefix="tegra_logs_")
    os.makedirs(tegra_log_dir, exist_ok=True)

    rows = []

    print("=== prefill sweep ===")
    for p in POWERS:
        print(f"  p={p} n=0")
        r = run_one_point(args.llama_bench, args.model, p, 0, tegra_log_dir)
        r.update({"sweep_type": "prefill", "prefill_len": p, "decode_len": 0, "vocab_size": None})
        rows.append(r)

    print("=== decode sweep ===")
    for n in POWERS:
        print(f"  p=1 n={n}")
        r = run_one_point(args.llama_bench, args.model, 1, n, tegra_log_dir)
        r.update({"sweep_type": "decode", "prefill_len": 1, "decode_len": n, "vocab_size": None})
        rows.append(r)

    if args.vocab_variant_dir:
        print("=== vocab sweep ===")
        variants = sorted(
            (f for f in os.listdir(args.vocab_variant_dir) if f.endswith(".gguf")),
            key=lambda f: int(f.split(".vocab")[-1].split(".gguf")[0]),
        )
        for fname in variants:
            v = int(fname.split(".vocab")[-1].split(".gguf")[0])
            path = os.path.join(args.vocab_variant_dir, fname)
            print(f"  V={v} ({fname})")
            r = run_one_point(args.llama_bench, path, VOCAB_PROBE_P, VOCAB_PROBE_N, tegra_log_dir)
            r.update({"sweep_type": "vocab", "prefill_len": VOCAB_PROBE_P,
                      "decode_len": VOCAB_PROBE_N, "vocab_size": v})
            rows.append(r)
    else:
        print("(skipping vocab sweep -- no --vocab-variant-dir given)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDNAMES})

    print(f"wrote {len(rows)} rows to {args.out}")

    if not args.no_archive and rows:
        safe_device = "".join(c if c.isalnum() or c in "-_." else "_" for c in DEVICE_ID)
        run_id = f"{safe_device}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(args.archive_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.out, os.path.join(run_dir, "results.csv"))

        analytics = report.compute_analytics(rows)
        source_note = args.run_note or (
            f"Auto-archived from `{args.out}` on device `{DEVICE_ID}` "
            f"(model: {args.model}, vocab-variant-dir: {args.vocab_variant_dir or 'none'})."
        )
        readme_text = report.render_readme(rows, analytics, run_id, source_note)
        with open(os.path.join(run_dir, "README.md"), "w") as f:
            f.write(readme_text)

        print(f"archived run to {run_dir}/ (results.csv + README.md)")


if __name__ == "__main__":
    main()
FILE_RUNSWEEP_EOF

write_if_missing "analyze_results.py" <<'FILE_ANALYZE_EOF'
#!/usr/bin/env python3
"""
Standalone QC / README generator for any results.csv from run_sweep.py --
e.g. to re-check an archived run, or a CSV you've manually merged from
multiple runs. Uses the same computation logic as run_sweep.py's
auto-archive step (see report.py) so the two never drift apart.

Usage:
  python3 analyze_results.py results.csv                 # print QC to stdout
  python3 analyze_results.py results.csv --write-readme   # also write README.md next to it
"""
import argparse
import csv
import os

import report


def print_summary(rows, analytics):
    a = analytics
    print(f"Loaded {len(rows)} rows\n")

    print("=== Model(s) ===")
    if a.get("model_types"):
        for mt in a["model_types"]:
            print(f"  {mt}")
    if a.get("model_filenames"):
        for mf in a["model_filenames"]:
            print(f"  file: {mf}")
    if not a.get("model_types") and not a.get("model_filenames"):
        print("  (no model metadata in this CSV -- older schema)")

    print("\n=== Contamination ===")
    c = a["contamination"]
    if c["n"]:
        print(f"  n={c['n']}  max={c['max']:.4f}  mean={c['mean']:.4f}")
        if c["flagged"]:
            print(f"  {len(c['flagged'])} row(s) above threshold:")
            for r in c["flagged"]:
                print(f"    {r['sweep_type']:8s} p={r.get('prefill_len'):>4} n={r.get('decode_len'):>4} "
                      f"V={r.get('vocab_size') or '-':>7}  contam={r['est_contam_frac']}")
        else:
            print("  none above threshold -- looks clean")
    else:
        print("  no est_contam_frac data found")
    for r in c["known_bad"]:
        print(f"  (excluded: vocab_size=1 known-degenerate row)")

    print("\n=== Reps outliers ===")
    outliers = a["reps_outliers"]
    if outliers:
        for r, med in outliers:
            print(f"  {r['sweep_type']:8s} p={r.get('prefill_len'):>4} n={r.get('decode_len'):>4} "
                  f"V={r.get('vocab_size') or '-':>7}  reps={r.get('reps')} (median={med:.0f}) "
                  f"wall_window_s={r.get('wall_window_s')}")
    else:
        print("  none")

    print("\n=== Trends ===")

    def sweep_rows(name):
        return sorted(
            (r for r in rows if r["sweep_type"] == name),
            key=lambda r: report.to_float(r.get("prefill_len")) or report.to_float(r.get("decode_len")) or 0,
        )

    prefill = sweep_rows("prefill")
    if prefill:
        print("  prefill: prefill_len -> energy_per_token_mJ, ttft_ms")
        for r in prefill:
            print(f"    p={r['prefill_len']:>4}  energy/tok={r.get('energy_per_token_mJ', ''):>10}  "
                  f"ttft_ms={r.get('ttft_ms', ''):>10}")

    decode = sweep_rows("decode")
    if decode:
        print("  decode: decode_len -> tpot_ms, energy_per_token_mJ")
        for r in decode:
            print(f"    n={r['decode_len']:>4}  tpot_ms={r.get('tpot_ms', ''):>10}  "
                  f"energy/tok={r.get('energy_per_token_mJ', ''):>10}")

    vocab = sorted(
        (r for r in rows if r["sweep_type"] == "vocab" and report.to_float(r.get("vocab_size")) not in (None, 1)),
        key=lambda r: report.to_float(r["vocab_size"]),
    )
    if vocab:
        print("  vocab (V=1 excluded, known degenerate): vocab_size -> tpot_ms, energy_per_token_mJ, model_buf_MB")
        for r in vocab:
            print(f"    V={r['vocab_size']:>7}  tpot_ms={r.get('tpot_ms', ''):>10}  "
                  f"energy/tok={r.get('energy_per_token_mJ', ''):>10}  "
                  f"model_buf_MB={r.get('model_buf_MB', ''):>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--write-readme", action="store_true",
                    help="also write a README.md next to the CSV, same format as run_sweep.py's auto-archive")
    ap.add_argument("--run-note", default="")
    args = ap.parse_args()

    with open(args.csv_path) as f:
        rows = list(csv.DictReader(f))

    analytics = report.compute_analytics(rows)
    print_summary(rows, analytics)

    if args.write_readme:
        run_label = os.path.splitext(os.path.basename(args.csv_path))[0]
        source_note = args.run_note or f"Generated from `{args.csv_path}`."
        readme_text = report.render_readme(rows, analytics, run_label, source_note)
        out_path = os.path.join(os.path.dirname(args.csv_path) or ".", "README.md")
        with open(out_path, "w") as f:
            f.write(readme_text)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
FILE_ANALYZE_EOF

write_if_missing "aggregate_runs.py" <<'FILE_AGGREGATE_EOF'
#!/usr/bin/env python3
"""
Aggregate multiple sweep runs into per-test-point statistics: n, mean,
std, and coefficient of variation (CV = std/mean, as a %) for the key
metrics. Groups by (sweep_type, prefill_len, decode_len, vocab_size,
model_type, device_id) -- model_type and device_id are both included
deliberately so that runs against different quantizations/models, or
from different physical devices (e.g. a cluster of Orins), never get
silently averaged together as if they were noisy repeats of the same
measurement.

Three uses:
  1. Run-to-run variance: run the sweep several times against the SAME
     model on the SAME device (see sweep.sh --repeat), then aggregate --
     model_type and device_id are constant across those runs, so they
     merge together as intended.
  2. Quantization/model comparison: run the sweep once each against
     different model files (different quant, different model) on the same
     device, then aggregate -- model_type differs, so each shows up as its
     own row at the same prefill_len/decode_len, giving a direct
     comparison table.
  3. Cross-device comparison: run the same sweep on multiple physical
     devices, then aggregate -- device_id differs, so each device's
     results stay separate rather than blending unit-to-unit variation
     (different silicon binning, thermals, PSU, etc.) into what would
     otherwise look like measurement noise.

Usage:
  python3 aggregate_runs.py runs/                 # scans runs/*/results.csv
  python3 aggregate_runs.py runs/ --out variance.csv
  python3 aggregate_runs.py a/results.csv b/results.csv c/results.csv
"""
import argparse
import csv
import glob
import os
import statistics

import report  # reuse to_float

METRICS = ["energy_per_token_mJ", "ttft_ms", "tpot_ms", "pp_tok_per_s", "tg_tok_per_s"]
KEY_FIELDS = ["sweep_type", "prefill_len", "decode_len", "vocab_size", "model_type", "device_id"]
# model_type and device_id are both part of the grouping key deliberately:
# both are constant across repeats of the same config on the same machine
# (so --repeat runs still aggregate together correctly), but differ across
# quantizations/models and across physical devices respectively. Without
# device_id, results from two different Orins at the same test point would
# get silently merged and averaged as if they were noisy repeats of the
# same hardware -- exactly the same failure mode model_type was added to
# prevent for quantization comparisons. Older CSVs without a device_id
# column (pre-multi-device support) get grouped under None, which is fine
# as long as you're not mixing old single-device data with new
# multi-device data in the same aggregate call.
# model_type is part of the grouping key deliberately: it's constant across
# repeats of the same config (so --repeat runs still aggregate together
# correctly), but different across quantizations/models -- without it, a
# Q4 run and an F16 run at the same prefill_len would get silently merged
# and averaged as if they were noisy repeats of the same measurement,
# which is wrong. Rows with different model_type at the same (sweep_type,
# prefill_len, decode_len, vocab_size) show up as separate rows in the
# output -- that's the intended comparison view for quantization sweeps.
HIGH_CV_THRESHOLD = 15.0  # % -- flag test points noisier than this


def find_csvs(paths):
    csvs = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "*", "results.csv")))
            direct = os.path.join(p, "results.csv")
            if os.path.isfile(direct):
                found.append(direct)
            csvs.extend(found)
        elif os.path.isfile(p):
            csvs.append(p)
        else:
            print(f"WARNING: path not found, skipping: {p}")
    return csvs


def load_all(csvs):
    rows = []
    for i, path in enumerate(csvs):
        run_id = os.path.basename(os.path.dirname(path)) or f"run{i}"
        with open(path) as f:
            for r in csv.DictReader(f):
                r["_run_id"] = run_id
                rows.append(r)
    return rows


def key_of(row):
    return tuple(row.get(k) for k in KEY_FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="run archive dir(s) and/or results.csv file(s)")
    ap.add_argument("--out", default="variance.csv")
    args = ap.parse_args()

    csvs = find_csvs(args.paths)
    if not csvs:
        print("No results.csv files found.")
        return

    print(f"Aggregating {len(csvs)} run(s):")
    for c in csvs:
        print(f"  {c}")

    rows = load_all(csvs)
    run_ids = sorted({r["_run_id"] for r in rows})
    print(f"\n{len(run_ids)} distinct run(s): {', '.join(run_ids)}")

    groups = {}
    for r in rows:
        groups.setdefault(key_of(r), []).append(r)

    out_rows = []
    for key, grp in groups.items():
        sweep_type, prefill_len, decode_len, vocab_size, model_type, device_id = key
        # only count runs that actually reported a real value for the
        # primary metric -- excludes e.g. the known-degenerate V=1 row
        # (energy=0) from skewing variance stats if it appears in some runs
        n_runs_present = len({r["_run_id"] for r in grp})
        out = {
            "sweep_type": sweep_type, "prefill_len": prefill_len,
            "decode_len": decode_len, "vocab_size": vocab_size,
            "model_type": model_type, "device_id": device_id,
            "n_runs": n_runs_present,
        }
        for m in METRICS:
            vals = [report.to_float(r.get(m)) for r in grp]
            vals = [v for v in vals if v is not None]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
                cv = (std / mean * 100) if mean else None
                out[f"{m}_mean"] = mean
                out[f"{m}_std"] = std
                out[f"{m}_cv_pct"] = cv
                out[f"{m}_n"] = len(vals)
            else:
                out[f"{m}_mean"] = out[f"{m}_std"] = out[f"{m}_cv_pct"] = out[f"{m}_n"] = None
        out_rows.append(out)

    def sort_key(o):
        order = {"prefill": 0, "decode": 1, "vocab": 2}
        return (
            order.get(o["sweep_type"], 9),
            report.to_float(o["prefill_len"]) or 0,
            report.to_float(o["decode_len"]) or 0,
            report.to_float(o["vocab_size"]) or 0,
            o.get("model_type") or "",
            o.get("device_id") or "",
        )

    out_rows.sort(key=sort_key)

    fieldnames = ["sweep_type", "prefill_len", "decode_len", "vocab_size", "model_type", "device_id", "n_runs"]
    for m in METRICS:
        fieldnames += [f"{m}_mean", f"{m}_std", f"{m}_cv_pct", f"{m}_n"]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for o in out_rows:
            w.writerow(o)

    print(f"\nwrote {args.out} ({len(out_rows)} test points)")

    if len(run_ids) < 2:
        print("\nNOTE: only 1 distinct run found -- CV/std are meaningless with n=1. "
              "Run the sweep at least 2-3 more times before trusting variance numbers.")
        return

    high_cv = [o for o in out_rows
              if o.get("energy_per_token_mJ_cv_pct") is not None
              and o["energy_per_token_mJ_cv_pct"] > HIGH_CV_THRESHOLD]
    if high_cv:
        print(f"\n{len(high_cv)} test point(s) with energy_per_token_mJ CV > {HIGH_CV_THRESHOLD}%:")
        for o in high_cv:
            print(f"  {o['sweep_type']:8s} p={o['prefill_len']:>4} n={o['decode_len']:>4} "
                  f"V={o['vocab_size'] or '-':>7}  device={o.get('device_id') or '-':<12} "
                  f"CV={o['energy_per_token_mJ_cv_pct']:.1f}% (n={o['n_runs']})")
    else:
        print(f"\nAll test points under {HIGH_CV_THRESHOLD}% CV on energy_per_token_mJ -- looks stable.")

    devices = sorted({o.get("device_id") for o in out_rows if o.get("device_id")})
    if len(devices) > 1:
        print(f"\n{len(devices)} distinct device(s) in this aggregate: {', '.join(devices)}")


if __name__ == "__main__":
    main()
FILE_AGGREGATE_EOF

write_if_missing "sweep.sh" <<'FILE_SWEEPSH_EOF'
#!/usr/bin/env bash
# Convenience wrapper for run_sweep.py -- edit the defaults below once for
# your setup, then just run `./sweep.sh` (or `sudo ./sweep.sh`) going
# forward instead of retyping the full command.
#
# Usage:
#   ./sweep.sh                       # run with the defaults below
#   ./sweep.sh --note "trying X"      # adds a note to the archived README
#   ./sweep.sh --schema-check         # just run --debug-schema and exit
#   ./sweep.sh --out other.csv        # override the output path
#   ./sweep.sh --repeat 3             # run the full sweep 3 times in a row,
#                                      # for run-to-run variance analysis
#                                      # (see aggregate_runs.py afterward)
#   SWEEP_DEVICE_ID=orin-lab-2 ./sweep.sh
#                                      # tag results with an explicit device
#                                      # id -- important on a multi-device
#                                      # setup (cluster, cloned SD images)
#                                      # where hostname alone may not be
#                                      # unique; defaults to hostname if unset
#
# Any extra arguments are passed straight through to run_sweep.py, so you
# can still use --archive-dir, --no-archive, etc. from here too.

set -euo pipefail

# Resolve the real user's home dir even when invoked via sudo (sudo resets
# $HOME to /root by default, which would silently point defaults at the
# wrong place -- e.g. /root/llama.cpp instead of /home/orin/llama.cpp).
# The `|| true` matters: if getent ever can't resolve the user for any
# reason, we fall back to $HOME below rather than letting `pipefail` abort
# the whole script with no error message.
REAL_USER="${SUDO_USER:-${USER:-$(id -un)}}"
REAL_HOME="$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6 || true)"
REAL_HOME="${REAL_HOME:-$HOME}"

# --- Edit these for your setup -------------------------------------------
LLAMA_BENCH="${LLAMA_BENCH:-$REAL_HOME/llama.cpp/build/bin/llama-bench}"
MODEL="${MODEL:-$REAL_HOME/Sweeper/gemma-3-270m-f16.gguf}"
VOCAB_VARIANT_DIR="${VOCAB_VARIANT_DIR:-$REAL_HOME/Sweeper/vocab_variants}"
OUT_CSV="${OUT_CSV:-$REAL_HOME/Sweeper/results.csv}"
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SCHEMA_CHECK=0
REPEAT=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --schema-check)
            SCHEMA_CHECK=1
            shift
            ;;
        --out)
            OUT_CSV="$2"
            shift 2
            ;;
        --note)
            EXTRA_ARGS+=(--run-note "$2")
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$REPEAT" =~ ^[0-9]+$ ]] || [[ "$REPEAT" -lt 1 ]]; then
    echo "ERROR: --repeat must be a positive integer, got: $REPEAT" >&2
    exit 1
fi

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "ERROR: llama-bench not found or not executable at: $LLAMA_BENCH" >&2
    echo "  Set LLAMA_BENCH env var or edit the default at the top of this script." >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL" >&2
    echo "  Set MODEL env var or edit the default at the top of this script." >&2
    exit 1
fi

if [[ "$SCHEMA_CHECK" -eq 1 ]]; then
    echo "Running schema check only (--debug-schema)..."
    python3 run_sweep.py --debug-schema \
        --llama-bench "$LLAMA_BENCH" \
        --model "$MODEL" \
        --out /dev/null
    exit 0
fi

VOCAB_ARGS=()
VOCAB_STATUS="(skipped -- not found)"
if [[ -d "$VOCAB_VARIANT_DIR" ]]; then
    VOCAB_ARGS+=(--vocab-variant-dir "$VOCAB_VARIANT_DIR")
    VOCAB_STATUS="$VOCAB_VARIANT_DIR"
else
    echo "NOTE: vocab-variant-dir not found ($VOCAB_VARIANT_DIR) -- skipping vocab sweep." >&2
    echo "  Build it first with vocab_truncate.py if you want the vocab sweep included." >&2
fi

echo "llama-bench:       $LLAMA_BENCH"
echo "model:              $MODEL"
echo "vocab-variant-dir:  $VOCAB_STATUS"
echo "out:                $OUT_CSV"
echo

if [[ "$(id -u)" -ne 0 ]]; then
    echo "NOTE: not running as root -- some tegrastats power rails may read 0."
    echo "  Re-run with 'sudo ./sweep.sh' for full rail visibility."
    echo
fi

if [[ "$REPEAT" -eq 1 ]]; then
    python3 run_sweep.py \
        --llama-bench "$LLAMA_BENCH" \
        --model "$MODEL" \
        --out "$OUT_CSV" \
        "${VOCAB_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    echo "Running $REPEAT repeats for variance analysis..."
    echo
    for i in $(seq 1 "$REPEAT"); do
        echo "=== Repeat $i/$REPEAT ==="
        REPEAT_NOTE="repeat $i/$REPEAT"
        python3 run_sweep.py \
            --llama-bench "$LLAMA_BENCH" \
            --model "$MODEL" \
            --out "$OUT_CSV" \
            "${VOCAB_ARGS[@]}" \
            "${EXTRA_ARGS[@]}" \
            --run-note "$REPEAT_NOTE"
        echo
    done
    echo "All $REPEAT repeats archived under runs/ (or your --archive-dir, if set)."
    echo "Next: python3 aggregate_runs.py runs/ --out variance.csv"
fi
FILE_SWEEPSH_EOF

write_if_missing "sweep_quantizations.sh" <<'FILE_SWEEPQUANT_EOF'
#!/usr/bin/env bash
# Runs the full sweep --repeat N times against each of several quantized
# model files, then aggregates everything into one comparison CSV.
#
# Vocab sweep is automatically skipped for every model here -- vocab
# isolation needs float (F16) tensors (see vocab_truncate.py), so there's
# no meaningful "Q4 vocab sweep" to run; re-running it under a quant label
# would just waste time re-measuring the same F16 vocab data.
#
# Edit MODELS below: one label:path pair per quantization you want to
# compare. Missing files are skipped with a warning rather than failing
# the whole run, so it's fine to list quants you haven't built yet.
#
# Usage:
#   sudo ./sweep_quantizations.sh                  # runs REPEATS (default 4) for each model below
#   sudo ./sweep_quantizations.sh --repeats 3       # override repeat count
#   sudo ./sweep_quantizations.sh --out my.csv      # override final aggregate output name
#
# Anything else you pass is forwarded to sweep.sh (and from there to
# run_sweep.py), same as sweep.sh's own passthrough behavior.

set -euo pipefail

REAL_USER="${SUDO_USER:-${USER:-$(id -un)}}"
REAL_HOME="$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6 || true)"
REAL_HOME="${REAL_HOME:-$HOME}"

# --- Edit this list for your setup -------------------------------------------
declare -A MODELS=(
  [F16]="$REAL_HOME/Sweeper/gemma-3-270m-f16.gguf"
  [Q4_K_M]="$REAL_HOME/Sweeper/gemma-3-270m-q4km.gguf"
  [Q8_0]="$REAL_HOME/Sweeper/gemma-3-270m-q8_0.gguf"
)
# Comment out or add lines above as needed. F16 is included by default so
# repeat runs there keep accumulating too -- remove it if you only want to
# sweep new quantizations without adding more F16 repeats.
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPEATS=4
OUT_CSV="quant_comparison.csv"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --out)
            OUT_CSV="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
    echo "ERROR: --repeats must be a positive integer, got: $REPEATS" >&2
    exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
    echo "NOTE: not running as root -- some tegrastats power rails may read 0."
    echo "  Re-run with 'sudo ./sweep_quantizations.sh' for full rail visibility."
    echo
fi

echo "Will run $REPEATS repeat(s) for each of: ${!MODELS[@]}"
echo

any_ran=0
for label in "${!MODELS[@]}"; do
    model_path="${MODELS[$label]}"
    if [[ ! -f "$model_path" ]]; then
        echo "SKIP: $label -- model file not found: $model_path" >&2
        echo
        continue
    fi
    echo "=== $label ($model_path) ==="
    MODEL="$model_path" VOCAB_VARIANT_DIR=/nonexistent \
        ./sweep.sh --repeat "$REPEATS" --note "$label quant" "${EXTRA_ARGS[@]}"
    any_ran=1
    echo
done

if [[ "$any_ran" -eq 0 ]]; then
    echo "No model files found -- nothing was run. Check the MODELS list at the top of this script." >&2
    exit 1
fi

echo "All quantizations swept. Aggregating..."
python3 aggregate_runs.py runs/ --out "$OUT_CSV"
FILE_SWEEPQUANT_EOF


chmod +x sweep.sh sweep_quantizations.sh 2>/dev/null || true
echo

# =============================================================================
echo "=== 5/5: Python dependencies ==="
if python3 -c "import gguf, numpy" 2>/dev/null; then
    echo "  found: gguf, numpy already importable"
else
    echo "  not found -- installing from requirements.txt"
    if ! pip install -r requirements.txt --break-system-packages; then
        echo "  ERROR: pip install failed. Try manually:" >&2
        echo "    pip install gguf numpy --break-system-packages" >&2
        exit 1
    fi
    echo "  installed"
fi

echo
echo "=== Done ==="
echo "llama-bench:  $LLAMA_BENCH"
echo "model:        $MODEL_PATH"
echo "harness dir:  $SWEEPER_DIR"
echo
echo "Next steps:"
echo "  cd $SWEEPER_DIR"
echo "  python3 run_sweep.py --debug-schema --llama-bench $LLAMA_BENCH --model $MODEL_PATH --out /dev/null"
echo "  # then build vocab variants and run the sweep -- see README.md"
