# CJK char / char-BPE / byte-fallback experiments

Both entry points share normalization, train/validation splits, IPA conversion,
model settings, BPB reporting and the three validation per-token metrics from
[PR #917](https://github.com/ReaLLMASIC/ReaLLM-Forge/pull/917).

| Entry point | Comparison | Default data/output/log suffix |
| --- | --- | --- |
| `demos/cjk_ipa_charbpe_compare.sh` | char vs char-BPE (8192 vocabulary) | data: `cjk_ipa_charbpe`; output/log: `cjk_ipa_charbpe_12l_3epochs` |
| `demos/cjk_ipa_char_byte_fallback_compare.sh` | char vs character whitelist + UTF-8 byte fallback | data: `cjk_ipa_char_byte_fallback`; output/log: `cjk_ipa_char_byte_fallback_12l_3epochs` |

Each experiment has 12 runs: Japanese / Korean / Chinese × original / IPA ×
two tokenizers. Defaults: 12 layers, 12 heads, embedding dimension 768,
MLP hidden size 3072
(`MLP_SIZE`, default `4 * N_EMBD`), block size 1024, batch size 16,
QK normalization and QK norm scale enabled, `torch.compile` enabled,
TensorBoard disabled, learning rate 0.001, seed 1337 and bfloat16.
Each run uses `--training_limiters max_epochs --max_epochs 3`; `MAX_EPOCHS`
overrides the epoch limit. The iteration limiter is not active. Evaluation
runs every 500 iterations using 50 sampled batches per split. Epochs are
measured as cumulative sampled training tokens divided by the training split
token count, so tokenizers have different iteration counts and the last batch
may slightly exceed 3 epochs. Sampling remains the trainer’s random-window
sampling; an epoch is not a shuffled pass over every window.
Training runs sequentially on `cuda:0`;
`CUDA_VISIBLE_DEVICES` chooses the physical GPU.

## Byte-fallback vocabulary

The new demo defaults to `FALLBACK_CHAR_LIMIT=7936`, giving a total vocabulary
of at most 8192 including the 256 byte tokens. No unused IDs are added to pad
smaller vocabularies. For each language and
representation independently, the whitelist contains the K most frequent
non-whitespace Unicode characters in the training split. Frequency ties are
broken by Unicode order. Validation text is never used to select this whitelist.
Whitespace is encoded as bytes because the existing line-based custom tokenizer
strips whitespace from its vocabulary file.

The demo calls `prepare.py --method custom_char_byte_fallback`. IDs 0–255
represent raw UTF-8 bytes; IDs 256 onward represent whitelist characters.
Unknown characters are preserved through their bytes. Actual vocabulary size
is `256 + min(K, available non-whitespace training characters)`. The baseline
char tokenizer retains its existing behavior of covering train and validation
characters. It is not a train-only/OOV baseline.

If K covers every non-whitespace training character, this comparison primarily measures the
additional byte vocabulary, changed IDs, and validation-only characters.
Several IPA streams have fewer than 256 distinct characters. Use a lower K
when you want fallback to handle more of their observed characters.

To use a fixed whitelist, set `FALLBACK_CHARS_FILE=/absolute/path/characters.txt`.
It must contain one Unicode character per line, with no duplicates. Multi-character
phonemes/tokens are rejected so this experiment remains character-level. The
same file applies to every selected stream; use `RUN_FILTER` to run streams with
different whitelist files separately. A lower-level alternative is
`COMPARISON_TOKENIZER=byte`, which compares char against pure bytes with a fixed
256-token vocabulary.

## On the other machine

### Download tokenized data and train directly

For a ready-to-paste PR comment with clone, setup, launch and monitoring commands,
see [CJK_CharBPE_PR_Comment.md](CJK_CharBPE_PR_Comment.md).

The [published prepared package](https://huggingface.co/datasets/xinyixuu/ted-2020/tree/main/cjk_char_byte_fallback_prepared)
contains the char baselines and character-byte-fallback datasets (K=7936). The
[new charBPE package](https://huggingface.co/datasets/xinyixuu/ted-2020/tree/main/cjk_charbpe_tokenized)
contains six charBPE datasets. The charBPE download stage combines the six char
baselines from the first package with the six charBPE datasets from the second,
covering Japanese, Korean and Chinese in original and IPA representations. Both
packages are pinned to the same Hub commit during each download. After installing
PyTorch with CUDA support and `requirements_gpu.txt`, choose one launch command:

```bash
# Download all 12 char vs charBPE datasets, then train sequentially on one GPU.
bash demos/run_cjk_experiment.sh charbpe download-train

# Or detach the same char vs charBPE experiment with status/log tracking.
bash demos/run_cjk_nohup.sh charbpe download-train

# Separate comparison: char vs byte fallback.
bash demos/run_cjk_experiment.sh byte-fallback download-train

# Or launch in the background with the existing status/log tracking.
bash demos/run_cjk_nohup.sh char_byte_fallback download-train

# Download only; no GPU or IPA conversion dependencies are needed for this stage.
bash demos/cjk_ipa_char_byte_fallback_compare.sh download
# Later (including offline):
bash demos/cjk_ipa_char_byte_fallback_compare.sh train
```

These are alternative launch commands; choose one to avoid duplicate training.
The existing `prepare`, `all` and `train` stages keep their previous behavior.
`download-train` skips source downloads, normalization, IPA conversion and
tokenization entirely. `espeak-ng` and the IPA converter packages are needed
only for local `prepare`/`all`, not for the prepared-data path.

```bash
# Example: download and train only the Japanese original-text pair.
RUN_FILTER='^ja_original_' \
CJK_DATA_ROOT="$PWD/data/cjk_downloaded" \
CJK_OUT_ROOT="$PWD/out/cjk_downloaded" \
CJK_LOG_ROOT="$PWD/logs/cjk_downloaded" \
bash demos/run_cjk_experiment.sh charbpe download-train
```

The downloader selects only the requested dataset directories, including
`train.bin`, `val.bin`, `meta.pkl`, preparation markers, whitelists, and charBPE
vocabulary/count JSONs. It excludes
`_source` text and checks file sizes and SHA256 against `manifest.json` before
installing files under `CJK_DATA_ROOT/<run_name>/`. Already matching local files
are reused; differing existing files require a fresh data root or moving them
aside. Metadata is validated before training. Custom `FALLBACK_CHAR_LIMIT` or
`FALLBACK_CHARS_FILE` settings require local preparation rather than downloading
the fixed published fallback vocabularies.

`HF_DATASET_REPO` and `HF_DATASET_REVISION` select the dataset repository/version;
the downloader pins all binaries to the manifest's commit. `HF_PREPARED_SUBDIR`
defaults to `cjk_char_byte_fallback_prepared` for char/fallback datasets;
`HF_CHARBPE_SUBDIR` defaults to `cjk_charbpe_tokenized` for charBPE datasets.
The downloader supports both manifest formats and installs both packages into
one data root. Published charBPE vocabularies have 8192 IDs including 256 byte
fallback IDs; use `prepare`/`all` with a separate data root for other vocabulary sizes.

Clone the feature branch (replace the directory name if desired):

```bash
git clone --branch feat/cjk-char-byte-fallback-metrics --single-branch \
  https://github.com/xinyixuu/nanoGPT.git
cd nanoGPT
```

Create/activate your Python environment, install CUDA-compatible `torch` and
`torchaudio` for that machine, then install repository dependencies:

```bash
python3 -m pip install -r requirements_cpu.txt
python3 -m pip install pytest
# Ubuntu/Debian: Korean IPA conversion needs espeak-ng.
sudo apt-get install espeak-ng
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
```

`requirements_cpu.txt` contains the general Python dependencies and IPA
converters; GPU-specific optional packages are not required for this demo.
Use `DTYPE=float32` or `float16` if needed for your GPU.
The `prepare` and `all` stages automatically download `ja.txt`, `ko.txt` and
`zh_cn.txt` from [xinyixuu/ted-2020](https://huggingface.co/datasets/xinyixuu/ted-2020)
to `data/ted-2020` using `huggingface_hub`. Downloads use the Hub cache.
Set `HF_DATASET_REVISION` to a commit hash to pin a dataset version;
`HF_DATASET_REPO` and `HF_DATASET_DIR` override the repository and download
directory. `train` uses prepared files and does not download source text.
For offline preparation, supply all three local UTF-8 corpus paths explicitly.
To use the uploaded data for the full char-BPE experiment:

```bash
bash demos/run_cjk_experiment.sh charbpe download-train
```

For local source overrides and byte-fallback comparison:

```bash
export PYTHON_BIN="$(command -v python3)"
export JA_TEXT=/path/to/ja.txt
export KO_TEXT=/path/to/ko.txt
export ZH_CN_TEXT=/path/to/zh_cn.txt
export FALLBACK_CHAR_LIMIT=7936

# Optionally select one physical GPU.
export CUDA_VISIBLE_DEVICES=0

bash demos/cjk_ipa_char_byte_fallback_compare.sh prepare
bash demos/cjk_ipa_char_byte_fallback_compare.sh train
# Or run both stages: bash demos/cjk_ipa_char_byte_fallback_compare.sh all
```

For pure byte comparison:

```bash
COMPARISON_TOKENIZER=byte bash demos/cjk_ipa_char_byte_fallback_compare.sh all
```

## Background execution with nohup and exit codes

`demos/run_cjk_experiment.sh` wraps either experiment and normalizes its exit
code: successful completion returns **0**; any nonzero underlying exit code
returns **1**. Both experiments default to downloading the three TED corpora from
`xinyixuu/ted-2020`. `JA_TEXT`, `KO_TEXT` and `ZH_CN_TEXT` optionally override
individual files with local paths. From the
repository root, launch the full char versus byte-fallback experiment with:

```bash
# Unique directories preserve previous runs and give each job its own status.
JOB_DIR="$(mktemp -d "$PWD/cjk_job.XXXXXX")"
export PYTHON_BIN="$(command -v python3)"
export CUDA_VISIBLE_DEVICES=0
export FALLBACK_CHAR_LIMIT=7936
export CJK_DATA_ROOT="$PWD/data/cjk_ipa_fallback_vocab8192"
export CJK_OUT_ROOT="$JOB_DIR/out"
export CJK_LOG_ROOT="$JOB_DIR/logs"
export CJK_EXIT_CODE_FILE="$JOB_DIR/exit_code"

nohup bash demos/run_cjk_experiment.sh byte-fallback all \
  >"$JOB_DIR/nohup.log" 2>&1 < /dev/null &
CJK_PID=$!
printf 'pid=%s\njob_dir=%s\n' "$CJK_PID" "$JOB_DIR"

# Follow progress (Ctrl-C stops tail only).
tail -f "$JOB_DIR/nohup.log"
```

For char versus char-BPE, replace `byte-fallback` with `charbpe` and use
`CJK_DATA_ROOT="$PWD/data/cjk_ipa_charbpe"`. Replace `all` with `prepare` or
`train` to run just one stage. Training requires prepared datasets.

The optional `CJK_EXIT_CODE_FILE` contains `running` during execution, then
`0` or `1` when the wrapper exits. Read it after reconnecting using the printed
absolute job directory: `cat /absolute/job/directory/exit_code`. A missing file
can mean the wrapper has not started yet. Forced termination with `SIGKILL`,
machine shutdown, or a failure to write the file can leave the status missing
or at `running`; check the process and log in those cases.

In the **same shell that launched the job**, obtain the actual final exit code:

```bash
if wait "$CJK_PID"; then
  CJK_STATUS=0
else
  CJK_STATUS=1
fi
printf 'Final exit code: %s\n' "$CJK_STATUS"
# In an automation shell script, propagate it with: exit "$CJK_STATUS"
```

`$?` immediately after `nohup ... &` describes background launch, not experiment
completion. A new shell cannot `wait` on the old shell's child; use the status
file instead. Errors are determined by process exit codes and the pipeline's
output validation, not by searching logs for the word `error`.

Use distinct artifact roots for different K values, whitelists, seeds or model
settings. For example:

```bash
FALLBACK_CHAR_LIMIT=64 \
CJK_DATA_ROOT="$PWD/data/cjk_ipa_fallback_k64" \
CJK_OUT_ROOT="$PWD/out/cjk_ipa_fallback_k64" \
CJK_LOG_ROOT="$PWD/logs/cjk_ipa_fallback_k64" \
bash demos/cjk_ipa_char_byte_fallback_compare.sh all
```

Preparation fingerprints include the source files, tokenizer code, K and fixed
whitelist contents. Cached datasets are rebuilt when those inputs change.
Training does not automatically overwrite or resume partial runs. Use a fresh
output directory to collect a complete metric history.

## Retrain the existing char-BPE experiment with new metrics

Existing datasets can be reused on the current machine. Keep the old outputs
and point training at new output/log directories:

```bash
CJK_OUT_ROOT="$PWD/out/cjk_ipa_charbpe_12l_3epochs" \
CJK_LOG_ROOT="$PWD/logs/cjk_ipa_charbpe_12l_3epochs" \
bash demos/cjk_ipa_charbpe_compare.sh train
```

On a fresh machine, use `download-train` to fetch the published tokenized data
and train directly. Use `all` only when intentionally rebuilding tokenization. The new
metrics cannot be recovered for earlier evaluation snapshots from old CSV
losses alone; full metric histories require retraining.

## Outputs and interpretation

For a detached run with a status file, use the launcher (it invokes `nohup`):

```bash
# Current machine: reuse prepared char-BPE datasets and retrain with new metrics.
bash demos/run_cjk_nohup.sh charbpe train

# Fresh machine: download tokenized data and train char vs charBPE directly.
bash demos/run_cjk_nohup.sh charbpe download-train
```

The default char-BPE log root is `logs/cjk_ipa_charbpe_12l_3epochs`.
The byte-fallback launch uses `logs/cjk_ipa_char_byte_fallback_12l_3epochs`.
Monitor a char-BPE run with:

```bash
tail -f logs/cjk_ipa_charbpe_12l_3epochs/nohup.log
cat logs/cjk_ipa_charbpe_12l_3epochs/status
cat logs/cjk_ipa_charbpe_12l_3epochs/pid
```

`status` is `RUNNING` while active, `0` when the whole selected pipeline succeeds,
and `1` when it exits with an error. `exit_code` retains the original exit code;
`started_at` and `finished_at` record UTC timestamps. A successful launcher exit
only means the job was started: use the status file to check experiment completion.
Each launch reserves its log root using `.nohup-run`; choose a new log/output root
for another attempt. Normal termination signals are recorded as failure; SIGKILL
or a machine shutdown cannot run the exit handler.

The original text and tokenized datasets are hosted on Hugging Face.
Use `download-train` for published data, or `all` to regenerate IPA/tokenization.
Alternatively, transfer the prepared dataset tree and use `train` with its
`CJK_DATA_ROOT`; training-only execution does not read the original text files.

Each run writes `full_config.json`, `ckpt.pt`, `best_val_loss_and_iter.txt` and
`per_token_metrics/`. The report directory includes detailed and summary CSVs,
six added probability/rank overview/history HTML pages, and eight static PNG
sort orders per evaluation. The three new CSV columns are:

- `avg_target_probability`: target softmax probability, higher is better.
- `avg_target_rank`: one-based rank, lower is better; ties share the best rank.
- `avg_left_probability`: probability mass strictly ahead of the target,
  lower is better; the target and tied competitors are excluded.

Each completed run also logs and writes `training_token_stats.json`. The
`comparison_summary.tsv` includes these counts for every tokenizer:

- `vocab_size`: actual number of vocabulary IDs, including unused byte IDs.
- `training_unique_token_ids`: number of distinct target IDs actually seen in
  training, excluding IDs whose cumulative `training_seen_count` is zero.
- `training_tokens_processed`: total training target occurrences, counting
  repeated tokens and repeated sampling of data, excluding padding (`-1`).
- `training_tokens_iteration`: latest exported snapshot used for these counts.

Counts come from the latest per-token snapshot, not the best-loss snapshot;
cumulative counts are never added across snapshots. They exclude validation
and training-split evaluation batches. Existing completed runs can regenerate
the JSON and log counts when selected by the `train` stage without retraining.
`best_tokens` remains the existing count at the best validation-loss iteration.

All three are validation-only means over positions with that target token.
Tokens not sampled in validation have `NaN`. Padding targets (`-1`) are ignored.
`per_token_summary.csv` describes the distribution across populated token IDs.

`comparison_summary.tsv` adds `val_avg_target_probability`, `val_avg_target_rank`
and `val_avg_left_probability` at each run's **best validation-loss iteration**.
These three aggregates are weighted by `val_eval_count`, so they represent
sampled validation positions rather than an unweighted average of token IDs.
They should not be interpreted as tokenizer-independent measures: vocabularies,
token units and context spans differ. BPB is the primary cross-tokenizer score
within the same representation. Original and IPA BPB use different byte streams
and cannot directly establish which representation is better. The existing
Chinese IPA converter has known mixed-number/Hanzi issues.

Model depth/width and the epoch budget match across runs. Processed token
counts differ with tokenization. Parameter counts,
processed raw bytes and FLOPs are not matched. This is a single-seed experiment.
Reports use Plotly CDN for interactive views; CSVs and static PNGs work offline.

## Git branch and preservation

The feature branch contains both demos, core metric code, tests and this guide.
The old pipeline and its three new metrics remain available. Commit code paths
explicitly; do not use `git add .` to include generated data accidentally.
Datasets, checkpoints, logs and the local results index are kept on the original
machine and are not needed to clone the code branch.

For subsequent code changes:

```bash
git switch feat/cjk-char-byte-fallback-metrics
git add demos/cjk_ipa_charbpe_compare.sh \
  demos/cjk_ipa_char_byte_fallback_compare.sh \
  data/template/utils/build_char_fallback_vocab.py \
  utils/cjk_comparison_metrics.py tests/test_cjk_fallback_pipeline.py \
  documentation/CJK_Tokenizer_Experiments.md \
  README.md utils/per_token_metrics.py utils/per_token_html.py \
  utils/per_token_static.py tests/test_per_token_metrics.py
git commit -m 'feat(experiments): add CJK char byte-fallback comparison with metrics'
git push -u origin feat/cjk-char-byte-fallback-metrics
```
