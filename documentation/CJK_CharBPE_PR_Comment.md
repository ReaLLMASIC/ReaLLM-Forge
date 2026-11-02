The char vs charBPE pipeline now uses the published tokenized TED-2020 datasets directly. It combines six char baselines from `cjk_char_byte_fallback_prepared` with six charBPE datasets from `cjk_charbpe_tokenized`, verifies manifest sizes/SHA256, and skips source preprocessing, IPA conversion, and BPE training.

Run in a Python 3.10+ environment with CUDA-compatible PyTorch and torchaudio installed:

```bash
git clone --branch feat/cjk-char-byte-fallback-metrics --single-branch \
  https://github.com/xinyixuu/nanoGPT.git
cd nanoGPT
python3 -m pip install -r requirements_gpu.txt

# Check the default bfloat16 training requirements.
python3 -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))'

# Fresh output/log directories preserve earlier experiments.
export PYTHON_BIN="$(command -v python3)"
export CUDA_VISIBLE_DEVICES=0
export CJK_DATA_ROOT="$PWD/data/cjk_ipa_charbpe_hf"
JOB_ROOT="$(mktemp -d "$PWD/cjk_charbpe_run.XXXXXX")"
export CJK_OUT_ROOT="$JOB_ROOT/out"
export CJK_LOG_ROOT="$JOB_ROOT/logs"

# Download both tokenized packages and train all 12 models in the background.
bash demos/run_cjk_nohup.sh charbpe download-train
printf 'Job directory: %s\n' "$JOB_ROOT"

# Ctrl-C stops tail; training keeps running.
tail -f "$CJK_LOG_ROOT/nohup.log"
```

After reconnecting, use the printed absolute job directory:

```bash
cat /absolute/path/to/cjk_charbpe_run.XXXXXX/logs/status
cat /absolute/path/to/cjk_charbpe_run.XXXXXX/logs/exit_code
cat /absolute/path/to/cjk_charbpe_run.XXXXXX/out/comparison_summary.tsv
```

`status` is `RUNNING`, `0` (success), or `1` (failure). Launcher success only confirms that the background process started. Outputs include `comparison_summary.tsv`, checkpoints, training token counts, and per-token CSV/HTML/PNG reports for BPB, target probability, target rank, and probability mass ahead of the target.

Defaults: Japanese/Korean/Chinese × original/IPA × char/charBPE; 12 layers, 12 heads, embedding 768, MLP 3072, batch 16, context 1024, 3 epochs, QK norm + scale, torch.compile, bfloat16, TensorBoard off. Runs execute sequentially on one GPU.

Optional alternatives (choose these instead of the background launch):

```bash
# Download only; training can later run offline with the same CJK_DATA_ROOT.
bash demos/cjk_ipa_charbpe_compare.sh download
bash demos/cjk_ipa_charbpe_compare.sh train

# Train only the Japanese original-text pair, with fresh output/log roots.
RUN_FILTER='^ja_original_' bash demos/run_cjk_experiment.sh charbpe download-train
```

Use a fresh `CJK_OUT_ROOT` / `CJK_LOG_ROOT` for each new experiment. `HF_DATASET_REVISION` can pin a Hub commit; both packages use the same resolved revision. `all` intentionally regenerates preprocessing/tokenization; use `download-train` for the uploaded data.
