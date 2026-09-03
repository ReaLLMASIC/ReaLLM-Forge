# Gemma 270M workflows

This folder includes scripts for training Gemma 270M from scratch, fine-tuning, and
experimenting with LM head acceleration using Johnson-Lindenstrauss (JL) projection.

## Experiment sequence (English → Spanish)

`finetune.py` now supports the exact experiment structure below.

If you want a single runnable walkthrough, use:

```bash
bash huggingface_model/gemma/270M/demo_gradual_blend_en_es.sh
```

That demo runs the gradual blend path and prints commented commands for the two baselines.

### A) Gradual blend recipe (Softmax → ReLUMax/ReLU2Max + output norm blending)

1. obtain checkpoint
2. optional fine-tune with standard softmax
3. gradual fine-tune with alpha schedule:
   - attention: `alpha * softmax + (1-alpha) * relu_variant`
   - output norms (post-attn + post-ffn): `alpha * output_norm + (1-alpha) * raw`
   - alpha decreases from 1 → 0 and is clamped at 0, with optional `post_zero_steps`

Example:

```bash
# Stage 1: stabilize on Softmax first
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --source_lang en \
  --target_lang es \
  --source_lang_name English \
  --target_lang_name Spanish \
  --dataset_split "train[:10%]" \
  --output_dir ./runs/gemma270_softmax_stage1 \
  --total_iterations 20000 \
  --sample_frequency 1000 \
  --attention_mode softmax

# Stage 2: switch activation and continue from Stage 1 checkpoint
python huggingface_model/gemma/270M/finetune.py \
  --model_name ./runs/gemma270_softmax_stage1 \
  --dataset_config en-es \
  --source_lang en \
  --target_lang es \
  --source_lang_name English \
  --target_lang_name Spanish \
  --dataset_split "train[:10%]" \
  --output_dir ./runs/gemma270_relu2max_stage2 \
  --total_iterations 10000 \
  --sample_frequency 1000 \
  --attention_mode gradual_blend \
  --attention_activation relu2max \
  --activation_divisor 256.0 \
  --alpha_start 1.0 \
  --alpha_end 0.0 \
  --post_zero_steps 1000 \
  --blend_output_norm
```

For `relumax`, only change:

```bash
--attention_activation relumax --activation_divisor 256.0
```

### B) Baseline recipe (Softmax only)

1. obtain checkpoint
2. fine-tune with standard softmax

```bash
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --output_dir ./runs/gemma270_softmax_only \
  --total_iterations 20000 \
  --sample_frequency 1000 \
  --attention_mode softmax
```

### C) Sum baseline (Softmax + ReLU variant scores)

1. obtain checkpoint
2. fine-tune with summed attention probabilities/scores

```bash
python huggingface_model/gemma/270M/finetune.py \
  --model_name google/gemma-3-270m \
  --dataset_config en-es \
  --output_dir ./runs/gemma270_sum_relu2max \
  --total_iterations 20000 \
  --sample_frequency 1000 \
  --attention_mode sum \
  --attention_activation relu2max \
  --activation_divisor 256.0
```

### Important implementation note

The activation swap is implemented by monkey-patching `torch.nn.functional.softmax`
during training, so this is intended as a practical experiment path and not a
production-safe kernel-level replacement.

After each run, the script prints **multi-shot translation outputs for 3 fixed EN→ES
test sentences** (`--print_multishot_after_train` defaults to true).

## Plot validation loss per iteration (Softmax vs ReLUMax vs ReLU2Max)

After running three stage-2 experiments (one each for softmax / relumax / relu2max),
you can plot validation-loss curves from each run's `trainer_state.json`:

```bash
python huggingface_model/gemma/270M/plot_validation_loss.py \
  --run "softmax=./runs/gemma270_softmax_stage2" \
  --run "relumax=./runs/gemma270_relumax_stage2" \
  --run "relu2max=./runs/gemma270_relu2max_stage2" \
  --title "Gemma 270M EN→ES: validation loss per iteration" \
  --output ./runs/gemma270_en_es_val_loss.png
```

## Benchmark EN→ES translation quality

You can run a quick benchmark on a held-out slice with exact-match plus BLEU/chrF
(if `sacrebleu` is installed):

```bash
python huggingface_model/gemma/270M/benchmark_en_es_translation.py \
  --model_name ./runs/gemma270_relu2max_stage2 \
  --dataset_config en-es \
  --dataset_split "train[10%:11%]" \
  --num_samples 200 \
  --max_new_tokens 64
```

## Loading a pre-trained checkpoint

Use the Hugging Face model hub (or a local checkpoint directory) as the `model_name`:

```bash
python huggingface_model/gemma/270M/jl_head_eval.py \
  --model_name google/gemma-3-270m \
  --dataset_split "train[:1%]" \
  --fineweb_subset sample-10BT
```

The scripts rely on `AutoModelForCausalLM.from_pretrained(...)`, so you can point
`--model_name` at any local checkpoint path created by `train_from_scratch.py` or
`finetune.py`.

## JL-projected LM head evaluation

`jl_head_eval.py` runs a two-stage LM head evaluation:

1. Project hidden states and LM head weights into a lower `target_dimension` using
   a JL random matrix.
2. Compute approximate logits, select the top-`n` candidate tokens, and then compute
   exact logits only for that subset.

It reports the average absolute token-id difference between the estimated top-k
tokens and the exact top-k tokens, and plots a heatmap for a sweep of `top_n`
values and `target_dimension` sizes.

**How `avg_id_delta` is computed**

For each token position, the script compares the top-`k` token IDs from the
exact logits against the top-`k` token IDs from the JL-approximated logits. It
sorts both ID lists, computes the elementwise absolute difference between the
sorted IDs, and averages those differences across all token positions. It also
tracks how many IDs changed when comparing the two top-`k` sets (order does not
matter), and summarizes that count with the same mean/median/min/max/std stats.

Example (top-`k` = 5):

* Exact top-`k` IDs: `[2, 7, 11, 19, 42]`
* Estimated top-`k` IDs: `[3, 8, 11, 21, 40]`
* Absolute differences: `|2-3|, |7-8|, |11-11|, |19-21|, |42-40|`
* Average: `(1 + 1 + 0 + 2 + 2) / 5 = 1.2`

```bash
python huggingface_model/gemma/270M/jl_head_eval.py \
  --model_name google/gemma-3-270m \
  --dataset_split "train[:1%]" \
  --fineweb_subset sample-10BT \
  --eval_tokens 1000 \
  --top_k 65 \
  --top_n_values 1000,2000,5000,10000 \
  --target_dimensions 500,400,300,200,100 \
  --projection orthonormal \
  --approx_logits_device cpu \
  --approx_logits_dtype float16 \
  --approx_chunk_size 5000 \
  --exact_chunk_size 5000 \
  --annotate_stats true \
  --output_dir jl_eval_outputs
```

## Token-angle dashboard + island exports

`vocab_angle_token_dashboard.py` adds an interactive Plotly dashboard for selected
tokens and exports connected-component island files for the full vocab graph at a
chosen angle threshold.

Highlights:

* `--tokens`: comma-separated tokens to inspect (e.g., digits, months, weekdays).
* `selected_token_dashboard.html`: interactive histogram + token-id scatter of
  selected-token angles to the rest of the vocab.
* `selected_token_reports/*.csv`: per-selected-token nearest-angle lists.
* `islands/*.txt`: one uniquely named file per connected island (`uuid` suffix),
  with token id, token string, and graph degree.

Demo scripts:

```bash
bash ./demo_angle_dashboard_digits.sh
bash ./demo_angle_dashboard_months.sh
bash ./demo_angle_dashboard_weekdays.sh
bash ./demo_islands_20deg.sh
```

## Interactive webapp: angle-neighborhood explorer (Gemma + Gemma-IT)

`vocab_angle_explorer_app.py` runs a local Dash webapp that compares:

* `google/gemma-3-270m`
* `google/gemma-3-270m-it`

It visualizes, per selected token, how many vocab vectors fall within angle
bins over a configurable range (defaults `10` to `90` degrees with 10-degree
stack bins).

Features:

* Presets: `digits` (default), `weekdays`, `months`, `alphabet`, `all`
* Regex token filter
* Sort mode: alphabetical, highest→lowest, lowest→highest
* Stacked histogram in both models
* Click a token bar, then export a CSV listing all tokens with:
  token id, degree separation, dot product, normalized dot product

Run:

```bash
python ./vocab_angle_explorer_app.py \
  --model-base google/gemma-3-270m \
  --model-it google/gemma-3-270m-it \
  --device cpu \
  --port 8050 \
  --output-dir ./gemma_angle_explorer_exports
```

## Digit-only quantization angle comparison

`digit_quant_angle_comparison.py` compares pairwise angles for digits `0-9`
under full precision (`fp32`) and symmetric quantization modes:

* int8, int7, int6, int5, int4, int3
* ternary
* binary

Outputs:

* `angles_<mode>.csv` angle matrix for each mode.
* `digit_<d>_relative_angles.png` one plot per digit, showing baseline vs all
  quantized curves against all other digits.
* `digit_token_ids.csv` for the resolved token IDs.
* `tsne_structure_all_modes.png` for approximate global structure view.
* `distortion_<token>.png` signed angular distortion per token vs fp32 baseline.
* `relative_angles_selector.html` interactive Plotly page with token dropdown and
  per-quantization disorder labels indicating rank-order changes vs fp32.

Demo:

```bash
bash ./demo_digit_quant_angles.sh
bash ./demo_weekday_quant_angles.sh
bash ./demo_month_quant_angles.sh
## Latin/punctuation/other manual router (OPUS-100 en-es)

`latin_punct_router_eval.py` builds three token groups from the Gemma vocabulary:

1. Tokens whose decoded UTF text contains at least one Latin-script codepoint.
2. Tokens that are punctuation/layout-only (Unicode punctuation plus common English/Spanish punctuation symbols, and whitespace/layout tokens like tabs/newlines).
3. Everything else (including byte/special tokens).

It then:

- prints a raw-count and percentage table for all three groups,
- computes one average LM-head vector per group,
- L2-normalizes those three vectors into unit routing prototypes,
- routes each decoding step by dot product of final-layer hidden state with the 3 prototypes,
- and performs next-token scoring only within the routed token subset.
- reports side-by-side teacher-forced token accuracy with and without routing.
- prints validation-set example translations before (full LM head) and after (routed LM head).
- color-highlights the actually generated continuation segment (excluding the fixed multi-shot prompt context).
- uses a fixed 3-shot English→Spanish prompt template before each evaluated/generated example.
- can optionally freeze the model and train 3 route scalars (initialized to 1.0) for the 3 prototypes on OPUS-100 `en-es`.

Example:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --model_name google/gemma-3-270m-it \
  --split "train[:1%]" \
  --max_samples 100 \
  --max_target_tokens 64 \
  --example_split "validation[:20]" \
  --num_examples 3 \
  --example_max_new_tokens 64 \
  --route_mode three_way \
  --byte_fallback \
  --device cuda
```

Notes:

- Evaluation is teacher-forced next-token prediction over OPUS-100 `en-es` translation targets.
- The script uses cosine-style scoring (unit-normalized hidden state and LM-head rows) for both routing and routed token selection.


Alternative 2-way routing (latin+punct vs other):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_vs_other \
  --byte_fallback
```

Use `--no-byte_fallback` to disable unioning byte tokens into non-`other` candidate sets.

Alternative 2-way routing that never selects `other` (latin vs punct only):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_vs_punct_only \
  --byte_fallback
```

Single-bucket latin+punct mode (no routing decision; only score in latin+punct section):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_only \
  --byte_fallback
```

Train only the three route scalars (model frozen) for `three_way` mode:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode three_way \
  --train_route_scales \
  --train_split "train[:1%]" \
  --train_max_samples 100 \
  --train_epochs 1 \
  --train_lr 1e-2
```

Interactive chat mode (type your own English sentence and compare full vs routed translation):

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --chat_mode \
  --route_mode three_way \
  --example_max_new_tokens 64
```

In chat mode, user input is color-highlighted and generated continuations are also highlighted in the printed full outputs.

Latin-trim sweep report mode (trim longest UTF-8 latin tokens from 0%..80% in 10% steps, emit report + graph):

```bash
bash huggingface_model/gemma/270M/demo_latin_trim_sweep.sh
```

The demo runs both trim strategies (`longest_bytes`, `highest_id`) and then writes a combined comparison plot:

- `latin_trim_reports_combined_accuracy.png` (full LM head + both routed strategies).
- `latin_trim_reports_combined_scores.csv` (raw scores from all plotted series and quantization bars).
- `latin_trim_reports_combined_accuracy.html` (interactive Plotly version of the main trim/two-pass comparison chart).
- The same demo also runs a quantization sweep (8/6/5/4/3 bits; vector/group32; symmetric/asymmetric) on the 100% latin+punct(+byte) candidate set and includes it in the combined plot.
- The demo now also runs two-pass experiments: first pass sweeps low-precision configs (`group32` and `vector` × `asymmetric`/`symmetric` × int4/int3), and second pass reranks shortlist sizes `top-1/top-10/top-100/top-1000/top-10000` in settable higher precision (`float16`/`bfloat16`/`float32`). Curves are included for both trimmed and untrimmed candidate variants in the combined plot.

Direct CLI equivalent:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_only \
  --latin_trim_sweep \
  --latin_trim_strategy longest_bytes \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --report_dir latin_trim_reports
```

This writes:

- `latin_trim_reports/latin_trim_sweep.csv`
- `latin_trim_reports/latin_trim_sweep_accuracy.png`
- `latin_trim_reports/latin_trim_sweep_report.txt`
- `latin_trim_reports/latin_trim_00.txt`, `latin_trim_reports/latin_trim_10.txt`, ... per-percent details
- `latin_trim_reports/latin_trim_00_latin_tokens.json`, `latin_trim_reports/latin_trim_10_latin_tokens.json`, ... per-percent latin token arrays (`id`, `token`, `length_bytes`, sorted by descending `length_bytes`)

The sweep also computes an additional score: average ASCII-string difference (via sequence ratio) between prediction and reference before the first newline.

You can also run a single trim without sweep via `--latin_trim_percent <pct>`.

Trimming strategy can be selected with `--latin_trim_strategy`:

- `longest_bytes` (default): trim by largest UTF-8 token byte length first.
- `highest_id`: trim by highest token id first.

Standalone quantization sweep:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --route_mode latin_punct_only \
  --quantization_sweep \
  --quant_group_size 32 \
  --quant_report_dir quantization_reports
```

## LM-head angle webapp (token-to-token geometry)

`lm_head_angle_webapp.py` provides a Streamlit UI for inspecting LM-head vector geometry:

- Pairwise mode: select any two vocabulary tokens and compute their angle in degrees plus each vector magnitude.
- Single-token mode: select one token and produce a full nearest→furthest angle-sorted vocabulary list, including magnitudes.
- Token picker supports both selection by token ID and case-insensitive substring matching across raw tokens and display-normalized variants, so typing partial strings like `refix` can match `prefix` and `Hello` can surface `hello`, `_hello`, `Hello`, etc. (depending on tokenizer vocabulary entries).

Run:

```bash
pip install streamlit transformers torch pandas
streamlit run huggingface_model/gemma/270M/lm_head_angle_webapp.py
```

Notes:

- Default model is `google/gemma-3-270m`; replace in the sidebar if you want `-it` or a local checkpoint path.
- For very large vocab scans, CPU mode may take longer; CUDA is supported when available.
Standalone trimmed-vocab two-pass sweep:

```bash
python huggingface_model/gemma/270M/latin_punct_router_eval.py \
  --two_pass_trim_sweep \
  --two_pass_first_configs group32_asymmetric:4,group32_symmetric:4,vector_asymmetric:4,vector_symmetric:4,group32_asymmetric:3,group32_symmetric:3,vector_asymmetric:3,vector_symmetric:3 \
  --two_pass_first_group_size 32 \
  --two_pass_second_topn_values 1,10,100,1000,10000 \
  --two_pass_second_dtype float16 \
  --latin_trim_strategy longest_bytes \
  --latin_trim_sweep_max 80 \
  --latin_trim_sweep_step 10 \
  --two_pass_report_dir two_pass_trim_reports
```
