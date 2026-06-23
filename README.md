# ReaLLMASIC

## Overview

ReaLLMAsic aims to bridge the gap between theoretical model design and practical
hardware implementation, ensuring efficient, scalable, and robust ML model
development.

Our project stands out for its extensive exploration of various model
configurations and modules, catering to a diverse range of use cases.

Key exploration features include:

* `Module Variation`: Explore with different module types -- e.g. Softmax, Softermax, ConSmax, and SigSoftmax -- discover which is best suited (PPA) to your application.
* `Flexible Tokenization`: Explore different tokenization: tiktoken, sentencepiece, phonemization, character level, custom tokenization, etc.
* `Diverse Dataset Performance Testing`: Evaluate model efficacy across various languages and datasets including: csv-timeseries, mathematics, music, lyrics, literature, and webtext.
* `Standard and Custom Hyperparameters`: Fine-tune models using conventional hyperparameters and explore the impact of custom settings on model performance and PPA impacts.

Key analysis features:

* `Exploration scripts`: Are encapsulated into bash scripts which loop over the train.py's argparse parameters.
* `Logging with automatic timestamps & labels`: run a suite of experiments and have the repo automatically organize and label them by timestamp and description

Hardware Related
* `Training with Hardware Emulation`: Implement different operations for forward and backward passes for hardware-implementation aware training.
* `PPA Implications Analysis`: Understand the power, performance, and area (PPA) implications of different model designs, guiding efficient hardware-software integration.


# TOC

* [Overview](#overview)
* [Installation](#installation)
  * [Step 1 (Recommended) Adding a Virtual Env](#step-1-recommended-adding-a-virtual-env)
  * [Step 2 Install Dependencies](#step-2-install-dependencies)
* [Testing Your Setup](#testing-your-setup)
  * [Prepare Training and Validation Data Sets](#prepare-training-and-validation-data-sets)
  * [Train Model From Scratch](#train-model-from-scratch)
  * [Perform Inference From Custom Model](#perform-inference-from-custom-model)
* [Explorations](#explorations)
  * [Start Exploration](#start-exploration)
  * [Inspect and Monitor Best Val Losses](#inspect-and-monitor-best-val-losses)
  * [Start Tensorboard Logging](#start-tensorboard-logging)
  * [Model Stats Table](#model-stats-table)
  * [Troubleshooting](#troubleshooting)
  * [Creating New Features and Exploration Scripts](#creating-new-features-and-exploration-scripts)
* [Contributing](#contributing)
* [Acknowledgements](#acknowledgements)

## Installation

This section contains installation locally with GPU acceleration.

(If you do not have a GPU, check out this [colab](./colabs/NanoGPT_Quickstart.ipynb), which has a T4 GPU
runtime (at time of writing) for ML acceleration.)

### Step 1 (Recommended) Adding a Virtual Env

We recommend creating a virtual env or conda environment before starting:

For venv:
```bash
python3 -m venv venv
source venv/bin/activate
```

Or for conda:
```bash
conda create -n nanogpt
conda activate nanogpt
```

### Step 2 Install Dependencies

If you are compatible with cu11.8, then use the following:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install numpy transformers datasets tiktoken wandb tqdm tensorboard rich torchinfo
```

If unsure, visit the pytorch page and subtitute the appropriate line for the `torch` installation line above: https://pytorch.org/get-started/locally/

## Testing Your Setup

### Prepare Training and Validation Data Sets

This downloads and parses a literature dataset into `train.bin` and `val.bin` files.

```bash
bash data/shakespeare_char/get_dataset.sh
```
### Train Model From Scratch

Training with a GPU is highly recommended, to do this now run (should take
around 3-20 minutes depending on one's GPU):

```bash
python3 train.py --compile
```

Highly recommend setting `--max_sample_tokens` which generates and shows outputs
at each new saved checkpoint.

```bash
python3 train.py --max_sample_tokens 100 --compile
```

### Train Model with MeZO (Forward-Only Updates)

This repo also includes a zeroth-order optimizer script, `train_mezo.py`, that
uses forward passes only to estimate gradients. It supports the standard single
dataset training mode and can start from scratch or an existing checkpoint.

Train from scratch:

```bash
python3 train_mezo.py --dataset shakespeare_char --max_iters 2000 --batch_size 64 --block_size 256
```

Resume from an existing checkpoint:

```bash
python3 train_mezo.py --init_from resume --out_dir out
```

You can adjust the perturbation scale with `--mezo_epsilon` and optionally fix
the perturbation seed per step with `--mezo_seed`.

### Perform Inference From Custom Model

minutes and the best validation loss is 1.4697. Based on the configuration, the
model checkpoints are being written into the `--out_dir` directory, that
defaults to `./out`. So once the training finishes we can sample from the
best model by pointing the sampling script at this directory:

```bash
python3 sample.py
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.
```

This looks pretty good for a model which just learned how to spell from scratch.
Keeping an eye on inference is very important, however, usually one can infer
levels from validation losses.

The next section goes over how to do a massive _exploration_ of different models
and quickly compare their quality using the `validation loss` as a proxy.

## Explorations

The explorations directory is intended to be have a set of fully encapsulated
replicable sweeps.

Using these, one can quickly and visually compare ultimate quality of
checkpoints created from training using `validation loss` as a figure of merit.

### Start Exploration

To run the experiment create or modify an existing json file in the `explorations` folder:

```bash
python3 optimization_and_search/run_experiments.py -c explorations/config.json
```

This will create logs in the following directories:

```
csv_logs/
logs/
```

This also saves timestamped and labelled folders within the `output_dir` (which
defaults to `out/` subdirectories)

### Inspect and Monitor Best Val Losses

Often for large explorations with `run_experiments` one wants to monitor the
the best validation losses so far (a metric for how well the model does on next
token prediction on the current dataset).

The included `inspect_ckpts.py` script reports the best valiation loss and
associated iteration number for all ckpt.pt files recursivel for a specified
parent directory.

Example usage:
```bash
python3 checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss
```

![image](./documentation/images/inspect_ckpts.png)

This can be wrapped with color via the watch command for a realtime dashboard.

For example to look at all checkpoint files in the out directory:
```bash
watch --color 'python3 checkpoint_analysis/inspect_ckpts.py --directory ./out --sort loss'
```

As with remainder of the repo, this script is provided as a base to open up for
additional community contributions.

### Start Tensorboard Logging

If using tensorboard for logging, we have provided a convenience script:

```bash
source ./logging/start_tensorboard.sh
```

### View CSV Logs (Tensorboard-style)

To plot the bulk CSV logs in a live-updating, tensorboard-like view (train/val on
the same chart with filenames as labels), run:

```bash
python3 logging/view_csv_logs.py --csv-dir csv_logs --pattern "**/bulk_*.csv"
```

By default the viewer renders an ASCII chart in the terminal, refreshes every 5
seconds, and can be switched to a tokens X-axis with `--x-axis tokens`. Use
`--layout horizontal` to place the legend beside the graph, or set the chart
height with `--height`. Use `--max-x` to control the X-axis scale (default 3000
iters/tokens). For a matplotlib window, add `--mode matplotlib`.

You can view live validation loss updates on url: [http://localhost:6006](http://localhost:6006)

Note: Only one tensorboard process can grab port 6006 at time, if you want to
run a second tensorboard, use the script and specify a different port e.g. 6007:

```bash
source ./logging/start_tensorboard.sh 6007
```

## Model Stats Table

The training script can output a per‑tensor statistics table to a CSV file with
``--print_model_stats_table``. A helper script then visualizes the table or
compares two runs with colour‑coded deltas.

```bash
python3 train.py --print_model_stats_table run1_stats.csv
python3 train.py --optimizer adamw --print_model_stats_table run2_stats.csv
python3 view_model_stats.py run1_stats.csv run2_stats.csv
```

See [documentation/Model_Stats_Table.md](documentation/Model_Stats_Table.md)
for more details.

## LM-head minimum-angle graph exports

`train.py` can optionally export the same closest-neighbor graph used by the
LM-head angle explorer's minimum-angular-distance view after each validation
loss. The export treats each LM-head row as a token vector, streams
row/column blocks through the selected compute device, excludes each token's
self-distance, and records the closest non-self token by signed `0°–180°`
angular distance. The full `vocab_size × vocab_size` angle matrix is never
materialized.

Enable the export by choosing a labelled output directory and turning on the
per-eval flag:

```bash
python3 train.py \
  --export_min_angle_graph_dir out/min_angle_graphs \
  --export_min_angle_graph_each_eval \
  --export_min_angle_graph_label shakespeare_char_baseline
```

Each validation step writes a labelled CSV and JSON sidecar whose filename
includes the label, iteration, and validation loss:

```text
out/min_angle_graphs/shakespeare_char_baseline_iter_00000250_val_1.234567.csv
out/min_angle_graphs/shakespeare_char_baseline_iter_00000250_val_1.234567.json
```

The CSV is an edge list with one directed nearest-neighbor edge per token:

```text
token_id,nearest_token_id,min_angle_deg,cosine,token_vector_length,nearest_token_vector_length,min_angle_rank
```

Use `--export_min_angle_graph_block_size` to tune temporary matrix multiply
size, and `--export_min_angle_graph_device` to choose `auto`, `cpu`, or a CUDA
device such as `cuda:0`. In `auto` mode, the export uses the LM-head tensor's
current CUDA device when possible, otherwise streams blocks to `cuda:0` if CUDA
is available, and falls back to CPU.

To review a sequence of exports visually, open
`analysis/min_angle_graph_plotly_viewer.html` in a browser and select the CSV
files from one export directory. The page sorts snapshots by the iteration in
their filenames and provides previous/next/play controls plus a slider to step
from the first validation export to the last.

## TODO Section:

TODO: Add links and descriptions to other Readme's and Demos.

## Normalization Options

The training CLI now supports `--use_peri_ln` for experimenting with
*Peri-LN*, a normalization strategy that applies layer normalization
around each sublayer (before and after). This can be combined with the
existing `--use_post_ln` flag for Post-LN training.

## Contributing

This repo is under active development and accepting PR's, please see the

See the [Contributing_Features.md](Contributing_Features.md) for details on how
to add new features and explorations.

## Citation

This work extends Andrej Karpathy's foundational
[nanoGPT](https://github.com/karpathy/nanoGPT), for which prior citation format can be found [here](
https://github.com/karpathy/nanoGPT/issues/471).

We ask that citations cite both projects.

To cite **ReaLLM-Forge**, please use this BibTeX entry:

```bibtex
@software{ReaLLM-Forge,
  author = {{ReaLLMASIC} and {Contributors}},
  title = {{ReaLLM-Forge: A Framework for Hardware-Aware LLM Exploration}},
  url = {https://github.com/ReaLLMASIC/ReaLLM-Forge},
  year = {2025},
}
```
