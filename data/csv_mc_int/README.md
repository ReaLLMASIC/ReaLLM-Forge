# CSV Integer Multicontext

`data/csv_mc_int` converts a generic integer CSV into one regular nanoGPT
multicontext dataset per CSV column. It is intended for integer-valued time
series or tabular streams where every column has a known inclusive range.

Each input value is range checked and encoded as:

```text
token_id = raw_integer_value - int_min
vocab_size = int_max - int_min + 1
```

That gives every column its own vocabulary while regular `--training_mode
multicontext` learns the columns together. Sampled token IDs are decoded back to
raw integer CSV values by `sample.py` using the metadata saved in each column
folder.

## Quick start

```bash
data/csv_mc_int/get_dataset.sh data/csv_mc_int/input.csv \
  --output_root csv_mc_int \
  --range time:0:100 \
  --range temp:0:100 \
  --range pressure:900:1100
```

Outputs are written under `data/<output_root>/<column>/`:

- `train.bin`
- `val.bin`
- `meta.pkl`
- `values.csv` (optional with `--save_values_csv`)

A `manifest.json` is also written to `data/<output_root>/` with the ordered
`multicontext_datasets` list for training and sampling scripts.

## Headerless CSV

Use zero-based generic columns with `--no_header`:

```bash
data/csv_mc_int/get_dataset.sh readings.csv --no_header \
  --output_root csv_mc_int_readings \
  --range 0:0:1023 \
  --range 1:-100:100
```

Headerless output folders are named `col_0`, `col_1`, and so on.

## Default ranges

If every column shares a range, use:

```bash
data/csv_mc_int/get_dataset.sh readings.csv --default_range 0:65535
```

Per-column `--range` values override `--default_range`.
