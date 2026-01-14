# MIDI Detailed Notes Dataset Utilities

This folder automates downloading and preparing the
[`nintorac/midi_etl`](https://huggingface.co/datasets/nintorac/midi_etl) Lakh
MIDI detailed note tables.

## Files

- `get_dataset.py` – downloads the parquet shards and produces derived JSON and
  CSV artefacts. Each MIDI note number (0–127) receives its own JSON array to
  simplify note-centric processing. A combined CSV with all columns is emitted
  for quick inspection, alongside a `note_counts.json` summary file.

## Requirements

Install the runtime dependencies before executing the script:

```bash
pip install pyarrow requests tqdm
```

## Usage

```bash
python3 get_dataset.py
```

Optional flags:

- `--output-dir DIR` – place downloads and generated files inside `DIR`
  (defaults to the current directory).
- `--skip-download` – reuse existing parquet files inside the output directory.
- `--batch-size N` – adjust the parquet streaming batch size (default 50,000).

## Ordering guard

While converting to JSON/CSV, the script keeps a rolling window keyed on
`floor(start_time)`. Whenever the integer portion of the time changes the
buffered rows are sorted by the precise `start_time` prior to emission. This
ensures the output remains chronologically ordered even if the source data
momentarily regresses (e.g. `285` → `285.25` → `285`).
