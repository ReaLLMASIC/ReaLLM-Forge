# Analysis — what every device recorded

The Analysis page of the app (`#/analysis`). It reads the recordings the other dashboards save
in `Documents/Bio-dash/` and talks to no device, so it needs no Bluetooth and runs on any computer
that has the recordings.

```bash
../launch.sh                           # then open http://localhost:5000/#/analysis
./launch.sh analysis                   # from the release folder, alongside others
```

Or install it as a background service with `./biodash.py` like the others.

## What it shows

- **What was recorded:** one strip per device for the chosen day or range.
- **Cards:** heart, SpO₂ (with links to each night's overnight report), air, water.
- **Day by day:** for ranges.
- **Signals on one clock:** any signals, one chart each on a shared time axis, a readout of all of
  them at the hovered moment, and **Download CSV** of exactly what's shown. **For ReaLLM-Forge** downloads the same
  signals as an integer CSV for ReaLLM-Forge's `data/csv_mc_int` (see the main README).

How each kind of signal is put on the clock, and why estimates are drawn dashed, is in the main
README (*Analysis*) and `biodata.py`'s header.

## Files

| File | |
|---|---|
| `biodata.py` | reads and merges the recordings, resampling, summaries, CSV (standard library only; also used by `tools/interpolate.py`) |
| `mcint.py` | the integer CSV for ReaLLM-Forge `csv_mc_int`: fixed scales and ranges per signal (also used by `tools/export_mcint.py`) |
| `api.py` | its web API (FastAPI), mounted by `../server.py` at `/api/d/analysis` |
| `sample_data.py` | writes made-up recordings: `python3 sample_data.py /tmp/biodash-demo --days 7`, then run with `BIODASH_DATA_DIR=/tmp/biodash-demo` |
| `test_biodata.py` | checks the resampling and summaries against hand-worked values: `python3 test_biodata.py` |
| `test_mcint.py` | checks the integer export: `python3 test_mcint.py` (`FORGE=~/ReaLLM-Forge` also runs ReaLLM-Forge's own prepare script on it) |

`BIODASH_DATA_DIR` points it at a different recordings folder (a copy from another computer, say).
