# Analysis — what every device recorded

The sixth Bio-dash dashboard (port **5005**). It reads the recordings the other dashboards save
in `Documents/Bio-dash/` and talks to no device, so it needs no Bluetooth and runs on any computer
that has the recordings.

```bash
./run_dashboard.sh                     # then open http://localhost:5005
./launch.sh analysis                   # from the release folder, alongside others
```

Or install it as a background service with `./biodash.py` like the others.

## What it shows

- **What was recorded:** one strip per device for the chosen day or range.
- **Cards:** heart, SpO₂ (with links to each night's overnight report), air, water.
- **Day by day:** for ranges.
- **Signals on one clock:** any signals, one chart each on a shared time axis, a readout of all of
  them at the hovered moment, and **Download CSV** of exactly what's shown.

How each kind of signal is put on the clock, and why estimates are drawn dashed, is in the main
README (*Analysis*) and `biodata.py`'s header.

## Files

| File | |
|---|---|
| `biodata.py` | reads and merges the recordings, resampling, summaries, CSV (standard library only; also used by `tools/interpolate.py`) |
| `app.py` | the web server (FastAPI) |
| `sample_data.py` | writes made-up recordings: `python3 sample_data.py /tmp/biodash-demo --days 7`, then run with `BIODASH_DATA_DIR=/tmp/biodash-demo` |
| `test_biodata.py` | checks the resampling and summaries against hand-worked values: `python3 test_biodata.py` |

`BIODASH_DATA_DIR` points it at a different recordings folder (a copy from another computer, say).
