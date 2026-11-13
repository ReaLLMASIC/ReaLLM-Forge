# Changelog

## V8.0 patch: a second person's strap shows up; smooth ECG on Compare

- **Fixed:** with two people, the second person's Heart page could stay on "Looking…" for good while the strap was
  advertising (found on an Orin: the worker sat in its first search for two hours). Both workers had started at
  the same instant and asked BlueZ to scan at once; one never got an answer, and nothing timed out. Now:
  - every worker's search goes through `bt_debug.scanning`, whose start and stop give up after 10 s and retry;
    after three hangs in a row the worker exits and the server starts it again with a fresh link to BlueZ;
  - the radio list (`bt_debug.list_adapters`) times out too;
  - the server starts workers 1.5 s apart, and restarts a Heart, Brain or Air worker that says it's searching
    but hasn't finished a sweep for 90 s (`BIODASH_SCAN_STUCK_S`).
- **Radios:** with several people, a connected device keeps the radio it's on (moving it would drop it), a strap
  still searching gets a radio no other strap uses (dongles first), and rings take the radio with the fewest
  straps (`plan_radios`, tested in `test_radios.py`). Settings → Bluetooth radios warns when two straps share one.
- **Finder:** a connected device's list says it's from before connecting (a connected worker doesn't search, so
  the list can be old), and "Not showing up?" mentions radios when several people are set up.
- **Compare: the ECG strips wiggled.** Two causes: with more samples than pixels, the chart averaged them into
  4 px columns measured from the moving right edge, so each frame grouped different samples and the R peaks
  shrank and grew; and the right edge was "now", so the gap after the last burst from the strap (one every
  ~0.5 s) grew and snapped shut. The strips now draw every sample (`raw`) and run 0.9 s behind the strap's own
  clock (`x.follow`), as the Heart page's monitor does. Measured in the demo: the right edge moved 63 px and the
  peaks 15 px between frames before; 1 px after.
- **Charts:** where averaging is still used (motion, the forehead pulse), its columns are fixed in time, so a
  scrolling chart no longer shimmers.
- **Fixed:** the Heart and Air workers no longer spin flat out when a search fails (no Bluetooth): they wait 2 s.
- **Tests:** `test_scan_hang.py` (a Bluetooth stack that never answers), `test_radios.py`, and `test_server.py`
  now has the second person's worker get stuck once and checks the server starts it again (70 checks).


- **Fixed:** switching on the beat-to-beat (R-R) forecast made all three Forecast buttons disappear. The page
  read the R-R error as `mae` / `hold`; forecast_service.py sends `model_ms` / `hold_ms`. The demo's made-up
  forecasts had the same wrong names, so the tests didn't see it; they now use the service's.
- **Fixed:** forecasts arriving slower than every 3 s (a CPU, or the Orin with all three overlays on) were
  taken for a stopped service and kept flashing "Starting…". The page now learns how often each one comes.
- A forecast the page can't read shows "Forecast on" and is logged to the browser console, instead of
  taking the buttons away. Checked with the real models and service against the demo strap.


**One app.** Everything Bio-dash does, in one server on one port with one service, built on the
first 6.0 one-app design and brought up to date with all that came after it. Recordings are
unchanged.

- **One server** (`server.py`, port 5000) runs the Bluetooth worker of each device that's on,
  serves each device's API at `/api/d/<device>` and sends live data to the pages over one
  WebSocket. The separate web servers (`app.py` on 5000 – 5006), their templates, their copies of
  the shared scripts and Omni's own engine are gone. Every device folder is now the same pair:
  a worker and `api.py`.
- **People** (Settings → People, up to 3). The first person's strap and ring are `heart` and
  `oxygen`, as with one person. Each extra person gets one more of the same worker and API
  (`heart-p2`, `oxygen-p2` …) with their state in `~/.config/bio-dash/people/<id>/`. A switcher on
  Heart and Oxygen picks whose; Home shows a card each.
  - **Compare** page: a row per person (heart rate, HRV, SpO₂, ring pulse, connection), shared
    charts over 5 min – 1 h, and everyone's ECG under each other. A colour and a line style each.
  - A strap or ring belongs to one person: the finder says who has it, and *Move here* takes it over.
  - Radios are shared out when there are two or more people (straps take the USB dongles first);
    Settings → Bluetooth radios warns when there are more straps than radios. A pinned radio wins.
  - Each page and API follows its own worker's recording instead of "the newest file".
  - **Analysis** no longer merges two people's straps or rings into one line: when a range holds
    more than one, you pick which (the first person's by default).
- **Brain** (Muse S Athena) is a page of the app: EEG lanes, bands (now and over 2 minutes),
  fNIRS with *Zero*, pulse, motion, head position, the preset, and a Home card.
- **Forecasts:** a *Forecast* button by the ECG, the beat-to-beat chart (the R-R model's next
  beats) and the motion chart (the ECG + accelerometer model), each with its live error. The
  model lab is served at `/lab`. Settings → Forecasts shows what the service is doing.
- **Settings**, in sections with a jump bar: Devices, People, Look (themes, *Keep this screen
  on*), Forecasts, Bluetooth radios, Updates, This computer, About.
- **Heart:** *Disconnect* in the Device sheet (Brain too). The Polar worker now has both V7's
  disconnect / switch commands and the stream-refusal handling ("the strap refused the ECG
  stream", with the reason), which had lived in different versions.
- **Water:** the "now tap *It's full*" calibration banner shows straight away.
- **Demo** (`./launch.sh --demo`): a made-up headband, made-up forecasts, and distinct readings
  per person; it keeps its state in the scratch folder, never in the device folders.
- **Fixed:** a page that finished loading after you had left it could throw a script error.
- **Tests:** `test_server.py` (69 checks: API, people, moving a strap, Analysis per unit,
  forecasts, recordings) and the new `test_web.py` (every page in a browser, adding a person,
  phone width). The worker, protocol and exporter tests are unchanged and pass.
- **Upgrading:** `./biodash.py install` removes the old per-dashboard services (including
  `biodash-muse` and `biodash-omni`) and turns on the devices they ran. Paired devices, radio pins
  and settings carry over; Omni's people (V7.0) are added again in Settings → People.
- **Not yet tried on hardware:** several people at once in this form (the radio sharing follows
  V7's rule), and the merged Polar worker.

## V7.0

**Omni compares people:** up to 3 at once, each with a Polar H10 and an O2 ring, on one page.

- **People.** Your V6 H10 and O2 ring become **Person 1** on the first start (from `last_device_*.json` and the
  Polar's radio pin), so one person carries on as before. **Add person** (up to 3), rename in place, reorder,
  **Remove** (two clicks; disconnects their devices). Saved in `Omni-dashboard/people.json`, which the updater
  carries to new versions like the other settings.
- **Devices nearby:** the H10s and O2 rings Omni hears, unassigned first (assigned ones folded away), each with
  **Give to <person>** / **New person** / **Move to**. A device belongs to one person; handing it over
  disconnects it from the first and connects it for the second (quick retries while it lets go). Each person's
  devices have **Disconnect** / **Connect** (auto-reconnect off / on) and **Forget**, and reconnect by themselves
  when back in range.
- **Compare:** a table of everyone's latest H10 heart rate, ring heart rate, SpO₂, R-R, RMSSD and SDNN, and shared
  5-minute charts of heart rate (H10), SpO₂ and HRV (RMSSD), one line per person: a colour **and** a line style
  each (solid, dashed, dotted), pre-filled from their recordings.
- **A card per person:** the six tiles, a live ECG sweep, R-R per beat and motion.
- **Radios:** each H10 gets a radio of its own (external first, in people order; more H10s than radios share, and
  the Radios panel warns); O2 rings take a radio no H10 uses, else their person's; scanning avoids H10 radios, and
  on a streaming H10's radio sweeps only while something is missing or after **Find devices** (at most every
  15 s). A radio pin per person in the Radios panel (**D**), saved by adapter address.
- **Recordings** unchanged: per device under `Omni/<device>/<date>/`, now with `<time>_person.json` saying whose
  session it was. The night report and the analysis dashboard read them as before.
- **API:** `/api/people` (GET, POST add), `/api/people/<id>` (POST name / colour / radio / move, DELETE),
  `/api/people/<id>/device` (assign, connect, disconnect, forget), `/api/scan`; `/api/stream` sends
  `{"people": {id: {polar, viatom}}}`; `/api/history` and `/api/radios` take a person. V6's `/api/connect` and
  `/api/scanners` still work (first person).
- **Test without hardware:** `OMNI_FAKE=1 python3 omni.py` (three stand-in H10s, O2 rings and radios,
  `fake_omni.py`); `test_omni.py` runs three people end to end, page included (37 checks).
- Omni's classic ECG view and tap-a-tile trends are gone (the per-person sweep and the shared charts replace them).

## V6.3

V6.2 plus a third live forecast: **R-R**, from the ECG + R-R model (the R-R plan v2 proof of concept,
2026-11-05).

- **🔮 Forecast on the Heart Rate Variability (R-R) card.** The next beats the model forecasts in the coming
  second, on the R-R chart: a ring at each beat's time and R-R (the median of the samples), a bar from the
  lowest to the highest R-R the samples gave, and small dots for the last beats as the model sees them. The status
  shows the next R-R and its heart rate, and the running error: each beat's R-R against the last forecast made
  before the strap reported it, over the last 60 beats, next to repeating the last R-R.
- **The model is included:** `polar_dashboard/forecast/models/ecg_rr` (`hf_g125_ecg_rr_d3_6L_384D_40k`, step
  26,250; 13.5 M parameters, 52 MB), the same weights as the Hugging Face folder `ecg_rr/`.
- **`polar_dashboard/rr_lane.py` builds the R-R lane live** from the session's `_rr.csv`, the way the training
  export made it (beats chained from the strap's R-R messages, anchored to their arrival; a straight line
  between beats), with the newest beat's R-R held after it ("hold": a live forecast can't know the next beat).
  It matches the PoC's export code beat for beat and row for row; `test_rr_lane.py` checks it.
- **`forecast_service.py`** tails the `_rr.csv` too, has an `rr` overlay (`--rr-model`, setting `rr`) and adds
  `beats`, `known` and `rr_score` to its R-R forecasts; `--bench` times all three models. `app.py` relays
  `rr_forecast`. `tools/orin_setup.sh --models` installs and times `ecg_rr` as well.
- **Expect it to be about as good as holding the last R-R.** That's what the proof of concept measured on
  held-out data (9.31 vs 9.48 ms), and its rebuilt beats sit a median 456 ms after the ECG's R peaks. The overlay
  is for watching the model live and for dropping plan v2's models into the same folder.

## V6.2

V6.0 with the forecasts working on a Jetson Orin out of the box.

- **`tools/orin_setup.sh` installs transformers 4.x and a current Pillow.** It used to take the newest
  transformers (5.x, which also needs PyTorch 2.6+), and with the system's Jetson PyTorch the environment also
  sees the system's packages, whose Pillow (9.0, JetPack 6 / Ubuntu 22.04) lacks `Image.Resampling`: every model
  load failed with "Could not import module 'PreTrainedModel'". Now: `transformers>=4.40,<5` (as the models were
  tested) and `pillow>=9.1` in the forecast environment itself, then a check that transformers imports, which
  stops the script with the real error if it doesn't. Running it again fixes an environment made before.
- **The forecast's "couldn't load" message says how to see the real error** (transformers hides it) and how to
  fix the usual cause.
- **Hydro-dash sees bottles again with a newer bleak.** bleak 1.x and later refuse a scan callback that
  doesn't take exactly two parameters (device, advertisement), and Hydro's scanner had a third, optional one:
  every scan failed with "callback must be callable with 2 parameters", so no bottle was ever listed (the other
  dashboards were fine). Both of Hydro's scans (one radio, and "Any" across several) now pass exactly two.
- **`tools/ble_scan.py --adapter` works with a newer bleak** (`bluez={"adapter": ...}` instead of the removed
  `adapter=`).

## V6.0

V5.8 plus a new device and dashboard: **Muse-dash**, for the **Muse S Athena** headband (port 5006).
The first time Bio-dash reads the brain: EEG, brain-wave band powers and blood oxygenation, beside the heart
(Polar), blood oxygen (O2 ring), air (Atmos) and water (HidrateSpark).

- **Muse-dash** (`muse_dashboard/`, `./run_dashboard.sh`, `./launch.sh muse`, `./biodash.py install muse`):
  - **Live EEG** from the four head sensors (TP9, AF7, AF8, TP10; **+ AUX** for the other 4), 256 Hz, with
    display filters (1-40 Hz, 60 / 50 Hz notch or none, or raw) and a scale from ±25 to ±500 µV.
  - **Brain waves:** delta, theta, alpha, beta and gamma as a share of the power every second (last 2 s,
    Hann window), averaged over the sensors in contact, now and over the last 2 minutes; an **Alpha share**
    tile with its trend.
  - **Sensor contact** per sensor (good / fair / bad / off).
  - **fNIRS:** oxy- and deoxy-haemoglobin change on each side of the forehead from the 730 and 850 nm light
    (modified Beer-Lambert) against a 10 s baseline, with **⟲ Zero** for a new one.
  - **Heart rate and HRV from the forehead PPG** (the clearest optical channel, picked every second), with
    the pulse waveform.
  - **Head motion and position** (tilt, turning, still / moving) and **battery**.
  - **Presets** in the debug panel (default `p1041`: EEG 8 ch, optics 16 ch, motion, battery); a new preset
    starts a new session. The debug panel also has the rates against 256 / 64 / 52 Hz, dropped samples,
    clock restarts, firmware, the pulse channel and the radio choice.
  - **Recordings** in `Muse/<headband>/<date>/`: `eeg`, `optics`, `imu`, `bands`, `heart`, `fnirs`,
    `battery`, timed by the headband's own clock mapped onto the computer's (as the H10's).
  - **In the V6 design** (from the single-app redesign: its colours, cards, tiles, status pill and charts, in
    every theme), as the **Brain** page; auto-reconnect, device names, tile trends, themes, the sidebar and
    updates as in the other dashboards.
  - The protocol follows [OpenMuse](https://github.com/DominiqueMakowski/OpenMuse) (MIT); see
    `muse_dashboard/PROTOCOL.md`. **Not yet tried on a real headband**: the decoder matches OpenMuse's on its
    test packets, everything else has run against the stand-in. `muse_dashboard/README.md` has a checklist
    for the first session.
  - **`MUSE_FAKE=1`**: a stand-in headband sending packets in the real format, to try everything without one.
  - Tests: `test_muse_protocol.py` (10, the decoder against OpenMuse's), `test_muse_signal.py` (5),
    `test_muse_dashboard.py` (28 checks: worker, files, app, page and presets with the stand-in).
- **Every dashboard in the V6 design** (a V6.0 patch): the redesign's colours, page heads (Heart, Oxygen, Air,
  Water, Omni, Analysis, Brain, each with its device's icon and colour), cards, tiles, buttons, status pills and
  chart colours, in every theme from the sidebar. One shared `static/css/v6.css` and `static/js/v6.js` in every
  dashboard (Analysis: its own stylesheet, on the same tokens); the pages work as before. The sidebar lists the
  dashboards by those names.
- **The forecast models are in the release** (a V6.0 patch): `polar_dashboard/forecast/models/{ecg,ecg_acc}/
  model.safetensors` (49 + 69 MB). A new release folder no longer starts without them; PyTorch is still set up
  on its own (`tools/orin_setup.sh`, which reuses a working system-wide Jetson PyTorch).
- The sidebar lists **Muse** (5006) in every dashboard; `launch.sh`, `biodash.py`, `setup_env.sh`,
  `fetch_vendor.sh` and `requirements.txt` know it.
- The single-app redesign that was being built as "6.0" is kept on its own branch (`v6-unified`), for later.

## V5.8

V5.6 plus a second page for the forecast models: the Stream Lane Lab viewer from ReaLLM-Forge PR #924,
on the live strap.

- **🧪 Lab** (`/lab`, the button next to Forecast): ReaLLM-Forge PR #924's Stream Lane Lab viewer, served
  by Polar-dash, with the live strap as a source beside PR #924's sine demo. Pick a model (`ecg`, or `acc`
  with its 4 columns), the horizon (up to 128 rows, 1 s), the decoding (mean, greedy, sample with
  temperature and top-k), press Start: it forecasts from the newest rows as soon as the last forecast is
  back, and the real signal then arrives over it, with the one-step and rollout error, latency and frames
  per second. It runs on the same forecast service as the Heart page (one at a time on the GPU), through
  `GET /api/lab/rows` (the strap's newest rows on the 8 ms grid, in mV / mg) and PR #924's `GET /api/signal`.
  Pausing, or leaving the tab, stops its forecasts. `test_lab.py` (19 checks, no PyTorch needed).
- `tools/forecast_timing.py`: finds the R peaks in a recording and in its forecast log (`Forecasts/`) and
  reports the forecast's timing error by how far ahead it was made (0-100 ms, 100-200 ms, ...), for the
  line drawn and the median. The same offset even at 0-100 ms ahead means an alignment problem; an error
  that grows with the lead is the model's beat timing. Standard library only; `test_forecast_timing.py`.
- **The line drawn follows your rhythm.** The model tends to put the next R peak early (about 100 ms at
  0.2-0.5 s ahead, measured with `tools/forecast_timing.py` on Oct 22; at 0-100 ms ahead it's within 1 ms,
  so nothing is misaligned). Of the 8 samples it draws, the dashboard now shows the one whose next R peak
  lands where the last 2 s of beats put it (the median of the last beat-to-beat intervals), instead of the
  one closest to the median. The median and band are unchanged, and so is the error readout (it uses the
  median). Falls back to the old choice when the last 2 s hold fewer than two clear beats.
  `BIODASH_FORECAST_REP=median` turns it off. `test_rhythm_pick.py` (9 checks).
- `tools/portable_polar_tests/`: the tests for **Portable Polar**, the Polar H10 in the browser (its own
  repo: Web Bluetooth, recordings in Bio-dash's format on the runner's disk, the ECG forecast on their GPU).
  They check it against the code here: parsing against polar-python, timestamps against `stream_clock.py`,
  the 8 ms rows against `waves.py`, the line drawn against `forecast_service.py`, the model in the browser
  against onnxruntime, and the page with a stand-in strap.

## V5.6

V5.5 plus a better look at the live ECG forecast, and changing straps without restarting:

- **The pen traces over the forecast.** What the model forecast stays on the monitor under the live ECG
  after the pen passes (it used to vanish), so each beat shows the prediction and the real thing on top
  of each other. Ahead of the pen it's still the dashed line and band.
- **🔮 Inference strip** under the monitor while the ECG forecast is on: what the model forecast for each
  moment, alone, on the same time axis as the live ECG above it, with the pen's position marked.
- **The inference is logged:** `<Documents>/Bio-dash/Forecasts/<day>/<session time>_forecast_ecg.csv` (and
  `_forecast_acc.csv`), one row per 8 ms: `t_ms` (the strap's clock, as in the session's own files),
  `lead_ms` (how far ahead it was forecast), the line drawn and the median and band of the samples, in
  mV / mg. It's what the strip shows. `Forecasts/` sits outside the recordings layout, so Analysis, the
  exporters and `fleet.py` leave it alone. `BIODASH_FORECAST_LOG=0` turns it off.
- **⏏ Disconnect** (header, while connected; tap twice): ends the session with the strap, and auto-reconnect
  waits until you pick a device in the scanner: the same strap, or another one. Picking another device
  while connected switches to it.
- **The model weights carry over** to a new version: the forecast uses the same model's weights from an
  older release folder when its own has none (so upgrading from V5.5 keeps the forecast working), and from
  V5.6 on the updater links them into the new folder.
- **The H10 models on stream-lane-lab's API** (ReaLLM-Forge PR #924): `GET /api/models` lists them with
  their columns (name, unit, range, levels), context and row interval (8 ms); `POST /api/forecast` takes
  `model_id`, `history` (rows in mV / mg on the 8 ms grid, oldest first), `horizon` (1-128 rows), `mode`
  (`mean`, `greedy`, `sample`), `temperature`, `top_k`, `seed`, and returns `predictions` with
  `latency_ms`, `history_frames_used`, `out_of_range_counts` as PR #924's server does. The forecast service
  answers it (on 127.0.0.1:5011, `BIODASH_FORECAST_API_PORT`); the live overlay's own route is now
  `/api/forecast/live`. `test_forecast_api.py` (13 checks, end to end).
- **Share link:** `tailscale serve --bg --https=443 localhost:5001` gives the team
  `https://<this computer>.<tailnet>.ts.net`. Requests through it count as other devices, so "this
  computer only" settings still hold (with `--allow-power` they are allowed anyway: install without it
  before sharing).
- `polar_dashboard/test_worker_commands.py` (8 checks, a stand-in strap) and `test_forecast_log.py` (11).

## V5.5

V5.3 plus live forecasts of the ECG and the accelerometer in Polar-dash, from the H10 models
trained in ReaLLM-Forge:

- **🔮 Forecast pills** on the ECG card and the accelerometer card show the next second ahead of the
  live trace: a shaded band (the range of 8 sampled futures) and a dashed line (one representative
  sample), with the running error next to the "repeat the last value" baseline. `forecast_service.py`
  runs as its own process (torch never loads into the web server); `run_dashboard.sh` starts it and
  restarts it if it stops. Details in `polar_dashboard/forecast/README.md`. A research demo, not a
  medical device. The model weights stay out of git: copy `forecast/models` from the shared zip.
- **`analysis_dashboard/waves.py`:** the ECG and motion on one even 8 ms (125 Hz) grid, the rows the
  models were trained on; `LiveAligner` builds the same rows live.
- **`tools/export_waves.py`:** those rows as a csv_mc_int dataset (`--with-acc`, `--from`), with
  `columns.json` and a `_pieces.csv` of the joins.
- **`polar_dashboard/replay_session.py`:** plays a recorded session back through the dashboard to
  test the forecasts. It refuses to write into your real recordings unless told to
  (`BIODASH_DATA_DIR` or `--into-recordings`), marks what it writes with `<time>_replay_info.json`,
  and the exporters, the analysis page and `fleet.py` (push and pull) all leave replays out.
- **`tools/fleet.py`** runs on Python 3.10 (the Orin) and never pulls a replayed session from
  another machine.
- **`tools/orin_setup.sh`: the forecasts on a Jetson Orin.** Installs PyTorch built for the Jetson
  (NVIDIA's Jetson AI Lab index for the JetPack it finds; never PyPI's, which can't use the GPU) and
  transformers into one environment every release shares (`~/.local/share/bio-dash/forecast-venv`,
  which the dashboards then use by themselves), copies the model weights in (`--models <zip>`), starts
  the forecasts at 4 samples, and times both models at 4 and 8 (`forecast_service.py --bench --samples`).
- Carries every V5.3 change, including the Polar H10 timestamp fix (`stream_clock.py`).

## V5.3

V5.2 plus an export made for ReaLLM-Forge, and one button to restart everything:

- **For ReaLLM-Forge:** the analysis page's new button (and `tools/export_mcint.py`) writes the
  signals on one clock as an integer CSV that ReaLLM-Forge's `data/csv_mc_int` takes as it is: a
  whole number in every cell, decimals kept by scaling (`temp_c_x10`), the time as `minute_of_day`
  and `second`, a fixed range per signal so different days share vocabularies, and `-1` where a
  device wasn't recording (csv_mc_int allows no blanks). Short gaps are held at the last reading
  first; `--missing drop` leaves incomplete rows out instead.
- **Restart everything** (sidebar → Restart, or `./restart_all.py`): stops every Bio-dash dashboard
  and Bluetooth worker on the computer, from any release folder, as a service or started by hand
  (an old release still running in another folder is caught too); resets Bluetooth; then starts
  again whatever was running. Workers get Ctrl+C first so recordings close properly, then SIGTERM,
  then SIGKILL. Bluetooth: the bluetooth service is restarted if that's allowed without a password
  (a sudoers line is offered), otherwise every connected device is disconnected and each radio
  switched off and on. "Reset Bluetooth too" can be unticked. Hand-started dashboards come back in
  the background with their output in `~/.cache/bio-dash/logs/`; Omni needs sudo so it's left for
  you. Same access rule as Reboot: this computer only, unless `BIODASH_ALLOW_POWER=1`.
- **Hydro: a banner after "It's empty now"** until you press "It's full now". The bottle takes empty
  and full as one calibration and counts no sips until it has both, which looked like the dashboard
  had stopped updating. The banner has its own "It's full now" button, survives reloads and
  restarts, and clears itself if a live sip arrives (the bottle is counting again).
- **Hydro isn't pinned to a radio any more:** it listens on every radio and connects through the one
  that hears the bottle best (the last radio that worked gets a small head start). A radio whose
  connection fails is passed over for 10 minutes, or for a week (remembered across restarts) when
  the bottle's service list came back empty through it, the Bluetooth 4.0 problem. The pin could lock
  onto the wrong radio now and then. Pinning is still in the debug panel; old pins don't carry over.
- **`tools/fleet.py`: recordings from your other machines.** `check` lists what each online
  Tailscale machine (tagged `tag:ssh`) has recorded; `pull` copies what's new or has grown, into one
  folder per machine or (`--merge`) into this computer's recordings without overwriting anything.
  Plain ssh, nothing installed on the other side. `push --to gdrive:Bio-dash` uploads the raw
  recordings (this computer's and every pulled machine's) to any cloud rclone knows: new and grown
  files only, nothing online deleted, the destination remembered (`push --pull-first` does both).
  `schedule` uploads by itself (systemd user timers): every 5 min without the ECG / motion
  waveforms, everything hourly, one upload at a time. `tools/test_fleet.py`: 33 checks with
  stand-in machines and a real rclone.
- **`tools/export_ecg.py`: just the ECG.** The raw samples with their timestamps, and an even
  200 Hz (no timestamps, interpolated linear or cubic within each unbroken stretch, a stretches file
  saying where the gaps were); `--uv` for whole microvolts. `--with-acc` adds the strap's
  motion (X/Y/Z mg) on the same 200 Hz rows, plus its raw file. `tools/test_export_ecg.py`: 27 checks.
- **Name your devices** ("Mine", "Boss's"): a tag next to every device in the scanner, the
  auto-reconnect bar and beside the status, click it to name the device. Names are kept per device
  (by the name it advertises, or its address) in `~/.config/bio-dash/device_names.json`, shared by
  every dashboard and carried across upgrades. Only what you see changes; recordings and their
  folders keep the device's own name.
- **Polar H10 timestamps fixed** (Polar-dash and Omni; `stream_clock.py`). The strap's accelerometer
  set to 200 Hz really runs at about 205.6 Hz; samples were stamped 5 ms apart, so each frame's
  first samples were up to 5 ms early and overlapped the previous frame. The rate is now measured
  from the strap's own frame times. ECG and accelerometer now share one clock offset, so the same
  instant has the same time in both files. And the jumps of 1-1.3 s (often backwards) that split
  recordings came from re-anchoring the clock whenever it drifted 1 s from the computer's: drift is
  now corrected 1 ms at a time and stamps never go backwards. The debug panel shows the measured
  accelerometer rate and any clock restarts. Older recordings are unchanged.
  `polar_dashboard/test_stream_clock.py`: 13 checks with a simulated strap.
  `tools/apply_clock_fix.py <release folder>` puts the same fix into another V5.x copy (e.g. V5.5).
- **`tools/radio_compare.py`** (from V6): which of two Bluetooth radios hears your devices better,
  side by side or one at a time with a `--report`.
- Each export comes with `<name>.columns.json` (scales, ranges, "no reading" values) and
  `<name>.prepare.sh`, which runs `get_dataset.sh` with every `--range` filled in.
  `tools/export_mcint.py --forge ~/ReaLLM-Forge` builds the datasets in one go.
- `tools/mcint_decode.py` turns what `sample.py` generates back into real units.
- `analysis_dashboard/test_mcint.py`: 29 checks against hand-worked values; with `FORGE=` set it also
  runs ReaLLM-Forge's own prepare script on the result.

## V5.2

Themes for every dashboard, and everything patched into V5.0 since its release (the Bluetooth scanner, water calculations with the
local weather, the Bluetooth 5 dongle confirmation, power-cut protection; see V5.0 below), gathered
into one release, plus what a night of testing radios on the Jetson Orin taught us:

- **Every Bluetooth adapter tried, and why it did or didn't work:** a table in
  `docs/ARCHITECTURE.md` section 9 (TP-Link UB500 Plus, ASUS USB-BT400, the Orin's built-in radio,
  a laptop's built-in radio), with two new diagrams: which radio serves which device on the Orin,
  and what happens at power-on.
- **UB500 Plus proven with the bottle:** 10 of 10 cold connects with every HidrateSpark
  characteristic present, heard 8 times in 25 s at −60 dBm, and picked by Hydro-dash by itself
  after a reboot. The radio diagram (section 7) carries the numbers; how to run the same checks on
  any adapter is in `hydro_dashboard/TROUBLESHOOTING.md` section 13.
- **Found: the Orin doesn't power on with a UB500 Plus and an ASUS BT400 both attached** (no logo,
  no screen, before Linux). Not power (two ASUS dongles boot fine, all four add about 1 W) and not
  fixable from the dongles' side; plug the ASUS dongles in after boot. TROUBLESHOOTING 14.
- **Broadcom BCM20702 firmware patch:** "BCM: firmware Patch file not found" explained, with the
  file, its checksum and how to install it (ASUS BT400 goes from build 0000 to 1467).
  TROUBLESHOOTING 15.
- **`hydro_test.py` says when the name is almost right:** a run that doesn't find the bottle lists
  the bottle-like names it did see ("Seen instead: h2o00001234 <- did you mean this one?"). A
  one-character typo in the name had made a working radio look dead.
- **Powering the Orin from USB-C:** why a 12 V trigger board works with some chargers only (12 V is
  optional in USB Power Delivery; 20 V is required above 45 W). ARCHITECTURE section 9.
- Radio pins and the last device carry across upgrades (the updater copies them); the docs now say
  so, instead of warning that they reset.
- **Themes for every dashboard:** Auto (follows the device's light / dark setting), Dark, Light,
  OLED black, High contrast and Warm night, picked under **Theme** in the sidebar. One choice
  applies to all six dashboards (a cookie shared by every port; open dashboards follow within
  seconds). Chart axes and grids and the ECG monitor are recoloured to match; data colours keep
  their meaning. `static/js/themes.js` + `static/css/themes.css`, the same in every dashboard.

## V5.0

A new dashboard that looks back instead of live: **Analysis** (port 5005). It reads the recordings
the other dashboards save and talks to no device. (No V4: four is an unlucky number.)

- **Summaries for a day or any range:** a strip per device showing when it was recording; cards for
  heart (average, range, resting estimate, HRV as median RMSSD), SpO₂ (average, lowest, time below
  95 / 90 / 88 %), air (CO₂ with time above 1000 ppm, and every other metric the Atmos reports) and
  water (drunk against the Hydro-dash goal, sips, refills, largest sip); for a range, a row per day.
- **Overnight reports from the dashboard:** each night with SpO₂ readings gets a link that builds
  and opens `tools/night_report.py`'s report.
- **Signals on one clock:** pick any signals; each gets its own chart on a shared time axis
  (never two scales on one chart), a crosshair across all of them, and a readout of every signal
  at the hovered moment. Step: auto, 1 s ... 1 h.
- **Better resampling, shared by the dashboard and `tools/interpolate.py`** (`analysis_dashboard/biodata.py`):
  readings get the mean of each step with its min / max (nothing between two ticks is dropped any
  more), state such as the bottle's fill level is carried forward, sips become mL per step plus a
  running daily total. Estimates are drawn dashed, and every exported value has a `.n` column with
  the number of real readings behind it (0 = estimate).
- **Signals merged across dashboards:** the same Polar strap recorded by Polar-dash in the day and
  by Omni at night is one heart-rate signal; heart rate is derived from the R-R intervals where a
  recording has no heart-rate column (Omni); sips replayed by the bottle are counted once.
- **CSV export** of exactly what's on screen, from the page.
- `tools/interpolate.py` now uses the same code: named signals (`heart.hr`, `oxygen.spo2`,
  `air.co2@scd30`, `water.fill` ...), `.n` columns, `--minmax`, `--curated`. Output file names start
  with `bio-dash_` instead of `interpolated_`.
- `analysis_dashboard/sample_data.py` writes made-up recordings to try it with; `test_biodata.py`
  checks the resampling against values worked out by hand.
- Sidebar, services menu (`./biodash.py`), `./launch.sh analysis` and the root requirements know
  the sixth dashboard.
- **Power-cut protection for recordings** (all dashboards): recordings are pushed onto the disk
  every 10 s (`BIODASH_FSYNC_S`), so a cut costs at most that much; on start, each dashboard removes
  the zero (NUL) bytes a cut can leave at the end of a recent recording; the analysis dashboard and
  `tools/night_report.py` read past them. (Found on a real night's SpO₂ recording, where they
  stopped the analysis page from loading.)
- The analysis page shows the actual error when something goes wrong, instead of "request failed (500)".
- **`tools/ble_scan.py`:** standalone Bluetooth LE scanner with plain-language advertisements (Apple Find My, Continuity, Fast Pair, iBeacon, Eddystone, heart-rate straps, makers by ID), address types, `--find-my`, `--services` to list a device's GATT services, and `--log` to CSV.
- **Patch notes:** each patch carries a `PATCH_NOTES.md`; `push_v5.sh` puts it into the commit message, and the sidebar's Updates section (and `updater.py check`) lists the notes under each patch before you apply it.
- **Hydro-dash with a Bluetooth 5 dongle, confirmed:** a TP-Link UB500 Plus on a Jetson Orin Nano (JetPack 6, kernel 5.15) works out of the box and streams the bottle with no `ExchangeMTU` cap. The radio diagram, `hydro_dashboard/TROUBLESHOOTING.md` (12) and the README say so.
- **Debug panels: the real radio maker and USB ID.** Every radio showed "Linux Foundation · 1d6b:0246": that is BlueZ's own device ID, which D-Bus reports for any adapter. The ID now comes from the USB device itself (sysfs).
- **Hydro-dash: water calculations.** *How Much Should I Drink?* suggests a daily goal from body weight, a reference intake, exercise and **the local weather** (today's feels-like high from Open-Meteo for your town: +250 mL from 25 °C up to +1 L from 36 °C; cold days unchanged), shows the working, and sets it as your goal in one click. *Update my goal every morning* does that by itself each day from 05:00. The goal maths is in `water.py` (`test_water.py`, `test_goal_settings.py`). The Today tile adds what's left (in mL and bottles) and your pace through the reminder hours: on pace, or how far behind, and about how much per hour until the end of the day. Pace maths in `static/js/water_calc.js` (`test_water_calc.js`).

## V3.1

Hydro-dash tuning, from reading how the official app talks to the bottle (no app code included).
Run on one PRO 2: connecting, sips, fill level, bottle size and glow work; **reminders are still being confirmed.**

- **Fill level without calibrating.** Every sip record carries the bottle's own calibrated "empty" and "full" weights (the two fields V3.0 logged as "constant"). The fill level is computed from those, from the newest record and from the settled live weight. The manual full / empty anchors and the "(estimated)" level are gone.
- **Sip volume from the weights**, as the app computes it, instead of the whole-percent byte (which reads up to 6 mL low on a 21 oz bottle). The percent byte remains the fallback. The sip log gains `Cal_min`, `Cal_max` and `Volume_from` columns.
- **Capacity read from the bottle** on connect.
- **Draining stored sips the app's way** on current (firmware 100+) bottles: acknowledge each record, then ask for the next. V3.0's single fast request is kept for older firmware and as an automatic fallback (`BIODASH_HYDRO_DRAIN=fast` forces it). The debug panel shows the bottle family, drain mode and requests / acks.
- **"It's full now" / "It's empty now"** now tell the bottle to redo its own calibration (two clicks to confirm), instead of storing an anchor in the dashboard.
- **Replays aren't double counted** when a sip first logged from the percent byte comes back with a weight-based volume (de-duplication allows a small volume difference at the same timestamp).
- **Glow and reminders.** *Glow now* button; a reminder schedule (from / to, every N minutes, only when behind on the goal or every time, optional sound), glow on reaching the goal and glow on each sip. The schedule is stored on the bottle, so it runs with the dashboard closed.
- **Custom glow.** Choose a style (pulse, spin, flash, solid, rainbow) and two colours, previewed as the LED ring, and upload it to the bottle, which keeps it for *Glow now* and reminders. Patterns are built by Hydro-dash and sent in the bottle's own format (a protobuf message in 20-byte packets); 10 LEDs on the small puck, 13 on the large.
- **Real sync instead of the canned handshake.** On every connect (and at midnight, and when settings change) the bottle gets the actual time of day, goal, today's total and the reminder schedule. V3.0 replayed 13 fixed writes that set the clock to 15:18 and someone else's schedule.
- **The radio catch, found and documented.** A laptop's built-in Bluetooth often can't hear the bottle; an old Bluetooth 4.0 dongle hears it but "connects" with an empty service list, because the bottle never sends its large replies over a radio limited to 27-byte packets. Capping the message size (`ExchangeMTU = 64` in `/etc/bluetooth/main.conf`) makes it work. Hydro-dash treats an empty service list as a failed connection and retries, finds the bottle by name (its address changes), and leaves the radio choice to the debug panel. New diagram: `docs/ARCHITECTURE.md` section 7; write-up: `hydro_dashboard/TROUBLESHOOTING.md` sections 11 and 12.
- **Bottle size is a list** of the sizes sold (17 / 20 / 21 / 24 / 30 / 32 oz) plus *Automatic*, which follows what the bottle reports.
- **Settings survive updates:** goal, reminders and the bottle's calibration are kept in `~/.config/bio-dash/`, not in the dashboard folder (an existing file is carried over once).
- **"It's full now" / "It's empty now" show the result straight away** instead of waiting for the next sip.
- **Services say so when they can't start.** If a dashboard's process stops on its own (most often: a new release folder with no `.venv` yet, so "No module named …"), the launcher now exits with an error instead of printing "System is LIVE" and ending quietly, so the service shows as failing and is retried. `./biodash.py` checks for the packages before installing services and tells you to run `./setup_env.sh` first.
- **The sidebar lists only the dashboards that are running.** Links to dashboards that aren't running are hidden, and appear within about 10 seconds of one starting.
- **Fill level on a second machine.** If no calibration is saved yet (a new computer with the recordings copied over), the bottle's empty / full weights are taken from the newest sip record in the log, so the fill level shows on connect instead of after the next sip.
- **`tools/interpolate.py`:** one evenly spaced CSV from every device's recordings, by linear interpolation (one row per time step, one column per signal). Cells stay empty outside a signal's readings and across pauses longer than `--max-gap`; waveforms and event logs are left out unless included. Standard library only.
- `hydro_test.py watch`: one mark per second showing when the bottle was heard; `listen` retries the connection while the bottle keeps advertising.
- `hydro_test.py listen` prints the firmware family, capacity, each sip's source and the level left, and reports requests / acks / skipped records; `--drain auto|ack|fast`.
- `PROTOCOL.md`: the full record layout, drain commands by firmware, bottle size and calibration commands.

## V3.0

### Sidebar and updates (all dashboards)
- **Sidebar** replaces the crowded top bar: dashboard links with live dots, the debug panel toggle, keep-screen-on, updates and power, identical in every dashboard (`static/js/sidebar.js`, which replaces `nav.js` and `power.js`). Closed when a page opens (☰ opens it; a drawer on phones); dashboards are listed in port order; the header keeps the title and connection status.
- **Update checker** (`updater.py`, sidebar **Updates**): *Check for updates* looks at GitHub and offers **Apply patch** (new commits for this version: pull, refresh `.venv` if needed, restart the services) and, separately, **Upgrade to Vx.y** (a newer release folder: pull, build its `.venv`, carry settings across, re-point the services, restart; the old folder is kept). Fast-forward only; refuses on local edits or a diverged branch. Same access rule as shutdown / reboot.
- **Shutdown / reboot** moved from the debug panel to the sidebar's **Power** section.
- **`.gitignore`** for `__pycache__`, `.venv` and the files the dashboards write while running, so a clone stays clean and updatable.

### Tools
- **Overnight SpO₂ report** — `tools/night_report.py` turns a night of `_vitals.csv` recordings into a self-contained HTML report (charts, time below thresholds, dips per hour, recording quality).

### Hydro-dash: HidrateSpark smart bottle (new dashboard, port 5004)
- **Status: still being tested** — verified on one PRO 2; fill-level calibration, capacity and long sessions are still being checked.
- **Sips** with their timestamps, live and **replayed from the bottle's buffer** after time out of range (drained with the bottle's `0x57` request, de-duplicated within ±2 s across sessions).
- **Today** vs. a daily goal (progress ring), **fill level** from the on-bottle weight sensor, **last sip**, **sips / refills today**, **battery**.
- **Charts:** intake by hour today, last 7 days vs. goal. **Tile trends** over 1 h / 6 h / 1 day / 7 days.
- **Bottle settings:** capacity, daily goal, mL or fl oz, **fill calibration** (full / empty), auto "full" latch on a detected refill (cap open → weight rise ≥ 25 → cap close).
- **Recordings:** `Documents/Bio-dash/HidrateSpark/<bottle>/<date>/` — `_sips`, `_status`, `_events`, and `_raw` (every notification from every characteristic, hex). Totals and charts are computed from the sip log.
- **Debug panel:** radio and link, bottle protocol (handshake, sip characteristic, live / replayed / duplicate / undecoded counts, drain writes, stuck-record guard), sensors, and **every characteristic with its last raw frame**.
- **Protocol:** our own implementation (`hydro_protocol.py`, with tests) of the community-documented HidrateSpark protocol (13-step handshake, drain, cap / weight signals). Both documented sip-record layouts are decoded automatically. Sources and licensing in `hydro_dashboard/PROTOCOL.md`. **Verified on a simulated bottle only; the PRO 2 is unverified** — the raw capture makes adapting it straightforward.
- **Scanner** also lists devices that advertise **without a name** (some firmware only names itself in a scan response, or not at all): matched by HidrateSpark service UUID, or shown under **Show all nearby devices** with their service and manufacturer IDs, strongest signal first.
- **Connecting:** the bottle is looked up fresh before every connection attempt (it only advertises briefly after being moved, and BlueZ forgets devices it hasn't seen recently — a handle from an earlier scan failed with *device … not found*). The page shows what it's waiting for, and the debug panel shows the last connection error.
- **PRO 2 verified connectable** (firmware 100.64.0): same HidrateSpark service, sip, cap and weight characteristics. The bottle rotates its address on every wake-up and BlueZ drops it the moment scanning stops, so the worker (and `hydro_test.py`) now follow it **by name** and **connect while the scan is still running**.
- **PRO 2 sip records decoded** from a real bottle (new little-endian layout: sip %, today's total, seconds since the sip, weight before/after), with tests from the captured frames. Sips are now drained **strictly one at a time** (overlapping `0x57` requests were skipping records), and the sip log records the weight before/after each sip.
- **`hydro_dashboard/TROUBLESHOOTING.md`:** how the PRO 2 was brought up — every problem, its cause and fix, a checklist and the tools that helped.
- **`hydro_test.py`:** standalone bottle test (like Viatom's `dump_gatt.py` / Omni's `test_scanner.py`) — scan (named or not, strongest first), dump services with readable values and a protocol check, and listen with the handshake + drain, printing every notification decoded and saving a capture.
- Shared features: radio selection, auto-reconnect, keep-screen-on, header links, `biodash.py` service (`hydro`), `./launch.sh hydro`.

### Suite
- **Atmos PM colours:** PM1.0 lime, PM2.5 cyan, PM4.0 fuchsia, PM10 orange (were four near-identical cyans / teals) in the chart, the tile trends and the tiles.
- **`docs/ARCHITECTURE.md`:** flow diagrams (Mermaid, rendered by GitHub) of the whole suite, a device's life cycle, the Hydro-dash bottle protocol, the Atmos parser, and how everything starts.
- **Shutdown / reboot** from every dashboard's debug panel (System card): two taps to confirm; only from the computer itself unless `BIODASH_ALLOW_POWER=1` (new `biodash.py` option *Allow shutdown / reboot from other devices*, `--allow-power`); requests need a custom header, so other websites can't trigger it. Uses `systemctl` when the session allows it, else a `sudo` rule the card shows how to add. `BIODASH_POWER_DRY_RUN=1` for testing.
- **Acknowledgements:** README credits [klei22/viatom-ble](https://github.com/klei22/viatom-ble) (fork of ecostech/viatom-ble), the work the Viatom dashboard and Bio-dash started from; the Viatom dashboard has its own README.
- **Header links:** a fifth entry, Hydro (all dashboards).
- **Tile trends module:** optional `windows` setting for day-scale windows (default unchanged: 1 / 5 / 15 / 60 min).


## V2.8

### Polar H10: debug panel
- **🐞 Debug panel** (button or the `D` key; remembers whether it was open). It refreshes once a second from `/api/status`.
- **Bluetooth radio:** lists every adapter from BlueZ over D-Bus, with address, alias, power state, chip vendor/product and USB ID from sysfs, and bus type. The adapter the H10 was found on and connected through is marked **IN USE**.
- **Link:** state, connect attempt, device, RSSI at scan, connected duration.
- **Measured sample rates:** ECG and ACC Hz against the nominal 130 / 200 (colour-coded), frames/s × samples per frame, HR notifications/s, and R-R intervals per minute.
- **Dropped-sample detection:** gaps between consecutive PMD frames in the H10's own timestamps are counted as lost samples. In testing, one deliberately lost frame reported exactly 73 dropped ECG samples.
- **Browser stats:** WebSocket messages/s, rows/s, ECG display delay and buffer depth, and library versions.
- New `bt_debug.py` (radio discovery and rate meters) and `static/js/debug_panel.js`.

### Viatom O2 Ultra: debug panel
- **🐞 Debug panel**, the same as the Polar's above (button or the `D` key, remembers whether it was open), in the Viatom's sky accent.
- **Bluetooth radio:** every BlueZ adapter with chip, bus and power state, with the one the O2 is on marked **IN USE**. It shares `bt_debug.py` with the Polar dashboard.
- **Link:** state, device, RSSI at scan, radio, connected duration, link-failure count.
- **Data rates** over a 20 s window (the O2 answers a poll every 2 s): valid readings/s against the nominal 0.5/s, notifications/s, polls sent/s, and time since the last reading.
- **Protocol counters:** continuation fragments skipped (non-`0x55`), calibration packets (`0xA5`), out-of-range SpO₂ frames, and poll write failures.
- **GATT:** the notify and write characteristics actually in use, the write mode the link settled on, and the MTU.
- The worker writes `debug.json` once a second and the app serves it at `/api/debug`. `debug.json` is gitignored.

### Atmos-Mini (SEN69C) dashboard redesign
- **Same look as Polar and Viatom:** the header with a live status badge, the scanner panel with a radar pulse and device cards, and slate tiles and chart cards, with an emerald accent.
- **Headline tiles:** CO₂ with ventilation guidance, PM2.5 with its US EPA AQI category (2024 breakpoints, instantaneous), and temperature / RH. **Secondary tiles:** VOC and NOx indices relative to Sensirion's learned baseline, HCHO, and PM1 / PM4 / PM10.
- **Trend charts** for CO₂, PM1 / 2.5 / 10, temperature & RH (dual axis), and VOC / NOx index & HCHO (dual axis), with a **5 min / 15 min / 1 hour** window selector that is remembered between visits.
- **`/api/history`** pre-fills the charts from the newest CSV (reading only the tail of the file, downsampled to 1500 points at most), so a page reload keeps the trend.

### Atmos-Mini: debug panel
- **🐞 Debug panel** (button or the `D` key), the same as the Polar and Viatom panels: Bluetooth radio with the adapter **IN USE** marked, and the link (state, device, RSSI, radio, connected duration, link failures).
- **UART stream health:** readings/s, line interval ± jitter, last-reading age, notifications/s, throughput, bytes per notification, rejected lines, buffer overflows, and rows logged. It also shows the NUS service, TX characteristic and MTU.

### Worker
- **Garbled lines no longer reach the CSV.** A line must have 10 numeric fields; `nan` is accepted, because the SEN69C reports NaN while warming up. Everything else is counted as a rejected line.
- **Same handshake fixes as the Polar:** worker status via `/api/status` (no device-card flicker, "attempt n/3"), 3 connect retries before rescanning, and a scan cut short by a Connect click no longer blanks the device list.

### Radio identification (all debug panels)
- **Radios are identified by Bluetooth address**, with `hciN` shown only as "currently hciN". Linux numbers adapters in enumeration order, so a dongle that comes up first takes hci0 and pushes an internal card to hci1. The address is stable across reboots and replugs.
- **INTERNAL / EXTERNAL tag** on every radio, from the kernel's USB port data (`removable`, then `port/connect_type`, then a vendor fallback for Intel). Native PCIe, SDIO and UART radios count as internal. Wi-Fi+BT combo cards report as USB because their Bluetooth half sits on the M.2 slot's USB lines; the tag disambiguates them. Hover to see how it was determined.

### Omni dashboard
- **Radio routing.** The H10 can monopolise the radio it streams on, so Omni now assigns the H10 and the O2 + scanning to radios:
  - **Auto:** with two or more radios, the H10 goes on an external radio and the O2 plus scanning on another; with one radio, everything shares it.
  - **Pinning:** set per device from the debug panel. Saved to `radio_config.json` by **adapter address**, so it survives hci renumbering. An unplugged pinned radio falls back to auto. Changes apply on the next connection.
  - Before connecting, each device is looked up on its assigned radio, because BlueZ connects through the adapter that discovered the device. The radio list refreshes every 10 s, so a dongle plugged in later is picked up.
- **Scanning no longer freezes while the Polar streams**, unless it would have to share the Polar's radio. On a single-radio machine it pauses as before; the debug panel shows the reason.
- **Fixed:** scanning was hardcoded to `hci1`, so it failed every sweep on single-radio machines. `POLAR_ADAPTER` was never used at all, so the H10 simply connected through whichever radio had scanned it. Both are replaced by the routing above.
- **Bedside-monitor ECG** (sweep, beat flash, QRS beep, gain modes), the same as the Polar dashboard.
- **🐞 Debug panel** (button or `D`):
  - radios with their roles and the routing pickers
  - Polar link (measured ECG/ACC Hz, frames, drops, RSSI)
  - O2 link (readings/s, last reading, fragments, write failures, write mode, MTU)
  - scanner state with the pause reason, and versions
  - new `/api/debug` and `/api/radios` endpoints
- **Fixed:** the tile value overlapped its unit on wide values ("837ms"). Tiles are now laid out as number + unit.

### ECG view toggle (Polar, Omni)
- **🩺 Monitor / 📈 Classic switch** on the ECG card: the bedside-monitor sweep or the original scrolling Chart.js line. The choice is remembered per dashboard.
- The monitor keeps receiving data in Classic, so beat detection and the QRS beep still work. The gain button hides in Classic because it only applies to the monitor.
- Shared as `static/js/ecg_view.js`. The Classic chart only fills while it's visible, so a hidden chart doesn't pile up samples.

### Tile trends (all dashboards)
- **Tap any tile to see its trend.** A live line graph opens under the tiles, with 1 / 5 / 15 / 60 min windows (remembered per dashboard), min / avg / max, and the selected tile highlighted in the dashboard's accent. Tap again, ✕, or Esc to close; tiles are keyboard-accessible.
- **One shared module**, `static/js/tile_trends.js`, used by all four dashboards. It builds its own panel, supports multi-series tiles (a second y-axis when needed), and takes an optional pre-fill function.
- **Polar H10:** heart rate, RMSSD, and **ECG amplitude (peak-to-peak per second)** for the ECG tile. An instantaneous voltage sampled once a second is meaningless; p-p amplitude drops when the strap dries or slips. Pre-filled from the session CSVs via the new `/api/history?metric=hr|rmssd|ecg_pp`.
- **Viatom O2:** SpO₂ (90–100%), pulse, and battery (0–100%). Pre-filled from the worker's daily CSV via the new `/api/history`, with the date taken from the filename and yesterday's file read when a window crosses midnight.
- **Atmos-Mini:** CO₂, PM2.5, **Climate** (temperature + RH on two axes), VOC, NOx, HCHO, and **Particulates** (PM1 / 4 / 10). Pre-filled from the existing `/api/history`.
- **Omni:** SpO₂, ring HR, ECG HR, R-R (per beat), RMSSD and SDNN. Browser history from page load, up to 1 hour.
- **Honest drawing:** whole-number readings are drawn as steps, curves use monotone interpolation (no overshoot), and each series has a minimum or fixed y-range. The built-in trend charts on Viatom and Atmos now use monotone interpolation too.

### Atmos dashboard (was the Atmos-Mini / nRF dashboard)
- **Renamed** `nrf-sen_dashboard/` → `atmos_dashboard/`, and `start_air.sh` → `run_dashboard.sh` to match the other dashboards. Logs now go to `logs_atmos/`.
- **Works with any Atmos device.** The new `atmos_parser.py` reads JSON, `key=value` / `key: value`, CSV with a device-sent header line, or bare numeric CSV, detected per line.
- **Field-name normalisation:** `CO2_ppm`, `PM2.5 [µg/m³]`, `Humidity_RH`, `SCD30 Temp` and `sen55_rh` map to canonical metrics, tagged with their source sensor (`temp@scd30`).
  - When several sensors report the same quantity, tiles use the best source: SEN5x/SEN6x → SHT4x → SCD30 → SFA3X.
  - Unrecognised numeric fields get their own **Other Readings** tiles, with trends.
  - JSON strings (e.g. firmware versions) and timestamps are ignored.
- **Device profiles** for bare CSV, selectable in the debug panel: **Auto** (by field count: 10 → Mini, 13 → S4, 14 → S5), **Atmos-Mini (SEN69C)**, **Atmos-Sphere S4 (SCD30 + SEN44 + SFA3X)**, **Atmos-Sphere S5 (SCD30 + SEN55 + SFA3X)**.
  - The S4/S5 column orders follow each Sensirion library's read order, but the sensor block order is assumed, and the panel says so when those layouts are in use.
  - A device-sent header line or a custom column map overrides the profile.
  - Changes apply live, with no reconnect, and are saved to `atmos_profile.json`.
- **Debug panel → Device Profile:**
  - the profile picker
  - the line format, fields per line, and where column names came from
  - the device header, if one was sent
  - the custom column map (Apply / Clear)
  - a live **Detected fields** list marking which source is **ON TILE** vs **ALT SOURCE**
- **Adaptive layout:** tiles, chart lines and chart cards for metrics the device doesn't report are hidden. The header subtitle shows the detected device, field count and format.
- **Self-describing logs:** the CSV header names every series. If new fields appear mid-session, logging rolls to a `_partN` file with the wider header. History reads any schema, including older logs.
- **Live stream:** now sends named records (`{ts, v: {series: value}}`) instead of positional CSV lines.
- The scanner also recognises "Sphere" and "XIAO" in device names, alongside any Nordic UART advertiser.
- The shared trend module gained `addMetric()` for tiles created at runtime.

### Atmos-Sphere S4 firmware support
- The **S4 profile is now confirmed**: it's exactly what the Atmos-Sphere-s4 firmware's new Bluetooth fallback sends, and the firmware also sends it as a header line, so the dashboard maps by name. The "assumed order" warning now only appears for the S5.

### Radio selection on every dashboard
- **Polar, Viatom and Atmos can now choose their Bluetooth radio**, as Omni already could. Pick it in the debug panel under **Radio for this dashboard**. It's saved to that dashboard's `radio_config.json` by adapter address, so hci renumbering doesn't matter, and applies on the next scan.
- **Complementary auto mode:** the Polar prefers an external radio (the H10 can monopolise its radio while streaming), while Viatom and Atmos prefer an internal one. Dashboards running side by side split sensibly however the radios were numbered.
- Scanning, device lookup and connecting all use the chosen radio. Viatom's continuous scanner restarts on the new radio within a second of a change.
- Shared as `RadioPin` / `bluez_args()` in `bt_debug.py`.

### Atmos over Wi-Fi
- **Connect over Wi-Fi (TCP)** from the scanner panel, alongside Bluetooth. The bytes go through the same receive path, so parsing, logging, trends and debug stats are identical. The last host/port is remembered.
- **New profile "Atmos-Sphere S4 over Wi-Fi"**, confirmed from the firmware: the S4's 9-field TCP stream (PM from the SEN44, averaged temperature/RH, VOC, HCHO, CO₂). Auto-selected by field count.
- The debug panel's Link section shows the **Transport** (Bluetooth / Wi-Fi). A closed connection returns to scanning immediately; a dead host is retried 3× first.

### Ports & launcher
- **Viatom moved to port 5003**, so each dashboard has its own port: Omni 5000 · Polar 5001 · Atmos 5002 · Viatom 5003. Viatom and Omni can now run at the same time.
- **Dashboard links in every header**, with a live dot for each dashboard that's running (`static/js/nav.js`).
- **`launch.sh`** starts several dashboards from one terminal with name-prefixed output; Ctrl+C stops them all. It refuses Omni together with the standalone Polar/Viatom (same devices) and starts Omni first, after its sudo prompt, so its Bluetooth restart can't cut the others' connections.

### Omni session logs
- **Omni now logs like the other dashboards:** raw ECG (130 Hz), ACC (200 Hz), R-R per beat, and 1 Hz vitals (Polar HR, RMSSD, SDNN, SpO₂, ring HR) to `logs_omni/`, archived on the next start.
- **Tile trends are pre-filled from those logs** via the new `/api/history`, so they survive a page reload.
- **Launcher fixes:** `run_dashboard.sh` now changes to its own folder (it failed when started from elsewhere) and stops `omni.py` on Ctrl+C / TERM (it used to be left running).

### Services, keep-awake, auto-reconnect
- **`biodash.py`**, a service manager with an interactive picker (curses TUI) and matching commands (`install`, `status`, `stop`, `uninstall`, `logs`).
  - It runs the chosen dashboards as systemd user services with restart-on-failure, and can start them at boot via lingering.
  - It enforces Omni ↔ standalone Polar/Viatom exclusivity and shows live status and logs.
  - Standard library only.
- **Keep the computer awake:** new `service_run.sh`, and `launch.sh` now runs its session under `systemd-inhibit` (suspend + idle; lid-close optional; `BIODASH_INHIBIT=0` to disable). It falls back gracefully when the inhibitor isn't permitted.
- **Keep the screen on:** a **☀** toggle in every dashboard header (Screen Wake Lock API), re-acquired when the tab becomes visible again. It's disabled, with an explanation, on non-secure origins.
- **Auto-reconnect to the last device** on all four dashboards (`bt_debug.LastDevice`):
  - Bluetooth devices reconnect as soon as they advertise again.
  - The Atmos Wi-Fi host is retried every 15 s.
  - Omni remembers the H10 and the O2 separately.
  - A reconnect bar in the scanner panel (`reconnect.js`) offers Pause / Resume / Forget, backed by `/api/last-device` on every dashboard.
- Omni's launcher skips its `sudo` Bluetooth restart when running as a service (no terminal to prompt on).

### Recordings saved to Documents
- **All four dashboards now save to `Documents/Bio-dash/<Dashboard>/<device>/<date>/<time>_<stream>.csv`** instead of `logs_*` folders inside the repo. The Documents folder is found via XDG user dirs (fallback `~/Documents`, override `BIODASH_DATA_DIR`).
- **One set of files per session**, opened when a device connects and closed when it disconnects, so every file belongs to exactly one sensor. Nothing is written before a device connects.
- **Device folders:** named after the device, with the Bluetooth address added when the name is shared by every unit of a model (`Atmos-Sphere-S4`, `Atmos-Mini`). Wi-Fi devices are named by host and port.
- **Polar:** `ecg`, `acc`, `rr` (renamed from `ppi`).
- **Viatom:** `vitals` per session with full epoch timestamps. This replaces the daily `health_log` with time-of-day-only stamps, and the midnight workaround is gone.
- **Atmos:** `readings` (+ `_partN` when new fields appear).
- **Omni:** writes into each device's own folder: the H10 gets `ecg` / `acc` / `rr` / `hrv`, the O2 gets `vitals`. Its old combined vitals file mixed both devices.
- **History:** trend history and chart pre-fill read the newest recording of the current device (Atmos and Viatom only use the connected unit's sessions, so switching units doesn't mix histories).
- Each debug panel shows **Saving to**. The launchers print the recordings folder instead of archiving into `logs_*/archive/`.
- Shared helpers in `bt_debug.py`: `data_root()`, `device_folder()`, `SessionFiles`, `latest_session_files()`.

### Fixes
- The Beep button's "on" tint never showed, because the base slate classes won over the rose ones in the compiled CSS. Toggles now swap classes rather than stacking them.
- **Units in uppercase headers.** CSS `uppercase` turned `µg/m³` into "MG/M³", because µ capitalizes to a Greek capital mu that reads as milligrams. Units and formulas (`µg/m³`, `ppm`, `NOx`, `SpO₂`) now keep their case, and the Polar accelerometer unit is corrected to `mG` (milli-g).

## V2.7

- **ECG monitor follows the dashboard theme.** The black-and-green screen is replaced by the same slate card, rose-500 trace (matching the HR tile), a subtle slate ECG grid, pill-style Gain and Beep buttons, and an amber "No Signal" badge. The sweep display, beat flash, beep and gain modes are unchanged.
- `EcgMonitor` now takes a `theme` option (trace, glow, background and grid colours), so the monitor can be re-coloured without touching the renderer.

## V2.6

- **Removed in-tree version archives.** The `v1-code/` … `v4-code/` folders are gone from every dashboard: 10 folders, including old Polar session logs. Earlier versions now live as their own tagged releases, so each dashboard folder holds only the current code.

## V2.5

### Polar H10: bedside-monitor ECG
- **Hospital-style sweep display.** The live ECG no longer scrolls. It is drawn like a patient monitor: a pen sweeps left to right and wipes a small gap just ahead of itself, over an ECG-paper grid (1 mm / 5 mm squares at the standard 25 mm/s).
- Phosphor-green trace with glow on a black screen, with a large HR readout and a heart that flashes on each detected QRS.
- **QRS beep** (off by default). The toggle is needed because browsers only allow audio after a click.
- **Gain button** cycles AUTO → 5 → 10 → 20 mm/mV. 10 mm/mV is the clinical standard; AUTO fits the waveform to the screen.
- A 0.5 Hz high-pass filter on the display removes breathing and movement baseline wander. It only affects what's drawn; the CSV logs are untouched raw data.
- Samples are replayed at true speed ~0.9 s behind live, so the pen moves smoothly even though the H10 delivers ECG in ~0.5 s bursts. This also works when the browser is on a different machine than the Orin, even if their clocks don't match.
- Shows **NO SIGNAL** when data stops.
- New `static/js/ecg_monitor.js`, a standalone renderer that doesn't use Chart.js.

## V2.4

### Viatom O2 Ultra dashboard redesign
- **Rebuilt on the Polar H10 layout.** It now has the same Tailwind dark theme, header with a live status badge, scanner panel with a radar pulse and device cards, and metric tiles and chart cards. It uses a sky-blue accent so the two dashboards are easy to tell apart.
- Three tiles: **SpO₂**, **Pulse Rate** (with the session min/max), and **Device Battery**, whose bar turns amber at 40% and red at 20%.
- SpO₂ and pulse trend charts show 60 s windows sized for the O₂'s ~2 s update rate.
- The scanner panel follows the worker state. It shows "Handshaking…" while connecting, hides once streaming, and returns on disconnect. Device cards only rebuild when the set of devices changes.
- Calibration (`0xA5`) shows as "Sensor Settling…" instead of blanking the readings.
- If the stream goes quiet for 15 s while connected, the page shows a **Signal Lost** badge. The badge shows **Server Offline** if the web server stops.
- The old hand-written `style.css` is removed. The page now uses compiled Tailwind, and `fetch_vendor.sh --tailwind` rebuilds it.

## V2.3

- **`setup_env.sh`** creates an isolated `.venv/` (whole suite or a single dashboard), checks for Python ≥ 3.10, and verifies imports. The launch scripts use `.venv/` automatically when it exists. `.venv/` is gitignored.

## V2.2

- **Fully offline.** Chart.js 3.9.1, Luxon 3.0.1, chartjs-adapter-luxon 1.2.0 and chartjs-plugin-streaming 2.0.0 are included in each dashboard's `static/vendor/`. These are the same versions and files the CDN served.
- **Tailwind is compiled instead of loaded from the play CDN.** The play CDN downloaded a ~400 KB script and compiled CSS in the browser on every page load. Each page now loads a ~11 KB static `tailwind.css`, which is also lighter on the Orin.
- **New `fetch_vendor.sh`.** It downloads the libraries with `wget`, verifies them against pinned SHA-256 checksums, and rebuilds Tailwind with `--tailwind`. Set `CDN=` to use a mirror.

## V2.1

### Structure
- **HTML, CSS and JS moved out of Python.** Every dashboard now has `templates/index.html`, `static/js/dashboard.js`, and `static/css/style.css` where it has custom styles. The Python files contain only server and BLE logic.
- FastAPI dashboards (Polar, air) serve the page with `FileResponse` and mount `/static`. Flask dashboards (Viatom, Omni) use `send_from_directory` with an explicit `root_path`, so they work regardless of the directory you launch from.
- **Per-dashboard `requirements.txt`** for all four dashboards. The root `requirements.txt` pulls in all of them with `-r`. `bleak` and `polar-python` are capped below their next major version to prevent surprise upgrades.

### Fixes
- **Air dashboard page was broken.** Its script used `split('\n')` inside a non-raw Python string, so Python turned `\n` into a real newline and the browser hit a JS syntax error. The whole page script failed. Moving the JS into its own file fixes this, and the rest of that bug class along with it.

## V2 — 2026-09-22

A performance and reliability pass across all four dashboards. Verified on hardware: Polar H10 handshake and streaming, and Viatom O2 Ultra connect → SpO₂/HR logging.

### Polar H10 (`polar_dashboard/`)

**Performance**
- WebSocket traffic is batched. The server used to send one message per sample (~330 msgs/s per open tab) and now sends one message per poll per stream: 365 ECG samples arrive as 5 messages instead of 365.
- Samples are timestamped from the H10's own PMD frame clock, locked to wall-clock time on the first frame (it re-anchors if drift exceeds 1 s). This removes BLE delivery jitter at the source; ECG spacing is a steady ~7.7 ms. The browser-side `lastEcgTs += 7.69` workaround is gone.
- CSV rows are written with `writerows` per frame instead of one call per sample.

**Fixes**
- The scanner panel used to vanish permanently after the first packet. It now comes back after 5 s without data, so you can reconnect without reloading.
- Connecting no longer flickers. The worker reports scanning / connecting / streaming through a new `/api/status` endpoint, and the page shows "Handshaking… (attempt n/3)" instead of redrawing stale device cards.
- Clicking Connect in the middle of a scan no longer blanks the device list.
- Up to 3 connect attempts before falling back to scanning, which handles BlueZ's frequent first-attempt abort.
- The WebSocket reconnects automatically.
- The app now follows a new log file when the worker restarts.
- Callbacks read `polar_python`'s real fields (`heartrate`, `rr_intervals`, `timestamp`, `data`) instead of guessing through `__dict__`.
- Device names are HTML-escaped in the scanner list.

### Viatom O2 Ultra (`viatom_dashboard/`)
- **Continuous scanning.** Short 4 s scan windows kept missing the O2's slow adverts. The worker now runs one long-lived scanner and keeps devices listed for 30 s after their last advert. It logs `Spotted …` and `Lost …` events.
- The name filter now matches `Band-WU` (the O2 Ultra's advertised name), not just the hardcoded MAC.
- Removed a hardcoded `~/Forks/Bio-dash/...` path, so the dashboard runs from any checkout location.
- The notify fallback used to subscribe to the *service* UUID, which can never work. It now fails with a clear error instead.
- One failed poll write no longer drops the link. The worker flips the write mode and retries, giving up only after 3 misses.
- Notification fragments that don't start with the `0x55` frame header are ignored instead of being parsed into garbage readings.
- Removed dead `static/dashboard.js` and `templates/index.html`, which called a nonexistent `/api/data` endpoint.

### Atmos-Mini / SEN69C (`nrf-sen_dashboard/`)
- **Fixed dropped lines.** The UART handler processed only the first line per BLE notification, so readings lagged and backed up when two lines arrived together.
- The tailer used to glob and stat the log directory on every 50 ms poll; that now happens every 2 s.
- The scanner also matches `Atmos` and any device advertising the Nordic UART service, so board swaps and renames still show up.
- Reuses the BLE device found during the scan instead of running a second full scan to connect.
- `start_air.sh` pointed to port 8000; it now points to 5002, where the app actually runs.
- Auto-reconnect, stale-data detection and escaping, same as the Polar dashboard.

### Omni (`Omni-dashboard/`)
- **Fixed lost samples.** The SSE handler cleared the shared ECG/ACC buffers while the BLE thread was appending to them, which dropped samples and split data between open tabs. It's replaced by a thread-safe ring buffer with a cursor per client, so every tab gets the full stream.
- The R-R chart gets one point per heartbeat. It used to replot the last value ~20×/s.
- A `0xA5` calibration byte from the O2 no longer tears down a healthy connection, because link state and display state are now separate.
- Uses the same sensor-clock timestamps as the Polar dashboard.

### Repo
- `requirements.txt` now lists FastAPI and uvicorn, which were missing, and drops the unused `unicorn`, `numpy` and `flask-cors`.
- Added `.gitignore` for `__pycache__/`, `logs*/` and runtime JSON so session and health data don't get committed.
- `bleak` minimum raised to 0.22; tested on 3.0.2.

### Known limitations
- Viatom and Omni both use port 5000.
- Omni defines `POLAR_ADAPTER = "hci0"` but doesn't use it yet, so both devices share `hci1` and scanning pauses while the Polar streams.
- Pages load Tailwind and Chart.js from CDNs, so a fully offline machine won't render them.

## V1

The original dashboards: Polar H10 (v1 → v4 iterations), Viatom O2, SEN69C air monitor and Omni. Earlier versions are available as tagged releases and are no longer kept as `vN-code/` folders in the tree.
