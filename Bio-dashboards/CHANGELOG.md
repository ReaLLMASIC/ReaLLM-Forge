# Changelog

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
