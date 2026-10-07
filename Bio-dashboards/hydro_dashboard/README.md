# Hydro-dash — HidrateSpark smart bottle

Live hydration dashboard for the **HidrateSpark** bottle (PRO / PRO 2): every sip with its time,
today's intake against a goal, fill level from the bottle's weight sensor, refills, battery.
Port **5004**.

> 🧪 **Still being tested.** Hydro-dash is new in V3.0 and has been run on one HidrateSpark PRO 2
> so far. Connecting, live and replayed sips and battery work; **fill-level calibration, bottle
> capacity and long-session behaviour are still being checked**, so treat fill level and totals as
> provisional. Other bottle models are untested.

## First run

1. **No phone app needed.** The bottle accepts one connection at a time, so if the HidrateSpark
   app is installed on a nearby phone, force-quit it while Hydro-dash is in use.
2. Start it: `./run_dashboard.sh` here, `./launch.sh hydro` from the release folder, or as a
   service via `./biodash.py`.
3. Open <http://localhost:5004>, **shake or tip the bottle** to wake it, and click **Connect**.
   If it isn't listed, tick **Show all nearby devices** — some firmware advertises another name.
4. **Bottle Settings:** set the capacity (e.g. 621 mL for 21 oz) and your daily goal.
5. **Calibrate the fill level** (optional but recommended): with the bottle upright and settled,
   press **It's full now** when full and **It's empty now** when empty. Until then the fill level
   is estimated; a detected refill also sets the "full" point automatically.

After that it reconnects to the bottle by itself whenever it wakes (auto-reconnect, like the other
dashboards).

## What you see

- **Today** — intake vs. goal (ring), **In the bottle**, **Last sip**, **Sips / Refills today**,
  **Battery**. Tap a tile for its trend (1 h · 6 h · 1 day · 7 days).
- **Today by hour** and **Last 7 days** vs. goal.
- **Today's sips** — each sip, marked *live* or *synced from bottle* (sips taken while away are
  buffered on the bottle and replayed with their original times on the next connection).
- Volumes in mL or fl oz (Bottle Settings).

## Recordings

`Documents/Bio-dash/HidrateSpark/<bottle>/<date>/<time>_…`:

| File | Contents |
|---|---|
| `_sips.csv` | one row per sip: time, mL, source (live / replayed), frame layout, raw frame |
| `_status.csv` | fill level, weight, battery, cap — every 5 s |
| `_events.csv` | connect / disconnect, cap open / close, refills, calibration |
| `_raw.csv` | every notification from every characteristic, as hex |

Totals and charts are computed from `_sips.csv`, so they survive restarts and include replayed sips.

## Debug panel (D)

Radio and link (as in the other dashboards), **Bottle Protocol** (handshake, sip characteristic,
live / replayed / duplicate / undecoded counts, drain writes), **Sensors** (weight raw / settled,
calibration anchors, cap, battery, serial, firmware), and **every characteristic the bottle
exposes with its last raw frame**.

## Troubleshooting

How the PRO 2 was brought up, every problem hit on the way, and a quick checklist:
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## PRO 2 status

The protocol is documented by the community for the PRO (v1) — see [PROTOCOL.md](PROTOCOL.md).
If your PRO 2 differs, the debug panel shows it (handshake failed, *Undecoded sip frames* rising,
or no sip characteristic found), and `_raw.csv` already holds what's needed to adapt the decoder.

## Testing the bottle without the dashboard

`hydro_test.py` (stop Hydro-dash first, phone Bluetooth off):

```bash
python3 hydro_test.py                  # scan (+ probe if it doesn't advertise) -> list its services -> listen 60 s
python3 hydro_test.py scan             # every device nearby, strongest first, named or not
python3 hydro_test.py probe            # scan, then connect briefly to the strongest unnamed devices to find the bottle
python3 hydro_test.py dump <address>   # services, characteristics, readable values, protocol check
python3 hydro_test.py listen <address> # handshake + drain, every notification printed live
```

The listen step prompts you to sip, open / close the cap and refill, decodes what it knows, and
saves a capture to `Documents/Bio-dash/HidrateSpark/test-captures/`.

## Files

`hydro_protocol.py` protocol + state machines (tests: `python3 test_hydro_protocol.py`) ·
`hydro_worker.py` Bluetooth worker · `hydro_test.py` standalone bottle test · `app.py` web server · `hydro_settings.json` capacity / goal /
calibration (created on first save).

## Acknowledgements

Bio-dash started from **[klei22/viatom-ble](https://github.com/klei22/viatom-ble)** (a fork of
[ecostech/viatom-ble](https://github.com/ecostech/viatom-ble), MIT), the Python script for reading
Viatom oximeters over Bluetooth that kicked off these dashboards. HidrateSpark protocol sources are
listed in `PROTOCOL.md`.
