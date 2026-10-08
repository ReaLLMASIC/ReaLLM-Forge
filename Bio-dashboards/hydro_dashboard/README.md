# Hydro-dash — HidrateSpark smart bottle

Live hydration dashboard for the **HidrateSpark** bottle (PRO / PRO 2): every sip with its time,
today's intake against a goal, fill level from the bottle's weight sensor, refills, battery.
Port **5004**.

> 🧪 **Still being tested.** Hydro-dash has been run on one HidrateSpark PRO 2 so far. Working on
> that bottle: connecting, live and replayed sips, battery, and new in V3.1 the fill level from the
> bottle's own calibration, bottle size read from the bottle, and glow. **Still being confirmed:**
> reminders. Other bottle models are untested.

> ⚠ **The radio matters.** If the page says connected but shows no firmware, battery or sips, you
> are probably on an old USB dongle: see `TROUBLESHOOTING.md` section 12 and the
> [radio diagram](../docs/ARCHITECTURE.md#7-hydro-dash-which-bluetooth-radio).

## First run

1. **No phone app needed.** The bottle accepts one connection at a time, so if the HidrateSpark
   app is installed on a nearby phone, force-quit it while Hydro-dash is in use.
2. Start it: `./run_dashboard.sh` here, `./launch.sh hydro` from the release folder, or as a
   service via `./biodash.py`.
3. Open <http://localhost:5004>, **shake or tip the bottle** to wake it, and click **Connect**.
   If it isn't listed, tick **Show all nearby devices** — some firmware advertises another name.
4. **Bottle Settings:** set your daily goal. The bottle size is read from the bottle ("Automatic"), or
   pick one of the sizes HidrateSpark sells (17 / 20 / 21 / 24 / 30 / 32 oz) from the list.
5. **Take a sip.** The bottle sends its own "empty" and "full" weights with every sip record, and
   the fill level appears from the first one. There is nothing to calibrate in the dashboard.

6. **Glow & Reminders:** press **✨ Glow now** to light the bottle. Pick a **style** (pulse, spin,
   flash, solid, rainbow) and **two colours** (the ring next to them previews the result) and press
   **Send glow to bottle**: the bottle stores it and plays it once, and uses it from then on for
   *Glow now* and reminders. Set when you want reminders
   (from / to, every N minutes), whether the bottle glows only when you're behind on your goal or at
   every reminder, with or without sound, plus glow on reaching the goal and on each sip. **Save &
   send to bottle** stores the schedule on the bottle itself, so it keeps reminding you with the
   dashboard closed. It's re-sent on every connect, along with the time of day, goal and today's total.

**Use a USB Bluetooth dongle if you can.** A laptop's built-in radio shares its antenna with Wi-Fi
and often misses the bottle's short adverts; Hydro-dash picks an external radio automatically when
one is plugged in.

**If the fill level is wrong**, the bottle's own calibration is off. Stand it upright with the cap
on, completely full or completely empty, and press **It's full now** / **It's empty now** (click
twice to confirm). That tells the bottle to store a new value, which arrives with the next sip.

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
live / replayed / duplicate / undecoded counts, bottle family, drain mode, requests / acks, skipped
records), **Sensors** (weight raw / settled, the bottle's empty / full calibration, whether the
live weight is on that scale, where the fill level comes from, capacity, cap, battery, serial,
firmware), and **every characteristic the bottle
exposes with its last raw frame**.

## Troubleshooting

How the PRO 2 was brought up, every problem hit on the way, and a quick checklist:
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## PRO 2 status

Verified on a 21 oz PRO 2 (firmware 100.64); see [PROTOCOL.md](PROTOCOL.md). If another bottle
differs, the debug panel shows it (handshake failed, *Undecoded sip frames* rising, or no sip
characteristic found), and `_raw.csv` already holds what's needed to adapt the decoder.

**Draining stored sips.** Current bottles (firmware 100+) are drained the way the official app does
it: each record is acknowledged, then the next is requested. *Skipped records* in the debug panel
should stay at 0. If the bottle doesn't answer that way, Hydro-dash falls back to the V3.0 method
for the connection and says so (*Drain mode: fast (fell back)*). To force the old method, start it
with `BIODASH_HYDRO_DRAIN=fast`.

## Testing the bottle without the dashboard

`hydro_test.py` (stop Hydro-dash first, phone Bluetooth off):

```bash
python3 hydro_test.py                  # scan (+ probe if it doesn't advertise) -> list its services -> listen 60 s
python3 hydro_test.py scan             # every device nearby, strongest first, named or not
python3 hydro_test.py probe            # scan, then connect briefly to the strongest unnamed devices to find the bottle
python3 hydro_test.py dump <address>   # services, characteristics, readable values, protocol check
python3 hydro_test.py listen <address> # handshake + drain, every notification printed live
python3 hydro_test.py listen <address> --drain fast   # drain the V3.0 way (default: auto, by firmware)
python3 hydro_test.py watch            # when does the bottle advertise? one mark per second for 60 s
python3 hydro_test.py listen <address> --adapter hci1   # any command on a specific radio (e.g. a USB dongle)
```

The listen step prompts you to sip, open / close the cap and refill, decodes what it knows, and
saves a capture to `Documents/Bio-dash/HidrateSpark/test-captures/`.

## Files

`hydro_protocol.py` protocol + state machines (tests: `python3 test_hydro_protocol.py`) ·
`hydro_worker.py` Bluetooth worker · `hydro_test.py` standalone bottle test · `app.py` web server · `~/.config/bio-dash/hydro_settings.json` (kept across updates) capacity / goal /
the bottle's last known calibration (created on first save).

## Acknowledgements

Bio-dash started from **[klei22/viatom-ble](https://github.com/klei22/viatom-ble)** (a fork of
[ecostech/viatom-ble](https://github.com/ecostech/viatom-ble), MIT), the Python script for reading
Viatom oximeters over Bluetooth that kicked off these dashboards. HidrateSpark protocol sources are
listed in `PROTOCOL.md`.
