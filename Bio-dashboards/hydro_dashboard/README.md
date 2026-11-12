# Hydro-dash — HidrateSpark smart bottle

Live hydration dashboard for the **HidrateSpark** bottle (PRO / PRO 2): every sip with its time,
today's intake against a goal, fill level from the bottle's weight sensor, refills, battery.
The Water page of the app (`#/water`).

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
2. Start Bio-dash: `./launch.sh` from the release folder, or as a
   service via `./biodash.py`.
3. Open <http://localhost:5000/#/water>, **shake or tip the bottle** to wake it, and click **Connect**.
   If it isn't listed, tick **Show all nearby devices** — some firmware advertises another name.
4. **Bottle Settings:** set your daily goal, or work one out in **How Much Should I Drink?** (below). The bottle size is read from the bottle ("Automatic"), or
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

**The radio isn't pinned.** Hydro-dash listens on every radio at once and connects through the one
that hears the bottle best (the one that worked last time gets a small head start). A radio whose
connection fails is passed over for 10 minutes, or for a week if the bottle's service list came back
empty through it (an old Bluetooth 4.0 dongle, TROUBLESHOOTING 12). A laptop's built-in radio
shares its antenna with Wi-Fi and often misses the bottle's short adverts, so a Bluetooth 5 dongle
still helps (tested: TP-Link UB500 Plus). To pin one radio anyway, pick it under *Radio for this
dashboard* under Device → Troubleshooting; *Any radio* undoes that. Pins saved by V5.2 and older don't
carry over: Hydro now starts unpinned.

**If the fill level is wrong**, the bottle's own calibration is off. Stand it upright with the cap
on, completely full or completely empty, and press **It's full now** / **It's empty now** (click
twice to confirm). That tells the bottle to store a new value, which arrives with the next sip.
**After "It's empty now", fill it and press "It's full now".** The bottle takes empty and full as
one calibration and counts no sips until it has both. The page keeps a banner up until you do
(with its own "It's full now" button), and it stays across reloads and restarts.

After that it reconnects to the bottle by itself whenever it wakes (auto-reconnect, like the other
dashboards).

## What you see

- **Today** — intake vs. goal (ring), what's left to go (and in bottles), and your **pace**: the
  goal spread evenly over the reminder hours (from / to), so it says *on pace* or how far behind
  you are, and roughly how much per hour gets you there. Also **In the bottle**, **Last sip**,
  **Sips / Refills today**, **Battery**. Tap a tile for its trend (1 h · 6 h · 1 day · 7 days).
- **How Much Should I Drink?** — a suggested daily goal from drinks, with the working shown:
  halfway between the National Academies reference intake (3.7 L total for men, 2.7 L for women,
  of which about 80 % usually comes from drinks) and your body weight × 33 mL/kg ("half your weight
  in pounds, in fl oz"), plus about 0.7 L per hour of exercise and extra for the day's weather,
  rounded and kept within 1.5–5 L. **Use as my daily goal** sets it (and sends it to the bottle).
  The inputs are kept with the other settings. It's a rule of thumb, not medical advice: a doctor's
  fluid limit or a kidney or heart condition comes first.
  - **Weather: today's local forecast.** Find your town once; the day's "feels like" high (heat
    and humidity together) adds 250 mL from 25 °C / 77 °F, 500 from 28 °C / 82 °F, 750 from
    32 °C / 90 °F and 1 L from 36 °C / 97 °F. Cold days add nothing and take nothing away: cold air
    is dry and thirst runs low, so the goal stays. The forecast comes from
    [Open-Meteo](https://open-meteo.com) (free, no account; only the town's coordinates are sent),
    needs internet on the computer running the dashboard, and is cached for 30 minutes. Offline,
    the goal is worked out without it and the card says so. *Hot (by hand)* adds 500 mL instead.
  - **Update my goal every morning:** from 05:00 the dashboard sets the day's goal from the
    suggestion, with that day's weather, and sends it to the bottle, even with no page open. It
    waits until noon for the forecast if the network isn't up yet. A goal you type in yourself
    lasts until the next morning.
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

## Device → Troubleshooting

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
differs, Device → Troubleshooting shows it (handshake failed, *Undecoded sip frames* rising, or no sip
characteristic found), and `_raw.csv` already holds what's needed to adapt the decoder.

**Draining stored sips.** Current bottles (firmware 100+) are drained the way the official app does
it: each record is acknowledged, then the next is requested. *Skipped records* under Device → Troubleshooting
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
`water.py` goal maths and the Open-Meteo forecast (tests: `python3 test_water.py`, `python3 test_goal_settings.py`) · `../web/js/water_calc.js` pace maths (tests: `node test_water_calc.js`) · `hydro_worker.py` Bluetooth worker · `hydro_test.py` standalone bottle test · `api.py` web API (mounted at `/api/d/water`) · `~/.config/bio-dash/hydro_settings.json` (kept across updates) capacity / goal /
the bottle's last known calibration (created on first save).

## Acknowledgements

Bio-dash started from **[klei22/viatom-ble](https://github.com/klei22/viatom-ble)** (a fork of
[ecostech/viatom-ble](https://github.com/ecostech/viatom-ble), MIT), the Python script for reading
Viatom oximeters over Bluetooth that kicked off these dashboards. HidrateSpark protocol sources are
listed in `PROTOCOL.md`.
