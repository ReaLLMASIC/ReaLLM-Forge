# Viatom O2 dashboard

Live SpO₂, pulse rate and battery from a **Viatom / Wellue** oximeter (Checkme O2 Ultra, O2Ring and
similar; they advertise as "O2…", "Checkme…" or "Band-WU …"). The Oxygen page of the app (`#/oxygen`).

## Credits

This dashboard — and the Bio-dash project that grew out of it — started from
**[klei22/viatom-ble](https://github.com/klei22/viatom-ble)**, a Python script for reading Viatom
ring and wrist oximeters over Bluetooth Low Energy, itself a fork of
[ecostech/viatom-ble](https://github.com/ecostech/viatom-ble) (both MIT-licensed). Their work on
talking to these devices is what kicked off the Bio-dash dashboards. Thank you!

## Running it

```bash
../launch.sh                # from the release folder: ./launch.sh
```

Open <http://localhost:5000/#/oxygen>, wear the band so it's awake, and click **Connect**. Close the Viatom /
ViHealth phone app first — the band accepts one connection at a time. After the first connection
it reconnects automatically whenever the band is in range.

## How it works

- `ble_worker.py` finds the band (by name or its health service `14839ac4-…`), connects, and polls
  it every 2 s for a reading (SpO₂, pulse, battery). A different band can be pinned with
  `BIODASH_VIATOM_MAC=<address>`, but isn't needed — bands are found by name.
- `api.py` (mounted by `../server.py` at `/api/d/oxygen`) serves the history; the server passes each
  new reading to the open pages.
- With several people, each extra person's ring is one more `ble_worker.py`, with its state in
  `~/.config/bio-dash/people/<id>/oxygen/`.
- Recordings: `Documents/Bio-dash/Viatom O2/<band>/<date>/<time>_vitals.csv`
  (time, SpO₂, pulse, battery), one file per connection.
- **Device → Troubleshooting** on the page: radio, link, data rates, versions.

## Utilities

- `dump_gatt.py <address>` — list a device's services and characteristics.
