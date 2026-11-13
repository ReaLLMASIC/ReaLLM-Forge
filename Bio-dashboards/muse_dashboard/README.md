# Muse-dash (Muse S Athena)

Live EEG, brain-wave band powers, sensor contact, fNIRS (blood oxygenation on the forehead), heart rate
from the forehead PPG, head motion and battery from a **Muse S Athena** headband: the Brain page of the app (`#/brain`). The page is
`../web/js/pages/brain.js`, in every theme.

```bash
../launch.sh                        # then open http://localhost:5000/#/brain (turn Brain on in Settings)
python3 fake_muse.py                # the worker against a stand-in headband: no Muse or Bluetooth needed
```

Switch the headband on (it allows **one connection at a time**, so close the Muse phone app), pick it in
the scanner and press **Connect Headband**. Later it reconnects by itself when it's back in range
(**Pause** / **Forget** in the scanner).

> 🧪 **Not yet tried on a real headband.** The protocol comes from OpenMuse, which was validated on real
> recordings, and the decoder here matches OpenMuse's (`test_muse_protocol.py`), but everything after
> the decoder has only run against the stand-in. Use the checklist below for the first session.

## On the page

| | |
|---|---|
| **Heart rate** | From the forehead PPG: every second the optical channel with the clearest pulse (autocorrelation over 8 s) is picked, beats are its peaks (at least 0.33 s apart, refined between samples), the rate is the median of the last 5 intervals. Shows "no clear pulse yet" when nothing is clear enough: keep still for a few seconds. HRV (RMSSD of the last 20 intervals) sits on the pulse card. |
| **Alpha share** | 8-13 Hz as a share of 1-44 Hz, averaged over the sensors in contact (smoothed over ~3 s on the page). |
| **Sensor contact** | TP9 / AF7 / AF8 / TP10: good, fair, bad or off, from how much each signal swings after its slow drift is removed (under 40 µV good, under 90 µV fair) or whether it sits stuck at the end of its range or flat (off the skin). |
| **Battery** | From the headband's battery packets (and its status answers). |
| **Live EEG** | The last 8 s per sensor. Display filters: 1-40 Hz and a 60 Hz notch (tap for 50 Hz or none), or **Raw**; scale ±25-500 µV; **+ AUX** adds the 4 extra channels. The recording is always raw. |
| **Brain waves** | Delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-44 Hz as shares, now and over the last 2 minutes. |
| **fNIRS** | Oxy- (HbO) and deoxy-haemoglobin (HbR) change on each side, from the 730 and 850 nm light (modified Beer-Lambert), outer optodes when the preset has them, against the first 10 s. **⟲ Zero** takes a new 10 s baseline (e.g. after settling in, before a task). The path length isn't measured, so it assumes 3 cm × DPF 6: read the direction and shape, not the µM. |
| **Pulse** | The PPG channel the heart rate comes from, high-passed and flipped (more blood, less light back). |
| **Head motion** | Accelerometer (g), tilt forward / back and left / right from gravity, turning (°/s), still / moving. |
| **Device** sheet | Disconnect, auto-reconnect, the radio, and under *Troubleshooting* the worker's numbers: rates against 256 / 64 / 52 Hz, dropped samples, clock restarts, firmware, versions. |
| **What the headband sends** | The **preset**. |

## Presets

What the headband sends, chosen at the bottom of the page; default **p1041** (everything). See `PROTOCOL.md` for
the table. `p21` sends EEG and motion only, with no light on the forehead. A new preset starts a new
session, because the files' columns change.

## Recordings

`<Documents>/Bio-dash/Muse/<headband>/<date>/<time>_<stream>.csv`, one set per connection:

| Stream | Columns | Rate |
|---|---|---|
| `eeg` | `Timestamp_Epoch_ms, TP9_uV, AF7_uV, AF8_uV, TP10_uV` (+ `AUX1_uV` .. `AUX4_uV` on 8-channel presets) | 256 Hz |
| `optics` | `Timestamp_Epoch_ms` + 4, 8 or 16 channels (`LO_NIR` ...) | 64 Hz |
| `imu` | `Timestamp_Epoch_ms, ACC_X_g, ACC_Y_g, ACC_Z_g, GYRO_X_dps, GYRO_Y_dps, GYRO_Z_dps` | 52 Hz |
| `bands` | `Timestamp_Epoch_ms`, `<sensor>_<band>` (log10 µV², 4 × 5), `<sensor>_contact` (1 good, 2 fair, 4 bad, 0 off) | 1 / s |
| `heart` | `Timestamp_Epoch_ms, IBI_ms, HR_BPM` | per beat |
| `fnirs` | `Timestamp_Epoch_ms, L_HbO_uM, L_HbR_uM, R_HbO_uM, R_HbR_uM` | 1 / s |
| `battery` | `Timestamp_Epoch_ms, Battery_pct` | as reported |

Times come from the headband's own clock, mapped onto the computer's (see `PROTOCOL.md`, *Times*).

## Files

| | |
|---|---|
| `muse_protocol.py` | Commands, packets, decoding, the clock (standard library only) |
| `muse_signal.py` | Band powers and contact, the pulse and heart rate, fNIRS (numpy) |
| `muse_worker.py` | Bluetooth (bleak): scan, connect, start the preset, decode, record, the once-a-second maths |
| `api.py` | Web API (mounted by `../server.py` at `/api/d/brain`): follows the session's files and hands new rows to the page |
| `fake_muse.py` | The stand-in headband (a fake bleak), for the tests |
| `../web/js/pages/brain.js`, `../web/js/eeg.js` | The page; the EEG traces with their display filters |

## Tests

```bash
python3 test_muse_protocol.py     # the decoder against OpenMuse's, the clock (10 tests)
python3 test_muse_signal.py       # bands, contact, the pulse, fNIRS on known signals (5 tests)
python3 test_muse_dashboard.py    # everything with the stand-in: worker, files, app, page, presets (28 checks, ~1 min)
```

## First session with the real headband: what to check

1. **It's listed.** The scanner lists names containing "Muse". If it isn't listed, the name may differ:
   send the name `tools/ble_scan.py` shows.
2. **It streams.** Under Device → Troubleshooting: EEG near 256 Hz, optics near 64 Hz, motion near 52 Hz, firmware and
   battery filled in. If the rates stay at 0 after connecting, send the worker's log (it prints what the
   headband sent that it couldn't use).
3. **Contact.** All four dots green once it sits well (wet the sensors behind the ears). Eyes closed for 10 s
   should push the alpha share up: the classic check that the EEG is real.
4. **Pulse.** The heart rate should match a Polar H10 worn at the same time within a few bpm once you keep
   still (the Pulse card says which optical channel it's using).
5. **fNIRS.** Holding your breath for ~20 s should move HbO and HbR on both sides; press **⟲ Zero** first.
6. **Head position.** Tilt forward: "Tilt forward / back" should go one way and come back. If it reads
   reversed, or left / right is swapped, say so: the axes are set from OpenMuse's channel order, not checked
   on a head yet.
