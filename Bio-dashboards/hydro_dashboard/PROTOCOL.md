# HidrateSpark BLE protocol (as used by Hydro-dash)

Hydro-dash implements the bottle protocol in its own code (`hydro_protocol.py`). The protocol
**facts** below come from public community reverse-engineering and from reading how the official
Android app talks to the bottle; no code from any of them is included (Bio-dash is Apache-2.0; one
source is GPL-3.0, the app is proprietary).

**Sources**
- [HidrateSpark-MQTT-bridge](https://github.com/loryanstrant/HidrateSpark-MQTT-bridge) (MIT) —
  `docs/SYNC_AND_PROTOCOL.md`: characteristics, cap/weight signals, refill heuristic. Measured on
  firmware 80.18, nRF52832 (HidrateSpark Steel / PRO).
- [HA-Hidratespark](https://github.com/bditter/HA-Hidratespark) (MIT) — the "percent / seconds-ago"
  sip layout and stuck-record guard. Tested on a PRO (v1) 21 oz.
- [HydroSync](https://github.com/maxperron/HydroSync) (GPL-3.0) — original handshake and frame layout.
- The official HidrateSpark Android app — the meaning of the MIN / MAX calibration fields, the
  weight-based sip volume, the bottle-size characteristic, the acknowledge / next commands and
  which firmware uses them, and the calibration commands.

**PRO 2: verified** on a 21 oz bottle (firmware 100.64.0, hardware "Sensor-SM-V1"): same service and
characteristics, a new little-endian sip-record layout (below), plus Apple Find My (`fd44`) and a
bottle-settings characteristic `316c4914-…` (first two bytes = capacity in mL, little-endian: `6d02` = 621).

## Characteristics

| UUID | Use |
|---|---|
| `b44b03f0-b850-4090-86eb-72863fb3618d` | handshake writes ("set point") |
| `e3578b0d-caa7-46d6-b7c2-7331c08de044` | handshake writes **and** cap-state notifications |
| `016e11b1-6c8a-4074-9e5a-076053f93784` | sip records — "legacy" path (firmware 80.18) |
| `bf2d1ba1-c473-49f2-9571-0ce69036c642` | sip records — "modern" path (used when present) |
| `1807a063-4e2d-4636-981a-35e93d1c7b94` | weight, 2 bytes big-endian, ~every 2 s |
| `00002a19-…` / `00002a25-…` / `00002a26-…` | battery %, serial number, firmware revision |

Bottles advertise as `h2o…`; Hydro-dash also matches "HidrateSpark…" names and the
`45855422-…` / `bf2d1ba0-…` services.

## Session

1. **Sync:** a series of writes, 50 ms apart, that set today's total, the goal, the bottle's clock
   and its glow-reminder schedule (see *Glow, reminders and the sync* below). The bottle sends no
   sip records before it. V3.0 sent a fixed copy of 13 such writes taken from the community
   projects ("the handshake"), which set the clock to 15:18 and somebody else's schedule every
   time; `build_sync()` in `hydro_protocol.py` now sends the real values.
2. **Drain:** ask for a record on the sip characteristic; the bottle answers with one. Byte 0 is the
   number of records still queued, so repeat until it's 0 (see *Draining* below for the two ways of
   asking). If the same frame repeats 5 times, stop (the bottle isn't advancing; avoids a write loop).
3. Sips taken while connected arrive the same way (often a short "1 pending" frame first).

## Sip record layouts

Two layouts are documented for different firmware; Hydro-dash decodes each frame with both and keeps
the one with plausible values (they're mutually exclusive in practice — see the tests):

| Bytes | "epoch" (firmware 80.18 notes) | "percent" (newer integration) |
|---|---|---|
| 0 | records pending | records pending |
| 1 | flags | sip volume, % of capacity |
| 2–3 | total mL so far (BE) | total mL so far (BE) |
| 4–7 / 5–8 | Unix time, seconds (bytes 4–7) | seconds since the sip (bytes 5–8) |
| 8–9 | sip volume mL (BE) | — |

The layout used is recorded per sip (`Layout` column) and counted in the debug panel.

### PRO 2 layout (firmware 100.x, verified on real captures)

All little-endian. Checked against 15 recorded sips: totals add up record to record, each record's
"weight after" equals the next record's "weight before", and the live sip arrived as "1 s ago".

| Bytes | Meaning |
|---|---|
| 0 | records still queued |
| 1 | sip volume, **whole % of capacity**, rounded down (21 oz: 1 % ≈ 6.2 mL). Fallback only |
| 2–3 | today's running total, % of capacity (resets overnight) |
| 4–7 | **seconds since the sip** |
| 8–9 | **MIN**: the bottle's calibrated *empty* weight |
| 10–11 | **MAX**: the bottle's calibrated *full* weight |
| 12–13 / 14–15 | bottle weight before / after the sip |

The bottle carries its own calibration and sends it with every record. With
`frac(w) = clamp((w − MIN) / (MAX − MIN), 0, 1)`:

- **sip volume** = `(frac(before) − frac(after)) × capacity`
- **fill level** = `frac(weight) × capacity`

This is linear for every model after the V3 (the V1 / V2 / V3 bottles have a correction curve in
the app, not implemented here). On the test bottle MIN = 32784 and MAX = 34238, so one unit is
0.427 mL of its 621 mL. Byte 1 is the same sip rounded down to a whole percent, which reads up to
6 mL low; Hydro-dash uses the weights and falls back to byte 1 only when the record has no usable
MIN / MAX or the two disagree by more than 3 % of capacity.

`01 00 00 …` (records queued, no data) means "a record is coming"; just wait for it.

### Draining

Writes to the sip characteristic:

| Byte | Meaning |
|---|---|
| `0x55` | ready: send the next record |
| `0x33` | acknowledge the record just received (the bottle drops it from its buffer) |
| `0x57` | ready, fast mode: send the next record, no acknowledgement |

Which one depends on the firmware's major version, as in the official app:

| Firmware | Family | Draining |
|---|---|---|
| 100+ | Telink (PRO 2 and current bottles) | `0x33` after each record, then `0x55` |
| 30–99 | Nordic / Cypress / V3 | `0x57` per record |
| below 30 | early | `0x33`, then `0x55` |

Hydro-dash V3.0 used `0x57` for every bottle; on the PRO 2 that showed gaps in the pending count.
Either way only one request is ever in flight: the next goes out after the previous record has
arrived (or 2 s passed), there's a 1.5 s wait after the handshake in case the bottle starts on its
own, and the periodic re-check only runs when no drain is under way. If three acknowledged requests
in a row get no answer while records are waiting, Hydro-dash switches to `0x57` for that
connection. Skips are counted in the debug panel.

## Bottle size and calibration commands

- **Bottle size:** characteristic `316c4914-…`, capacity in mL, 16-bit little-endian (`6d02` = 621).
  Read on every connect.
- **Recalibrate** (debug characteristic `e3578b0d-…`): `0x41` = "the bottle is full now",
  `0x40` = "the bottle is empty now" (`0x66` on firmware below 50). The bottle stores the new MAX /
  MIN itself; they show up in the next sip record.
- **Never sent:** `0xF0` on the debug characteristic puts the bottle into firmware-update mode.

## Glow, reminders and the sync

Writes to the debug characteristic `e3578b0d-…`:

| Bytes | Meaning |
|---|---|
| `21 hi lo` | today's total so far, as % of capacity (big-endian) |
| `22 hi lo` | daily goal, as % of capacity (big-endian) |
| `B1` / `B0` | glow on each sip: on / off |

Writes to the set-point characteristic `b44b03f0-…`:

| Bytes | Meaning |
|---|---|
| `77 00 00 00` + 4-byte LE seconds | the bottle's clock: seconds since local midnight |
| `93 nn` / `92` | glow when the goal is reached: on (light pattern nn) / off |
| reminder slot (below) | one glow reminder |

A **reminder slot** means "at this time of day, glow if less than this much has been drunk":

| Bytes | Meaning |
|---|---|
| 0 | slot number (0–47 on firmware 100+, 0–11 on older bottles) |
| 1 | light pattern id |
| 2–3 | amount, % of capacity, little-endian; `FFFF` = glow regardless |
| 4–7 | time of day in seconds, little-endian; 0 = slot unused |
| 8 | (firmware 100+ only) glow on / off |
| 9 | (firmware 100+ only) sound on / off |

Hydro-dash fills one slot per reminder time and clears the rest. A reminder's amount is the share
of the goal that should be done by then: reminder *k* of *n* gets `goal × k / (n + 1)`.

**Glow now:** write one byte to the LED characteristic `a1d9a5bf-…`: `47` (71) on firmware 100+
plays the glow pattern stored on the bottle; older bottles take a light pattern id (e.g. `30`).

### Custom glow patterns (firmware 100+)

The bottle stores one glow pattern; "glow now" and the reminders play it. A pattern is a set of
**frame groups** (animations) and a **sequence** saying which group to play, how many extra times,
and with what pause. A **frame** sets every LED of the ring to a colour, holds for `ticksShown`
and fades to the next frame over `transitionTicks`. The small puck (hardware revision
`Sensor-SM-V1`, read from `2a27`) has 10 LEDs, the large one 13.

On the wire it's a protobuf message (all fields are integers; zero values are omitted, so a colour
channel of 0 is sent as 1):

```
Pattern    { 1: numSequences   2: repeated Sequence   3: numFrameGroups   4: repeated FrameGroup }
Sequence   { 1: frameGroupId   2: scale   3: repeats   4: delay }
FrameGroup { 1: id   2: numFrames   3: repeated Frame }
Frame      { 2: ticksShown   3: transitionTicks   4: repeated LEDColor }
LEDColor   { 1: r   2: g   3: b }
```

**Upload:** split the bytes into 20-byte packets (a 23-byte MTU minus the 3-byte header). Write
the number of packets (one byte, so at most 255 packets = 5100 bytes) to `3bbd83e1-…`, then each
packet in order to `3bbd83e2-…`.

Hydro-dash builds its own patterns (`glow_pattern()` in `hydro_protocol.py`): pulse, spin, flash,
solid and rainbow, from one or two colours blended round the ring. They are 200–1100 bytes.

Not implemented: the sound library, and the meaning of a tick in milliseconds (not measured).

## Cap, weight, refills, fill level

- **Cap:** bit 0 of byte 0 on the debug characteristic — `81…` open, `80…` closed.
- **Weight:** raw 16-bit value that rises as the bottle fills. A reading counts once 3 consecutive
  samples agree within ±2 (readings while the bottle moves don't settle).
- **Refill:** settled weight when the cap opens vs. the first settled weight within 30 s after it
  closes; a rise ≥ 25 raw units is a refill.
- **Fill level:** `frac(weight) × capacity` with the bottle's MIN / MAX. It's taken from the newest
  sip record's "weight after", and from the settled live weight whenever that reading lies on the
  calibration's scale (within 25 % of the MIN–MAX span outside either end). The last known MIN /
  MAX are remembered, so the level shows right after a reconnect.

## Constraints

- **One BLE central at a time** — the phone app and Hydro-dash can't both be connected.
- **Pair once with the official app** before first use.
- Cap and refill events are notify-only: events while out of range are lost (the resulting fill
  level is recovered on reconnect). Sips are buffered on the bottle and replayed.
