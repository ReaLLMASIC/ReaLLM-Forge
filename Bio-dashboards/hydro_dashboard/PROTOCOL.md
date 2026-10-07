# HidrateSpark BLE protocol (as used by Hydro-dash)

Hydro-dash implements the bottle protocol in its own code (`hydro_protocol.py`). The protocol
**facts** below come from public community reverse-engineering; no code from those projects is
included (Bio-dash is Apache-2.0; one source is GPL-3.0).

**Sources**
- [HidrateSpark-MQTT-bridge](https://github.com/loryanstrant/HidrateSpark-MQTT-bridge) (MIT) —
  `docs/SYNC_AND_PROTOCOL.md`: characteristics, cap/weight signals, refill heuristic. Measured on
  firmware 80.18, nRF52832 (HidrateSpark Steel / PRO).
- [HA-Hidratespark](https://github.com/bditter/HA-Hidratespark) (MIT) — the "percent / seconds-ago"
  sip layout and stuck-record guard. Tested on a PRO (v1) 21 oz.
- [HydroSync](https://github.com/maxperron/HydroSync) (GPL-3.0) — original handshake and frame layout.

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

1. **Handshake:** 13 writes, 50 ms apart (see `HANDSHAKE` in `hydro_protocol.py`). The bottle sends
   no sip records before it.
2. **Drain:** write `0x57` to the sip characteristic; the bottle answers with one record. Byte 0 is
   the number of records still queued — write `0x57` again after each record until it's 0. If the
   same frame repeats 5 times, stop (the bottle isn't advancing; avoids a write loop).
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
| 1 | sip volume, **% of capacity** (21 oz: 1 % ≈ 6.2 mL) |
| 2–3 | today's running total, % of capacity (resets overnight) |
| 4–7 | **seconds since the sip** |
| 8–11 | constant within a session (purpose unknown) |
| 12–13 / 14–15 | bottle weight before / after the sip (≈ 0.38 mL per unit on a 21 oz) |

`01 00 00 …` (records queued, no data) means "a record is coming"; just wait for it.

### Draining one record at a time

Each `0x57` acknowledges the current record. Two requests in flight skip a record (a real PRO 2 session
showed the pending count jumping 18 → 15). Hydro-dash sends the next request only after the previous
record has arrived (or 2 s passed), waits 1.5 s after the handshake in case the bottle starts on its
own, and only re-checks for queued sips when no drain is under way. Skips are counted in the debug panel.

## Cap, weight, refills, fill level

- **Cap:** bit 0 of byte 0 on the debug characteristic — `81…` open, `80…` closed.
- **Weight:** raw 16-bit value that rises as the bottle fills. A reading counts once 3 consecutive
  samples agree within ±2 (readings while the bottle moves don't settle).
- **Refill:** settled weight when the cap opens vs. the first settled weight within 30 s after it
  closes; a rise ≥ 25 raw units (≈ 25 mL) is a refill. Without an "empty" anchor, the refill's
  weight is latched as "full".
- **Fill level:** linear between the full / empty anchors (or ≈ 1 mL per raw unit below "full");
  until calibrated, estimated by subtracting sips.

## Constraints

- **One BLE central at a time** — the phone app and Hydro-dash can't both be connected.
- **Pair once with the official app** before first use.
- Cap and refill events are notify-only: events while out of range are lost (the resulting fill
  level is recovered on reconnect). Sips are buffered on the bottle and replayed.
