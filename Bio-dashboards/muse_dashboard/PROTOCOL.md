# The Muse S Athena's Bluetooth protocol

Interaxon doesn't publish it. Everything here comes from
[OpenMuse](https://github.com/DominiqueMakowski/OpenMuse) (MIT, commit `a9be252`), which worked it out from
recordings and validated it. OpenMuse credits [amused-py](https://github.com/Amused-EEG/amused-py) (MIT) for
the communication and [AbosaSzakal's MuseAthenaDataformatParser](https://github.com/AbosaSzakal/MuseAthenaDataformatParser)
for the data format. `muse_protocol.py` implements it, and `test_muse_protocol.py` checks it against
OpenMuse's own decoder (`test_data/openmuse_reference.json`: random packets and what OpenMuse made of them).

## Characteristics

| UUID | What |
|---|---|
| `273e0001-4c4d-454d-96be-f03bac821358` | Control: commands in, answers out (notify) |
| `273e0013-4c4d-454d-96be-f03bac821358` | Data (notify) |
| `273e0014-4c4d-454d-96be-f03bac821358` | Data (notify) |

Both data characteristics carry the same packet format; which sensor arrives on which isn't fixed, so
the worker subscribes to both and decodes everything that arrives.

## Commands

A command is an ASCII token plus a newline, prefixed with its length: `p1041` is sent as
`06 70 31 30 34 31 0a`. Written without response.

To start (as OpenMuse does): `v6` (version), `s` (status), `h` (halt), the preset, `s`, `dc001` twice (start),
`L1` (low latency), `s`. `h` stops. The answers to `s` arrive on the control characteristic as JSON-like text
in pieces: `"fw"` firmware, `"bp"` battery %, `"hn"` the headband's name, and others.

## Presets

| Preset | EEG | Optics | Motion | Battery | Red LED |
|---|:-:|:-:|:-:|:-:|---|
| p20, p21, p50, p51, p60, p61 | 4 ch | — | ✓ | ✓ | off |
| p1034, p1043 | 8 ch | 8 ch | ✓ | ✓ | bright |
| p1044 | 8 ch | 8 ch | ✓ | ✓ | dim |
| p1035 | 4 ch | 4 ch | ✓ | ✓ | dim |
| **p1041** (default here), p1042 | 8 ch | 16 ch | ✓ | ✓ | bright |
| p1045 | 8 ch | 4 ch | ✓ | ✓ | dim |
| p1046 | 8 ch | 4 ch | ✓ | ✓ | — |
| p4129 | 8 ch | 4 ch | ✓ | ✓ | dim |

(OpenMuse's table, derived from the packets each preset produces; more may exist.)

## Packets

A notification holds one or more packets, back to back:

```
byte  0       packet length (header included)
      1       packet index (0-255, wraps)
      2-5     device clock, u32 little-endian, 256 kHz (wraps every ~4.7 h)
      6-8     ?
      9       the packet's sensor id (a tag, below)
      10-13   ?
      14-     the first block of that sensor (no header), then tagged blocks:
              [tag][block index][3 bytes ?][data] ... to the end of the packet
```

| Tag | Sensor | Channels × samples per block | Rate | Data bytes |
|---|---|---|---|---|
| 0x11 | EEG | 4 × 4 | 256 Hz | 28 |
| 0x12 | EEG | 8 × 2 | 256 Hz | 28 |
| 0x34 | Optics | 4 × 3 | 64 Hz | 30 |
| 0x35 | Optics | 8 × 2 | 64 Hz | 40 |
| 0x36 | Optics | 16 × 1 | 64 Hz | 40 |
| 0x47 | Accelerometer + gyroscope | 6 × 3 | 52 Hz | 36 |
| 0x53 | ? | | | 24 |
| 0x88 | Battery (newer firmware, 188-230 bytes) | 1 | every few s | 188+ |
| 0x98 | Battery | 1 | every few s | 20 |

An unknown tag ends its packet (its length isn't known).

## Values

- **EEG:** unsigned 14-bit, packed LSB first (bit *i* of byte *k* is bit 8*k*+*i* of the stream),
  samples × channels in order; × 1450 / 16383 → µV (0-1450, mid-scale ~725). Channels: TP9, AF7, AF8, TP10,
  then AUX1-4.
- **Optics:** unsigned 20-bit, packed the same way; / 32768. Channels (16): LO_NIR, RO_NIR, LO_IR, RO_IR,
  LI_NIR, RI_NIR, LI_IR, RI_IR, LO_RED, RO_RED, LO_AMB, RO_AMB, LI_RED, RI_RED, LI_AMB, RI_AMB
  (L / R left / right, O / I outer / inner optode, NIR 730 nm, IR 850 nm, RED, AMB ambient light). 8 channels:
  the first 8; 4 channels: LI_NIR, RI_NIR, LI_IR, RI_IR.
- **Motion:** 18 × int16 little-endian, 3 samples of [ax ay az gx gy gz]; acceleration × 0.0000610352 → g,
  rotation × −0.0074768 → °/s.
- **Battery:** the first 2 bytes, u16 little-endian / 256 → %.

## Times

Blocks in one packet share its clock value, so each stream counts its samples at the nominal rate and
re-syncs to the clock when it says samples are missing (counted as dropped). The device clock maps to the
computer's through one offset for all streams (set at the first packet, slewed 1 ms per packet when they
drift over 250 ms apart, reset after a jump of over 5 s), so EEG, optics and motion line up. The 32-bit
wrap is followed. See `MuseClock` in `muse_protocol.py`.
