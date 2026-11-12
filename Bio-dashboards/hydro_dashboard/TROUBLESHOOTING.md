# HidrateSpark PRO 2 — how we got it working

A record of bringing up Hydro-dash on a real **HidrateSpark PRO 2** (21 oz, firmware 100.64.0,
hardware "Sensor-SM-V1"), on 2026-10-06, on a Framework 13 (Ubuntu, BlueZ, bleak 3.0.2).
Hydro-dash had been built from community documentation for the original PRO; this is everything
that went wrong between that and a working dashboard, and why.

Real Bluetooth addresses and the bottle's serial number are replaced with placeholders below.

**Quick fixes, if you're here because it isn't working:** see the [checklist](#checklist) at the end.


## The investigation at a glance

🟥 symptom · ⬜ (dashed) wrong guess · 🟪 tool that showed it · 🟧 real cause · 🟩 fix · ⬛ result

```mermaid
flowchart TB
    subgraph R1["1 · Not listed"]
        direction LR
        a1["Chimes, but<br/>not listed"]:::symptom -.-> w1["Phone app?<br/>(never connected)"]:::wrong -.-> b1["Advertises only a few s;<br/>unnamed devices hidden"]:::cause --> c1["Show unnamed;<br/>shake first"]:::fix
    end
    subgraph R2["2 · Finding it"]
        direction LR
        a2["Unnamed devices,<br/>rotating addresses"]:::symptom -.-> w2["Apple beacons?<br/>(neighbours')"]:::wrong -.-> b2["scan / probe"]:::tool --> c2["Found by name<br/>h2o0000xxxx"]:::fix
    end
    subgraph R3["3 · Connect fails instantly"]
        direction LR
        a3["'device not found'"]:::symptom --> b3["Stale handle<br/>from earlier scan"]:::cause --> c3["Fresh lookup<br/>each attempt"]:::fix
    end
    subgraph R4["4 · Still not found"]
        direction LR
        a4["'not found' right<br/>after seeing it"]:::symptom --> t4["BLEAK_LOGGING"]:::tool -.-> w4["Not connectable?<br/>(btmon: it is)"]:::wrong -.-> b4["Address rotates;<br/>BlueZ drops it when<br/>scanning stops"]:::cause --> c4["By name, connect<br/>while scanning"]:::fix
    end
    subgraph R5["5 · Sip format"]
        direction LR
        a5["UNKNOWN frames,<br/>wrong times"]:::symptom --> t5["listen capture"]:::tool --> b5["New PRO 2 layout<br/>(little-endian)"]:::cause --> c5["pro2 decoder<br/>+ tests"]:::fix
    end
    subgraph R6["6 · Lost sips"]
        direction LR
        a6["Queue jumped<br/>18→15, 8→5"]:::symptom --> b6["Two 0x57 requests<br/>in flight skip one"]:::cause --> c6["Drain one<br/>at a time"]:::fix
    end
    subgraph R7["7 · Crash on Connect"]
        direction LR
        a7["Callback needs<br/>2 parameters"]:::symptom --> c7["Fixed; scanner errors<br/>no longer fatal"]:::fix --> d7(["✅ Live"]):::done
    end
    R1 --> R2 --> R3 --> R4 --> R5 --> R6 --> R7
    classDef symptom fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef cause fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef fix fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef wrong fill:#F1F5F9,stroke:#94A3B8,color:#475569,stroke-width:2px,stroke-dasharray:6 4
    classDef tool fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef done fill:#1E293B,stroke:#0F172A,color:#F8FAFC,stroke-width:2px
    classDef q fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    style R1 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style R2 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style R3 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style R4 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style R5 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style R6 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style R7 fill:none,stroke:#CBD5E1,stroke-width:1.5px
```

## If it isn't working

```mermaid
flowchart LR
    Q1{"Listed?"}:::q -- yes --> Q2{"Connects?"}:::q -- yes --> Q3{"Handshake ok,<br/>sip char found?"}:::q -- yes --> Q4{"Undecoded<br/>frames?"}:::q -- no --> Q5{"Skipped<br/>records = 0?"}:::q -- yes --> Q6{"Volumes<br/>~30% off?"}:::q -- no --> OK(["✅ Good"]):::done
    Q1 -- no --> X1["Shake it right before<br/>scanning; tick Show all"]:::fix
    Q2 -- no --> X2["Shake while 'waiting';<br/>check last error"]:::fix
    Q3 -- no --> X3["hydro_test.py dump<br/>→ protocol check"]:::tool
    Q4 -- yes --> X4["New firmware:<br/>listen → send capture"]:::tool
    Q5 -- no --> X5["Sips lost:<br/>report with capture"]:::symptom
    Q6 -- yes --> X6["Capacity: 21 oz 621 mL<br/>30 oz 887 mL"]:::fix
    classDef symptom fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef cause fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef fix fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef wrong fill:#F1F5F9,stroke:#94A3B8,color:#475569,stroke-width:2px,stroke-dasharray:6 4
    classDef tool fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef done fill:#1E293B,stroke:#0F172A,color:#F8FAFC,stroke-width:2px
    classDef q fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
```

---

## Timeline

### 1. The bottle didn't appear in Hydro-dash

**Symptom:** the bottle chimed, but Hydro-dash's scanner never listed it.

**Causes (on our side — the bottle was never connected to a phone):**
- The bottle **only advertises for a few seconds after it's moved**, then goes quiet. Hydro-dash
  scanned in ~4 s windows, so it often missed that burst.
- The scanner **skipped every device without a name**. Unnamed advertisements weren't even shown
  under *Show all*, so there was nothing to spot.

**Fixes:**
- Unnamed devices are now listed under *Show all nearby devices*, with their service and
  manufacturer IDs.
- **Shake or tip the bottle right before / during a scan**, and keep it next to the computer.

### 2. Finding the bottle among dozens of devices

`hydro_test.py scan` showed six strong **unnamed devices that advertised nothing at all**, and a
few weak Apple (`004c`) beacons.

- **Red herring:** the weak Apple beacons (−92 to −100 dBm). A bottle next to the laptop reads
  −30 to −60 dBm; these were neighbours' AirTags / iPhones, not the bottle's Find My signal.
- The strong unnamed devices had **different addresses on every scan** (private, rotating
  addresses), so "scan, then dump the strongest" kept chasing moving targets.
- Added `hydro_test.py probe`: connects briefly to each strong unnamed device right after the scan
  and checks for HidrateSpark's services.

Woken right before the scan, the bottle then showed up properly under its name, **`h2o0000xxxx`**,
recognised as a HidrateSpark, at −57 dBm. (Its advertisement carries only the name, which is why it
looked like nothing at all whenever the name wasn't captured.)

### 3. "device … not found" — instantly

**Symptom:** clicking **Connect** failed three times within seconds:
`BleakError("device 'dev_XX_XX_XX_XX_XX_XX' not found")`.

**Cause (first layer):** the worker connected using the device handle from its *earlier* scan.
The bottle only advertises for a few seconds after it's moved, and BlueZ forgets devices it hasn't
seen recently, so the handle was stale.

**Fix:** look the bottle up fresh before every connection attempt. The page now shows *"waiting
for the bottle to advertise — shake or tip it"*, and Device → Troubleshooting shows the last connection error.

### 4. Still "not found" — and the address keeps changing

**Symptom:** the fresh lookup *found* the bottle, but the connection still failed with
`device … not found`.

**Discovery 1 — the address rotates on every wake-up.** Over one afternoon the bottle used at least
four different addresses. They're *static random* addresses that change each time it wakes (it's an
Apple Find My accessory). Connecting by address is a race you lose; **the name (`h2o0000xxxx`) is
the only stable handle**.

**Discovery 2 — `bluetoothctl` by hand is too slow.** `scan le` showed the bottle, but by the time
`connect` ran in a second terminal it said `Device … not available`, and the scan showed
`[DEL] Device … h2o0000xxxx`: BlueZ had already dropped it.

**Fix:** `hydro_test.py dump <name>` waits for the name to advertise and connects immediately.

### 5. The real cause: BlueZ drops the bottle when scanning stops

`BLEAK_LOGGING=1` gave the timing:

```
09.007  Discovering: False                          ← scan stops, bottle just seen
09.009  Connecting to device @ XX:XX:XX:XX:XX:XX
09.015  device 'dev_XX_…' not found                 ← 6 ms later, before any connection attempt
```

**Wrong turns:** early on we wrongly blamed the phone app for holding the connection (it never
was connected). Here, we first concluded the bottle was *non-connectable* (BlueZ drops non-connectable
devices fast). `sudo btmon` disproved it:

```
Legacy PDU Type: ADV_IND (0x0013)          ← connectable
Address: XX:XX:XX:XX:XX:XX (Static)
Name (complete): h2o0000xxxx               ← the advert carries only the name
```

So the bottle **is connectable**; BlueZ simply removes it the instant discovery stops — and bleak's
normal sequence is *find → stop scanning → connect*.

**Fix:** **connect while the scan is still running**, and stop scanning once the link is up.
First successful connection:

```
seen at XX:XX:XX:XX:XX:XX -- connecting with the scan still running
✅ Connected.
```

### 6. What the PRO 2 exposes

`hydro_test.py dump h2o0000xxxx` listed every service. Everything Hydro-dash needs is there:

| Needed | Found |
|---|---|
| Handshake `b44b03f0-…` and sip records `016e11b1-…` | ✅ in the HidrateSpark service `45855422-…` |
| Cap state `e3578b0d-…` | ✅ (moved to its own service `593f756e-…`, same characteristic) |
| Weight `1807a063-…` | ✅ plus a new `2007a063-…` next to it |
| Battery, firmware, serial, model | ✅ 100 %, 100.64.0, "HidrateSpark PRO 2" |

New on the PRO 2: Apple **Find My** (`fd44` service, `4f86000x-…` characteristics), a
**bottle-settings** characteristic `316c4914-…` (`6d02` = 621 = capacity in mL, little-endian),
and several vendor services (OTA, device info).

### 7. Sip records in an unknown format

**Symptom:** `hydro_test.py listen` drained the 19 sips stored on the bottle, but most showed
`⚠ sip frame in an UNKNOWN layout`, and the few that "decoded" had **wrong times**.

**Cause:** the PRO 2 uses a new, little-endian record layout. Decoded from the capture:

| Bytes | Meaning | How we checked |
|---|---|---|
| 0 | records still queued | counts down 19 → 1 |
| 1 | sip, % of capacity | 17 % → 106 mL; 3 % → 19 mL |
| 2–3 | today's running total, % | 102 → 109 = +7 ✓ each step; resets overnight |
| 4–7 | seconds since the sip | the live sip read "1 s ago" |
| 12–15 | weight before → after | each "after" = the next "before"; ≈ 0.38 mL per unit |

**Fix:** a `pro2` layout in `hydro_protocol.py`, with tests built from the captured frames.
Details in [PROTOCOL.md](PROTOCOL.md).

### 8. Skipped sip records

**Symptom:** the queued count jumped **18 → 15** and **8 → 5**, so four sips were lost.

**Cause:** each `0x57` drain request *acknowledges* the current record. Two requests in flight (the
bottle starting on its own after the handshake, plus our request, plus a retry fallback) skip one.

**Fix:** drain **strictly one request at a time** (the next only after a record arrives or 2 s pass),
wait 1.5 s after the handshake in case the bottle starts by itself, never re-check mid-queue, and
count any skips (*Skipped records* under Device → Troubleshooting).

### 9. The worker crashed on Connect

**Symptom:** `TypeError: callback must be callable with 2 parameters`.

**Cause:** bleak requires the scan callback to take exactly `(device, advertisement)`; ours had a
third, defaulted parameter. The scanner was also created outside the error handling, so the error
killed the whole worker.

**Fix:** a two-parameter callback; scanner errors now count as a failed attempt instead of crashing.
(The simulated scanner used in testing now enforces bleak's rule too.)

### 10. Working

Hydro-dash connected to the bottle by name, showed the serial and firmware, today's sips and total,
the last sip "just now", and battery 100 %. Remaining setup: fill-level calibration and checking the
bottle size (21 oz = 621 mL vs. the 30 oz tumbler = 887 mL; sips are a % of capacity).

### 11. A day later: "can't find it" again, and it was the laptop's radio

**Symptom:** after the bottle ran flat and was recharged, scans showed only unnamed devices, the
occasional sighting was followed by a connection timeout, and the system's own scanner caught the
name only now and then. A raw capture showed the bottle advertising normally (`ADV_IND`, its name,
a new address).

**Cause:** the laptop's built-in radio. It shares a chip and antenna with Wi-Fi and simply wasn't
listening often enough to catch the bottle's adverts. An old USB Bluetooth dongle picked the bottle
up almost instantly.

**Fix:** use a USB dongle for the bottle. Hydro-dash now prefers an external radio automatically
when one is plugged in (pin it under *Radio for this dashboard* under Device → Troubleshooting if you have
several); for the test script, `--adapter hci1`. `python3 hydro_test.py watch` prints one mark per
second showing when the bottle was actually heard, which makes a deaf radio obvious.

**Wrong turns:** suspecting the charge, the phone, a Find My-only mode, and the new V3.1 code. The
connection code hadn't changed.

---

### 12. Connected, but no services: an old dongle and big replies

The USB dongle from step 11 heard the bottle instantly, and then "connected" with an empty service
list: firmware unknown, 0 notifying characteristics. `btmon` showed why. The computer asks the
bottle for a larger message size (the bottle agrees to 200 bytes), then asks for its
characteristics. The bottle answers the small replies, and never sends the first large one
(about 190 bytes). 30 s later the computer gives up. The link itself stays healthy throughout.

The dongle was Bluetooth 4.0: 27 bytes per radio packet, so a large reply has to be cut into many
pieces, and this bottle doesn't manage that. A radio that carries long packets whole (the laptop's
built-in one, or a Bluetooth 5 dongle) doesn't hit it.

How it was pinned down:

- `gatttool -i hci0 -t random -b <addr> --characteristics` listed everything. It never asks for the
  larger size, so every reply is small.
- Capping the size system-wide made the dashboard's own connection work on the old dongle:
  in `/etc/bluetooth/main.conf` add `[GATT]` / `ExchangeMTU = 64`, then restart bluetooth.
  (Not 23: bluetoothd then fails to register any adapter.) This applies to **every** device on
  the machine, and the Polar H10 wants the larger size, so it's a diagnostic more than a fix.

Also learned: the bottle's address changes from time to time. Each change makes the computer read
the service list afresh, so a setup that "worked this morning" from a remembered list can stop.
Always find the bottle by name, never by a saved address.

Because of this, Hydro-dash no longer picks a dongle automatically. Choose the radio in the debug
menu: a Bluetooth 5 dongle is the best of both; an old one needs the cap above.

*Since V5.3:* the pin turned out to lock onto the wrong radio now and then, so Hydro-dash isn't
pinned by default any more: every radio listens and the one that hears the bottle best connects. A
radio through which the service list comes back empty (this problem) is passed over for a week,
remembered across restarts, so an old dongle is tried once and then left alone.

**Confirmed with a Bluetooth 5 dongle:** a TP-Link UB500 Plus (Realtek RTL8761BU, reports HCI 5.1)
on a Jetson Orin Nano connects to the bottle and streams sips with no `ExchangeMTU` line at all.
It worked out of the box on JetPack 6 (kernel 5.15): the kernel loads `rtl_bt/rtl8761bu_fw.bin`
itself and no `btusb new_id` is needed (it shares the plain UB500's USB ID, `2357:0604`). Check
yours with `hciconfig -a`: the dongle should show `Bus: USB`, `UP RUNNING` and `HCI Version: 5.x`.

---

### 13. Proving a radio works with the bottle (the UB500 Plus test night)

How the UB500 Plus was confirmed on the Orin, so the same checks can vet any new adapter. First
**stop Hydro-dash**: a connected bottle stops advertising, and every test below then reports
"wasn't seen".

```bash
systemctl --user stop biodash-hydro
hcitool con                                   # must list no connection to the bottle
# the UB500's hciN, found by its USB ID (2357:0604) since the number changes between boots
UB=$(for h in /sys/class/bluetooth/hci*; do [ "$(cat $h/device/../idVendor 2>/dev/null)" = 2357 ] && basename $h; done | head -1); echo $UB
```

For another adapter, use its USB vendor ID from `lsusb`, or read its hci number off the debug
panel's Bluetooth Radio list, where each radio shows its address.

Set `UB` in every new terminal: shell variables don't carry over, and an empty `--adapter` quietly
uses the default radio.

| Test | Command | Pass |
|---|---|---|
| Hears the bottle | `python3 hydro_test.py watch <name> --adapter $UB --seconds 45` (tip the bottle) | █ bursts after each tip (8 sightings in 25 s at −60 dBm on the UB500) |
| Ten cold connects (the section 12 bug) | `for i in $(seq 10); do python3 hydro_test.py dump <name> --adapter $UB 2>&1 \| grep -c "✅"; sleep 5; done` | **6** every time: "✅ Connected" plus the 5 protocol characteristics. The old bug gives 0 or 1 |
| Which radio the dashboard uses | `hcitool -i $UB con` while Hydro-dash is connected | lists the bottle |

**Wrong name, all zeros.** The tools match the bottle's name exactly: `h2o00001234` is not
`h2o000001234`. A run that never connects prints no ✅ at all, so a typo looks exactly like a dead
radio. The tools now list the bottle-like names they did see ("Seen instead: h2o00001234 <- did you
mean this one?"), and `tools/ble_scan.py` shows every name in range.

### 14. The Orin won't power on with a UB500 Plus and an ASUS BT400 plugged in

**Symptom:** nothing at all. No NVIDIA logo, no screen; the charger shows a few watts and stays
there. Unplug a dongle and it boots.

What was tried, one boot each:

| Plugged in at power-on | Result |
|---|---|
| UB500 Plus + 2 × ASUS BT400 | ❌ no boot |
| UB500 Plus + 1 × ASUS BT400 | ❌ no boot |
| 2 × ASUS BT400 | ✅ boots |
| UB500 Plus alone | ✅ boots |

- **It isn't power.** Two ASUS dongles boot fine, and with everything running all four dongles add
  about 1 W (7–8 W total) with no over-current messages in `dmesg`.
- **It happens before Linux.** No log is written for the failed boot (`journalctl -b -1` shows the
  boot before it) and the screen never lights, so it's the Jetson boot firmware's USB scan meeting
  that pair of devices. L4T 36.5.2, the newest JetPack 6 release, still does it.
- **Firmware files don't help.** The dongles' improved firmware (section 15, and `rtl8761bu_fw.bin`)
  is uploaded by Linux into the chip's RAM at every power-up, so the boot firmware always meets the
  original.

**Workarounds:** boot with the ASUS dongles unplugged and plug them in afterwards (everything
works once Linux is up, and the dashboards find their radios by address); or put them on a powered
hub with a switch, turned on after boot; or use UB500 Pluses throughout (test two together at
power-on first). Not yet tried: other USB ports, and a serial console on the button header (J14)
to see exactly where the firmware stops.

### 15. "BCM: firmware Patch file not found" (ASUS BT400 and other BCM20702 dongles)

```text
Bluetooth: hci2: BCM20702A1 (001.002.014) build 0000
Bluetooth: hci2: BCM: firmware Patch file not found, tried:
Bluetooth: hci2: BCM: 'brcm/BCM20702A1-0b05-17cb.hcd'
```

The dongle still works on the firmware built into the chip ("build 0000"), but that's the oldest,
buggiest version. The patch is in the community
[broadcom-bt-firmware](https://github.com/winterheart/broadcom-bt-firmware) collection, named after
the dongle's USB ID (`0b05:17cb` for the ASUS BT400):

```bash
cd /tmp
wget -q https://raw.githubusercontent.com/winterheart/broadcom-bt-firmware/master/brcm/BCM20702A1-0b05-17cb.hcd
echo "02204ae0958e7af3ffa6193713f5b3847d4ad37b9b9b5064b56cd96f2fdd18d1  BCM20702A1-0b05-17cb.hcd" | sha256sum -c
sudo mkdir -p /lib/firmware/brcm && sudo cp BCM20702A1-0b05-17cb.hcd /lib/firmware/brcm/
```

Unplug and replug the dongle (no reboot needed), then `sudo dmesg | grep -i "build 1467"`: one line
per patched dongle. It loads by itself from then on. Another dongle ID needs the file with that ID
in its name; the collection's `DEVICES.md` lists them.

## Checklist

| Symptom | Check |
|---|---|
| Bottle not listed | Shake or tip the bottle right before / during the scan (it only advertises briefly); keep it within ~1 m; tick *Show all nearby devices* |
| Bottle rarely seen, or seen but the connection times out | The radio isn't hearing it. Plug in a USB Bluetooth dongle and select it under Device → Troubleshooting; check with `python3 hydro_test.py watch --adapter hci1` |
| `device … not found` | Use the current Hydro-dash (it follows the bottle by name and connects while scanning) |
| Connects, fill level updates, but no sips since you pressed *It's empty now* | The bottle is waiting for the other half of its calibration: fill it, cap on, press **It's full now** (the banner at the top does the same). Sips drunk in between aren't recorded. |
| Connects, but no sips | Debug panel: *Sync after connecting* = complete, *Sip records via* found; *Undecoded sip frames* rising → new firmware layout, send `_raw.csv` |
| Sips missing | Debug panel: *Skipped records* should be 0 |
| Volumes look ~30 % off | Capacity setting vs. your bottle (21 oz 621 mL, 30 oz tumbler 887 mL) — in the HidrateSpark app *and* Hydro-dash |
| "wasn't seen" every time, but the scan lists the bottle | Hydro-dash still connected (stop it; `hcitool con` must be empty), or the name differs by a character (section 13) |
| Orin doesn't power on with dongles attached | Section 14: plug the ASUS dongles in after boot |
| `BCM: firmware Patch file not found` | Section 15 |
| Anything else | `python3 hydro_test.py listen h2o0000xxxx` prints every notification decoded and saves a capture |

## Tools that helped

| Tool | Used for |
|---|---|
| `hydro_test.py scan` / `probe` / `dump <name>` / `listen <name>` | finding, inspecting and capturing the bottle without the dashboard |
| `bluetoothctl --timeout 60 scan le \| grep -i h2o` | seeing the bottle appear (`[NEW]`) and be dropped (`[DEL]`) |
| `BLEAK_LOGGING=1 …` | the millisecond timing that showed BlueZ dropping the device |
| `sudo btmon \| grep -B16 h2o \| grep -iE "PDU\|Address\|Name"` | the advertisement type (connectable or not) and address type |
| Hydro-dash debug panel (D) | handshake, frames, skips, last error, every characteristic's last raw frame |
| `hydro_test.py watch <name> --adapter hciN` | whether a given radio hears the bottle's adverts, second by second |
| `tools/ble_scan.py --adapter hciN` | every device in range with its exact name and signal |
