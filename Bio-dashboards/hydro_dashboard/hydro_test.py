"""Hydro-dash bottle test: check how your HidrateSpark advertises and talks, without the dashboard.

    python3 hydro_test.py                    scan (+ probe if needed) -> dump its services -> listen 60 s
    python3 hydro_test.py scan               15 s scan: every device, strongest signal first
    python3 hydro_test.py probe              scan, then connect briefly to the strongest unnamed devices
                                             to find the one with HidrateSpark services
    python3 hydro_test.py dump <address|name>     services, characteristics, and every readable value
    python3 hydro_test.py listen <address|name>   handshake + drain, then print every notification live
(Use the bottle's name, e.g. h2o00003095 -- its address rotates every few minutes.)
Options: --seconds N (listen time, default 60) · --no-handshake · --adapter hciN

Stop Hydro-dash first (the bottle accepts one connection), and turn off Bluetooth on your phone.
The listen step saves a capture (time, characteristic, hex) to
Documents/Bio-dash/HidrateSpark/test-captures/ -- send it along if the bottle needs decoding.
"""
import argparse
import asyncio
import contextlib
import csv
import datetime
import os
import re
import sys
import time

from bleak import BleakClient, BleakScanner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import bt_debug              # noqa: E402
    import hydro_protocol as hp  # noqa: E402
except ImportError:
    sys.exit("hydro_test.py needs to run from the hydro_dashboard folder (it uses hydro_protocol.py and\n"
             "bt_debug.py next to it). From the Bio-dash release folder:\n"
             "    cd hydro_dashboard && python3 hydro_test.py")

CAPACITY_ML = 621          # only used to show sip volumes for the "percent" frame layout


MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$")


async def resolve(target, adapter=None, timeout=25.0):
    """An address, or a name like h2o00003095: wait for it to advertise and return the device
    right away. Bottles rotate their address every few minutes, so the name is the stable handle,
    and connecting the moment it's seen beats BlueZ forgetting it again."""
    if MAC_RE.match(target):
        return target
    print(f"🔎 Waiting up to {timeout:.0f} s for '{target}' to advertise -- shake or tip the bottle now...")
    want = target.lower()
    dev = await BleakScanner.find_device_by_filter(
        lambda d, adv: (d.name or adv.local_name or "").lower() == want, timeout=timeout, **bluez(adapter))
    if dev is None:
        sys.exit(f"❌ '{target}' wasn't seen. Shake the bottle and keep it next to the computer; phone Bluetooth off.")
    print(f"   seen at {dev.address} -- connecting immediately")
    return dev


@contextlib.asynccontextmanager
async def open_bottle(target, adapter=None, timeout=25.0):
    """Connect to an address or a name (e.g. h2o00003095) WHILE STILL SCANNING.
    BlueZ drops this bottle the instant discovery stops, so the usual
    find -> stop scan -> connect fails with "device ... not found"."""
    by_mac = bool(MAC_RE.match(target))
    want = target.lower()
    found, box = asyncio.Event(), {}

    def on_adv(d, adv):
        name = (d.name or adv.local_name or "").lower()
        if not found.is_set() and ((by_mac and d.address.lower() == want) or (not by_mac and name == want)):
            box["dev"] = d
            found.set()

    print(f"🔎 Waiting up to {timeout:.0f} s for '{target}' -- shake or tip the bottle now...")
    scanner = BleakScanner(on_adv, **bluez(adapter))
    await scanner.start()
    try:
        await asyncio.wait_for(found.wait(), timeout)
    except asyncio.TimeoutError:
        await scanner.stop()
        sys.exit(f"❌ '{target}' wasn't seen. Shake the bottle and keep it next to the computer; phone Bluetooth off.")
    dev = box["dev"]
    print(f"   seen at {dev.address} -- connecting with the scan still running")
    client = BleakClient(dev, timeout=20.0, **bluez(adapter))
    try:
        await client.connect()
    except Exception as e:
        with contextlib.suppress(Exception):
            await scanner.stop()
        sys.exit(f"❌ Connection refused / failed: {e.__class__.__name__}: {e}")
    with contextlib.suppress(Exception):
        await scanner.stop()
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            await client.disconnect()


def bluez(adapter):
    return {"bluez": {"adapter": adapter}} if adapter else {}


# ---------------------------------------------------------------------------
async def scan(seconds=15, adapter=None):
    print(f"🔎 Scanning {seconds} s{' on ' + adapter if adapter else ''} -- shake or tip the bottle to wake it, "
          "keep it next to the computer, phone Bluetooth off.\n")
    seen = {}

    def on_adv(device, adv):
        name = device.name or adv.local_name or ""
        seen[device.address] = {
            "address": device.address, "name": name, "rssi": adv.rssi or -100,
            "services": [u.lower() for u in (adv.service_uuids or [])],
            "mfr": sorted(f"{k:04x}" for k in (adv.manufacturer_data or {})),
            "bottle": hp.is_bottle(name, adv.service_uuids),
        }

    try:
        async with BleakScanner(on_adv, **bluez(adapter)):
            await asyncio.sleep(seconds)
    except Exception as e:
        sys.exit(f"❌ Scan failed: {e!r}\n   Is Bluetooth on? (bluetoothctl show) Is Hydro-dash still running and scanning?")

    found = sorted(seen.values(), key=lambda d: (not d["bottle"], -d["rssi"]))
    print(f"{'':2}{'RSSI':>5}  {'address':17}  {'name':22}  services / manufacturer")
    for d in found[:20]:
        mark = "💧" if d["bottle"] else "  "
        extra = " ".join(u[:8] for u in d["services"][:3])
        if d["mfr"]:
            extra += ("  " if extra else "") + "mfr " + " ".join(d["mfr"]) + (" (Apple / Find My?)" if "004c" in d["mfr"] else "")
        print(f"{mark}{d['rssi']:>5}  {d['address']:17}  {(d['name'] or '(no name)')[:22]:22}  {extra}")
    if len(found) > 20:
        print(f"   … and {len(found) - 20} weaker devices")
    bottles = [d for d in found if d["bottle"]]
    print()
    if bottles:
        print(f"💧 Looks like a HidrateSpark: {bottles[0]['name'] or '(no name)'} [{bottles[0]['address']}] at {bottles[0]['rssi']} dBm")
        return bottles[0]["address"], found
    print("No device advertised HidrateSpark's name or service.")
    return None, found


async def probe(found, adapter=None, max_devices=6, min_rssi=-80):
    """Connect briefly to the strongest unnamed devices and look inside for HidrateSpark's services.
    Unnamed, silent devices use rotating private addresses, so this runs right after the scan."""
    cands = [d for d in found if not d["name"] and d["rssi"] >= min_rssi and "004c" not in d["mfr"]][:max_devices]
    if not cands:
        print("No strong unnamed devices to probe -- is the bottle awake and next to the computer?")
        return None
    print(f"\n🔬 Probing the {len(cands)} strongest unnamed devices for HidrateSpark services "
          "(a quick read-only connection each; shake the bottle again now)...")
    for d in cands:
        try:
            async with BleakClient(d["address"], timeout=8.0, **bluez(adapter)) as client:
                svcs = [svc.uuid.lower() for svc in client.services]
                chars = {ch.uuid.lower() for svc in client.services for ch in svc.characteristics}
            hit = any(u in hp.SERVICE_HINTS for u in svcs) or hp.CHAR_SET_POINT in chars or hp.CHAR_USER_DATA in chars
            shown = " ".join(u[4:8] if u.endswith("-0000-1000-8000-00805f9b34fb") else u[:8] for u in svcs[:6])
            print(f"  {'💧' if hit else '✗ '} {d['address']}  {d['rssi']:>4} dBm  services: {shown or '(none)'}")
            if hit:
                print(f"\n💧 Found the bottle: {d['address']}")
                return d["address"]
        except Exception as e:
            print(f"  ·  {d['address']}  {d['rssi']:>4} dBm  couldn't connect ({e.__class__.__name__})")
    print("\nNone of them is a HidrateSpark. Things to check:\n"
          "  • phone Bluetooth fully OFF (a connected bottle stops advertising)\n"
          "  • shake the bottle right before / during the scan (it sleeps when idle)\n"
          "  • if it's registered in Apple Find My, it may only show up as an Apple (004c) beacon")
    return None


# ---------------------------------------------------------------------------
async def dump(address, adapter=None):
    async with open_bottle(address, adapter) as client:
        print("✅ Connected.\n\n=== SERVICES & CHARACTERISTICS ===")
        for svc in client.services:
            print(f"\n[Service] {svc.uuid}  {svc.description or ''}")
            for ch in svc.characteristics:
                known = hp.KNOWN_CHARS.get(ch.uuid.lower())
                line = f"  └─ {ch.uuid}  [{', '.join(ch.properties)}]" + (f"  ← {known}" if known else "")
                if "read" in ch.properties:
                    try:
                        val = bytes(await client.read_gatt_char(ch.uuid))
                        text = val.decode("utf-8", "replace").strip("\x00 ")
                        printable = text if len(text) >= 3 and all(c.isprintable() for c in text) else ""
                        if ch.uuid.lower() == hp.CHAR_BATTERY and val:
                            printable = f"{val[0]} %"
                        line += f"\n       value: {val.hex() or '(empty)'}" + (f"  ({printable})" if printable else "")
                    except Exception as e:
                        line += f"\n       value: (read failed: {e.__class__.__name__})"
                print(line)
        print("\n==================================")
        expected = [hp.CHAR_SET_POINT, hp.CHAR_DEBUG, hp.CHAR_DATA_POINT, hp.CHAR_USER_DATA, hp.CHAR_WEIGHT]
        present = {ch.uuid.lower() for svc in client.services for ch in svc.characteristics}
        print("\nProtocol check (characteristics the documented protocol uses):")
        for uuid in expected:
            print(f"  {'✅' if uuid in present else '❌'} {hp.KNOWN_CHARS.get(uuid, uuid):24s} {uuid}")


# ---------------------------------------------------------------------------
def describe(uuid, data):
    """One-line decoding for known characteristics."""
    if uuid in (hp.CHAR_DATA_POINT, hp.CHAR_USER_DATA):
        r = hp.parse_sip_frame(data, CAPACITY_ML)
        if r["kind"] == "sip":
            when = datetime.datetime.fromtimestamp(r["ts"]).strftime("%a %H:%M:%S")
            w = f", weight {r['weight_before']}→{r['weight_after']}" if r.get("weight_before") else ""
            return f"💧 SIP {r['volume_ml']} mL ({r['pct']}%) at {when}{w} [{r['layout']}, {r['remaining'] - 1} more queued]"
        return {"empty": "sip queue empty", "pending": f"{r['remaining']} sip record(s) pending",
                "unknown": "⚠ sip frame in an UNKNOWN layout"}[r["kind"]]
    if uuid == hp.CHAR_DEBUG:
        c = hp.parse_cap(data)
        return "cap OPEN" if c else "cap CLOSED" if c is False else ""
    if uuid == hp.CHAR_WEIGHT:
        w = hp.parse_weight(data)
        return f"weight {w}" if w is not None else ""
    if uuid == hp.CHAR_BATTERY:
        b = hp.parse_battery(data)
        return f"battery {b} %" if b is not None else ""
    return ""


async def listen(address, seconds=60, handshake=True, adapter=None):
    folder = os.path.join(bt_debug.dashboard_dir("HidrateSpark"), "test-captures")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, datetime.datetime.now().strftime("capture_%Y-%m-%d_%H-%M-%S.csv"))
    out = open(path, "w", newline="")
    w = csv.writer(out)
    w.writerow(["Timestamp_Epoch_ms", "Characteristic", "Hex", "Decoded"])
    counts = {}
    t0 = time.time()

    async with open_bottle(address, adapter) as client:
        chars = {ch.uuid.lower(): ch for svc in client.services for ch in svc.characteristics}
        sip_char = hp.CHAR_USER_DATA if hp.CHAR_USER_DATA in chars else hp.CHAR_DATA_POINT if hp.CHAR_DATA_POINT in chars else None
        repeat = {"last": None, "n": 0}

        async def write_drain():
            await client.write_gatt_char(sip_char, hp.DRAIN, response=True)
        drainer = hp.Drainer(write_drain)
        last_rem = {"n": 0}

        async def drain():
            await drainer.request()

        def on_notify(uuid, data):
            data = bytes(data)
            text = describe(uuid, data)
            counts[uuid] = counts.get(uuid, 0) + 1
            w.writerow([int(time.time() * 1000), uuid, data.hex(), text]); out.flush()
            if uuid != hp.CHAR_WEIGHT or counts[uuid] % 5 == 1:          # weight arrives ~every 2 s: show 1 in 5
                label = hp.KNOWN_CHARS.get(uuid, uuid[:8] + "…")
                print(f"  {time.time() - t0:6.1f}s  {label:22s} {data.hex():42s} {text}")
            if uuid == sip_char:
                drainer.on_frame()
                if last_rem["n"] > 1 and data and 0 < data[0] < last_rem["n"] - 1 and any(data[1:]):
                    skipped = last_rem["n"] - 1 - data[0]
                    counts["skipped"] = counts.get("skipped", 0) + skipped
                    print(f"  ⚠ {skipped} record(s) skipped")
                if data and (data[0] == 0 or any(data[1:])):
                    last_rem["n"] = data[0]
            if uuid == sip_char and data and data[0] > 0:                 # records pending: ask for the next
                repeat["n"] = repeat["n"] + 1 if data.hex() == repeat["last"] else 0
                repeat["last"] = data.hex()
                if repeat["n"] < hp.MAX_IDENTICAL_FRAMES:
                    asyncio.get_running_loop().create_task(drain())

        n_sub = 0
        for uuid, ch in chars.items():
            if {"notify", "indicate"} & set(ch.properties):
                try:
                    await client.start_notify(uuid, lambda c, d, u=uuid: on_notify(u, d)); n_sub += 1
                except Exception as e:
                    print(f"  (couldn't subscribe to {uuid}: {e.__class__.__name__})")
        print(f"✅ Connected; listening to {n_sub} notifying characteristics.")

        if handshake and hp.CHAR_SET_POINT in chars and hp.CHAR_DEBUG in chars:
            try:
                for uuid, payload in hp.HANDSHAKE:
                    await client.write_gatt_char(uuid, bytes.fromhex(payload), response=True)
                    await asyncio.sleep(hp.HANDSHAKE_INTERVAL_S)
                print("✅ Handshake sent (13 writes).")
            except Exception as e:
                print(f"❌ Handshake failed: {e!r}")
        elif handshake:
            print("⚠ Handshake characteristics not found -- this firmware may use a different protocol.")
        if sip_char:
            await asyncio.sleep(1.5)            # the bottle may start sending by itself after the handshake
            if drainer.last_frame_at == 0:
                await drain()
            print(f"↺ Draining buffered sips on {hp.KNOWN_CHARS[sip_char]} (one at a time).")
        else:
            print("⚠ No sip-record characteristic found.")

        print(f"\nListening {seconds} s -- now: take a sip, then open and close the cap, then refill if you can.\n")
        end = time.time() + seconds
        while time.time() < end and client.is_connected:
            await asyncio.sleep(0.5)
        if not client.is_connected:
            print("\n(the bottle disconnected -- it sleeps when idle)")
    out.close()

    print("\n=== SUMMARY ===")
    if counts.get("skipped"):
        print(f"  ⚠ {counts.pop('skipped')} record(s) skipped while draining")
    for uuid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {hp.KNOWN_CHARS.get(uuid, 'unknown'):22s} {uuid}")
    if not counts:
        print("  no notifications at all -- the bottle may need the official app's pairing first, or uses a different protocol")
    print(f"\nCapture saved: {path}")


async def main():
    ap = argparse.ArgumentParser(description="HidrateSpark bottle test (scan / dump / listen).")
    ap.add_argument("command", nargs="?", default="all", choices=["all", "scan", "probe", "dump", "listen"])
    ap.add_argument("address", nargs="?")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--no-handshake", action="store_true")
    ap.add_argument("--adapter", help="Bluetooth adapter, e.g. hci1 (default: system default)")
    a = ap.parse_args()
    if a.command in ("dump", "listen") and not a.address:
        sys.exit(f"usage: python3 hydro_test.py {a.command} <address or name, e.g. h2o00003095>")
    if a.command == "scan":
        await scan(adapter=a.adapter); return
    if a.command == "probe":
        _, found = await scan(adapter=a.adapter)
        await probe(found, a.adapter); return
    if a.command == "dump":
        await dump(a.address, a.adapter); return
    if a.command == "listen":
        await listen(a.address, a.seconds, not a.no_handshake, a.adapter); return
    address = a.address
    if not address:
        address, found = await scan(adapter=a.adapter)
        if not address:
            address = await probe(found, a.adapter)
    if not address:
        return
    await dump(address, a.adapter)
    await listen(address, a.seconds, not a.no_handshake, a.adapter)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
