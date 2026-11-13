#!/usr/bin/env python3
"""Standalone Bluetooth LE scanner: what's around, and what each device is advertising.

Lists every device the radio hears with its name, signal, address type and a plain reading of
its advertisement: Apple Find My (AirTags, Find My-enabled accessories, iPhones' offline-finding
beacons), other Apple Continuity messages, Google Fast Pair, Eddystone, iBeacon, heart-rate
straps, and the maker of anything else (from its manufacturer ID).

    python3 tools/ble_scan.py                        30 s, every device, a live table
    python3 tools/ble_scan.py --seconds 60 --adapter hci1
    python3 tools/ble_scan.py --find-my              only Apple Find My traffic
    python3 tools/ble_scan.py --name h2o             only names containing "h2o"
    python3 tools/ble_scan.py --services h2o00001234 connect and list its GATT services
    python3 tools/ble_scan.py --log scan.csv         also save every advertisement heard

Address types: "public" is fixed by the maker. "random static" stays the same until the device
reboots (or, for Find My accessories, until it rotates on purpose). "random private" changes
every few minutes so the device can't be followed; that's what phones, AirTags and most
Find My traffic use. A device that keeps changing address can only be found again by name.

Only bleak is needed (pip install bleak), the same as the dashboards. Linux / BlueZ, macOS
and Windows all work for scanning; the address type is only known on Linux.
"""
import argparse
import asyncio
import csv
import datetime as dt
import os
import sys
import time

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    sys.exit("This needs bleak: pip install bleak  (or run it with the Bio-dash .venv: ../.venv/bin/python3)")

# Bluetooth SIG company IDs (the first two bytes of manufacturer data)
COMPANIES = {
    0x004C: "Apple", 0x0006: "Microsoft", 0x0075: "Samsung", 0x00E0: "Google", 0x0059: "Nordic Semiconductor",
    0x02E5: "Espressif", 0x006B: "Polar", 0x0087: "Garmin", 0x038F: "Xiaomi", 0x012D: "Sony",
    0x000D: "Texas Instruments", 0x000F: "Broadcom", 0x0131: "Cypress", 0x005D: "Realtek",
}
# 16-bit service UUIDs worth naming (advertised in the service list or as service data)
SERVICES = {
    0xFD44: "Apple Find My network (accessory)", 0xFD6F: "Exposure Notification", 0xFE2C: "Google Fast Pair",
    0xFEAA: "Eddystone beacon", 0xFE9F: "Google", 0xFCF1: "Google", 0xFE03: "Amazon",
    0x180D: "Heart Rate", 0x180F: "Battery", 0x180A: "Device Information", 0x1809: "Health Thermometer",
    0x1810: "Blood Pressure", 0x1816: "Cycling Speed and Cadence", 0x1818: "Cycling Power",
    0x181A: "Environmental Sensing", 0x1822: "Pulse Oximeter", 0xFEED: "Tile", 0xFEEC: "Tile",
}
# Apple Continuity message types (first byte after Apple's company ID)
APPLE_TYPES = {
    0x02: "iBeacon", 0x05: "AirDrop", 0x07: "AirPods / Beats", 0x09: "AirPlay target", 0x0A: "AirPlay source",
    0x0C: "Handoff", 0x0D: "Hotspot", 0x0E: "Hotspot", 0x0F: "Nearby Action", 0x10: "Nearby Info (iPhone, Mac…)",
    0x12: "Find My (offline-finding beacon)", 0x16: "Nearby Info",
}


def uuid16(u):
    """'0000fd44-0000-1000-8000-00805f9b34fb' -> 0xFD44 (or None for a full 128-bit UUID)."""
    u = str(u).lower()
    if u.endswith("-0000-1000-8000-00805f9b34fb") and u.startswith("0000"):
        return int(u[4:8], 16)
    return None


def address_kind(device):
    """public / random static / random private / random (non-resolvable); '?' where the OS doesn't say (macOS, Windows)."""
    details = getattr(device, "details", None)
    props = details.get("props", {}) if isinstance(details, dict) else {}
    kind = props.get("AddressType")              # BlueZ: "public" or "random"
    if kind == "public":
        return "public"
    if kind != "random":
        return "?"
    try:
        top = int(str(device.address).split(":")[0], 16) >> 6
    except ValueError:
        return "random"
    return {3: "random static", 1: "random private", 0: "random (non-resolvable)"}.get(top, "random")


def describe(adv):
    """Plain-language list of what an advertisement says."""
    out = []
    for cid, data in (adv.manufacturer_data or {}).items():
        if cid == 0x004C:
            i = 0
            while i + 1 < len(data):                 # Apple packs several [type, length, payload] items
                t, n = data[i], data[i + 1]
                if t == 0x12:
                    out.append("Apple Find My: offline-finding beacon")
                elif t == 0x02 and n >= 21 and i + 2 + 21 <= len(data):
                    p = data[i + 2:i + 2 + 21]
                    out.append(f"iBeacon {p[:16].hex()} major {int.from_bytes(p[16:18], 'big')} minor {int.from_bytes(p[18:20], 'big')}")
                else:
                    out.append(f"Apple {APPLE_TYPES.get(t, f'type 0x{t:02x}')}")
                i += 2 + n
            if not data:
                out.append("Apple")
        else:
            out.append(f"maker: {COMPANIES.get(cid, f'0x{cid:04x}')}")
    for u in list(adv.service_uuids or []) + list((adv.service_data or {}).keys()):
        s = uuid16(u)
        if s is not None and s in SERVICES:
            label = SERVICES[s]
            if s == 0xFD44:
                label = "Apple Find My network accessory (0xFD44)"
            if label not in out:
                out.append(label)
        elif s is not None:
            out.append(f"service 0x{s:04x}")
    seen, uniq = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    return uniq


def is_find_my(desc):
    return any("Find My" in d for d in desc)


class Table:
    def __init__(self, args):
        self.a = args
        self.dev = {}                                 # address -> dict
        self.started = time.time()
        self.log = None
        if args.log:
            self.log_f = open(args.log, "w", newline="")
            self.log = csv.writer(self.log_f)
            self.log.writerow(["time", "address", "address_type", "name", "rssi", "manufacturer_data", "service_uuids", "service_data", "meaning"])

    def heard(self, device, adv):
        desc = describe(adv)
        name = adv.local_name or device.name or ""
        if self.a.name and self.a.name.lower() not in name.lower():
            return
        if self.a.find_my and not is_find_my(desc):
            return
        d = self.dev.setdefault(device.address, {"first": time.time(), "count": 0, "name": "", "rssi": None,
                                                 "kind": address_kind(device), "desc": [], "best": -999})
        d["count"] += 1
        d["last"] = time.time()
        d["name"] = name or d["name"]
        d["rssi"] = adv.rssi
        d["best"] = max(d["best"], adv.rssi if adv.rssi is not None else -999)
        for x in desc:
            if x not in d["desc"]:
                d["desc"].append(x)
        if self.log:
            self.log.writerow([dt.datetime.now().isoformat(timespec="milliseconds"), device.address, d["kind"], name, adv.rssi,
                               ";".join(f"{k:04x}:{v.hex()}" for k, v in (adv.manufacturer_data or {}).items()),
                               ";".join(adv.service_uuids or []),
                               ";".join(f"{k}:{v.hex()}" for k, v in (adv.service_data or {}).items()), " | ".join(desc)])

    def render(self, final=False):
        rows = sorted(self.dev.items(), key=lambda kv: (-(kv[1]["best"])))
        now = time.time()
        lines = [f"Bluetooth LE scan · {now - self.started:4.0f} s · {len(rows)} device(s)"
                 + (f" · {sum(1 for _, d in rows if is_find_my(d['desc']))} Find My" if rows else "")
                 + ("" if final else "   (Ctrl+C to stop)"), ""]
        lines.append(f"{'address':19s} {'type':15s} {'rssi':>5s} {'seen':>5s}  {'name':22s} what it is")
        for addr, d in rows[: self.a.rows]:
            age = now - d.get("last", now)
            rssi = f"{d['rssi']}" if d["rssi"] is not None else "?"
            what = ", ".join(d["desc"]) or "—"
            stale = "" if age < 10 else f"  (last {age:.0f} s ago)"
            lines.append(f"{addr:19s} {d['kind']:15s} {rssi:>5s} {d['count']:>5d}  {d['name'][:22]:22s} {what}{stale}")
        if len(rows) > self.a.rows:
            lines.append(f"… and {len(rows) - self.a.rows} more (--rows to show more)")
        return "\n".join(lines)


async def scan(a):
    t = Table(a)
    scanner = BleakScanner(detection_callback=t.heard, bluez={"adapter": a.adapter}) if a.adapter else BleakScanner(detection_callback=t.heard)
    live = sys.stdout.isatty() and not a.plain
    await scanner.start()
    try:
        end = time.time() + a.seconds
        while time.time() < end:
            await asyncio.sleep(1.0)
            if live:
                sys.stdout.write("\x1b[H\x1b[2J" + t.render() + "\n")
                sys.stdout.flush()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await scanner.stop()
        if t.log:
            t.log_f.close()
    if live:
        sys.stdout.write("\x1b[H\x1b[2J")
    print(t.render(final=True))
    fm = [d for d in t.dev.values() if is_find_my(d["desc"])]
    if fm:
        print(f"\n{len(fm)} device(s) sent Apple Find My traffic. Most use random private addresses that change every few"
              " minutes, so one AirTag or iPhone can show up several times.")
    if a.log:
        print(f"\nEvery advertisement saved to {a.log}")


async def services(a):
    target = a.services
    print(f"Looking for '{target}' (wake it if it only advertises when moved)…")
    dev = None
    if ":" in target and len(target) == 17:
        dev = await BleakScanner.find_device_by_address(target, timeout=a.seconds, bluez={"adapter": a.adapter}) if a.adapter \
            else await BleakScanner.find_device_by_address(target, timeout=a.seconds)
    else:
        f = lambda d, ad: target.lower() in ((ad.local_name or d.name or "").lower())
        dev = await BleakScanner.find_device_by_filter(f, timeout=a.seconds, bluez={"adapter": a.adapter}) if a.adapter \
            else await BleakScanner.find_device_by_filter(f, timeout=a.seconds)
    if not dev:
        sys.exit("Not heard. Check the name with a plain scan first.")
    print(f"Found {dev.name or '?'} at {dev.address}; connecting…")
    async with BleakClient(dev, timeout=30.0) as c:
        found_fm = False
        for svc in c.services:
            s16 = uuid16(svc.uuid)
            label = SERVICES.get(s16, "") if s16 is not None else ""
            if s16 == 0xFD44:
                found_fm = True
            print(f"  service {svc.uuid}  {label or svc.description}")
            for ch in svc.characteristics:
                print(f"      {ch.uuid}  [{','.join(ch.properties)}]  {ch.description}")
        if not list(c.services):
            print("  (empty service list: on an old Bluetooth 4.0 dongle, see Hydro-dash TROUBLESHOOTING section 12)")
        if found_fm:
            print("\nThis device carries Apple's Find My network accessory service (0xFD44): it can be located through"
                  " nearby Apple devices once paired to an Apple ID, which is also why its address keeps changing.")


def main():
    ap = argparse.ArgumentParser(description="Standalone Bluetooth LE scanner (plain-language advertisements, Apple Find My included).",
                                 epilog="More: the notes at the top of this file.")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--adapter", help="e.g. hci0 (Linux); default: the system's default radio")
    ap.add_argument("--name", help="only devices whose name contains this")
    ap.add_argument("--find-my", action="store_true", help="only Apple Find My traffic")
    ap.add_argument("--services", metavar="NAME_OR_ADDRESS", help="connect and list this device's GATT services")
    ap.add_argument("--log", metavar="FILE.csv", help="save every advertisement heard")
    ap.add_argument("--rows", type=int, default=40, help="rows in the table (default 40)")
    ap.add_argument("--plain", action="store_true", help="no live refresh; print the table once at the end")
    a = ap.parse_args()
    try:
        asyncio.run(services(a) if a.services else scan(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
