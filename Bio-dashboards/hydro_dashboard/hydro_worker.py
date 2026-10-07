"""Hydro-dash worker: finds the HidrateSpark bottle, runs the handshake, drains buffered
sips, follows weight / cap / battery, and records everything.

Recordings: <Documents>/Bio-dash/HidrateSpark/<bottle>/<date>/<time>_{sips,status,events,raw}.csv
  sips    one row per sip (live or replayed from the bottle's buffer, de-duplicated)
  status  fill level / weight / battery / cap every 5 s
  events  connect, cap open/close, refill, calibration
  raw     every notification from every characteristic (hex) -- for checking or decoding
          new firmware (e.g. the PRO 2)
"""
import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from importlib.metadata import version as _pkg_version

from bleak import BleakClient, BleakScanner

import bt_debug
import hydro_protocol as hp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_hydro")       # runtime state only; recordings go to Documents
os.makedirs(LOGS_DIR, exist_ok=True)
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "hydro_settings.json")

radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"), prefer="internal")
last_device = bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))

CONNECT_ATTEMPTS = 3
LOOKUP_TIMEOUT_S = 15.0      # per attempt: how long to wait for the bottle to advertise
STATUS_ROW_EVERY_S = 5
REDRAIN_EVERY_S = 30          # ask for buffered sips periodically, in case live ones only queue
DEFAULT_SETTINGS = {"capacity_ml": 621, "goal_ml": 2500, "units": "ml",
                    "weight_full_raw": None, "weight_empty_raw": None, "weight_scale_ml": 1.0}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

SESSION_STREAMS = {
    "sips":   ["Timestamp_Epoch_ms", "Volume_ml", "Pct", "Total_Reported", "Source", "Layout", "Raw",
               "Weight_before", "Weight_after"],
    "status": ["Timestamp_Epoch_ms", "Fill_ml", "Fill_pct", "Weight_raw", "Weight_stable_raw", "Battery_pct", "Cap"],
    "events": ["Timestamp_Epoch_ms", "Event", "Detail"],
    "raw":    ["Timestamp_Epoch_ms", "Characteristic", "Hex"],
}
session = bt_debug.SessionFiles("HidrateSpark")


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE) as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s


def save_settings(s):
    with open(SETTINGS_FILE + ".tmp", "w") as f:
        json.dump(s, f, indent=2)
    os.replace(SETTINGS_FILE + ".tmp", SETTINGS_FILE)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
selected_target = None
ble_device_cache = {}
last_rssi = {}
radio_hci = None
_adapters_at = 0.0

live = {}
counters = {}
meters = {}
debug = {
    "adapters": [], "adapter": None, "device": None, "radio": {}, "connected_at": None,
    "recording": None, "gatt": [], "handshake": None, "sip_char": None,
    "counters": counters, "rates": {}, "last_frames": {}, "versions": {}, "last_error": None,
}
for _pkg in ("bleak", "dbus-fast", "fastapi"):
    try:
        debug["versions"][_pkg] = _pkg_version(_pkg)
    except Exception:
        debug["versions"][_pkg] = None
debug["versions"]["python"] = sys.version.split()[0]

weight = hp.StableWeight()
refills = hp.RefillDetector()
dedupe = hp.SipDeduper()


def reset_session_state():
    global weight, refills
    live.clear()
    live.update(fill_ml=None, fill_pct=None, weight_raw=None, weight_stable=None, battery=None,
                cap_open=None, last_sip=None, serial=None, firmware=None, last_notify_at=None)
    counters.clear()
    counters.update(notifications=0, sip_frames=0, sips_live=0, sips_replayed=0, duplicates=0,
                    unknown_frames=0, drain_writes=0, drain_paused=0, skipped_records=0, refills=0, link_failures=counters.get("link_failures", 0))
    meters.clear()
    meters.update({k: bt_debug.RateMeter(window=30) for k in ("notifications", "weight")})
    debug["last_frames"] = {}
    weight = hp.StableWeight()
    refills = hp.RefillDetector()


reset_session_state()


def set_status(state, address=None, attempt=None):
    debug["rates"] = {k: round(m.rate(), 2) for k, m in meters.items()}
    _write_json(STATUS_FILE, {"state": state, "address": address, "attempt": attempt,
                              "max_attempts": CONNECT_ATTEMPTS, "live": live,
                              "last_device": last_device.load(), "settings": load_settings(), "debug": debug})


def event(name, detail=""):
    session.writerows("events", [[int(time.time() * 1000), name, detail]])
    session.flush("events")
    logging.info(f"event: {name} {detail}")


def seed_dedupe_from_today():
    """Sips already recorded today (any session) -- so a replay after a restart isn't double counted."""
    start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    seeds = []
    for path in bt_debug.latest_session_files("HidrateSpark", "sips", limit=20):
        try:
            with open(path) as f:
                next(f, None)
                for line in f:
                    p = line.split(",")
                    ts = int(p[0]) / 1000
                    if ts >= start - 86400:
                        seeds.append((ts, int(float(p[1]))))
        except Exception:
            continue
    dedupe.seed(seeds)
    return len(seeds)


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------
class Session:
    """Everything that needs the connected client."""

    def __init__(self, client, sip_char):
        self.client = client
        self.sip_char = sip_char
        self.connected_at = time.time()
        self.last_frame_hex = None
        self.frame_repeat = 0
        self.drain_task = None
        self.drainer = hp.Drainer(self._write_drain)
        self.last_remaining = 0

    async def _write_drain(self):
        await self.client.write_gatt_char(self.sip_char, hp.DRAIN, response=True)
        counters["drain_writes"] += 1

    async def drain(self):
        """One request at a time (each 0x57 acknowledges a record -- overlapping requests skip one)."""
        if self.sip_char:
            await self.drainer.request()

    def on_notify(self, uuid, data):
        now = time.time()
        data = bytes(data)
        counters["notifications"] += 1
        meters["notifications"].add()
        live["last_notify_at"] = now
        debug["last_frames"][uuid] = {"hex": data.hex(), "at": now}
        session.writerows("raw", [[int(now * 1000), uuid, data.hex()]])
        if uuid == self.sip_char:
            self.on_sip_frame(data, now)
        elif uuid == hp.CHAR_DEBUG:
            self.on_cap(data, now)
        elif uuid == hp.CHAR_WEIGHT:
            self.on_weight(data, now)
        elif uuid == hp.CHAR_BATTERY:
            b = hp.parse_battery(data)
            if b is not None:
                live["battery"] = b

    def on_sip_frame(self, data, now):
        counters["sip_frames"] += 1
        settings = load_settings()
        r = hp.parse_sip_frame(data, settings["capacity_ml"], now)
        self.drainer.on_frame(now)
        if self.last_remaining > 1 and r["kind"] == "sip" and r["remaining"] < self.last_remaining - 1:
            counters["skipped_records"] += self.last_remaining - 1 - r["remaining"]
        if r["kind"] in ("sip", "empty"):
            self.last_remaining = r["remaining"]
        if r["remaining"] > 0:
            hexed = data.hex()
            self.frame_repeat = self.frame_repeat + 1 if hexed == self.last_frame_hex else 0
            self.last_frame_hex = hexed
            if self.frame_repeat >= hp.MAX_IDENTICAL_FRAMES:
                counters["drain_paused"] += 1          # the bottle isn't advancing; don't write-loop
            else:
                asyncio.get_running_loop().create_task(self.drain())
        else:
            self.last_frame_hex, self.frame_repeat = None, 0
        if r["kind"] == "unknown":
            counters["unknown_frames"] += 1
            return
        if r["kind"] != "sip":
            return
        if not dedupe.is_new(r["ts"], r["volume_ml"]):
            counters["duplicates"] += 1
            return
        source = "replayed" if r["ts"] < self.connected_at - 5 else "live"
        counters["sips_" + source] += 1
        session.writerows("sips", [[int(r["ts"] * 1000), r["volume_ml"], r["pct"] if r["pct"] is not None else "",
                                    r["total_reported"], source, r["layout"], data.hex(),
                                    r.get("weight_before", ""), r.get("weight_after", "")]])
        session.flush("sips")
        if live["last_sip"] is None or r["ts"] >= live["last_sip"]["ts"]:
            live["last_sip"] = {"ts": r["ts"], "ml": r["volume_ml"]}
        # without a weight-based fill yet, estimate by subtracting live sips
        if source == "live" and live["fill_ml"] is not None and settings.get("weight_full_raw") is None:
            live["fill_ml"] = max(0, live["fill_ml"] - r["volume_ml"])
            live["fill_pct"] = round(100 * live["fill_ml"] / settings["capacity_ml"])
        logging.info(f"sip {r['volume_ml']} ml ({source}, {r['layout']} layout, {r['remaining'] - 1} more queued)")

    def on_cap(self, data, now):
        is_open = hp.parse_cap(data)
        if is_open is None or is_open == live["cap_open"]:
            return
        live["cap_open"] = is_open
        refills.on_cap(is_open, weight.stable, now)
        event("cap_open" if is_open else "cap_close")

    def on_weight(self, data, now):
        raw = hp.parse_weight(data)
        if raw is None:
            return
        meters["weight"].add()
        live["weight_raw"] = raw
        stable = weight.add(raw, now)
        if stable is None:
            return
        live["weight_stable"] = stable
        settings = load_settings()
        ev = refills.on_stable(stable, now)
        if ev:
            counters["refills"] += 1
            event("refill", f"weight {ev['pre']} -> {ev['post']} (+{ev['rise']})")
            if settings.get("weight_empty_raw") is None:
                settings["weight_full_raw"] = ev["post"]       # latch "full" on a refill (no empty anchor yet)
                save_settings(settings)
        fill_ml, fill_pct = hp.fill_from_weight(stable, settings)
        if fill_ml is not None:
            live["fill_ml"], live["fill_pct"] = fill_ml, fill_pct


def read_command():
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE) as f:
            cmd = json.load(f)
        os.remove(COMMAND_FILE)
        return cmd
    except Exception:
        return None


def handle_command(cmd):
    """Commands that apply while connected (calibration)."""
    if not cmd or cmd.get("action") != "calibrate":
        return cmd
    which = cmd.get("which")
    if which not in ("full", "empty"):
        return None
    if weight.stable is None:
        event("calibrate_failed", f"{which}: no settled weight yet -- stand the bottle upright for a few seconds")
        return None
    settings = load_settings()
    settings[f"weight_{which}_raw"] = weight.stable
    save_settings(settings)
    event(f"calibrate_{which}", f"weight {weight.stable}")
    fill_ml, fill_pct = hp.fill_from_weight(weight.stable, settings)
    if fill_ml is not None:
        live["fill_ml"], live["fill_pct"] = fill_ml, fill_pct
    return None


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
async def scan():
    global selected_target, _adapters_at, radio_hci
    if time.time() - _adapters_at > 10:
        debug["adapters"] = await bt_debug.list_adapters()
        _adapters_at = time.time()
    radio_hci, reason = radio_pin.resolve(debug["adapters"])
    debug["radio"] = {"config": radio_pin.config(), "hci": radio_hci, "reason": reason}
    set_status("scanning")
    found, cut_short = {}, False

    def on_adv(device, adv):
        # Unnamed devices are kept too (shown with "Show all"): some bottles / firmware
        # advertise the name only in a scan response, or not at all.
        name = device.name or adv.local_name or ""
        uuids = list(adv.service_uuids or [])
        ble_device_cache[device.address] = device
        last_rssi[device.address] = adv.rssi
        found[device.address] = {
            "name": name or "(no name)", "address": device.address, "rssi": adv.rssi or -100,
            "bottle": hp.is_bottle(name, uuids),
            "services": [u[:8] for u in uuids][:4],
            "mfr": [f"{k:04x}" for k in (adv.manufacturer_data or {})][:3],
        }

    try:
        async with BleakScanner(on_adv, **bt_debug.bluez_args(radio_hci)):
            for _ in range(20):            # up to 4 s; leave early when the UI picks a device
                await asyncio.sleep(0.2)
                if os.path.exists(COMMAND_FILE):
                    cut_short = True
                    break
    except Exception as e:
        logging.error(f"Scan error: {e}")
        await asyncio.sleep(2)          # e.g. adapter off: don't spin

    if not cut_short:   # bottles first, then everything else (for "show all devices")
        _write_json(DEVICES_FILE, sorted(found.values(), key=lambda d: (not d["bottle"], -d["rssi"]))[:40])

    cmd = read_command()
    if cmd and cmd.get("action") == "connect":
        addr = cmd.get("address")
        name = (found.get(addr) or {}).get("name") or next(
            (d.get("name") for d in _read_devices() if d.get("address") == addr), None)
        selected_target = {"address": addr, "name": name if name and name != "(no name)" else None}
        logging.info(f"Connect requested: {name or ''} [{addr}]")
    else:
        last = last_device.target()
        if last:
            # match by name first: HidrateSpark PRO 2 bottles change address every wake-up
            hit = next((d for d in found.values() if last.get("name") and d["name"] == last["name"]), None) \
                  or found.get(last["address"])
            if hit:
                selected_target = {"address": hit["address"], "name": last.get("name")}
                logging.info(f"Auto-reconnecting to {last.get('name') or ''} [{hit['address']}]")


def _read_devices():
    try:
        with open(DEVICES_FILE) as f:
            return json.load(f)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# One connected session
# ---------------------------------------------------------------------------
async def run_session(device, on_connected=None):
    async with BleakClient(device, **bt_debug.bluez_args(radio_hci)) as client:
        if on_connected:
            await on_connected()
        reset_session_state()
        session.start(getattr(device, "name", None), device.address, SESSION_STREAMS)
        debug["recording"] = session.dir
        debug["connected_at"] = time.time()
        n_seeded = seed_dedupe_from_today()
        event("connect", f"{getattr(device, 'name', '')}; {n_seeded} recent sips known")

        # Map everything the bottle exposes (also the capture for new firmware)
        chars = {}
        debug["gatt"] = []
        for svc in client.services:
            for ch in svc.characteristics:
                chars[ch.uuid.lower()] = ch
                debug["gatt"].append({"service": svc.uuid, "char": ch.uuid, "props": list(ch.properties),
                                      "known": hp.KNOWN_CHARS.get(ch.uuid.lower())})
        sip_char = hp.CHAR_USER_DATA if hp.CHAR_USER_DATA in chars else hp.CHAR_DATA_POINT
        if sip_char not in chars:
            event("protocol_mismatch", "no sip-record characteristic -- capture only (see the debug panel)")
            sip_char = None
        debug["sip_char"] = sip_char
        sess = Session(client, sip_char)

        for uuid, field in ((hp.CHAR_SERIAL, "serial"), (hp.CHAR_FIRMWARE, "firmware"), (hp.CHAR_BATTERY, None)):
            if uuid in chars and "read" in chars[uuid].properties:
                try:
                    val = bytes(await client.read_gatt_char(uuid))
                    if field:
                        live[field] = val.decode("utf-8", "replace").strip("\x00 ")
                    else:
                        live["battery"] = hp.parse_battery(val)
                except Exception:
                    pass

        # Subscribe to every notify/indicate characteristic (known ones are decoded, all are captured)
        for uuid, ch in chars.items():
            if {"notify", "indicate"} & set(ch.properties):
                try:
                    await client.start_notify(uuid, lambda c, d, u=uuid: sess.on_notify(u, d))
                except Exception as e:
                    logging.debug(f"notify {uuid} failed: {e!r}")

        # Handshake (only if this bottle has the characteristics it targets)
        if hp.CHAR_SET_POINT in chars and hp.CHAR_DEBUG in chars:
            try:
                for uuid, payload in hp.HANDSHAKE:
                    await client.write_gatt_char(uuid, bytes.fromhex(payload), response=True)
                    await asyncio.sleep(hp.HANDSHAKE_INTERVAL_S)
                debug["handshake"] = "complete"
            except Exception as e:
                debug["handshake"] = f"failed: {e!r}"
                event("handshake_failed", repr(e))
        else:
            debug["handshake"] = "skipped (characteristics not found)"
        if sip_char:
            await asyncio.sleep(1.5)                   # the bottle may start sending on its own after the handshake
            if sess.drainer.last_frame_at == 0:
                await sess.drain()

        set_status("streaming", device.address)
        last_device.remember(device.address, getattr(device, "name", None))
        logging.info(f"Connected; recording to {session.dir}")

        next_status_row = next_drain = 0
        while client.is_connected:
            await asyncio.sleep(1)
            now = time.time()
            handle_command(read_command())
            if now >= next_status_row:
                next_status_row = now + STATUS_ROW_EVERY_S
                cap = "" if live["cap_open"] is None else ("open" if live["cap_open"] else "closed")
                session.writerows("status", [[int(now * 1000), live["fill_ml"] if live["fill_ml"] is not None else "",
                                              live["fill_pct"] if live["fill_pct"] is not None else "",
                                              live["weight_raw"] or "", live["weight_stable"] or "",
                                              live["battery"] if live["battery"] is not None else "", cap]])
                session.flush()
            # periodic check for queued sips -- only when no drain is under way (never mid-queue)
            if sip_char and now >= next_drain and sess.drainer.idle(now) and now - sess.drainer.last_frame_at > 10:
                next_drain = now + REDRAIN_EVERY_S
                await sess.drain()
            set_status("streaming", device.address)
        logging.warning("Bottle disconnected (it sleeps when idle) -- back to scanning.")
        event("disconnect")


async def main():
    global selected_target
    logging.info("Hydro-dash worker online.")
    _write_json(DEVICES_FILE, [])
    set_status("scanning")
    if os.path.exists(COMMAND_FILE):
        os.remove(COMMAND_FILE)
    while True:
        if not selected_target:
            await scan()
            continue
        target = selected_target
        mac = target["address"]
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            # The bottle only advertises briefly, rotates its address, and BlueZ drops it the
            # moment discovery stops -- so wait for it (by name or address) and connect while
            # the scan is still running; the scan stops once the link is up.
            debug["phase"] = "waiting for the bottle to advertise -- shake or tip it"
            set_status("connecting", mac, attempt)
            seen, box = asyncio.Event(), {}

            def on_adv(dev, adv):                     # bleak requires exactly (device, advertisement)
                name = dev.name or adv.local_name or ""
                if not seen.is_set() and (dev.address == target["address"] or (target["name"] and name == target["name"])):
                    box["dev"] = dev
                    last_rssi[dev.address] = adv.rssi
                    seen.set()

            scanner = None

            async def stop_scan():
                if scanner is not None:
                    with contextlib.suppress(Exception):
                        await scanner.stop()
            try:
                scanner = BleakScanner(on_adv, **bt_debug.bluez_args(radio_hci))
                await scanner.start()
                await asyncio.wait_for(seen.wait(), LOOKUP_TIMEOUT_S)
            except (asyncio.TimeoutError, Exception) as e:
                await stop_scan()
                counters["link_failures"] = counters.get("link_failures", 0) + 1
                msg = (f"bottle not seen for {LOOKUP_TIMEOUT_S:.0f} s -- shake it to wake it, and keep it close"
                       if isinstance(e, asyncio.TimeoutError) else f"scan failed: {e!r}")
                debug["last_error"] = {"at": time.time(), "attempt": attempt, "error": msg}
                logging.warning(f"{msg} (attempt {attempt}/{CONNECT_ATTEMPTS})")
                continue
            device = box["dev"]
            mac = device.address
            debug["phase"] = "connecting"
            debug["adapter"] = bt_debug.adapter_of(device)
            debug["device"] = {"name": device.name, "address": device.address, "rssi": last_rssi.get(device.address)}
            set_status("connecting", mac, attempt)
            try:
                await run_session(device, on_connected=stop_scan)
                break
            except Exception as e:
                counters["link_failures"] = counters.get("link_failures", 0) + 1
                debug["last_error"] = {"at": time.time(), "attempt": attempt,
                                       "error": f"{e.__class__.__name__}: {e}"[:300]}
                logging.warning(f"Link error (attempt {attempt}/{CONNECT_ATTEMPTS}): {e!r}")
                await asyncio.sleep(1)
            finally:
                await stop_scan()
                session.close()
                debug["recording"] = None
        debug["phase"] = None
        selected_target = None
        debug["connected_at"] = None


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        session.close()
