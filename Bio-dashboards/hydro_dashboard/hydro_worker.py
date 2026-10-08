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
def _settings_path():
    """Settings live outside the dashboard folder, so replacing the folder with a new release
    keeps the goal, reminders and the bottle's calibration. An old in-folder file is adopted once."""
    d = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "bio-dash")
    new, old = os.path.join(d, "hydro_settings.json"), os.path.join(BASE_DIR, "hydro_settings.json")
    try:
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(new) and os.path.exists(old):
            with open(old, "rb") as src, open(new, "wb") as dst:
                dst.write(src.read())
    except OSError:
        return old
    return new


SETTINGS_FILE = _settings_path()

radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"), prefer="internal")   # same default as the other dashboards; pick a dongle in the debug menu if the built-in radio can't hear the bottle (see TROUBLESHOOTING 12)
last_device = bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))

CONNECT_ATTEMPTS = 3
CONNECT_TIMEOUT_S = 45.0
LOOKUP_TIMEOUT_S = 15.0      # per attempt: how long to wait for the bottle to advertise
STATUS_ROW_EVERY_S = 5
REDRAIN_EVERY_S = 30          # ask for buffered sips periodically, in case live ones only queue
# capacity_auto: take the capacity the bottle reports (turned off when you type your own).
# cal_min / cal_max: the bottle's own empty / full weights, remembered from its last sip record.
# reminders: the glow-reminder schedule sent to the bottle. goal_glow: glow when the goal is
# reached. sip_glow: glow on each sip (None = leave the bottle as it is).
DEFAULT_SETTINGS = {"capacity_ml": 621, "capacity_auto": True, "goal_ml": 2500, "units": "ml",
                    "cal_min": None, "cal_max": None, "cal_at": None,
                    "reminders": dict(hp.DEFAULT_REMINDERS), "goal_glow": True, "sip_glow": None,
                    "glow": {"style": "pulse", "color1": "#2bff00", "color2": "#1499ff"}}
# BIODASH_HYDRO_DRAIN=fast forces the V3.0 way of draining (one 0x57 per record, no acknowledge)
DRAIN_OVERRIDE = os.environ.get("BIODASH_HYDRO_DRAIN", "").strip().lower()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

SESSION_STREAMS = {
    "sips":   ["Timestamp_Epoch_ms", "Volume_ml", "Pct", "Total_Reported", "Source", "Layout", "Raw",
               "Weight_before", "Weight_after", "Cal_min", "Cal_max", "Volume_from"],
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
                cap_open=None, last_sip=None, serial=None, firmware=None, last_notify_at=None,
                fill_from=None, capacity_from=None, can_glow=None, synced_at=None, reminders_set=None, can_upload_glow=None, hardware=None, glow_upload=None, family=None, drain_mode=None, cal_min=None, cal_max=None)
    counters.clear()
    counters.update(notifications=0, sip_frames=0, sips_live=0, sips_replayed=0, duplicates=0,
                    unknown_frames=0, drain_writes=0, drain_acks=0, drain_paused=0, drain_fallback=0, skipped_records=0,
                    weight_out_of_band=0, refills=0, link_failures=counters.get("link_failures", 0))
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


def today_total_ml():
    """Today's intake so far, from the sip log (all sessions, replays de-duplicated)."""
    start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    seen, total = hp.SipDeduper(volume_tol=dedupe.volume_tol), 0
    for path in bt_debug.latest_session_files("HidrateSpark", "sips", limit=20):
        try:
            with open(path) as f:
                next(f, None)
                for line in f:
                    p = line.split(",")
                    ts, ml = int(p[0]) / 1000, int(float(p[1]))
                    if ts >= start and seen.is_new(ts, ml):
                        total += ml
        except Exception:
            continue
    return total


def calibration_from_log():
    """The bottle's empty / full weights from the newest sip record on disk that has them."""
    best = None
    for path in bt_debug.latest_session_files("HidrateSpark", "sips", limit=20):
        try:
            with open(path) as f:
                head = next(f, "").strip().split(",")
                lo, hi = head.index("Cal_min"), head.index("Cal_max")
                for line in f:
                    p = line.rstrip("\n").split(",")
                    try:
                        ts, cal = int(p[0]), (int(p[lo]), int(p[hi]))
                    except (ValueError, IndexError):
                        continue
                    if hp.valid_calibration(*cal) and (best is None or ts > best[0]):
                        best = (ts, cal)
        except Exception:
            continue
    return best[1] if best else None


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

    def __init__(self, client, sip_char, drain_mode="fast"):
        self.client = client
        self.sip_char = sip_char
        self.connected_at = time.time()
        self.last_frame_hex = None
        self.frame_repeat = 0
        self.drain_task = None
        self.drainer = hp.Drainer(self._write_sip_char, mode=drain_mode)
        self.last_remaining = 0
        self.newest_sip_ts = 0.0
        self.has_led = False
        self.can_upload = False
        self.glow_data_response = True
        self.synced_day = None

    async def _write_sip_char(self, payload):
        await self.client.write_gatt_char(self.sip_char, payload, response=True)
        counters["drain_acks" if payload == hp.ACK else "drain_writes"] += 1

    async def drain(self, ack=False):
        """One request at a time (overlapping requests skip records). ack=True: acknowledge the
        record just received first (bottles that need it -- see hp.drain_mode_for)."""
        if self.sip_char:
            await self.drainer.request(ack=ack, pending=self.last_remaining > 0)
            if self.drainer.fell_back and not counters["drain_fallback"]:
                counters["drain_fallback"] = 1
                event("drain_fallback", "acknowledged draining got no answer -- using the fast (0x57) mode for this connection")
            live["drain_mode"] = self.drainer.mode

    async def sync(self, why="connect"):
        """Bring the bottle in step: today's total, goal, clock, glow settings and the reminder
        schedule. Returns True when every write went through."""
        settings = load_settings()
        lt = time.localtime()
        writes = hp.build_sync(settings, live.get("firmware"), today_total_ml(),
                               lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
        try:
            for uuid, payload in writes:
                await self.client.write_gatt_char(uuid, payload, response=True)
                await asyncio.sleep(hp.HANDSHAKE_INTERVAL_S)
        except Exception as e:
            debug["handshake"] = f"failed: {e!r}"
            event("sync_failed", repr(e))
            return False
        n = len(hp.reminder_times(settings.get("reminders")))
        live["synced_at"], live["reminders_set"] = time.time(), min(n, hp.reminder_slots(live.get("firmware")))
        self.synced_day = lt.tm_yday
        debug["handshake"] = "complete"
        if why != "connect":
            event("sync", f"{why}: {live['reminders_set']} reminder(s), clock and goal sent")
        return True

    async def glow_now(self):
        if not self.has_led:
            event("glow_failed", "this bottle doesn't expose the LED control characteristic")
            return
        try:
            await self.client.write_gatt_char(hp.CHAR_LED, hp.glow_now_bytes(live.get("firmware")), response=True)
            event("glow", "glow now")
        except Exception as e:
            event("glow_failed", repr(e))

    async def upload_glow(self):
        """Send the glow chosen in the dashboard to the bottle (it keeps it), then play it once."""
        if not self.can_upload:
            event("glow_upload_failed", "this bottle doesn't take custom glows")
            return
        g = {**DEFAULT_SETTINGS["glow"], **(load_settings().get("glow") or {})}
        try:
            pattern = hp.glow_pattern(g["style"], [g["color1"], g["color2"]], hp.led_count(live.get("hardware")))
            count, packets = hp.glow_packets(hp.encode_glow_pattern(pattern))
            live["glow_upload"] = {"state": "sending", "packets": len(packets), "at": time.time()}
            await self.client.write_gatt_char(hp.CHAR_GLOW_COUNT, count, response=True)
            for chunk in packets:
                await self.client.write_gatt_char(hp.CHAR_GLOW_DATA, chunk, response=self.glow_data_response)
                if not self.glow_data_response:
                    await asyncio.sleep(0.03)
            live["glow_upload"] = {"state": "done", "packets": len(packets), "at": time.time(), "style": g["style"]}
            event("glow_upload", f"{g['style']} {g['color1']} {g['color2']}: {len(packets)} packets")
        except Exception as e:
            live["glow_upload"] = {"state": "failed", "error": repr(e)[:160], "at": time.time()}
            event("glow_upload_failed", repr(e))
            return
        await asyncio.sleep(0.3)
        await self.glow_now()

    def set_calibration(self, cal_min, cal_max):
        """The bottle's own empty / full weights, from a sip record. Remembered across restarts."""
        if not hp.valid_calibration(cal_min, cal_max):
            return
        if (live["cal_min"], live["cal_max"]) != (cal_min, cal_max):
            first = live["cal_min"] is None
            live["cal_min"], live["cal_max"] = cal_min, cal_max
            settings = load_settings()
            if (settings.get("cal_min"), settings.get("cal_max")) != (cal_min, cal_max):
                settings.update(cal_min=cal_min, cal_max=cal_max, cal_at=int(time.time()))
                save_settings(settings)
                event("calibration" if first else "calibration_changed", f"empty {cal_min} / full {cal_max}")

    def set_fill(self, raw, source):
        fill_ml, fill_pct = hp.fill_from_calibration(raw, live["cal_min"], live["cal_max"], load_settings()["capacity_ml"])
        if fill_ml is not None:
            live["fill_ml"], live["fill_pct"], live["fill_from"] = fill_ml, fill_pct, source

    async def calibrate(self, which):
        """Tell the bottle to redo its own calibration: "this is full" / "this is empty"."""
        if which == "full":
            payload = hp.CAL_FULL
        else:
            major = hp.firmware_major(live.get("firmware"))
            payload = hp.CAL_EMPTY if major is None or major >= 50 else hp.CAL_EMPTY_OLD
        try:
            await self.client.write_gatt_char(hp.CHAR_DEBUG, payload, response=True)
            event(f"calibrate_{which}", f"sent {payload.hex()} -- the bottle confirms its new value with the next sip record")
            self.assume_calibrated(which)
        except Exception as e:
            event("calibrate_failed", f"{which}: {e!r}")

    def assume_calibrated(self, which):
        """Show the result of "it's full / empty now" straight away. The bottle takes its current
        weight as the new full / empty point but only reports that with its next sip record, so
        until then use the settled weight we last saw. Kept for this session only: what is
        remembered across restarts is always the bottle's own figure."""
        w = live.get("weight_stable") or live.get("weight_raw")
        cap = load_settings()["capacity_ml"]
        if w is not None:
            lo, hi = (live["cal_min"], w) if which == "full" else (w, live["cal_max"])
            if hp.valid_calibration(lo, hi):
                live["cal_min"], live["cal_max"] = lo, hi
        live["fill_ml"], live["fill_pct"] = (cap, 100) if which == "full" else (0, 0)
        live["fill_from"] = f"you said it's {which}"

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
                # a full record gets acknowledged before the next is requested; a bare
                # "more waiting" frame is just asked again
                asyncio.get_running_loop().create_task(self.drain(ack=r["kind"] in ("sip", "unknown")))
        else:
            self.last_frame_hex, self.frame_repeat = None, 0
        if r["kind"] == "unknown":
            counters["unknown_frames"] += 1
            return
        if r["kind"] != "sip":
            return
        self.set_calibration(r.get("cal_min"), r.get("cal_max"))
        # the newest record's "weight after" is the level right now, unless a settled live
        # reading has arrived since
        if r["ts"] >= self.newest_sip_ts and r.get("weight_after"):
            self.newest_sip_ts = r["ts"]
            if weight.stable_at is None or weight.stable_at < r["ts"] or live["fill_from"] != "live weight":
                self.set_fill(r["weight_after"], "last sip record")
        if not dedupe.is_new(r["ts"], r["volume_ml"]):
            counters["duplicates"] += 1
            return
        source = "replayed" if r["ts"] < self.connected_at - 5 else "live"
        counters["sips_" + source] += 1
        session.writerows("sips", [[int(r["ts"] * 1000), r["volume_ml"], r["pct"] if r["pct"] is not None else "",
                                    r["total_reported"], source, r["layout"], data.hex(),
                                    r.get("weight_before", ""), r.get("weight_after", ""),
                                    r.get("cal_min") or "", r.get("cal_max") or "", r.get("volume_source", "")]])
        session.flush("sips")
        if live["last_sip"] is None or r["ts"] >= live["last_sip"]["ts"]:
            live["last_sip"] = {"ts": r["ts"], "ml": r["volume_ml"]}
        logging.info(f"sip {r['volume_ml']} ml ({source}, {r['layout']} layout, from {r.get('volume_source', 'record')}, "
                     f"{r['remaining'] - 1} more queued)")

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
        ev = refills.on_stable(stable, now)
        if ev:
            counters["refills"] += 1
            event("refill", f"weight {ev['pre']} -> {ev['post']} (+{ev['rise']})")
        # fill level from the settled live weight and the bottle's own calibration -- but only
        # when the reading is on the calibration's scale (a tilted or lifted bottle isn't)
        if live["cal_min"] is not None:
            if hp.weight_in_band(stable, live["cal_min"], live["cal_max"]):
                self.set_fill(stable, "live weight")
            else:
                counters["weight_out_of_band"] += 1


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


async def handle_command(cmd, sess):
    """Commands that apply while connected: bottle calibration, glow now, re-sync."""
    if not cmd:
        return cmd
    action = cmd.get("action")
    if action == "calibrate" and cmd.get("which") in ("full", "empty"):
        await sess.calibrate(cmd["which"])
    elif action == "glow":
        await sess.glow_now()
    elif action == "glow_pattern":
        await sess.upload_glow()
    elif action == "sync":
        await sess.sync("settings changed")
    else:
        return cmd
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
    # timeout: connecting includes reading the service list, which is slow the first time at a
    # new address (the puck changes address when it restarts) and on older dongles
    async with BleakClient(device, timeout=CONNECT_TIMEOUT_S, **bt_debug.bluez_args(radio_hci)) as client:
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
        if not chars:
            # The link came up but reading the service list failed. The bottle has been seen
            # to stop answering part-way through that read; fail this attempt so it's retried
            # instead of sitting "connected" with nothing to talk to.
            raise RuntimeError("connected, but the bottle's service list came back empty -- "
                               "it stopped answering part-way through")
        sip_char = hp.CHAR_USER_DATA if hp.CHAR_USER_DATA in chars else hp.CHAR_DATA_POINT
        if sip_char not in chars:
            event("protocol_mismatch", "no sip-record characteristic -- capture only (see the debug panel)")
            sip_char = None
        debug["sip_char"] = sip_char

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

        # What kind of bottle this is decides how stored records are drained
        live["family"] = hp.bottle_family(live.get("firmware"))
        drain_mode = "fast" if DRAIN_OVERRIDE == "fast" else hp.drain_mode_for(live.get("firmware"))
        live["drain_mode"] = drain_mode
        sess = Session(client, sip_char, drain_mode)

        # Capacity straight from the bottle (unless you typed your own)
        settings = load_settings()
        if hp.CHAR_BOTTLE_SIZE in chars and "read" in chars[hp.CHAR_BOTTLE_SIZE].properties:
            try:
                size = hp.parse_bottle_size(await client.read_gatt_char(hp.CHAR_BOTTLE_SIZE))
            except Exception:
                size = None
            if size:
                live["capacity_bottle"] = size
                if settings.get("capacity_auto", True) and settings.get("capacity_ml") != size:
                    settings["capacity_ml"] = size
                    save_settings(settings)
                    event("capacity", f"{size} mL (read from the bottle)")
        live["capacity_from"] = "bottle" if live.get("capacity_bottle") and settings.get("capacity_auto", True) else "settings"
        # The calibration remembered from the last session, until this session's first record
        if hp.valid_calibration(settings.get("cal_min"), settings.get("cal_max")):
            live["cal_min"], live["cal_max"] = settings["cal_min"], settings["cal_max"]
        else:
            # none saved here (new machine, recordings copied over): the sip log has it
            cal = calibration_from_log()
            if cal:
                live["cal_min"], live["cal_max"] = cal
                event("calibration", f"empty {cal[0]} / full {cal[1]} (from the sip log)")
        # a sip first logged from the whole-percent byte may be replayed with a weight-based volume
        dedupe.volume_tol = round(settings["capacity_ml"] * 0.012) + 2

        # Subscribe to every notify/indicate characteristic (known ones are decoded, all are captured)
        for uuid, ch in chars.items():
            if {"notify", "indicate"} & set(ch.properties):
                try:
                    await client.start_notify(uuid, lambda c, d, u=uuid: sess.on_notify(u, d))
                except Exception as e:
                    logging.debug(f"notify {uuid} failed: {e!r}")

        # Sync (only if this bottle has the characteristics it targets): total, goal, clock,
        # glow settings and reminders. The bottle sends no sip records before it.
        sess.has_led = hp.CHAR_LED in chars
        live["can_glow"] = sess.has_led
        sess.can_upload = hp.CHAR_GLOW_COUNT in chars and hp.CHAR_GLOW_DATA in chars
        sess.glow_data_response = sess.can_upload and "write" in chars[hp.CHAR_GLOW_DATA].properties
        live["can_upload_glow"] = sess.can_upload
        if hp.CHAR_HARDWARE in chars and "read" in chars[hp.CHAR_HARDWARE].properties:
            with contextlib.suppress(Exception):
                live["hardware"] = bytes(await client.read_gatt_char(hp.CHAR_HARDWARE)).decode("utf-8", "replace").strip("\x00 ")
        if hp.CHAR_SET_POINT in chars and hp.CHAR_DEBUG in chars:
            await sess.sync()
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
            await handle_command(read_command(), sess)
            # a new day: the total and the clock need sending again
            if sess.synced_day is not None and time.localtime().tm_yday != sess.synced_day and sess.drainer.idle(now):
                await sess.sync("new day")
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
