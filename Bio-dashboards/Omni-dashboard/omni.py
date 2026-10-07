import asyncio
import csv
import datetime
import json
import logging
import os
import sys
import threading
import time
import math
from collections import deque
from itertools import islice
from flask import Flask, Response, jsonify, request, send_from_directory
from bleak import BleakClient, BleakScanner
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice
from polar_python import PolarDevice
from importlib.metadata import version as _pkg_version

import bt_debug

# ==========================================
# 1. CONFIGURATION & GLOBAL STATE
# ==========================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ECG_HZ = 130
ACC_HZ = 200

# ---- Recordings: <Documents>/Bio-dash/Omni/<device>/<date>/<time>_<stream>.csv ----
# One session per device, opened when it connects (bt_debug.SessionFiles):
#   H10 folder: ecg, acc, rr (per beat), hrv (1 Hz: HR, RMSSD, SDNN)
#   O2 folder:  vitals (1 Hz: SpO2, HR)
_log_lock = threading.Lock()   # callbacks run on the BLE thread; keep writes whole
POLAR_STREAMS = {
    "ecg": ["Timestamp_Epoch_ms", "ECG_mV"],
    "acc": ["Timestamp_Epoch_ms", "X_mg", "Y_mg", "Z_mg"],
    "rr":  ["Timestamp_Epoch_ms", "RR_ms"],
    "hrv": ["Timestamp_Epoch_ms", "HR_BPM", "RMSSD_ms", "SDNN_ms"],
}
O2_STREAMS = {"vitals": ["Timestamp_Epoch_ms", "SpO2_pct", "HR_BPM"]}
sessions = {"polar": bt_debug.SessionFiles("Omni"), "viatom": bt_debug.SessionFiles("Omni")}

def log_rows(role, stream, rows):
    if not rows:
        return
    with _log_lock:
        sessions[role].writerows(stream, rows)
        sessions[role].flush(stream)

class SampleRing:
    """Thread-safe ring of recent samples with a running sequence number.

    The BLE thread appends; every SSE client keeps its own cursor and asks for
    everything since it last looked. Nothing is ever cleared out from under the
    writer, and N browser tabs each get the full stream.
    """
    def __init__(self, maxlen):
        self._buf = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()

    def extend(self, items):
        with self._lock:
            self._buf.extend(items)
            self._seq += len(items)

    def since(self, seq):
        with self._lock:
            n = min(self._seq - seq, len(self._buf))
            items = list(islice(self._buf, len(self._buf) - n, None)) if n > 0 else []
            return items, self._seq

    @property
    def seq(self):
        with self._lock:
            return self._seq

class StreamClock:
    """Sensor-clock -> wall-clock mapping for PMD frames (timestamp = last sample).
    Gives jitter-free sample spacing; re-anchors if drift exceeds 1 s."""
    def __init__(self, hz):
        self.period_ms = 1000.0 / hz
        self.offset_ms = None
        self.prev_last_ms = None
        self.dropped = 0          # samples missing between frames (from sensor timestamps)
        self.last_frame_len = 0

    def reset(self):
        self.offset_ms = None
        self.prev_last_ms = None
        self.dropped = 0

    def stamps(self, sensor_ts_ns, n):
        now_ms = time.time() * 1000.0
        self.last_frame_len = n
        if sensor_ts_ns:
            last_ms = sensor_ts_ns / 1e6
            if self.prev_last_ms is not None:
                gap = (last_ms - self.prev_last_ms) - n * self.period_ms
                if gap > 1.5 * self.period_ms:
                    self.dropped += round(gap / self.period_ms)
            self.prev_last_ms = last_ms
            if self.offset_ms is None or abs(last_ms + self.offset_ms - now_ms) > 1000.0:
                self.offset_ms = now_ms - last_ms
            end_ms = last_ms + self.offset_ms
        else:
            end_ms = now_ms
        return [int(end_ms - (n - 1 - i) * self.period_ms) for i in range(n)]

ecg_ring = SampleRing(ECG_HZ * 10)   # [ts_ms, mV]
acc_ring = SampleRing(ACC_HZ * 10)   # [ts_ms, x, y, z]
rr_ring = SampleRing(200)            # [ts_ms, rr_ms] -- one entry per beat
ecg_clock = StreamClock(ECG_HZ)
acc_clock = StreamClock(ACC_HZ)

# Shared Memory Matrix (scalars only; waveform data lives in the rings above)
omni_state = {
    "polar": {"status": "Disconnected", "hr": 0, "rr": 0, "rmssd": 0.0, "sdnn": 0.0},
    "viatom": {"status": "Disconnected", "spo2": 0, "hr": 0}
}

discovered_devices = {"polar": [], "viatom": []}
active_targets = {"polar": None, "viatom": None}
# Auto-reconnect: the last H10 and the last O2, remembered separately
last_devices = {role: bt_debug.LastDevice(os.path.join(BASE_DIR, f"last_device_{role}.json")) for role in ("polar", "viatom")}

# DBus Cache
ble_device_cache = {}

polar_ppi_history = deque(maxlen=40)
polar_last_heartbeat = 0
viatom_link_up = False

# Hardware UUIDs
# Optional: pin a specific O2 band by address (it is found by name / service anyway).
VIATOM_MAC = os.environ.get("BIODASH_VIATOM_MAC", "").strip().upper()
VIATOM_SVC_UUID = "14839ac4-7d7e-415c-9a42-167340cf2339"
VIATOM_WRITE_UUID = "8b00ace7-eb0a-49b0-b977-10a8d4d5e82f"

# ==========================================
# RADIO ROUTING
# ==========================================
# The H10 streams ECG + ACC on a tight connection schedule and can starve every
# other link on its radio. With two radios, give it one to itself.
#
# Assignments are stored by adapter ADDRESS (stable), never by hciN (which follows
# plug-in order and can swap between boots). "auto" picks:
#   2+ radios -> Polar on an external radio (else the first), O2 + scanning on another
#   1 radio   -> everything shares it (scanning pauses while the Polar streams)
RADIO_CONFIG_FILE = os.path.join(BASE_DIR, "radio_config.json")
ADAPTER_REFRESH_S = 10

def _load_radio_config():
    try:
        with open(RADIO_CONFIG_FILE) as f:
            cfg = json.load(f)
        return {"polar": cfg.get("polar", "auto"), "viatom": cfg.get("viatom", "auto")}
    except Exception:
        return {"polar": "auto", "viatom": "auto"}

radio_config = _load_radio_config()
adapters = []                  # latest bt_debug.list_adapters() result
routing = {"polar": None, "viatom": None, "scan": None, "why": {}}   # resolved hciN names


def resolve_routing():
    """Turn the address-based config into hciN names for this boot."""
    usable = [a for a in adapters if a.get("powered", True) and not a.get("error")]
    by_addr = {(a.get("address") or "").upper(): a["name"] for a in usable}
    names = [a["name"] for a in usable]
    why = {}

    def pinned(role):
        want = (radio_config.get(role) or "auto").upper()
        if want != "AUTO":
            if want in by_addr:
                why[role] = f"pinned to {want}"
                return by_addr[want]
            why[role] = f"pinned radio {want} not present -- using auto"
        return None

    polar = pinned("polar")
    if polar is None and names:
        ext = [a["name"] for a in usable if a.get("placement") == "external"]
        polar = ext[0] if (len(names) > 1 and ext) else names[0]
        why.setdefault("polar", "auto: external radio" if (len(names) > 1 and ext)
                       else "auto: first radio (no external radio found)" if len(names) > 1 else "auto: only radio")

    viatom = pinned("viatom")
    if viatom is None and names:
        others = [n for n in names if n != polar]
        viatom = others[0] if others else polar
        why.setdefault("viatom", "auto: a radio the Polar isn't using" if others else "auto: shared with Polar (one radio)")

    routing.update(polar=polar, viatom=viatom, scan=viatom, why=why)


def set_radio(role, address):
    radio_config[role] = address or "auto"
    tmp = RADIO_CONFIG_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(radio_config, f, indent=2)
    os.replace(tmp, RADIO_CONFIG_FILE)
    resolve_routing()


def _bluez(hci):
    return {"adapter": hci} if hci else {}


# ==========================================
# DEBUG TELEMETRY
# ==========================================
meters = {k: bt_debug.RateMeter() for k in ("ecg", "acc", "ecg_frames", "acc_frames", "hr")}
viatom_meters = {k: bt_debug.RateMeter(window=20) for k in ("notifications", "readings", "polls")}
viatom_counters = {"fragments_skipped": 0, "calibrating_packets": 0, "out_of_range": 0, "write_failures": 0}
link_failures = {"polar": 0, "viatom": 0}
last_rssi = {}
link_info = {
    "polar": {"adapter": None, "device": None, "connected_at": None},
    "viatom": {"adapter": None, "device": None, "connected_at": None, "last_reading_at": None, "gatt": {}},
}
scanner_state = {"scanning": False, "paused_reason": None, "last_sweep_at": None}
versions = {}
for _pkg in ("bleak", "polar-python", "dbus-fast", "flask"):
    try:
        versions[_pkg] = _pkg_version(_pkg)
    except Exception:
        versions[_pkg] = None
versions["python"] = sys.version.split()[0]

# ==========================================
# 2. FLASK WEB DASHBOARD (STABILIZED UI)
# ==========================================
app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))  # works from any cwd


@app.route('/')
def home():
    return send_from_directory(os.path.join(app.root_path, "templates"), "index.html")

@app.route('/api/scanners')
def get_scanners():
    return jsonify({"devices": discovered_devices, "state": omni_state})

@app.route('/api/connect', methods=['POST'])
def command_connect():
    req = request.json
    t_type = req.get('type')
    t_mac = req.get('address')
    if t_type in active_targets:
        active_targets[t_type] = t_mac
    return jsonify({"status": "locked", "target": t_mac})

@app.route('/api/last-device', methods=['GET', 'POST'])
def last_device_api():
    """GET: the remembered H10 / O2. POST {"role": "polar"|"viatom", "auto": bool} or {"role", "forget": true}."""
    if request.method == 'POST':
        req = request.json or {}
        ld = last_devices.get(req.get("role"))
        if ld is None:
            return jsonify({"error": "role must be polar or viatom"}), 400
        if req.get("forget"):
            ld.forget()
        elif req.get("auto") is not None:
            ld.set_auto(bool(req["auto"]))
    return jsonify({"devices": {role: ld.load() for role, ld in last_devices.items()}})

@app.route('/api/radios', methods=['GET', 'POST'])
def radios():
    """GET: adapters + config + resolved routing. POST {role, address|"auto"}: pin a radio."""
    if request.method == 'POST':
        req = request.json or {}
        role, addr = req.get("role"), req.get("address")
        if role not in ("polar", "viatom"):
            return jsonify({"error": "role must be polar or viatom"}), 400
        set_radio(role, addr)
    return jsonify({"adapters": adapters, "config": radio_config, "routing": routing})

HISTORY_MAX_POINTS = 1500
# metric -> (stream, column) in the device session files
_HISTORY_SRC = {"rr": ("rr", 1), "polar_hr": ("hrv", 1), "rmssd": ("hrv", 2), "sdnn": ("hrv", 3),
                "spo2": ("vitals", 1), "viatom_hr": ("vitals", 2)}

def _tail_lines(path, max_bytes):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    return lines[1:] if size > max_bytes else lines

@app.route('/api/history')
def history():
    """Tile-trend history from the newest recording of the relevant device:
    ?metric=spo2|viatom_hr|polar_hr|rmssd|sdnn|rr&minutes=N"""
    metric = request.args.get("metric", "")
    try:
        minutes = max(1, min(int(request.args.get("minutes", 15)), 60))
    except ValueError:
        minutes = 15
    cutoff = time.time() * 1000 - minutes * 60_000
    pts = []
    if metric not in _HISTORY_SRC:
        return jsonify({})
    stream, col = _HISTORY_SRC[metric]
    files = bt_debug.latest_session_files("Omni", stream, limit=1)
    if not files:
        return jsonify({metric: []})
    path, budget = files[0], minutes * 60 * (4 * 24 if stream == "rr" else 40) + 4096
    try:
        for line in _tail_lines(path, budget):
            p = line.split(",")
            try:
                ts, y = int(p[0]), float(p[col])
            except (ValueError, IndexError):
                continue                    # header or blank (device not connected)
            if ts >= cutoff and y > 0:
                pts.append({"x": ts, "y": y})
    except OSError:
        pass
    step = max(1, len(pts) // HISTORY_MAX_POINTS)
    return jsonify({metric: pts[::step]})

@app.route('/api/debug')
def debug_info():
    p, v = link_info["polar"], link_info["viatom"]
    return jsonify({
        "adapters": adapters, "config": radio_config, "routing": routing,
        "scanner": scanner_state, "versions": versions,
        "polar": {**p, "state": omni_state["polar"]["status"], "link_failures": link_failures["polar"],
                  "nominal": {"ecg_hz": ECG_HZ, "acc_hz": ACC_HZ},
                  "rates": {"ecg_hz": round(meters["ecg"].rate(), 1), "acc_hz": round(meters["acc"].rate(), 1),
                            "ecg_frames_per_s": round(meters["ecg_frames"].rate(), 2),
                            "ecg_samples_per_frame": ecg_clock.last_frame_len,
                            "hr_notifications_per_s": round(meters["hr"].rate(), 2),
                            "ecg_dropped": ecg_clock.dropped, "acc_dropped": acc_clock.dropped}},
        "viatom": {**v, "state": omni_state["viatom"]["status"], "link_failures": link_failures["viatom"],
                   "counters": viatom_counters, "nominal": {"readings_per_s": 0.5},
                   "rates": {"readings_per_s": round(viatom_meters["readings"].rate(), 2),
                             "notifications_per_s": round(viatom_meters["notifications"].rate(), 2),
                             "polls_per_s": round(viatom_meters["polls"].rate(), 2)}},
    })

@app.route('/api/stream')
def stream_data():
    def event_stream():
        # Start each client at "now" so it doesn't replay the ring on connect
        ecg_seq, acc_seq, rr_seq = ecg_ring.seq, acc_ring.seq, rr_ring.seq
        while True:
            ecg, ecg_seq = ecg_ring.since(ecg_seq)
            acc, acc_seq = acc_ring.since(acc_seq)
            rr, rr_seq = rr_ring.since(rr_seq)
            payload = {
                "polar": {**omni_state["polar"], "ecg_buffer": ecg, "acc_buffer": acc, "rr_buffer": rr},
                "viatom": omni_state["viatom"],
            }
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")


# ==========================================
# 3. BLUETOOTH ENGINE (ASYNCIO)
# ==========================================

def _first_attr(obj, names, default=None):
    for n in names:
        v = getattr(obj, n, None)
        if v:
            return v
    return default

def polar_hr_cb(data):
    # polar_python HRData: heartrate (int BPM), rr_intervals (list[float] ms)
    global polar_last_heartbeat
    polar_last_heartbeat = time.time()
    meters["hr"].add()

    bpm = _first_attr(data, ("heartrate", "bpm", "heart_rate"))
    if bpm is not None:
        omni_state["polar"]["hr"] = int(bpm)

    rr_list = _first_attr(data, ("rr_intervals", "rrs", "rrs_ms"), [])
    new_rr = []
    now_ms = int(time.time() * 1000)
    for rr in rr_list:
        try:
            rr_val = int(round(float(rr)))
        except (ValueError, TypeError):
            continue
        if 200 < rr_val < 2000:
            polar_ppi_history.append(rr_val)
            new_rr.append([now_ms, rr_val])

    if not new_rr:
        return
    omni_state["polar"]["rr"] = new_rr[-1][1]
    rr_ring.extend(new_rr)
    log_rows("polar", "rr", new_rr)

    h = polar_ppi_history
    if len(h) > 2:
        sq_diff = sum((h[i] - h[i - 1]) ** 2 for i in range(1, len(h)))
        omni_state["polar"]["rmssd"] = math.sqrt(sq_diff / (len(h) - 1))
        mean_rr = sum(h) / len(h)
        omni_state["polar"]["sdnn"] = math.sqrt(sum((x - mean_rr) ** 2 for x in h) / (len(h) - 1))

def _acc_xyz(val):
    if isinstance(val, (list, tuple)):
        return val[0], val[1], val[2]
    if isinstance(val, dict):
        return val.get('x', val.get('X', 0)), val.get('y', val.get('Y', 0)), val.get('z', val.get('Z', 0))
    return getattr(val, 'x', 0), getattr(val, 'y', 0), getattr(val, 'z', 0)

def polar_acc_cb(data):
    # polar_python ACCData: timestamp (ns, last sample), data (list[(x, y, z)] mG)
    samples = _first_attr(data, ("data", "samples", "acc"), [])
    if not samples: return
    stamps = acc_clock.stamps(getattr(data, "timestamp", None), len(samples))
    out = []
    for ts, val in zip(stamps, samples):
        try:
            x, y, z = _acc_xyz(val)
            out.append([ts, int(x), int(y), int(z)])
        except (ValueError, TypeError, IndexError):
            continue
    acc_ring.extend(out)
    log_rows("polar", "acc", out)
    meters["acc"].add(len(out))
    meters["acc_frames"].add()

def polar_ecg_cb(data):
    # polar_python ECGData: timestamp (ns, last sample), data (list[int] µV)
    samples = _first_attr(data, ("data", "samples", "ecg", "voltages"), [])
    if not samples: return
    stamps = ecg_clock.stamps(getattr(data, "timestamp", None), len(samples))
    out = []
    for ts, val in zip(stamps, samples):
        try:
            out.append([ts, round(int(getattr(val, 'voltage', getattr(val, 'ecg_uv', val))) / 1000.0, 3)])
        except (ValueError, TypeError):
            continue
    if out:
        ecg_ring.extend(out)
        log_rows("polar", "ecg", out)
        meters["ecg"].add(len(out))
        meters["ecg_frames"].add()
        sys.stdout.write(f"\r[ POLAR ] ❤️ HR: {omni_state['polar']['hr']:3d} | ⚡ ECG: {out[-1][1]:>6.3f} mV    ")
        sys.stdout.flush()

def viatom_rx_cb(sender, data):
    if len(data) == 0: return
    viatom_meters["notifications"].add()

    # 0xA5 = sensor settling. This is a DISPLAY state only -- it must not touch the
    # link flag, or the write loop would exit and drop a perfectly good connection.
    if len(data) == 1 and data[0] == 0xA5:
        viatom_counters["calibrating_packets"] += 1
        omni_state["viatom"]["status"] = "Calibrating"
        return

    # Real-time frames start with 0x55; continuation fragments of a longer frame
    # don't, and parsing them at fixed offsets produces garbage readings.
    if data[0] != 0x55:
        viatom_counters["fragments_skipped"] += 1
        return
    if len(data) >= 9:
        spo2_val, hr_val = int(data[7]), int(data[8])
        if not 40 <= spo2_val <= 100:
            viatom_counters["out_of_range"] += 1
        else:
            viatom_meters["readings"].add()
            link_info["viatom"]["last_reading_at"] = time.time()
            omni_state["viatom"]["spo2"] = spo2_val
            omni_state["viatom"]["hr"] = hr_val
            omni_state["viatom"]["status"] = "Connected"


async def adapter_refresher():
    """Keeps the radio list + routing current (dongles can be plugged in any time)."""
    global adapters
    while True:
        try:
            adapters = await bt_debug.list_adapters()
        except Exception as e:
            logging.error(f"Adapter listing failed: {e!r}")
        prev = dict(routing)
        resolve_routing()
        if (routing["polar"], routing["viatom"]) != (prev.get("polar"), prev.get("viatom")):
            logging.info(f"Radio routing: Polar -> {routing['polar'] or 'default'}, "
                         f"O2 + scanning -> {routing['viatom'] or 'default'}")
        await asyncio.sleep(ADAPTER_REFRESH_S)


async def omni_scanner():
    while True:
        scan_hci = routing["scan"]
        is_connecting = any(omni_state[k]["status"] == "Connecting" for k in omni_state)
        polar_live = omni_state["polar"]["status"] == "Connected"

        # Pause while a link is being set up. Pause during Polar streaming ONLY when the
        # scan would run on the Polar's own radio -- on a separate radio, keep scanning.
        reason = None
        if is_connecting:
            reason = "a link is connecting"
        elif polar_live and scan_hci == routing["polar"]:
            reason = "sharing the Polar's radio while it streams"
        elif all(active_targets.values()):
            reason = "both devices selected"

        if reason:
            scanner_state.update(scanning=False, paused_reason=reason)
            await asyncio.sleep(1)
            continue

        found = {"polar": {}, "viatom": {}}

        def scan_cb(device: BLEDevice, adv: AdvertisementData):
            ble_device_cache[device.address] = device
            last_rssi[device.address] = adv.rssi
            name = (device.name or adv.local_name or "").upper()
            addr = device.address.upper()
            uuids = [u.lower() for u in (adv.service_uuids or [])]

            info = {"name": device.name or f"Device ({addr[-5:]})", "address": device.address, "rssi": adv.rssi or -100}
            if "POLAR H10" in name: found["polar"][addr] = info
            elif (VIATOM_MAC and addr == VIATOM_MAC) or VIATOM_SVC_UUID.lower() in uuids or any(x in name for x in ["O2", "CHECKME", "VIATOM", "BAND-WU"]): found["viatom"][addr] = info

        scanner_state.update(scanning=True, paused_reason=None)
        try:
            async with BleakScanner(scan_cb, bluez=_bluez(scan_hci)):
                await asyncio.sleep(2.5)
            for key in discovered_devices:
                discovered_devices[key] = sorted(list(found[key].values()), key=lambda x: x["rssi"], reverse=True)
            # Auto-reconnect: a remembered device is advertising and nothing is selected for its role
            for role, ld in last_devices.items():
                last = ld.target()
                if last and not active_targets[role] and last["address"].upper() in found[role]:
                    active_targets[role] = found[role][last["address"].upper()]["address"]
                    logging.info(f"Auto-reconnecting {role} to {last.get('name') or ''} [{active_targets[role]}]")
            scanner_state["last_sweep_at"] = time.time()
        except Exception as e:
            logging.error(f"Radar blocked on {scan_hci or 'default adapter'}: {repr(e)}")
        await asyncio.sleep(2)


async def _device_on(role, mac):
    """A BLEDevice for `mac` as seen by the radio assigned to `role`.

    BlueZ connects through whichever adapter discovered the device, so if the role's
    radio differs from the scan radio, look the device up on that radio first."""
    hci = routing[role]
    cached = ble_device_cache.get(mac)
    if cached is not None and (hci is None or bt_debug.adapter_of(cached) in (None, hci)):
        return cached
    logging.info(f"Looking up {mac} on {hci} for the {role}...")
    return await BleakScanner.find_device_by_address(mac, timeout=10.0, bluez=_bluez(hci))


async def polar_worker():
    global polar_last_heartbeat
    while True:
        target_mac = active_targets["polar"]
        if not target_mac or omni_state["polar"]["status"] in ["Connected", "Connecting"]:
            await asyncio.sleep(1); continue
            
        omni_state["polar"]["status"] = "Connecting"
        try:
            device = await _device_on("polar", target_mac)
            if device:
                link_info["polar"].update(adapter=bt_debug.adapter_of(device) or routing["polar"],
                                          device={"name": device.name, "address": device.address,
                                                  "rssi": last_rssi.get(device.address)})
                logging.info(f"Polar -> radio {link_info['polar']['adapter'] or 'default'}")
                async with PolarDevice(device) as p:
                    ecg_clock.reset()
                    acc_clock.reset()
                    for m in meters.values(): m.reset()
                    link_info["polar"]["connected_at"] = time.time()
                    omni_state["polar"]["status"] = "Connected"
                    last_devices["polar"].remember(device.address, getattr(device, "name", None))
                    with _log_lock:
                        sessions["polar"].start(getattr(device, "name", None), device.address, POLAR_STREAMS)
                    link_info["polar"]["recording"] = sessions["polar"].dir
                    logging.info(f"Polar recording to {sessions['polar'].dir}")
                    polar_last_heartbeat = time.time()
                    
                    logging.info("Polar handshake complete. Activating Heart Rate Matrix...")
                    try: await p.start_hr_stream(polar_hr_cb)
                    except Exception as e: logging.error(f"[DIAGNOSTIC] HR Error: {e}")
                    await asyncio.sleep(1.5) 
                    
                    logging.info("Activating Kinematics...")
                    try:
                        try: await p.start_acc_stream(polar_acc_cb, 200, 16, 8)
                        except TypeError: await p.start_acc_stream(polar_acc_cb)
                    except Exception as e: logging.error(f"[DIAGNOSTIC] ACC Error: {e}")
                    await asyncio.sleep(1.5)

                    logging.info("Activating ECG...")
                    try:
                        try: await p.start_ecg_stream(polar_ecg_cb, 130, 14)
                        except TypeError: await p.start_ecg_stream(polar_ecg_cb)
                    except Exception as e: logging.error(f"[DIAGNOSTIC] ECG Error: {e}")
                    
                    while time.time() - polar_last_heartbeat < 10.0 and omni_state["polar"]["status"] == "Connected":
                        await asyncio.sleep(1)
                    logging.warning("\n⚠️ POLAR DROPPED: Watchdog timeout.")
            else:
                logging.warning(f"\n⚠️ Target not found in radar cache. Retrying...")
                
        except Exception as e:
            link_failures["polar"] += 1
            logging.error(f"Polar Error: {repr(e)}")

        link_info["polar"]["connected_at"] = None
        with _log_lock:
            sessions["polar"].close()
        link_info["polar"]["recording"] = None
        omni_state["polar"]["status"] = "Disconnected"
        active_targets["polar"] = None
        sys.stdout.write("\n")
        await asyncio.sleep(2)


async def viatom_worker():
    global viatom_link_up

    def handle_disconnect(client):
        global viatom_link_up
        logging.warning("\n⚠️ VIATOM DROPPED: Hardware disconnect detected.")
        viatom_link_up = False

    write_bytes = bytearray([0xAA, 0x17, 0xE8, 0x00, 0x00, 0x00, 0x00, 0x1B])

    while True:
        target_mac = active_targets["viatom"]
        if not target_mac or omni_state["viatom"]["status"] != "Disconnected":
            await asyncio.sleep(1); continue

        omni_state["viatom"]["status"] = "Connecting"
        try:
            device = await _device_on("viatom", target_mac) or target_mac
            link_info["viatom"].update(adapter=(bt_debug.adapter_of(device) if not isinstance(device, str) else None) or routing["viatom"],
                                       device={"name": getattr(device, "name", None), "address": target_mac,
                                               "rssi": last_rssi.get(target_mac)}, gatt={}, last_reading_at=None)
            for m in viatom_meters.values(): m.reset()
            for k in viatom_counters: viatom_counters[k] = 0
            async with BleakClient(device, timeout=15.0, disconnected_callback=handle_disconnect,
                                   bluez=_bluez(routing["viatom"])) as client:
                viatom_link_up = True
                link_info["viatom"]["connected_at"] = time.time()
                logging.info("Viatom connected. Waiting 3s for GATT table to boot...")
                await asyncio.sleep(3.0)

                svc = client.services.get_service(VIATOM_SVC_UUID)
                chars = svc.characteristics if svc else []
                notify_uuid = next((c.uuid for c in chars if "notify" in c.properties), None)
                write_uuid = next((c.uuid for c in chars if "write" in c.properties or "write-without-response" in c.properties), VIATOM_WRITE_UUID)
                if not notify_uuid:
                    raise RuntimeError("no notify characteristic in Viatom service (GATT not resolved)")

                link_info["viatom"]["gatt"] = {"notify_uuid": notify_uuid, "write_uuid": write_uuid,
                                               "mtu": getattr(client, "mtu_size", None), "write_mode": "with response"}
                await client.start_notify(notify_uuid, viatom_rx_cb)
                last_devices["viatom"].remember(target_mac, (link_info["viatom"].get("device") or {}).get("name"))
                with _log_lock:
                    sessions["viatom"].start((link_info["viatom"].get("device") or {}).get("name"), target_mac, O2_STREAMS)
                link_info["viatom"]["recording"] = sessions["viatom"].dir
                logging.info(f"O2 recording to {sessions['viatom'].dir}")
                omni_state["viatom"]["status"] = "Connected"

                use_response = True
                while viatom_link_up and client.is_connected:
                    try:
                        await client.write_gatt_char(write_uuid, write_bytes, response=use_response)
                        viatom_meters["polls"].add()
                    except Exception:
                        viatom_counters["write_failures"] += 1
                        use_response = not use_response  # some firmwares only accept one mode
                        link_info["viatom"]["gatt"]["write_mode"] = "with response" if use_response else "without response"
                    await asyncio.sleep(2)
        except Exception as e:
            link_failures["viatom"] += 1
            logging.error(f"Viatom Error: {repr(e)}")

        link_info["viatom"]["connected_at"] = None
        with _log_lock:
            sessions["viatom"].close()
        link_info["viatom"]["recording"] = None
        viatom_link_up = False
        omni_state["viatom"]["status"] = "Disconnected"
        active_targets["viatom"] = None
        await asyncio.sleep(2)

async def vitals_logger():
    """One vitals row per second while either device is connected (blank = not connected)."""
    while True:
        await asyncio.sleep(1)
        p, v = omni_state["polar"], omni_state["viatom"]
        p_on, v_on = p["status"] == "Connected", v["status"] == "Connected"
        if not (p_on or v_on):
            continue
        num = lambda x, d=0: (round(x, d) if d else int(x)) if x else ""
        now_ms = int(time.time() * 1000)
        if p_on:
            log_rows("polar", "hrv", [[now_ms, num(p["hr"]), num(p["rmssd"], 1), num(p["sdnn"], 1)]])
        if v_on:
            log_rows("viatom", "vitals", [[now_ms, num(v["spo2"]), num(v["hr"])]])


async def async_master():
    # Resolve radios once before scanning so the first sweep uses the right one
    global adapters
    try:
        adapters = await bt_debug.list_adapters()
    except Exception:
        adapters = []
    resolve_routing()
    await asyncio.gather(adapter_refresher(), omni_scanner(), polar_worker(), viatom_worker(), vitals_logger())

def start_ble_engine():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_master())

# ---------- shutdown / reboot (debug panel "System" card) ----------
@app.route("/api/system/power", methods=["GET", "POST"])
def system_power():
    host = request.remote_addr or ""
    if request.method == "GET":
        return jsonify(bt_debug.power_info(host))
    action = (request.get_json(silent=True) or {}).get("action")
    code, out = bt_debug.system_power(action, host, request.headers.get(bt_debug.POWER_HEADER))
    return jsonify(out), code


if __name__ == "__main__":
    ble_thread = threading.Thread(target=start_ble_engine, daemon=True)
    ble_thread.start()
    logging.info("OMNI-DASH LIVE: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
