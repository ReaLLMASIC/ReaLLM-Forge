# app.py
import json
import os
import time
from flask import Flask, Response, jsonify, request, send_from_directory

import bt_debug

app = Flask(__name__, root_path=os.path.dirname(os.path.abspath(__file__)))  # works from any cwd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
DEVICES_FILE = os.path.join(BASE_DIR, "devices.json")
COMMAND_FILE = os.path.join(BASE_DIR, "command.json")
DEBUG_FILE = os.path.join(BASE_DIR, "debug.json")


@app.route('/')
def home():
    return send_from_directory(os.path.join(app.root_path, "templates"), "index.html")

@app.route('/api/scan-results')
def scan_results():
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, 'r') as f: return jsonify(json.load(f))
        except Exception: pass
    return jsonify([])

HISTORY_MAX_POINTS = 1500

@app.route('/api/history')
def history():
    """Trend history from the recordings in <Documents>/Bio-dash/Viatom O2/<device>/<date>/
    (one row per 5 s while connected). Only the current band's sessions are used."""
    try:
        minutes = max(1, min(int(request.args.get('minutes', 15)), 60))
    except ValueError:
        minutes = 15
    cutoff_ms = (time.time() - minutes * 60) * 1000
    out = {"spo2": [], "hr": [], "battery": []}
    for path in reversed(bt_debug.latest_session_files("Viatom O2", "vitals", limit=6, same_device=True)):
        try:
            with open(path) as f:
                next(f, None)                                  # header
                for line in f:
                    p = line.strip().split(",")
                    try:
                        x, spo2, hr, batt = int(p[0]), int(float(p[1])), int(float(p[2])), int(float(p[3]))
                    except (ValueError, IndexError):
                        continue
                    if x < cutoff_ms:
                        continue
                    if 40 <= spo2 <= 100: out["spo2"].append({"x": x, "y": spo2})
                    if hr > 0:            out["hr"].append({"x": x, "y": hr})
                    if batt > 0:          out["battery"].append({"x": x, "y": batt})
        except OSError:
            continue
    for k, pts in out.items():
        step = max(1, len(pts) // HISTORY_MAX_POINTS)
        out[k] = pts[::step]
    return jsonify(out)

@app.route('/api/debug')
def debug_info():
    """Radio, link, GATT and rate telemetry written by ble_worker.py once a second."""
    try:
        with open(DEBUG_FILE, 'r') as f: return jsonify(json.load(f))
    except Exception:
        return jsonify({})

# ---------- auto-reconnect to the last device (see bt_debug.LastDevice) ----------
LAST_DEVICES = {"device": bt_debug.LastDevice(os.path.join(BASE_DIR, "last_device.json"))}

@app.route('/api/last-device', methods=['GET', 'POST'])
def last_device_api():
    """GET: remembered devices. POST {"role", "auto": bool} or {"role", "forget": true}."""
    if request.method == 'POST':
        req = request.json or {}
        ld = LAST_DEVICES.get(req.get("role", "device"))
        if ld is None:
            return jsonify({"error": "unknown role"}), 400
        if req.get("forget"):
            ld.forget()
        elif req.get("auto") is not None:
            ld.set_auto(bool(req["auto"]))
    return jsonify({"devices": {role: ld.load() for role, ld in LAST_DEVICES.items()}})


radio_pin = bt_debug.RadioPin(os.path.join(BASE_DIR, "radio_config.json"))

@app.route('/api/radio', methods=['GET', 'POST'])
def radio():
    """Pin the O2 to a radio by adapter address ("auto" to unpin). The worker
    re-resolves within a second and restarts its scan there; a live link isn't dropped."""
    if request.method == 'POST':
        radio_pin.save((request.json or {}).get("address", "auto"))
    return jsonify({"radio": radio_pin.config()})

@app.route('/api/connect', methods=['POST'])
def connect():
    target_mac = request.json.get('address')
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({"action": "connect", "address": target_mac}, f)
    os.replace(tmp, COMMAND_FILE)
    return jsonify({"status": "command_sent"})

@app.route('/api/vitals-stream')
def vitals_stream():
    def event_stream():
        last_mtime = 0
        while True:
            if os.path.exists(DATA_FILE):
                try:
                    current_mtime = os.path.getmtime(DATA_FILE)
                    if current_mtime != last_mtime:
                        last_mtime = current_mtime
                        with open(DATA_FILE, 'r') as f: yield f"data: {f.read()}\n\n"
                except Exception: pass
            time.sleep(0.05)
    return Response(event_stream(), mimetype="text/event-stream")

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
    app.run(host="0.0.0.0", port=5003, debug=False, threaded=True)   # 5000 is Omni
