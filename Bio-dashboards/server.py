#!/usr/bin/env python3
"""Bio-dash: one app for every device.

    python3 server.py                  http://<this computer>:5000
    python3 server.py --port 8080
    python3 server.py --demo           made-up readings, no Bluetooth (to try the app out)

One process, one port, one service. It
  * runs the Bluetooth worker of each device you turned on (and restarts it if it stops),
  * serves each device's API under /api/d/<device>/... (the code in <device folder>/api.py),
  * pushes live readings and connection status to every open page over one websocket (/ws),
  * serves the app itself (web/).

Up to three people can wear a strap and a ring at once. The first person's are "heart" and "oxygen",
exactly as with one person; each extra person gets one more worker and API of the same code
("heart-p2", "oxygen-p2" ...), keeping its small state files in ~/.config/bio-dash/people/<id>/.

The workers are the same programs as before and write the same recordings, in the same
format, to <Documents>/Bio-dash/<device>/<unit>/<date>/<time>_<stream>.csv.
"""
import argparse
import asyncio
import collections
import contextlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
import time
import types

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
sys.path.insert(0, HERE)
import bt_debug  # noqa: E402

CONFIG_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "bio-dash")
CONFIG_FILE = os.environ.get("BIODASH_APP_CONFIG") or os.path.join(CONFIG_DIR, "app.json")
DEMO = os.environ.get("BIODASH_DEMO") == "1"


def version():
    try:
        m = re.search(r"^## V(\d+(?:\.\d+)*)", open(os.path.join(HERE, "CHANGELOG.md")).read(), re.M)
        return m.group(1) if m else "?"
    except OSError:
        return "?"


VERSION = version()


# ---------------------------------------------------------------------------------- devices
class Kind:
    """A kind of device: its folder holds the Bluetooth worker and api.py."""

    def __init__(self, key, label, product, folder, worker, recordings, personal=False):
        self.key, self.label, self.product = key, label, product
        self.folder = os.path.join(HERE, folder)
        self.worker = worker                      # the Bluetooth program in that folder
        self.recordings = recordings              # its folder name under Documents/Bio-dash
        self.personal = personal                  # worn: one per person (the others are one per room / owner)


KINDS = {k.key: k for k in [
    Kind("heart", "Heart", "Polar H10", "polar_dashboard", "advanced_worker.py", "Polar H10", personal=True),
    Kind("oxygen", "Oxygen", "Checkme O2 ring", "viatom_dashboard", "ble_worker.py", "Viatom O2", personal=True),
    Kind("brain", "Brain", "Muse S Athena", "muse_dashboard", "muse_worker.py", "Muse"),
    Kind("air", "Air", "Atmos", "atmos_dashboard", "sensor_worker.py", "Atmos"),
    Kind("water", "Water", "HidrateSpark bottle", "hydro_dashboard", "hydro_worker.py", "HidrateSpark"),
]}
SLOTS = ("p1", "p2", "p3")                    # the people; p1 is always there
STATE_ROOT = os.path.dirname(CONFIG_FILE)     # extra people's (and the demo's) worker state lives under here


class Instance:
    """One worker + API of a kind. key: "heart", or "heart-p2" for an extra person's."""

    def __init__(self, kind, person=None):
        self.kind, self.person = kind, person
        extra = person not in (None, SLOTS[0])
        self.key = f"{kind.key}-{person}" if extra else kind.key
        self.folder, self.worker = kind.folder, kind.worker
        if DEMO:                                  # the demo never touches the real state files
            self.state_dir = os.path.join(STATE_ROOT, "demo-state", self.key)
        elif extra:
            self.state_dir = os.path.join(STATE_ROOT, "people", person, kind.key)
        else:
            self.state_dir = kind.folder
        self.api = None                           # the loaded api.py module
        self.proc = None                          # Worker
        self.lifespan = None

    def env(self):
        return {} if self.state_dir == self.folder else {"BIODASH_STATE_DIR": self.state_dir}

    async def load(self):
        """Import this instance's api.py, mount it at /api/d/<key> and start its file followers."""
        if self.api is not None:
            return
        self.api = load_module(f"biodash_{self.key.replace('-', '_')}_api", self.folder, self.env())
        app.mount(f"/api/d/{self.key}", self.api.app)
        self.lifespan = self.api.app.router.lifespan_context(self.api.app)
        await self.lifespan.__aenter__()
        self.proc = Worker(self)

    async def unload(self):
        if self.proc:
            await self.proc.stop()
        if self.lifespan:
            with contextlib.suppress(Exception):
                await self.lifespan.__aexit__(None, None, None)
            self.lifespan = None

    def last_device(self):
        try:
            return self.api.LAST_DEVICES["device"]
        except (AttributeError, KeyError):
            return None


INSTANCES = {}
for _k in KINDS.values():
    for _p in (SLOTS if _k.personal else (None,)):
        _i = Instance(_k, _p)
        INSTANCES[_i.key] = _i
ANALYSIS_FOLDER = os.path.join(HERE, "analysis_dashboard")
ANALYSIS = None                               # its api.py, once loaded


def load_module(name, folder, env=None):
    """A device's api.py, as its own module (they share names like `app`, so no plain import).
    env: set while it loads (BIODASH_STATE_DIR: where this instance keeps its state)."""
    if folder not in sys.path:
        sys.path.insert(1, folder)            # its helpers: atmos_parser, water, biodata ...
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, "api.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    os.environ.update(env or {})
    try:
        spec.loader.exec_module(mod)
    finally:
        for k in env or {}:
            os.environ.pop(k, None)
    return mod


# ---------------------------------------------------------------------------------- settings
def read_config():
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except Exception:
        cfg = None
    if not isinstance(cfg, dict) and DEMO:
        fresh = os.environ.get("BIODASH_DEMO_FRESH") == "1"     # demo the first-run setup too
        cfg = {"devices": {} if fresh else {k: True for k in KINDS}, "setup_done": not fresh}
    if not isinstance(cfg, dict):
        cfg = {"devices": {}, "setup_done": False}
        # coming from the separate dashboards: a device that was paired before is one you own
        for k in KINDS.values():
            try:
                if json.load(open(os.path.join(k.folder, "last_device.json"))).get("address"):
                    cfg["devices"][k.key] = True
            except Exception:
                pass
        cfg["setup_done"] = bool(cfg["devices"])
    cfg.setdefault("devices", {})
    cfg.setdefault("setup_done", False)
    people = [p for p in cfg.get("people") or [] if isinstance(p, dict) and p.get("id") in SLOTS]
    if not any(p["id"] == SLOTS[0] for p in people):
        people.insert(0, {"id": SLOTS[0], "name": ""})
    cfg["people"] = sorted(people, key=lambda p: SLOTS.index(p["id"]))
    return cfg


def write_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_FILE)


CONFIG = read_config()


def person(pid):
    return next((p for p in CONFIG["people"] if p["id"] == pid), None)


def enabled(key):
    inst = INSTANCES.get(key)
    if inst is None:
        return False
    if inst.person in (None, SLOTS[0]):
        return bool(CONFIG["devices"].get(inst.kind.key))
    p = person(inst.person)
    return bool(p and (p.get("devices") or {}).get(inst.kind.key))


def active():
    """The instances that exist now: every kind's first, plus the extra people's."""
    return [i for i in INSTANCES.values() if i.person in (None, SLOTS[0]) or person(i.person)]


# ---------------------------------------------------------------------------------- workers
def _die_with_parent():
    """In the worker, before it starts: if the server dies without stopping it (killed, crashed),
    Linux sends it Ctrl+C, so no worker is left holding a Bluetooth device."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGINT)     # PR_SET_PDEATHSIG
    except Exception:
        pass


class Worker:
    """Runs one device's Bluetooth program, restarting it (with a growing pause) if it stops."""

    def __init__(self, device):
        self.d = device
        self.wanted = False
        self.proc = None
        self.task = None
        self.state = "off"                   # off | starting | running | restarting | stopping
        self.started_at = None
        self.restarts = 0
        self.last_exit = None
        self.log = collections.deque(maxlen=400)

    def info(self):
        return {"state": self.state, "pid": self.proc.pid if self.proc and self.proc.returncode is None else None,
                "started_at": self.started_at, "restarts": self.restarts, "last_exit": self.last_exit}

    def _note(self, line):
        self.log.append([time.time(), line])
        print(f"[{self.d.key}] {line}", flush=True)

    def start(self):
        if self.wanted:
            return
        self.wanted = True
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        self.wanted = False
        if self.proc and self.proc.returncode is None:
            self.state = "stopping"
            with contextlib.suppress(ProcessLookupError):
                self.proc.send_signal(signal.SIGINT)        # the workers close their recordings on Ctrl+C
            try:
                await asyncio.wait_for(self.proc.wait(), 8)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.proc.kill()
                await self.proc.wait()
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
        self.task, self.state = None, "off"

    async def restart(self):
        await self.stop()
        self.start()

    def command(self):
        if DEMO:
            return [sys.executable, "-u", os.path.join(HERE, "tools", "demo_feed.py"), self.d.kind.key,
                    str(SLOTS.index(self.d.person or SLOTS[0]))]
        return [sys.executable, "-u", self.d.worker]

    async def _run(self):
        pause = 2
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   PYTHONPATH=os.pathsep.join(p for p in (HERE, os.environ.get("PYTHONPATH")) if p), **self.d.env())
        while self.wanted:
            self.state = "starting"
            began = time.time()
            try:
                # its own session: a Ctrl+C in the terminal reaches the server, which then stops
                # each worker in turn, instead of every worker at once racing the restart logic
                self.proc = await asyncio.create_subprocess_exec(
                    *self.command(), cwd=self.d.folder, env=env, start_new_session=True, preexec_fn=_die_with_parent,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            except Exception as e:
                self._note(f"couldn't start {self.d.worker}: {e}")
                self.proc = None
            else:
                self.state, self.started_at = "running", began
                self._note(f"started {self.d.worker} (pid {self.proc.pid})")
                async for raw in self.proc.stdout:
                    self._note(raw.decode("utf-8", "replace").rstrip())
                code = await self.proc.wait()
                self.last_exit = {"code": code, "at": time.time()}
                if not self.wanted:
                    break
                self._note(f"{self.d.worker} stopped (exit {code}); starting it again in {pause} s")
            if not self.wanted:
                break
            self.state = "restarting"
            self.restarts += 1
            await asyncio.sleep(pause)
            pause = 2 if time.time() - began > 120 else min(60, pause * 2)
        self.state = "off"


class ForecastWorker(Worker):
    """polar_dashboard/forecast_service.py: the Heart page's live forecasts (ECG, accelerometer, R-R) and the
    Lab's model API, for the first person's strap. Started with the Heart worker when a model is in
    polar_dashboard/forecast/models. It needs PyTorch and transformers in python() (tools/orin_setup.sh sets
    that up on a Jetson); without them it says so and idles. BIODASH_FORECAST=0 leaves it off."""

    FOLDER = os.path.join(HERE, "polar_dashboard")

    def __init__(self):
        super().__init__(types.SimpleNamespace(key="forecast", worker="forecast_service.py", folder=self.FOLDER, env=dict))

    @classmethod
    def wanted_here(cls):
        if os.environ.get("BIODASH_FORECAST", "1") == "0" or not enabled("heart"):
            return False
        if DEMO:
            return True                       # made-up forecasts, so the overlays can be tried
        base = os.path.join(cls.FOLDER, "forecast", "models")
        return any(os.path.isfile(os.path.join(base, m, w))
                   for m in ("ecg", "ecg_acc", "ecg_rr") for w in ("pytorch_model.bin", "model.safetensors"))

    @staticmethod
    def python():
        """BIODASH_FORECAST_PYTHON, else the environment tools/orin_setup.sh made, else this Python."""
        venv = os.environ.get("BIODASH_FORECAST_VENV") or os.path.expanduser("~/.local/share/bio-dash/forecast-venv")
        mine = os.path.join(venv, "bin", "python")
        return os.environ.get("BIODASH_FORECAST_PYTHON") or (mine if os.access(mine, os.X_OK) else sys.executable)

    def command(self):
        if DEMO:
            return [sys.executable, "-u", os.path.join(HERE, "tools", "demo_feed.py"), "forecast", self.url()]
        return [self.python(), "-u", self.d.worker, "--url", self.url()]

    @staticmethod
    def url():
        a = ARGS or parse_args([])
        host = "127.0.0.1" if a.host in ("0.0.0.0", "::", "", "localhost") else a.host
        return f"http://{host}:{a.port}/api/d/heart"


FORECAST = None


# ---------------------------------------------------------------------------------- live hub
def _rmssd(ppis):
    if len(ppis) < 3:
        return None
    sq = sum((ppis[i] - ppis[i - 1]) ** 2 for i in range(1, len(ppis)))
    return round((sq / (len(ppis) - 1)) ** 0.5, 1)


class Tap:
    """Sits in a device API's websocket list like a browser would, and hands what it's sent to the hub."""

    def __init__(self, hub, key):
        self.hub, self.key = hub, key

    async def send_text(self, msg):
        self.hub.on_stream(self.key, msg)


class Hub:
    """One websocket per page. Every page gets status changes and a once-a-second summary of
    the latest readings; a page also gets a device's full stream (ECG ...) while it shows it."""

    def __init__(self):
        self.clients = {}                     # websocket -> {"subs": set(), "queue": asyncio.Queue}
        self.taps = {}
        self.latest = {k: {} for k in INSTANCES}
        self.status = {}
        self._status_sig = {}
        self._ppi = collections.defaultdict(lambda: collections.deque(maxlen=20))
        self._o2_mtime = {}
        self._water_at = 0.0
        self._radios_at = 0.0

    # ---- streams coming from the device APIs
    def on_stream(self, key, msg):
        latest = self.latest[key]
        try:
            kind = INSTANCES[key].kind.key
            if kind == "heart":
                if msg.startswith('{"type":"ppi"'):
                    rows = json.loads(msg)["rows"]
                    for ts, ppi, hr in rows:
                        self._ppi[key].append(ppi)
                        if hr > 0:
                            latest.update(hr=hr, ts=ts)
                    latest["rmssd"] = _rmssd(list(self._ppi[key]))
                    latest["rr"] = rows[-1][1]
                elif msg.startswith('{"type":"ecg"'):
                    latest["ecg_at"] = time.time()
            elif kind == "air":
                rows = json.loads(msg).get("rows") or []
                if rows:
                    self.latest[key] = {"ts": rows[-1]["ts"], "v": rows[-1]["v"]}
            elif kind == "brain":
                if msg.startswith('{"type":"heart"'):
                    ts, _ibi, hr = json.loads(msg)["rows"][-1][:3]
                    latest.update(hr=hr, ts=ts)
                elif msg.startswith('{"type":"battery"'):
                    latest["battery"] = json.loads(msg)["rows"][-1][1]
                elif msg.startswith('{"type":"bands"'):
                    r = json.loads(msg)["rows"][-1]
                    latest.update(ts=r[0], alpha=_alpha_share(r), contact=r[21:25])
                elif msg.startswith('{"type":"eeg"'):
                    latest["eeg_at"] = time.time()
        except Exception:
            pass
        self.send(key, '{"t":"live","d":"%s","m":%s}' % (key, msg))

    def send(self, key, text):
        for ws, c in list(self.clients.items()):
            if key is None or key in c["subs"]:
                q = c["queue"]
                if q.qsize() < 400:                # a slow phone drops frames rather than piling up
                    q.put_nowait(text)

    def broadcast(self, obj):
        self.send(None, json.dumps(obj, separators=(",", ":")))

    def attach_taps(self):
        """Taps only while a page is open, so the device APIs don't parse streams nobody sees."""
        want = bool(self.clients)
        for inst in INSTANCES.values():
            mgr = getattr(inst.api, "manager", None) if inst.api else None
            if mgr is None:
                continue
            tap = self.taps.setdefault(inst.key, Tap(self, inst.key))
            if want and enabled(inst.key):
                mgr.active_connections.add(tap)
            else:
                mgr.active_connections.discard(tap)

    # ---- status of every device, once a second
    def compact_status(self, inst):
        key, kind = inst.key, inst.kind.key
        st = {}
        if inst.api is not None:
            try:
                st = inst.api.get_status() if hasattr(inst.api, "get_status") else inst.api.status()
            except Exception as e:
                st = {"state": "scanning", "error": str(e)}
        dbg = st.get("debug") or {}
        out = {k: st.get(k) for k in ("state", "address", "attempt", "max_attempts", "detail")}
        dev = dbg.get("device") or {}
        out["name"] = dev.get("name")
        out["rssi"] = dev.get("rssi")
        radio = dbg.get("radio") or {}
        out["radio"] = {"config": radio.get("config"), "hci": radio.get("hci"), "reason": radio.get("reason"),
                        "avoided": dbg.get("radio_avoided") or []}
        out["last_device"] = st.get("last_device")
        out["held"] = bool(st.get("auto_held"))             # disconnected on purpose: waits for a pick
        out["worker"] = inst.proc.info() if inst.proc else {"state": "off"}
        out["enabled"] = enabled(key)
        out["warning"] = dbg.get("warning")                 # something the worker wants you to know
        if kind == "water":
            live = st.get("live") or {}
            out["live"] = {k: live.get(k) for k in ("fill_ml", "fill_pct", "battery", "last_sip", "cap_open",
                                                    "fill_from", "capacity_bottle", "can_glow", "can_upload_glow",
                                                    "glow_upload", "synced_at", "reminders_set", "firmware", "family")}
            # waiting for "It's full now" after "It's empty now" (from the saved settings, so it shows when disconnected too)
            out["live"]["cal_pending"] = (st.get("settings") or {}).get("cal_pending")
            self.latest[key].update({k: live.get(k) for k in ("fill_ml", "fill_pct", "battery")})
            if live.get("last_sip"):
                self.latest[key]["last_sip"] = live["last_sip"]
        elif kind == "oxygen":
            out["detail"] = (st.get("live") or {}).get("status") or out.get("detail")
        elif kind == "air":
            parser = dbg.get("parser") or {}
            out["profile"] = parser.get("effective_profile") or parser.get("profile")
        elif kind == "brain":
            out["signals"] = dbg.get("signals")
            out["layout"] = dbg.get("layout")
            out["preset"] = dbg.get("preset")
            out["firmware"] = (dbg.get("info") or {}).get("fw")
        return out

    async def status_loop(self):
        while True:
            for inst in active():
                st = self.compact_status(inst)
                sig = json.dumps(st, sort_keys=True, default=str)
                self.status[inst.key] = st
                if sig != self._status_sig.get(inst.key):
                    self._status_sig[inst.key] = sig
                    self.broadcast({"t": "status", "d": inst.key, "s": st})
            now = time.time()
            water = INSTANCES["water"]
            if enabled("water") and water.api and now - self._water_at > 15:
                self._water_at = now
                try:
                    s = await run_in_threadpool(water.api.summary, 1)
                    self.latest["water"].update(today_ml=s["today_ml"], goal_ml=s["goal_ml"], sips_today=s["sips_today"])
                    if s.get("last_sip") and not self.latest["water"].get("last_sip"):
                        self.latest["water"]["last_sip"] = s["last_sip"]
                except Exception:
                    pass
            if now - self._radios_at > 10:
                self._radios_at = now
                with contextlib.suppress(Exception):
                    await share_radios()
            if self.clients:
                self.broadcast({"t": "latest", "v": {i.key: self.latest[i.key] for i in active()}, "at": now})
            await asyncio.sleep(1)

    async def oxygen_loop(self):
        """An O2 ring's worker rewrites data.json at each reading: pass each new one on."""
        while True:
            await asyncio.sleep(0.25)
            for inst in active():
                if inst.kind.key != "oxygen" or not enabled(inst.key):
                    continue
                try:
                    path = os.path.join(inst.state_dir, "data.json")
                    m = os.stat(path).st_mtime_ns
                    if m == self._o2_mtime.get(inst.key):
                        continue
                    self._o2_mtime[inst.key] = m
                    with open(path) as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    continue
                data["ts"] = int(time.time() * 1000)
                self.latest[inst.key] = data
                self.send(inst.key, json.dumps({"t": "live", "d": inst.key, "m": data}, separators=(",", ":")))


def _alpha_share(row):
    """Alpha's share (%) of the 1-44 Hz power, mean of the EEG channels in contact (a `bands` row)."""
    shares = []
    for c in range(4):
        if row[21 + c] in (1, 2):
            p = [10 ** v for v in row[1 + 5 * c:6 + 5 * c]]
            shares.append(100 * p[2] / sum(p))
    return round(sum(shares) / len(shares), 1) if shares else None


HUB = Hub()


# ---------------------------------------------------------------------------------- people
async def share_radios():
    """With two or more people, give each strap its own radio where there are enough (a streaming H10 can
    monopolise one): straps take the USB dongles first, rings start from the other end. Each worker reads
    its share from its radio_config.json and follows it unless you pinned a radio (bt_debug.RadioPin).
    With one person nothing is shared out and "auto" works as it always has."""
    if DEMO:
        return
    worn = [i for i in active() if i.kind.personal and enabled(i.key)]
    several = len({i.person for i in worn}) > 1
    order = []
    if several:
        radios = [a for a in await bt_debug.list_adapters()
                  if a.get("address") and a.get("powered", True) and not a.get("error")]
        order = sorted(radios, key=lambda a: a.get("placement") != "external")       # dongles first
    for kind, ranked in (("heart", order), ("oxygen", order[::-1])):
        for n, inst in enumerate(i for i in worn if i.kind.key == kind):
            share = ranked[n % len(ranked)]["address"] if ranked else None
            bt_debug.RadioPin(os.path.join(inst.state_dir, RADIO_FILE)).set_share(share)
    if not several:
        for inst in INSTANCES.values():
            if inst.kind.personal and inst not in worn and os.path.isdir(inst.state_dir):
                bt_debug.RadioPin(os.path.join(inst.state_dir, RADIO_FILE)).set_share(None)


RADIO_FILE = "radio_config.json"


async def release(inst):
    """Make this instance let go of its device: forget it and start the worker afresh (it then only lists what's nearby)."""
    ld = inst.last_device()
    if ld is not None:
        ld.forget()
    if inst.proc and enabled(inst.key):
        await inst.proc.restart()


def people_view():
    out = []
    for p in CONFIG["people"]:
        devs = {}
        for i in INSTANCES.values():
            if i.kind.personal and i.person == p["id"]:
                ld = (i.last_device().load() if i.last_device() else None) or {}
                devs[i.kind.key] = {"key": i.key, "enabled": enabled(i.key),          # unit: its recordings folder
                                    "unit": bt_debug.device_folder(ld.get("name"), ld["address"]) if ld.get("address") else None}
        out.append({"id": p["id"], "name": p.get("name") or "", "devices": devs})
    return out


# ---------------------------------------------------------------------------------- keep awake
def take_inhibitor():
    """Block suspend / idle sleep while Bio-dash runs (lid-close too with BIODASH_INHIBIT_LID=1),
    by holding a systemd-inhibit lock for as long as this process lives. BIODASH_INHIBIT=0: off."""
    import subprocess
    if os.environ.get("BIODASH_INHIBIT", "1") == "0" or DEMO:
        return None
    if not shutil.which("systemd-inhibit"):
        print("⚠ systemd-inhibit isn't available, so the computer may sleep while recording.", flush=True)
        return None
    what = "sleep:idle" + (":handle-lid-switch" if os.environ.get("BIODASH_INHIBIT_LID") == "1" else "")
    p = subprocess.Popen(["systemd-inhibit", f"--what={what}", "--mode=block", "--who=Bio-dash",
                          "--why=Recording live sensor data", "sleep", "infinity"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(0.4)
    if p.poll() is not None:
        err = (p.stderr.read() or b"").decode(errors="replace").strip()[:200]
        print(f"⚠ Couldn't take a sleep inhibitor ({err or 'not permitted'}); the computer may sleep.", flush=True)
        return None
    print(f"☀ Keeping this computer awake while Bio-dash runs ({what}).", flush=True)
    return p


# ---------------------------------------------------------------------------------- the app
@contextlib.asynccontextmanager
async def lifespan(app):
    inhibitor = take_inhibitor()
    global ANALYSIS, FORECAST
    ANALYSIS = load_module("biodash_analysis_api", ANALYSIS_FOLDER)
    app.mount("/api/d/analysis", ANALYSIS.app)
    for inst in active():
        await inst.load()
        if enabled(inst.key):
            inst.proc.start()
    tasks = [asyncio.create_task(HUB.status_loop()), asyncio.create_task(HUB.oxygen_loop())]
    if ForecastWorker.wanted_here():
        FORECAST = ForecastWorker()
        FORECAST.start()
    print(f"Bio-dash V{VERSION} is up: http://{hostname()}:{ARGS.port}"
          + ("  (demo: made-up readings)" if DEMO else ""), flush=True)
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*(i.unload() for i in INSTANCES.values()),
                             *([FORECAST.stop()] if FORECAST else []), return_exceptions=True)
        if inhibitor:
            inhibitor.terminate()
        ANALYSIS._stop_pool()                           # its worker process, if it's running


def hostname():
    import socket
    return socket.gethostname()


app = FastAPI(lifespan=lifespan, title="Bio-dash", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(GZipMiddleware, minimum_size=2048)      # history JSON shrinks ~20x for phones on Wi-Fi


@app.middleware("http")
async def proxied_is_not_local(request, call_next):
    """Behind a proxy on this computer (tailscale serve) every request comes from 127.0.0.1: use the forwarded
    address, so "this computer only" (power, updates, restart, the forecast service) still means this computer."""
    h = request.headers
    if h.get("tailscale-user-login") or h.get("x-forwarded-for") or h.get("forwarded"):
        first = (h.get("x-forwarded-for") or "").split(",")[0].strip()
        request.scope["client"] = (first if first and first not in ("127.0.0.1", "::1") else "proxied", 0)
    return await call_next(request)


@app.exception_handler(Exception)
async def explain_errors(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


def client_host(request):
    return request.client.host if request.client else ""


# ---- the app's own API
@app.get("/api/app")
def app_info(request: Request):
    return {"version": VERSION, "demo": DEMO, "setup_done": CONFIG["setup_done"], "host": hostname(),
            "data_root": bt_debug.data_root(), "local": bt_debug._power_allowed(client_host(request))[0],
            "devices": {k.key: {"label": k.label, "product": k.product, "personal": k.personal, "enabled": enabled(k.key),
                                "status": HUB.status.get(k.key)} for k in KINDS.values()},
            "instances": {i.key: {"kind": i.kind.key, "person": i.person, "enabled": enabled(i.key)} for i in active()},
            "people": people_view(), "max_people": len(SLOTS),
            "forecast": FORECAST is not None}


class DeviceToggle(BaseModel):
    device: str
    enabled: bool


async def apply(inst):
    """Start or stop an instance's worker to match the settings."""
    if enabled(inst.key):
        await inst.load()
        inst.proc.start()
    elif inst.proc:
        await inst.proc.stop()


@app.post("/api/app/device")
async def toggle_device(req: DeviceToggle):
    inst = INSTANCES.get(req.device)
    if inst is None or inst not in active():
        return JSONResponse({"error": "unknown device"}, status_code=404)
    if inst.person in (None, SLOTS[0]):
        CONFIG["devices"][inst.kind.key] = req.enabled
    else:
        person(inst.person).setdefault("devices", {})[inst.kind.key] = req.enabled
    write_config(CONFIG)
    await apply(inst)
    HUB.attach_taps()
    return {"ok": True, "device": inst.key, "enabled": req.enabled}


class SetupRequest(BaseModel):
    devices: list[str] = []
    done: bool = True


@app.post("/api/app/setup")
async def setup(req: SetupRequest):
    """The first-run choice: which kinds of device you have (the first person's, for the worn ones)."""
    for k in KINDS:
        CONFIG["devices"][k] = k in req.devices
        await apply(INSTANCES[k])
    CONFIG["setup_done"] = req.done
    write_config(CONFIG)
    HUB.attach_taps()
    return {"ok": True}


class PersonRequest(BaseModel):
    name: str = ""


def _clean_name(name):
    return " ".join((name or "").split())[:24]


@app.post("/api/app/people")
async def add_person(req: PersonRequest):
    """One more person wearing a strap and / or a ring (up to three)."""
    free = [s for s in SLOTS if not person(s)]
    if not free:
        return JSONResponse({"error": f"Bio-dash follows up to {len(SLOTS)} people at once"}, status_code=409)
    pid = free[0]
    CONFIG["people"].append({"id": pid, "name": _clean_name(req.name), "devices": {k.key: True for k in KINDS.values() if k.personal}})
    CONFIG["people"].sort(key=lambda p: SLOTS.index(p["id"]))
    write_config(CONFIG)
    for inst in active():
        if inst.person == pid:
            await apply(inst)
    HUB.attach_taps()
    return {"ok": True, "id": pid, "people": people_view()}


@app.post("/api/app/people/{pid}")
async def rename_person(pid: str, req: PersonRequest):
    p = person(pid)
    if p is None:
        return JSONResponse({"error": "no such person"}, status_code=404)
    p["name"] = _clean_name(req.name)
    write_config(CONFIG)
    return {"ok": True, "people": people_view()}


@app.delete("/api/app/people/{pid}")
async def remove_person(pid: str):
    """Stops following this person: their workers stop and their devices are forgotten. Recordings stay."""
    p = person(pid)
    if p is None or pid == SLOTS[0]:
        return JSONResponse({"error": "the first person stays (turn their devices off instead)"}, status_code=409)
    mine = [i for i in INSTANCES.values() if i.person == pid]
    CONFIG["people"].remove(p)
    write_config(CONFIG)
    for inst in mine:
        if inst.proc:
            await inst.proc.stop()
        for root, _dirs, files in os.walk(inst.state_dir):          # last device, radio pin, status: a later person starts clean
            for name in files:
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(root, name))
        HUB.status.pop(inst.key, None)
        HUB._status_sig.pop(inst.key, None)
        HUB.latest[inst.key] = {}
    HUB.attach_taps()
    return {"ok": True, "people": people_view()}


class ClaimRequest(BaseModel):
    device: str
    address: str


@app.post("/api/app/claim")
async def claim(req: ClaimRequest):
    """Before connecting `address` for one person: a strap or ring belongs to one person at a time, so whoever
    else had it (connected, or saved for auto-reconnect) lets go of it."""
    inst = INSTANCES.get(req.device)
    if inst is None:
        return JSONResponse({"error": "unknown device"}, status_code=404)
    addr, moved = req.address.upper(), []
    for other in active():
        if other is inst or other.kind is not inst.kind or other.api is None:
            continue
        ld = other.last_device()
        saved = ((ld.load() or {}).get("address") or "").upper() if ld else ""
        using = ((HUB.status.get(other.key) or {}).get("address") or "").upper()
        if addr in (saved, using):
            await release(other)
            moved.append(other.key)
    return {"ok": True, "released": moved}


def running(key):
    inst = INSTANCES.get(key)
    return inst if inst is not None and inst.proc is not None else None


@app.post("/api/app/restart/{key}")
async def restart_worker(key: str):
    inst = running(key)
    if inst is None or not enabled(key):
        return JSONResponse({"error": "that device is off"}, status_code=409)
    await inst.proc.restart()
    return {"ok": True}


@app.get("/api/app/log/{key}")
def worker_log(key: str, since: float = 0):
    proc = FORECAST if key == "forecast" else getattr(running(key), "proc", None)
    if proc is None:
        return {"lines": [], "worker": {"state": "off"}}
    return {"lines": [l for l in proc.log if l[0] > since], "worker": proc.info()}


@app.get("/api/app/device-names")
def device_names_get():
    """Names you gave devices ("Mine", "Boss's"), by the name they advertise (or their address)."""
    return {"names": bt_debug.device_names()}


@app.post("/api/app/device-names")
async def device_names_set(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    code, out = bt_debug.set_device_name(body.get("key"), body.get("name"))
    return JSONResponse(out, status_code=code)


@app.get("/api/app/radios")
async def radios():
    """Every Bluetooth radio, and which device uses which."""
    adapters = await bt_debug.list_adapters()
    use = {i.key: (HUB.status.get(i.key) or {}).get("radio") for i in active() if enabled(i.key)}
    return {"adapters": adapters, "devices": use}


# ---- shutdown / reboot / updates (this computer only, unless BIODASH_ALLOW_POWER=1)
@app.get("/api/system/power")
def power_info(request: Request):
    return bt_debug.power_info(client_host(request))


@app.post("/api/system/power")
async def power(request: Request):
    try:
        action = (await request.json()).get("action")
    except Exception:
        action = None
    code, out = bt_debug.system_power(action, client_host(request), request.headers.get(bt_debug.POWER_HEADER))
    return JSONResponse(out, status_code=code)


@app.get("/api/system/update")
async def update_info(request: Request, check: int = 0):
    return await run_in_threadpool(bt_debug.update_info, client_host(request), bool(check))


@app.post("/api/system/update")
async def update(request: Request):
    try:
        action = (await request.json()).get("action")
    except Exception:
        action = None
    code, out = await run_in_threadpool(bt_debug.system_update, action, client_host(request),
                                        request.headers.get(bt_debug.POWER_HEADER))
    return JSONResponse(out, status_code=code)


@app.get("/api/system/restart")
def restart_info(request: Request):
    return bt_debug.restart_info(client_host(request))


@app.post("/api/system/restart")
async def restart_all(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    code, out = await run_in_threadpool(bt_debug.system_restart, body if isinstance(body, dict) else {},
                                        client_host(request), request.headers.get(bt_debug.POWER_HEADER))
    return JSONResponse(out, status_code=code)


# ---- the one websocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    client = {"subs": set(), "queue": asyncio.Queue()}
    HUB.clients[ws] = client
    HUB.attach_taps()
    await ws.send_text(json.dumps({"t": "hello", "version": VERSION, "status": HUB.status, "latest": HUB.latest},
                                  separators=(",", ":")))

    async def pump():
        while True:
            text = await client["queue"].get()
            await ws.send_text(text)

    sender = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if "sub" in msg:                       # {"sub": ["heart"]}: the devices this page shows in full
                client["subs"] = {k for k in msg["sub"] if k in INSTANCES}
    except (WebSocketDisconnect, ValueError, RuntimeError):
        pass
    finally:
        sender.cancel()
        HUB.clients.pop(ws, None)
        HUB.attach_taps()


# ---- the page
NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/")
def index():
    html = open(os.path.join(WEB, "index.html")).read().replace("{{V}}", VERSION)
    return Response(html, media_type="text/html", headers=NO_CACHE)


@app.get("/lab")
def lab():
    """stream-lane-lab's viewer (ReaLLM-Forge PR #924) for the H10 models, with the live strap as a source."""
    html = open(os.path.join(WEB, "lab.html")).read().replace("{{V}}", VERSION)
    return Response(html, media_type="text/html", headers=NO_CACHE)


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(os.path.join(WEB, "manifest.webmanifest"), media_type="application/manifest+json")


class CachedStatic(StaticFiles):
    """The browser keeps the files but checks they're current each load (a quick 304 when they
    are), so an update shows up straight away without anyone clearing a cache."""

    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


app.mount("/static", CachedStatic(directory=WEB), name="static")


# ---------------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Bio-dash: one app for every device.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("BIODASH_PORT", 5000)))
    ap.add_argument("--host", default=os.environ.get("BIODASH_HOST", "0.0.0.0"))
    ap.add_argument("--demo", action="store_true", help="made-up readings instead of Bluetooth (try the app out)")
    return ap.parse_args(argv)


ARGS = parse_args([]) if __name__ != "__main__" else None

if __name__ == "__main__":
    ARGS = parse_args()
    if ARGS.demo and not DEMO:
        # demo: fresh made-up recordings in a scratch folder; your real ones aren't touched
        import subprocess
        import tempfile
        demo = os.path.join(tempfile.gettempdir(), "bio-dash-demo")
        os.environ["BIODASH_DEMO"] = "1"
        os.environ["BIODASH_POWER_DRY_RUN"] = "1"                     # the demo never really shuts down
        os.environ.setdefault("BIODASH_DATA_DIR", os.path.join(demo, "Bio-dash"))
        os.environ.setdefault("BIODASH_APP_CONFIG", os.path.join(demo, "app.json"))
        os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(demo, "config"))
        if not os.path.isdir(os.environ["BIODASH_DATA_DIR"]):          # a few past days for Analysis
            subprocess.run([sys.executable, os.path.join(ANALYSIS_FOLDER, "sample_data.py"),
                            os.environ["BIODASH_DATA_DIR"], "--days", "6"], check=False)
        os.execv(sys.executable, [sys.executable, *sys.argv])
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning", ws_ping_interval=20)
