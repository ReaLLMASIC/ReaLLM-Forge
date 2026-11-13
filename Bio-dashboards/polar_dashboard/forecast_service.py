#!/usr/bin/env python3
"""Live ECG / accelerometer / R-R forecasts for Polar-dash (Bio-dash V6.3).

Runs next to the server as its own process, so a crash or a GPU stall can't freeze the dashboard and torch
stays out of the web server:

  * follows the session the Bluetooth worker is recording
    (<Documents>/Bio-dash/Polar H10/<device>/<date>/<time>_{ecg,acc,rr}.csv) and tails the accelerometer,
    ECG and R-R files, accelerometer first;
  * builds model rows with waves.LiveAligner, exactly like the training data (the 125 Hz grid; the settings
    come from each model folder's config.json);
  * every refresh (0.5 s) and for each overlay that is switched on, forecasts the next second as 8 samples
    with the fast loop (KV cache + CUDA graph) and POSTs the band, the median and one representative sample
    to the Heart API's /api/forecast/live (api.py), which passes them on to the open pages;
  * answers stream-lane-lab's API (ReaLLM-Forge PR #924: GET /api/models, POST /api/forecast, GET /api/signal,
    plus GET /rows: the strap's latest rows for the Lab viewer) on
    127.0.0.1:5011, which api.py passes on;
  * sends a heartbeat once a second (state, device, timings) and gets the overlay switches back.

Three models: forecast/models/ecg (ECG only; the ECG overlay), forecast/models/ecg_acc (ECG +
accelerometer; the accelerometer overlay) and forecast/models/ecg_rr (ECG + R-R, a proof of concept; the R-R
overlay, with the R-R lane rebuilt live from the strap's R-R messages by rr_lane.py). Each is loaded the first
time its overlay is switched on.
Without torch / transformers / the model folders the service keeps running and reports why, so the
dashboard can show it. Run the models on a GPU with bfloat16 autocast, as trained (CPU works, slowly).

    python3 forecast_service.py [--url http://127.0.0.1:5000/api/d/heart] [--models forecast/models] [--device auto]
    python3 forecast_service.py --bench          (no dashboard: load the models, time forecasts, GPU memory)
    python3 forecast_service.py --session <.../HH-MM-SS_ecg.csv> --record contexts.jsonl --seed 1   (checks)
"""
import argparse
import bisect
import collections
import csv
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FC_DIR = os.path.join(HERE, "forecast")
for p in (HERE, os.path.join(HERE, "..", "analysis_dashboard"), FC_DIR, os.path.dirname(HERE)):   # last: the shared bt_debug.py
    if p not in sys.path:
        sys.path.insert(0, p)

import bt_debug  # noqa: E402  (session folders, standard library only)
import rr_lane  # noqa: E402  (the R-R lane, as the ECG + R-R model's training data made it)
import waves  # noqa: E402  (the shared interpolation: the same rows as the training data)

KINDS = ("ecg", "acc", "rr")
DEFAULTS = {"ecg": False, "acc": False, "rr": False, "horizon_s": 1.0, "refresh_s": 0.5, "samples": 8}
RR_BEATS = 3             # next beats read out of an R-R forecast
RR_SCORED = 60           # the R-R overlay's running error: over the last this many beats
STALE_MS = 5000          # no new rows for this long: waiting for data
MIN_CONTEXT = 64         # rows (0.5 s) before the first forecast after a gap
TOP_K, TEMPERATURE = 100, 1.0   # as in the held-out evaluation (calibrated band)
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never via a proxy


def log(msg):
    print(f"[forecast {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def post_json(url, obj, timeout=2.0):
    req = urllib.request.Request(url, data=json.dumps(obj, separators=(",", ":")).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with _opener.open(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body else None


# ------------------------------------------------------------------------------------------------ data
class Tail:
    """New complete lines of a growing CSV file (a half-written last line waits for the next read)."""

    def __init__(self, path, tail_bytes):
        self.path = path
        self.f = open(path, "r", errors="replace")
        self.partial = ""
        size = os.path.getsize(path)
        if size > tail_bytes:            # joined mid-session: start a few seconds back, not at the top
            self.f.seek(size - tail_bytes)
            self.f.readline()

    def lines(self):
        chunk = self.f.read()
        if not chunk:
            return []
        parts = (self.partial + chunk).split("\n")
        self.partial = parts.pop()
        return [s.strip() for s in parts if s.strip()]

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def parse(line, n):
    """'ts,a,b[,c]' -> [int ts, floats...] or None (header, NUL bytes, garbage)."""
    parts = line.split(",")
    if len(parts) < n:
        return None
    try:
        return [int(parts[0])] + [float(x) for x in parts[1:n]]
    except ValueError:
        return None


def get_json(url, timeout=2.0):
    with _opener.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


def newest_session(fixed=None, url=None):
    """(ecg csv, acc csv or None, rr csv or None) of the session being recorded: the one the Heart API says its
    worker is writing (so, with several people recording, the first person's), else the newest ECG file."""
    ecg = fixed
    if not ecg and url:
        try:
            ecg = (get_json(url + "/api/session") or {}).get("ecg")
        except (urllib.error.URLError, OSError, ValueError, AttributeError):
            ecg = None
        if not ecg:
            return None, None, None
    if not ecg:
        files = bt_debug.latest_session_files("Polar H10", "ecg", limit=1)
        if not files:
            return None, None, None
        ecg = files[0]
    stem = ecg[: -len("_ecg.csv")] if ecg.endswith("_ecg.csv") else None
    acc, rr = (stem + "_acc.csv", stem + "_rr.csv") if stem else (None, None)
    return ecg, (acc if acc and os.path.exists(acc) else None), (rr if rr and os.path.exists(rr) else None)


def to_real(lane, v):
    """Integer lane value -> mV / mg (the lane's decode rule: value * step / scale)."""
    return v * float(lane.get("step", 1)) / float(lane.get("scale") or 1)


# ---- picking the line drawn by the heart's rhythm
# The model tends to put the next R peak early (tools/forecast_timing.py: ~100 ms at 0.2-0.5 s ahead, Oct 22).
# Of the samples it drew, the one whose next R peak falls where the recent beat-to-beat interval says it should
# is drawn instead of the one closest to the median. BIODASH_FORECAST_REP=median goes back to the old choice.
REP_PICK = os.environ.get("BIODASH_FORECAST_REP", "rhythm")
R_HALF = 12               # rows each side of the moving mean taken off before looking for R peaks (~200 ms)
R_REFRACTORY = 31         # rows (~250 ms): two R peaks are never closer


def _highpass(x):
    n, out = len(x), []
    for i in range(n):
        a, b = max(0, i - R_HALF), min(n, i + R_HALF + 1)
        out.append(x[i] - sum(x[a:b]) / (b - a))
    return out


def _peaks(hp, thr, lo=1, hi=None):
    """Indices of R peaks: local maxima above thr, the tallest first within R_REFRACTORY rows."""
    hi = len(hp) - 1 if hi is None else hi
    cand = sorted((i for i in range(max(1, lo), hi) if hp[i] > thr and hp[i] >= hp[i - 1] and hp[i] > hp[i + 1]),
                  key=lambda i: -hp[i])
    taken = []
    for i in cand:
        if all(abs(i - j) >= R_REFRACTORY for j in taken):
            taken.append(i)
    return sorted(taken)


def rhythm_pick(ctx, samples):
    """ctx: the context's ECG (one lane, oldest first, on the model's grid); samples: N lists of the forecast ECG
    (the rows after ctx). -> the index of the sample whose first R peak lands nearest the next beat expected from
    the context's own beat-to-beat intervals (median of the last three), or None when the context has fewer than
    two clear R peaks."""
    T = len(ctx)
    hp = _highpass(ctx)
    top = max(hp) if hp else 0
    if top <= 0:
        return None
    pk = _peaks(hp, 0.5 * top, hi=T - 3)                 # a peak in the last rows may still be rising
    if len(pk) < 2:
        return None
    rr = statistics.median([b - a for a, b in zip(pk, pk[1:])][-3:])
    if not R_REFRACTORY <= rr <= 250:                     # 30-240 bpm on an 8 ms grid
        return None
    expect = max(1.0, pk[-1] + rr - (T - 1))              # rows after the last context row (overdue: next row)
    thr = 0.5 * statistics.median(hp[i] for i in pk)
    best = None
    for n, s in enumerate(samples):
        hs = _highpass(ctx[-R_HALF:] + list(s))[R_HALF:]
        first = next((j + 1 for j in _peaks(hs, thr) if j + 1 > max(0, (pk[-1] + R_REFRACTORY) - (T - 1))), None)
        when = first if first is not None else len(s) + rr / 2  # no beat in the horizon: later than its end
        score = abs(when - expect)
        if best is None or score < best[0]:
            best = (score, n)
    return best[1]


def summarize(samples, lanes, prefer=None):
    """samples: N x K x H integer values -> median, band (lo / hi) and the representative sample (`prefer` if
    given, else the one closest to the median; averaging samples would smear a heartbeat whose timing is
    uncertain)."""
    N, K, H = len(samples), len(samples[0]), len(samples[0][0])
    real = [[[to_real(lanes[k], samples[n][k][h]) for h in range(H)] for k in range(K)] for n in range(N)]
    med = [[statistics.median(real[n][k][h] for n in range(N)) for h in range(H)] for k in range(K)]
    lo = [[min(real[n][k][h] for n in range(N)) for h in range(H)] for k in range(K)]
    hi = [[max(real[n][k][h] for n in range(N)) for h in range(H)] for k in range(K)]
    spread = [sum(hi[k][h] - lo[k][h] for h in range(H)) / H + 1e-9 for k in range(K)]
    dist = [sum(sum(abs(real[n][k][h] - med[k][h]) for h in range(H)) / spread[k] for k in range(K)) for n in range(N)]
    rep = prefer if prefer is not None and 0 <= prefer < N else min(range(N), key=dist.__getitem__)
    dec = [3 if lane.get("unit") == "mV" else 1 for lane in lanes]

    def rnd(rows):
        return [[round(v, d + 1) for v in row] for row, d in zip(rows, dec)]
    return {"median": rnd(med), "rep": rnd(real[rep]), "lo": rnd(lo), "hi": rnd(hi), "rep_index": rep}


def has_weights(d):
    return any(os.path.exists(os.path.join(d, w)) for w in ("model.safetensors", "pytorch_model.bin"))


def weights_elsewhere(model_dir, cfg):
    """Before V6.0 the weights stayed out of git, so a release folder could start without them: use the same model's (an
    identical config.json) from another release folder next to this one (<Bio-dash>/V*_Dashboard/...),
    newest first. tools/orin_setup.sh --models copies them in for good."""
    rel = os.path.relpath(os.path.abspath(model_dir), os.path.dirname(HERE))    # within the release folder
    top = os.path.dirname(os.path.dirname(HERE))                                # the folder holding V*_Dashboard
    keep = lambda c: {k: v for k, v in c.items() if k not in ("transformers_version", "torch_dtype", "dtype")}
    here = os.path.basename(os.path.dirname(HERE))

    def version(name):
        try:
            return tuple(int(x) for x in name[1:].split("_")[0].split("."))
        except ValueError:
            return ()
    try:
        names = sorted((n for n in os.listdir(top) if n.startswith("V") and n.endswith("_Dashboard") and n != here),
                       key=version, reverse=True)
    except OSError:
        return None
    for name in names:
        d = os.path.join(top, name, rel)
        try:
            if has_weights(d) and keep(json.load(open(os.path.join(d, "config.json")))) == keep(cfg or {}):
                return d
        except (OSError, ValueError):
            continue
    return None


API_MAX_HORIZON = 128     # rows, as PR #924's API


def start_api(svc, port):
    """The PR #924 API for api.py to pass on: GET /models, POST /forecast on 127.0.0.1 only."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            from urllib.parse import parse_qs, urlsplit
            u = urlsplit(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            path = u.path.rstrip("/")
            try:
                if path == "/models":
                    return self._send(200, svc.api_models())
                if path == "/rows":
                    return self._send(*svc.api_rows(q))
                if path == "/signal":
                    return self._send(*svc.api_signal(q))
            except Exception as e:     # noqa: BLE001  (report, keep serving)
                return self._send(500, {"detail": f"{type(e).__name__}: {e}"})
            self._send(404, {"detail": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/forecast":
                return self._send(404, {"detail": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(min(n, 8 << 20)) or b"null")
            except (ValueError, OSError):
                return self._send(422, {"detail": "bad JSON"})
            try:
                code, body = svc.api_forecast(req)
            except Exception as e:     # noqa: BLE001  (report, keep serving)
                code, body = 500, {"detail": f"{type(e).__name__}: {e}"}
            self._send(code, body)

        def log_message(self, *_):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        log(f"API: can't listen on 127.0.0.1:{port} ({e}); /api/models and /api/forecast are off")
        return None
    threading.Thread(target=server.serve_forever, name="api", daemon=True).start()
    log(f"API (stream-lane-lab contract) on 127.0.0.1:{port}")
    return server


class ForecastLog:
    """<data>/Forecasts/<day>/<HH-MM-SS>_forecast_<kind>.csv for the session being followed (_replay added for a
    replayed one): one row per grid instant, the forecast the dashboard showed for it -- the representative
    sample (the line drawn) and the median and band of the samples -- with how far ahead it was forecast
    (lead_ms: from the last row of context). Raw units (mV / mg), not high-passed. Times are the strap's,
    as in the session's own files. Forecasts/<day>/ is outside the <dashboard>/<device>/<day>/ layout, so
    Analysis, the exporters and fleet.py leave it alone."""

    def __init__(self, session, ov):
        folder, name = os.path.split(session)
        stamp, day = name[:8], os.path.basename(folder)
        replay = os.path.exists(os.path.join(folder, stamp + "_replay_info.json"))
        out = os.path.join(bt_debug.data_root(), "Forecasts", day)
        os.makedirs(out, exist_ok=True)
        self.path = os.path.join(out, f"{stamp}_forecast_{ov.kind}{'_replay' if replay else ''}.csv")
        self.session = session
        names = [f"{ln['column'].rsplit('_', 1)[0]}_{ln.get('unit') or 'value'}" for ln in ov.lanes]
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self.f = open(self.path, "a", newline="", buffering=1)
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["t_ms", "lead_ms"] + [c for n in names for c in (n, n + "_median", n + "_lo", n + "_hi")])

    def write(self, prev, t0_new):
        K = len(prev["rep"])
        for h in range(len(prev["rep"][0])):
            t = prev["t0"] + h * prev["dt"]
            if t >= t0_new:
                break
            row = [int(round(t)), int(round(t - prev["from"]))]
            for k in range(K):
                row += [prev["rep"][k][h], prev["median"][k][h], prev["lo"][k][h], prev["hi"][k][h]]
            self.w.writerow(row)

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass


def next_beats(times, rr_ctx, samples_rr, beat_t, dt, grid_ms, max_beats=RR_BEATS):
    """The next beats an R-R forecast holds. times / rr_ctx: the context's grid instants and rr_ms lane;
    samples_rr: each sample's forecast rr_ms rows (after the context); beat_t: the newest beat known.
    Read per sample from the row of that beat on (rr_lane.read_beats), then the samples' median per beat while at
    least half of them have it. -> [{"t": ms, "rr": ms, "lo": ms, "hi": ms, "of": samples with it}]."""
    p = bisect.bisect_right(times, beat_t) - 1             # the beat's row (the grid instant at or before it)
    if p < 0:
        return []
    tail = list(rr_ctx[p + 1:])
    per = [rr_lane.read_beats(tail + list(s), -1, max_beats, grid_ms) for s in samples_rr]
    out = []
    for j in range(max_beats):
        got = [b[j] for b in per if len(b) > j]
        if 2 * len(got) < len(per):
            break
        out.append({"t": round(times[p] + statistics.median(g[0] + 1 for g in got) * dt, 1),
                    "rr": round(statistics.median(g[1] for g in got), 1),
                    "lo": min(g[1] for g in got), "hi": max(g[1] for g in got), "of": len(got)})
    return out


class Overlay:
    """One overlay (ecg | acc | rr): its model folder, LiveAligner and context ring. The ring holds the
    LiveAligner's rows (ECG, plus the accelerometer for acc); the rr overlay adds its R-R lane from its beats when
    it makes a context (model_rows), so the lane always uses every beat known by then."""

    def __init__(self, kind, model_dir):
        self.kind, self.model_dir = kind, model_dir
        self.cfg_error = None
        self.cfg = None
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                self.cfg = json.load(f)
            if not has_weights(model_dir):
                other = weights_elsewhere(model_dir, self.cfg)
                if other:
                    log(f"{kind}: no weights in {model_dir}; using the same model's from {other}")
                    self.model_dir = other
                else:
                    self.cfg_error = f"no weights in {model_dir}"
        except OSError:
            self.cfg_error = f"model folder missing: {model_dir}"
        run_file = os.path.join(model_dir, "RUN.txt")
        self.run = open(run_file).read().strip() if os.path.exists(run_file) else os.path.basename(model_dir)
        cfg = self.cfg or {}
        self.lanes = cfg.get("lanes") or [{"column": "ecg_uv", "int_min": -2500, "scale": 1000, "unit": "mV"}]
        self.names = [ln["column"] for ln in self.lanes]
        self.units = [ln.get("unit", "") for ln in self.lanes]
        self.block = int(cfg.get("block_size", 256))
        st = (cfg.get("source") or {}).get("settings") or {"grid_ms": 8, "with_acc": kind == "acc"}
        self.settings = waves.Settings.from_dict(st)
        self.dt_ms = float(self.settings.grid_ms or round(1000.0 / float(cfg.get("sample_rate_hz") or 130), 3))
        self.aligner = waves.LiveAligner(self.settings)
        self.ring = collections.deque(maxlen=self.block)
        self.times = collections.deque(maxlen=self.block)
        self.lock = threading.Lock()     # ring + times, read by the API thread (/rows for the /lab viewer)
        self.piece = 0                   # counts gaps / restarts, so a viewer knows its rows no longer join up
        self.last_t, self.fresh = None, 0
        self.prev = None                 # the last forecast, until the next one says how much of it to keep
        self.log = None                  # ForecastLog for the session being followed
        self.model = self.fc = None
        self.load_error = None
        self.compute_ms, self.count, self.posted = None, 0, 0
        self.beats = rr_lane.LiveBeats() if "rr_ms" in self.names else None
        self.rr_open = {}                # beat index known when forecast -> (predicted next R-R, last R-R)
        self.rr_scores = collections.deque(maxlen=RR_SCORED)   # (|model - actual|, |last - actual|) per beat

    @property
    def needs_acc(self):
        return bool(self.settings.with_acc)

    @property
    def needs_rr(self):
        return self.beats is not None

    def model_rows(self, times, rows):
        """The rows as the model takes them: for rr, each with its rr_ms appended, from the first beat on."""
        if not self.needs_rr:
            return times, rows
        lane = self.beats.lane(times)
        start = next((i for i, v in enumerate(lane) if v is not None), len(lane))
        return times[start:], [list(r) + [v] for r, v in zip(rows[start:], lane[start:])]

    def score_beats(self):
        """The R-R overlay's live error: each beat, once its R-R is known, against the last forecast made before
        it (its first predicted beat), next to holding the last R-R."""
        if not self.needs_rr:
            return
        n = len(self.beats)
        for idx in sorted(k for k in self.rr_open if k + 1 < n):
            pred, last = self.rr_open.pop(idx)
            actual = self.beats.rrs[idx + 1]
            self.rr_scores.append((abs(pred - actual), abs(last - actual)))

    def rr_score(self):
        if not self.rr_scores:
            return None
        m = len(self.rr_scores)
        return {"n": m, "model_ms": round(sum(a for a, _ in self.rr_scores) / m, 1),
                "hold_ms": round(sum(b for _, b in self.rr_scores) / m, 1)}

    def reset(self):
        self.aligner.reset()
        if self.beats is not None:
            self.beats.reset()
            self.rr_open.clear()
            self.rr_scores.clear()
        with self.lock:
            self.ring.clear()
            self.times.clear()
            self.piece += 1
        self.last_t, self.fresh = None, 0
        self.prev = None
        if self.log:
            self.log.close()
            self.log = None

    def take_rows(self):
        for t, row, new_piece in self.aligner.rows():
            with self.lock:
                if new_piece:
                    self.ring.clear()
                    self.times.clear()
                    self.piece += 1
                    self.prev = None             # a gap: the page starts afresh too
                self.ring.append(row)
                self.times.append(t)
            self.last_t = t
            self.fresh += 1

    def snapshot(self):
        with self.lock:
            times, rows, piece = list(self.times), list(self.ring), self.piece
        times, rows = self.model_rows(times, rows)
        return times, rows, piece

    def usable_rows(self):
        """How many rows a context could have now (for rr: from the first beat on)."""
        if not self.needs_rr:
            return len(self.ring)
        if len(self.beats) < 2:
            return 0
        with self.lock:
            first = bisect.bisect_left(list(self.times), self.beats.times[0])
            return len(self.times) - first

    def status(self, on):
        return {"on": on, "model": self.run, "loaded": self.model is not None,
                "error": self.cfg_error or self.load_error, "rows": len(self.ring),
                **({"beats": len(self.beats), "rr_score": self.rr_score()} if self.needs_rr else {}),
                "compute_ms": None if self.compute_ms is None else round(self.compute_ms, 1),
                "forecasts": self.count, "last_t": self.last_t}


# ------------------------------------------------------------------------------------------------ service
class Service:
    def __init__(self, a):
        self.a = a
        self.gpu = threading.Lock()          # the live loop and the HTTP API take turns on the models
        self.url = a.url.rstrip("/")
        self.settings = dict(DEFAULTS)
        self.overlays = {"ecg": Overlay("ecg", a.ecg_model or os.path.join(a.models, "ecg")),
                         "acc": Overlay("acc", a.acc_model or os.path.join(a.models, "ecg_acc")),
                         "rr": Overlay("rr", a.rr_model or os.path.join(a.models, "ecg_rr"))}
        self.session = None
        self.tails = {}
        self.state, self.detail = "starting", ""
        self.dashboard_ok = True
        self.stop = False
        self.record = open(a.record, "a") if a.record else None
        self.torch = self.dev = None
        self.torch_error = None
        self.device_name = "none"
        try:
            import torch
            self.torch = torch
            want = a.device
            if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
                self.dev = torch.device("cuda")
                self.device_name = f"cuda ({torch.cuda.get_device_name(0)})"
            else:
                self.dev = torch.device("cpu")
                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                self.device_name = "cpu (slow: no CUDA GPU found)" if want == "auto" else "cpu"
        except Exception as e:     # noqa: BLE001
            self.torch_error = (f"PyTorch isn't available for {sys.executable} ({type(e).__name__}); "
                                f"see polar_dashboard/forecast/README.md")
        log(f"device: {self.device_name if self.torch else self.torch_error}"
            + (" · torch.compile on" if a.compile and self.torch else ""))
        for ov in self.overlays.values():
            log(f"{ov.kind}: {ov.model_dir} ({ov.run}) {ov.cfg_error or ''}".rstrip())

    # ---------------- the PR #924 API (stream-lane-lab: GET /api/models, POST /api/forecast)
    def api_models(self):
        out = []
        for kind, ov in self.overlays.items():
            if ov.cfg_error:
                continue
            cols = []
            for ln in ov.lanes:
                step, scale = float(ln.get("step", 1)), float(ln.get("scale") or 1)
                cols.append({"name": f"{ln['column'].rsplit('_', 1)[0]}_{ln.get('unit') or 'value'}",
                             "unit": ln.get("unit"), "minimum": ln["int_min"] * step / scale,
                             "maximum": ln["int_max"] * step / scale, "levels": ln["int_max"] - ln["int_min"] + 1,
                             "kind": "integer_range"})
            out.append({"id": kind, "name": ov.run, "channels": len(cols), "columns": cols,
                        "context_length": ov.block, "interval_ms": ov.dt_ms, "max_horizon": API_MAX_HORIZON})
        return {"models": out, "device": self.device_name, "threads": self.torch.get_num_threads() if self.torch else 0,
                "message": "" if out else "No forecast models are installed (polar_dashboard/forecast/models)."}

    def api_rows(self, q):
        """GET /rows?model_id=ecg&after=<t_ms>&piece=<n>: the strap's newest rows on the model's grid, in real
        units (mV / mg), for the /lab viewer's live source. Rows after `after` (strap clock, ms); all of the
        context ring when `after` is missing, older than the ring, or from another piece (a gap since)."""
        ov = self.overlays.get(q.get("model_id", "ecg"))
        if ov is None or ov.cfg_error:
            return 404, {"detail": "Checkpoint not found; refresh the model list."}
        times, rows, piece = ov.snapshot()
        try:
            after = float(q["after"]) if q.get("after") not in (None, "") else None
            same = q.get("piece") not in (None, "") and int(q["piece"]) == piece
        except ValueError:
            return 422, {"detail": "after is a time in ms; piece an integer"}
        restart = not (same and after is not None and times and times[0] <= after + ov.dt_ms / 2)
        start = 0 if restart else next((i for i, t in enumerate(times) if t > after + 1e-6), len(times))
        real = [[round(to_real(ln, r[k]), 4) for k, ln in enumerate(ov.lanes)] for r in rows[start:]]
        return 200, {"model_id": q.get("model_id", "ecg"), "piece": piece, "restart": restart,
                     "interval_ms": ov.dt_ms, "times": times[start:], "rows": real,
                     "columns": [f"{ln['column'].rsplit('_', 1)[0]}_{ln.get('unit') or 'value'}" for ln in ov.lanes],
                     "state": self.state, "detail": self.detail}

    def api_signal(self, q):
        """PR #924's GET /api/signal: its sine mixtures (lane_lab/data.py sine_series), the viewer's demo source."""
        import numpy as np
        try:
            seed, frames = int(q.get("seed", 2026)), int(q.get("frames", 4096))
            channels, start = int(q.get("channels", 1)), int(q.get("start", 0))
        except ValueError:
            return 422, {"detail": "seed, frames, channels and start are integers"}
        if not (0 <= seed < 2 ** 31 and 1 <= frames <= 16384 and 1 <= channels <= 4 and 0 <= start <= 10 ** 9):
            return 422, {"detail": "seed 0..2^31-1, frames 1..16384, channels 1..4, start 0..1e9"}
        t = np.arange(start, start + frames, dtype=np.float64)
        out = []
        for c in range(channels):
            rng = np.random.default_rng(np.random.SeedSequence([seed, c]))
            amplitude = np.array([0.52, 0.29, 0.19]) * rng.uniform(0.85, 1.15, 3)
            amplitude *= 0.90 / amplitude.sum()
            phase = rng.uniform(-np.pi, np.pi, 3)
            frequency = np.array([1 / 128, 1 / 47, 1 / 23]) * (1 + 0.07 * c)
            out.append((amplitude[:, None] * np.sin(2 * np.pi * frequency[:, None] * t + phase[:, None])).sum(axis=0))
        values = np.stack(out, axis=-1).astype(np.float32)
        return 200, {"values": values.tolist(), "start": start, "channels": channels, "seed": seed}

    def api_forecast(self, req):
        """PR #924's POST /api/forecast for the H10 models: history rows in real units (mV / mg) on the
        model's grid (8 ms), oldest first -> predictions for the next `horizon` rows. Returns (status, body)."""
        torch = self.torch
        if torch is None:
            return 503, {"detail": self.torch_error}
        if not isinstance(req, dict):
            return 422, {"detail": "Send a JSON object."}
        ov = self.overlays.get(req.get("model_id"))
        if ov is None or ov.cfg_error:
            return 404, {"detail": "Checkpoint not found; refresh the model list."}
        try:
            horizon = int(req.get("horizon", 24))
            mode = req.get("mode", "mean")
            temperature = float(req.get("temperature", 1.0))
            top_k = int(req.get("top_k", 0))
            seed = int(req.get("seed", 0))
            history = req.get("history")
        except (TypeError, ValueError):
            return 422, {"detail": "horizon, top_k and seed are integers; temperature a number."}
        if not 1 <= horizon <= API_MAX_HORIZON:
            return 422, {"detail": f"horizon must be 1..{API_MAX_HORIZON}"}
        if mode not in ("mean", "greedy", "sample"):
            return 422, {"detail": "mode must be mean, greedy or sample"}
        if not 0 < temperature <= 10 or not 0 <= top_k <= 4096 or not 0 <= seed < 2 ** 31:
            return 422, {"detail": "temperature in (0, 10], top_k 0..4096, seed 0..2^31-1"}
        K = len(ov.lanes)
        if not isinstance(history, list) or not 1 <= len(history) <= 16384:
            return 422, {"detail": "Provide between 1 and 16384 observed frames."}
        if any(not isinstance(r, list) or len(r) != K for r in history):
            return 422, {"detail": f"Every history row must contain {K} columns."}
        try:
            rows = [[float(v) for v in r] for r in history]
        except (TypeError, ValueError):
            return 422, {"detail": "History must contain numbers."}
        if any(not math.isfinite(v) for r in rows for v in r):
            return 422, {"detail": "History must contain finite numbers."}
        rows = rows[-(ov.block - horizon):]                  # context + horizon fit the model's window
        ints, clipped = [], [0] * K
        for r in rows:
            q = []
            for k, (v, ln) in enumerate(zip(r, ov.lanes)):
                i = int(round(v * float(ln.get("scale") or 1) / float(ln.get("step", 1))))
                if i < ln["int_min"] or i > ln["int_max"]:
                    clipped[k] += 1
                    i = min(ln["int_max"], max(ln["int_min"], i))
                q.append(i)
            ints.append(q)
        with self.gpu:
            loading = ov.model is None
            if not self.ensure_model(ov):
                return 500, {"detail": ov.load_error or ov.cfg_error or "the model didn't load"}
            t0 = time.perf_counter()
            preds = self._rollout(ov, ints, horizon, mode, temperature, top_k, seed)
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            latency = (time.perf_counter() - t0) * 1000.0
        return 200, {"predictions": preds, "latency_ms": latency, "model_id": req["model_id"],
                     "history_frames_used": len(rows), "horizon": horizon, "mode": mode, "model_loaded": loading,
                     "out_of_range_counts": clipped, "interval_ms": ov.dt_ms,
                     "note": "Forecast uses supplied history only."}

    def _rollout(self, ov, ints, horizon, mode, temperature, top_k, seed):
        """Autoregressive forecast with the model's KV cache. mean: each lane's expected value (returned as is,
        fed back rounded to its integer level); greedy: the most likely level; sample: a draw at `temperature`
        from the top_k levels (0 = all), seeded."""
        torch = self.torch
        model, dev = ov.model, self.dev
        lo = torch.tensor([ln["int_min"] for ln in ov.lanes], device=dev)
        real = [float(ln.get("step", 1)) / float(ln.get("scale") or 1) for ln in ov.lanes]
        ids = torch.tensor([[r[k] for r in ints] for k in range(len(ov.lanes))], device=dev).unsqueeze(0) - lo.view(1, -1, 1)
        gen = torch.Generator(device=dev).manual_seed(seed)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
        out_rows, past, x = [], None, ids
        with torch.inference_mode(), amp:
            for _ in range(horizon):
                o = model(x, past_key_values=past, use_cache=True, last_step_only=True)
                past = o.past_key_values
                nxt, row = [], []
                for k, lg in enumerate(o.logits):
                    lg = lg[0, -1].float()
                    if mode == "greedy":
                        q = int(lg.argmax())
                        val = (q + int(lo[k])) * real[k]
                    elif mode == "mean":
                        p = torch.softmax(lg, -1)
                        levels = torch.arange(lg.numel(), device=dev, dtype=torch.float32)
                        m = float((p * levels).sum())
                        q = int(round(m))
                        val = (m + int(lo[k])) * real[k]
                    else:
                        lg = lg / temperature
                        if 0 < top_k < lg.numel():
                            v, _ = torch.topk(lg, top_k)
                            lg = lg.masked_fill(lg < v[-1], float("-inf"))
                        q = int(torch.multinomial(torch.softmax(lg, -1), 1, generator=gen))
                        val = (q + int(lo[k])) * real[k]
                    nxt.append(q)
                    row.append(round(val, 6))
                out_rows.append(row)
                x = torch.tensor(nxt, device=dev).view(1, -1, 1)
        return out_rows

    # ---------------- the inference log
    def keep(self, ov, t0_new):
        """The previous forecast's rows up to where the new one starts are what the dashboard keeps on screen (the
        inference): write them to Forecasts/<day>/<session time>_forecast_<kind>.csv."""
        prev = ov.prev
        if prev is None or self.a.no_log or not self.session:
            return
        if ov.log is None or ov.log.session != self.session:
            if ov.log:
                ov.log.close()
            try:
                ov.log = ForecastLog(self.session, ov)
                log(f"{ov.kind}: logging the inference to {ov.log.path}")
            except OSError as e:
                log(f"{ov.kind}: can't write the inference log: {e}")
                self.a.no_log = True
                return
        ov.log.write(prev, t0_new)

    # ---------------- dashboard link
    def heartbeat(self):
        if self.a.bench:
            return
        body = {"state": self.state, "detail": self.detail, "device": self.device_name, "pid": os.getpid(),
                "session": self.session, "overlays": {k: ov.status(bool(self.settings.get(k)))
                                                      for k, ov in self.overlays.items()},
                "horizon_s": self.settings.get("horizon_s"), "samples": self.settings.get("samples")}
        try:
            got = post_json(self.url + "/api/forecast/heartbeat", body)
            if isinstance(got, dict):
                self.settings = {**DEFAULTS, **{k: v for k, v in got.items() if k in DEFAULTS}}
            if not self.dashboard_ok:
                log("dashboard reachable again")
            self.dashboard_ok = True
        except (urllib.error.URLError, OSError, ValueError) as e:
            if self.dashboard_ok:
                log(f"dashboard not reachable at {self.url} ({e}); retrying")
            self.dashboard_ok = False

    # ---------------- data
    def follow(self):
        ecg, acc, rr = newest_session(self.a.session, None if self.a.bench else self.url)
        if ecg == self.session and (acc is None or "acc" in self.tails) and (rr is None or "rr" in self.tails):
            return
        if ecg != self.session:
            for t in self.tails.values():
                t.close()
            self.tails = {}
            for ov in self.overlays.values():
                ov.reset()
            self.session = ecg
            if ecg:
                log(f"following {os.path.dirname(ecg)} ({os.path.basename(ecg)[:8]})")
                self.tails["ecg"] = Tail(ecg, 48 * 1024)
        if acc and "acc" not in self.tails and self.session:
            self.tails["acc"] = Tail(acc, 64 * 1024)
        if rr and "rr" not in self.tails and self.session:
            self.tails["rr"] = Tail(rr, 4 * 1024)            # ~2 minutes of beats

    def read(self):
        if "acc" in self.tails:            # accelerometer first, so ECG rows don't wait for it
            for line in self.tails["acc"].lines():
                r = parse(line, 4)
                if r:
                    for ov in self.overlays.values():
                        if ov.needs_acc:
                            ov.aligner.push_acc(r[0], r[1], r[2], r[3])
        if "rr" in self.tails:
            for line in self.tails["rr"].lines():
                r = parse(line, 2)
                if r:
                    for ov in self.overlays.values():
                        if ov.needs_rr:
                            ov.beats.push(r[0], r[1])
        if "ecg" in self.tails:
            for line in self.tails["ecg"].lines():
                r = parse(line, 2)
                if r:
                    for ov in self.overlays.values():
                        ov.aligner.push_ecg(r[0], r[1])
        for ov in self.overlays.values():
            ov.take_rows()
            ov.score_beats()

    # ---------------- models
    def ensure_model(self, ov):
        if ov.model is not None or ov.cfg_error or ov.load_error or self.torch is None:
            return ov.model is not None
        self.state, self.detail = "loading model", ov.run          # the heartbeat thread reports it
        try:
            from modeling_csv_mc_int import CsvMcIntForCausalLM
            t0 = time.time()
            ov.model = CsvMcIntForCausalLM.load_dir(ov.model_dir, map_location=self.dev)
            log(f"{ov.kind}: loaded {ov.run} on {self.dev} in {time.time() - t0:.1f} s")
        except Exception as e:     # noqa: BLE001
            ov.load_error = f"couldn't load {ov.run}: {type(e).__name__}: {e}"
            if "PreTrainedModel" in str(e) or isinstance(e, ImportError):
                # transformers hides the real import error; say how to see it, and the usual fix
                ov.load_error += (f" | the real error: {sys.executable} -c 'from transformers.modeling_utils import PreTrainedModel'"
                                  " | usually fixed by: tools/orin_setup.sh (V6.2+), or: pip install 'transformers>=4.40,<5' 'pillow>=9.1'")
            log(ov.load_error)
        return ov.model is not None

    def forecaster(self, ov, samples):
        if ov.fc is None or ov.fc.N != samples:
            from fast_forecast import FastForecaster

            def make():
                return FastForecaster(ov.model, num_samples=samples, top_k=TOP_K, temperature=TEMPERATURE,
                                      dtype=self.torch.bfloat16)
            fc = make()
            if self.a.compile and self.dev.type == "cuda":
                # Optional: fuse the decode step's ~300 small kernels (about 2.3x faster on an A100). Compiling
                # takes ~35 s per model there (longer on an Orin) and needs Triton; on any failure the plain
                # CUDA graph is used.
                self.state, self.detail = "loading model", f"{ov.run}: compiling (torch.compile)"
                t0 = time.time()
                try:
                    fc._step = self.torch.compile(fc._step, mode="max-autotune-no-cudagraphs", dynamic=False)
                    warm = self.torch.tensor([[-ln["int_min"]] * 8 for ln in ov.lanes], dtype=self.torch.long)
                    fc.forecast(warm, 2)                 # compiles, then captures the CUDA graph around it
                    log(f"{ov.kind}: compiled the decode step in {time.time() - t0:.0f} s")
                except Exception as e:     # noqa: BLE001
                    log(f"{ov.kind}: torch.compile failed ({type(e).__name__}: {str(e)[:200]}); "
                        f"using the plain CUDA graph")
                    fc = make()
            ov.fc = fc
        return ov.fc

    def forecast(self, ov):
        torch = self.torch
        horizon = max(1, int(round(float(self.settings["horizon_s"]) * 1000.0 / ov.dt_ms)))
        horizon = min(horizon, ov.block - MIN_CONTEXT + 1)
        samples = max(1, min(32, int(self.settings["samples"])))
        with ov.lock:
            times, rows = ov.model_rows(list(ov.times), list(ov.ring))
        T = min(len(rows), ov.block - horizon + 1)
        rows, times = rows[-T:], times[-T:]
        ctx = torch.tensor([[r[k] - ov.lanes[k]["int_min"] for r in rows] for k in range(len(ov.lanes))],
                           dtype=torch.long)
        fc = self.forecaster(ov, samples)
        t0 = time.perf_counter()
        toks = fc.forecast(ctx, horizon, generator_seed=self.a.seed)          # (N, K, H) token ids
        vals = (toks.cpu() + torch.tensor([ln["int_min"] for ln in ov.lanes]).view(1, -1, 1)).tolist()
        ov.compute_ms = (time.perf_counter() - t0) * 1000.0
        ov.count += 1
        ov.fresh = 0
        prefer = None
        if REP_PICK == "rhythm" and ov.lanes[0]["column"].startswith("ecg"):
            prefer = rhythm_pick([r[0] for r in ov.ring], [v[0] for v in vals])   # all 2 s: more beats than the context
        out = summarize(vals, ov.lanes, prefer)
        out["rep_pick"] = "rhythm" if prefer is not None else "median"
        if ov.needs_rr:
            k = ov.names.index("rr_ms")
            beats = next_beats(times, [r[k] for r in rows], [v[k] for v in vals], ov.beats.times[-1], ov.dt_ms,
                               ov.settings.grid_ms or rr_lane.GRID_MS)
            out["beats"] = beats
            out["last_beat"] = {"t": ov.beats.times[-1], "rr": ov.beats.rrs[-1], "index": len(ov.beats) - 1}
            out["known"] = [[t, r] for t, r in zip(ov.beats.times[-8:], ov.beats.rrs[-8:])]   # as the model sees them
            if beats:
                ov.rr_open[len(ov.beats) - 1] = (beats[0]["rr"], ov.beats.rrs[-1])   # the latest one counts
            out["rr_score"] = ov.rr_score()
        t0_new = times[-1] + ov.dt_ms
        self.keep(ov, t0_new)
        ov.prev = {"t0": t0_new, "dt": ov.dt_ms, "from": times[-1], **{k: out[k] for k in ("rep", "median", "lo", "hi")}}
        last = [to_real(ov.lanes[k], rows[-1][k]) for k in range(len(ov.lanes))]
        payload = {"kind": ov.kind, "model": ov.run, "t0": times[-1] + ov.dt_ms, "dt_ms": ov.dt_ms,
                   "lanes": ov.names, "units": ov.units, "last": last, "context_rows": T, "samples": samples,
                   "compute_ms": round(ov.compute_ms, 1), "made_at": int(time.time() * 1000), **out}
        if self.record:
            self.record.write(json.dumps({"kind": ov.kind, "t0": payload["t0"], "times": times, "rows": rows,
                                          "rep_index": out["rep_index"], "median": out["median"]}) + "\n")
            self.record.flush()
        try:
            post_json(self.url + "/api/forecast/live", payload)
            ov.posted += 1
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.dashboard_ok = False
            log(f"couldn't send the {ov.kind} forecast: {e}")

    # ---------------- main loop
    def _heartbeats(self):
        """Once a second, from its own thread: the dashboard keeps seeing the service (and its state) while a
        model loads or compiles or a slow forecast runs."""
        while not self.stop:
            t0 = time.time()
            self.heartbeat()
            time.sleep(max(0.05, 1.0 - (time.time() - t0)))

    def run(self):
        threading.Thread(target=self._heartbeats, name="heartbeat", daemon=True).start()
        if self.a.api_port:
            start_api(self, self.a.api_port)
        next_follow = next_fc = 0.0
        while not self.stop:
            now = time.time()
            if now >= next_follow:
                next_follow = now + 2.0
                try:
                    self.follow()
                except OSError as e:
                    log(f"session files: {e}")
            try:
                self.read()
            except OSError as e:
                log(f"read: {e}")
                self.session = None
            on = [k for k in KINDS if self.settings.get(k)]
            for k, ov in self.overlays.items():
                if k not in on and ov.prev is not None:      # switched off: the page dropped it too
                    ov.prev = None
            newest = max([ov.last_t for ov in self.overlays.values() if ov.last_t is not None], default=None)
            stale = newest is None or now * 1000.0 - newest > STALE_MS
            if self.torch is None:
                self.state, self.detail = "unavailable", self.torch_error
            elif not on:
                self.state, self.detail = "idle", "the overlays are off"
            elif not self.session:
                self.state, self.detail = "waiting for data", "no Polar H10 recording yet"
            elif stale:
                self.state, self.detail = "waiting for data", "no new samples from the strap"
            else:
                self.state, self.detail = "forecasting", ""
            if self.state == "forecasting" and now >= next_fc:
                next_fc = now + max(0.1, float(self.settings["refresh_s"]))
                for k in on:
                    ov = self.overlays[k]
                    if ov.usable_rows() >= MIN_CONTEXT and ov.fresh > 0 and self.ensure_model(ov):
                        try:
                            with self.gpu:
                                self.forecast(ov)
                        except Exception as e:     # noqa: BLE001  (keep serving; report it)
                            ov.load_error = f"forecast failed: {type(e).__name__}: {e}"
                            log(ov.load_error)
                    elif ov.cfg_error or ov.load_error:
                        self.detail = ov.cfg_error or ov.load_error
                    elif ov.needs_rr and len(ov.beats) < 2:
                        self.detail = "R-R: waiting for the strap's beats"
            time.sleep(0.02)
        for t in self.tails.values():
            t.close()
        for ov in self.overlays.values():
            if ov.log:
                ov.log.close()
        if self.record:
            self.record.close()
        log("stopped")


def bench(svc, repeats):
    """Time warm forecasts for every model at the dashboard settings and report GPU memory (the Orin check:
    the overlays together must fit well inside the refresh interval)."""
    torch = svc.torch
    if torch is None:
        sys.exit(svc.torch_error)
    samples = int(svc.settings["samples"])
    sync = torch.cuda.synchronize if svc.dev.type == "cuda" else (lambda: None)
    total = 0.0
    for ov in svc.overlays.values():
        if ov.cfg_error or not svc.ensure_model(ov):
            log(f"{ov.kind}: {ov.cfg_error or ov.load_error}")
            continue
        H = max(1, int(round(float(svc.settings["horizon_s"]) * 1000.0 / ov.dt_ms)))
        T = ov.block - H + 1
        ctx = torch.tensor([[-ln["int_min"]] * T for ln in ov.lanes], dtype=torch.long)   # value 0 (timing only)
        fc = svc.forecaster(ov, samples)
        t0 = time.perf_counter()
        fc.forecast(ctx, H)
        sync()
        first = time.perf_counter() - t0
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            toks = fc.forecast(ctx, H)
            summarize((toks.cpu() + torch.tensor([ln["int_min"] for ln in ov.lanes]).view(1, -1, 1)).tolist(),
                      ov.lanes)
            sync()
            times.append(time.perf_counter() - t0)
        med = statistics.median(times) * 1000.0
        total += med
        mem = (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB"
               if svc.dev.type == "cuda" else "")
        log(f"{ov.kind} ({ov.run}): {samples} samples x {H} rows from {T} context rows: median {med:.1f} ms, "
            f"max {max(times) * 1000.0:.1f} ms per forecast incl. summary (first one, with set-up: "
            f"{first * 1000.0:.0f} ms){mem}")
    if total:
        budget = float(svc.settings["refresh_s"]) * 1000.0
        log(f"all overlays: {total:.0f} ms of every {budget:.0f} ms refresh ("
            + ("OK" if total < 0.75 * budget else
               "tight: try --compile, a longer refresh_s or a shorter horizon_s; fewer samples barely help") + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("BIODASH_FORECAST_URL", "http://127.0.0.1:5000/api/d/heart"))
    ap.add_argument("--models", default=os.environ.get("BIODASH_FORECAST_MODELS", os.path.join(FC_DIR, "models")))
    ap.add_argument("--ecg-model", help="model folder for the ECG overlay (default <models>/ecg)")
    ap.add_argument("--acc-model", help="model folder for the accelerometer overlay (default <models>/ecg_acc)")
    ap.add_argument("--rr-model", help="model folder for the R-R overlay (default <models>/ecg_rr)")
    ap.add_argument("--device", default=os.environ.get("BIODASH_FORECAST_DEVICE", "auto"), choices=("auto", "cuda", "cpu"))
    ap.add_argument("--compile", action="store_true", default=os.environ.get("BIODASH_FORECAST_COMPILE") == "1",
                    help="torch.compile the decode step (~2.3x faster on an A100; needs Triton; compiles for "
                         "~35 s per model when its overlay is first switched on). Env: BIODASH_FORECAST_COMPILE=1")
    ap.add_argument("--session", help="follow this *_ecg.csv instead of the newest recording")
    ap.add_argument("--record", help="append each forecast's context rows to this JSONL file (checks)")
    ap.add_argument("--seed", type=int, help="fixed sampling seed for every forecast (checks)")
    ap.add_argument("--api-port", type=int, default=int(os.environ.get("BIODASH_FORECAST_API_PORT", "5011")),
                    help="local port for the PR #924 API that api.py passes on (0 = off; default 5011)")
    ap.add_argument("--no-log", action="store_true", default=os.environ.get("BIODASH_FORECAST_LOG") == "0",
                    help="don't write the inference to Forecasts/<day>/<time>_forecast_<kind>.csv "
                         "(env BIODASH_FORECAST_LOG=0)")
    ap.add_argument("--bench", type=int, nargs="?", const=20, default=0, metavar="N",
                    help="no dashboard: load the models, time N forecasts each (default 20) and exit")
    ap.add_argument("--samples", type=int, help="with --bench: samples per forecast (default: the dashboard's 8)")
    a = ap.parse_args()
    svc = Service(a)
    if a.bench:
        if a.samples:
            svc.settings["samples"] = max(1, min(32, a.samples))
        bench(svc, a.bench)
        return

    def _stop(*_):
        svc.stop = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    svc.run()


if __name__ == "__main__":
    main()
