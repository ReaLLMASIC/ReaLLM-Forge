"""Atmos line parsing: turns whatever an Atmos device sends into {series_id: value}.

Accepted line formats (detected per line):
  json        {"co2": 612, "scd30_temp": 22.4}
  key=value   co2=612, temp=22.4, pm2.5=3.1        (also key:value, ; or space separated)
  csv header  CO2_ppm,Temp_C,RH ...                 (names for the numeric lines that follow)
  csv         612,22.4,41.0,...                     (positional; mapped by header / profile)

Series ids are canonical metric names, optionally qualified by the sensor they came
from: "co2", "temp@scd30", "rh@sen55". The Atmos-Sphere boards carry three sensors
that all report temperature and humidity, so the source keeps them apart; the
dashboard shows the best source per metric (see PRIMARY_SOURCE_ORDER).

Headerless numeric CSV needs a column order. Priority:
  1. a header line the device sent this session
  2. a custom column map set in the debug panel
  3. the selected profile (or, on "auto", the profile whose field count matches)
  4. generic names: field1, field2, ...
"""
import json
import math
import re

CANONICAL = ("co2", "pm1", "pm25", "pm4", "pm10", "temp", "rh", "voc", "nox", "hcho")

SOURCES = ("scd30", "scd40", "scd41", "scd4x", "sen44", "sen54", "sen55", "sen5x", "sen66",
           "sen68", "sen69c", "sfa30", "sfa3x", "sht4x", "sgp41")

# Temperature / RH: SEN5x/SEN6x are compensated, SCD30 runs warm from its own IR source,
# SFA3X last. Single-source metrics (CO2, PM, HCHO) just take whatever exists.
PRIMARY_SOURCE_ORDER = ("sen69c", "sen68", "sen66", "sen55", "sen54", "sen5x", "sen44",
                        "sht4x", "scd30", "scd41", "scd40", "scd4x", "sfa3x", "sfa30", "sgp41", "")

# Column orders for headerless CSV. Sensor blocks follow each Sensirion library's own
# read order. S4 is confirmed: it's exactly what the Atmos-Sphere-s4 firmware's BLE
# fallback sends (it also sends this as a header line). S5's block order (SCD30,
# SEN55, SFA3X) is still an assumption -- set a custom column map if it differs.
PROFILES = {
    "mini": {
        "label": "Atmos-Mini (SEN69C)",
        "sensors": ["SEN69C"],
        "columns": ["pm1@sen69c", "pm25@sen69c", "pm4@sen69c", "pm10@sen69c", "rh@sen69c",
                    "temp@sen69c", "voc@sen69c", "nox@sen69c", "hcho@sen69c", "co2@sen69c"],
        "order_confirmed": True,
    },
    "s4": {
        "label": "Atmos-Sphere S4 (SCD30 + SEN44 + SFA3X)",
        "sensors": ["SCD30", "SEN44", "SFA3X"],
        "columns": ["co2@scd30", "temp@scd30", "rh@scd30",
                    "pm1@sen44", "pm25@sen44", "pm4@sen44", "pm10@sen44", "voc@sen44", "rh@sen44", "temp@sen44",
                    "hcho@sfa3x", "rh@sfa3x", "temp@sfa3x"],
        "order_confirmed": True,    # matches the Atmos-Sphere-s4 firmware's BLE output
    },
    "s4wifi": {
        # The Atmos-Sphere-s4 firmware's Wi-Fi TCP stream (port 8080): no header, temp and
        # RH already averaged across its three sensors, no NOx. Confirmed from the sketch.
        "label": "Atmos-Sphere S4 over Wi-Fi (averaged temp / RH)",
        "sensors": ["SCD30", "SEN44", "SFA3X"],
        "columns": ["pm1@sen44", "pm25@sen44", "pm4@sen44", "pm10@sen44", "rh", "temp",
                    "voc@sen44", "hcho@sfa3x", "co2@scd30"],
        "order_confirmed": True,
    },
    "s5": {
        "label": "Atmos-Sphere S5 (SCD30 + SEN55 + SFA3X)",
        "sensors": ["SCD30", "SEN55", "SFA3X"],
        "columns": ["co2@scd30", "temp@scd30", "rh@scd30",
                    "pm1@sen55", "pm25@sen55", "pm4@sen55", "pm10@sen55", "rh@sen55", "temp@sen55",
                    "voc@sen55", "nox@sen55",
                    "hcho@sfa3x", "rh@sfa3x", "temp@sfa3x"],
        "order_confirmed": False,
    },
}
PROFILE_BY_FIELD_COUNT = {len(p["columns"]): k for k, p in PROFILES.items()}

_UNIT_TOKENS = {"ppm", "ppb", "ugm3", "ug", "m3", "c", "degc", "deg", "pct", "percent", "index", "idx",
                "raw", "val", "value", "ug_m3", "mgm3"}
_TS_NAMES = {"ts", "time", "timestamp", "timestamp_epoch_ms", "epoch", "epoch_ms", "millis", "ms", "uptime"}


def _metric_from_tokens(toks):
    n = "_".join(toks)
    squashed = "".join(toks)
    if re.fullmatch(r"pm_?1(_0)?", n) or squashed in ("pm1p0",):
        return "pm1"
    if re.fullmatch(r"pm_?2_?5", n) or squashed in ("pm25", "pm2p5"):
        return "pm25"
    if re.fullmatch(r"pm_?4(_0)?", n) or squashed in ("pm4p0",):
        return "pm4"
    if re.fullmatch(r"pm_?10(_0)?", n) or squashed in ("pm10p0",):
        return "pm10"
    if squashed in ("co2", "carbondioxide"):
        return "co2"
    if squashed in ("hcho", "formaldehyde", "ch2o"):
        return "hcho"
    if squashed in ("t", "temp", "temperature", "tmp"):
        return "temp"
    if squashed in ("rh", "hum", "humidity", "relativehumidity"):
        return "rh"
    if squashed in ("voc", "vocindex", "voci"):
        return "voc"
    if squashed in ("nox", "noxindex", "noxi"):
        return "nox"
    return None


def normalize_name(raw):
    """'SCD30_CO2_ppm' -> 'co2@scd30', 'PM2.5' -> 'pm25', 'Temp_C' -> 'temp',
    'Timestamp_Epoch_ms' -> '__ts__', 'weird thing' -> 'weird_thing' (unmapped)."""
    s = raw.strip().strip('"').lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)            # drop "(ppm)" / "[C]"
    s = s.replace("µ", "u").replace("°", "").replace("%", " pct ").replace(".", "_")
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    if not toks:
        return None
    if "_".join(toks) in _TS_NAMES:
        return "__ts__"
    source = next((t for t in toks if t in SOURCES), "")
    rest = [t for t in toks if t != source]
    core = [t for t in rest if t not in _UNIT_TOKENS] or rest
    metric = _metric_from_tokens(core)
    if metric is None:
        # "humidity_rh" / "temp_c" style: try again without trailing unit-ish token
        metric = _metric_from_tokens(core[:1]) if len(core) > 1 else None
    if metric is None:
        return "_".join(toks)                         # unknown field: keep a clean raw name
    return f"{metric}@{source}" if source else metric


def metric_of(series_id):
    return series_id.split("@", 1)[0]


def is_canonical(series_id):
    return metric_of(series_id) in CANONICAL


def _num(tok):
    try:
        v = float(tok)
    except ValueError:
        return None
    return v if not math.isinf(v) else None


class LineParser:
    """Stateful: remembers a device-sent header and the chosen profile."""

    def __init__(self):
        self.profile = "auto"        # auto | mini | s4 | s5
        self.custom_columns = None   # list of names from the debug panel, or None
        self.device_header = None    # names from a header line the device sent
        self.last_format = None
        self.last_field_count = None
        self.column_source = None    # which rule mapped the last positional line
        self.effective_profile = None

    def configure(self, profile="auto", custom_columns=None):
        self.profile = profile if profile in PROFILES or profile == "auto" else "auto"
        cols = [c.strip() for c in (custom_columns or []) if c.strip()]
        self.custom_columns = [normalize_name(c) or c for c in cols] or None

    def reset_session(self):
        self.device_header = None

    def _columns_for(self, n):
        if self.device_header and len(self.device_header) == n:
            self.effective_profile = None
            return self.device_header, "device header"
        if self.custom_columns:
            self.effective_profile = None
            cols = self.custom_columns[:n] + [f"field{i + 1}" for i in range(len(self.custom_columns), n)]
            return cols, "custom column map"
        prof = self.profile if self.profile != "auto" else PROFILE_BY_FIELD_COUNT.get(n)
        if prof:
            self.effective_profile = prof
            cols = PROFILES[prof]["columns"]
            cols = cols[:n] + [f"field{i + 1}" for i in range(len(cols), n)]
            how = "profile" if self.profile != "auto" else "auto (field count)"
            return cols, f"{how}: {PROFILES[prof]['label']}"
        self.effective_profile = None
        return [f"field{i + 1}" for i in range(n)], "generic (unknown layout)"

    def parse(self, line):
        """Returns ({series_id: float}, kind). kind: 'data' | 'header' | 'reject'."""
        line = line.strip().strip("\r")
        if not line:
            return {}, "reject"

        # JSON object
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                return {}, "reject"
            self.last_format = "json"
            self.column_source = "names in each line"
            # JSON: only real numbers are readings ("fw": "1.2" is metadata, not a sensor)
            return self._named((k, v) for k, v in obj.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool))

        # key=value / key:value
        if "=" in line or re.search(r"[A-Za-z_][\w.]*\s*:\s*-?[\d.]", line):
            pairs = []
            for part in re.split(r"[,;\t]|\s{2,}|\s(?=[A-Za-z_][\w.]*\s*[=:])", line):
                m = re.match(r"\s*([^=:]+?)\s*[=:]\s*(\S+)", part)
                if m:
                    pairs.append((m.group(1), m.group(2)))
            if pairs:
                self.last_format = "key=value"
                self.column_source = "names in each line"
                return self._named(pairs)
            return {}, "reject"

        # CSV
        toks = [t.strip() for t in line.split(",")]
        nums = [_num(t) for t in toks]
        if all(v is not None for v in nums):
            self.last_format = "csv"
            self.last_field_count = len(toks)
            cols, self.column_source = self._columns_for(len(toks))
            return {c: v for c, v in zip(cols, nums) if c != "__ts__" and not math.isnan(v)}, "data"
        if not any(v is not None for v in nums) and len(toks) >= 2:
            names = [normalize_name(t) or f"field{i + 1}" for i, t in enumerate(toks)]
            self.device_header = names
            self.last_format = "csv header"
            return {}, "header"
        return {}, "reject"

    def _named(self, items):
        out = {}
        for k, v in items:
            sid = normalize_name(str(k))
            val = v if isinstance(v, (int, float)) and not isinstance(v, bool) else _num(str(v))
            if sid and sid != "__ts__" and val is not None and not (isinstance(val, float) and math.isnan(val)):
                out[sid] = float(val)
        return (out, "data") if out else ({}, "reject")
