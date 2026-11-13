"""Water maths for Hydro-dash: a suggested daily goal (body weight, reference intake, exercise and
the day's weather), and the local weather from Open-Meteo (free, no API key; only the town's
coordinates are sent). Standard library only. Checks: python3 test_water.py"""
import datetime as dt
import json
import threading
import time
import urllib.parse
import urllib.request

LB_PER_KG = 2.20462
# National Academies adequate intake of total water (drinks + food), mL/day; about 80 % of it
# usually comes from drinks, which is what the bottle measures
REFERENCE_TOTAL = {"men": 3700, "women": 2700}
FROM_DRINKS = 0.8
ML_PER_KG = 33              # "half your body weight (lb) in fl oz" = 0.5 * 29.57 / 0.4536 ≈ 33 mL/kg
ML_PER_EXERCISE_MIN = 12    # ≈ 0.7 L per hour of sweaty activity (sports-medicine range 0.4–0.8)
HOT_DAY_ML = 500            # the manual "hot weather" choice
MIN_GOAL, MAX_GOAL = 1500, 5000

# Extra water for the day's "feels like" high (°C): heat and humidity both raise sweating.
# (from, extra mL, word). Cold days add nothing and take nothing away: cold air is dry and
# breathing loses water, but thirst runs low, so the goal stays and the note says so.
HEAT_BANDS = [(36, 1000, "extreme heat"), (32, 750, "very hot"), (28, 500, "hot"), (25, 250, "warm")]
COLD_BELOW = 10

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIMEOUT_S = 8
CACHE_S = 30 * 60


def round50(x):
    return int(round(x / 50.0)) * 50


def weather_extra(feels_max_c):
    """-> (extra mL, word) for the day's feels-like high, or (0, None) when unknown."""
    if feels_max_c is None:
        return 0, None
    for start, ml, word in HEAT_BANDS:
        if feels_max_c >= start:
            return ml, word
    return 0, "cold" if feels_max_c < COLD_BELOW else "mild"


def suggest_goal(calc, weather=None):
    """calc: {weight, weight_unit 'lb'|'kg', reference 'men'|'women', exercise_min, weather 'auto'|'hot'|'off'}
    weather: today's forecast (see parse_forecast) or None.
    -> dict with the goal and every step of the working."""
    try:
        w = float(calc.get("weight") or 0)
    except (TypeError, ValueError):
        w = 0
    kg = None if w <= 0 else (w if calc.get("weight_unit") == "kg" else w / LB_PER_KG)
    ref = round(REFERENCE_TOTAL.get(calc.get("reference"), REFERENCE_TOTAL["men"]) * FROM_DRINKS)
    by_weight = round(kg * ML_PER_KG) if kg else None
    base = (ref + by_weight) / 2 if by_weight else ref
    try:
        minutes = max(0, min(600, int(calc.get("exercise_min") or 0)))
    except (TypeError, ValueError):
        minutes = 0
    exercise = minutes * ML_PER_EXERCISE_MIN

    mode = calc.get("weather") or "off"
    weather_ml, word = 0, None
    if mode == "hot":
        weather_ml, word = HOT_DAY_ML, "hot (set by hand)"
    elif mode == "auto" and weather:
        weather_ml, word = weather_extra(weather.get("feels_max_c"))

    raw = base + exercise + weather_ml
    goal = max(MIN_GOAL, min(MAX_GOAL, round50(raw)))
    return {"goal": goal, "kg": round(kg, 1) if kg else None, "reference": ref, "by_weight": by_weight,
            "base": round(base), "exercise": exercise, "weather_ml": weather_ml, "weather_word": word,
            "weather_mode": mode, "weather": weather if mode == "auto" else None,
            "clamped": goal != round50(raw)}


# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------
def _get_json(url, params):
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}",
                                 headers={"User-Agent": "Bio-dash Hydro (github.com/Meapy011/Bio-dash)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_forecast(j):
    """Open-Meteo forecast JSON -> {date, max_c, min_c, feels_max_c, now_c} for the place's today."""
    daily = j.get("daily") or {}

    def first(key):
        v = daily.get(key) or []
        return v[0] if v and v[0] is not None else None

    if not (daily.get("time") or []):
        raise ValueError("no daily forecast in the reply")
    out = {"date": daily["time"][0], "max_c": first("temperature_2m_max"), "min_c": first("temperature_2m_min"),
           "feels_max_c": first("apparent_temperature_max"),
           "now_c": (j.get("current") or {}).get("temperature_2m"), "timezone": j.get("timezone")}
    if out["feels_max_c"] is None:
        out["feels_max_c"] = out["max_c"]
    return out


_cache, _lock = {}, threading.Lock()


def forecast(lat, lon, fetch=None, now=time.time):
    """Today's weather at (lat, lon), cached for 30 min. Raises on network / format errors."""
    fetch = fetch or _get_json
    key = (round(float(lat), 3), round(float(lon), 3))
    with _lock:
        hit = _cache.get(key)
        if hit and now() - hit[0] < CACHE_S and hit[1]["date"] == dt.date.today().isoformat():
            return hit[1]
    j = fetch(FORECAST_URL, {"latitude": key[0], "longitude": key[1], "timezone": "auto", "forecast_days": 1,
                             "current": "temperature_2m",
                             "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max"})
    w = parse_forecast(j)
    w["fetched_at"] = now()
    with _lock:
        _cache[key] = (now(), w)
    return w


def find_places(name, fetch=None):
    """Town search -> [{name, lat, lon, label}] (up to 6)."""
    fetch = fetch or _get_json
    name = (name or "").strip()
    if len(name) < 2:
        return []
    j = fetch(GEOCODE_URL, {"name": name, "count": 6, "language": "en", "format": "json"})
    out = []
    for r in j.get("results") or []:
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        label = ", ".join(x for x in (r.get("name"), r.get("admin1"), r.get("country")) if x)
        out.append({"name": r.get("name") or label, "lat": round(lat, 4), "lon": round(lon, 4), "label": label})
    return out


def weather_for(calc, fetch=None):
    """(weather or None, error text or None) for the calculator's saved place."""
    place = calc.get("place") or {}
    if calc.get("weather") != "auto":
        return None, None
    if place.get("lat") is None or place.get("lon") is None:
        return None, "pick your town to use the local weather"
    try:
        return forecast(place["lat"], place["lon"], fetch=fetch), None
    except Exception as e:
        return None, f"weather unavailable ({e.__class__.__name__}); no adjustment today"


def due_for_daily_update(calc, today, hour, earliest_hour=5):
    """Should the morning update run now? Once per day, from `earliest_hour` local time."""
    return bool(calc.get("follow")) and hour >= earliest_hour and calc.get("followed_on") != today
