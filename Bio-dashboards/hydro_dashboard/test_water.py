#!/usr/bin/env python3
"""Checks for water.py with values worked out by hand. No network: Open-Meteo replies are canned.
Run: python3 test_water.py"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the shared bt_debug.py
import water as w

ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


me = {"weight": 200, "weight_unit": "lb", "reference": "men", "exercise_min": 0, "weather": "off"}
# 200 lb = 90.72 kg -> x33 = 2994; reference 3700 x 0.8 = 2960; halfway 2977 -> 3000
g = w.suggest_goal(me)
check("reference from drinks", g["reference"], 2960)
check("by weight", g["by_weight"], 2994)
check("goal", g["goal"], 3000)
# + 30 min exercise (360) + hot by hand (500) = 3837 -> 3850
check("exercise + hot by hand", w.suggest_goal(dict(me, exercise_min=30, weather="hot"))["goal"], 3850)
check("kg, women", w.suggest_goal({"weight": 60, "weight_unit": "kg", "reference": "women"})["goal"], 2050)
check("no weight", w.suggest_goal({"weight": "", "reference": "men"})["goal"], 2950)
big = w.suggest_goal({"weight": 400, "weight_unit": "kg", "exercise_min": 600, "weather": "hot"})
check("capped", (big["goal"], big["clamped"]), (5000, True))

# the day's feels-like high
check("mild", w.weather_extra(20), (0, "mild"))
check("cold adds and removes nothing", w.weather_extra(4), (0, "cold"))
check("warm", w.weather_extra(25), (250, "warm"))
check("hot", w.weather_extra(30.5), (500, "hot"))
check("very hot", w.weather_extra(33), (750, "very hot"))
check("extreme", w.weather_extra(38), (1000, "extreme heat"))
check("unknown", w.weather_extra(None), (0, None))
hot_day = {"feels_max_c": 31.0}
check("auto uses the forecast", w.suggest_goal(dict(me, weather="auto"), hot_day)["goal"], 3500)   # 2977 + 500
check("auto without a forecast", w.suggest_goal(dict(me, weather="auto"), None)["goal"], 3000)
check("off ignores the forecast", w.suggest_goal(me, hot_day)["goal"], 3000)

# Open-Meteo replies (shape as documented: daily arrays indexed by day, current values)
today = dt.date.today().isoformat()
reply = {"latitude": 39.80, "longitude": -89.64, "timezone": "America/Chicago",
         "current": {"time": f"{today}T14:00", "interval": 900, "temperature_2m": 17.3},
         "daily": {"time": [today], "temperature_2m_max": [19.8], "temperature_2m_min": [12.1],
                   "apparent_temperature_max": [18.6]}}
f = w.parse_forecast(reply)
check("forecast parsed", (f["date"], f["max_c"], f["min_c"], f["feels_max_c"], f["now_c"]), (today, 19.8, 12.1, 18.6, 17.3))
no_feels = {"daily": {"time": [today], "temperature_2m_max": [29.0], "temperature_2m_min": [20.0]}}
check("feels-like falls back to the high", w.parse_forecast(no_feels)["feels_max_c"], 29.0)
try:
    w.parse_forecast({"error": True, "reason": "bad"})
    raise AssertionError("an error reply should raise")
except ValueError:
    ok += 1

calls = []
def fake(url, params):
    calls.append((url, params))
    return reply
w._cache.clear()
a = w.forecast(39.8017, -89.6437, fetch=fake)
b = w.forecast(39.8017, -89.6437, fetch=fake)
check("cached for 30 min", (len(calls), a["max_c"], b["max_c"]), (1, 19.8, 19.8))
check("asks for the place's own day", (calls[0][1]["timezone"], calls[0][1]["forecast_days"]), ("auto", 1))
w._cache.clear()
check("stale after 30 min", (w.forecast(39.8017, -89.6437, fetch=fake, now=lambda: 0),
                             w.forecast(39.8017, -89.6437, fetch=fake, now=lambda: 3600)) and len(calls), 3)

geo = {"results": [{"name": "Springfield", "latitude": 39.80172, "longitude": -89.64371, "admin1": "Illinois",
                    "country": "United States"},
                   {"name": "Springfield", "latitude": 37.21533, "longitude": -93.29824, "admin1": "Missouri",
                    "country": "United States"}]}
p = w.find_places("Springfield", fetch=lambda u, q: geo)
check("places", [x["label"] for x in p], ["Springfield, Illinois, United States", "Springfield, Missouri, United States"])
check("place coordinates", (p[0]["lat"], p[0]["lon"]), (39.8017, -89.6437))
check("no results", w.find_places("zzzz", fetch=lambda u, q: {}), [])
check("too short to search", w.find_places("a", fetch=lambda u, q: geo), [])

def down(url, params):
    raise OSError("no network")
w._cache.clear()
calc = dict(me, weather="auto", place={"name": "Springfield", "lat": 39.8017, "lon": -89.6437})
check("offline: no adjustment, says why", w.weather_for(calc, fetch=down)[0], None)
check("offline message", "unavailable" in w.weather_for(calc, fetch=down)[1], True)
check("no town yet", w.weather_for(dict(me, weather="auto"))[1], "pick your town to use the local weather")

# the morning update: once a day, not before 05:00
check("due", w.due_for_daily_update({"follow": True}, "2026-10-11", 7), True)
check("too early", w.due_for_daily_update({"follow": True}, "2026-10-11", 4), False)
check("already done today", w.due_for_daily_update({"follow": True, "followed_on": "2026-10-11"}, "2026-10-11", 9), False)
check("not following", w.due_for_daily_update({"follow": False}, "2026-10-11", 9), False)

print(f"{ok} checks passed")
