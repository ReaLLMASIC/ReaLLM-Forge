#!/usr/bin/env python3
"""Checks for the goal calculator's saved settings and the morning goal update in api.py.
Uses a throwaway settings folder and a canned forecast. Run: python3 test_goal_settings.py"""
import datetime as dt
import os
import sys
import tempfile

tmp = tempfile.mkdtemp()
os.environ["XDG_CONFIG_HOME"] = tmp
os.environ["BIODASH_DATA_DIR"] = tmp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the shared bt_debug.py
import water
import api as app

ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


today = dt.date.today().isoformat()
hot = {"daily": {"time": [today], "temperature_2m_max": [30.2], "temperature_2m_min": [16.0],
                 "apparent_temperature_max": [31.0]}, "current": {"temperature_2m": 27.0}}
water._get_json = lambda url, params: hot
town = {"name": "Springfield", "lat": 39.8017, "lon": -89.6437}

c = app.clean_goal_calc({"weight": 200, "weather": "auto", "place": town}, None)
check("saved", (c["weight"], c["weather"], c["place"]["lat"]), (200.0, "auto", 39.8017))
check("a partial update keeps the weight", app.clean_goal_calc({"follow": True}, c)["weight"], 200.0)
check("clearing the weight", app.clean_goal_calc({"weight": ""}, c)["weight"], None)
check("silly weight ignored", app.clean_goal_calc({"weight": 5000}, c)["weight"], 200.0)
check("old hot checkbox becomes 'hot'", app.clean_goal_calc({}, {"hot": True})["weather"], "hot")
check("bad place ignored", app.clean_goal_calc({"place": {"lat": 999, "lon": 0}}, c)["place"]["lat"], 39.8017)
check("place removed", app.clean_goal_calc({"place": None}, c)["place"], None)

s = app.settings()
s["goal_ml"], s["goal_calc"] = 2500, app.clean_goal_calc({"follow": True}, c)
app._write_json(app.SETTINGS_FILE, s)
at = lambda h: dt.datetime.now().replace(hour=h, minute=0)
check("not before 05:00", app.apply_daily_goal(at(4)), None)
# 200 lb: halfway 2977 + hot (feels 31 °C) 500 = 3477 -> 3500
check("morning update", app.apply_daily_goal(at(6)), 3500)
check("goal saved", app.settings()["goal_ml"], 3500)
check("once a day", app.apply_daily_goal(at(9)), None)

# forecast unreachable in the morning: wait for it (until noon), then go without the weather
s = app.settings(); s["goal_calc"]["followed_on"] = None; app._write_json(app.SETTINGS_FILE, s)
water._cache.clear()
def down(url, params):
    raise OSError("offline")
water._get_json = down
check("waits for the forecast", app.apply_daily_goal(at(8)), None)
check("after noon, no weather", app.apply_daily_goal(at(13)), 3000)

print(f"{ok} checks passed")
