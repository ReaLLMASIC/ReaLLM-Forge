#!/usr/bin/env python3
"""Run Bio-dash as a background service (one service for everything).

    ./biodash.py                     set it up, answering a few questions
    ./biodash.py install             install / update the service and start it
          [--port 5000] [--no-keep-awake] [--lid] [--no-reconnect] [--allow-power] [--boot | --no-boot]
    ./biodash.py status              is it running, and where to open it
    ./biodash.py start | stop | restart
    ./biodash.py logs [-f]           its output (every device's Bluetooth messages too)
    ./biodash.py uninstall           stop and remove the service (recordings and settings stay)

The service is a systemd *user* unit (~/.config/systemd/user/biodash-app.service), so no root is
needed. "Start at boot" turns on lingering (loginctl enable-linger) so it runs without a login.
Which devices it looks for is chosen in the app (Settings, or the first-run setup).

Upgrading from V5: `install` removes the old per-dashboard services (biodash-polar ... ) and
turns on the devices they ran, so `./biodash.py install polar hydro` (what V5's updater runs)
still does the right thing. Standard library only.
"""
import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
UNIT_DIR = os.environ.get("BIODASH_UNIT_DIR") or os.path.expanduser("~/.config/systemd/user")
DRY_RUN = os.environ.get("BIODASH_DRY_RUN") == "1"
UNIT = "biodash-app.service"
DEFAULT_PORT = 5000
CONFIG_FILE = os.environ.get("BIODASH_APP_CONFIG") or os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "bio-dash", "app.json")
# the separate dashboards of V5 - V7 -> the devices they drove
LEGACY = {"polar": ["heart"], "viatom": ["oxygen"], "omni": ["heart", "oxygen"], "muse": ["brain"], "atmos": ["air"],
          "hydro": ["water"], "analysis": []}
OLD_UNITS = [f"biodash-{k}.service" for k in LEGACY]


def run(cmd):
    if DRY_RUN:
        print("DRY-RUN:", " ".join(shlex.quote(c) for c in cmd))
        return True, ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def systemctl(*args):
    return run(["systemctl", "--user", *args])


def unit_path(name=UNIT):
    return os.path.join(UNIT_DIR, name)


def current_user():
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def linger_enabled():
    if DRY_RUN:
        return False
    ok, out = run(["loginctl", "show-user", current_user(), "-p", "Linger", "--value"])
    return ok and out.strip() == "yes"


def python():
    venv = os.path.join(REPO, ".venv", "bin", "python3")
    return venv if os.access(venv, os.X_OK) else "python3"


def installed_opts():
    """The options the installed service was set up with (so a reinstall keeps them)."""
    try:
        text = open(unit_path()).read()
    except OSError:
        return None
    env = dict(re.findall(r"^Environment=(\w+)=(.*)$", text, re.M))
    return {"keep_awake": env.get("BIODASH_INHIBIT", "1") == "1", "lid": env.get("BIODASH_INHIBIT_LID") == "1",
            "reconnect": env.get("BIODASH_AUTORECONNECT", "1") == "1", "allow_power": env.get("BIODASH_ALLOW_POWER") == "1",
            "port": int(env.get("BIODASH_PORT", DEFAULT_PORT))}


def unit_text(o):
    env = {"BIODASH_SERVICE": "1", "BIODASH_PORT": str(o["port"]),
           "BIODASH_INHIBIT": "1" if o["keep_awake"] else "0", "BIODASH_INHIBIT_LID": "1" if o["lid"] else "0",
           "BIODASH_AUTORECONNECT": "1" if o["reconnect"] else "0", "BIODASH_ALLOW_POWER": "1" if o["allow_power"] else "0",
           "PYTHONUNBUFFERED": "1"}
    lines = "\n".join(f"Environment={k}={v}" for k, v in env.items())
    return f"""# Installed by Bio-dash biodash.py -- change it with ./biodash.py, not by hand.
[Unit]
Description=Bio-dash (http://localhost:{o['port']})
After=network-online.target bluetooth.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={REPO}
ExecStart=/usr/bin/env bash {shlex.quote(os.path.join(REPO, 'service_run.sh'))}
{lines}
Restart=on-failure
RestartSec=5
# SIGTERM to the server only: it stops each device's Bluetooth worker cleanly (recordings closed)
KillMode=mixed
TimeoutStopSec=20

[Install]
WantedBy=default.target
"""


def enable_devices(keys):
    """Turn devices on in the app's settings (used when V5 services are carried over)."""
    if not keys:
        return
    try:
        cfg = json.load(open(CONFIG_FILE))
    except Exception:
        cfg = {"devices": {}, "setup_done": False}
    for k in keys:
        cfg.setdefault("devices", {})[k] = True
    cfg["setup_done"] = True
    if DRY_RUN:
        print(f"DRY-RUN: devices on: {', '.join(keys)}")
        return
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE + ".tmp", "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(CONFIG_FILE + ".tmp", CONFIG_FILE)


def packages_problem():
    ok, _ = run([python(), "-c", "import bleak, fastapi, uvicorn"])
    if ok:
        return None
    return f"The Python packages aren't installed for this release. Run ./setup_env.sh in {REPO} first, then install again."


def url(port):
    return f"http://{socket.gethostname()}.local:{port}  (or http://localhost:{port} on this computer)"


# ---------------------------------------------------------------------------------- commands
def cmd_install(a):
    names = [n for n in a.names if n != "app"]
    bad = [n for n in names if n not in LEGACY]
    if bad:
        sys.exit(f"Unknown name(s): {', '.join(bad)}. (Devices are chosen in the app now; just run ./biodash.py install.)")
    err = packages_problem()
    if err:
        sys.exit(err)
    o = installed_opts() or {"keep_awake": True, "lid": False, "reconnect": True, "allow_power": False, "port": DEFAULT_PORT}
    if a.port: o["port"] = a.port
    if a.no_keep_awake: o["keep_awake"] = False
    if a.lid: o["lid"] = True
    if a.no_reconnect: o["reconnect"] = False
    if a.allow_power: o["allow_power"] = True
    # carry the separate dashboards' services over: their devices are turned on, the old units removed
    devices = sorted({d for n in names for d in LEGACY[n]})
    for old in OLD_UNITS:
        if os.path.exists(unit_path(old)):
            devices = sorted(set(devices) | set(LEGACY[old[len('biodash-'):-len('.service')]]))
            systemctl("disable", "--now", old)
            if not DRY_RUN:
                os.remove(unit_path(old))
            print(f"removed the old service {old}")
    enable_devices(devices)
    os.makedirs(UNIT_DIR, exist_ok=True)
    if DRY_RUN:
        print(unit_text(o))
    else:
        with open(unit_path(), "w") as f:
            f.write(unit_text(o))
    problems = []
    for args in (("daemon-reload",), ("enable", UNIT), ("restart", UNIT)):
        ok, out = systemctl(*args)
        if not ok:
            problems.append(f"systemctl --user {' '.join(args)}: {out}")
    if a.boot is not None:
        ok, out = run(["loginctl", "enable-linger" if a.boot else "disable-linger", current_user()])
        if not ok:
            problems.append(f"start at boot: {out} (try: sudo loginctl {'enable' if a.boot else 'disable'}-linger $USER)")
    for p in problems:
        print("⚠ " + p)
    if not problems:
        print(f"✅ Bio-dash is running. Open {url(o['port'])}")
    return 1 if problems else 0


def cmd_status(_a):
    o = installed_opts()
    if not o:
        print("Not installed as a service. Run ./biodash.py to set it up (or python3 server.py to try it in this terminal).")
        return 0
    _, active = systemctl("is-active", UNIT)
    _, enabled = systemctl("is-enabled", UNIT)
    print(f"Bio-dash service: {active or '?'} (starts at login: {enabled or '?'}; at boot: {'yes' if linger_enabled() else 'no'})")
    print(f"Open: {url(o['port'])}")
    print(f"Keep the computer awake: {'yes' if o['keep_awake'] else 'no'}{' (lid closed too)' if o['lid'] else ''} · "
          f"auto-reconnect: {'on' if o['reconnect'] else 'off'} · shut down / update from other devices: {'yes' if o['allow_power'] else 'no'}")
    left = [u for u in OLD_UNITS if os.path.exists(unit_path(u))]
    if left:
        print(f"⚠ Old V5 services are still installed ({', '.join(left)}): run ./biodash.py install to replace them.")
    return 0


def cmd_simple(verb):
    def f(_a):
        ok, out = systemctl(verb, UNIT)
        print(out or (f"{verb}ed" if ok else "failed"))
        return 0 if ok else 1
    return f


def cmd_uninstall(_a):
    for u in [UNIT] + OLD_UNITS:
        if os.path.exists(unit_path(u)):
            systemctl("disable", "--now", u)
            if not DRY_RUN:
                os.remove(unit_path(u))
            print(f"removed {u}")
    systemctl("daemon-reload")
    print("Recordings and settings are untouched.")
    return 0


def cmd_logs(a):
    cmd = ["journalctl", "--user", "-u", UNIT, "-n", "200", "--no-pager"] + (["-f"] if a.follow else [])
    if DRY_RUN:
        print("DRY-RUN:", " ".join(cmd))
        return 0
    return subprocess.call(cmd)


def ask(q, default):
    try:
        a = input(f"{q} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    except EOFError:
        return default
    return default if not a else a.startswith("y")


def cmd_wizard(_a):
    print("Bio-dash: set up the background service.\n")
    o = installed_opts() or {}
    ns = argparse.Namespace(names=[], port=None, no_keep_awake=not ask("Keep this computer awake while it records?", o.get("keep_awake", True)),
                            lid=False, no_reconnect=False, allow_power=False, boot=None)
    if not ns.no_keep_awake:
        ns.lid = ask("…even with the laptop lid closed?", o.get("lid", False))
    ns.boot = ask("Start at boot, before anyone logs in (good for a headless Orin / Pi)?", linger_enabled())
    ns.allow_power = ask("Allow shut down, restart and updates from other devices (your phone)?", o.get("allow_power", False))
    return cmd_install(ns)


def main():
    ap = argparse.ArgumentParser(description="Run Bio-dash as a background service.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("install")
    p.add_argument("names", nargs="*", help=argparse.SUPPRESS)        # V5 dashboard names (upgrades)
    p.add_argument("--port", type=int)
    p.add_argument("--no-keep-awake", action="store_true")
    p.add_argument("--lid", action="store_true")
    p.add_argument("--no-reconnect", action="store_true")
    p.add_argument("--allow-power", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--boot", dest="boot", action="store_true", default=None)
    g.add_argument("--no-boot", dest="boot", action="store_false")
    sub.add_parser("status")
    for v in ("start", "stop", "restart"):
        sub.add_parser(v)
    sub.add_parser("uninstall")
    p = sub.add_parser("logs"); p.add_argument("-f", "--follow", action="store_true"); p.add_argument("name", nargs="?", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cmd is None:
        return cmd_wizard(a) if sys.stdin.isatty() else cmd_status(a)
    return {"install": cmd_install, "status": cmd_status, "start": cmd_simple("start"), "stop": cmd_simple("stop"),
            "restart": cmd_simple("restart"), "uninstall": cmd_uninstall, "logs": cmd_logs}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
