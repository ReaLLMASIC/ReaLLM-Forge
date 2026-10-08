#!/usr/bin/env python3
"""Bio-dash service manager: choose which dashboards run as background services.

    ./biodash.py                         interactive picker (TUI)
    ./biodash.py status                  what's installed / running
    ./biodash.py install polar atmos     run these as services (replaces the current set)
          [--no-keep-awake] [--lid] [--no-reconnect] [--allow-power] [--boot | --no-boot]
    ./biodash.py stop polar | all        stop (they start again at next login / boot)
    ./biodash.py uninstall [names | all] stop and remove
    ./biodash.py logs polar [-f]         the service's output

Services are systemd *user* units (~/.config/systemd/user/biodash-<name>.service): no
root needed, and stopping one stops the whole dashboard (worker + web server).
"Start at boot" turns on lingering (loginctl enable-linger) so they run without a login.
Each service runs through service_run.sh, which keeps the computer awake while it runs.
Standard library only (curses).
"""
import argparse
import curses
import json
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
UNIT_DIR = os.environ.get("BIODASH_UNIT_DIR") or os.path.expanduser("~/.config/systemd/user")
DRY_RUN = os.environ.get("BIODASH_DRY_RUN") == "1"   # print systemctl/loginctl calls instead (testing)

DASHBOARDS = [   # (key, label, folder, port)
    ("polar",  "Polar H10",  "polar_dashboard",  5001),
    ("atmos",  "Atmos",      "atmos_dashboard",  5002),
    ("viatom", "Viatom O2",  "viatom_dashboard", 5003),
    ("omni",   "Omni (Polar + O2 together)", "Omni-dashboard", 5000),
    ("hydro",  "Hydro-dash (HidrateSpark)",  "hydro_dashboard",  5004),
]
KEYS = [d[0] for d in DASHBOARDS]
INFO = {d[0]: d for d in DASHBOARDS}
# Omni drives the Polar and the O2 itself: running it alongside the standalone
# Polar / Viatom dashboards would have two programs fighting over the same devices.
CONFLICTS = {"omni": {"polar", "viatom"}, "polar": {"omni"}, "viatom": {"omni"}, "atmos": set(), "hydro": set()}

DEFAULT_OPTS = {"keep_awake": True, "lid": False, "reconnect": True, "allow_power": False}


# ---------------------------------------------------------------------------
# systemd plumbing
# ---------------------------------------------------------------------------
def unit_name(key):
    return f"biodash-{key}.service"


def unit_path(key):
    return os.path.join(UNIT_DIR, unit_name(key))


def run(cmd, check=False):
    """Runs a command; returns (ok, output). In dry-run mode just records it."""
    if DRY_RUN:
        print("DRY-RUN:", " ".join(shlex.quote(c) for c in cmd))
        return True, ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {out}")
    return p.returncode == 0, out


def systemctl(*args):
    return run(["systemctl", "--user", *args])


def user_systemd_available():
    if DRY_RUN:
        return True, ""
    ok, out = run(["systemctl", "--user", "show-environment"])
    return ok, out


def unit_text(key, opts):
    _, label, folder, port = INFO[key]
    env = {
        "BIODASH_SERVICE": "1",
        "BIODASH_INHIBIT": "1" if opts["keep_awake"] else "0",
        "BIODASH_INHIBIT_LID": "1" if opts["lid"] else "0",
        "BIODASH_AUTORECONNECT": "1" if opts["reconnect"] else "0",
        "BIODASH_ALLOW_POWER": "1" if opts.get("allow_power") else "0",
        "PYTHONUNBUFFERED": "1",
    }
    env_lines = "\n".join(f"Environment={k}={v}" for k, v in env.items())
    return f"""# Installed by Bio-dash biodash.py -- edit with ./biodash.py, not by hand.
[Unit]
Description=Bio-dash {label} dashboard (http://localhost:{port})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={REPO}
ExecStart=/usr/bin/env bash {shlex.quote(os.path.join(REPO, "service_run.sh"))} {folder}
{env_lines}
Restart=on-failure
RestartSec=5
KillMode=control-group
TimeoutStopSec=15

[Install]
WantedBy=default.target
"""


def read_opts_from_units():
    """Options of the currently installed services (so the TUI shows what's in effect)."""
    for key in KEYS:
        try:
            text = open(unit_path(key)).read()
        except OSError:
            continue
        get = lambda k, d: (f"Environment={k}=1" in text) if f"Environment={k}=" in text else d
        return {"keep_awake": get("BIODASH_INHIBIT", True), "lid": get("BIODASH_INHIBIT_LID", False),
                "reconnect": get("BIODASH_AUTORECONNECT", True), "allow_power": get("BIODASH_ALLOW_POWER", False)}
    return dict(DEFAULT_OPTS)


def installed():
    return [k for k in KEYS if os.path.exists(unit_path(k))]


def state(key):
    """(installed, active_state, enabled_state)"""
    if not os.path.exists(unit_path(key)):
        return False, "-", "-"
    if DRY_RUN:
        return True, "unknown", "unknown"
    _, active = systemctl("is-active", unit_name(key))
    _, enabled = systemctl("is-enabled", unit_name(key))
    return True, (active.splitlines() or ["?"])[0], (enabled.splitlines() or ["?"])[0]


def current_user():
    # not os.getlogin(): it fails without a controlling terminal (scripts, services)
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def linger_enabled():
    if DRY_RUN:
        return False
    user = current_user()
    ok, out = run(["loginctl", "show-user", user, "-p", "Linger", "--value"])
    return ok and out.strip() == "yes"


def set_linger(on):
    user = current_user()
    return run(["loginctl", "enable-linger" if on else "disable-linger", user])


def validate(selected):
    sel = set(selected)
    for k in sel:
        clash = CONFLICTS[k] & sel
        if clash:
            names = ", ".join(INFO[c][1] for c in sorted(clash))
            return f"{INFO[k][1]} can't run together with {names} (they'd fight over the same devices)."
    return None


def packages_problem():
    """Each release folder has its own .venv; a freshly pulled release has none until
    ./setup_env.sh has run, and its services would start and quit straight away."""
    venv_py = os.path.join(REPO, ".venv", "bin", "python3")
    py = venv_py if os.access(venv_py, os.X_OK) else "python3"
    ok, _ = run([py, "-c", "import bleak"])
    if ok:
        return None
    return ("The Python packages for this release aren't installed here. Run ./setup_env.sh in "
            f"{REPO} first (each release has its own .venv), then Apply again.")


def apply(selected, opts, boot=None, log=print):
    """Make the installed services exactly `selected`. Returns list of problems."""
    problems = []
    err = validate(selected) or (packages_problem() if selected else None)
    if err:
        return [err]
    os.makedirs(UNIT_DIR, exist_ok=True)
    # remove what's no longer wanted
    for key in installed():
        if key not in selected:
            systemctl("disable", "--now", unit_name(key))
            try:
                os.remove(unit_path(key))
            except OSError:
                pass
            log(f"removed {unit_name(key)}")
    for key in selected:
        with open(unit_path(key), "w") as f:
            f.write(unit_text(key, opts))
    ok, out = systemctl("daemon-reload")
    if not ok:
        problems.append(f"systemctl --user daemon-reload failed: {out}")
    for key in selected:
        # restart so changed options take effect; enable so it starts at login/boot
        ok1, out1 = systemctl("enable", unit_name(key))
        ok2, out2 = systemctl("restart", unit_name(key))
        if ok1 and ok2:
            log(f"running {unit_name(key)} -> http://localhost:{INFO[key][3]}")
        else:
            problems.append(f"{unit_name(key)}: {(out1 + ' ' + out2).strip()}")
    if boot is not None:
        ok, out = set_linger(boot)
        if not ok:
            problems.append(f"loginctl {'enable' if boot else 'disable'}-linger failed: {out} "
                            f"(try: sudo loginctl {'enable' if boot else 'disable'}-linger $USER)")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_status(_args):
    ok, why = user_systemd_available()
    if not ok:
        print(f"⚠ systemd user session not available: {why}")
    print(f"{'dashboard':28s} {'port':>5s}  {'installed':9s} {'running':9s} {'at login':9s}")
    for key in KEYS:
        inst, active, enabled = state(key)
        print(f"{INFO[key][1]:28s} {INFO[key][3]:5d}  {('yes' if inst else 'no'):9s} {active:9s} {enabled:9s}")
    o = read_opts_from_units()
    print(f"\nkeep computer awake: {'on' if o['keep_awake'] else 'off'}{' (incl. lid close)' if o['lid'] else ''}"
          f" · auto-reconnect: {'on' if o['reconnect'] else 'off'} · start at boot: {'on' if linger_enabled() else 'off'}"
          f" · shutdown/reboot/updates from other devices: {'on' if o.get('allow_power') else 'off'}")


def names_from(args_names):
    if not args_names or args_names == ["all"]:
        return installed() if args_names == ["all"] else []
    bad = [n for n in args_names if n not in KEYS]
    if bad:
        sys.exit(f"Unknown dashboard(s): {', '.join(bad)}. Choose from: {', '.join(KEYS)}")
    return args_names


def cmd_install(args):
    names = names_from(args.names)
    if not names:
        sys.exit("Name at least one dashboard: " + ", ".join(KEYS))
    opts = {"keep_awake": not args.no_keep_awake, "lid": args.lid, "reconnect": not args.no_reconnect,
            "allow_power": args.allow_power}
    boot = True if args.boot else False if args.no_boot else None
    problems = apply(names, opts, boot)
    for p in problems:
        print("✗", p)
    sys.exit(1 if problems else 0)


def cmd_stop(args):
    for key in names_from(args.names) or installed():
        ok, out = systemctl("stop", unit_name(key))
        print(("stopped " if ok else "✗ ") + unit_name(key) + ("" if ok else f": {out}"))


def cmd_uninstall(args):
    keep = [k for k in installed() if k not in (names_from(args.names) or installed())]
    problems = apply(keep, read_opts_from_units())
    for p in problems:
        print("✗", p)


def cmd_logs(args):
    cmd = ["journalctl", "--user", "-u", unit_name(args.name), "-n", "80", "--no-pager"]
    if args.follow:
        cmd.append("-f")
    os.execvp(cmd[0], cmd)


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------
class Tui:
    def __init__(self, scr):
        self.scr = scr
        self.selected = set(installed())
        self.opts = read_opts_from_units()
        self.boot = linger_enabled()
        self.cursor = 0
        self.message = ""
        self.sysd_ok, self.sysd_why = user_systemd_available()
        self.states = {}
        self.refresh_states()

    # rows: the dashboards, then the options, then 2 actions
    def rows(self):
        r = [("dash", k) for k in KEYS]
        r += [("opt", "keep_awake"), ("opt", "lid"), ("opt", "reconnect"), ("opt", "allow_power"), ("opt", "boot")]
        r += [("act", "apply"), ("act", "quit")]
        return r

    def refresh_states(self):
        self.states = {k: state(k) for k in KEYS}

    def color(self, n):
        return curses.color_pair(n) if curses.has_colors() else 0

    def draw(self):
        s = self.scr
        s.erase()
        h, w = s.getmaxyx()
        put = lambda y, x, text, attr=0: s.addnstr(y, x, text, max(0, w - x - 1), attr) if 0 <= y < h else None
        put(0, 2, "Bio-dash services", curses.A_BOLD | self.color(1))
        put(1, 2, "Choose which dashboards run in the background as systemd user services.", self.color(4))
        y = 3
        if not self.sysd_ok:
            put(y, 2, f"⚠ No systemd user session: {self.sysd_why[:w - 30]}", self.color(3)); y += 2
        put(y, 4, f"{'Dashboard':30s} {'port':>5s}   {'status':22s}", curses.A_UNDERLINE); y += 1
        rows = self.rows()
        for i, (kind, key) in enumerate(rows):
            sel = i == self.cursor
            attr = curses.A_REVERSE if sel else 0
            if kind == "dash":
                inst, active, enabled = self.states[key]
                box = "[x]" if key in self.selected else "[ ]"
                status = ("running" if active == "active" else active) if inst else "not installed"
                if inst and enabled == "enabled":
                    status += " · at login"
                col = self.color(2) if active == "active" else self.color(4)
                put(y, 2, f"{box} {INFO[key][1]:28s} {INFO[key][3]:5d}   ", attr)
                put(y, 44, f"{status:24s}", attr | col)
                y += 1
                if key == KEYS[-1]:
                    y += 1
                    put(y, 2, "Options", curses.A_UNDERLINE); y += 1
            elif kind == "opt":
                label, on = {
                    "keep_awake": ("Keep this computer awake while a dashboard runs", self.opts["keep_awake"]),
                    "lid":        ("   …even with the laptop lid closed", self.opts["lid"]),
                    "reconnect":  ("Auto-reconnect to the last device", self.opts["reconnect"]),
                    "allow_power": ("Allow shutdown / reboot / updates from other devices (e.g. a phone)", self.opts.get("allow_power", False)),
                    "boot":       ("Start at boot, before anyone logs in (linger)", self.boot),
                }[key]
                put(y, 2, f"{'[x]' if on else '[ ]'} {label}", attr); y += 1
                if key == "boot":
                    y += 1
            else:
                label = {"apply": "Apply", "quit": "Quit"}[key]
                put(y, 2, f"  {label}  ", attr | curses.A_BOLD); y += 1
        y += 1
        put(y, 2, "↑/↓ move · Space toggle · Enter apply / choose · L logs for the dashboard under the cursor · Q quit", self.color(4))
        if self.message:
            for j, line in enumerate(self.message.splitlines()[:max(1, h - y - 3)]):
                put(y + 2 + j, 2, line, self.color(3) if line.startswith("✗") else self.color(2))
        s.refresh()

    def toggle(self, kind, key):
        if kind == "dash":
            if key in self.selected:
                self.selected.discard(key)
            else:
                dropped = CONFLICTS[key] & self.selected
                self.selected -= dropped
                self.selected.add(key)
                if dropped:
                    self.message = (f"{INFO[key][1]} can't run with " + ", ".join(INFO[d][1] for d in sorted(dropped))
                                    + " (same devices) -- unticked " + ("it." if len(dropped) == 1 else "them."))
                    return
            self.message = ""
        elif kind == "opt":
            if key == "boot":
                self.boot = not self.boot
            else:
                self.opts[key] = not self.opts[key]
                if key == "keep_awake" and not self.opts[key]:
                    self.opts["lid"] = False
                if key == "lid" and self.opts[key]:
                    self.opts["keep_awake"] = True

    def do_apply(self):
        lines = []
        problems = apply(sorted(self.selected, key=KEYS.index), self.opts,
                         boot=self.boot if self.boot != linger_enabled() else None, log=lines.append)
        lines += [f"✗ {p}" for p in problems]
        if not self.selected and not problems:
            lines.append("No dashboards selected -- all Bio-dash services removed.")
        self.message = "\n".join(lines) or "Nothing to change."
        self.refresh_states()

    def show_logs(self, key):
        if DRY_RUN:
            self.message = f"(dry run) would show: journalctl --user -u {unit_name(key)}"; return
        _, out = run(["journalctl", "--user", "-u", unit_name(key), "-n", "300", "--no-pager"])
        lines = out.splitlines() or ["(no output yet)"]
        top = max(0, len(lines) - 1)
        while True:
            self.scr.erase()
            h, w = self.scr.getmaxyx()
            self.scr.addnstr(0, 2, f"{unit_name(key)} -- ↑/↓/PgUp/PgDn scroll · any other key returns", w - 3, curses.A_BOLD)
            view = lines[max(0, top - (h - 3)):top + 1]
            for i, l in enumerate(view):
                self.scr.addnstr(2 + i, 1, l, w - 2)
            self.scr.refresh()
            c = self.scr.getch()
            if c == curses.KEY_UP: top = max(0, top - 1)
            elif c == curses.KEY_DOWN: top = min(len(lines) - 1, top + 1)
            elif c == curses.KEY_PPAGE: top = max(0, top - (h - 3))
            elif c == curses.KEY_NPAGE: top = min(len(lines) - 1, top + (h - 3))
            else: return

    def loop(self):
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color(); curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1); curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1); curses.init_pair(4, curses.COLOR_WHITE, -1)
        self.scr.timeout(3000)            # redraw every 3 s so the status column stays live
        while True:
            self.draw()
            c = self.scr.getch()
            rows = self.rows()
            kind, key = rows[self.cursor]
            if c == -1:
                self.refresh_states()
            elif c in (curses.KEY_UP, ord("k")):
                self.cursor = (self.cursor - 1) % len(rows)
            elif c in (curses.KEY_DOWN, ord("j"), 9):
                self.cursor = (self.cursor + 1) % len(rows)
            elif c == ord(" "):
                if kind in ("dash", "opt"):
                    self.toggle(kind, key)
            elif c in (curses.KEY_ENTER, 10, 13):
                if kind == "act" and key == "quit":
                    return
                if kind in ("dash", "opt"):
                    self.toggle(kind, key)
                else:
                    self.do_apply()
            elif c in (ord("a"), ord("A")):
                self.do_apply()
            elif c in (ord("l"), ord("L")) and kind == "dash":
                self.show_logs(key)
            elif c in (ord("q"), ord("Q")):     # not Esc: a stray escape sequence mustn't quit
                return


def main():
    ap = argparse.ArgumentParser(description="Run Bio-dash dashboards as background services.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    p = sub.add_parser("install"); p.add_argument("names", nargs="+")
    p.add_argument("--no-keep-awake", action="store_true"); p.add_argument("--lid", action="store_true")
    p.add_argument("--no-reconnect", action="store_true")
    p.add_argument("--allow-power", action="store_true", help="allow the sidebar's shutdown / reboot / updates from other devices")
    g = p.add_mutually_exclusive_group(); g.add_argument("--boot", action="store_true"); g.add_argument("--no-boot", action="store_true")
    p = sub.add_parser("stop"); p.add_argument("names", nargs="*")
    p = sub.add_parser("uninstall"); p.add_argument("names", nargs="*")
    p = sub.add_parser("logs"); p.add_argument("name", choices=KEYS); p.add_argument("-f", "--follow", action="store_true")
    args = ap.parse_args()
    if args.cmd is None:
        if not sys.stdin.isatty():
            sys.exit("The picker needs a terminal; see ./biodash.py --help for commands.")
        curses.wrapper(lambda scr: Tui(scr).loop())
        return
    {"status": cmd_status, "install": cmd_install, "stop": cmd_stop,
     "uninstall": cmd_uninstall, "logs": cmd_logs}[args.cmd](args)


if __name__ == "__main__":
    main()
