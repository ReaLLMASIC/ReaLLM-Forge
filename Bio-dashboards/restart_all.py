#!/usr/bin/env python3
"""Restart everything Bio-dash: stop every running dashboard and Bluetooth worker (from any
release folder, as a service or started by hand), reset Bluetooth, then start again whatever
was running.

    ./restart_all.py                 stop, reset Bluetooth, start again
    ./restart_all.py --stop          stop everything and leave it stopped
    ./restart_all.py --no-bluetooth  don't touch Bluetooth
    ./restart_all.py --dry-run       show what it would do, change nothing

What counts as Bio-dash: any process of yours whose program or working folder is inside a
V<version>_Dashboard folder (so an old release still running in another folder is caught too),
and the biodash-* services. The updater is left alone; this refuses to run during an update.

Stopping: services with `systemctl --user stop`; everything else gets Ctrl+C (SIGINT, so
recordings are closed properly), then SIGTERM after 6 s, then SIGKILL after 3 more.

Bluetooth reset, in order of preference:
  1. restart the bluetooth service, if allowed without a password (polkit, or this sudoers rule:
     echo "$USER ALL=(root) NOPASSWD: $(command -v systemctl) restart bluetooth" \\
       | sudo tee /etc/sudoers.d/biodash-bluetooth && sudo chmod 440 /etc/sudoers.d/biodash-bluetooth )
  2. otherwise: disconnect every connected device and switch each radio off and on again.
Either way other Bluetooth devices (headphones, a keyboard) drop for a few seconds and
reconnect on their own.

Starting again: services that were running are started; dashboards that were started by hand
(launch.sh, run_dashboard.sh, server.py ...) are started again the same way in the background,
with their output in ~/.cache/bio-dash/logs/. Omni needs sudo, so it's left for you to start.

Progress goes to ~/.cache/bio-dash/restart.json (the dashboards' sidebar reads it).
Standard library only.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

RELEASE_RE = re.compile(r"^V\d+(?:\.\d+)*_Dashboard$")
MARKERS = ("biodash.py", "launch.sh", "server.py", "bt_debug.py")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "bio-dash")
STATUS = os.path.join(CACHE, "restart.json")
LOGS = os.path.join(CACHE, "logs")
UPDATE_STATUS = os.path.join(os.path.expanduser("~"), ".config", "bio-dash", "update_status.json")
LEAVE_ALONE = ("updater.py", "restart_all.py", "biodash.py")
BT_FIX = ('echo "$USER ALL=(root) NOPASSWD: $(command -v systemctl) restart bluetooth" '
          '| sudo tee /etc/sudoers.d/biodash-bluetooth && sudo chmod 440 /etc/sudoers.d/biodash-bluetooth')


class Progress:
    def __init__(self, dry, quiet=False):
        self.dry, self.quiet = dry, quiet
        self.data = {"state": "running", "started": int(time.time()), "finished": None, "stage": "Starting",
                     "log": [], "pid": os.getpid(), "dry_run": dry, "bt_fix": None, "problems": False}
        self.save()

    def save(self):
        try:
            os.makedirs(CACHE, exist_ok=True)
            tmp = STATUS + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f)
            os.replace(tmp, STATUS)
        except OSError:
            pass

    def stage(self, text):
        self.data["stage"] = text
        self.log(text + "…")

    def log(self, text):
        if not self.quiet:
            print(text, flush=True)
        self.data["log"] = (self.data["log"] + [text])[-80:]
        self.save()

    def finish(self, state, text):
        self.data.update(state=state, stage=text, finished=int(time.time()))
        self.log(text)


def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)


# ------------------------------------------------------------------------------ processes
def release_of(path):
    """The V*_Dashboard folder a path is in, or None."""
    path = os.path.realpath(path)
    parts = path.split(os.sep)
    for i in range(len(parts) - 1, 0, -1):
        if RELEASE_RE.match(parts[i]):
            root = os.sep.join(parts[:i + 1])
            if any(os.path.exists(os.path.join(root, m)) for m in MARKERS):
                return root
    return None


def proc_info(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
        cwd = os.readlink(f"/proc/{pid}/cwd")
        with open(f"/proc/{pid}/stat") as f:
            ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return None
    except (OSError, ValueError, IndexError):
        return None
    return {"pid": pid, "ppid": ppid, "argv": argv, "cwd": cwd}


INTERPRETER = re.compile(r"^(python[\d.]*|bash|sh|dash|systemd-inhibit|uvicorn)$")


def biodash_release(info):
    """The release folder this process belongs to, or None. Only programs Bio-dash runs as
    (python, bash, systemd-inhibit, uvicorn) count, and only with a script inside a release folder
    (or `python -m ...` / uvicorn started there): an editor or a terminal open in the folder is
    left alone."""
    argv = info["argv"]
    if not argv or not INTERPRETER.match(os.path.basename(argv[0])):
        return None
    for a in argv[1:]:
        if a.startswith("-"):
            continue
        p = a if os.path.isabs(a) else os.path.join(info["cwd"], a)
        if os.path.isfile(p):
            rel = release_of(p)
            if rel:
                return rel
    if "-m" in argv[1:] or os.path.basename(argv[0]) == "uvicorn":
        return release_of(info["cwd"])
    if "-c" in argv[1:] and "multiprocessing" in " ".join(argv):   # a helper process left by a dashboard
        return release_of(info["cwd"])
    return None


def find_processes():
    me = os.getpid()     # not its parents: started from a dashboard, that dashboard is stopped too
    found = {}
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        info = proc_info(int(name))
        if not info:
            continue
        rel = biodash_release(info)
        if not rel or any(os.path.basename(a) in LEAVE_ALONE for a in info["argv"]):
            continue
        info["release"] = rel
        info["unit"] = unit_of(info["pid"])
        found[info["pid"]] = info
    # anything a Bio-dash process started (a sleep inhibitor, a helper) goes with it
    children = {}
    for name in os.listdir("/proc"):
        if name.isdigit() and int(name) != me and int(name) not in found:
            info = proc_info(int(name))
            if info:
                children.setdefault(info["ppid"], []).append(info)
    todo = list(found)
    while todo:
        for info in children.get(todo.pop(), []):
            if info["pid"] not in found and not any(os.path.basename(a) in LEAVE_ALONE for a in info["argv"]):
                parent = found[info["ppid"]]
                info.update(release=parent["release"], unit=unit_of(info["pid"]))
                found[info["pid"]] = info
                todo.append(info["pid"])
    return found


def unit_of(pid):
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            m = re.search(r"/(biodash-[\w@.-]+\.service)", f.read())
        return m.group(1) if m else None
    except OSError:
        return None


def roots(procs):
    """The outermost Bio-dash process of each tree that isn't a service: what to start again."""
    out = []
    for info in procs.values():
        if info["unit"]:
            continue
        parent = procs.get(info["ppid"])
        if parent is None or parent["unit"]:
            out.append(info)
    return sorted(out, key=lambda i: i["pid"])


def describe(info):
    argv = [os.path.basename(a) if i < 2 else a for i, a in enumerate(info["argv"])]
    return f"{' '.join(argv)[:90]} ({os.path.basename(info['release'])}, pid {info['pid']})"


def alive(pid):
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def stop_processes(pids, prog):
    waves = ((signal.SIGINT, 6), (signal.SIGTERM, 3), (signal.SIGKILL, 2))
    left = [p for p in pids if alive(p)]
    for sig, wait in waves:
        if not left:
            break
        for p in left:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        end = time.time() + wait
        while time.time() < end and any(alive(p) for p in left):
            time.sleep(0.2)
        still = [p for p in left if alive(p)]
        if still and sig != signal.SIGKILL:
            prog.log(f"{len(still)} still running after {signal.Signals(sig).name}; trying harder.")
        left = still
    return left


# ------------------------------------------------------------------------------ services
def services():
    """{unit: active?} for the biodash-* user services (not update / restart helpers)."""
    if not shutil.which("systemctl"):
        return {}
    ok, out = run(["systemctl", "--user", "list-units", "biodash-*", "--all", "--plain", "--no-legend", "--type=service"])
    units = {}
    for line in out.splitlines() if ok else []:
        parts = line.split()
        if parts and parts[0].endswith(".service") and not re.match(r"biodash-(update|restart)", parts[0]):
            units[parts[0]] = len(parts) > 2 and parts[2] == "active"
    return units


# ------------------------------------------------------------------------------ bluetooth
def bluez_objects():
    ok, out = run(["busctl", "--json=short", "call", "org.bluez", "/", "org.freedesktop.DBus.ObjectManager",
                   "GetManagedObjects"], timeout=10)
    if not ok:
        return None
    try:
        return json.loads(out)["data"][0]
    except (ValueError, KeyError, IndexError):
        return None


def adapters_and_connected():
    objs = bluez_objects()
    if objs is None:
        return None, None
    adapters, connected = [], []
    for path, ifaces in objs.items():
        if "org.bluez.Adapter1" in ifaces:
            adapters.append(path)
        dev = ifaces.get("org.bluez.Device1")
        if dev and dev.get("Connected", {}).get("data"):
            connected.append((path, dev.get("Name", {}).get("data") or dev.get("Address", {}).get("data", path)))
    return sorted(adapters), connected


def restart_bluetooth_service():
    systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
    ok, out = run([systemctl, "--no-ask-password", "restart", "bluetooth"], timeout=40)
    if ok:
        return True, "polkit"
    ok2, _ = run(["sudo", "-n", systemctl, "restart", "bluetooth"], timeout=40)
    return (True, "sudo") if ok2 else (False, out)


def set_powered(adapter, on):
    return run(["busctl", "set-property", "org.bluez", adapter, "org.bluez.Adapter1", "Powered", "b",
                "true" if on else "false"], timeout=10)


def reset_bluetooth(prog, dry):
    adapters, connected = adapters_and_connected()
    if adapters is None:
        prog.log("Bluetooth: couldn't reach BlueZ (is the bluetooth service running?); trying to restart it.")
    if dry:
        prog.log(f"DRY-RUN: would restart the bluetooth service, or disconnect {len(connected or [])} device(s) "
                 f"and switch {len(adapters or [])} radio(s) off and on")
        return True
    ok, how = restart_bluetooth_service()
    if ok:
        prog.log(f"Restarted the bluetooth service ({how}).")
    else:
        prog.data["bt_fix"] = BT_FIX
        if not adapters:
            prog.log("Bluetooth: no radios found and the service can't be restarted without a password.")
            return False
        for path, name in connected or []:
            okd, out = run(["busctl", "call", "org.bluez", path, "org.bluez.Device1", "Disconnect"], timeout=15)
            prog.log(f"Disconnected {name}." if okd else f"Couldn't disconnect {name}: {out[:120]}")
        for a in adapters:
            okf, out = set_powered(a, False)
            if not okf:
                prog.log(f"Couldn't switch {a.rsplit('/', 1)[-1]} off: {out[:120]}")
        time.sleep(2)
        for a in adapters:
            set_powered(a, True)
        prog.log(f"Switched {len(adapters)} radio(s) off and on (the bluetooth service needs a password to restart; "
                 f"see the sudoers line to allow it).")
    # wait for the radios to be back
    end = time.time() + 15
    while time.time() < end:
        objs = bluez_objects()
        if objs:
            ads = [i["org.bluez.Adapter1"] for i in objs.values() if "org.bluez.Adapter1" in i]
            if ads and all(a.get("Powered", {}).get("data") for a in ads):
                prog.log(f"{len(ads)} radio(s) on.")
                return True
        time.sleep(0.5)
    prog.log("Some radios didn't come back on within 15 s (rfkill? unplugged?).")
    return False


# ------------------------------------------------------------------------------ start again
def environ_of(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            pairs = [p.split(b"=", 1) for p in f.read().split(b"\0") if b"=" in p]
        return {k.decode(errors="replace"): v.decode(errors="replace") for k, v in pairs}
    except OSError:
        return None


def relaunch(info, env, prog, n):
    argv = info["argv"]
    if any("omni" in a.lower() for a in argv[:4]) or os.path.basename(info["cwd"]) == "Omni-dashboard":
        prog.log(f"Not starting {describe(info)} again: Omni needs sudo. Start it yourself.")
        return False
    os.makedirs(LOGS, exist_ok=True)
    where = os.path.relpath(info["cwd"], info["release"])
    name = re.sub(r"[^\w.-]+", "_", os.path.basename(info["release"]) + ("" if where == "." else "_" + where))[:80]
    log_path = os.path.join(LOGS, f"restart_{name or n}.log")
    env = dict(env or os.environ)
    if os.path.basename(argv[0]) != "systemd-inhibit":
        env.pop("BIODASH_INHIBITED", None)    # launch.sh takes the inhibitor again
    try:
        with open(log_path, "ab") as log:
            log.write(f"\n==== started again by restart_all.py at {time.ctime()} ====\n".encode())
            log.flush()
            subprocess.Popen(argv, cwd=info["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as e:
        prog.log(f"Couldn't start {describe(info)} again: {e}")
        return False
    where = os.path.relpath(info["cwd"], info["release"])
    prog.log(f"Started again: {' '.join(os.path.basename(a) for a in argv[:2])} in "
             f"{os.path.basename(info['release'])}{'' if where == '.' else '/' + where} "
             f"(output: ~/.cache/bio-dash/logs/{os.path.basename(log_path)}).")
    return True


# ------------------------------------------------------------------------------ main
def update_running():
    try:
        with open(UPDATE_STATUS) as f:
            st = json.load(f)
        return st.get("state") == "running" and alive(int(st.get("pid", 0)))
    except (OSError, ValueError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser(description="Restart everything Bio-dash and reset Bluetooth.",
                                 epilog="Full notes: the top of this file.")
    ap.add_argument("--stop", action="store_true", help="stop everything and leave it stopped")
    ap.add_argument("--no-bluetooth", action="store_true", help="don't reset Bluetooth")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run or os.environ.get("BIODASH_DRY_RUN") == "1"
    prog = Progress(dry, a.quiet)
    if update_running():
        prog.finish("failed", "An update is running; try again when it has finished.")
        sys.exit(1)

    prog.stage("Looking for running dashboards")
    units = services()
    active_units = [u for u, on in units.items() if on]
    procs = find_processes()
    by_hand = roots(procs)
    envs = {r["pid"]: environ_of(r["pid"]) for r in by_hand}
    prog.log(f"Found {len(active_units)} running service(s) and {len(procs)} Bio-dash process(es)"
             + (f", {len(by_hand)} started by hand." if by_hand else "."))
    for u in active_units:
        prog.log(f"  service {u}")
    for info in sorted(procs.values(), key=lambda i: i["pid"]):
        if not info["unit"]:
            prog.log(f"  {describe(info)}")

    prog.stage("Stopping")
    if dry:
        prog.log(f"DRY-RUN: would stop {len(active_units)} service(s) and {len(procs)} process(es)")
    else:
        for u in active_units:
            ok, out = run(["systemctl", "--user", "stop", u], timeout=60)
            prog.log(f"Stopped {u}." if ok else f"Couldn't stop {u}: {out[:160]}")
        left = stop_processes([p for p in find_processes()], prog)
        if left:
            prog.log(f"{len(left)} process(es) couldn't be stopped: {', '.join(map(str, left))}")

    bt_ok = True
    if not a.no_bluetooth:
        prog.stage("Resetting Bluetooth")
        bt_ok = reset_bluetooth(prog, dry)

    if a.stop:
        prog.finish("done", "Everything is stopped." + ("" if bt_ok else " (Bluetooth reset had problems; see above.)"))
        return
    prog.stage("Starting again")
    started = 0
    for u in active_units:
        if dry:
            prog.log(f"DRY-RUN: would start {u}")
            continue
        ok, out = run(["systemctl", "--user", "start", u], timeout=60)
        prog.log(f"Started {u}." if ok else f"Couldn't start {u}: {out[:160]}")
        started += ok
    for i, r in enumerate(by_hand):
        if dry:
            prog.log(f"DRY-RUN: would start again: {describe(r)}")
            continue
        started += relaunch(r, envs.get(r["pid"]), prog, i)
    if not active_units and not by_hand:
        msg = "Nothing was running." + (" Bluetooth was reset." if not a.no_bluetooth and not dry and bt_ok else "")
    else:
        msg = f"Restarted: {started} of {len(active_units) + len(by_hand)} started again."
        if started < len(active_units) + len(by_hand):
            prog.data["problems"] = True
    if not bt_ok:
        prog.data["problems"] = True
        msg += " The Bluetooth reset had problems; see the log."
    if dry:
        prog.data["problems"] = False
        msg = "Dry run: nothing was changed."
    prog.finish("done" if not dry else "dry_run", msg)


if __name__ == "__main__":
    main()
