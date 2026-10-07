#!/usr/bin/env python3
"""Bio-dash updater: looks for patches and new versions on GitHub and applies them.

    ./updater.py check              what's available (add --json for the raw result)
    ./updater.py patch              update this version in place, then restart the services
    ./updater.py upgrade            move to the newest version folder (e.g. V3.0 -> V3.1)
    ./updater.py status             progress / result of the last patch or upgrade

Two kinds of update, kept apart on purpose:
  patch    new commits that touch THIS release folder. Same folder, same services:
           git pull, refresh .venv if requirements changed, restart.
  upgrade  a newer V*_Dashboard folder exists. Pull, build its .venv, copy your settings
           across, re-point the services at it and restart. The old folder stays on disk.

It needs a git clone (not an unzipped copy) and only ever fast-forwards from the clone's
own upstream: it refuses when tracked files were edited locally or the branch has diverged,
so nothing of yours is overwritten. Recordings live in Documents/Bio-dash and aren't touched.
The sidebar's "Updates" section in every dashboard calls this script. Standard library only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.environ.get("BIODASH_STATE_DIR") or os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "bio-dash")
STATUS_FILE = os.path.join(STATE_DIR, "update_status.json")
CHECK_FILE = os.path.join(STATE_DIR, "update_check.json")
DRY_RUN = os.environ.get("BIODASH_DRY_RUN") == "1"        # don't touch systemd (testing)
VERSION_RE = re.compile(r"^V(\d+(?:\.\d+)*)_Dashboard$")
# runtime scratch files that are NOT settings (never copied to a new version)
VOLATILE = re.compile(r"^(data|devices|command|debug|status)[\w-]*\.json$")
SKIP_DIRS = {".venv", "__pycache__", "node_modules", "vendor", ".git"}


class Refuse(Exception):
    """An update that can't go ahead; the message says why and what to do."""


def git(*args, cwd=HERE, timeout=60, check=True):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise Refuse(f"git {' '.join(args[:2])} failed: {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout.strip()


def vtuple(v):
    return tuple(int(x) for x in v.split("."))


def layout():
    """Where this copy lives: repo root, release folder (relative), version, upstream ref."""
    if not shutil.which("git"):
        raise Refuse("git isn't installed, so updates can't be checked.")
    try:
        top = git("rev-parse", "--show-toplevel")
    except Refuse:
        raise Refuse("This copy isn't a git clone (it was probably unzipped), so it can't update itself. "
                     "Clone https://github.com/Meapy011/Bio-dash and run the dashboards from there.")
    folder = os.path.relpath(HERE, top)
    m = VERSION_RE.match(os.path.basename(HERE))
    version = m.group(1) if m else None
    if version is None:                       # unversioned mirror (e.g. a synced copy): read the changelog
        try:
            head = re.search(r"^## V(\d+(?:\.\d+)*)", open(os.path.join(HERE, "CHANGELOG.md")).read(), re.M)
            version = head.group(1) if head else None
        except OSError:
            pass
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        raise Refuse("This clone is on a detached commit (a tag?), not a branch. Run: git checkout main")
    up = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False) or f"origin/{branch}"
    return {"top": top, "folder": "" if folder == "." else folder, "versioned": bool(m),
            "version": version, "branch": branch, "upstream": up}


def changelog_section(text, version):
    m = re.search(rf"^## V{re.escape(version)}\b.*?$(.*?)(?=^## V|\Z)", text, re.M | re.S)
    if not m:
        return ""
    lines = [l.rstrip() for l in m.group(1).strip().splitlines() if l.strip()]
    return "\n".join(lines[:14])[:1400]


def check(fetch=True):
    """Everything the UI needs. Never raises: problems come back as {"error": ...}."""
    out = {"ok": False, "checked_at": int(time.time()), "version": None, "patch": None, "upgrade": None,
           "blockers": [], "services": installed_units()}
    try:
        lay = layout()
        out.update(version=lay["version"], folder=lay["folder"] or ".", branch=lay["branch"])
        if fetch:
            remote = lay["upstream"].split("/", 1)[0]
            git("fetch", "--quiet", "--prune", remote, cwd=lay["top"], timeout=90)
        up, top, folder = lay["upstream"], lay["top"], lay["folder"]
        if not git("rev-parse", "--verify", "--quiet", up, cwd=top, check=False):
            raise Refuse(f"This clone has nothing to compare against ({up} doesn't exist). "
                         "It needs a GitHub remote: git remote -v")
        behind = int(git("rev-list", "--count", f"HEAD..{up}", cwd=top))
        ahead = int(git("rev-list", "--count", f"{up}..HEAD", cwd=top))
        # patch: upstream commits that touch this release folder
        log = git("log", "--format=%h%x09%ad%x09%s", "--date=short", f"HEAD..{up}", "--", folder or ".", cwd=top)
        commits = [dict(zip(("id", "date", "subject"), l.split("\t", 2))) for l in log.splitlines() if l]
        if commits:
            out["patch"] = {"count": len(commits), "commits": commits[:12]}
        # upgrade: a newer V*_Dashboard folder upstream
        if lay["versioned"] and lay["version"]:
            parent = os.path.dirname(folder)
            names = git("ls-tree", "-d", "--name-only", f"{up}:{parent}" if parent else up, cwd=top).splitlines()
            newer = sorted((vtuple(m.group(1)), n) for n in names for m in [VERSION_RE.match(n)]
                           if m and vtuple(m.group(1)) > vtuple(lay["version"]))
            if newer:
                name = newer[-1][1]
                ver = VERSION_RE.match(name).group(1)
                path = os.path.join(parent, name) if parent else name
                notes = git("show", f"{up}:{path}/CHANGELOG.md", cwd=top, check=False)
                out["upgrade"] = {"version": ver, "folder": path, "notes": changelog_section(notes, ver)}
        # reasons an update would be refused
        dirty = [l.split(None, 1)[-1] for l in git("status", "--porcelain", "--untracked-files=no", cwd=top).splitlines() if l.strip()]
        if dirty:
            out["blockers"].append("Files in the clone were edited locally: " + ", ".join(dirty[:5])
                                   + (" …" if len(dirty) > 5 else "") + ". Commit or undo them first (git status).")
        if ahead and behind:
            out["blockers"].append(f"The clone has {ahead} local commit(s) GitHub doesn't have, and is {behind} behind. "
                                   "Sort that out by hand (git status) before updating.")
        out.update(ok=True, behind=behind, ahead=ahead)
    except Refuse as e:
        out["error"] = str(e)
    except subprocess.TimeoutExpired:
        out["error"] = "GitHub didn't answer in time. Is this computer online?"
    except Exception as e:                                    # never take the dashboard down over a check
        out["error"] = f"{type(e).__name__}: {e}"
    write_json(CHECK_FILE, out)
    return out


# ---------------------------------------------------------------------------
# services
# ---------------------------------------------------------------------------
def unit_dir():
    return os.environ.get("BIODASH_UNIT_DIR") or os.path.expanduser("~/.config/systemd/user")


def installed_units():
    try:
        return sorted(f[len("biodash-"):-len(".service")] for f in os.listdir(unit_dir())
                      if f.startswith("biodash-") and f.endswith(".service") and not f.startswith("biodash-update"))
    except OSError:
        return []


def unit_opts():
    """The options the installed services were set up with, as biodash.py install flags."""
    for key in installed_units():
        try:
            text = open(os.path.join(unit_dir(), f"biodash-{key}.service")).read()
        except OSError:
            continue
        flags = []
        if "Environment=BIODASH_INHIBIT=0" in text: flags.append("--no-keep-awake")
        if "Environment=BIODASH_INHIBIT_LID=1" in text: flags.append("--lid")
        if "Environment=BIODASH_AUTORECONNECT=0" in text: flags.append("--no-reconnect")
        if "Environment=BIODASH_ALLOW_POWER=1" in text: flags.append("--allow-power")
        return flags
    return []


def restart_services(log):
    keys = installed_units()
    if not keys:
        log("No Bio-dash services are installed: restart the dashboards yourself (Ctrl+C, then launch again).")
        return False
    for key in keys:
        unit = f"biodash-{key}.service"
        if DRY_RUN:
            log(f"DRY-RUN: systemctl --user restart {unit}")
            continue
        p = subprocess.run(["systemctl", "--user", "restart", unit], capture_output=True, text=True, timeout=60)
        log(f"Restarted {key}." if p.returncode == 0 else f"Couldn't restart {key}: {(p.stderr or p.stdout).strip()[:200]}")
    return True


# ---------------------------------------------------------------------------
# status file (the dashboards poll it to show progress, and the result after the restart)
# ---------------------------------------------------------------------------
def write_json(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Progress:
    def __init__(self, action, from_version):
        self.data = {"state": "running", "action": action, "from": from_version, "to": None,
                     "started": int(time.time()), "finished": None, "stage": "Starting", "log": [], "pid": os.getpid()}
        self.save()

    def save(self):
        write_json(STATUS_FILE, self.data)

    def stage(self, text):
        self.data["stage"] = text
        self.log(text)

    def log(self, text):
        print(text, flush=True)
        self.data["log"] = (self.data["log"] + [text])[-40:]
        self.save()

    def finish(self, state, message):
        self.data.update(state=state, stage=message, finished=int(time.time()))
        self.log(message)


def busy():
    s = read_json(STATUS_FILE)
    if not s or s.get("state") != "running":
        return False
    try:
        os.kill(int(s.get("pid", 0)), 0)                   # is that updater still alive?
    except (OSError, ValueError):
        return False
    return time.time() - s.get("started", 0) < 1800


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------
def guard(info):
    if info.get("error"):
        raise Refuse(info["error"])
    if info["blockers"]:
        raise Refuse(" ".join(info["blockers"]))


def pull(lay, prog):
    old = git("rev-parse", "HEAD", cwd=lay["top"])
    prog.stage("Downloading the update (git pull)")
    remote, _, ref = lay["upstream"].partition("/")
    git("pull", "--ff-only", "--quiet", remote, ref or lay["branch"], cwd=lay["top"], timeout=180)
    new = git("rev-parse", "HEAD", cwd=lay["top"])
    prog.log(f"Updated {old[:7]} → {new[:7]}." if old != new else "Already downloaded.")
    return old, new


def build_env(folder_abs, prog, required):
    """(Re)builds .venv for a release folder. required=False: a failure is a warning."""
    script = os.path.join(folder_abs, "setup_env.sh")
    if not os.path.exists(script):
        return
    prog.stage("Installing Python packages (this can take a few minutes)")
    p = subprocess.run(["bash", script], cwd=folder_abs, capture_output=True, text=True, timeout=1500)
    if p.returncode != 0:
        tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
        msg = "Installing packages failed: " + " | ".join(tail)
        if required:
            raise Refuse(msg + " — nothing was switched; the current version keeps running.")
        prog.log("⚠ " + msg)


def do_patch():
    info = check(fetch=True)
    guard(info)
    if not info["patch"]:
        raise Refuse("No patch is available for this version.")
    lay = layout()
    prog = Progress("patch", info["version"])
    try:
        old, new = pull(lay, prog)
        changed = git("diff", "--name-only", old, new, "--", lay["folder"] or ".", cwd=lay["top"]).splitlines()
        if any(c.endswith("requirements.txt") for c in changed) and os.path.isdir(os.path.join(HERE, ".venv")):
            build_env(HERE, prog, required=False)
        prog.data["to"] = info["version"]
        prog.stage("Restarting the dashboards")
        restarted = restart_services(prog.log)
        prog.finish("done", f"Patch applied ({len(changed)} file(s) changed)." + ("" if restarted else " Restart the dashboards to use it."))
    except Exception as e:
        prog.finish("error", str(e) if isinstance(e, Refuse) else f"{type(e).__name__}: {e}")
        raise
    check(fetch=False)


def copy_settings(old_abs, new_abs, log):
    """Settings the dashboards saved next to their code (untracked *.json), minus scratch files."""
    copied = 0
    for root, dirs, files in os.walk(old_abs):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".json") or VOLATILE.match(name):
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, old_abs)
            if git("ls-files", "--error-unmatch", rel, cwd=old_abs, check=False):
                continue                                         # part of the release, not a setting
            dst = os.path.join(new_abs, rel)
            if os.path.isdir(os.path.dirname(dst)) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
    log(f"Copied {copied} settings file(s) to the new version.")


def do_upgrade():
    info = check(fetch=True)
    guard(info)
    if not info["upgrade"]:
        raise Refuse("No newer version is available.")
    lay = layout()
    target = info["upgrade"]
    prog = Progress("upgrade", info["version"])
    prog.data["to"] = target["version"]
    try:
        pull(lay, prog)
        new_abs = os.path.join(lay["top"], target["folder"])
        if not os.path.isfile(os.path.join(new_abs, "biodash.py")):
            raise Refuse(f"{target['folder']} wasn't found after the pull.")
        build_env(new_abs, prog, required=True)
        prog.stage("Carrying your settings across")
        copy_settings(HERE, new_abs, prog.log)
        keys = installed_units()
        if keys:
            prog.stage(f"Pointing the services at V{target['version']} and restarting")
            env = dict(os.environ)
            p = subprocess.run([sys.executable, os.path.join(new_abs, "biodash.py"), "install", *keys, *unit_opts()],
                               cwd=new_abs, capture_output=True, text=True, timeout=180, env=env)
            for line in (p.stdout + p.stderr).strip().splitlines()[-6:]:
                prog.log(line)
            if p.returncode != 0:
                raise Refuse("Re-installing the services failed (see the lines above). The old version is still on disk: "
                             f"run ./biodash.py from {os.path.basename(HERE)} to go back.")
            prog.finish("done", f"Upgraded to V{target['version']}. The old version is still in {os.path.basename(HERE)}.")
        else:
            prog.finish("done", f"V{target['version']} is downloaded and set up in {new_abs}. "
                                "No services are installed, so stop these dashboards and launch them from that folder.")
    except Exception as e:
        prog.finish("error", str(e) if isinstance(e, Refuse) else f"{type(e).__name__}: {e}")
        raise


def main():
    ap = argparse.ArgumentParser(description="Check for and apply Bio-dash updates.")
    ap.add_argument("command", choices=["check", "patch", "upgrade", "status"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="check: use what was last downloaded")
    a = ap.parse_args()
    if a.command == "status":
        print(json.dumps(read_json(STATUS_FILE) or {"state": "idle"}, indent=None if a.json else 2))
        return 0
    if a.command == "check":
        info = check(fetch=not a.no_fetch)
        if a.json:
            print(json.dumps(info))
            return 0
        if info.get("error"):
            print("✗", info["error"]); return 1
        print(f"Bio-dash V{info['version']} ({info['folder']}, branch {info['branch']})")
        if info["patch"]:
            print(f"  Patch available: {info['patch']['count']} change(s)")
            for c in info["patch"]["commits"]:
                print(f"    {c['date']}  {c['subject']}")
        if info["upgrade"]:
            print(f"  New version available: V{info['upgrade']['version']}")
        if not info["patch"] and not info["upgrade"]:
            print("  Up to date.")
        for b in info["blockers"]:
            print("  ⚠", b)
        return 0
    if busy():
        print("Another update is already running."); return 1
    began = int(time.time())
    try:
        do_patch() if a.command == "patch" else do_upgrade()
    except Refuse as e:
        last = read_json(STATUS_FILE) or {}
        if last.get("pid") != os.getpid():            # refused before any progress was recorded
            write_json(STATUS_FILE, {"state": "error", "action": a.command, "started": began, "finished": int(time.time()),
                                     "stage": str(e), "log": [str(e)], "pid": os.getpid()})
        print("✗", e); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
