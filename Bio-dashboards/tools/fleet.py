#!/usr/bin/env python3
"""Find Bio-dash recordings on your other machines (over Tailscale + SSH) and pull them here.

    ./tools/fleet.py list                    machines on your tailnet, and which are online
    ./tools/fleet.py check                   what each online machine has recorded
    ./tools/fleet.py pull                    copy what's new from every machine that has recordings
    ./tools/fleet.py pull roomba ubuntu-1    only these machines
    ./tools/fleet.py pull --since 2026-10-01 --only polar,hydro
    ./tools/fleet.py pull --merge            into this computer's own recordings folder (see below)
    ./tools/fleet.py push --to gdrive:Bio-dash   upload everything here to a cloud (rclone, see below)
    ./tools/fleet.py push --pull-first           pull from the machines, then upload (the last --to is remembered)
    ./tools/fleet.py schedule                    upload by itself: every 5 min, the ECG / motion waveforms hourly
    ./tools/fleet.py schedule --off              stop that

Machines come from `tailscale status`: the online ones with the tag `tag:ssh` (change with --tag,
or --tag "" for every online machine). You can also name machines (or user@host) yourself; then
Tailscale isn't needed at all, just SSH. This computer is skipped.

It connects with plain `ssh` (Tailscale SSH works with it, so does a normal SSH server and your
~/.ssh/config). Nothing is installed on the other machines; they need python3, which Ubuntu has.

Where recordings are looked for on each machine: <Documents>/Bio-dash (the same rule the dashboards
use, from ~/.config/user-dirs.dirs), or --remote-root.

Where they go:
    default    ~/Documents/Bio-dash-machines/<machine>/   one folder per machine, each laid out
               like a recordings folder, so the analysis dashboard or tools/interpolate.py can read
               one directly:  BIODASH_DATA_DIR=~/Documents/Bio-dash-machines/roomba ...
    --merge    straight into this computer's recordings folder, next to its own. A file that's
               already there is only replaced by a longer copy of itself (a recording that kept
               growing: checked byte for byte). A different recording with the same name is kept
               in a device folder of its own, "<device> @<machine>", so nothing is overwritten.
    --into DIR somewhere else (one sub-folder per machine).

Pushing online (push): uploads the raw recordings with rclone, so any cloud rclone knows works
(Google Drive, Dropbox, OneDrive, S3, Backblaze B2, Nextcloud/WebDAV, an SFTP box ...). Set one up
once with `rclone config` (it names it, e.g. "gdrive"), then:  ./tools/fleet.py push --to gdrive:Bio-dash
What goes up: this computer's recordings as <dest>/<this computer>/..., and each machine pulled into
Bio-dash-machines/ as <dest>/<machine>/..., laid out like a recordings folder. Only new and grown
files are uploaded (rclone compares size and time) and nothing online is ever deleted. These are
health recordings: an rclone "crypt" remote on top of the cloud one encrypts them before they leave
(rclone config -> new remote -> crypt), and the cloud only ever sees scrambled names and contents.

Scheduled uploads (schedule): systemd user timers run `push` by themselves: every --every (5min)
without the --skip streams (default ecg,acc), and everything every --full-every (1h). Why two: a
recording file grows for as long as its session lasts, and a grown file goes up whole again. The
ECG grows ~11 MB an hour, so a night's ECG re-sent every 5 minutes would be gigabytes an hour;
heart rate, SpO2, air and water are small and can go every few minutes. --skip "" sends everything
every time. Runs at low priority and never two at once; starts 3 min after boot.

Only what's missing or has grown is copied (rsync when both sides have it, else tar over ssh),
so running it again is quick. Exports/ and Reports/ (made from recordings) are left out.
Options: --dry-run (show what would be copied), --jobs N (machines at once, default 4),
--user NAME (ssh user), --timeout S (ssh connect timeout, default 8).
Standard library only.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY_RE = re.compile(r"^\d{4}-\d\d-\d\d$")
SKIP_TOP = {"Exports", "Reports"}

# runs on the other machine (python3 reads it from stdin): where are the recordings, and what's there
REMOTE_LIST = r'''
import hashlib, json, os, socket, sys
args = json.loads(sys.argv[1])
home = os.path.expanduser("~")
def documents():
    try:
        for line in open(os.path.join(home, ".config", "user-dirs.dirs")):
            line = line.strip()
            if line.startswith("XDG_DOCUMENTS_DIR="):
                p = line.split("=", 1)[1].strip().strip('"').replace("$HOME", home)
                if p and os.path.abspath(p) != os.path.abspath(home):
                    return p
    except OSError:
        pass
    return os.path.join(home, "Documents")
root = args.get("root") or os.environ.get("BIODASH_DATA_DIR") or os.path.join(documents(), "Bio-dash")
root = os.path.expanduser(root)
files = []
if os.path.isdir(root):
    for dash in sorted(os.listdir(root)):
        dp = os.path.join(root, dash)
        if dash in ("Exports", "Reports") or not os.path.isdir(dp):
            continue
        for dev in sorted(os.listdir(dp)):
            vp = os.path.join(dp, dev)
            if not os.path.isdir(vp):
                continue
            for day in sorted(os.listdir(vp)):
                yp = os.path.join(vp, day)
                if len(day) != 10 or not os.path.isdir(yp) or (args.get("since") and day < args["since"]):
                    continue
                for name in os.listdir(yp):
                    # a session written by Polar-dash's replay_session.py isn't a recording: never pull it
                    if name.endswith(".csv") and not os.path.exists(os.path.join(yp, name[:8] + "_replay_info.json")):
                        path = os.path.join(yp, name)
                        try:
                            st = os.stat(path)
                            with open(path, "rb") as f:
                                head = hashlib.sha1(f.read(256)).hexdigest()[:16]
                        except OSError:
                            continue
                        files.append([dash + "/" + dev + "/" + day + "/" + name, st.st_size, int(st.st_mtime), head])
print(json.dumps({"host": socket.gethostname(), "root": root, "exists": os.path.isdir(root), "files": files,
                  "rsync": any(os.access(os.path.join(p, "rsync"), os.X_OK) for p in os.environ.get("PATH", "").split(":"))}))
'''


def documents_dir():
    home = os.path.expanduser("~")
    try:
        with open(os.path.join(home, ".config", "user-dirs.dirs")) as f:
            for line in f:
                line = line.strip()
                if line.startswith("XDG_DOCUMENTS_DIR="):
                    p = line.split("=", 1)[1].strip().strip('"').replace("$HOME", home)
                    if p and os.path.abspath(p) != os.path.abspath(home):
                        return p
    except OSError:
        pass
    return os.path.join(home, "Documents")


def local_root():
    return os.environ.get("BIODASH_DATA_DIR") or os.path.join(documents_dir(), "Bio-dash")


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# ------------------------------------------------------------------------------ machines
def tailnet(tag):
    """[{name, ip, online, tags, os, self}] from `tailscale status --json`, or None without Tailscale."""
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        p = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=15)
        data = json.loads(p.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    out = []
    for peer, is_self in [(data.get("Self") or {}, True)] + [(p, False) for p in (data.get("Peer") or {}).values()]:
        if not peer:
            continue
        name = (peer.get("DNSName") or "").split(".")[0] or peer.get("HostName")
        out.append({"name": name, "ip": (peer.get("TailscaleIPs") or [None])[0], "online": bool(peer.get("Online")) or is_self,
                    "tags": peer.get("Tags") or [], "os": peer.get("OS", ""), "self": is_self})
    return sorted(out, key=lambda m: m["name"] or "")


def pick_machines(a):
    """[(label, ssh target)]"""
    me = {socket.gethostname().lower(), "localhost", "127.0.0.1"}
    if a.machines:
        out = []
        for m in a.machines:
            host = m.split("@")[-1]
            if host.lower() in me and not a.include_self:
                print(f"  {m}: this computer, skipped (--include-self to read it over ssh anyway)")
                continue
            out.append((re.sub(r"[^\w.-]+", "_", host), m if "@" in m or not a.user else f"{a.user}@{m}"))
        return out
    net = tailnet(a.tag)
    if net is None:
        sys.exit("Tailscale isn't installed or isn't running here. Name the machines instead:  ./tools/fleet.py check host1 host2")
    out = []
    for m in net:
        if m["self"] or not m["online"]:
            continue
        if a.tag and a.tag not in m["tags"]:
            continue
        target = m["name"]
        out.append((m["name"], f"{a.user}@{target}" if a.user else target))
    if not out:
        sys.exit(f"No online machines{' tagged ' + a.tag if a.tag else ''} on the tailnet. (`./tools/fleet.py list` shows them all.)")
    return out


# ------------------------------------------------------------------------------ ssh
def ssh_base(a):
    if a.ssh_cmd:                          # for testing: a command that stands in for ssh
        return shlex.split(a.ssh_cmd)
    return ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={a.timeout}", "-o", "StrictHostKeyChecking=accept-new"]


def remote_list(a, target, since=None):
    cmd = ssh_base(a) + [target, "python3 - " + shlex.quote(json.dumps({"root": a.remote_root, "since": since}))]
    try:
        p = subprocess.run(cmd, input=REMOTE_LIST, capture_output=True, text=True, timeout=a.timeout + 60)
    except subprocess.TimeoutExpired:
        return None, "no answer within a minute"
    except OSError as e:
        return None, f"couldn't run ssh: {e}"
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip().splitlines()
        msg = err[-1] if err else f"ssh exit {p.returncode}"
        if "python3" in msg and "not found" in msg:
            msg = "python3 isn't installed there"
        elif "Connection closed" in msg or "closed by remote host" in msg:
            user = target.split("@")[0] if "@" in target else os.environ.get("USER", "you")
            msg += (f"  (the machine hung up straight away: usually no user '{user}' there, or Tailscale SSH not "
                    f"allowing it. Try `ssh -v {target} true`, and --user NAME if your user name differs there)")
        elif "Permission denied" in msg or "Host key" in msg or "check" in msg.lower() and "tailscale" in msg.lower():
            msg += "  (try `ssh " + target + "` by hand once: it may want you to log in or approve it)"
        return None, msg
    try:
        return json.loads(p.stdout.strip().splitlines()[-1]), None
    except (ValueError, IndexError):
        return None, "unexpected reply: " + p.stdout.strip()[:120]


def summarize(info):
    files = info["files"]
    dashes = {}
    for rel, size, *_ in files:
        d = rel.split("/", 1)[0]
        n, s = dashes.get(d, (0, 0))
        dashes[d] = (n + 1, s + size)
    days = sorted({f[0].split("/")[2] for f in files})
    newest = max((f[2] for f in files), default=None)
    return dashes, days, newest


# ------------------------------------------------------------------------------ commands
def cmd_list(a):
    net = tailnet(a.tag)
    if net is None:
        sys.exit("Tailscale isn't installed or isn't running here.")
    print(f"{'machine':22s} {'online':7s} {'address':16s} {'os':8s} tags")
    for m in net:
        mark = " (this computer)" if m["self"] else ""
        print(f"{m['name'][:22]:22s} {'yes' if m['online'] else 'no':7s} {m['ip'] or '':16s} {m['os'][:8]:8s} "
              f"{' '.join(m['tags'])}{mark}")
    if a.tag:
        n = sum(1 for m in net if m["online"] and not m["self"] and a.tag in m["tags"])
        print(f"\ncheck / pull will use the {n} online machine(s) tagged {a.tag} (--tag \"\" for all).")


def cmd_check(a):
    machines = pick_machines(a)
    with cf.ThreadPoolExecutor(max(1, a.jobs)) as ex:
        results = list(ex.map(lambda m: (m, *remote_list(a, m[1], a.since)), machines))
    for (label, target), info, err in results:
        if err:
            print(f"\n{label}: ✗ {err}")
            continue
        if not info["exists"]:
            print(f"\n{label}: no recordings folder ({info['root']})")
            continue
        dashes, days, newest = summarize(info)
        total = sum(f[1] for f in info["files"])
        print(f"\n{label}: {len(info['files']):,} recordings, {human(total)} in {info['root']}")
        if days:
            print(f"  days {days[0]} to {days[-1]} ({len(days)}), newest file "
                  f"{dt.datetime.fromtimestamp(newest).strftime('%Y-%m-%d %H:%M')}")
        for d, (n, s) in sorted(dashes.items()):
            print(f"  {d:22s} {n:>6,} files {human(s):>10s}")
        if a.into or a.merge:
            continue
        dest = os.path.join(a.dest_base, label)
        new, grown, _ = plan(a, target, info, info["files"], dest, label, merge=False)
        if os.path.isdir(dest):
            print(f"  here: {'all pulled' if not new and not grown else f'{len(new):,} new and {len(grown):,} grown not pulled yet'} ({dest})")


REMOTE_PREFIX = r'''
import hashlib, json, os, sys
root, items = json.loads(sys.stdin.read())
out = {}
for rel, n in items:
    try:
        with open(os.path.join(root, rel), "rb") as f:
            out[rel] = hashlib.sha1(f.read(n)).hexdigest()
    except OSError:
        out[rel] = None
print(json.dumps(out))
'''


def remote_prefix_hashes(a, target, root, items):
    """{rel: sha1 of its first n bytes on the other machine} for [(rel, n)]."""
    if not items:
        return {}
    code = REMOTE_PREFIX.replace("sys.stdin.read()", "sys.argv[1]")
    cmd = ssh_base(a) + [target, "python3 -c " + shlex.quote(code) + " " + shlex.quote(json.dumps([root, items]))]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout + 120)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {}


def file_hash(path, n=None):
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read(n) if n is not None else f.read()).hexdigest()
    except OSError:
        return None


def other_device_folder(rel, label):
    """<dashboard>/<device>/<day>/<file> -> <dashboard>/<device> @<machine>/<day>/<file>: a recording
    that clashes by name with one of this computer's is kept beside it, as its own device folder (the
    dashboards read the time and stream from the file name, so that has to stay as it is)."""
    dash, dev, rest = rel.split("/", 2)
    return f"{dash}/{dev} @{label}/{rest}"


def plan(a, target, info, files, dest, label, merge):
    """What to copy: (new [(rel, local rel)], grown [...], renamed [...]).
    A file here with the same name counts as the same recording, grown over there, only when this copy
    is exactly the start of that one (checked on the other machine). Otherwise: in a machine's own
    folder the machine's copy wins; with --merge both are kept (see other_device_folder)."""
    new, grown, renamed = [], [], []
    later = []                                 # (rel, local rel, have): is our copy the start of theirs?
    for rel, size, _, head in files:
        loc = rel
        if merge and not os.path.exists(os.path.join(dest, rel)) and os.path.exists(os.path.join(dest, other_device_folder(rel, label))):
            loc = other_device_folder(rel, label)          # pulled before under the other name
        lp = os.path.join(dest, loc)
        if not os.path.exists(lp):
            new.append((rel, rel))
            continue
        have = os.path.getsize(lp)
        same_start = (file_hash(lp, 256) or "")[:16] == head
        if size == have and same_start:
            continue                                       # already here
        if not merge:
            grown.append((rel, loc))                       # a machine's own folder: its copy wins
        elif size > have:
            later.append((rel, loc, have))
        elif loc == rel:
            alt = other_device_folder(rel, label)          # a different recording with the same name
            ap_ = os.path.join(dest, alt)
            if not os.path.exists(ap_):
                renamed.append((rel, alt))
            elif os.path.getsize(ap_) < size:
                later.append((rel, alt, os.path.getsize(ap_)))     # pulled before: maybe grown since
    sizes = {f[0]: f[1] for f in files}
    for _ in range(2):                         # a second round for clashes whose own copy may have grown
        theirs = remote_prefix_hashes(a, target, info["root"], [(rel, have) for rel, _, have in later])
        again = []
        for rel, loc, have in later:
            if theirs.get(rel) and theirs[rel] == file_hash(os.path.join(dest, loc)):
                (grown if loc == rel else renamed).append((rel, loc))
            elif loc == rel:
                alt = other_device_folder(rel, label)
                ap_ = os.path.join(dest, alt)
                if not os.path.exists(ap_):
                    renamed.append((rel, alt))
                elif os.path.getsize(ap_) < sizes[rel]:
                    again.append((rel, alt, os.path.getsize(ap_)))
        later = again
    return new, grown, renamed


def copy_files(a, target, info, pairs, dest):
    """Copy [(remote rel, local rel)] from the machine into dest."""
    if not pairs:
        return 0, None
    if a.dry_run:
        return len(pairs), None
    same_names = all(r == l for r, l in pairs)
    if same_names and info.get("rsync") and shutil.which("rsync") and not a.ssh_cmd:
        with tempfile.NamedTemporaryFile("w", delete=False) as lst:
            lst.write("\n".join(r for r, _ in pairs) + "\n")
        try:
            os.makedirs(dest, exist_ok=True)
            ssh = " ".join(shlex.quote(x) for x in ssh_base(a))
            p = subprocess.run(["rsync", "-a", "--partial", "-e", ssh, f"--files-from={lst.name}",
                                f"{target}:{info['root'].rstrip('/')}/", dest + "/"], capture_output=True, text=True)
            if p.returncode == 0:
                return len(pairs), None
            return 0, (p.stderr or p.stdout).strip().splitlines()[-1:] or ["rsync failed"]
        finally:
            os.unlink(lst.name)
    # tar over ssh: works with nothing but python3 / tar on the other side
    names = "\n".join(r for r, _ in pairs)
    remote = f"cd {shlex.quote(info['root'])} && tar --warning=no-file-changed -cf - -T -"
    try:
        p = subprocess.Popen(ssh_base(a) + [target, remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.stdin.write(names.encode() + b"\n")
        p.stdin.close()
        where = dict(pairs)
        n = 0
        with tarfile.open(fileobj=p.stdout, mode="r|") as tar:
            for m in tar:
                if not m.isfile() or m.name not in where:
                    continue
                out = os.path.join(dest, where[m.name])
                if os.path.commonpath([os.path.abspath(out), os.path.abspath(dest)]) != os.path.abspath(dest):
                    continue                  # never write outside dest
                os.makedirs(os.path.dirname(out), exist_ok=True)
                src = tar.extractfile(m)
                tmp = out + ".part"
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f)
                os.replace(tmp, out)
                os.utime(out, (m.mtime, m.mtime))
                n += 1
        err = p.stderr.read().decode(errors="replace").strip()
        p.wait()
        # tar exits 1 when a file grew while it was read (a recording in progress): that's fine
        bad = p.returncode not in (0, 1) or (p.returncode == 1 and n == 0)
        return n, ((err.splitlines() or [f"tar exit {p.returncode}"])[-1] if bad else None)
    except (OSError, tarfile.TarError) as e:
        return 0, str(e)


def cmd_pull(a):
    machines = pick_machines(a)
    with cf.ThreadPoolExecutor(max(1, a.jobs)) as ex:
        listed = list(ex.map(lambda m: (m, *remote_list(a, m[1], a.since)), machines))
    only = [w.strip().lower() for w in (a.only or "").split(",") if w.strip()]
    total = 0
    for (label, target), info, err in listed:
        if err:
            print(f"{label}: ✗ {err}")
            continue
        files = [f for f in info["files"] if not only or any(w in f[0].split("/", 1)[0].lower() for w in only)]
        if not info["exists"] or not files:
            print(f"{label}: nothing to pull")
            continue
        dest = local_root() if a.merge else os.path.join(a.into or a.dest_base, label)
        new, grown, renamed = plan(a, target, info, files, dest, label, a.merge)
        todo = new + grown
        wanted = {r for r, _ in todo + renamed}
        size = sum(f[1] for f in files if f[0] in wanted)
        verb = "would copy" if a.dry_run else "copying"
        if not todo and not renamed:
            print(f"{label}: up to date ({len(files):,} recordings, {dest})")
            continue
        print(f"{label}: {len(files):,} recordings there; {verb} {len(new):,} new, {len(grown):,} grown"
              + (f", {len(renamed):,} under another name" if renamed else "") + f" ({human(size)}) to {dest}")
        if a.dry_run and a.verbose:
            for r, l in todo + renamed:
                print(f"    {r}" + (f" -> {l}" if l != r else ""))
        n1, e1 = copy_files(a, target, info, todo, dest)
        n2, e2 = copy_files(a, target, info, renamed, dest)
        for e in (e1, e2):
            if e:
                print(f"  ✗ {e if isinstance(e, str) else ' '.join(e)}")
        total += n1 + n2
    if not a.dry_run:
        print(f"\nDone: {total:,} file(s) copied.")
        if not a.merge and total:
            print(f"Look at one machine's recordings:  BIODASH_DATA_DIR={os.path.join(a.into or a.dest_base, '<machine>')} "
                  f"python3 tools/interpolate.py --list")


# ------------------------------------------------------------------------------ push (rclone)
CONFIG = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"),
                      "bio-dash", "fleet.json")


PUSH_LOCK = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
                         "bio-dash", "push.lock")


def load_config():
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(c):
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    with open(CONFIG + ".tmp", "w") as f:
        json.dump(c, f, indent=2)
    os.replace(CONFIG + ".tmp", CONFIG)


def push_sources(a):
    """[(label, folder)]: this computer's recordings, then each machine pulled here."""
    out = []
    here = re.sub(r"[^\w.-]+", "_", socket.gethostname()) or "this-computer"
    if os.path.isdir(local_root()):
        out.append((here, local_root()))
    base = a.into or a.dest_base
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, name)) and name != here:
                out.append((name, os.path.join(base, name)))
    if a.machines:
        want = {m.split("@")[-1] for m in a.machines}
        out = [(l, p) for l, p in out if l in want]
    return out


def replay_rules(src):
    """rclone rules leaving out sessions written by Polar-dash's replay_session.py (a <time>_replay_info.json
    beside them): a replay copies a real session, often a held-out one, with new timestamps."""
    rules = []
    for dirpath, dirnames, files in os.walk(src):
        dirnames[:] = sorted(d for d in dirnames if d not in ("Exports", "Reports"))
        for f in sorted(files):
            if f.endswith("_replay_info.json"):
                rel = os.path.relpath(os.path.join(dirpath, f[: -len("_replay_info.json")]), src)
                rules.append("- /" + re.sub(r"([\[\]{}*?\\])", r"\\\1", rel.replace(os.sep, "/")) + "_*")
    return rules


def rclone_rules(src, only, skip=()):
    """rclone filter rules: <dashboard>/<device>/<YYYY-MM-DD>/*.csv only, not Exports/ or Reports/,
    not replayed sessions, and not the streams in `skip` (e.g. ecg, acc: the big waveforms)."""
    names = [re.sub(r"[^\w-]", "", st) for st in skip]          # (no backslash inside an f-string: Python 3.10)
    rules = ["- /Exports/**", "- /Reports/**"] + replay_rules(src) + [f"- *_{n}.csv" for n in names if n]
    dashes = ["*"]
    if only:
        dashes = [d for d in sorted(os.listdir(src)) if any(w in d.lower() for w in only)
                  and os.path.isdir(os.path.join(src, d))] or ["(none)"]
    for d in dashes:
        esc = re.sub(r"([\[\]{}*?\\])", r"\\\1", d) if d != "*" else "*"
        rules.append(f"+ /{esc}/*/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*.csv")
    rules.append("- **")
    return rules


def cmd_push(a):
    rclone = a.rclone or shutil.which("rclone")
    if not rclone:
        sys.exit("rclone isn't installed. Ubuntu:  sudo apt install rclone   (or https://rclone.org/install/)\n"
                 "Then set up a cloud once with  rclone config  and run:  ./tools/fleet.py push --to <name>:Bio-dash")
    conf = load_config()
    dest = a.to or conf.get("push_to")
    if not dest:
        sys.exit("Where to? e.g.  ./tools/fleet.py push --to gdrive:Bio-dash   (the name is one from `rclone config`;\n"
                 "`rclone listremotes` lists them). It's remembered for next time.")
    if a.to and a.to != conf.get("push_to") and not a.dry_run:
        conf["push_to"] = a.to
        save_config(conf)
    if a.pull_first:
        print("== pulling from the machines first")
        cmd_pull(a)
        print("\n== pushing")
    # one push at a time (the 5-minute and the hourly one can meet): wait up to 10 min for the other
    import fcntl
    import time as _time
    os.makedirs(os.path.dirname(PUSH_LOCK), exist_ok=True)
    lock = open(PUSH_LOCK, "w")
    for i in range(600):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if i == 0:
                print("Another upload is running; waiting for it.")
            _time.sleep(1)
    else:
        sys.exit("Another upload has been running for 10 minutes; leaving it be.")
    sources = push_sources(a)
    if not sources:
        sys.exit(f"Nothing to push: no recordings in {local_root()} or {a.into or a.dest_base}.")
    only = [w.strip().lower() for w in (a.only or "").split(",") if w.strip()]
    skip = [w.strip().lower() for w in (a.skip or "").split(",") if w.strip()]
    failed = 0
    for label, src in sources:
        target = f"{dest.rstrip('/')}/{label}"
        with tempfile.NamedTemporaryFile("w", suffix=".rules", delete=False) as rf:
            rf.write("\n".join(rclone_rules(src, only, skip)) + "\n")
        cmd = [rclone, "copy", src, target, "--filter-from", rf.name, "-v", "--stats", "0", "--transfers", "4"]
        if a.since:
            cmd += ["--max-age", a.since]
        if a.dry_run:
            cmd.append("--dry-run")
        print(f"{label}: {src} -> {target}{' (dry run)' if a.dry_run else ''}")
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            copied = errors = 0
            for line in p.stdout:
                if "Copied (" in line or "Skipped copy as --dry-run" in line:
                    copied += 1
                    if a.verbose:
                        print("   ", line.rstrip().split(" : ", 1)[-1])
                elif " ERROR " in line or "Failed to" in line or "CRITICAL" in line:
                    errors += 1
                    print("  ✗", line.rstrip().split(" : ", 1)[-1][:200])
            p.wait()
        finally:
            os.unlink(rf.name)
        if p.returncode != 0:
            failed += 1
            print(f"  ✗ rclone stopped (exit {p.returncode}); nothing online was deleted. Re-run to carry on.")
        else:
            print(f"  {'would upload' if a.dry_run else 'uploaded'} {copied:,} file(s)" + (" (new or grown)" if copied else ": up to date"))
    if failed:
        sys.exit(1)


# ------------------------------------------------------------------------------ schedule
UNIT_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"),
                        "systemd", "user")


def span_text(text):
    """'5min' / '1h' / '30s' -> a systemd time span; exits on anything else."""
    m = re.fullmatch(r"\s*(\d+)\s*(s|sec|m|min|h|hr|d)\s*", text or "")
    if not m:
        sys.exit(f"'{text}': give a time like 5min, 1h or 30s")
    n, u = int(m.group(1)), m.group(2)
    secs = n * {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400}[u]
    if secs < 60:
        sys.exit("At least a minute apart, please.")
    return f"{secs // 60}min" if secs < 3600 or secs % 3600 else f"{secs // 3600}h"


def unit_files(every, full_every, skip):
    me, py = os.path.abspath(__file__), sys.executable
    def service(desc, args):
        return (f"[Unit]\nDescription={desc}\n\n"
                f"[Service]\nType=oneshot\nNice=10\nIOSchedulingClass=idle\n"
                f'ExecStart="{py}" "{me}" push {args}'.rstrip() + "\n")
    def timer(desc, span):
        return (f"[Unit]\nDescription={desc}\n\n[Timer]\nOnBootSec=3min\nOnUnitActiveSec={span}\n"
                f"RandomizedDelaySec=30\n\n[Install]\nWantedBy=timers.target\n")
    files = {}
    skip_args = f"--skip {','.join(skip)}" if skip else ""
    files["biodash-push.service"] = service(f"Upload Bio-dash recordings{' (not ' + ', '.join(skip) + ')' if skip else ''}",
                                            skip_args)
    files["biodash-push.timer"] = timer(f"Upload Bio-dash recordings every {every}", every)
    if skip and full_every:
        files["biodash-push-full.service"] = service("Upload all Bio-dash recordings, waveforms too", "")
        files["biodash-push-full.timer"] = timer(f"Upload all Bio-dash recordings every {full_every}", full_every)
    return files


def cmd_schedule(a):
    """Install / replace / remove the systemd user timers that run `push` by themselves."""
    names = ["biodash-push.timer", "biodash-push.service", "biodash-push-full.timer", "biodash-push-full.service"]
    systemctl = shutil.which("systemctl")
    run = (lambda *args: subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)) if systemctl else None
    if a.off:
        if run:
            run("disable", "--now", "biodash-push.timer", "biodash-push-full.timer")
        for n in names:
            with contextlib_suppress():
                os.remove(os.path.join(UNIT_DIR, n))
        if run:
            run("daemon-reload")
        print("Uploads are no longer scheduled.")
        return
    if not load_config().get("push_to") and not a.to:
        sys.exit("Push once by hand first, so the destination is known:  ./tools/fleet.py push --to gdrive:Bio-dash")
    every = span_text(a.every)
    full = span_text(a.full_every) if a.full_every else None
    skip = [w.strip().lower() for w in (a.skip if a.skip is not None else "ecg,acc").split(",") if w.strip()]
    files = unit_files(every, full, skip)
    if a.dry_run:
        for n, text in files.items():
            print(f"--- {os.path.join(UNIT_DIR, n)}\n{text}")
        return
    if a.to:
        c = load_config(); c["push_to"] = a.to; save_config(c)
    os.makedirs(UNIT_DIR, exist_ok=True)
    for n in names:                                  # replace the whole set (the full timer may go away)
        if n not in files:
            if run:
                run("disable", "--now", n) if n.endswith(".timer") else None
            with contextlib_suppress():
                os.remove(os.path.join(UNIT_DIR, n))
    for n, text in files.items():
        with open(os.path.join(UNIT_DIR, n), "w") as f:
            f.write(text)
    if not run:
        print(f"Wrote the timers to {UNIT_DIR}, but systemctl isn't here to start them.")
        return
    run("daemon-reload")
    bad = [n for n in files if n.endswith(".timer") and run("enable", "--now", n).returncode != 0]
    if bad:
        sys.exit(f"Couldn't start {', '.join(bad)}: try  systemctl --user enable --now {bad[0]}")
    dest = a.to or load_config().get("push_to")
    print(f"Uploading to {dest} every {every}" + (f" (not {', '.join(skip)})" if skip else "")
          + (f", and everything every {full}" if full and skip else "") + ".")
    print(f"Runs from {os.path.abspath(__file__)}: after moving to a new release, run schedule again from there.")
    linger = subprocess.run(["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
                            capture_output=True, text=True).stdout if shutil.which("loginctl") else ""
    if "Linger=yes" not in linger:
        print("To keep it going when you're not logged in:  sudo loginctl enable-linger $USER")
    print("Next runs:  systemctl --user list-timers 'biodash-push*'    what happened:  journalctl --user -u biodash-push")


class contextlib_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return exc[0] is not None and issubclass(exc[0], OSError)


def main():
    ap = argparse.ArgumentParser(description="Find Bio-dash recordings on your other machines and pull them here.",
                                 epilog="Full notes: the top of this file.")
    ap.add_argument("command", choices=["list", "check", "pull", "push", "schedule"])
    ap.add_argument("machines", nargs="*", help="machine names or user@host (default: online tailnet machines with --tag)")
    ap.add_argument("--tag", default="tag:ssh", help='Tailscale tag to pick machines by ("" = every online machine)')
    ap.add_argument("--user", help="ssh user (default: yours, or ~/.ssh/config)")
    ap.add_argument("--since", help="only days from YYYY-MM-DD on")
    ap.add_argument("--only", help="only these dashboards (words in the folder name: polar,hydro,atmos,viatom,omni)")
    ap.add_argument("--merge", action="store_true", help="into this computer's own recordings folder")
    ap.add_argument("--into", help="another folder (one sub-folder per machine)")
    ap.add_argument("--remote-root", help="the recordings folder on the other machines, if not <Documents>/Bio-dash")
    ap.add_argument("--to", help="push: where, as an rclone remote and folder, e.g. gdrive:Bio-dash (remembered)")
    ap.add_argument("--pull-first", action="store_true", help="push: pull from the machines first")
    ap.add_argument("--skip", help="push: leave out these streams, e.g. ecg,acc (schedule: default ecg,acc; \"\" = none)")
    ap.add_argument("--every", default="5min", help="schedule: how often to push (default 5min)")
    ap.add_argument("--full-every", default="1h", help="schedule: how often to push everything, waveforms too (default 1h)")
    ap.add_argument("--off", action="store_true", help="schedule: stop the scheduled uploads")
    ap.add_argument("--rclone", help=argparse.SUPPRESS)           # path to rclone (testing)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=8)
    ap.add_argument("--include-self", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--ssh-cmd", help=argparse.SUPPRESS)          # testing: stands in for ssh
    a = ap.parse_intermixed_args()       # names may come after the options (Python 3.10 needs this)
    if a.since and not DAY_RE.match(a.since):
        sys.exit("--since: YYYY-MM-DD")
    if a.merge and a.into:
        sys.exit("--merge or --into, not both")
    a.dest_base = os.path.join(documents_dir(), "Bio-dash-machines")
    {"list": cmd_list, "check": cmd_check, "pull": cmd_pull, "push": cmd_push, "schedule": cmd_schedule}[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
