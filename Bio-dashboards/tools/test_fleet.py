#!/usr/bin/env python3
"""Checks for fleet.py with stand-in machines: a fake `ssh` that runs the command locally with HOME set
to the machine's own folder, and a fake `tailscale`. No network needed. Run: python3 tools/test_fleet.py"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.path.join(HERE, "fleet.py")
SAMPLE = os.path.join(os.path.dirname(HERE), "analysis_dashboard", "sample_data.py")
ok = 0


def check(name, got, want):
    global ok
    if got != want:
        raise AssertionError(f"{name}: got {got!r}, want {want!r}")
    ok += 1


T = tempfile.mkdtemp()
REMOTE, LOCAL, BIN = os.path.join(T, "remote"), os.path.join(T, "local"), os.path.join(T, "bin")
os.makedirs(BIN)
with open(os.path.join(BIN, "fakessh"), "w") as f:
    f.write(f"""#!/bin/bash
target="${{1#*@}}"; shift
[ -d "{REMOTE}/$target" ] || {{ echo "ssh: connect to host $target port 22: Connection timed out" >&2; exit 255; }}
export HOME="{REMOTE}/$target"; cd "$HOME"; exec bash -c "$*"
""")
with open(os.path.join(BIN, "tailscale"), "w") as f:
    status = {"Self": {"DNSName": "here.tail0.ts.net.", "HostName": "here", "TailscaleIPs": ["100.64.0.1"], "Online": True,
                       "Tags": ["tag:ssh"], "OS": "linux"},
              "Peer": {"a": {"DNSName": "alpha.tail0.ts.net.", "TailscaleIPs": ["100.64.0.2"], "Online": True, "Tags": ["tag:ssh"], "OS": "linux"},
                       "b": {"DNSName": "beta.tail0.ts.net.", "TailscaleIPs": ["100.64.0.3"], "Online": True, "Tags": ["tag:ssh"], "OS": "linux"},
                       "c": {"DNSName": "gamma.tail0.ts.net.", "TailscaleIPs": ["100.64.0.4"], "Online": False, "Tags": ["tag:ssh"], "OS": "linux"},
                       "d": {"DNSName": "phone.tail0.ts.net.", "TailscaleIPs": ["100.64.0.5"], "Online": True, "Tags": [], "OS": "android"}}}
    f.write("#!/bin/bash\ncat <<'EOF'\n" + json.dumps(status) + "\nEOF\n")
for n in ("fakessh", "tailscale"):
    os.chmod(os.path.join(BIN, n), 0o755)
for name, seed, days in (("alpha", 1, 2), ("beta", 7, 1)):
    subprocess.run([sys.executable, SAMPLE, "--days", str(days), "--seed", str(seed),
                    os.path.join(REMOTE, name, "Documents", "Bio-dash")], check=True, capture_output=True)
os.makedirs(os.path.join(LOCAL, "Documents"))
ENV = dict(os.environ, HOME=LOCAL, PATH=BIN + os.pathsep + os.environ["PATH"])
ENV.pop("BIODASH_DATA_DIR", None)
ENV.pop("XDG_CONFIG_HOME", None)          # never the real ~/.config


def fleet(*args):
    p = subprocess.run([sys.executable, FLEET, *args, "--ssh-cmd", os.path.join(BIN, "fakessh")],
                       capture_output=True, text=True, env=ENV, timeout=120)
    return p.returncode, p.stdout + p.stderr


def tree(root):
    out = {}
    for d, _, files in os.walk(root):
        for n in files:
            p = os.path.join(d, n)
            out[os.path.relpath(p, root)] = hashlib.sha1(open(p, "rb").read()).hexdigest()
    return out


# machines from Tailscale: online, tagged, not this computer
code, out = fleet("check")
check("check runs", code, 0)
check("alpha and beta, not the offline one, the phone or this computer",
      ("alpha:" in out, "beta:" in out, "gamma" in out, "phone" in out, "here:" in out), (True, True, False, False, False))
code, out = fleet("list")
check("list shows everything", all(n in out for n in ("alpha", "beta", "gamma", "phone", "(this computer)")), True)
code, out = fleet("check", "nowhere")
check("an unreachable machine is reported, not fatal", ("nowhere: ✗" in out, code), (True, 0))

# pull into one folder per machine; again = nothing; a grown recording is copied again
fleet("pull")
dest = os.path.join(LOCAL, "Documents", "Bio-dash-machines")
for m in ("alpha", "beta"):
    check(f"{m}: identical copy", tree(os.path.join(dest, m)), tree(os.path.join(REMOTE, m, "Documents", "Bio-dash")))
code, out = fleet("pull")
check("second pull: up to date", out.count("up to date"), 2)
rr = next(p for p in tree(os.path.join(REMOTE, "alpha", "Documents", "Bio-dash")) if p.endswith("_rr.csv"))
with open(os.path.join(REMOTE, "alpha", "Documents", "Bio-dash", rr), "a") as f:
    f.write("9999999999999,60,1000\n")
code, out = fleet("pull", "alpha")
check("grown recording copied", "1 grown" in out, True)
check("and now identical", tree(os.path.join(dest, "alpha")), tree(os.path.join(REMOTE, "alpha", "Documents", "Bio-dash")))
code, out = fleet("pull", "--only", "polar", "--dry-run", "beta")
check("--only / --dry-run", ("up to date" in out, code), (True, 0))

# a replayed session on a machine is never pulled (it has a <time>_replay_info.json beside it)
rp = os.path.join(REMOTE, "alpha", "Documents", "Bio-dash", "Polar H10", "Polar H10 REPLAY", "2026-10-21")
os.makedirs(rp)
for n in ("09-00-00_ecg.csv", "09-00-00_acc.csv"):
    open(os.path.join(rp, n), "w").write("Timestamp_Epoch_ms,ECG_mV,HR_BPM\n1,0.1,60\n")
open(os.path.join(rp, "09-00-00_replay_info.json"), "w").write("{}")
fleet("pull", "alpha")
check("replayed sessions aren't pulled", any("REPLAY" in k for k in tree(os.path.join(dest, "alpha"))), False)
shutil.rmtree(os.path.join(REMOTE, "alpha", "Documents", "Bio-dash", "Polar H10", "Polar H10 REPLAY"))

# --merge into this computer's own recordings: its own files are never touched
mine = os.path.join(LOCAL, "Documents", "Bio-dash")
subprocess.run([sys.executable, SAMPLE, "--days", "1", "--seed", "99", mine], check=True, capture_output=True)
before = tree(mine)
code, out = fleet("pull", "beta", "--merge")
after = tree(mine)
check("own recordings untouched", {k: after[k] for k in before}, before)
clashes = [k for k in after if " @beta" in k]
check("clashing names kept as '<device> @beta'", len(clashes) > 0, True)
check("everything from beta is here once", len(after) - len(before), len(tree(os.path.join(REMOTE, "beta", "Documents", "Bio-dash"))))
code, out = fleet("pull", "beta", "--merge")
check("merge again: up to date", "up to date" in out, True)
# one of ours is the start of one of theirs: the same recording, grown there -> replaced by theirs
theirs = os.path.join(REMOTE, "beta", "Documents", "Bio-dash", clashes[0].replace(" @beta", ""))
ours = os.path.join(mine, clashes[0].replace(" @beta", ""))
shutil.rmtree(os.path.join(mine, clashes[0].split(" @beta")[0] + " @beta"))
with open(theirs, "rb") as f:
    data = f.read()
with open(ours, "wb") as f:
    f.write(data[:len(data) // 2])
code, out = fleet("pull", "beta", "--merge")
check("prefix of theirs: completed in place", (open(ours, "rb").read() == data, "1 grown" in out), (True, True))

# push: with a real rclone when there is one (a local folder stands in for the cloud)
RCLONE = os.environ.get("RCLONE") or shutil.which("rclone")
if RCLONE:
    cloud = os.path.join(T, "cloud")
    code, out = fleet("push", "--rclone", RCLONE)
    check("push without --to asks where", ("Where to?" in out, code != 0), (True, True))
    os.makedirs(os.path.join(mine, "Exports"), exist_ok=True)
    open(os.path.join(mine, "Exports", "x.csv"), "w").write("not a recording\n")
    code, out = fleet("push", "--rclone", RCLONE, "--to", cloud)
    check("push ok", code, 0)
    here = [d for d in os.listdir(cloud) if d not in ("alpha", "beta")]
    check("this computer and both machines", sorted(os.listdir(cloud)), sorted(["alpha", "beta"] + here))
    check("this computer's recordings, not Exports/", tree(os.path.join(cloud, here[0])), {k: v for k, v in tree(mine).items()
                                                                                         if not k.startswith("Exports")})
    check("a machine's", tree(os.path.join(cloud, "alpha")), tree(os.path.join(dest, "alpha")))
    code, out = fleet("push", "--rclone", RCLONE)
    check("the destination is remembered, and a second push uploads nothing", out.count("up to date"), 3)
    with open(os.path.join(dest, "alpha", rr), "a") as f:
        f.write("9999999999999,61,990\n")
    code, out = fleet("push", "--rclone", RCLONE, "alpha")
    check("a grown recording goes up again (only that machine)", ("uploaded 1 file" in out, "beta" in out), (True, False))
    os.remove(os.path.join(dest, "alpha", rr))
    fleet("push", "--rclone", RCLONE)
    check("nothing online is deleted", os.path.exists(os.path.join(cloud, "alpha", rr)), True)
    code, out = fleet("push", "--rclone", RCLONE, "--only", "polar", "--dry-run", "beta")
    check("--only / --dry-run", code, 0)
    code, out = fleet("push", "--rclone", RCLONE, "--skip", "ecg,status", "--dry-run", "beta")
    check("--skip runs", code, 0)
else:
    print("(no rclone: push not tested)")

# schedule: writes the timers (a stand-in systemctl records what it's asked)
with open(os.path.join(BIN, "systemctl"), "w") as f:
    f.write(f"#!/bin/bash\necho \"$*\" >> {T}/systemctl.log\n")
os.chmod(os.path.join(BIN, "systemctl"), 0o755)
units = os.path.join(LOCAL, ".config", "systemd", "user")
code, out = fleet("schedule", "--to", "gdrive:Bio-dash")
check("schedule ok", code, 0)
check("two timers and their services", sorted(os.listdir(units)),
      ["biodash-push-full.service", "biodash-push-full.timer", "biodash-push.service", "biodash-push.timer"])
svc = open(os.path.join(units, "biodash-push.service")).read()
check("frequent one leaves out the waveforms", ("push --skip ecg,acc" in svc, os.path.abspath(FLEET) in svc), (True, True))
check("every 5 min / hourly", ("OnUnitActiveSec=5min" in open(os.path.join(units, "biodash-push.timer")).read(),
                               "OnUnitActiveSec=1h" in open(os.path.join(units, "biodash-push-full.timer")).read()), (True, True))
check("timers enabled", "enable --now biodash-push.timer" in open(os.path.join(T, "systemctl.log")).read(), True)
fleet("schedule", "--every", "10min", "--skip", "")
check("--skip \"\": one timer, everything", sorted(os.listdir(units)), ["biodash-push.service", "biodash-push.timer"])
code, out = fleet("schedule", "--every", "20s")
check("too often refused", code != 0, True)
fleet("schedule", "--off")
check("--off removes them", os.listdir(units), [])

shutil.rmtree(T)
print(f"{ok} checks passed")
