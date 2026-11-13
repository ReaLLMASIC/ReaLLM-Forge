#!/usr/bin/env python3
"""Compare two Bluetooth radios side by side: which one hears your devices better?

Both radios listen at the same time, to the same adverts, so the comparison is fair (a bottle
that only advertises after it moves, or a strap that drifts in and out, is the same for both).
For each device it reports how often each radio heard it and how strong the signal was.

    python3 tools/radio_compare.py                       list the radios, then stop
    python3 tools/radio_compare.py hci1 hci2             60 s, every device both can hear
    python3 tools/radio_compare.py 12:38 10:21 --labels stock antenna
                                                         radios by (the end of) their address, which
                                                         stays put while hci numbers shuffle
    python3 tools/radio_compare.py hci1 hci2 --labels stock antenna --seconds 120
    python3 tools/radio_compare.py hci1 hci2 --name h2o --name Polar --csv run1.csv

One at a time (same port, same spot, so only the dongle changes):

    python3 tools/radio_compare.py 12:38 --label stock    --seconds 60     round 1
    (swap dongles into the SAME port)
    python3 tools/radio_compare.py 10:21 --label antenna  --seconds 60     round 1
    ... repeat stock / antenna two or three times ...
    python3 tools/radio_compare.py --report                                 compare the rounds

Each solo run is added to radio_results.json (--results to choose another file). The report compares
only devices heard in every round (fixed beacons are ideal; phones change address and drop out),
and shows how much the same dongle varies from round to round, which is the noise to beat.

Tips for a fair test
  * Stop Bio-dash first (./biodash.py stop), or at least turn off the device you're measuring:
    a connected device stops advertising.
  * Plug both dongles into USB 2 ports (or short USB extension cables) side by side. USB 3 ports
    and cables give off noise right on 2.4 GHz and can cost a dongle 10 dB or more.
  * Run it twice, swapping the two dongles' ports in between: a difference that follows the dongle
    is the dongle; one that stays with the port is the port.
  * Repeat near and far (same room, another room, another floor): an antenna matters most when
    the signal is weak.

Reading the results: RSSI is in dBm, closer to 0 is stronger (−50 is very strong, −90 is the edge).
3 dB is twice the power; under about 3 dB of difference is within the noise. "Heard" counts how
many adverts each radio reported; a radio that hears more of them finds a device sooner.
Only bleak is needed, as for the dashboards. Linux / BlueZ.
"""
import argparse
import asyncio
import csv
import os
import statistics
import sys
import time

try:
    from bleak import BleakScanner
except ImportError:
    sys.exit("This needs bleak: run it with the Bio-dash .venv:  .venv/bin/python3 tools/radio_compare.py")


async def list_radios():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)                                     # V6: bt_debug.py in the release folder
        sys.path.insert(1, os.path.join(root, "analysis_dashboard"))  # V5: one copy per dashboard
        import bt_debug
        adapters = await bt_debug.list_adapters()
    except Exception as e:
        print(f"Couldn't list the radios ({e!r}). Try: bluetoothctl list")
        return
    if not adapters:
        print("No Bluetooth radios found.")
        return
    print("Radios on this computer:\n")
    for a in adapters:
        bits = [a.get("vendor"), a.get("usb_id") and f"USB {a['usb_id']}", a.get("placement")]
        print(f"  {a['name']:6s} {a.get('address') or '':20s} {' · '.join(b for b in bits if b)}")
    print("\nTwo dongles of the same model show the same USB ID. To tell them apart, unplug one and"
          "\nsee which hci disappears (or note each one's address with only it plugged in).")
    print("\nThen:  python3 tools/radio_compare.py <hciA or address end> <hciB or address end> --labels stock antenna")


async def resolve(names):
    """hciN as given; anything else is matched against the end of each radio's address."""
    if all(n.startswith("hci") for n in names):
        return names
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)                                     # V6: bt_debug.py in the release folder
        sys.path.insert(1, os.path.join(root, "analysis_dashboard"))  # V5: one copy per dashboard
        import bt_debug
        adapters = await bt_debug.list_adapters()
    except Exception as e:
        sys.exit(f"Couldn't list the radios to find {names} ({e!r}); use hci names instead.")
    out = []
    for n in names:
        if n.startswith("hci"):
            out.append(n)
            continue
        hits = [x for x in adapters if (x.get("address") or "").upper().endswith(n.upper())]
        if len(hits) != 1:
            sys.exit(f"{n!r} matches {len(hits)} radios; run without arguments to list them.")
        out.append(hits[0]["name"])
        print(f"{n} is {hits[0]['name']} ({hits[0]['address']})")
    return out


def summary(rssis):
    if not rssis:
        return None
    return {"n": len(rssis), "median": statistics.median(rssis), "best": max(rssis), "worst": min(rssis)}


async def main():
    ap = argparse.ArgumentParser(description="Compare two Bluetooth radios side by side.")
    ap.add_argument("radios", nargs="*", help="two adapters, e.g. hci1 hci2")
    ap.add_argument("--labels", nargs=2, metavar=("A", "B"), help="names for the two radios (default: their hci names)")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--name", action="append", default=[], help="only devices whose name contains this (repeatable)")
    ap.add_argument("--min-heard", type=int, default=3, help="leave out devices heard fewer times than this by both")
    ap.add_argument("--csv", help="also save every advert heard: time, radio, address, name, rssi")
    ap.add_argument("--label", help="one radio at a time: a name for this run's radio (e.g. stock)")
    ap.add_argument("--results", default="radio_results.json", help="where one-at-a-time runs are kept")
    ap.add_argument("--report", action="store_true", help="compare the one-at-a-time runs saved so far")
    a = ap.parse_args()

    if a.report:
        report(a)
        return
    if len(a.radios) == 1:
        await solo(a)
        return
    if len(a.radios) != 2:
        await list_radios()
        return
    radios = await resolve(a.radios)
    labels = a.labels or a.radios
    a.radios = radios
    heard = {}                      # address -> {"name", 0: [rssi...], 1: [rssi...]}
    log = []
    t0 = time.time()

    def make_cb(i):
        def cb(device, adv):
            name = device.name or adv.local_name or ""
            if a.name and not any(n.lower() in name.lower() for n in a.name):
                return
            e = heard.setdefault(device.address, {"name": name, 0: [], 1: []})
            if name and not e["name"]:
                e["name"] = name
            if adv.rssi is not None:
                e[i].append(adv.rssi)
            if a.csv:
                log.append([round(time.time() - t0, 3), labels[i], device.address, name, adv.rssi])
        return cb

    # each scanner gets its OWN settings dict (bleak writes the adapter into it, so a shared one
    # would put both scanners on the same radio); report every advert, not only changes
    scanners = [BleakScanner(make_cb(i), bluez={"adapter": r, "filters": {"DuplicateData": True}})
                for i, r in enumerate(a.radios)]
    print(f"Listening on {labels[0]} ({a.radios[0]}) and {labels[1]} ({a.radios[1]}) for {a.seconds:g} s"
          + (f", names containing {', '.join(a.name)}" if a.name else "") + " …")
    try:
        for s in scanners:
            await s.start()
    except Exception as e:
        for s in scanners:
            try:
                await s.stop()
            except Exception:
                pass
        sys.exit(f"Couldn't start scanning: {e!r}\nSomething else is scanning on one of these radios: stop Bio-dash "
                 "(./biodash.py stop) and close the desktop's Bluetooth settings, then try again.")
    try:
        end = time.time() + a.seconds
        while time.time() < end:
            await asyncio.sleep(1)
            left = int(end - time.time())
            counts = [sum(len(e[i]) for e in heard.values()) for i in (0, 1)]
            print(f"\r  {left:4d} s left · adverts heard: {labels[0]} {counts[0]} · {labels[1]} {counts[1]}   ", end="", flush=True)
    finally:
        for s in scanners:
            await s.stop()
    print("\n")

    rows = []
    for addr, e in heard.items():
        sa, sb = summary(e[0]), summary(e[1])
        if max(len(e[0]), len(e[1])) < a.min_heard:
            continue
        rows.append((addr, e["name"], sa, sb))
    if not rows:
        print("Nothing heard often enough. Wake the device (tip the bottle, wear the strap) and run it again.")
        return
    rows.sort(key=lambda r: -max(r[2]["n"] if r[2] else 0, r[3]["n"] if r[3] else 0))

    w = max(20, *(len(lbl) for lbl in labels))
    fmt = lambda s: f"{s['n']:5d} × {s['median']:6.1f} dBm" if s else "—"
    print(f"{'Device':30s} {labels[0]:>{w}s} {labels[1]:>{w}s}   difference")
    print(f"{'':30s} {'heard × median':>{w}s} {'heard × median':>{w}s}")
    wins = [0, 0]
    for addr, name, sa, sb in rows[:25]:
        diff = ""
        if sa and sb:
            d = sb["median"] - sa["median"]
            if abs(d) < 3:
                diff = "about the same"
            else:
                better = 1 if d > 0 else 0
                wins[better] += 1
                diff = f"{labels[better]} {abs(d):.0f} dB stronger"
        elif sa or sb:
            only = 0 if sa else 1
            wins[only] += 1
            diff = f"only {labels[only]} heard it"
        print(f"{(name or addr)[:30]:30s} {fmt(sa):>{w}s} {fmt(sb):>{w}s}   {diff}")

    both = [(sa, sb) for _, _, sa, sb in rows if sa and sb]
    print()
    if both:
        avg = statistics.mean(sb["median"] - sa["median"] for sa, sb in both)
        heard_a = sum(sa["n"] for sa, _ in both)
        heard_b = sum(sb["n"] for _, sb in both)
        print(f"Over {len(both)} device(s) both heard: {labels[1]} is {avg:+.1f} dB vs {labels[0]} on average "
              f"(positive = {labels[1]} stronger), and heard {heard_b} adverts to {heard_a}.")
    print(f"Clearly better: {labels[0]} on {wins[0]} device(s), {labels[1]} on {wins[1]}. "
          "Swap the ports and run again before trusting a small difference.")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["seconds", "radio", "address", "name", "rssi"])
            w_.writerows(log)
        print(f"Every advert saved to {a.csv} (it holds device addresses: keep it out of the repo).")


# ---------------------------------------------------------------- one radio at a time
def load_runs(path):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


async def solo(a):
    import json
    if not a.label:
        sys.exit("One radio at a time needs a --label (e.g. --label stock), so the rounds can be compared.")
    radio = (await resolve(a.radios))[0]
    heard = {}

    def cb(device, adv):
        name = device.name or adv.local_name or ""
        if a.name and not any(n.lower() in name.lower() for n in a.name):
            return
        e = heard.setdefault(device.address, {"name": name, "rssi": []})
        if name and not e["name"]:
            e["name"] = name
        if adv.rssi is not None:
            e["rssi"].append(adv.rssi)

    s = BleakScanner(cb, bluez={"adapter": radio, "filters": {"DuplicateData": True}})
    print(f"Listening on {a.label} ({radio}) for {a.seconds:g} s …")
    try:
        await s.start()
    except Exception as e:
        sys.exit(f"Couldn't start scanning: {e!r}\nStop Bio-dash (./biodash.py stop) and close the desktop's Bluetooth settings.")
    try:
        end = time.time() + a.seconds
        while time.time() < end:
            await asyncio.sleep(1)
            print(f"\r  {int(end - time.time()):4d} s left · adverts heard: {sum(len(e['rssi']) for e in heard.values())}   ",
                  end="", flush=True)
    finally:
        await s.stop()
    print()
    runs = load_runs(a.results)
    rnd = 1 + sum(1 for r in runs if r["label"] == a.label)
    runs.append({"label": a.label, "radio": radio, "round": rnd, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "seconds": a.seconds,
                 "devices": {addr: {"name": e["name"], "n": len(e["rssi"]), "median": statistics.median(e["rssi"])}
                             for addr, e in heard.items() if e["rssi"]}})
    with open(a.results, "w") as f:
        json.dump(runs, f, indent=1)
    total = sum(len(e["rssi"]) for e in heard.values())
    print(f"{a.label} round {rnd}: {len(heard)} devices, {total} adverts ({total / a.seconds * 60:.0f} a minute). "
          f"Saved to {a.results} (it holds device addresses: keep it out of the repo).")
    labels = sorted({r["label"] for r in runs})
    print("Rounds so far: " + ", ".join(f"{l} ×{sum(1 for r in runs if r['label'] == l)}" for l in labels)
          + ("   → python3 tools/radio_compare.py --report" if len(labels) > 1 else ""))


def report(a):
    runs = load_runs(a.results)
    labels = []
    for r in runs:
        if r["label"] not in labels:
            labels.append(r["label"])
    if len(labels) < 2:
        sys.exit(f"{a.results} has runs for {labels or 'nothing'} so far; it needs two labels to compare.")
    if len(labels) > 2:
        print(f"Comparing the first two labels: {labels[0]} and {labels[1]} (also saved: {', '.join(labels[2:])})")
    A, B = labels[:2]
    ra = [r for r in runs if r["label"] == A]
    rb = [r for r in runs if r["label"] == B]
    # devices heard well in every round of both
    common = None
    for r in ra + rb:
        ok = {addr for addr, d in r["devices"].items() if d["n"] >= a.min_heard}
        common = ok if common is None else common & ok
    common = common or set()
    print(f"{A}: {len(ra)} round(s) · {B}: {len(rb)} round(s) · {len(common)} device(s) heard in every round\n")
    if not common:
        print("No device was heard in every round. Run longer (--seconds 120), or keep a fixed beacon nearby.")
        return

    def med(rs, addr):
        return statistics.mean(r["devices"][addr]["median"] for r in rs)

    rows = []
    for addr in common:
        name = next((r["devices"][addr]["name"] for r in ra + rb if r["devices"][addr]["name"]), "") or addr
        rows.append((name, med(ra, addr), med(rb, addr)))
    rows.sort(key=lambda x: -max(x[1], x[2]))
    w = max(12, len(A) + 4, len(B) + 4)
    print(f"{'Device':30s} {A:>{w}s} {B:>{w}s}   difference")
    for name, va, vb in rows[:30]:
        d = vb - va
        diff = "about the same" if abs(d) < 3 else f"{B if d > 0 else A} {abs(d):.0f} dB stronger"
        print(f"{name[:30]:30s} {va:>{w - 4}.1f} dBm {vb:>{w - 4}.1f} dBm   {diff}")

    mean_d = statistics.mean(vb - va for _, va, vb in rows)
    faint = [vb - va for _, va, vb in rows if max(va, vb) < -90]
    faint_lead = None
    if len(faint) >= 3:
        fd = statistics.mean(faint)
        faint_lead = f"{B if fd > 0 else A} by {abs(fd):.1f} dB over {len(faint)} devices"
    rate = lambda rs: statistics.mean(sum(d["n"] for d in r["devices"].values()) / r["seconds"] * 60 for r in rs)
    # noise: how much the same radio moves between its own rounds, on the same devices
    spreads = []
    for rs in (ra, rb):
        if len(rs) > 1:
            for addr in common:
                vals = [r["devices"][addr]["median"] for r in rs]
                spreads.append(max(vals) - min(vals))
    lead = B if mean_d > 0 else A
    print(f"\nOn average {lead} is {abs(mean_d):.1f} dB stronger"
          + (f"; on the faint devices (below −90 dBm) {faint_lead}" if faint_lead else "") + ".")
    print(f"Adverts a minute, all devices: {A} {rate(ra):.0f} · {B} {rate(rb):.0f}.")
    if spreads:
        noise = statistics.median(spreads)
        verdict = ("a real difference" if abs(mean_d) > max(3, noise) else
                   "within the round-to-round noise: no real difference")
        print(f"The same dongle moves about {noise:.1f} dB between its own rounds, so {abs(mean_d):.1f} dB is {verdict}.")
    else:
        print("Run each dongle at least twice to see how much a dongle varies on its own (the noise to beat).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
