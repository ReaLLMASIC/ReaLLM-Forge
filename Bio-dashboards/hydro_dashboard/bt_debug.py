"""Debug telemetry for the dashboard's debug panel.

- list_adapters():   every Bluetooth radio BlueZ knows about (hciN, address, name,
                     powered, vendor/chip, bus, internal/external) -- via D-Bus,
                     with sysfs details.

hciN numbers are NOT stable: they follow enumeration order, so a USB dongle that
comes up first becomes hci0 and pushes an internal card to hci1. The adapter's
Bluetooth address is the stable identity; treat hciN as a label.
- adapter_of(dev):   which radio a scanned BLEDevice was seen on (hci0, hci1, ...).
- RateMeter:         measured samples/sec over a sliding window, per stream.
"""
import os
import re
import shutil
import socket
import subprocess
import time
from collections import deque

# USB/PCI vendor IDs commonly found on Bluetooth radios
VENDORS = {
    "8087": "Intel", "0bda": "Realtek", "0a12": "Cambridge Silicon Radio (CSR)",
    "0a5c": "Broadcom", "13d3": "IMC Networks / AzureWave", "0cf3": "Qualcomm Atheros",
    "0489": "Foxconn / Hon Hai", "04ca": "Lite-On", "2357": "TP-Link", "0b05": "ASUS",
    "10ec": "Realtek", "14e4": "Broadcom", "168c": "Qualcomm Atheros", "17cb": "Qualcomm",
    "1d6b": "Linux Foundation", "2c7c": "Quectel", "0e8d": "MediaTek", "14c3": "MediaTek",
}

_ADAPTER_RE = re.compile(r"/org/bluez/(hci\d+)")


def adapter_of(device):
    """hciN for a BLEDevice from a BlueZ scan (details['path'] = /org/bluez/hciN/dev_...)."""
    details = getattr(device, "details", None)
    path = details.get("path") if isinstance(details, dict) else getattr(details, "path", None)
    m = _ADAPTER_RE.search(str(path or ""))
    return m.group(1) if m else None


SYSFS = "/sys"   # overridable for tests


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _placement(usbdev, vid):
    """internal / external for a USB radio.

    M.2/PCIe Wi-Fi+BT combo cards put the Bluetooth half on the slot's USB lines,
    so "USB" alone doesn't mean "dongle". The kernel exposes where the port is:
      <usbdev>/removable          fixed | removable | unknown   (from ACPI)
      <usbdev>/port/connect_type  hardwired | hotplug | not used
    Returns (placement, how_we_know).
    """
    rem = _read(f"{usbdev}/removable")
    if rem == "fixed":
        return "internal", "port is fixed"
    if rem == "removable":
        return "external", "port is removable"
    ct = _read(f"{usbdev}/port/connect_type")
    if ct == "hardwired":
        return "internal", "port is hardwired"
    if ct == "hotplug":
        return "external", "port is hot-pluggable"
    if (vid or "").lower() == "8087":
        return "internal", "Intel BT is almost always an onboard combo card (guess)"
    return "unknown", "firmware doesn't say"


def _sysfs_info(hci):
    """Bus type, placement and chip identity from /sys/class/bluetooth/hciN/device."""
    info = {}
    dev = f"{SYSFS}/class/bluetooth/{hci}/device"
    real = os.path.realpath(dev) if os.path.exists(dev) else ""
    if not real:
        return info
    info["bus"] = ("USB" if "/usb" in real else "PCIe" if "/pci" in real else
                   "SDIO" if "/mmc" in real else "UART" if ("serial" in real or "tty" in real) else "other")
    if info["bus"] == "USB":
        # the hci's device is a USB interface; the USB device (with idVendor) is its parent
        usbdev = os.path.dirname(real)
        vid, pid = _read(f"{usbdev}/idVendor"), _read(f"{usbdev}/idProduct")
        if vid:
            info["usb_id"] = f"{vid}:{pid}"
            info["vendor"] = _read(f"{usbdev}/manufacturer") or VENDORS.get(vid.lower())
            info["product"] = _read(f"{usbdev}/product")
        info["placement"], info["placement_basis"] = _placement(usbdev, vid)
    elif info["bus"] in ("PCIe", "SDIO", "UART"):
        info["placement"], info["placement_basis"] = "internal", f"{info['bus']} radios are onboard"
    return info


def _vendor_from_modalias(modalias):
    # e.g. "usb:v8087p0032d0000" -> Intel
    m = re.match(r"(usb|pci|bluetooth):v([0-9A-Fa-f]{4})p([0-9A-Fa-f]{4})", modalias or "")
    return (VENDORS.get(m.group(2).lower()), f"{m.group(2).lower()}:{m.group(3).lower()}") if m else (None, None)


async def list_adapters():
    """All radios BlueZ exposes. Returns [] if BlueZ/D-Bus isn't reachable."""
    adapters = []
    try:
        from dbus_fast import BusType, Message
        from dbus_fast.aio import MessageBus

        bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
        try:
            reply = await bus.call(Message(
                destination="org.bluez", path="/",
                interface="org.freedesktop.DBus.ObjectManager", member="GetManagedObjects"))
            objects = reply.body[0] if reply and reply.body else {}
        finally:
            bus.disconnect()

        for path, ifaces in sorted(objects.items()):
            props = ifaces.get("org.bluez.Adapter1")
            if not props:
                continue
            val = lambda k: props[k].value if k in props else None
            hci = path.rsplit("/", 1)[-1]
            vendor, usb_id = _vendor_from_modalias(val("Modalias"))
            a = {
                "name": hci,
                "address": val("Address"),
                "alias": val("Alias") or val("Name"),
                "powered": bool(val("Powered")),
                "discovering": bool(val("Discovering")),
                "vendor": vendor,
                "usb_id": usb_id,
            }
            for k, v in _sysfs_info(hci).items():
                if v and not a.get(k):
                    a[k] = v
            adapters.append(a)
    except Exception as e:
        # No system bus (container, macOS) -> fall back to whatever sysfs shows
        base = f"{SYSFS}/class/bluetooth"
        for hci in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            if re.fullmatch(r"hci\d+", hci):
                adapters.append({"name": hci, "error": f"D-Bus unavailable: {e.__class__.__name__}", **_sysfs_info(hci)})
    return adapters


class RateMeter:
    """Counts events and reports the rate over the last `window` seconds."""

    def __init__(self, window=5.0):
        self.window = window
        self.count = 0
        self.snaps = deque()

    def add(self, n=1):
        self.count += n

    def reset(self):
        self.count = 0
        self.snaps.clear()

    def rate(self):
        now = time.monotonic()
        self.snaps.append((now, self.count))
        while len(self.snaps) > 2 and now - self.snaps[0][0] > self.window:
            self.snaps.popleft()
        t0, c0 = self.snaps[0]
        return (self.count - c0) / (now - t0) if now - t0 > 0.5 else 0.0


# ---------------------------------------------------------------------------
# Radio pinning: which Bluetooth radio a dashboard scans and connects on.
# Stored by adapter ADDRESS (stable) and resolved to hciN every time, because
# hciN follows enumeration order and can swap between boots.
# ---------------------------------------------------------------------------
import json as _json


def bluez_args(hci):
    """kwargs for BleakScanner / BleakClient / find_device_by_address."""
    return {"bluez": {"adapter": hci}} if hci else {}


class RadioPin:
    """radio_config.json next to a dashboard: {"radio": "auto" | "<adapter address>"}.

    "auto" with two or more radios picks by placement, so dashboards running side
    by side split sensibly no matter how the radios were numbered:
      prefer="external"  the Polar H10, which can monopolise its radio while streaming
      prefer="internal"  everything else, leaving any dongle to the Polar
    With one radio (or no placement info) auto uses the system default radio.
    """

    def __init__(self, path, prefer=None):
        self.path = path
        self.prefer = prefer

    def config(self):
        try:
            with open(self.path) as f:
                v = (_json.load(f).get("radio") or "auto").strip()
            return v or "auto"
        except Exception:
            return "auto"

    def save(self, address):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            _json.dump({"radio": (address or "auto").strip() or "auto"}, f, indent=2)
        os.replace(tmp, self.path)

    def resolve(self, adapters):
        """-> (hciN, or None for the system default radio; human-readable reason)"""
        usable = [a for a in (adapters or []) if a.get("powered", True) and not a.get("error")]
        want = self.config().upper()
        reason = None
        if want != "AUTO":
            for a in usable:
                if (a.get("address") or "").upper() == want:
                    return a["name"], f"pinned to {want}"
            reason = f"pinned radio {want} not present, using auto"
        if self.prefer and len(usable) > 1:
            match = [a["name"] for a in usable if a.get("placement") == self.prefer]
            if match:
                return match[0], reason or f"auto: {self.prefer} radio"
        return None, reason or "auto: system default radio"


class LastDevice:
    """Remembers the last device a dashboard streamed from, for auto-reconnect.

    <dashboard>/last_device.json:
      {"address": "...", "name": "...", "auto": true, "saved_at": 1790000000.0, ...extra}
    extra: e.g. {"transport": "tcp", "host": "192.168.0.50", "port": 8080} for Atmos Wi-Fi.
    "auto" defaults to BIODASH_AUTORECONNECT (set by the service installer; on unless "0").
    """

    def __init__(self, path):
        self.path = path

    @staticmethod
    def _default_auto():
        return os.environ.get("BIODASH_AUTORECONNECT", "1") != "0"

    def load(self):
        try:
            with open(self.path) as f:
                d = _json.load(f)
        except Exception:
            return None
        if not d.get("address"):
            return None
        d.setdefault("auto", self._default_auto())
        return d

    def _write(self, d):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(d, f, indent=2)
        os.replace(tmp, self.path)

    def remember(self, address, name=None, **extra):
        prev = self.load() or {}
        d = {"address": address, "name": name or prev.get("name"), "saved_at": time.time(),
             "auto": prev.get("auto", self._default_auto()) if prev.get("address") == address else self._default_auto()}
        d.update(extra)
        self._write(d)

    def set_auto(self, on):
        d = self.load()
        if d:
            d["auto"] = bool(on)
            self._write(d)

    def forget(self):
        try:
            os.remove(self.path)
        except OSError:
            pass

    def target(self):
        """The remembered device if auto-reconnect is on, else None."""
        d = self.load()
        return d if d and d.get("auto") else None


# ---------------------------------------------------------------------------
# Where recordings go:  <Documents>/Bio-dash/<Dashboard>/<Device>/<YYYY-MM-DD>/<HH-MM-SS>_<stream>.csv
# ---------------------------------------------------------------------------
def documents_dir():
    """The desktop's Documents folder (XDG user dirs), else ~/Documents."""
    home = os.path.expanduser("~")
    try:
        with open(os.path.join(home, ".config", "user-dirs.dirs")) as f:
            for line in f:
                line = line.strip()
                if line.startswith("XDG_DOCUMENTS_DIR="):
                    path = line.split("=", 1)[1].strip().strip('"').replace("$HOME", home)
                    if path and os.path.abspath(path) != os.path.abspath(home):
                        return path
    except OSError:
        pass
    return os.path.join(home, "Documents")


def data_root():
    """<Documents>/Bio-dash, or BIODASH_DATA_DIR if set."""
    return os.environ.get("BIODASH_DATA_DIR") or os.path.join(documents_dir(), "Bio-dash")


def safe_name(s, fallback="unknown device"):
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", str(s or "")).strip(" .-")
    return (s or fallback)[:80]


def device_folder(name, address=None):
    """'Polar H10 D793BB2E' stays as is (the name carries an ID); names shared by every
    unit of a model ('Atmos-Sphere-S4') get the address added so units stay apart."""
    name = (name or "").strip()
    addr = (address or "").replace("tcp://", "")
    if name and re.search(r"[0-9A-Fa-f]{4,}", name):
        return safe_name(name)
    if name and addr:
        return safe_name(f"{name} ({addr})")
    return safe_name(name or addr)


def dashboard_dir(dashboard):
    return os.path.join(data_root(), safe_name(dashboard))


class SessionFiles:
    """CSV files for one device session, opened when the device connects.

        s = SessionFiles("Polar H10")
        s.start(device_name, address, {"ecg": ["Timestamp_Epoch_ms", "ECG_mV"], ...})
        s.writerows("ecg", rows); s.flush(); s.close()
    """

    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.dir = None
        self.stamp = None
        self.files = {}
        self.writers = {}
        self.paths = {}

    @property
    def active(self):
        return bool(self.files)

    def start(self, device_name, address, streams, when=None):
        import csv as _csv
        import datetime as _dt
        self.close()
        now = when or _dt.datetime.now()
        self.dir = os.path.join(dashboard_dir(self.dashboard), device_folder(device_name, address), now.strftime("%Y-%m-%d"))
        os.makedirs(self.dir, exist_ok=True)
        self.stamp = now.strftime("%H-%M-%S")
        for stream, header in streams.items():
            path = os.path.join(self.dir, f"{self.stamp}_{stream}.csv")
            f = open(path, "a", newline="")
            w = _csv.writer(f)
            if header:
                w.writerow(header)
            f.flush()
            self.files[stream], self.writers[stream], self.paths[stream] = f, w, path
        return self.dir

    def writerows(self, stream, rows):
        w = self.writers.get(stream)
        if w is not None and rows:
            w.writerows(rows)

    def flush(self, stream=None):
        for k, f in self.files.items():
            if stream is None or k == stream:
                f.flush()

    def close(self):
        for f in self.files.values():
            try:
                f.close()
            except Exception:
                pass
        self.files, self.writers, self.paths = {}, {}, {}


def latest_session_files(dashboard, stream, limit=None, same_device=False):
    """Newest-first paths of '<time>_<stream>.csv' under a dashboard's folder.
    same_device=True keeps only files from the device of the newest one."""
    import glob as _glob
    pattern = os.path.join(dashboard_dir(dashboard), "*", "*", f"*_{stream}.csv")
    files = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if same_device and files:
        dev = os.path.dirname(os.path.dirname(files[0]))
        files = [f for f in files if os.path.dirname(os.path.dirname(f)) == dev]
    return files[:limit] if limit else files


# ---------------------------------------------------------------------------
# System power: shutdown / reboot from the debug panel
# ---------------------------------------------------------------------------
# Allowed only from this computer, unless BIODASH_ALLOW_POWER=1 (e.g. a phone controlling a
# backpack / headless setup). Requests must carry POWER_HEADER, which other websites can't add
# without CORS permission -- so a page open in your browser can't trigger it.
# BIODASH_POWER_DRY_RUN=1 does everything except the actual shutdown (for testing).
POWER_HEADER = "X-Biodash-Power"
POWER_ACTIONS = {"shutdown": "poweroff", "reboot": "reboot"}


def _power_allowed(client_host):
    if os.environ.get("BIODASH_ALLOW_POWER") == "1":
        return True, "allowed from any device (BIODASH_ALLOW_POWER=1)"
    host = (client_host or "").strip()
    if host in ("::1", "localhost") or host.startswith("127.") or host.startswith("::ffff:127."):
        return True, "allowed from this computer"
    return False, ("only from this computer -- to allow other devices (e.g. your phone), "
                   "start the dashboards with BIODASH_ALLOW_POWER=1")


def _power_method():
    """How a shutdown can run without a password prompt: "systemctl" (logind allows this user),
    "sudo" (a NOPASSWD sudoers rule for systemctl poweroff/reboot), or None."""
    try:
        out = subprocess.run(["busctl", "call", "org.freedesktop.login1", "/org/freedesktop/login1",
                              "org.freedesktop.login1.Manager", "CanPowerOff"],
                             capture_output=True, text=True, timeout=3).stdout
        if '"yes"' in out:
            return "systemctl"
    except Exception:
        pass
    try:
        systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
        if subprocess.run(["sudo", "-n", "-l", systemctl, "poweroff"], capture_output=True, timeout=3).returncode == 0:
            return "sudo"
    except Exception:
        pass
    return None


POWER_FIX = ('echo "$USER ALL=(root) NOPASSWD: $(command -v systemctl) poweroff, $(command -v systemctl) reboot" '
             '| sudo tee /etc/sudoers.d/biodash-power && sudo chmod 440 /etc/sudoers.d/biodash-power')


def power_info(client_host):
    allowed, reason = _power_allowed(client_host)
    method = _power_method() if allowed else None
    dry = os.environ.get("BIODASH_POWER_DRY_RUN") == "1"
    return {"allowed": allowed and (method is not None or dry), "reason": reason, "method": method,
            "dry_run": dry, "host": socket.gethostname(), "fix": None if (method or dry or not allowed) else POWER_FIX}


def system_power(action, client_host, header_value):
    """(http_status, json) for a shutdown / reboot request."""
    if action not in POWER_ACTIONS:
        return 400, {"error": "action must be shutdown or reboot"}
    if header_value != "1":
        return 403, {"error": "missing confirmation header"}
    allowed, reason = _power_allowed(client_host)
    if not allowed:
        return 403, {"error": reason}
    if os.environ.get("BIODASH_POWER_DRY_RUN") == "1":
        return 200, {"ok": True, "dry_run": True, "action": action}
    method = _power_method()
    if method is None:
        return 403, {"error": "this user isn't allowed to power off without a password", "fix": POWER_FIX}
    prefix = "sudo -n " if method == "sudo" else ""
    # run detached and slightly delayed, so this HTTP reply still reaches the browser
    subprocess.Popen(["sh", "-c", f"sleep 2; {prefix}systemctl {POWER_ACTIONS[action]}"],
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 200, {"ok": True, "action": action, "method": method}
