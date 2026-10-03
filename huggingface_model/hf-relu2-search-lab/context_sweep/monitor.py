"""Refresh the local sweep dashboard and export figures without using the GPU."""
import argparse
import fcntl
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import threading
import time
from .report import refresh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8774)
    p.add_argument("--interval", type=float, default=10)
    p.add_argument("--plot-interval", type=float, default=60)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    if min(a.interval, a.plot_interval) <= 0:
        p.error("Refresh intervals must be positive")
    root = Path(a.root).resolve()
    if not (root / "plan.json").is_file():
        p.error(f"No sweep plan found in {root}; use the sweep's output directory")
    live = root / "live"
    live.mkdir(parents=True, exist_ok=True)
    with (live / ".monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refresh(root, plots=True)
        if a.once:
            return
        class Handler(SimpleHTTPRequestHandler):
            def translate_path(self, path):
                resolved = Path(super().translate_path(path)).resolve()
                return str(resolved if resolved.is_relative_to(live) else live / "missing")
            def end_headers(self):
                self.send_header("Cache-Control", "no-store")
                super().end_headers()
            def log_message(self, *_):
                pass
        server = None
        try:
            server = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(live)))
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except OSError as error:
            print(f"HTTP dashboard unavailable: {error}. Continuing file reports in {live}", flush=True)
        stopped = threading.Event()
        def stop(*_):
            stopped.set()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        last_plot = time.monotonic()
        if server is not None:
            print(f"Dashboard: http://{a.host}:{a.port}", flush=True)
        try:
            while not stopped.wait(a.interval):
                plots = time.monotonic() - last_plot >= a.plot_interval
                refresh(root, plots)
                if plots:
                    last_plot = time.monotonic()
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    main()
