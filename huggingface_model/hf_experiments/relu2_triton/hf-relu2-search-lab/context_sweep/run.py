"""Resumable, sequential GPU sweep with live reports and fixed-token milestones."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .recipe import ROOT, read, write, build, fingerprint, progress, scheduled_cells


def stop_group(proc, sig=signal.SIGTERM):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def supervise(command, log_path, cancelled, grace=120, started=lambda pid: None):
    begin = time.time()
    if cancelled():
        return dict(status="not_started", seconds=0)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("\nCOMMAND " + json.dumps(command) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false"), start_new_session=True)
        requested = None
        try:
            started(proc.pid)
            while proc.poll() is None:
                if requested is None and cancelled():
                    requested = time.time()
                    stop_group(proc)
                if requested is not None and time.time() - requested >= grace:
                    stop_group(proc, signal.SIGKILL)
                    proc.wait()
                    break
                time.sleep(0.25)
        finally:
            if proc.poll() is None:
                stop_group(proc, signal.SIGKILL)
                proc.wait()
    return dict(status=("paused" if requested is not None else "ok" if proc.returncode == 0 else "failed"),
        returncode=proc.returncode, seconds=time.time() - begin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default=str(ROOT / "configs/sweep_50m.json"))
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/muon_output_norm_kda_50m")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--contexts", type=int, nargs="+")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--optimizers", nargs="+")
    p.add_argument("--output-norms", choices=["none", "capped"], nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--diagnostics-on-complete", action="store_true")
    p.add_argument("--probe-tokens", type=int, default=32768)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8774)
    p.add_argument("--dry-run", action="store_true", help="Print the recipe and selected jobs without loading models")
    p.add_argument("--check-only", action="store_true", help="Check environment, data and model counts, then exit")
    p.add_argument("--grace-seconds", type=float, default=120, help="Checkpoint grace after a manual stop; not a training deadline")
    a = p.parse_args()
    if a.grace_seconds <= 0:
        p.error("Positive shutdown grace required")
    plan = build(read(a.preset))
    selected = scheduled_cells(plan, a.contexts, a.variants, a.seeds, optimizers=a.optimizers, output_norms=a.output_norms)
    if a.diagnostics_on_complete and (a.probe_tokens < 1 or any(a.probe_tokens % n for n in plan['contexts']+[plan['spec']['common_validation_context']])):
        p.error('Every probe context must divide --probe-tokens exactly')
    print(json.dumps(dict(run_count=plan["run_count"], selected_run_count=len(selected),
        tokens_per_run=plan["tokens_per_run"], tokens_per_step=plan["tokens_per_step"],
        selected_training_tokens=plan["tokens_per_run"] * len(selected), contexts=plan["contexts"],
        model=plan["spec"]["model"], optimizers=plan["optimizers"], output_norms=plan["output_norms"], time_limit=None,
        milestones=plan["milestones"], jobs=selected), indent=2), flush=True)
    if a.dry_run:
        return
    from .preflight import check
    checked = check(plan, a.data, a.device, {cell["variant"] for cell in selected})
    if a.diagnostics_on_complete and read(Path(a.data)/'manifest.json')['splits']['validation']['tokens'] < a.probe_tokens+1:
        p.error('Prepared validation data must contain --probe-tokens + 1 tokens for checkpoint diagnostics')
    print(json.dumps(checked, indent=2), flush=True)
    if a.check_only:
        return
    out = Path(a.output).resolve()
    data = Path(a.data).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(recipe_sha256=plan["recipe_sha256"], data=str(data), device=a.device,
        data_manifest_sha256=checked["data_manifest_sha256"], environment=checked["environment"],
        source_hashes=checked["source_hashes"])
    with (out / ".sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(out / "sweep_state.json")
        if state:
            if not a.resume:
                raise ValueError("Sweep exists; use --resume")
            if state["identity"] != identity:
                raise ValueError("Recipe, data, model/trainer code or environment changed; keep this experiment intact and use a separate output")
            active = state.get("active") or {}
            pid = active.get("pid")
            if pid:
                cmd = Path(f"/proc/{pid}/cmdline")
                if cmd.exists() and str(out).encode() in cmd.read_bytes():
                    raise RuntimeError(f"A previous worker ({pid}) is still running; wait for it to save and exit")
            for job in state["jobs"]:
                if job["status"] == "running":
                    job["status"] = "interrupted"
        else:
            if a.resume:
                raise ValueError("No saved sweep exists; omit continuation flags for the first launch")
            if list(out.glob("ctx*/run.json")):
                raise ValueError("Output contains untracked training runs; choose a new output")
            state = dict(identity=identity, started_at=time.time(), time_limit=None, jobs=[], active=None)
        write(out / "plan.json", plan)
        write(out / "preflight.json", checked)
        for context, cfg in plan["configs"].items():
            write(out / "configs" / f"context_{context}.json", cfg)
        state.update(status="running", active=None)
        def persist():
            write(out / "sweep_state.json", state)
        persist()
        cancelled = False
        def stop(*_):
            nonlocal cancelled
            cancelled = True
            print("Stopping after the active optimizer step; waiting for its checkpoint.", flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        monitor, monitor_log = None, None
        if not a.no_monitor:
            monitor_log = (out / "monitor.log").open("a")
            monitor = subprocess.Popen([sys.executable, "-m", "context_sweep.monitor", "--root", str(out),
                "--host", a.host, "--port", str(a.port)], cwd=ROOT, stdout=monitor_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            print(f"Dashboard: http://{a.host}:{a.port}", flush=True)
        failed = set()
        try:
            for round_index, target in enumerate(plan["milestones"]):
                for cell in scheduled_cells(plan, a.contexts, a.variants, a.seeds, round_index, a.optimizers, a.output_norms):
                    if cancelled:
                        break
                    name = cell["name"]
                    run = out / name
                    cp, saved = progress(out, name)
                    summary = read(run / "train_summary.json", {})
                    final_ready = summary.get("status") == "complete" and summary.get("tokens") == cell["target_tokens"]
                    if (saved.get("step", 0) >= target and (target < plan["steps"] or final_ready)) or name in failed:
                        continue
                    if (run / "run.json").exists() and not (cp / "training_state.pt").is_file():
                        failed.add(name)
                        state["jobs"].append(dict(run=name, target_step=target, status="failed", seconds=0,
                            error="No complete resume checkpoint. Preserve this directory for diagnosis; a fresh output is required."))
                        persist()
                        continue
                    command = [sys.executable, "-m", "experiment.train", "--config", str(out / "configs" / f"context_{cell['config_key']}.json"),
                        "--data", str(data), "--output", str(run), "--variant", cell["variant"],
                        "--optimizer", cell["optimizer"], "--seed", str(cell["seed"]),
                        "--device", a.device, "--stop-after", str(target)]
                    if saved:
                        command += ["--resume"]
                    log = out / "logs" / f"{name}.log"
                    job = dict(run=name, target_step=target, status="running", started_at=time.time(), log=str(log))
                    state["jobs"].append(job)
                    state["active"] = dict(run=name, target_step=target)
                    persist()
                    print(f"{name}: step {saved.get('step', 0)} → {target}; log {log}", flush=True)
                    def started(pid):
                        state["active"]["pid"] = pid
                        persist()
                    result = supervise(command, log, lambda: cancelled, a.grace_seconds, started)
                    job.update(result)
                    _, saved_after = progress(out, name)
                    if result["status"] == "ok" and saved_after.get("step", 0) < target:
                        job.update(status="failed", error="Worker exited without reaching or saving the requested step")
                    if job["status"] == "failed":
                        failed.add(name)
                        with log.open("rb") as stream:
                            stream.seek(0, 2)
                            stream.seek(max(0, stream.tell() - 4000))
                            job["error"] = stream.read().decode(errors="replace")[-2000:]
                        print(f"{name}: failed; see {log}", flush=True)
                    if job["status"] == "paused":
                        cancelled = True
                    state["active"] = None
                    persist()
                    if a.diagnostics_on_complete and target == plan['steps'] and job['status'] == 'ok' and not cancelled:
                        probe_log=out/'logs'/f'{name}_diagnostics.log'
                        probe=supervise([sys.executable,'-m','diagnostics.sweep','--root',str(out),
                            '--data',str(data),'--run',name,'--device',a.device,'--tokens',str(a.probe_tokens)],
                            probe_log,lambda:cancelled,grace=10)
                        if probe['status']=='failed':
                            print(f'Diagnostic failed; training checkpoint preserved. See {probe_log}',flush=True)
                if cancelled:
                    break
            complete = all(progress(out, c["name"])[1].get("step", 0) >= plan["steps"] and
                           read(out / c["name"] / "train_summary.json", {}).get("status") == "complete"
                           for c in plan["cells"])
            state.update(status="complete" if complete else "paused" if cancelled else "partial",
                         active=None, finished_at=time.time())
            persist()
        finally:
            if monitor:
                stop_group(monitor)
                try:
                    monitor.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop_group(monitor, signal.SIGKILL)
                    monitor.wait()
                monitor_log.close()
            from .report import refresh
            refresh(out, plots=True)
        print(f"{state['status']}: {out / 'live/report.md'}", flush=True)
        if failed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
