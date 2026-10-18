"""Bounded, serial HF/Triton search with explicit data/code identity and reports."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from context_sweep.recipe import ROOT, read, write, digest, fingerprint
from .planning import build
from .report import render

GB = 10**9


def completed(output, job):
    summary = read(output / 'runs' / job['name'] / 'train_summary.json', {})
    return summary.get('status') == 'complete' and summary.get('tokens') == job['target_tokens']


def disk_estimate(plan, output, retire):
    from .common import count_parameters
    counts = {}
    for job in plan['jobs']:
        key = job['architecture']
        if key not in counts:
            counts[key] = count_parameters(plan['configurations'][job['name']])['total']
    pending = [j for j in plan['jobs'] if not completed(output, j)]
    retained = sum((4 if retire else 20) * counts[j['architecture']] for j in pending)
    transient = 32 * max((counts[j['architecture']] for j in pending), default=0)
    return dict(parameter_counts=counts, future_retained_bytes=retained,
                transient_checkpoint_bytes=transient, overhead_bytes=3*GB,
                estimated_additional_peak_bytes=retained+transient+3*GB)


def child(command, log, output):
    from context_sweep.run import supervise
    stop, last_report, last_plot = [False], [0.], [0.]
    def handler(signum, frame): stop[0] = True
    def cancelled():
        if time.monotonic()-last_report[0] >= 15:
            plots = time.monotonic()-last_plot[0] >= 60
            render(output,plots=plots)
            if plots: last_plot[0] = time.monotonic()
            last_report[0] = time.monotonic()
        return stop[0]
    previous = {sig:signal.signal(sig,handler) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        result = supervise(command,log,cancelled,grace=120)
    finally:
        for sig, handler in previous.items(): signal.signal(sig,handler)
    if result['status'] == 'paused': raise KeyboardInterrupt
    return result.get('returncode',1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan','prepare','run','all','report','monitor'))
    p.add_argument('--preset', default=str(ROOT/'configs/hf_search_100m.json'))
    p.add_argument('--output', default='runs/hf_search')
    p.add_argument('--data', default='data/hf_search_100m')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Absolute per-run step; preserves full LR horizon for resume')
    p.add_argument('--disk-budget-gb', type=float, default=30.)
    p.add_argument('--disk-reserve-gb', type=float, default=2.)
    p.add_argument('--gpu-reserve-gib', type=float, default=1.)
    p.add_argument('--retire-completed', action='store_true', help='Delete verified completed optimizer state; final weights remain. Completed jobs then cannot resume optimizer training.')
    a = p.parse_args()
    if min(a.disk_budget_gb, a.disk_reserve_gb, a.gpu_reserve_gib) <= 0 or a.disk_reserve_gb >= a.disk_budget_gb:
        p.error('Choose positive memory/disk budgets, with disk reserve smaller than disk budget')
    output, data = Path(a.output).resolve(), Path(a.data).resolve()
    if a.action in ('report','monitor'):
        while True:
            render(output)
            if a.action == 'report': return
            time.sleep(15)
    spec = read(a.preset)
    if not spec: p.error('Cannot read preset')
    plan = build(spec)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.hf_search.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Another launcher is using this output directory; monitor its report instead')
    prior = read(output/'plan.json')
    if prior and prior != plan:
        p.error('Existing plan differs; use a fresh output directory')
    write(output/'plan.json', plan)
    for name, cfg in plan['configurations'].items():
        write(output/'configs'/(name+'.json'), cfg)
    estimate = disk_estimate(plan, output, a.retire_completed)
    write(output/'disk_estimate.json', dict(estimate, retire_completed=a.retire_completed))
    render(output)
    print(f"{plan['run_count']} runs, {plan['total_training_tokens']:,} total tokens; "
          f"estimated additional peak disk {estimate['estimated_additional_peak_bytes']/GB:.2f} GB. "
          f"Report: {output/'report.md'}", flush=True)
    if a.action == 'plan': return
    if estimate['estimated_additional_peak_bytes'] > min(int(a.disk_budget_gb*GB),shutil.disk_usage(output).free)-int(a.disk_reserve_gb*GB):
        p.error('Search exceeds free-space budget. Reduce max_architectures, reuse prepared data, or explicitly choose --retire-completed. Nothing was deleted.')
    if a.action == 'all':
        subprocess.run([sys.executable,'scripts/check_install.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-m','pytest','-q','tests/test_search_model.py','tests/test_hf_search.py','tests/test_hf_trainer.py'],
                       cwd=ROOT,check=True,env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'))
    if a.action in ('prepare','all') and not (data/'manifest.json').is_file():
        if spec['base'].get('smoke_only'):
            p.error('Smoke requires the offline fixture; use python -m hf_search.smoke')
        needed = max(j['target_tokens'] for j in plan['jobs'])+1
        subprocess.run([sys.executable,'-m','context_sweep.prepare','--output',str(data),
            '--train-tokens',str(needed),'--validation-tokens',str(max(2_000_000,spec['base']['validation_tokens']+1))],cwd=ROOT,check=True)
    if a.action == 'prepare': return
    if not (data/'manifest.json').is_file(): p.error('Prepare data first, or pass --data with an existing token stream')
    from .common import source_hashes, environment
    identity = dict(plan_sha256=fingerprint(plan), data_manifest_sha256=digest(data/'manifest.json'),
                    source_hashes=source_hashes(), environment=environment(), retire_completed=a.retire_completed,
                    device=a.device)
    prior_identity = read(output/'identity.json')
    if prior_identity and prior_identity != identity:
        p.error('Source/data/environment/retention policy changed. Resume requires identical experiment identity.')
    if prior_identity and not a.resume:
        p.error('Training already started; pass --resume or choose a fresh output')
    write(output/'identity.json',identity)
    state = read(output/'state.json',{})
    failures = 0
    for job in plan['jobs']:
        name = job['name']
        if completed(output,job):
            if a.retire_completed:
                from depth_sweep.storage import retire_completed_checkpoint
                retire_completed_checkpoint(output/'runs'/name)
            continue
        from depth_sweep.storage import check_job_space
        check_job_space(output,estimate['parameter_counts'][job['architecture']],a.disk_reserve_gb)
        if a.device.startswith('cuda'):
            request = output/'probes'/(name+'.request.json')
            result = output/'probes'/(name+'.result.json')
            write(request,dict(config=plan['configurations'][name],cell=job,device=a.device,reserve_gib=a.gpu_reserve_gib))
            if result.exists(): result.unlink()
            state[name] = dict(status='probing',updated=time.time())
            write(output/'state.json',state)
            code = child([sys.executable,'-m','hf_search.probe','--request',str(request),'--result',str(result)],output/'logs'/(name+'.probe.log'),output)
            probe = read(result,{})
            if code or probe.get('status') != 'fit':
                state[name] = dict(status=probe.get('status','failed'), updated=time.time(), phase='preflight', error=probe.get('error'),log=str(output/'logs'/(name+'.probe.log')))
                write(output/'state.json',state); render(output)
                failures += 1
                print(f"Preflight failed: {name}; inspect {output/'logs'/(name+'.probe.log')}", flush=True)
                continue
        elif not spec['base'].get('smoke_only'):
            p.error('Only the tiny smoke preset can run on CPU')
        state[name] = dict(status='running',updated=time.time())
        write(output/'state.json',state); render(output,plots=False)
        run = output/'runs'/name
        command = [sys.executable,'-m','hf_search.train','--config',str(output/'configs'/(name+'.json')),
                   '--data',str(data),'--output',str(run),'--variant',job['variant'],'--optimizer','muon',
                   '--seed',str(job['seed']),'--device',a.device]
        if a.resume and (run/'latest.json').is_file(): command.append('--resume')
        if a.stop_after is not None: command += ['--stop-after',str(a.stop_after)]
        print(f"Training {name}; log {output/'logs'/(name+'.log')}",flush=True)
        try:
            code = child(command,output/'logs'/(name+'.log'),output)
        except KeyboardInterrupt:
            state[name] = dict(status='paused',updated=time.time())
            write(output/'state.json',state); render(output)
            raise
        summary = read(run/'train_summary.json',{})
        status = summary.get('status','failed') if code == 0 else 'failed'
        state[name] = dict(status=status,returncode=code,updated=time.time())
        write(output/'state.json',state)
        if code: failures += 1
        elif status == 'complete' and a.retire_completed:
            from depth_sweep.storage import retire_completed_checkpoint
            retire_completed_checkpoint(run)
        render(output)
    print(f"Report: {output/'report.md'}; failures: {failures}")
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
