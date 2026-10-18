"""Run an explicit 512-wide, 2048-context depth comparison with resource gates."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from context_sweep.recipe import ROOT, read, write, digest
from context_sweep.run import supervise, stop_group
from depth_sweep.launch import bind_data
from depth_sweep.storage import available_budget, check_job_space, retire_completed_checkpoint
from .recipe import specifications, cells, storage_estimate, summary


def probe_outcome(process, measured, log):
    if process['status'] != 'ok' or measured.get('status') not in ('fit', 'oom', 'budget_exceeded'):
        raise RuntimeError(f'Capacity worker failed; this is not an OOM estimate. Inspect {log}')
    if measured['status'] != 'fit':
        raise RuntimeError(f'An explicitly requested depth did not fit the VRAM reserve. '
                           f'Inspect {log}; select smaller --layers or --microbatch-tokens 2048 '
                           'with a fresh --output. No training started.')


def identity(specs, args):
    import torch
    from experiment.common import configure_device, source_hashes, environment
    configure_device(args.device, 'bfloat16')
    props = torch.cuda.get_device_properties(args.device)
    sources = source_hashes()
    for folder in ('quick_depth', 'context_sweep', 'depth_sweep'):
        sources.update({str(p.relative_to(ROOT)): digest(p) for p in (ROOT/folder).glob('*.py')})
    return dict(specs={str(k): v for k, v in specs.items()}, sources=sources, environment=environment(),
        gpu_uuid=str(getattr(props, 'uuid', props.name)), gpu_bytes=props.total_memory,
        device=args.device, allocator=torch.cuda.get_allocator_backend(),
        allocator_env={k: os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')},
        reserve_gib=args.reserve_gib, disk_budget_gb=args.disk_budget_gb,
        disk_reserve_gb=args.disk_reserve_gb, storage_policy='retire_only_completed_optimizer_state')


def probe(specs, args, out, signature, cancelled):
    import torch
    plan = read(out/'quick_plan.json')
    if plan and plan['identity'] != signature:
        raise ValueError('Settings, model/trainer source, GPU or environment changed; use a fresh output directory')
    free, _ = torch.cuda.mem_get_info()
    if plan and plan.get('capacity_passed'):
        if free + (128 << 20) < plan['initial_free_bytes']:
            raise RuntimeError('Less VRAM is free than at preflight; stop other GPU jobs before resuming')
        return plan
    storage = storage_estimate(specs)
    budget = available_budget(out, args.disk_budget_gb, args.disk_reserve_gb)
    if storage['estimated_peak_bytes'] > budget:
        raise RuntimeError(f'Sweep disk estimate {storage["estimated_peak_bytes"]/1e9:.2f} GB exceeds '
                           f'{budget/1e9:.2f} GB available after reserve; reduce --layers or free space.')
    plan = dict(identity=signature, specs=signature['specs'], **summary(specs),
                storage=dict(storage, budget_bytes=budget), capacity_passed=False,
                initial_free_bytes=free, observations=[])
    write(out/'quick_plan.json', plan)
    for depth, spec in specs.items():
        write(out/'presets'/f'L{depth:04d}.json', spec)
        for cell, cfg in cells(spec):
            if cancelled():
                raise KeyboardInterrupt
            tag = f'L{depth:04d}_{cell["name"]}'
            folder = out/'capacity'
            request, result_path, log = (folder/f'{tag}.{suffix}' for suffix in ('request.json', 'result.json', 'log'))
            # Do not accept a stale result if a new worker fails before writing.
            if result_path.exists():
                result_path.unlink()
            write(request, dict(config=cfg, cell=cell, device=args.device, reserve_gib=args.reserve_gib))
            write(out/'quick_status.json', dict(phase='capacity_probe', layers=depth, cell=cell['name'], log=str(log)))
            print(f'VRAM preflight: {tag} (two optimizer steps + validation)', flush=True)
            result = supervise([sys.executable, '-m', 'depth_sweep.worker', '--request', str(request),
                                '--result', str(result_path)], log, cancelled, grace=15)
            if cancelled():
                raise KeyboardInterrupt
            measured = read(result_path, {})
            plan['observations'].append(dict(layers=depth, cell=cell['name'], **measured))
            write(out/'quick_plan.json', plan)
            probe_outcome(result, measured, log)
    plan['capacity_passed'] = True
    write(out/'quick_plan.json', plan)
    write(out/'quick_status.json', dict(phase='ready'))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'probe', 'check', 'all', 'train', 'report', 'monitor'])
    p.add_argument('--layers', type=int, nargs='+', default=[16, 24, 32])
    p.add_argument('--tokens', type=int, default=100_000_000)
    p.add_argument('--microbatch-tokens', type=int, default=4096)
    p.add_argument('--data', default='data/fineweb_100m')
    p.add_argument('--output', default='runs/quick_d512_ctx2048')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--reserve-gib', type=float, default=1.)
    p.add_argument('--disk-budget-gb', type=float, default=30.)
    p.add_argument('--disk-reserve-gb', type=float, default=2.)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not all(math.isfinite(x) for x in (a.reserve_gib, a.disk_budget_gb, a.disk_reserve_gb)) or \
       a.reserve_gib < 0 or not 0 <= a.disk_reserve_gb < a.disk_budget_gb:
        p.error('Use finite nonnegative reserves and a larger positive disk budget')
    os.chdir(ROOT)
    out = Path(a.output).resolve()
    if a.command in ('report', 'monitor'):
        from .report import refresh, watch
        (watch if a.command == 'monitor' else refresh)(out)
        return
    specs = specifications(a.layers, a.tokens, a.microbatch_tokens)
    print(json.dumps(summary(specs), indent=2), flush=True)
    if a.command == 'plan':
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.quick.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'depth_plan.json').exists() or (out/'plan.json').exists():
            raise ValueError('This output belongs to a different lab sweep; use a fresh directory')
        from context_sweep.launch import require_cuda, tests
        require_cuda()
        if a.command == 'all':
            subprocess.run([sys.executable, 'scripts/check_install.py'], check=True)
            tests()
        signature = identity(specs, a)
        cancelled = False
        def stop(*_):
            nonlocal cancelled
            cancelled = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        saved = read(out/'quick_plan.json', {})
        if a.command == 'train' and not saved.get('capacity_passed'):
            raise ValueError('First run probe (or use all) to measure every requested shape')
        from .report import refresh
        try:
            plan = probe(specs, a, out, signature, lambda: cancelled)
        except KeyboardInterrupt:
            write(out/'quick_status.json', dict(phase='paused', error='Capacity probe interrupted; no training started.'))
            refresh(out)
            raise
        except Exception as error:
            write(out/'quick_status.json', dict(phase='failed', error=str(error)))
            refresh(out)
            raise
        refresh(out)
        if a.command in ('probe', 'check'):
            print(f'Preflight passed. File report: {out/"live/report.md"}', flush=True)
            return
        if not a.resume and any((out/f'L{d:04d}'/'sweep_state.json').exists() for d in specs):
            raise ValueError('Training already started; use --resume with the same options')
        first = out/'presets'/f'L{next(iter(specs)):04d}.json'
        subprocess.run([sys.executable, '-m', 'context_sweep.launch', 'prepare', '--preset', str(first),
                        '--data', str(Path(a.data).resolve())], check=True)
        bind_data(out, a.data)
        monitor_log = (out/'monitor.log').open('a')
        monitor = subprocess.Popen([sys.executable, '-m', 'quick_depth.report', '--root', str(out), '--watch'],
            stdout=monitor_log, stderr=subprocess.STDOUT, start_new_session=True)
        failed = None
        try:
            for depth, spec in specs.items():
                target, preset = out/f'L{depth:04d}', out/'presets'/f'L{depth:04d}.json'
                if read(preset) != spec:
                    raise ValueError(f'Preset changed after preflight: {preset}')
                for cell, _ in cells(spec):
                    if cancelled:
                        break
                    run = target/cell['name']
                    completed = read(run/'train_summary.json', {})
                    if completed.get('status') == 'complete' and completed.get('tokens') == cell['target_tokens']:
                        retire_completed_checkpoint(run)
                        continue
                    check_job_space(out, plan['storage']['parameter_counts'][str(depth)][cell['variant']], a.disk_reserve_gb)
                    command = [sys.executable, '-m', 'context_sweep.run', '--preset', str(preset),
                        '--data', str(Path(a.data).resolve()), '--output', str(target), '--device', a.device,
                        '--no-monitor', '--variants', cell['variant']]
                    if (target/'sweep_state.json').exists():
                        command += ['--resume']
                    log = out/'logs'/f'L{depth:04d}_{cell["variant"]}.log'
                    write(out/'quick_status.json', dict(phase='training', layers=depth, cell=cell['name'], log=str(log)))
                    result = supervise(command, log, lambda: cancelled, grace=180)
                    if result['status'] == 'failed':
                        failed = dict(layers=depth, cell=cell['name'], log=str(log))
                        break
                    if not cancelled:
                        retire_completed_checkpoint(run)
                if cancelled or failed:
                    break
            write(out/'quick_status.json', dict(phase='paused' if cancelled else 'failed' if failed else 'finished', failure=failed))
        except BaseException as error:
            write(out/'quick_status.json', dict(phase='failed', error=str(error)))
            raise
        finally:
            stop_group(monitor)
            try:
                monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop_group(monitor, signal.SIGKILL)
                monitor.wait()
            monitor_log.close()
            refresh(out)
        print(f'File report: {out/"live/report.md"}', flush=True)
        if failed:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
