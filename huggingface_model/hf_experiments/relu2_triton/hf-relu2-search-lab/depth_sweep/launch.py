"""Calibrate a common depth limit, freeze a grid, and run independent depth cohorts."""
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
from context_sweep.recipe import ROOT, read, write, build, digest, fingerprint
from context_sweep.run import supervise, stop_group
from .planning import depth_spec, depth_grid, find_limit, probe_cells
from .storage import available_budget, choose_grid, counts_for, check_job_space, retire_completed_checkpoint


def bind_data(out, data):
    data = Path(data).resolve()
    current = dict(path=str(data), manifest_sha256=digest(data/'manifest.json'))
    saved = read(out/'depth_data_identity.json')
    if saved is not None and saved != current:
        raise ValueError('Prepared data changed across depths; use the original data or a fresh output')
    write(out/'depth_data_identity.json', current)


def identity(base, args):
    from experiment.common import environment, source_hashes, configure_device
    import torch
    configure_device(args.device, base['dtype'])
    props = torch.cuda.get_device_properties(args.device)
    sources = source_hashes()
    for folder in ('depth_sweep', 'context_sweep'):
        sources.update({str(p.relative_to(ROOT)): digest(p) for p in (ROOT/folder).glob('*.py')})
    return dict(base=base, device=args.device, gpu_uuid=str(getattr(props, 'uuid', props.name)),
                gpu_bytes=props.total_memory, environment=environment(), source_hashes=sources,
                allocator_backend=torch.cuda.get_allocator_backend(),
                allocator_env={k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
                fraction=args.fraction, points=args.points, reserve_gib=args.reserve_gib,
                disk_budget_gb=args.disk_budget_gb, disk_reserve_gb=args.disk_reserve_gb,
                storage_policy='keep_final_weights_retire_completed_optimizer',
                max_search_layers=args.max_search_layers)


def verify_capacity_environment(plan, signature):
    import torch
    if plan['identity'] != signature:
        raise ValueError('Depth settings, sources, environment or GPU changed; choose a new output directory')
    free, _ = torch.cuda.mem_get_info()
    if free + (128 << 20) < plan['initial_free_bytes']:
        raise RuntimeError('Less GPU memory is free than during calibration; stop other GPU jobs before continuing')


def calibrate(base, args, out, signature, cancelled):
    import torch
    previous = read(out/'depth_plan.json')
    if previous:
        verify_capacity_environment(previous, signature)
        return previous
    initial_free, _ = torch.cuda.mem_get_info()
    start = base['model']['num_hidden_layers']
    observations = {}
    stamp = str(time.time_ns())
    trials = out/'capacity'/stamp
    history = []

    def fits(depth):
        if depth in observations:
            return observations[depth]
        # Capacity is common to all cells: stop at the first insufficient cell.
        # All chosen depths are checked for all cells once more via this function.
        for cell, cfg in probe_cells(base, depth):
            if cancelled():
                raise KeyboardInterrupt
            if available_budget(out,args.disk_budget_gb,args.disk_reserve_gb)<10**9:
                raise RuntimeError('Low disk space during capacity calibration; stop and free space before retrying')
            tag = f'L{depth:04d}_{cell["name"]}'
            request, result_path = trials/f'{tag}.request.json', trials/f'{tag}.result.json'
            write(request, dict(config=cfg, cell=cell, device=args.device, reserve_gib=args.reserve_gib))
            status = dict(phase='capacity_probe', layers=depth, cell=cell['name'],
                          log=str(trials/f'{tag}.log'), completed_probes=len(history))
            write(out/'depth_status.json', status)
            print(f'VRAM probe: {tag} (two full optimizer steps + validation)', flush=True)
            result = supervise([sys.executable, '-m', 'depth_sweep.worker', '--request', str(request),
                                '--result', str(result_path)], trials/f'{tag}.log', cancelled, grace=15)
            if cancelled():
                raise KeyboardInterrupt
            measured = read(result_path, {})
            if result['status'] != 'ok' or measured.get('status') not in ('fit','oom','budget_exceeded'):
                raise RuntimeError(f'Probe failed for a non-capacity reason. See {trials/f"{tag}.log"} '
                                   f'and {result_path}; do not interpret this as an OOM limit')
            history.append(dict(layers=depth, cell=cell['name'], **measured))
            write(out/'capacity_progress.json', dict(identity=signature, observations=history))
            if measured['status'] != 'fit':
                observations[depth] = False
                return False
        observations[depth] = True
        return True

    maximum, first_failure = find_limit(start, args.max_search_layers, fits)
    grid, storage = choose_grid(base,maximum,args.fraction,args.points,
        available_budget(out,args.disk_budget_gb,args.disk_reserve_gb))
    for depth in grid:
        if not fits(depth):
            raise RuntimeError(f'Chosen depth {depth} failed revalidation. Memory is not monotone '
                               'or GPU usage changed; no training plan was issued')
    plan = dict(identity=signature, initial_free_bytes=initial_free, base=base, layers=grid,
                storage=storage,
                measured_common_limit=maximum, first_failing_depth=first_failure,
                endpoint_fraction=args.fraction, contexts=[512,1024], reserve_gib=args.reserve_gib,
                tokens_per_run=build(depth_spec(base,start))['tokens_per_run'],
                runs_per_depth=len(probe_cells(base,start)), observations=history,
                note='Empirical two-step synthetic training/validation capacity, not a guarantee for arbitrary workloads. '
                     'Common limit across both contexts and every selected variant/norm/optimizer/seed. '
                     '70% applies to layer count, not total VRAM. All chosen grid points tested.')
    for depth in grid:
        write(out/'presets'/f'layers_{depth:04d}.json', depth_spec(base,depth))
    write(out/'depth_plan.json', plan)
    write(out/'depth_status.json', dict(phase='ready', layers=grid))
    print(json.dumps({k:v for k,v in plan.items() if k not in ('identity','observations','base')}, indent=2))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan','probe','all','train','report','monitor'])
    p.add_argument('--preset', default='configs/context_ablation_50m.json')
    p.add_argument('--data', default='data/fineweb_100m')
    p.add_argument('--output', default='runs/depth_16_3_disk30')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--points', type=int, default=3)
    p.add_argument('--start-layers', type=int, default=16)
    p.add_argument('--disk-budget-gb', type=float, default=30.)
    p.add_argument('--disk-reserve-gb', type=float, default=2.)
    p.add_argument('--fraction', type=float, default=.7)
    p.add_argument('--max-search-layers', type=int, default=256)
    p.add_argument('--reserve-gib', type=float, default=1.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--variants', nargs='+', help='Optional subset; defaults to all eight current arms')
    p.add_argument('--output-norms', nargs='+', choices=['none','capped'])
    a = p.parse_args()
    if a.points < 2 or a.start_layers<1 or not 0 < a.fraction <= 1 or not math.isfinite(a.reserve_gib) or a.reserve_gib < 0 or \
       not math.isfinite(a.disk_budget_gb) or not math.isfinite(a.disk_reserve_gb) or not 0<=a.disk_reserve_gb<a.disk_budget_gb:
        p.error('Invalid points, fraction or reserve')
    os.chdir(ROOT)
    out = Path(a.output).resolve()
    if a.command in ('report','monitor'):
        from .report import refresh, watch
        (watch if a.command=='monitor' else refresh)(out)
        return
    base = read(a.preset)
    if a.variants:
        base['variants'] = a.variants
    if a.output_norms:
        base['output_norms'] = a.output_norms
    base['model']['num_hidden_layers']=a.start_layers
    start = a.start_layers
    base = depth_spec(base,start)
    plan0 = build(base)
    if a.command=='plan':
        print(json.dumps(dict(start_layers=start, width=base['model']['hidden_size'],
            heads=base['model']['num_attention_heads'], contexts=[512,1024],
            endpoint=f'min(floor({a.fraction} * measured common layer limit), disk-fitting endpoint)', points=a.points,
            maximum_run_count=a.points*plan0['run_count'], tokens_per_run=plan0['tokens_per_run'],
            disk_budget_gb=a.disk_budget_gb,disk_reserve_gb=a.disk_reserve_gb,
            storage_policy='Final weights and reports kept; completed optimizer states retired; interrupted jobs resumable',
            variants=base['variants'], output_norms=base['output_norms'],
            batches={k:v['train']['micro_batch_size'] for k,v in plan0['configs'].items()},
            note='Run probe on an idle 4090 to resolve the grid; no VRAM estimate is fabricated.'), indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.depth.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'plan.json').exists() or (out/'sweep_state.json').exists():
            raise ValueError('This is an old context sweep output; choose a separate depth output')
        from context_sweep.launch import tests, require_cuda
        require_cuda()
        if a.command=='all':
            subprocess.run([sys.executable,'scripts/check_install.py'], check=True)
            tests()
        signature = identity(base,a)
        if not (out/'depth_plan.json').exists():
            # Reject impossible minimum grids before spending time on GPU capacity search.
            choose_grid(base,start+a.points-1,1.,a.points,
                        available_budget(out,a.disk_budget_gb,a.disk_reserve_gb))
        cancelled = False
        def stop(*_):
            nonlocal cancelled
            cancelled = True
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,stop)
        if a.command=='train':
            plan = read(out/'depth_plan.json')
            if not plan:
                raise ValueError('Run probe first (or use all)')
            verify_capacity_environment(plan,signature)
        else:
            plan = calibrate(base,a,out,signature,lambda:cancelled)
        if a.command=='probe':
            from .report import refresh
            refresh(out)
            return
        if not a.resume and any((out/f'L{d:04d}'/'sweep_state.json').exists() for d in plan['layers']):
            raise ValueError('Depth training already started; use --resume')
        # Prepare once; same token stream and original ~100M-token LR horizon.
        first_preset = out/'presets'/f'layers_{plan["layers"][0]:04d}.json'
        subprocess.run([sys.executable,'-m','context_sweep.launch','prepare','--preset',str(first_preset),
                        '--data',str(Path(a.data).resolve())], check=True)
        bind_data(out, a.data)
        monitor_log = (out/'depth_monitor.log').open('a')
        monitor = subprocess.Popen([sys.executable,'-m','depth_sweep.report','--root',str(out),'--watch'],
                                   stdout=monitor_log,stderr=subprocess.STDOUT,start_new_session=True)
        failed = []
        try:
            for depth in plan['layers']:
                if cancelled:
                    break
                target = out/f'L{depth:04d}'
                preset = out/'presets'/f'layers_{depth:04d}.json'
                # Frozen calibration must not silently diverge from training.
                if read(preset) != depth_spec(base,depth):
                    raise ValueError(f'Generated preset changed: {preset}')
                counts=counts_for(base,depth)
                for cell,_ in probe_cells(base,depth):
                    if cancelled:break
                    run=target/cell['name']
                    summary=read(run/'train_summary.json',{})
                    if summary.get('status')=='complete' and summary.get('tokens')==cell['target_tokens']:
                        retire_completed_checkpoint(run)
                        continue
                    check_job_space(out,counts[(cell['variant'],cell['output_norm'])],a.disk_reserve_gb)
                    args = [sys.executable,'-m','context_sweep.run','--preset',str(preset),
                            '--data',str(Path(a.data).resolve()),'--output',str(target),'--device',a.device,'--no-monitor',
                            '--contexts',str(cell['context']),'--variants',cell['variant'],
                            '--output-norms',cell['output_norm'],'--optimizers',cell['optimizer'],'--seeds',str(cell['seed'])]
                    if (target/'sweep_state.json').exists():args += ['--resume']
                    write(out/'depth_status.json',dict(phase='training',layers=depth,cell=cell['name'],log=str(out/f'L{depth:04d}.log')))
                    result = supervise(args,out/f'L{depth:04d}.log',lambda:cancelled,grace=180)
                    if result['status']=='failed':
                        failed.append(depth)
                        break  # Do not accumulate failed full optimizer checkpoints.
                    if not cancelled:retire_completed_checkpoint(run)
                if failed:break
            write(out/'depth_status.json',dict(phase='paused' if cancelled else 'failed' if failed else 'finished',
                                             failed_depths=failed))
        finally:
            stop_group(monitor)
            try:
                monitor.wait(timeout=15)
            except subprocess.TimeoutExpired:
                stop_group(monitor,signal.SIGKILL)
                monitor.wait()
            monitor_log.close()
            from .report import refresh
            refresh(out)
        print(f'File report: {out/"live/report.md"}',flush=True)
        if failed:
            raise SystemExit(2)


if __name__=='__main__':
    main()
