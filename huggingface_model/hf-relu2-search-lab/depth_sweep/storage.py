"""Bound retained weights; retire only verified, completed resume checkpoints."""
from pathlib import Path
import shutil
import os
from context_sweep.recipe import build, read, write, digest
from .planning import depth_spec, depth_grid, probe_cells

GB = 10**9  # Drive budgets are decimal GB, not GiB.
OVERHEAD = 3 * GB  # Packed data, reports, logs and compiler/download caches.


def counts_for(base, depth):
    import torch
    from hf_model.configuration_comparison import ComparisonConfig
    from hf_model.modeling_comparison import ComparisonForCausalLM
    counts = {}
    for cell, cfg in probe_cells(base, depth):
        key = (cell['variant'], cell['output_norm'])
        if key not in counts:
            with torch.device('meta'):
                model = ComparisonForCausalLM(ComparisonConfig(**dict(cfg['model'],variant=cell['variant'])))
            counts[key] = sum(p.numel() for p in model.parameters())
    return counts


def estimate(base, layers, counter=counts_for):
    retained, largest, by_depth = 0, 0, {}
    for depth in layers:
        counts = counter(base,depth)
        subtotal = sum(4 * counts[(c['variant'],c['output_norm'])] for c in build(depth_spec(base,depth))['cells'])
        by_depth[str(depth)] = subtotal
        retained += subtotal
        largest = max(largest,max(counts.values()))
    # FP32 model + Muon/AdamW state, old/new atomic checkpoints, final export.
    # 32 bytes/parameter exceeds two full AdamW checkpoints plus one export.
    # Retained weights already include every final export, including the active run.
    transient = 32 * largest
    return dict(final_weight_bytes=retained, transient_checkpoint_bytes=transient,
                overhead_bytes=OVERHEAD, estimated_peak_bytes=retained+transient+OVERHEAD,
                final_weight_bytes_by_depth=by_depth)


def choose_grid(base, maximum, fraction, points, budget, counter=counts_for):
    start=base['model']['num_hidden_layers']
    upper=depth_grid(start,maximum,fraction,points)[-1]
    low=start+points-1
    if upper<low:
        raise ValueError('VRAM limit cannot supply the requested number of distinct depths')
    def candidate(end):
        grid=depth_grid(start,end,1.,points)
        return grid,estimate(base,grid,counter)
    grid,cost=candidate(low)
    if cost['estimated_peak_bytes']>budget:
        raise ValueError(f'Even the smallest {points}-depth grid needs approximately '
                         f'{cost["estimated_peak_bytes"]/GB:.1f} GB plus the disk reserve. '
                         'Reuse the existing environment/data, free space, or explicitly select fewer variants/norms. '
                         'No models were deleted and training has not started.')
    high=upper
    while low<high:
        mid=(low+high+1)//2
        if candidate(mid)[1]['estimated_peak_bytes']<=budget:low=mid
        else:high=mid-1
    grid,cost=candidate(low)
    return grid,dict(cost,budget_bytes=budget,vram_endpoint=upper,disk_limited=low<upper)


def available_budget(path, budget_gb=30., reserve_gb=2.):
    return min(int(budget_gb*GB),shutil.disk_usage(path).free)-int(reserve_gb*GB)


def check_job_space(path, parameters, reserve_gb=2.):
    required=32*parameters+GB+int(reserve_gb*GB)
    free=shutil.disk_usage(path).free
    if free<required:
        raise RuntimeError(f'Low disk space: {free/GB:.2f} GB free; need {required/GB:.2f} GB '
                           'for checkpoint rotation and reserve. Existing results/checkpoints are preserved.')


def retire_completed_checkpoint(run):
    """New sweep only: preserve final weights, metrics, progress; remove duplicate state.

    Caller must run one job at a time and invoke this only after successful exit.
    A completed HF export is opened and tensor inventories compared before deletion.
    Marker hashes record exactly which final files were verified. No run tree is removed.
    """
    from safetensors import safe_open
    run=Path(run)
    summary=read(run/'train_summary.json',{})
    latest=read(run/'latest.json',{})
    checkpoint=run/latest.get('checkpoint','missing')
    progress=read(checkpoint/'progress.json',{})
    info=read(run/'run.json',{})
    if summary.get('status')!='complete' or not info.get('actual_token_budget') or \
       summary.get('tokens')!=info['actual_token_budget'] or progress.get('cursor')!=info['actual_token_budget']:
        raise ValueError('Refusing checkpoint retirement for an incomplete or unverified run')
    if checkpoint.parent!=run or checkpoint.is_symlink() or not checkpoint.name.startswith('checkpoint-'):
        raise ValueError('Unsafe checkpoint path')
    final=run/'final'
    if final.is_symlink() or not (final/'config.json').is_file():
        raise ValueError('Missing final HF export')
    marker=read(run/'storage_retention.json')
    if marker:
        if any(not (final/n).is_file() or digest(final/n)!=h for n,h in marker['final_sha256'].items()):
            raise ValueError('Final export changed after checkpoint retirement')
        # Continue an interrupted cleanup only after re-verifying recorded exports.
    else:
        def inventory(folder):
            result={}
            files=sorted(folder.glob('*.safetensors'))
            if not files:raise ValueError('Missing safetensors weights; preserve resume checkpoint')
            for file in files:
                if file.is_symlink():raise ValueError('Refusing linked weights')
                with safe_open(file,framework='pt',device='cpu') as f:
                    for key in f.keys():
                        if key in result:raise ValueError('Duplicate tensor name')
                        # Hash tensor content too: intact but wrong final weights must not qualify.
                        import hashlib
                        tensor=f.get_tensor(key)
                        result[key]=(tuple(tensor.shape),str(tensor.dtype),hashlib.sha256(tensor.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest())
            return result,files
        original,_=inventory(checkpoint)
        exported,files=inventory(final)
        if original!=exported:raise ValueError('Final weights differ; resume checkpoint preserved')
        marker=dict(policy='final_weights_and_metrics',exact_optimizer_resume=False,
                    final_sha256={p.name:digest(p) for p in files})
        write(run/'storage_retention.json',marker)
    # Keep checkpoint model loading/diagnostics working, without a second copy.
    # Atomic hardlink replacement consumes no additional model-data blocks.
    names=list(marker['final_sha256'])
    if (final/'model.safetensors.index.json').is_file():names.append('model.safetensors.index.json')
    for name in names:
        source=final/name
        target=checkpoint/name
        if source.is_symlink() or target.is_symlink():raise ValueError('Refusing linked checkpoint path')
        if target.exists() and os.path.samefile(source,target):continue
        temporary=checkpoint/('.retention-'+name)
        if temporary.exists():temporary.unlink()
        os.link(source,temporary)  # Fail safely if filesystem does not support hardlinks.
        os.replace(temporary,target)
    obsolete=[p for p in checkpoint.glob('*.safetensors') if p.name not in names]
    if 'model.safetensors.index.json' not in names:obsolete.append(checkpoint/'model.safetensors.index.json')
    # Retain progress.json and shared model files; retire only duplicate/optimizer data.
    for path in [checkpoint/'training_state.pt',*obsolete]:
        if path.is_symlink():raise ValueError('Refusing linked checkpoint file')
        if path.is_file():path.unlink()
