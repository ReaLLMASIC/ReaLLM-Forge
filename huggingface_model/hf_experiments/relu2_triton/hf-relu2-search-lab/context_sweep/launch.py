"""One entry point for preparation, tests, training, diagnostics and file reports."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .recipe import ROOT, read, build


def run(module, args=(), env=None):
    subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=ROOT, check=True,
                   env=dict(os.environ, **(env or {})))


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the full sweep and GPU tests. Use smoke for the offline CPU pipeline.')
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('The fused kernels require SM80 or newer.')


def tests(cuda=False):
    if cuda:
        require_cuda()  # Skipped CUDA tests must never count as the GPU gate.
    args=['-q', 'tests'] if not cuda else ['-q', 'tests/test_context_cuda.py', 'tests/test_cuda_training.py']
    run('pytest', args, {'PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan','prepare','check','test','test-cuda','smoke','train','diagnose','report','monitor','all'])
    p.add_argument('--preset', default='configs/context_ablation_50m.json')
    p.add_argument('--data', default='data/fineweb_100m')
    p.add_argument('--output', default='runs/context_ablation_50m')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--contexts', type=int, nargs='+')
    p.add_argument('--variants', nargs='+')
    p.add_argument('--output-norms', choices=['none','capped'], nargs='+')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--no-monitor', action='store_true')
    p.add_argument('--port', type=int, default=8774)
    p.add_argument('--probe-tokens', type=int, default=32768)
    p.add_argument('--local-jsonl')
    p.add_argument('--tokenizer', default='openai-community/gpt2')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    a=p.parse_args()
    os.chdir(ROOT)
    if a.command in ('test','test-cuda'):
        tests(a.command=='test-cuda');return
    if a.command=='smoke':
        out=a.output if a.output!='runs/context_ablation_50m' else 'runs/context_smoke'
        run('experiment.smoke',['--output',out,'--preset','configs/context_ablation_smoke.json','--diagnostics']);return
    if a.command in ('report','monitor'):
        run('context_sweep.monitor',['--root',a.output,'--port',a.port]+(['--once'] if a.command=='report' else []));return
    if a.command=='diagnose':
        args=['--root',a.output,'--data',a.data,'--device',a.device,'--tokens',a.probe_tokens]
        if a.contexts:args+=['--contexts',*a.contexts]
        run('diagnostics.sweep',args);return
    args=['--preset',a.preset,'--data',a.data,'--output',a.output,'--device',a.device,'--port',a.port]
    for flag in ('contexts','variants','output_norms','seeds'):
        value=getattr(a,flag)
        if value:args+=['--'+flag.replace('_','-'),*value]
    if a.resume:args+=['--resume']
    if a.no_monitor:args+=['--no-monitor']
    if a.command=='plan':run('context_sweep.run',args+['--dry-run']);return
    if a.command=='check':run('context_sweep.run',args+['--check-only']);return
    if a.command=='all':
        require_cuda()
        subprocess.run([sys.executable,'scripts/check_install.py'],cwd=ROOT,check=True)
        tests()  # Includes the actual GPU tests on this host.
    if a.command in ('prepare','all'):
        plan=build(read(a.preset))
        data=Path(a.data)
        if (data/'manifest.json').is_file():
            print(f'Using prepared data: {data}; training preflight verifies its identity.',flush=True)
        else:
            prep=['--output',data,'--train-tokens',plan['tokens_per_run']+1,
                  '--validation-tokens',max(plan['spec']['validation_tokens'],a.probe_tokens)+1,
                  '--tokenizer',a.tokenizer,'--dataset',a.dataset,'--dataset-config',a.dataset_config]
            if a.local_jsonl:prep+=['--local-jsonl',a.local_jsonl]
            run('context_sweep.prepare',prep)
        if a.command=='prepare':return
    if a.command in ('train','all'):
        run('context_sweep.run',args+['--diagnostics-on-complete','--probe-tokens',a.probe_tokens])
        # Also fills gaps after an interrupted diagnostic or a previously completed run.
        run('diagnostics.sweep',['--root',a.output,'--data',a.data,'--device',a.device,'--tokens',a.probe_tokens])


if __name__=='__main__':main()
