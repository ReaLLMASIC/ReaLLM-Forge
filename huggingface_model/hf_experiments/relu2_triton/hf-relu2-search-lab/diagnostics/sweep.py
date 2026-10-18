"""Probe saved sweep checkpoints; keep failures separate from training results."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from context_sweep.recipe import ROOT, read, write, progress, fingerprint, digest


def complete_result(path, contexts, tokens):
    result=read(path,{})
    evaluations=result.get('evaluations',[])
    return (sorted(row['context'] for row in evaluations)==sorted(contexts) and
            all(row['evaluated_tokens']==tokens for row in evaluations))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tokens', type=int, default=32768)
    p.add_argument('--contexts', type=int, nargs='+', help='Default: all training contexts and the common evaluation context')
    p.add_argument('--run', help='Probe this run only')
    p.add_argument('--include-partial', action='store_true', help='Allow saved incomplete training checkpoints; labels retain their actual token count')
    a=p.parse_args()
    root=a.root.resolve(); data=a.data.resolve()
    plan=read(root/'plan.json')
    if not plan:p.error('--root must be a sweep output with plan.json')
    contexts=a.contexts or sorted(set(plan['contexts']+[plan['spec']['common_validation_context']]))
    if a.tokens<1 or len(set(contexts))!=len(contexts) or any(n<2 or a.tokens % n for n in contexts):
        p.error('Every evaluation context must divide --tokens exactly')
    if a.run and a.run not in {c['name'] for c in plan['cells']}:p.error('Unknown run name')
    directory=root/'diagnostics';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'diagnostics_state.json',{'runs':{}})
        failures=[]
        for cell in plan['cells']:
            name=cell['name']
            if a.run and name!=a.run:continue
            checkpoint,saved=progress(root,name)
            if not saved:continue
            if saved.get('cursor')!=cell['target_tokens'] and not a.include_partial:continue
            if cell['variant']=='kda':
                state['runs'][name]=dict(status='not_applicable',reason='KDA has no full attention rows',training_tokens=saved['cursor'])
                continue
            identity=dict(checkpoint=str(checkpoint.resolve()),training_tokens=saved['cursor'],
                config_sha256=hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
                weights_sha256={f.name:digest(f) for f in sorted(checkpoint.glob('*.safetensors'))},
                data_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
                source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                    for folder in ('hf_model','diagnostics') for f in sorted((ROOT/folder).glob('*.py'))},
                contexts=contexts,evaluation_tokens=a.tokens,device=a.device,dtype=plan['spec']['dtype'])
            key=fingerprint(identity)
            old=state['runs'].get(name,{})
            if old.get('identity')==key and old.get('status')=='complete' and complete_result(root/old['result'],contexts,a.tokens):
                continue
            base=directory/name/f"step-{saved['step']:08d}"
            attempt=1
            dest=base/f'attempt-{attempt}'
            while dest.exists():
                attempt+=1;dest=base/f'attempt-{attempt}'
            log=dest.parent/f'attempt-{attempt}.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            job=dict(status='running',identity=key,settings=identity,training_tokens=saved['cursor'],
                checkpoint_step=saved['step'],result=str((dest/'diagnostics.json').relative_to(root)),
                log=str(log.relative_to(root)),started_at=time.time())
            state['runs'][name]=job;write(root/'diagnostics_state.json',state)
            command=[sys.executable,'-m','diagnostics.probe_context','--project',str(ROOT),
                '--checkpoint',str(checkpoint),'--data',str(data),'--contexts',*map(str,contexts),
                '--tokens',str(a.tokens),'--device',a.device,'--dtype',plan['spec']['dtype'],'--output',str(dest)]
            print(f"Diagnostics: {name} at {saved['cursor']:,} training tokens; log {log}",flush=True)
            with log.open('w') as stream:
                stream.write('COMMAND '+json.dumps(command)+'\n');stream.flush()
                proc=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            success=proc.returncode==0 and complete_result(dest/'diagnostics.json',contexts,a.tokens)
            job.update(status='complete' if success else 'failed',returncode=proc.returncode,finished_at=time.time())
            if not success:
                failures.append(name);job['error']=log.read_text()[-3000:]
                print(f'Diagnostics failed: {name}; see {log}',flush=True)
            write(root/'diagnostics_state.json',state)
        write(root/'diagnostics_state.json',state)
    from context_sweep.report import refresh
    refresh(root,plots=True)
    if failures:raise SystemExit('Failed diagnostics (training is preserved): '+', '.join(failures))


if __name__=='__main__':main()
