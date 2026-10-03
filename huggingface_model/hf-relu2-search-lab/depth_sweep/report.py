"""File-first progress, CSV and matched-budget loss-versus-depth plots."""
import argparse
import fcntl
import html
import math
import os
from pathlib import Path
import signal
import statistics
import threading
import time
from context_sweep.recipe import read, write, build
from context_sweep.report import collect, atomic_csv, COLORS, LABELS
from .planning import depth_spec


def gather(root):
    root = Path(root)
    plan = read(root/'depth_plan.json', {})
    rows = []
    for depth in plan.get('layers', []):
        child = root/f'L{depth:04d}'
        data = collect(child)
        if data['runs']:
            rows.extend(dict(r,layers=depth,initialization_ok=not data['initialization_mismatch_seeds']) for r in data['runs'])
        else:
            for cell in build(depth_spec(plan['base'],depth))['cells']:
                rows.append(dict(cell,layers=depth,status='pending',tokens=0,completed_budget=False,
                                 initialization_ok=True,best_validation_loss=None,best_common_loss=None,
                                 final_validation_loss=None))
    return dict(plan=plan,runs=rows,status=read(root/'depth_status.json',{}),updated_at=time.time())


def completed_mean(rows, metric, seeds):
    # Never turn missing/failed runs into zero or mix incomplete token budgets.
    if len(rows)!=len(seeds) or {r['seed'] for r in rows}!=set(seeds):
        return math.nan
    if not all(r.get('completed_budget') and r.get('initialization_ok') and
               isinstance(r.get(metric),(int,float)) and math.isfinite(r[metric]) for r in rows):
        return math.nan
    return statistics.mean(r[metric] for r in rows)


def plots(live,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=data['plan']
    if not plan:
        return
    base=plan['base']
    for metric,name,title in (
        ('best_validation_loss','best_validation_vs_layers','Best scheduled native-context validation'),
        ('best_common_loss','best_common512_vs_layers','Best scheduled shared-512-context validation'),
        ('final_validation_loss','final_validation_vs_layers','Final native-context validation')):
        norms=base['output_norms']
        fig,axes=plt.subplots(len(norms),2,squeeze=False,figsize=(13,4.5*len(norms)),layout='constrained')
        for i,norm in enumerate(norms):
            for j,context in enumerate((512,1024)):
                ax=axes[i,j]
                has_values=False
                for variant in base['variants']:
                    for optimizer in base['optimizers']:
                        values=[]
                        for depth in plan['layers']:
                            members=[r for r in data['runs'] if r['layers']==depth and r['context']==context and
                                     r['variant']==variant and r['optimizer']==optimizer and r['output_norm']==norm]
                            values.append(completed_mean(members,metric,base['seeds']))
                        has_values |= any(math.isfinite(v) for v in values)
                        ax.plot(plan['layers'],values,marker='o',color=COLORS[variant],
                                linestyle='--' if optimizer=='adamw' else '-',
                                label=LABELS[variant]+(' / '+optimizer if len(base['optimizers'])>1 else ''))
                ax.set(title=f'Train context {context} · output norm {norm}',xlabel='Decoder layers',
                       ylabel='Validation CE (nats/token)',xticks=plan['layers'])
                margin=max(1,(max(plan['layers'])-min(plan['layers']))*.08)
                ax.set_xlim(min(plan['layers'])-margin,max(plan['layers'])+margin)
                if not has_values:
                    ax.set_yticks([])
                    ax.text(.5,.5,'No completed-budget points yet',ha='center',va='center',
                            transform=ax.transAxes,color='#666666')
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
        fig.suptitle(title+f'\n{plan["tokens_per_run"]:,} training tokens/run; completed budgets only · '+
                     ('single seed, no uncertainty estimate' if len(base['seeds'])==1 else 'mean over planned seeds'))
        for ext in ('png','pdf'):
            target=live/f'{name}.{ext}'
            temp=live/f'{name}.{os.getpid()}.tmp.{ext}'
            fig.savefig(temp,dpi=150)
            os.replace(temp,target)
        plt.close(fig)


def refresh(root,draw=True):
    root=Path(root)
    live=root/'live'
    live.mkdir(parents=True,exist_ok=True)
    data=gather(root)
    write(live/'depth_summary.json',data)
    atomic_csv(live/'depth_runs.csv',data['runs'])
    plan=data['plan']
    lines=['# Depth sweep', '', f'Updated: {time.ctime(data["updated_at"])}', '',
           f'Phase: {data["status"].get("phase","not started")}', '',
           'Partial runs are progress only. Curves require completed, equal-token budgets for every planned seed.', '']
    if plan:
        storage=plan.get('storage',{})
        if storage:
            lines += [f'Disk estimate: {storage["estimated_peak_bytes"]/1e9:.2f} GB; '
                      f'planned budget excluding free-space reserve: {storage["budget_bytes"]/1e9:.2f} GB. '
                      f'Disk-limited endpoint: {storage["disk_limited"]}.',
                      'Completed runs retain final weights and metrics; optimizer state is retired. Interrupted jobs remain resumable.', '']
        lines += [f'Layers: {plan["layers"]}; common measured limit: {plan["measured_common_limit"]}; '
                  f'endpoint fraction: {plan["endpoint_fraction"]}.',
                  f'Tokens/run: {plan["tokens_per_run"]:,}; '
                  f'completed: {sum(r["completed_budget"] for r in data["runs"])}/{len(data["runs"])}.', '',
                  'Best-so-far numbers below are not matched-budget comparisons until complete.', '',
                  '| Layers | Context | Variant | Optimizer | Norm | Seed | Status | Tokens | Tokens/s | Best loss so far |',
                  '|---:|---:|---|---|---|---:|---|---:|---:|---:|']
        def fmt(x):
            return f'{x:.4f}' if isinstance(x,(int,float)) else '—'
        for r in data['runs']:
            lines.append(f'| {r["layers"]} | {r["context"]} | {r["variant"]} | {r["optimizer"]} | '
                         f'{r["output_norm"]} | {r["seed"]} | {r["status"]} | {r["tokens"]:,} | '
                         f'{fmt(r.get("recent_tokens_per_second"))} | {fmt(r.get("best_validation_loss"))} |')
        lines += ['', '## Jobs needing attention', '']
        lines += [f'- L{r["layers"]:04d}/{r["name"]}: {r["log"]}' for r in data['runs'] if r['status']=='failed']
    text='\n'.join(lines)+'\n'
    temp=live/f'report.{os.getpid()}.tmp'
    temp.write_text(text)
    os.replace(temp,live/'report.md')
    if draw:
        plots(live,data)
    page='<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Depth sweep</title></head>'
    page+='<body style="font-family:system-ui"><h1>Depth sweep — live files</h1>'
    page+='<p><a href="depth_runs.csv">CSV</a> · <a href="depth_summary.json">JSON</a> · <a href="report.md">Markdown report</a></p>'
    for name in ('best_validation_vs_layers','best_common512_vs_layers','final_validation_vs_layers'):
        if (live/f'{name}.png').exists():
            page+=f'<img style="max-width:100%" src="{name}.png?t={int(data["updated_at"])}" alt="{name}">'
    page+='<pre>'+html.escape(text)+'</pre></body></html>'
    temp=live/f'index.{os.getpid()}.tmp'
    temp.write_text(page)
    os.replace(temp,live/'index.html')
    return data


def watch(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.depth-report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop=threading.Event()
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,lambda *_:stop.set())
        last=0
        while not stop.is_set():
            draw=time.monotonic()-last>=60
            refresh(root,draw)
            if draw:
                last=time.monotonic()
            stop.wait(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    (watch if a.watch else refresh)(a.root)
