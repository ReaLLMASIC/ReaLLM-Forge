"""Read-only file reports: missing or failed jobs remain missing measurements."""
import argparse
import csv
import html
import io
import os
from pathlib import Path
import time
from context_sweep.recipe import read, write


def atomic_text(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(value)
    os.replace(temporary,path)


def collect(output):
    output = Path(output)
    plan = read(output / 'plan.json')
    if not plan:
        raise ValueError('No plan.json found; create the plan first')
    state = read(output / 'state.json', {})
    rows = []
    for job in plan['jobs']:
        path = output / 'runs' / job['name']
        summary = read(path / 'train_summary.json', {})
        info = read(path / 'run.json', {})
        events = []
        if (path / 'metrics.jsonl').is_file():
            import json
            for line in (path / 'metrics.jsonl').read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # An in-flight final write is not a zero measurement.
        trains = [e for e in events if e.get('kind') == 'train']
        vals = [e for e in events if e.get('kind') == 'validation']
        eligible = [e for e in vals if e.get('eligible_for_best')]
        best = min((e['loss'] for e in eligible), default=None)
        common = min((e['common_loss'] for e in eligible), default=None)
        architecture = next(a for a in plan['architectures'] if a['name'] == job['architecture'])
        status = state.get(job['name'], {}).get('status', summary.get('status', 'pending'))
        tokens = max((e.get('tokens', 0) for e in events), default=summary.get('tokens', 0))
        complete = summary.get('status') == 'complete' and summary.get('tokens') == job['target_tokens']
        if complete:
            status = 'complete'
        row = dict(run=job['name'], architecture=job['architecture'], variant=job['variant'], seed=job['seed'],
                   status=status, tokens=tokens, target_tokens=job['target_tokens'], context=job['context'],
                   parameters=info.get('parameter_counts', {}).get('total'),
                   best_validation_loss=best, best_common_loss=common,
                   final_validation_loss=(summary.get('validation') or {}).get('loss') if complete else None,
                   recent_tokens_per_second=trains[-1].get('tokens_per_second') if trains else None,
                   steady_tokens_per_second=summary.get('steady_tokens_per_second'),
                   training_seconds=summary.get('train_step_seconds'),
                   peak_allocated_bytes=summary.get('peak_allocated_bytes'), completed_budget=complete,
                   padded_kernel_dim=architecture['padding']['kernel_dim'],
                   qk_padding_ratio=architecture['padding']['qk_ratio'], value_padding_ratio=architecture['padding']['value_ratio'],
                   backends=','.join(summary.get('backends',trains[-1].get('backends',[]) if trains else [])),
                   log=state.get(job['name'],{}).get('log',str(output / 'logs' / (job['name'] + '.log'))))
        rows.append(row)
    return plan, rows


def render(output, plots=True):
    output = Path(output)
    plan, rows = collect(output)
    write(output / 'results.json', dict(updated_unix=time.time(), runs=rows))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output / 'results.csv',buffer.getvalue())
    lines = ['# HF attention architecture search', '',
        'Fixed-token screening. Compare native validation only within the same context. '
        'Common-context validation is comparable across contexts. Missing/failed jobs are not zero scores.', '',
        f'Updated: {time.ctime()}', '',
        '| Run | Status | Tokens | Best native loss | Best common loss | Recent tokens/s |',
        '|---|---|---:|---:|---:|---:|']
    def fmt(x):
        return '—' if x is None else f'{x:.4f}' if isinstance(x, float) else str(x)
    for r in rows:
        lines.append('| ' + ' | '.join(fmt(r[k]) for k in ('run','status','tokens','best_validation_loss','best_common_loss','recent_tokens_per_second')) + ' |')
    lines += ['', '## Files to monitor', '', '`results.csv`, `results.json`, `report.md`, `report.html`, and per-run `metrics.jsonl`.', '',
              'Graphs rank only runs that completed the identical token budget. Single-seed differences are preliminary.', '', '## Jobs needing attention', '']
    for row in rows:
        if row['status'] in ('failed', 'oom', 'budget_exceeded', 'error'):
            lines.append(f"- {row['run']}: {row['status']}; log `{row['log']}`")
    text = '\n'.join(lines) + '\n'
    atomic_text(output / 'report.md',text)
    atomic_text(output / 'report.html','<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15">'
        '<title>HF search results</title><style>body{font-family:system-ui;max-width:1400px;margin:2em auto}pre{white-space:pre-wrap}</style>'
        '<pre>' + html.escape(text) + '</pre><img width="900" src="validation_progress.png"><img width="900" src="loss_by_architecture.png"><img width="900" src="loss_vs_training_time.png">')
    if plots:
        plot(output, rows)
    return rows


def plot(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'softmax_sdpa':'#1976d2', 'relu2':'#e67e22', 'linear16':'#228b22'}
    complete = [r for r in rows if r['completed_budget']]
    names = list(dict.fromkeys(r['architecture'] for r in rows))
    import json
    fig, axes = plt.subplots(len(names),1,figsize=(8,max(4,3.4*len(names))),squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index,0]
        for row in rows:
            if row['architecture'] != name: continue
            path = output/'runs'/row['run']/'metrics.jsonl'
            events=[]
            if path.is_file():
                for line in path.read_text().splitlines():
                    try: event=json.loads(line)
                    except json.JSONDecodeError: continue
                    if event.get('kind') == 'validation': events.append(event)
            if events:
                ax.plot([e['tokens']/1e6 for e in events],[e['loss'] for e in events],marker='.',
                    label=f"{row['variant']} seed{row['seed']}",color=colors[row['variant']])
        ax.set_title(f'{name} — native context validation (live, including partial runs)')
        ax.set_xlabel('Training tokens (millions)'); ax.set_ylabel('Validation loss'); ax.grid(alpha=.2)
        if ax.lines: ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(output / ('validation_progress.'+ext))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.4), 4.8))
    for variant, color in colors.items():
        selected = [r for r in complete if r['variant'] == variant and r['best_common_loss'] is not None]
        ax.scatter([names.index(r['architecture']) for r in selected], [r['best_common_loss'] for r in selected], label=variant, color=color)
    ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
    ax.set_ylabel('Best common-context validation loss')
    ax.set_title('Completed fixed-token runs only (individual seeds)')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(output / ('loss_by_architecture.'+ext))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.8))
    for variant, color in colors.items():
        selected = [r for r in complete if r['variant'] == variant and r['best_common_loss'] is not None and r['training_seconds'] is not None]
        ax.scatter([r['training_seconds']/3600 for r in selected], [r['best_common_loss'] for r in selected], label=variant, color=color)
    ax.set_xlabel('Full optimizer-step training hours (excludes evaluation/export)')
    ax.set_ylabel('Best common-context validation loss')
    ax.set_title('Loss / training time trade-off; completed runs only')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(output / ('loss_vs_training_time.'+ext))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/hf_search')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--interval', type=float, default=15)
    a = p.parse_args()
    if a.interval < 1: p.error('interval must be at least one second')
    while True:
        render(a.output)
        if not a.watch: break
        time.sleep(a.interval)


if __name__ == '__main__': main()
