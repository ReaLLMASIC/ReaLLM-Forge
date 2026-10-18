"""Live files and matched-token depth curves; a localhost server is not required."""
import argparse
import fcntl
import html
import math
import os
from pathlib import Path
import signal
import threading
import time
from context_sweep.recipe import read, write, build
from context_sweep.report import collect, atomic_csv, COLORS, LABELS
from depth_sweep.report import completed_mean


def gather(root):
    root = Path(root)
    plan = read(root/'quick_plan.json', {})
    runs, validations = [], []
    for depth in plan.get('layers', []):
        data = collect(root/f'L{depth:04d}')
        if data['runs']:
            runs.extend(dict(row, layers=depth, initialization_ok=not data['initialization_mismatch_seeds']) for row in data['runs'])
            validations.extend(dict(row, layers=depth) for row in data['validations'])
        else:
            for cell in build(plan['specs'][str(depth)])['cells']:
                runs.append(dict(cell, layers=depth, status='pending', tokens=0, completed_budget=False,
                    initialization_ok=True, best_validation_loss=None, final_validation_loss=None))
    return dict(plan=plan, runs=runs, validations=validations,
                status=read(root/'quick_status.json', {}), updated_at=time.time())


def plots(live, data):
    if not data['plan']:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = data['plan']
    for metric, name, title in (
        ('best_validation_loss', 'best_validation_vs_layers', 'Best scheduled validation'),
        ('final_validation_loss', 'final_validation_vs_layers', 'Final validation')):
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        any_values = False
        for variant in plan['variants']:
            values = [completed_mean([r for r in data['runs'] if r['layers'] == depth and r['variant'] == variant],
                metric, plan['specs'][str(depth)]['seeds']) for depth in plan['layers']]
            any_values |= any(math.isfinite(v) for v in values)
            ax.plot(plan['layers'], values, marker='o', color=COLORS[variant], label=LABELS[variant])
        if not any_values:
            ax.set_yticks([])
            ax.text(.5, .5, 'No completed-budget points yet', ha='center', transform=ax.transAxes)
        ax.set(xlabel='Decoder layers', ylabel='Validation CE (nats/token)', xticks=plan['layers'],
            title=f'{title} · width 512 · context 2048\n{plan["tokens_per_run"]:,} tokens/run · Muon decay 0 · single seed')
        ax.grid(alpha=.2)
        ax.legend()
        save_figure(fig, live, name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for depth in plan['layers']:
        for variant in plan['variants']:
            # Partial training curves are explicitly separated from full-budget
            # comparisons; pause-only validation and random-init checks excluded.
            rows = sorted([r for r in data['validations'] if r['layers'] == depth and r['variant'] == variant
                and r.get('eligible_for_best') and isinstance(r.get('loss'), (int, float))
                and math.isfinite(r['loss'])], key=lambda r: r['tokens'])
            if rows:
                ax.plot([r['tokens']/1e6 for r in rows], [r['loss'] for r in rows],
                    label=f'L{depth} {LABELS[variant]}', linestyle='--' if variant == 'softmax_sdpa' else '-')
    if ax.lines:
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, 'Waiting for scheduled validation', ha='center', transform=ax.transAxes)
    ax.set(xlabel='Training tokens (millions)', ylabel='Validation CE (nats/token)',
        title='Live validation progress · partial curves included\nCompare at equal tokens; depth curves require completed runs')
    ax.grid(alpha=.2)
    save_figure(fig, live, 'validation_progress')
    plt.close(fig)


def save_figure(fig, live, name):
    for ext in ('png', 'pdf'):
        target = live/f'{name}.{ext}'
        temp = live/f'{name}.{os.getpid()}.tmp.{ext}'
        fig.savefig(temp, dpi=150)
        os.replace(temp, target)


def atomic_text(path, text):
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(text)
    os.replace(temp, path)


def refresh(root, draw=True):
    root = Path(root)
    live = root/'live'
    live.mkdir(parents=True, exist_ok=True)
    data = gather(root)
    write(live/'quick_summary.json', data)
    atomic_csv(live/'depth_runs.csv', data['runs'])
    atomic_csv(live/'validations.csv', data['validations'])
    plan = data['plan']
    lines = ['# Width 512 / context 2048 depth comparison', '',
        f'Updated: {time.ctime(data["updated_at"])}', '',
        f'Phase: {data["status"].get("phase", "not started")}', '',
        'ReLU² versus softmax SDPA; Muon decay 0; auxiliary AdamW matrix decay 0.1. No output norm.', '',
        'Single seed. Compare equal training tokens. Larger depths use more parameters and compute.',
        'Partial results are progress only; missing/failed runs are never zero-valued graph points.', '']
    if plan:
        cost = plan['storage']
        lines += [f'Layers: {plan["layers"]}. Tokens/run: {plan["tokens_per_run"]:,}. '
            f'Complete: {sum(r["completed_budget"] for r in data["runs"])}/{len(data["runs"])}.', '',
            f'Disk estimate: {cost["estimated_peak_bytes"]/1e9:.2f} GB; available planning budget '
            f'after reserve: {cost["budget_bytes"]/1e9:.2f} GB.',
            'Final model weights and reports are retained. Optimizer state is retired only after successful completion; interrupted runs remain resumable.', '',
            '| Layers | Variant | Status | Tokens | Recent tokens/s | Best loss so far | Final loss |',
            '|---:|---|---|---:|---:|---:|---:|']
        def fmt(value):
            return f'{value:.4f}' if isinstance(value, (int, float)) else '—'
        for row in data['runs']:
            lines.append(f'| {row["layers"]} | {row["variant"]} | {row["status"]} | {row["tokens"]:,} | '
                f'{fmt(row.get("recent_tokens_per_second"))} | {fmt(row.get("best_validation_loss"))} | '
                f'{fmt(row.get("final_validation_loss"))} |')
    failed = [r for r in data['runs'] if r['status'] == 'failed']
    if failed or data['status'].get('error') or data['status'].get('failure'):
        lines += ['', '## Jobs needing attention', '']
        lines += [f'- L{r["layers"]}: {r["variant"]}; log `{r.get("log", "unavailable")}`' for r in failed]
        if data['status'].get('error'):
            lines += [data['status']['error']]
        if data['status'].get('failure'):
            lines += [str(data['status']['failure'])]
    text = '\n'.join(lines)+'\n'
    atomic_text(live/'report.md', text)
    if draw:
        plots(live, data)
    page = '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Quick depth sweep</title></head><body style="font-family:system-ui">'
    page += '<p><a href="report.md">Report</a> · <a href="depth_runs.csv">Runs CSV</a> · <a href="validations.csv">Validation CSV</a></p>'
    for name in ('validation_progress', 'best_validation_vs_layers', 'final_validation_vs_layers'):
        if (live/f'{name}.png').exists():
            page += f'<img style="max-width:100%" src="{name}.png?t={int(data["updated_at"])}" alt="{name}">'
    page += '<pre>'+html.escape(text)+'</pre></body></html>'
    atomic_text(live/'index.html', page)
    return data


def watch(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.quick-report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stopped.set())
        last_plot = 0
        while not stopped.is_set():
            draw = time.monotonic()-last_plot >= 60
            refresh(root, draw)
            if draw:
                last_plot = time.monotonic()
            stopped.wait(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    (watch if a.watch else refresh)(a.root)
