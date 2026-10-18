"""Export per-head measurements and per-checkpoint diagnostic figures."""
from collections import defaultdict
import html
import hashlib
import os
from pathlib import Path
import statistics
from context_sweep.recipe import read, write


def export(root):
    from context_sweep.report import atomic_csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);live=root/'live';folder=live/'diagnostics';folder.mkdir(parents=True,exist_ok=True)
    state=read(root/'diagnostics_state.json',{'runs':{}})
    tables={name:[] for name in ('attention','loss_by_position','branch_norms')}
    listing=['<!doctype html><meta charset="utf-8"><title>Attention diagnostics</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto}img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Checkpoint diagnostics</h1><p><a href="../index.html">Training dashboard</a></p>',
        '<p>Read-only sampled-row reconstructions in FP32. Timings include probes and are not throughput benchmarks. Each graph identifies its training budget.</p>']
    for name,job in sorted(state['runs'].items()):
        listing.append(f'<h2>{html.escape(name)} · {html.escape(job["status"])}</h2>')
        if job['status']!='complete':
            listing.append('<pre>'+html.escape(job.get('error',job.get('reason','Probe interrupted or running; rerun diagnose to recover.')))+'<br>'+html.escape(job.get('log',''))+'</pre>');continue
        payload=read(root/job['result'],{})
        if not payload:
            listing.append('<p>Saved diagnostic file missing; run diagnose again.</p>');continue
        stamp=dict(run=name,training_tokens=payload['training_tokens'],training_context=payload['training_context'],
            variant=payload['variant'],output_norm=payload['output_norm'])
        for evaluation in payload['evaluations']:
            for key in tables:
                tables[key].extend(dict(stamp,evaluation_context=evaluation['context'],**row) for row in evaluation[key])
        listing.append(f'<p>{payload["training_tokens"]:,} training tokens. <a href="{html.escape(name)}.pdf">PDF</a></p><img src="{html.escape(name)}.png">')
        cache_key=hashlib.sha256((root/job['result']).read_bytes()+Path(__file__).read_bytes()).hexdigest()
        cache=folder/f'{name}.stamp.json'
        if read(cache,{})=={'sha256':cache_key} and all((folder/f'{name}.{ext}').is_file() for ext in ('png','pdf')):
            continue
        fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for evaluation in payload['evaluations']:
            context=evaluation['context'];label=f'eval {context}'
            attn=evaluation['attention'];positions=sorted({r['visible_keys'] for r in attn})
            for ax,metric,title in ((axs[0,0],'row_mass','Attention weight sum'),(axs[0,1],'effective_key_count','Effective number of keys')):
                vals=[statistics.mean(r[metric] for r in attn if r['visible_keys']==n) for n in positions]
                ax.plot(positions,vals,marker='.',label=label);ax.set_title(title+' (mean over layers/heads)')
            losses=evaluation['loss_by_position']
            axs[1,0].plot([r['last_position'] for r in losses],[r['loss'] for r in losses],marker='.',label=label)
            norms=evaluation['branch_norms']
            for branch,style in (('attn_output_norm','-'),('ffn_output_norm','--')):
                ns=[r for r in norms if r['branch']==branch];ends=sorted({r['last_position'] for r in ns})
                axs[1,1].plot(ends,[statistics.mean(r['mean_input_rms'] for r in ns if r['last_position']==n) for n in ends],
                    linestyle=style,marker='.',label=f'{context} '+('attn' if branch.startswith('attn') else 'FFN'))
        axs[1,0].set_title('Loss by position bin (nats/token)')
        axs[1,0].ticklabel_format(axis='y',useOffset=False,style='plain')
        axs[1,1].set_title('Branch RMS before optional output norm')
        for ax in axs.flat:
            ax.set_xlabel('Causal token position');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f"{name}\n{payload['training_tokens']:,} training tokens · diagnostic sample, not a performance benchmark",fontsize=10)
        for ext in ('png','pdf'):
            dest=folder/f'{name}.{ext}';temp=folder/f'{name}.{os.getpid()}.tmp.{ext}'
            fig.savefig(temp,dpi=150);os.replace(temp,dest)
        plt.close(fig)
        write(cache,{'sha256':cache_key})
    for name,rows in tables.items():atomic_csv(live/f'diagnostic_{name}.csv',rows)
    tmp=folder/f'index.{os.getpid()}.tmp';tmp.write_text('\n'.join(listing));os.replace(tmp,folder/'index.html')
