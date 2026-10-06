from pathlib import Path
import json
import csv
import numpy as np


def plots(inputs,output,metric="value_exact_match"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for f in inputs:
        for s in Path(f).read_text().splitlines():
            r=json.loads(s)
            if "exact_match" in r:
                r["series"] = r.get("model_id","legacy").split('/')[-1] + "/" + r["method"]
                if r.get("memory_disabled",False): r["series"] += "/memory-off"
                rows.append(r)
    if not rows: raise ValueError("No scored synthetic examples")
    # Raw machine-readable data accompany every figure; grouping remains explicit.
    if any(metric not in r for r in rows): raise ValueError(f"Missing {metric}; choose an available metric explicitly")
    columns=["series","model_id","model_revision","run_id","method","training_seed","task","total_length","distance","fact_count",
             "position_fraction","all_evidence_evicted","exact_match","value_exact_match","elapsed_seconds",
             "peak_allocated_bytes","state_bytes"]
    with (out/"plot_data.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r.get(k) for k in columns} for r in rows)
    for axis,label in [("total_length","Sequence length (tokens)"),("fact_count","Independent facts"),
                       ("distance","Distance from evidence (tokens)")]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        plotted=[]
        for method in sorted({r["series"] for r in rows}):
            selected=[r for r in rows if r["series"]==method]
            xs=sorted({r[axis] for r in selected})
            ys=[np.mean([r[metric] for r in selected if r[axis]==x]) for x in xs]
            ax.plot(xs,ys,"o-",label=method)
            plotted.extend({"method":method,"metric":metric,"x":x,"accuracy":float(y)} for x,y in zip(xs,ys))
        ax.set(xlabel=label,ylabel="Code-value accuracy" if metric=="value_exact_match" else "Strict exact-match accuracy",ylim=(-.03,1.03))
        if axis!="distance": ax.set_xscale("log",base=2)
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        for ext in ["png","pdf"]:fig.savefig(out/f"accuracy-{axis}.{ext}",dpi=160)
        (out/f"accuracy-{axis}.json").write_text(json.dumps(plotted,indent=2)+"\n")
        plt.close(fig)
    # Pair by content hash, task and settings; bootstrap examples within each training-seed pair.
    comparisons=[]
    methods=sorted({r["method"] for r in rows})
    lookup={m:{(r.get("model_id"),r.get("model_revision"),r.get("training_seed"),r["task"],r["prompt_sha256"]):r for r in rows
               if r["method"]==m and not r.get("memory_disabled",False)} for m in methods}
    rng=np.random.default_rng(318)
    if "slerp" in lookup:
        for other in methods:
            if other=="slerp":continue
            keys=set(lookup["slerp"]) & set(lookup[other])
            for model_id in sorted({r.get("model_id","legacy") for r in rows}):
             for length in sorted({r["total_length"] for r in rows}):
                selected=[k for k in keys if k[0]==model_id and lookup["slerp"][k]["total_length"]==length
                          and lookup["slerp"][k]["all_evidence_evicted"]]
                if not selected:continue
                diff=np.array([lookup["slerp"][k][metric]-lookup[other][k][metric] for k in sorted(selected)])
                boot=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)])
                comparisons.append({"comparison":f"slerp-minus-{other}","metric":metric,"model_id":model_id,"length":length,"n":len(diff),
                    "training_seeds":sorted({k[2] for k in selected}),
                    "ci_scope":"paired examples pooled within supplied seeds; not a between-training-seed confidence interval",
                    "difference":float(diff.mean()),"ci_low":float(np.quantile(boot,.025)),
                    "ci_high":float(np.quantile(boot,.975))})
    (out/"paired-comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")
    return out
