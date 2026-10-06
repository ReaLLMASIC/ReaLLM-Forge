"""Raw tables, paired document bootstrap, plots, and blind human-review sheets."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import random
import statistics
import numpy as np
from .config import file_digest,digest


def completed_rows(files):
    result=[];seen=set();metas={}
    for file in files:
        p=Path(file);meta=json.loads(p.with_suffix(".meta.json").read_text())
        if meta.get("status")!="complete" or meta.get("results_sha256")!=file_digest(p):
            raise ValueError(f"Incomplete or modified evaluation: {p}")
        if meta["run_id"] in seen:raise ValueError("Duplicate evaluation run")
        seen.add(meta["run_id"]);metas[meta["run_id"]]=meta
        result.extend(json.loads(line) for line in p.open())
    return result,metas


def paired_difference(a,b,metric,samples=2000,seed=17):
    def key(r):
        base=(r["source_sha256"],r["reference_sha256"],r["model_id"],r["model_revision"],
              r["dataset_revision"],r["max_new_tokens"],r["prompt_sha256"])
        if metric.endswith('_s'):
            base+=tuple(r.get(k) for k in ['device_name','runtime_sha256','weight_dtype','compute_dtype'])
        return base
    aa={key(r):r for r in a if r["status"]=="ok" and r.get(metric) is not None}
    bb={key(r):r for r in b if r["status"]=="ok" and r.get(metric) is not None}
    keys=sorted(aa.keys()&bb.keys())
    diffs=np.array([bb[k][metric]-aa[k][metric] for k in keys],dtype=float)
    if not len(diffs):return {"paired_documents":0,"difference":None,"ci95":None}
    rng=np.random.default_rng(seed)
    draws=np.array([rng.choice(diffs,len(diffs),replace=True).mean() for _ in range(samples)])
    return {"paired_documents":len(diffs),"difference":float(diffs.mean()),
        "ci95":np.quantile(draws,[0.025,0.975]).tolist() if len(diffs)>1 else None,
        "interval_scope":"paired documents in selected set; not training-seed uncertainty"}


def write_csv(path,rows,fields=None):
    if fields is None:fields=list(rows[0]) if rows else ["empty"]
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def report(files,out,baseline="rolling",semantic_file=None):
    rows,metas=completed_rows(files);out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError("Choose a new report directory")
    out.mkdir(parents=True,exist_ok=True)
    if semantic_file:
        smeta=json.loads(Path(semantic_file).with_suffix('.meta.json').read_text())
        if smeta.get('status')!='complete' or smeta.get('sha256')!=file_digest(semantic_file):
            raise ValueError('Incomplete or modified semantic scores')
        scores={(r["run_id"],r["source_sha256"]):r for r in map(json.loads,Path(semantic_file).read_text().splitlines())}
        for r in rows:
            s=scores.get((r["run_id"],r["source_sha256"]))
            if s:
                if s['prediction_sha256']!=digest(r['prediction']) or s['reference_sha256']!=r['reference_sha256']:
                    raise ValueError('Semantic scores belong to different text')
                for k in ["semantic_precision","semantic_recall","semantic_f1"]:r[k]=s[k]
    grouped=defaultdict(list)
    for r in rows:grouped[r["run_id"]].append(r)
    metrics=["rouge1","rouge2","rougeL","rougeLsum","semantic_f1","end_to_end_s","final_first_token_s",
             "rss_peak_bytes","gpu_peak_allocated_bytes","gpu_peak_reserved_bytes","max_call_cache_bytes",
             "all_generation_tokens","all_call_input_tokens","final_generation_tokens","prediction_words",
             "generation_limit_hit","repeated_trigram_fraction"]
    aggregates=[]
    for run,rr in grouped.items():
        for scope,selection in [("all",rr),("source_over_native",[r for r in rr if r["source_over_native"]])]:
            good=[r for r in selection if r["status"]=="ok"]
            item={"run_id":run,"method":rr[0]["method"],"model_id":rr[0]["model_id"],"scope":scope,
                  "n_requested":len(selection),"n_ok":len(good),"n_oom":sum(r["status"]=="oom" for r in selection),
                  "n_skipped":sum(r["status"].startswith("skipped") for r in selection)}
            for m in metrics:
                values=[r[m] for r in good if r.get(m) is not None]
                item[m]=statistics.mean(values) if values else None
            aggregates.append(item)
    write_csv(out/"summary.csv",aggregates)
    raw_fields=["run_id","method","id","source_tokens","reference_tokens","source_over_native","status"]+metrics
    write_csv(out/"per-document.csv",rows,raw_fields)
    comparisons=[]
    for left,a in grouped.items():
        if a[0]["method"]!=baseline:continue
        for right,b in grouped.items():
            if left==right:continue
            for scope in ["all","source_over_native"]:
                aa=a if scope=="all" else [r for r in a if r["source_over_native"]]
                bb=b if scope=="all" else [r for r in b if r["source_over_native"]]
                for metric in ["rougeLsum","semantic_f1","end_to_end_s"]:
                    comparisons.append({"baseline":left,"candidate":right,"scope":scope,"metric":metric,
                        **paired_difference(aa,bb,metric)})
    (out/"paired-comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")
    # Blind summaries for human scoring; source/reference mapping is separate.
    blinded=[];key=[]
    good=[r for r in rows if r["status"]=="ok"];random.Random(17).shuffle(good)
    for index,r in enumerate(good):
        code=f"summary-{index:05d}"
        blinded.append({"blind_id":code,"source_sha256":r["source_sha256"],"prediction":r["prediction"],
            "coverage_1_to_5":"","faithfulness_1_to_5":"","coherence_1_to_5":"","notes":""})
        key.append({"blind_id":code,"run_id":r["run_id"],"method":r["method"],"id":r["id"]})
    write_csv(out/"human-review.csv",blinded);write_csv(out/"human-review-key.csv",key)
    plot(rows,metas,out)
    (out/"report.json").write_text(json.dumps({"runs":metas,"aggregate":aggregates,
        "score_note":"ROUGE overlap and optional chunked semantic token matching are not factuality scores.",
        "pairing_note":"same whole source/reference/model revision/prompt/output budget; training histories may differ and remain in metadata",
        "sampling_note":"deterministic first matching documents per length bin; CI is conditional on this selected set"},indent=2)+"\n")
    return aggregates


def plot(rows,metas,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for r in rows:
        if r["status"]=="ok":groups[r["run_id"]].append(r)
    specs=[("source_tokens","rougeLsum","quality-vs-length","Source tokens","ROUGE-Lsum F1 (%)"),
           ("source_tokens","end_to_end_s","latency-vs-length","Source tokens","Total summary latency (s)"),
           ("end_to_end_s","rougeLsum","quality-vs-latency","Total summary latency (s)","ROUGE-Lsum F1 (%)"),
           ("source_tokens","max_call_cache_bytes","state-vs-length","Source tokens","Largest call cache/state (MiB)")]
    if any(r.get("gpu_peak_allocated_bytes") for r in rows):
        specs.append(("gpu_peak_allocated_bytes","rougeLsum","quality-vs-gpu-memory","Peak allocated GPU memory (GiB)","ROUGE-Lsum F1 (%)"))
    for x,y,name,xlabel,ylabel in specs:
        fig,ax=plt.subplots(figsize=(9.2,4.6));raw=[]
        markers=['o','s','^','D','v','P','X','>','<','h','*']
        for series,(run,rr) in enumerate(groups.items()):
            bins=defaultdict(list)
            for r in rr:
                if r.get(x) is not None and r.get(y) is not None:bins[r["length_bin"]].append(r)
            points=[]
            for b,items in sorted(bins.items()):
                xx=statistics.mean(r[x] for r in items);yy=statistics.mean(r[y] for r in items)
                if x=="gpu_peak_allocated_bytes":xx/=2**30
                if y=="max_call_cache_bytes":yy/=2**20
                points.append((xx,yy));raw.append({"run_id":run,"method":rr[0]["method"],"bin":b,"n":len(items),"x":xx,"y":yy})
            if points:
                ax.plot(*zip(*points),marker=markers[series%len(markers)],
                        color=plt.get_cmap('tab20')(series%20),label=f"{rr[0]['method']} [{run[:6]}]")
        ax.set(xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.25)
        if groups:ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1))
        if y=='rougeLsum':ax.set_ylim(0,max(1,ax.get_ylim()[1]))
        if x=="source_tokens":
            for n in sorted({m["native_context"] for m in metas.values()}):ax.axvline(n,color="gray",ls=":",alpha=.5)
        fig.tight_layout();fig.savefig(out/(name+".png"),dpi=180);fig.savefig(out/(name+".pdf"));plt.close(fig)
        (out/(name+".json")).write_text(json.dumps(raw,indent=2)+"\n")
