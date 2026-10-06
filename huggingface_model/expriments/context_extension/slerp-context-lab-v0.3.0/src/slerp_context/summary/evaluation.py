from dataclasses import asdict
from pathlib import Path
import gc
import json
import platform
import time
import torch
from ..data import tokenizer_for
from ..storage import append_json,disk_guard,versions
from .data import load_split,prompt_parts,encode
from .models import device_for,resolve_cfg,construct,restore,weight_identity
from .inference import Engine,run_document,BASELINES,LEARNED,sync
from .config import digest,file_digest
from .metrics import score
from .telemetry import RSSMonitor


def cpu_name():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):return line.split(':',1)[1].strip()
    except OSError:pass
    return platform.processor() or platform.machine()


def evaluate(study,method,output,split="validation",checkpoint=None,device=None,
             limit=None,min_source_tokens=0,max_source_tokens=None):
    if method not in BASELINES|LEARNED:raise ValueError("Unknown method")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    meta_path=output.with_suffix(".meta.json")
    if output.exists() or meta_path.exists():raise FileExistsError("Choose a fresh evaluation path")
    rows,meta=load_split(study,split)
    rows=[r for r in rows if min_source_tokens<=r["source_tokens"] and
          (max_source_tokens is None or r["source_tokens"]<=max_source_tokens)]
    if limit is not None:rows=rows[:limit]
    if not rows:raise ValueError("No whole documents selected")
    device=device_for(device);cfg=resolve_cfg(study,meta,method if method in LEARNED else "native")
    load_start=time.perf_counter()
    if checkpoint:
        model,path=restore(checkpoint,device)
        if (model.cfg.model_id,model.cfg.revision)!=(cfg.model_id,cfg.revision):
            raise ValueError("Checkpoint and dataset tokenizer differ")
        if method in LEARNED:
            for k in ["method","window","chunk","slots"]:
                if getattr(model.cfg,k)!=getattr(cfg,k):raise ValueError(f"Study/checkpoint mismatch: {k}")
    else:
        if method in LEARNED-{"local"}:
            raise ValueError("Learned memory requires a trained checkpoint; use train/profile first")
        model=construct(cfg,device);path=None
    dtype=getattr(torch,study.eval_weight_dtype)
    if device.type!="cuda" and dtype!=torch.float32:raise ValueError("Use float32 for CPU evaluation")
    # Keep geometry/memory parameters in FP32; every arm uses the same selected
    # backbone weight dtype and BF16 matmul autocast on CUDA.
    model.backbone.to(dtype=dtype);tokenizer=tokenizer_for(model.cfg,path)
    engine=Engine(model,tokenizer,study);sync(device)
    load_seconds=time.perf_counter()-load_start
    identity=weight_identity(model,checkpoint)
    overhead=sum(map(len,prompt_parts(tokenizer,study)))
    info={"format_version":"0.3.0","status":"in_progress","method":method,"split":split,
        "study":asdict(study),"identity":identity,"data_manifest":meta,
        "source_hashes":[r["source_sha256"] for r in rows],
        "native_context":model.native_context,"working_context":cfg.window,
        "weight_dtype":study.eval_weight_dtype,"compute_dtype":"bfloat16_autocast" if device.type=="cuda" else "float32",
        "device":str(device),"device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else cpu_name(),
        "model_load_seconds_excluded":load_seconds,"versions":versions(),
        "timing_scope":"after prepared source token IDs are loaded; includes all prompt construction, intermediate generation, and final generation",
        "rouge_recipe":"rouge-score 0.1.2; Porter stemming; deterministic punctuation/newline sentence split",
        "rss_scope":"sampled 20ms process RSS; includes weights; CUDA allocator reported separately",
        "cache_scope":"largest returned persistent cache/state across calls; not transient workspaces or activation peaks",
        "energy_joules":None,"energy_note":"not measured; use external/Jetson power instrumentation for energy claims"}
    info["run_id"]=digest({k:info[k] for k in ["method","study","identity","source_hashes","device","device_name","versions"]})[:16]
    meta_path.write_text(json.dumps(info,indent=2)+"\n")
    # Untimed fixed warmup. No reference text is provided to any generator.
    engine.generate(encode(tokenizer,"Summarize: A pilot study evaluated a proposed method."),4,recurrent=method in LEARNED)
    sync(device);gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()
    counts={};output.touch(exist_ok=False)
    for row in rows:
        disk_guard(cfg)
        if device.type=="cuda":
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        with RSSMonitor() as rss:
            try:
                result=run_document(engine,row,method)
            except torch.OutOfMemoryError as exc:
                result={"status":"oom","prediction":None,"error":str(exc)[:500]}
                gc.collect()
                if device.type=="cuda":torch.cuda.empty_cache()
        metrics=score(result["prediction"],row["reference"]) if result["status"]=="ok" else {}
        output_row={"run_id":info["run_id"],"method":method,"id":row["id"],"split":split,
            "model_id":cfg.model_id,"model_revision":cfg.revision,"checkpoint_sha256":identity["checkpoint_sha256"],
            "source_sha256":row["source_sha256"],"reference_sha256":row["reference_sha256"],
            "dataset_revision":meta["dataset_revision"],"length_bin":row["length_bin"],
            "source_tokens":row["source_tokens"],"reference_tokens":row["reference_tokens"],
            "source_over_native":row["source_tokens"]>model.native_context,
            "full_request_over_native":row["source_tokens"]+overhead+study.max_new_tokens>model.native_context,
            "source_over_working":row["source_tokens"]>cfg.window,
            "native_context":model.native_context,"working_context":cfg.window,
            "max_new_tokens":study.max_new_tokens,"prompt_sha256":digest(study.prompt),
            "weight_dtype":study.eval_weight_dtype,"compute_dtype":info["compute_dtype"],
            "device_name":info["device_name"],"runtime_sha256":digest(info["versions"]),
            "reference":row["reference"],"rss_peak_bytes":rss.peak,
            "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            "gpu_peak_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None,
            **result,**metrics}
        append_json(output,output_row);counts[result["status"]]=counts.get(result["status"],0)+1
        print(json.dumps({k:output_row.get(k) for k in ["id","method","status","source_tokens","rougeLsum","end_to_end_s"]}),flush=True)
    info.update(status="complete",counts=counts,rows=len(rows),results_sha256=file_digest(output))
    meta_path.write_text(json.dumps(info,indent=2)+"\n")
    return info
