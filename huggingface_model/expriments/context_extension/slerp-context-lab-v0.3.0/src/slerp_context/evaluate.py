from pathlib import Path
from contextlib import nullcontext
import csv
import json
import math
import re
import time
import numpy as np
import torch
from .config import Config
from .model import RecurrentLM
from .data import tokenizer_for, synthetic, TokenDocuments
from .storage import append_json, checkpoint_path, load_weights, disk_guard


def load_run(path, device=None):
    path=checkpoint_path(path)
    cfg=Config.load(path/"config.json")
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model=RecurrentLM.from_pretrained(path,device);model.eval()
    return model,tokenizer_for(cfg,path)


def autocast(model):
    device=next(model.parameters()).device
    return torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else nullcontext()


def stop_ids(model, tokenizer):
    eos=getattr(model.backbone.generation_config,"eos_token_id",None)
    return tokenizer.eos_token_id if eos is None else eos


def normalize_answer(text):
    text=text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"\s*,\s*",",",text.strip())


def exact(pred,target):
    return float(normalize_answer(pred)==normalize_answer(target))


def value_exact(pred, target):
    """Target-independent extraction of ALL six-digit codes in the final answer.

    Strip a completed thinking prefix, then use a single explicit answer block if
    present. Extra, duplicated, or reordered codes fail. Never select a substring
    because it matches the reference answer. This metric is synthetic-task-only.
    """
    if "</think>" in pred: pred=pred.rsplit("</think>",1)[-1]
    elif "<think>" in pred: return 0.0
    answers=re.findall(r"<answer>(.*?)</answer>",pred,flags=re.S)
    if len(answers)>1: return 0.0
    if answers: pred=answers[0]
    pattern=r"(?<![A-Za-z0-9])\d{6}(?![A-Za-z0-9])"
    expected=re.findall(pattern,target)
    return float(bool(expected) and re.findall(pattern,pred)==expected)


@torch.no_grad()
def evaluate(run, output, lengths, samples=4, tasks=("recall","update","trace","multi"),
             facts=(1,8,32), positions=(0.05,0.5,0.9), seed=91991, device=None, disable_memory=False,
             config=None, max_new_tokens=128):
    if samples < 1 or max_new_tokens < 1: raise ValueError("Positive samples and generation limit required")
    if run:
        model,tok=load_run(run,device);cfg=model.cfg
    else:
        from .native import NativeLM
        cfg = Config.load(config) if not isinstance(config,Config) else config
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = NativeLM.build(cfg,dev) if cfg.method == "native" else RecurrentLM.build(cfg,dev).eval()
        tok = tokenizer_for(cfg)
    output=Path(output)
    if output.exists(): raise FileExistsError("Evaluation output already exists; choose a new filename")
    output.parent.mkdir(parents=True,exist_ok=True)
    dev=next(model.parameters()).device
    label = cfg.method if run or cfg.method == "native" else "pretrained-"+cfg.method
    metadata_path = output.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps({"status":"in_progress","model_id":cfg.model_id,"revision":cfg.revision,
        "method":label,"source":str(run or config),"lengths":list(lengths),"tasks":list(tasks),
        "facts":list(facts),"positions":list(positions),"seed":seed,"samples":samples,
        "max_new_tokens":max_new_tokens,"thinking":False,"chat_template":cfg.chat_template,
        "do_sample":False,"repetition_penalty":1.0,
        "metrics":["exact_match","value_exact_match"],
        "skipped_combinations":[{"task":"trace","facts":n,"reason":"two-hop needs two records"}
            for n in facts if n<2 and "trace" in tasks]},indent=2)+"\n")
    for length in lengths:
      for task in tasks:
       for fact_count in facts:
        if task == "trace" and fact_count < 2: continue
        for position in positions:
         for i in range(samples):
            disk_guard(cfg)
            ep=synthetic(tok,length,seed,i,"test",task,fact_count,position,cfg.window,cfg.chunk,cfg.chat_template)
            prompt=torch.tensor([ep.prompt],device=dev)
            if dev.type=="cuda": torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            start=time.monotonic()
            with autocast(model):
                generated,state=model.generate_stream(prompt,max_new_tokens,
                    eos_token_id=stop_ids(model,tok),memory_enabled=not disable_memory)
            if dev.type=="cuda": torch.cuda.synchronize()
            text=tok.decode(generated[0],skip_special_tokens=True)
            row={**ep.metadata,"method":label,"memory_disabled":disable_memory,
                "model_id":cfg.model_id,"model_revision":cfg.revision,
                "training_mode":"full" if run else "pretrained", "run_id":str(run or "pretrained"),
                "training_seed":cfg.seed,"training_length_limit":max(cfg.lengths) if run else None,
                "native_context":model.native_context,"output":text,"target":ep.target,
                "exact_match":exact(text,ep.target),"value_exact_match":value_exact(text,ep.target),
                "generated_tokens":generated.numel(),
                "generation_limit_hit":generated.numel() == max_new_tokens,
                "elapsed_seconds":time.monotonic()-start,"state_bytes":state.nbytes(),
                "peak_allocated_bytes":torch.cuda.max_memory_allocated() if dev.type=="cuda" else 0,
                "peak_reserved_bytes":torch.cuda.max_memory_reserved() if dev.type=="cuda" else 0}
            append_json(output,row)
         print(json.dumps({"length":length,"task":task,"facts":fact_count,"position":position,"done":samples}),flush=True)
    summary = summarize(output)
    meta = json.loads(metadata_path.read_text())
    meta.update(status="complete",completed_examples=sum(r["n"] for r in summary))
    metadata_path.write_text(json.dumps(meta,indent=2)+"\n")
    return summary


def assess(inputs, output, threshold=None, metric="value_exact_match"):
    """Screen candidates and gate runs using their actual generated answers."""
    groups = {}
    for filename in inputs:
        meta = Path(filename).with_suffix(".meta.json")
        if meta.exists() and json.loads(meta.read_text()).get("status") != "complete":
            raise ValueError(f"Incomplete evaluation: {filename}. Use a new output filename for a rerun.")
        for line in Path(filename).read_text().splitlines():
            row = json.loads(line)
            if "exact_match" not in row: continue
            key = (row.get("model_id","unknown"), row["method"], row.get("run_id","unknown"), row["total_length"])
            groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        evicted = [r for r in rows if r["all_evidence_evicted"]]
        score = sum(r[metric] for r in evicted)/len(evicted) if evicted else None
        summary.append(dict(zip(["model_id","method","run_id","length"],key),
            n=len(rows),accuracy=sum(r[metric] for r in rows)/len(rows),
            strict_accuracy=sum(r["exact_match"] for r in rows)/len(rows),
            evicted_n=len(evicted),evicted_accuracy=score,
            passed=(score >= threshold) if threshold is not None and score is not None else None))
    if not summary: raise ValueError("No scored results")
    result={"threshold":threshold,"metric":metric,"groups":summary,
        "note":"These are package synthetic tasks, not official RULER/LongBench scores."}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if threshold is not None and any(g["passed"] is not True for g in summary):
        raise RuntimeError("Learning gate failed or lacked evicted examples; inspect the report before larger training")
    return result


def summarize(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()]
    groups={}
    for r in rows:
        key=(r.get("method"),r["task"],r["total_length"],r["fact_count"],r["position_fraction"],r.get("memory_disabled",False))
        groups.setdefault(key,[]).append(r)
    summary=[]
    rng=np.random.default_rng(1)
    for key,rs in sorted(groups.items()):
        values=np.array([r["exact_match"] for r in rs])
        boot=rng.choice(values,(2000,len(values)),replace=True).mean(1)
        ev=[r["exact_match"] for r in rs if r["all_evidence_evicted"]]
        value_scores=[r["value_exact_match"] for r in rs]
        value_ev=[r["value_exact_match"] for r in rs if r["all_evidence_evicted"]]
        summary.append(dict(zip(["method","task","total_length","fact_count","position_fraction","memory_disabled"],key),
            n=len(rs),exact_match=float(values.mean()),ci_low=float(np.quantile(boot,.025)),
            ci_high=float(np.quantile(boot,.975)),evicted_n=len(ev),
            evicted_exact_match=sum(ev)/len(ev) if ev else None,
            value_exact_match=sum(value_scores)/len(value_scores),
            evicted_value_exact_match=sum(value_ev)/len(value_ev) if value_ev else None))
    out=Path(path).with_suffix(".summary.csv")
    if summary:
        with out.open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    return summary


@torch.no_grad()
def perplexity(run,data_dir,output,lengths,suffix=512,samples=8,split="validation",device=None):
    model,tok=load_run(run,device);dev=next(model.parameters()).device
    docs=TokenDocuments(data_dir,split)
    docs.verify_tokenizer(model.cfg)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    maxlen=max(lengths)
    if min(lengths)<=suffix: raise ValueError("All lengths must exceed scored suffix")
    for i in range(samples):
        full=docs.episode(maxlen,7219,i)
        ids=full.prompt+full.answer
        for length in lengths:
            window=ids[-length:]
            inp=torch.tensor([window[:-1]],device=dev)
            labels=torch.tensor([window[1:]],device=dev)
            labels[:,:-suffix]=-100
            with autocast(model): nll,state=model.episode_loss(inp,labels)
            value=float(nll)
            append_json(out,{"example_id":i,"book_id":full.metadata["book_id"],"total_length":length,
                "suffix_tokens":suffix,"nll":value,"perplexity":math.exp(min(value,80)),
                "state_bytes":state.nbytes(),"method":model.cfg.method,"token_level":True})


@torch.no_grad()
def generated_history(run,output,samples=8,gap=2304,device=None):
    """Free-running diagnostic; fact must actually be emitted, then leave local memory."""
    model,tok=load_run(run,device);dev=next(model.parameters()).device
    if gap<=model.cfg.window: raise ValueError("Gap must exceed local window")
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    torch.manual_seed(812)
    tensor=lambda text:torch.tensor([tok.encode(text,add_special_tokens=False)],device=dev)
    for i in range(samples):
        with autocast(model):
            first,st=model.generate_stream(tensor("Choose a six digit code. Write CODE: followed by the code.\nCODE:"),32,temperature=1)
            text=tok.decode(first[0],skip_special_tokens=True)
            match=re.search(r"(?<!\d)\d{6}(?!\d)",text)
            if match is None:
                append_json(out,{"example_id":i,"valid_fact":False,"early_output":text});continue
            value=match.group()
            # No EOS stopping: exactly gap free-running tokens, same recurrent state.
            continuation,st=model.generate_stream(tensor("\nContinue writing a story:\n"),gap,
                temperature=1,state=st)
            recent=tok.decode(continuation[0,-model.cfg.window:],skip_special_tokens=True)
            all_middle=tok.decode(continuation[0],skip_special_tokens=True)
            answer,st=model.generate_stream(tensor("\nWhat was the original CODE? Return its six digits only.\nAnswer: "),24,state=st)
        pred=tok.decode(answer[0],skip_special_tokens=True)
        append_json(out,{"example_id":i,"valid_fact":True,"target":value,"output":pred,
            "consistency_exact_match":exact(pred,value),"task_target_exact_match":None,
            "recent_repeat":value in recent,"any_continuation_repeat":value in all_middle,
            "primary_eligible":value not in all_middle,"gap_tokens":gap,
            "early_output":text,"continuation":all_middle,"state_bytes":st.nbytes()})


@torch.no_grad()
def predict_jsonl(run,input_path,output,device=None,config=None,chat_template=False,max_new_tokens=128):
    """Bridge for official RULER records. External official scoring remains authoritative."""
    if run:
        model,tok=load_run(run,device)
    else:
        from .native import NativeLM
        cfg=Config.load(config)
        dev=device or ("cuda" if torch.cuda.is_available() else "cpu")
        model=NativeLM.build(cfg,dev) if cfg.method=="native" else RecurrentLM.build(cfg,dev).eval()
        tok=tokenizer_for(cfg)
    dev=next(model.parameters()).device
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    for line in Path(input_path).open():
        record=json.loads(line)
        text=record.get("input",record.get("prompt"))
        if not isinstance(text,str): raise ValueError("Expected input/prompt string")
        ids=tok.encode(text,add_special_tokens=False)
        if chat_template:
            from .data import chat_parts
            before,after=chat_parts(tok,True)
            ids=before+ids+after
        if not ids: raise ValueError("Empty prompt")
        with autocast(model):
            answer,st=model.generate_stream(torch.tensor([ids],device=dev),max_new_tokens,eos_token_id=stop_ids(model,tok))
        record.update(pred=tok.decode(answer[0],skip_special_tokens=True),input_tokens=len(ids),
                      actual_consumed_tokens=st.consumed,model_id=model.cfg.model_id,
                      model_revision=model.cfg.revision,max_new_tokens=max_new_tokens,chat_template=chat_template)
        append_json(out,record)
