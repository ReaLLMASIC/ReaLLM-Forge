"""Pinned HF streaming data; whole documents, whole reference summaries.

Length stratification selects rather than truncates source documents. Only
accepted rows are persisted. Each split has an atomic completion manifest.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
from ..data import tokenizer_for, chat_parts, Episode,ByteTokenizer
from ..storage import disk_guard
from .config import digest, file_digest


def encode(tokenizer,text):
    # Tokenization intentionally handles whole texts longer than the model input.
    kw={} if isinstance(tokenizer,ByteTokenizer) else {"truncation":False,"verbose":False}
    return tokenizer.encode(text,add_special_tokens=False,**kw)


def prompt_parts(tokenizer,study,instruction=None):
    before,after=chat_parts(tokenizer,study.base().chat_template)
    prefix=before+encode(tokenizer,(instruction or study.prompt)+"\n\nDocument:\n")
    suffix=encode(tokenizer,"\n\nSummary:\n")+after
    return prefix,suffix


def prompt_ids(tokenizer,study,source_ids,instruction=None):
    a,b=prompt_parts(tokenizer,study,instruction)
    return a+list(source_ids)+b


def bin_index(n,edges):
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        if lo <= n < hi:return i
    return None


def source_hash(text): return hashlib.sha256(text.encode()).hexdigest()


def normalize_row(row,study,tokenizer,index,split):
    source,target=row.get(study.source_field),row.get(study.target_field)
    if not isinstance(source,str) or not isinstance(target,str):
        raise ValueError("Dataset source and summary columns must be strings")
    source,target=source.strip(),target.strip()
    # Reject overlarge rows rather than truncating an input paired with a full summary.
    if not source or not target or len(source)>2_000_000 or len(target)>100_000:return None
    ids=encode(tokenizer,source); b=bin_index(len(ids),study.length_edges)
    if b is None:return None
    ref_ids=encode(tokenizer,target)
    return {"id":f"{split}-{index}-{source_hash(source)[:16]}","split":split,
        "source_index":index,"source_id":str(row.get("id",index)),
        "source_sha256":source_hash(source),"reference_sha256":source_hash(target),
        "source":source,"reference":target,"source_ids":ids,"reference_ids":ref_ids,
        "source_tokens":len(ids),"reference_tokens":len(ref_ids),
        "length_bin":b,"source_truncated":False,"reference_truncated":False}


def prepare_split(study,tokenizer,split,rows,provenance):
    out=Path(study.data_dir);out.mkdir(parents=True,exist_ok=True)
    dest=out/f"{split}.jsonl"; manifest=out/f"{split}.manifest.json"
    if dest.exists() or manifest.exists() or dest.with_suffix(".pending").exists():
        raise FileExistsError(f"Refusing to overwrite prepared split: {dest}")
    wanted=getattr(study,f"{split}_per_bin")
    counts=[0]*(len(study.length_edges)-1); seen=set(); other=set()
    for p in out.glob("*.jsonl"):
        if not p.with_suffix(".manifest.json").exists():
            raise ValueError(f"Incomplete prepared split {p}; inspect it before continuing")
        for line in p.open():other.add(json.loads(line)["source_sha256"])
    skipped={"length_or_empty":0,"duplicate":0,"full_bin":0}
    pending=dest.with_suffix(".pending"); written=0; scanned=0
    existing=sum(p.stat().st_size for p in out.glob("*.jsonl"))
    with pending.open("x") as f:
        for index,raw in enumerate(rows):
            if index>=study.max_scan:break
            scanned+=1
            row=normalize_row(raw,study,tokenizer,index,split)
            if row is None:skipped["length_or_empty"]+=1;continue
            h=row["source_sha256"]
            if h in other or h in seen:skipped["duplicate"]+=1;continue
            b=row["length_bin"]
            if counts[b]>=wanted:skipped["full_bin"]+=1;continue
            line=json.dumps(row,ensure_ascii=False)+"\n";size=len(line.encode())
            if existing+written+size>study.max_data_mb*1_000_000:
                raise RuntimeError("Prepared-data byte cap reached; choose fewer samples")
            disk_guard(study.base(),size)
            f.write(line);written+=size;seen.add(h);counts[b]+=1
            print(json.dumps({"split":split,"accepted":sum(counts),"bin_counts":counts}),flush=True)
            if min(counts)>=wanted:break
        f.flush();os.fsync(f.fileno())
    if not sum(counts):raise RuntimeError("No whole documents meet the requested bounds")
    os.replace(pending,dest)
    meta={**provenance,"split":split,"counts":counts,"scanned":scanned,"skipped":skipped,
        "length_edges":study.length_edges,"requested_per_bin":wanted,
        "quota_complete":min(counts)>=wanted,"source_policy":"whole_document_selection",
        "sha256":file_digest(dest),"complete":True,"format_version":"0.3.0"}
    manifest.write_text(json.dumps(meta,indent=2)+"\n")
    return meta


def prepare_hf(study,splits):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api=HfApi();cfg=study.base()
    cfg.revision=api.model_info(cfg.model_id,revision=cfg.revision).sha
    ds_revision=api.dataset_info(study.dataset_id,revision=study.dataset_revision).sha
    tokenizer=tokenizer_for(cfg)
    provenance={"dataset":study.dataset_id,"dataset_config":study.dataset_config,
        "dataset_revision":ds_revision,"model_id":cfg.model_id,"model_revision":cfg.revision,
        "source_field":study.source_field,"target_field":study.target_field,
        "tokenizer_class":type(tokenizer).__name__}
    for split in splits:
        # Repository contains Parquet; no remote dataset script or corpus download.
        stream=load_dataset(study.dataset_id,name=study.dataset_config,revision=ds_revision,
                            split=split,streaming=True)
        prepare_split(study,tokenizer,split,stream,provenance)
    Path(study.data_dir,"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance


def load_split(study,split):
    path=Path(study.data_dir)/f"{split}.jsonl"
    meta=json.loads(path.with_suffix(".manifest.json").read_text())
    if not meta.get("complete") or file_digest(path)!=meta["sha256"]:
        raise ValueError("Prepared split incomplete or changed")
    if meta["model_id"]!=study.base().model_id:
        raise ValueError("Dataset tokenization belongs to a different model")
    if (meta["dataset"],meta["dataset_config"])!=(study.dataset_id,study.dataset_config):
        raise ValueError("Prepared corpus differs from study")
    if study.dataset_revision not in {"main",meta['dataset_revision']}:
        raise ValueError('Prepared dataset revision differs from study')
    rows=[json.loads(line) for line in path.open()]
    if len({r["source_sha256"] for r in rows})!=len(rows):raise ValueError("Duplicate sources")
    return rows,meta


def summarization_episode(row,tokenizer,study):
    prompt=prompt_ids(tokenizer,study,row["source_ids"])
    reference=row["reference_ids"]
    if len(reference)>study.train_target_limit:
        raise ValueError("Training target too long; select whole targets rather than truncating them")
    eos=tokenizer.eos_token_id
    answer=list(reference)+([] if eos is None else [eos])
    return Episode(prompt,answer,row["reference"],{"task":"summary","id":row["id"]})
