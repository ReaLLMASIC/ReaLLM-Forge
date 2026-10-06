from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import time
import pytest
import torch
from safetensors.torch import load_file
from slerp_context.data import ByteTokenizer
from slerp_context.summary.config import Study
from slerp_context.summary.data import (prepare_split,load_split,normalize_row,
    summarization_episode,prompt_ids,source_hash)
from slerp_context.summary.inference import run_document,Generated,select_source
from slerp_context.summary.training import train,memory_distillation
from slerp_context.summary.report import report,paired_difference
from slerp_context.summary.evaluation import evaluate
from slerp_context.storage import checkpoint_path


def study(tmp_path,method="slerp"):
    return Study(model=dict(tiny=True,tiny_arch="llama",model_id="tiny-llama",revision="fixture",
        method=method,window=512,chunk=64,slots=4,checkpoint=True,token_budget=500,
        tokens_per_update=200,save_every=1,min_free_gb=0,max_project_gb=100),
        data_dir=str(tmp_path/"data"),dataset_revision='fixture',length_edges=[16,256,1024,4096],
        train_per_bin=1,validation_per_bin=1,test_per_bin=1,
        prompt="Summarize the document.",max_new_tokens=8,intermediate_tokens=8,
        train_target_limit=128,max_epochs=4)


def prepare(s,split):
    tok=ByteTokenizer()
    rows=[{"report":f"{split}: "+"A report with findings. "*n,"summary":"There were qualified benefits."} for n in [3,15,100]]
    return prepare_split(s,tok,split,rows,{"dataset":s.dataset_id,"dataset_config":s.dataset_config,
        "dataset_revision":"fixture","model_id":"tiny-llama","model_revision":"fixture"})


def test_whole_document_selection_integrity_and_duplicate_guard(tmp_path,monkeypatch):
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path));s=study(tmp_path);tok=ByteTokenizer()
    meta=prepare(s,"train");rows,_=load_split(s,"train")
    assert meta["counts"]==[1,1,1]
    assert all(tok.decode(r["source_ids"])==r["source"] for r in rows)
    assert rows[-1]["source_tokens"]>2048 and not rows[-1]["source_truncated"]
    raw=[{"report":r["source"],"summary":r["reference"]} for r in rows]
    raw.append({"report":"Different "+"abc "*30,"summary":"New summary."})
    m=prepare_split(s,tok,"validation",raw,meta)
    assert m["skipped"]["duplicate"]==3 and m["counts"]==[1,0,0]
    with pytest.raises(FileExistsError):prepare(s,"train")
    with Path(s.data_dir,"train.jsonl").open("a") as f:f.write("\n")
    with pytest.raises(ValueError,match="changed"):load_split(s,"train")


def test_reference_is_never_clipped(tmp_path):
    s=study(tmp_path);tok=ByteTokenizer();s.train_target_limit=10
    r=normalize_row({"report":"A long enough document.","summary":"A target over ten bytes."},s,tok,0,"train")
    with pytest.raises(ValueError,match="rather than truncating"):summarization_episode(r,tok,s)


class FakeEngine:
    def __init__(self,s):
        self.study=s;self.tokenizer=ByteTokenizer();self.device=torch.device("cpu");self.native=2048
        self.model=SimpleNamespace(cfg=s.base());self.inputs=[]
    def generate(self,prompt,budget,recurrent=False):
        self.inputs.append(prompt)
        return Generated(self.tokenizer.encode("brief"),len(prompt),5,time.perf_counter(),.01,100,False)
    def text(self,ids):return self.tokenizer.decode(ids)


@pytest.mark.parametrize("method",["rolling","map_reduce"])
def test_text_compression_covers_whole_source_and_charges_all_passes(method,tmp_path):
    s=study(tmp_path);e=FakeEngine(s);source=list(range(1000,7000))
    out=run_document(e,{"source_ids":source},method)
    visited=[x for call in e.inputs for x in call if x>=1000]
    assert visited==source
    assert out["source_tokens_seen"]==len(source) and out["num_calls"]>2
    assert out["all_call_input_tokens"]==sum(map(len,e.inputs))
    assert out["all_generation_tokens"]==5*len(e.inputs)
    assert out["final_first_token_s"]>=0
    assert all(len(x)+(s.max_new_tokens if i==len(e.inputs)-1 else s.intermediate_tokens)<=s.base().window
               for i,x in enumerate(e.inputs))


def test_native_skip_and_truncation_budgets(tmp_path):
    s=study(tmp_path);e=FakeEngine(s);row={"source_ids":list(range(3,2303))}
    assert run_document(e,row,"native_full")["status"]=="skipped_native_limit"
    assert not e.inputs
    r=run_document(e,row,"head_tail")
    assert 0<r["source_tokens_seen"]<512 and len(e.inputs[0])+8==512
    assert select_source([1,2,3,4],1,"head_tail")==[1]


@pytest.mark.parametrize("method",["slerp","delta","rmt"])
def test_summary_training_exact_resume(method,tmp_path,monkeypatch):
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path));s=study(tmp_path,method)
    # Two small documents keep the test short, while RMT crosses boundaries.
    s.length_edges=[16,256,1024];s.train_source_limit=1024
    prepare(s,"train")
    train(s,tmp_path/"resumed",device="cpu",max_updates=1)
    train(s,tmp_path/"resumed",device="cpu",resume=True)
    train(s,tmp_path/"continuous",device="cpu")
    a=load_file(str(checkpoint_path(tmp_path/"resumed")/"model.safetensors"))
    b=load_file(str(checkpoint_path(tmp_path/"continuous")/"model.safetensors"))
    assert all(torch.equal(a[k],b[k]) for k in a)
    s.max_new_tokens+=1
    with pytest.raises(ValueError,match="differ"):train(s,tmp_path/"resumed",device="cpu",resume=True)


def test_evaluation_manifest_report_and_true_beyond_native(tmp_path,monkeypatch):
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path));s=study(tmp_path);prepare(s,"validation")
    paths=[]
    for method in ["native_full","head","rolling"]:
        p=tmp_path/(method+".jsonl");evaluate(s,method,p,device="cpu");paths.append(p)
    native=[json.loads(l) for l in paths[0].read_text().splitlines()]
    assert native[-1]["source_over_native"] and native[-1]["status"]=="skipped_native_limit"
    result=report(paths,tmp_path/"report")
    assert len(result)==6
    assert (tmp_path/"report/quality-vs-length.png").exists()
    assert (tmp_path/"report/human-review.csv").exists()
    paths[0].write_text(paths[0].read_text()+"\n")
    with pytest.raises(ValueError,match="modified"):report(paths,tmp_path/"bad")


def test_pairs_match_reference_and_prompt():
    a={"source_sha256":"s","reference_sha256":"r","model_id":"m","model_revision":"v",
       "dataset_revision":"d","max_new_tokens":50,"prompt_sha256":"p","status":"ok","rougeLsum":10}
    assert paired_difference([a],[dict(a,rougeLsum=13)],"rougeLsum")["difference"]==3
    assert paired_difference([a],[dict(a,reference_sha256="other")],"rougeLsum")["paired_documents"]==0


def test_distillation_zero_when_equal_and_skip_beyond_teacher_context(tmp_path):
    from slerp_context.model import RecurrentLM,build_base
    s=study(tmp_path,'local');cfg=s.base();tok=ByteTokenizer()
    model=RecurrentLM.build(cfg).eval();teacher=build_base(cfg).eval().requires_grad_(False)
    row=normalize_row({'report':'This report documents qualified benefits.',
        'summary':'Benefits had limitations.'},s,tok,0,'train')
    ep=summarization_episode(row,tok,s)
    kl,n=memory_distillation(model,teacher,ep,s,torch.device('cpu'))
    assert abs(float(kl.detach()))<1e-6 and n>0
    kl.backward();assert all(p.grad is None for p in teacher.parameters())
    ep.prompt=ep.prompt*100
    assert memory_distillation(model,teacher,ep,s,torch.device('cpu'))==(None,0)


def test_latency_does_not_pair_across_different_hardware():
    a={'source_sha256':'s','reference_sha256':'r','model_id':'m','model_revision':'v',
       'dataset_revision':'d','max_new_tokens':50,'prompt_sha256':'p','status':'ok',
       'device_name':'4090','end_to_end_s':3.0}
    assert paired_difference([a],[dict(a,device_name='Orin')],'end_to_end_s')['paired_documents']==0
