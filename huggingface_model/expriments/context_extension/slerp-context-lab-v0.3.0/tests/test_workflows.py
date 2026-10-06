from pathlib import Path
import json
import pytest
import torch
from slerp_context.config import Config
from slerp_context.model import RecurrentLM
from slerp_context.data import ByteTokenizer,synthetic


def test_custom_pretrained_roundtrip(tmp_path):
    cfg=Config(tiny=True,window=16,chunk=4,slots=4)
    model=RecurrentLM.build(cfg);model.eval()
    model.save_pretrained(tmp_path/"adapter")
    loaded=RecurrentLM.from_pretrained(tmp_path/"adapter");loaded.eval()
    x=torch.randint(3,259,(1,40))
    with torch.no_grad():
        assert torch.equal(model.consume(x)[0],loaded.consume(x)[0])


def test_resume_matches_uninterrupted(tmp_path,monkeypatch):
    from slerp_context.train import train
    from slerp_context.storage import checkpoint_path
    from safetensors.torch import load_file
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path))
    cfg=Config(tiny=True,window=64,chunk=16,slots=4,lengths=[512],length_weights=[1.0],
        tokens_per_update=511,token_budget=1022,min_free_gb=0,max_project_gb=100,save_every=1)
    cfg.save(tmp_path/"config.json")
    train(Config.load(tmp_path/"config.json"),tmp_path/"resumed",max_updates=1,device="cpu")
    train(Config.load(tmp_path/"config.json"),tmp_path/"resumed",resume=True,device="cpu")
    train(Config.load(tmp_path/"config.json"),tmp_path/"continuous",device="cpu")
    a=load_file(str(checkpoint_path(tmp_path/"resumed")/"model.safetensors"))
    b=load_file(str(checkpoint_path(tmp_path/"continuous")/"model.safetensors"))
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_grpo_update_and_checkpoint(tmp_path,monkeypatch):
    import slerp_context.rl as rl
    from slerp_context.storage import checkpoint_path
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path))
    def build():
        return RecurrentLM.build(Config(tiny=True,window=64,chunk=16,slots=4,
            lengths=[512],length_weights=[1.0],min_free_gb=0,max_project_gb=100))
    policy,reference=build(),build();tok=ByteTokenizer()
    original=policy.memories[0].writer.weight.detach().clone()
    calls=iter([(policy,tok),(reference,tok)])
    monkeypatch.setattr(rl,"load_run",lambda *a,**k:next(calls))
    episode=synthetic(tok,512,42873,0,task="recall",fact_count=2,window=64,chunk=16)
    answers=iter([episode.answer,tok.encode("wrong\n")])
    # Deterministic test rollouts exercise mixed-reward optimization; not a quality test.
    def sample(*a,**k):return torch.tensor([next(answers)]),policy.initial_state()
    monkeypatch.setattr(policy,"generate_stream",sample)
    result=rl.grpo("unused",tmp_path/"rl",groups=1,group_size=2,length=512,device="cpu")
    assert result["optimizer_updates"]==1
    assert not torch.equal(original,policy.memories[0].writer.weight)
    assert (checkpoint_path(tmp_path/"rl")/"COMPLETE").exists()


def test_archive_latest_only(tmp_path):
    import tarfile
    from slerp_context.storage import archive_run
    run=tmp_path/"run";run.mkdir()
    (run/"latest.json").write_text(json.dumps({"directory":"step-00000002"}))
    for name in ["step-00000001","step-00000002"]:
        (run/name).mkdir();(run/name/"COMPLETE").write_text("test")
    result=archive_run(run,tmp_path/"archive",mbps=1000)
    with tarfile.open(result) as archive:
        names=archive.getnames()
    assert any("step-00000002" in n for n in names)
    assert not any("step-00000001" in n for n in names)


def test_tokenizer_revision_guard(tmp_path):
    import numpy as np
    from slerp_context.data import TokenDocuments
    np.arange(40,dtype=np.uint32).tofile(tmp_path/"train.bin")
    (tmp_path/"train.index.jsonl").write_text(json.dumps({"offset":0,"length":40,"book_id":"x","split":"train"})+"\n")
    (tmp_path/"train.manifest.json").write_text(json.dumps({"tokens":40,"model":"different","model_revision":"abc"}))
    docs=TokenDocuments(tmp_path,"train")
    with pytest.raises(ValueError,match="revision"):
        docs.verify_tokenizer(Config())
