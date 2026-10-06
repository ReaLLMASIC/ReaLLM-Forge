from dataclasses import replace
import json
import pytest
import torch
from slerp_context.config import Config
from slerp_context.model import RecurrentLM, build_base
from slerp_context.data import ByteTokenizer, synthetic
from slerp_context.storage import save_checkpoint, finalize_run, checkpoint_path
from slerp_context.train import optimizer_for, retention_kl
from slerp_context.evaluate import evaluate, assess


def config(arch="hunyuan_v1_dense", **kw):
    return Config(tiny=True,tiny_arch=arch,window=16,chunk=4,slots=4,
                  min_free_gb=0,max_project_gb=100,**kw)


@pytest.mark.parametrize("arch",["hunyuan_v1_dense","llama","gpt_neox"])
def test_native_parity_with_nontrivial_qk_gains_and_gqa(arch):
    m=RecurrentLM.build(config(arch,method="local")).eval()
    if arch=="hunyuan_v1_dense":
        with torch.no_grad():
            for layer in m.layers:
                layer.self_attn.query_layernorm.weight.uniform_(0.5,1.5)
                layer.self_attn.key_layernorm.weight.uniform_(0.5,1.5)
    x=torch.randint(3,259,(1,16))
    with torch.no_grad():
        expected=m.original(x,use_cache=False).logits
        actual,st=m.consume(x)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    assert st.layers[0].k.shape[1]==m.kv_heads
    if arch!="gpt_neox": assert m.dim==12 and m.dim!=32//4


@pytest.mark.parametrize("arch",["hunyuan_v1_dense","llama"])
@pytest.mark.parametrize("method",["slerp","nlerp","local"])
def test_gqa_causality_splits_and_checkpoint_gradients(arch,method):
    a=RecurrentLM.build(config(arch,method=method,checkpoint=False))
    b=RecurrentLM.build(config(arch,method=method,checkpoint=True))
    x=torch.randint(3,259,(1,45));y=x.clone();y[:,:-4]=-100
    la,_=a.episode_loss(x,y);lb,_=b.episode_loss(x,y)
    la.backward();lb.backward()
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-3)
    a.eval()
    with torch.no_grad():
        full,_=a.consume(x)
        first,st=a.consume(x[:,:19]);second,st=a.consume(x[:,19:],st)
        torch.testing.assert_close(torch.cat([first,second],1),full,atol=2e-6,rtol=2e-5)
        changed=x.clone();changed[:,22:]=3
        torch.testing.assert_close(a.consume(changed)[0][:,:22],full[:,:22],atol=2e-6,rtol=2e-5)


def test_full_weight_update_and_self_contained_reload(tmp_path,monkeypatch):
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path))
    m=RecurrentLM.build(config())
    assert all(p.requires_grad and p.dtype==torch.float32 for p in m.backbone.parameters())
    assert not any("lora" in name.lower() for name,_ in m.named_parameters())
    before={name:p.detach().clone() for name,p in m.backbone.named_parameters()}
    opt=optimizer_for(m);x=torch.randint(3,259,(1,40))
    m.episode_loss(x,x)[0].backward();opt.step()
    changed=[name for name,p in m.backbone.named_parameters() if not torch.equal(p,before[name])]
    assert len(changed)==len(before)
    save_checkpoint(m,opt,tmp_path/"run",{"step":1,"episode":1})
    clone=RecurrentLM.from_pretrained(tmp_path/"run")
    for (name,p),(other,q) in zip(m.named_parameters(),clone.named_parameters()):
        assert name==other and torch.equal(p,q)
    assert clone.embedding.weight.data_ptr()==clone.lm_head.weight.data_ptr()
    assert (checkpoint_path(tmp_path/"run")/"model.safetensors").stat().st_size>0
    result=finalize_run(tmp_path/"run")
    assert result["removed_optimizer_bytes"]>0 and not result["resumable"]
    assert not (checkpoint_path(tmp_path/"run")/"training.pt").exists()
    RecurrentLM.from_pretrained(tmp_path/"run")


def test_late_answer_reaches_early_backbone_features():
    m=RecurrentLM.build(config(checkpoint=True,detach_local=False))
    activations=[]
    def save(module,args,out):
        out.retain_grad();activations.append(out)
    handle=m.embedding.register_forward_hook(save)
    x=torch.randint(3,259,(1,40));y=x.clone();y[:,:-2]=-100
    m.episode_loss(x,y)[0].backward();handle.remove()
    assert activations[0].grad is not None and activations[0].grad.norm()>0
    assert m.memories[0].writer.weight.grad.norm()>0


def test_exact_retention_kl_and_frozen_reference():
    cfg=config(kl_weight=.02,kl_context=16,kl_tokens=4)
    m=RecurrentLM.build(cfg)
    teacher=build_base(cfg).eval().requires_grad_(False)
    x=torch.randint(3,259,(1,16))
    assert abs(float(retention_kl(m,teacher,x).detach()))<1e-6
    with torch.no_grad(): m.embedding.weight[3].add_(0.5)
    loss=retention_kl(m,teacher,x)
    assert loss>0
    loss.backward()
    assert m.embedding.weight.grad.norm()>0
    assert all(p.grad is None for p in teacher.parameters())


def test_full_training_learns_a_small_evicted_example():
    cfg=config(backbone_lr=.003,memory_lr=.005,checkpoint=False)
    m=RecurrentLM.build(cfg);opt=optimizer_for(m)
    x=torch.randint(3,259,(1,36));y=torch.full_like(x,-100);y[:,-2:]=torch.tensor([17,19])
    losses=[]
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss,_=m.episode_loss(x,y);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1)
        opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<losses[0]*.5


def test_random_query_indices_and_multifact_order():
    tok=ByteTokenizer()
    indices={synthetic(tok,2048,17,i,fact_count=8).metadata["query_record_index"] for i in range(64)}
    assert indices==set(range(8))
    ep=synthetic(tok,2048,17,0,task="multi",fact_count=8)
    assert len(set(ep.metadata["evidence_record_indices"]))==3
    assert ep.metadata["evidence_record_indices"]!=[0,1,2]


def test_native_evaluation_uses_hf_generate(tmp_path,monkeypatch):
    from transformers import HunYuanDenseV1ForCausalLM
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path))
    called=[];original=HunYuanDenseV1ForCausalLM.generate
    def capture(self,*args,**kw):
        called.append(kw.get("logits_to_keep"));return original(self,*args,**kw)
    monkeypatch.setattr(HunYuanDenseV1ForCausalLM,"generate",capture)
    out=tmp_path/"native.jsonl"
    evaluate(None,out,[512],1,["recall"],[2],[.05],device="cpu",
        config=config(method="native"),max_new_tokens=2)
    row=json.loads(out.read_text())
    assert called==[1] and row["method"]=="native" and row["training_mode"]=="pretrained"
    assert row["state_bytes"]>0
    result=assess([out],tmp_path/"report.json")
    assert len(result["groups"])==1


def test_bf16_compute_keeps_fp32_trainable_weights():
    m=RecurrentLM.build(config())
    x=torch.randint(3,259,(1,36))
    with torch.autocast("cpu",dtype=torch.bfloat16): loss,state=m.episode_loss(x,x)
    loss.backward()
    assert all(p.dtype==torch.float32 for p in m.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert state.layers[0].k.dtype==torch.bfloat16


def test_value_scoring_never_selects_by_reference():
    from slerp_context.evaluate import exact,value_exact
    text='<answer>\nThe value is **123456**.\n</answer>'
    assert exact(text,'123456')==0 and value_exact(text,'123456')==1
    assert value_exact('123456, 654321','123456')==0
    assert value_exact('654321, 123456','123456, 654321')==0
    assert value_exact('123456 and again 123456','123456')==0
    assert value_exact('1234567','123456')==0
    assert value_exact('<think>123456','123456')==0
    assert value_exact('<think>654321</think><answer>123456</answer>','123456')==1
    assert value_exact('<answer>654321</answer><answer>123456</answer>','123456')==0
