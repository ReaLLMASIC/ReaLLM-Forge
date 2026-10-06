import copy
from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from slerp_context.config import Config
from slerp_context.geometry import slerp,nlerp,unit
from slerp_context.model import RecurrentLM
from slerp_context.state import StreamState
from slerp_context.data import ByteTokenizer,synthetic
from slerp_context.storage import save_checkpoint,load_weights,checkpoint_path,disk_guard
from slerp_context.train import optimizer_for
from slerp_context.rl import grpo_loss

torch.set_num_threads(1)


def tiny(method="slerp",**kwargs):
    return RecurrentLM.build(Config(tiny=True,tiny_arch="gpt_neox",keep_checkpoints=2,window=16,chunk=4,slots=4,
        min_free_gb=0,max_project_gb=100,method=method,**kwargs))


@pytest.mark.parametrize("kind",["random","parallel","near_parallel","antipodal","near_antipodal"])
def test_geometry_endpoints_norm_gradients(kind):
    torch.manual_seed(5)
    u=unit(torch.randn(8,12,dtype=torch.float64)).requires_grad_()
    v={"random":unit(torch.randn_like(u)),"parallel":u.detach().clone(),
       "near_parallel":unit(u.detach()+torch.randn_like(u)*1e-6),
       "antipodal":-u.detach(),"near_antipodal":unit(-u.detach()+torch.randn_like(u)*1e-6)}[kind].requires_grad_()
    a=torch.full((8,1),.37,dtype=torch.float64,requires_grad=True)
    y=slerp(u,v,a)
    assert torch.allclose(y.norm(dim=-1),torch.ones(8,dtype=torch.float64),atol=1e-7)
    assert torch.allclose(slerp(u,v,a*0),u,atol=1e-6)
    assert torch.allclose(slerp(u,v,a*0+1),v,atol=1e-6)
    (y*torch.randn_like(y)).sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in [u,v,a])


def test_slerp_gradcheck_and_nlerp_reparameterization():
    torch.manual_seed(2)
    u=unit(torch.randn(2,5,dtype=torch.double)).requires_grad_()
    v=unit(torch.randn(2,5,dtype=torch.double)).requires_grad_()
    a=torch.tensor([[.21],[.83]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(slerp,(u,v,a),atol=1e-4)
    theta=torch.acos((u*v).sum(-1,keepdim=True))
    b=torch.sin(a*theta)/(torch.sin((1-a)*theta)+torch.sin(a*theta))
    assert torch.allclose(slerp(u,v,a),nlerp(u,v,b),atol=1e-8)


@pytest.mark.parametrize("method",["slerp","nlerp","ema","fixed","local","pi"])
def test_future_causality_and_arbitrary_prefill_splits(method):
    model=tiny(method);model.eval()
    ids=torch.randint(3,259,(1,49));changed=ids.clone();changed[:,25:]=torch.randint(3,259,(1,24))
    with torch.no_grad():
        expected,state=model.consume(ids)
        future,_=model.consume(changed)
        assert torch.allclose(expected[:,:25],future[:,:25],atol=2e-6)
        st=None;parts=[]
        for a,b in [(0,3),(3,18),(18,19),(19,33),(33,49)]:
            y,st=model.consume(ids[:,a:b],st);parts.append(y)
        assert torch.allclose(torch.cat(parts,1),expected,atol=2e-6)
        st=None;one=[]
        for i in range(ids.shape[1]):
            y,st=model.consume(ids[:,i:i+1],st);one.append(y)
        assert torch.allclose(torch.cat(one,1),expected,atol=2e-6)
        assert state.consumed==49
        assert state.layers[0].k.shape[-2] <= (49 if method=="pi" else 16)


def test_native_hf_equivalence_and_interpolation():
    local=tiny("local");local.eval()
    ids=torch.randint(3,259,(1,16))
    with torch.no_grad():
        native=local.original(ids,use_cache=False).logits
        actual,_=local.consume(ids)
        assert torch.allclose(native,actual,atol=2e-6,rtol=2e-5)
        mem=tiny();mem.eval()
        assert torch.allclose(mem.consume(ids)[0],native,atol=2e-6,rtol=2e-5)
        pi=tiny("pi");pi.eval()
        # HF supports linear RoPE interpolation in this pinned API; verify all dimensions.
        from transformers import GPTNeoXForCausalLM
        config=copy.deepcopy(local.original.config)
        config.rope_scaling={"rope_type":"linear","factor":4.0}
        hf=GPTNeoXForCausalLM(config)
        # Remove PEFT wrappers for exact original projection tensors.
        weights={k.replace(".base_layer",""):v for k,v in local.original.state_dict().items()
                 if "lora_" not in k}
        hf.load_state_dict(weights);hf.eval()
        assert torch.allclose(hf(ids,use_cache=False).logits,pi.consume(ids)[0],atol=2e-6,rtol=2e-5)


def test_next_token_loss_keeps_chunk_boundaries():
    model=tiny();model.eval()
    seq=torch.randint(3,259,(1,44));ids,labels=seq[:,:-1],seq[:,1:]
    logits,_=model.consume(ids)
    loss,_=model.episode_loss(ids,labels)
    assert torch.allclose(loss,F.cross_entropy(logits.reshape(-1,259),labels.flatten()),atol=1e-6)


def test_earliest_write_receives_late_loss_gradient():
    model=tiny(checkpoint=False);model.train()
    ids=torch.randint(3,259,(1,48))
    first,state=model.consume(ids[:,:20])
    early=state.layers[0].direction
    early.retain_grad()
    logits,state=model.consume(ids[:,20:],state)
    F.cross_entropy(logits[:,-1],torch.tensor([8])).backward()
    assert early.grad is not None and early.grad.norm()>1e-9
    assert model.memories[0].writer.weight.grad.norm()>1e-9
    assert model.memories[0].update_gate[-1].weight.grad.norm()>1e-9


def test_checkpointed_recurrence_gradients_match():
    a=tiny(checkpoint=False);b=tiny(checkpoint=True)
    x=torch.randint(3,259,(1,48));y=x.clone();y[:,:-5]=-100
    la,_=a.episode_loss(x,y);lb,_=b.episode_loss(x,y)
    la.backward();lb.backward()
    assert torch.allclose(la,lb,atol=1e-7)
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        if pa.grad is not None:
            assert pb.grad is not None,na
            assert torch.allclose(pa.grad,pb.grad,atol=1e-6,rtol=1e-3),na


def test_empty_and_zero_writes():
    model=tiny();mem=model.memories[0];state=model.initial_state().layers[0]
    q=torch.randn(1,4,3,8)
    assert mem.read(q,state.direction,state.log_radius,state.valid).abs().sum()==0
    k=torch.zeros(1,4,4,8)
    m,r,v=mem.write(k,k,state.direction,state.log_radius,state.valid)
    assert torch.equal(m,state.direction) and not v.any()


def test_session_continuation_and_no_aliasing(tmp_path):
    model=tiny();model.eval();x=torch.randint(3,259,(1,37))
    with torch.no_grad():
        _,st=model.consume(x[:,:23]);st.save(tmp_path/"session.pt")
        restored=StreamState.load(tmp_path/"session.pt")
        assert torch.allclose(model.consume(x[:,23:],st)[0],model.consume(x[:,23:],restored)[0])
        original=[t.clone() for t in st.layers[0].tensors()]
        model.consume(x[:,23:],st)
        assert all(torch.equal(a,b) for a,b in zip(original,st.layers[0].tensors()))
        assert torch.allclose(model.consume(x)[0],model.consume(x)[0])


def test_generation_consumes_each_token_once():
    model=tiny();model.eval();x=torch.randint(3,259,(1,23))
    with torch.no_grad():
        out,st=model.generate_stream(x,9)
        _,replayed=model.consume(torch.cat((x,out),1))
        assert st.consumed==32
        for a,b in zip(st.layers,replayed.layers):
            for u,v in zip(a.tensors(),b.tensors()):
                assert torch.allclose(u,v,atol=2e-6)


def test_checkpoint_roundtrip_optimizer_retention(tmp_path,monkeypatch):
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path))
    model=tiny();optimizer=optimizer_for(model)
    x=torch.randint(3,259,(1,48))
    loss,_=model.episode_loss(x,x);loss.backward();optimizer.step()
    for i in [1,2,3]:
        save_checkpoint(model,optimizer,tmp_path/"run",{"step":i,"episode":i})
    assert len(list((tmp_path/"run").glob("step-*")))==2
    clone=tiny();load_weights(clone,tmp_path/"run")
    model.eval();clone.eval()
    with torch.no_grad(): assert torch.allclose(model.consume(x)[0],clone.consume(x)[0],atol=1e-7)
    state=torch.load(checkpoint_path(tmp_path/"run")/"training.pt",weights_only=True)
    other=optimizer_for(clone);other.load_state_dict(state["optimizer"])
    assert state["progress"]["step"]==3
    assert next(iter(other.state.values()))["step"]==1


def test_disk_guard_before_writes(tmp_path,monkeypatch):
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path))
    cfg=Config(min_free_gb=1e9)
    with pytest.raises(RuntimeError,match="Disk guard"):disk_guard(cfg)
    cfg=Config(min_free_gb=0,max_project_gb=.000000001)
    (tmp_path/"file").write_bytes(b"123")
    with pytest.raises(RuntimeError,match="project uses"):disk_guard(cfg)


@pytest.mark.parametrize("task",["recall","update","trace","multi","capacity"])
def test_data_length_masks_eviction_and_disjoint_split(task):
    tok=ByteTokenizer()
    ep=synthetic(tok,1024,99,2,task=task,fact_count=2,position=.1,window=64,chunk=16)
    ids,labels=ep.tensors("cpu")
    assert len(ep.prompt)+len(ep.answer)==1024
    assert (labels[:,:len(ep.prompt)-1]==-100).all()
    assert int((labels!=-100).sum())==len(ep.answer)
    assert ep.metadata["all_evidence_evicted"]
    other=synthetic(tok,1024,99,2,split="test",task=task,fact_count=2)
    assert ep.target!=other.target


def test_rl_prefix_gradient_and_policy_objective():
    model=tiny();model.train()
    prompt=torch.randint(3,259,(1,44));answer=torch.randint(3,259,(1,4))
    lp=model.completion_logps(prompt,answer)
    assert lp.shape==(4,)
    with torch.no_grad():
        logits,_=model.consume(torch.cat((prompt,answer),1)[:,:-1])
        direct=logits[:,43:].float().log_softmax(-1).gather(-1,answer.unsqueeze(-1)).flatten()
    assert torch.allclose(lp,direct,atol=1e-6)
    loss,kl=grpo_loss(lp,lp.detach(),lp.detach(),torch.tensor(1.))
    assert torch.allclose(loss,torch.tensor(-1.)) and abs(float(kl.detach()))<1e-7
    loss.backward()
    assert model.memories[0].writer.weight.grad.norm()>1e-9


def test_reject_padding_batch_and_invalid_architecture():
    model=tiny()
    with pytest.raises(ValueError,match="Batch 1"):
        model.consume(torch.ones(2,4,dtype=torch.long))
