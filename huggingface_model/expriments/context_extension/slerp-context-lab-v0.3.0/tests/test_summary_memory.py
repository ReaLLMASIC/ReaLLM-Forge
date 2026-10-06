from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from slerp_context.config import Config
from slerp_context.model import RecurrentLM
from slerp_context.memory import delta_block
from slerp_context.train import optimizer_for
from slerp_context.storage import save_checkpoint


def cfg(method,**kw):
    return Config(tiny=True,tiny_arch="llama",method=method,window=32,chunk=8,slots=4,
                  min_free_gb=0,max_project_gb=100,**kw)


def test_delta_triangular_solve_matches_sequential_values_and_gradients():
    torch.manual_seed(7)
    state=torch.randn(2,3,4,4,dtype=torch.float64,requires_grad=True)
    keys=F.normalize(torch.randn(2,3,7,4,dtype=torch.float64),dim=-1).requires_grad_()
    values=torch.randn_like(keys,requires_grad=True)
    beta=torch.rand(2,3,7,1,dtype=torch.float64,requires_grad=True)
    expected=state
    for i in range(keys.shape[-2]):
        k,v,b=keys[...,i,:],values[...,i,:],beta[...,i,:]
        residual=v-(expected@k.unsqueeze(-1)).squeeze(-1)
        expected=expected+(b*residual).unsqueeze(-1)*k.unsqueeze(-2)
    actual=delta_block(state,keys,values,beta)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(actual.square().sum(),(state,keys,values,beta),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(state,keys,values,beta))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize("method",["rmt","delta"])
def test_memory_causality_and_incremental_generation(method):
    m=RecurrentLM.build(cfg(method,checkpoint=False)).eval()
    x=torch.randint(3,259,(1,55));changed=x.clone();changed[:,23:]=3
    with torch.no_grad():
        if method=="rmt":
            whole=m.tail_logits(x,55)
            prefix=m.tail_logits(changed,55)[:,:23]
            logits,state=m.prefill(x[:,:19]);parts=[logits]
            for i in range(19,55):
                logits,state=m.step(x[:,i:i+1],state);parts.append(logits)
            actual=torch.cat(parts,1)
        else:
            whole,_=m.consume(x);prefix=m.consume(changed)[0][:,:23]
            first,state=m.consume(x[:,:19]);second,state=m.consume(x[:,19:],state)
            actual=torch.cat([first[:,-1:],second],1)
        torch.testing.assert_close(prefix,whole[:,:23],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(actual,whole[:,18:],atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize("method",["rmt","delta"])
def test_checkpointed_gradient_matches_and_reaches_old_inputs(method):
    a=RecurrentLM.build(cfg(method,checkpoint=False))
    b=RecurrentLM.build(cfg(method,checkpoint=True))
    x=torch.randint(3,259,(1,55));y=torch.full_like(x,-100);y[:,-2:]=x[:,-2:]
    early=[]
    def hook(module,args,out):out.retain_grad();early.append(out)
    handle=a.embedding.register_forward_hook(hook)
    la,_=a.episode_loss(x,y);lb,_=b.episode_loss(x,y)
    la.backward();lb.backward();handle.remove()
    assert early[0].grad is not None and early[0].grad.norm()>0
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-3)
    assert any(p.grad is not None and p.grad.norm()>0 for p in a.memories.parameters())


@pytest.mark.parametrize("method",["rmt","delta"])
def test_full_backbone_update_and_offline_restore(method,tmp_path,monkeypatch):
    monkeypatch.setenv("SLERP_WORK_ROOT",str(tmp_path))
    m=RecurrentLM.build(cfg(method));opt=optimizer_for(m)
    before={n:p.detach().clone() for n,p in m.backbone.named_parameters()}
    x=torch.randint(3,259,(1,55));m.episode_loss(x,x)[0].backward();opt.step()
    assert all(p.requires_grad and not torch.equal(p,before[n]) for n,p in m.backbone.named_parameters())
    save_checkpoint(m,opt,tmp_path/"run",{"step":1})
    restored=RecurrentLM.from_pretrained(tmp_path/"run")
    assert all(torch.equal(p,q) for p,q in zip(m.parameters(),restored.parameters()))
    assert m.embedding.weight.data_ptr()==m.lm_head.weight.data_ptr()


def test_rmt_rejects_impossible_segment_budget():
    with pytest.raises(ValueError,match="chunk"):
        RecurrentLM.build(replace(cfg("rmt"),slots=20))
