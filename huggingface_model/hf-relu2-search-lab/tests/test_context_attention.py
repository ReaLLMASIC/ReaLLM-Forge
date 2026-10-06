"""Independent math, causality, optimizer and HF tests for both new rules."""
import copy
import json
from types import SimpleNamespace
import pytest
import torch
from transformers import AutoModelForCausalLM
from hf_model.context_attention import (CONTEXT_ARMS, context_attention_reference as ref,
                                        visible_counts, length_factors)
from hf_model.configuration_comparison import ComparisonConfig
from hf_model.modeling_comparison import ComparisonForCausalLM
from experiment.common import make_model, state_digest, token_loss
from experiment.optimizers import partition, optimizer_for
from context_sweep.recipe import ROOT, read, build
from diagnostics.probe_context import evaluate, weight_statistics


def tiny(variant="relu2_bias_scale_half", norm="none"):
    return ComparisonForCausalLM(ComparisonConfig(vocab_size=32, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=3,
        max_position_embeddings=32, variant=variant, output_norm=norm,
        causal_length_anchor=4, qk_norm_scale_init=3., use_qk_norm=True,
        use_rotary_embeddings=True, use_absolute_position_embeddings=False,
        sdpa_backend="math", require_fused=False, relu2max_accelerator="torch"))


@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_all_first_order_derivatives_with_signed_per_head_bias(alpha):
    torch.manual_seed(921)
    q=torch.randn(1,2,3,2,dtype=torch.float64,requires_grad=True)
    k=torch.randn(1,2,4,2,dtype=torch.float64,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    g=torch.tensor(-.7,dtype=torch.float64,requires_grad=True)
    tau=torch.tensor([-.3,.15],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[1,0,1,1]])
    fn=lambda q,k,v,g,t:ref(q,k,v,g,t,divisor=2.,alpha=alpha,anchor=1.5,attention_mask=mask)
    assert torch.autograd.gradcheck(fn,(q,k,v,g,tau),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize("alpha",[0.,.5,1.])
def test_closed_form_bias_sign_and_independent_heads(alpha):
    q=torch.ones(2,2,4,1,dtype=torch.float64)
    k=torch.ones_like(q); v=torch.ones_like(q)
    tau=torch.tensor([.5,1.5],dtype=torch.float64,requires_grad=True)
    out=ref(q,k,v,scale=2.,bias=tau,divisor=4.,alpha=alpha,anchor=2.)
    out.sum().backward()
    counts=torch.arange(1,5,dtype=torch.float64)
    factors=torch.maximum(counts/2,torch.ones_like(counts)).pow(-alpha)
    expected_out=(2-tau.detach())[None,:,None,None].square()*counts[None,None,:,None]*factors[None,None,:,None]/4
    torch.testing.assert_close(out,expected_out.expand_as(out))
    expected_grad=-2*2*(2-tau.detach())*(counts*factors).sum()/4
    torch.testing.assert_close(tau.grad,expected_grad)


def test_counts_ignore_future_padding_and_all_masked_prefixes():
    mask=torch.tensor([[0,0,1,1,0,1],[1,0,0,1,1,0]])
    counts=visible_counts(2,6,6,"cpu",mask)
    torch.testing.assert_close(counts,torch.tensor([[0,0,1,2,2,3],[1,1,1,2,3,3]]))
    torch.testing.assert_close(visible_counts(2,2,6,"cpu",mask),counts[:,-2:])
    torch.testing.assert_close(visible_counts(2,2,6,"cpu",mask,causal=False),torch.tensor([[3,3],[3,3]]))


@pytest.mark.parametrize("alpha",[.5,1.])
def test_anchor_preserves_short_rows_and_scales_long_rows(alpha):
    counts=torch.tensor([0,1,256,512,1024,2048])
    actual=length_factors(counts,alpha,512)
    torch.testing.assert_close(actual[:4],torch.ones(4),atol=0,rtol=0)
    torch.testing.assert_close(actual[4:],torch.tensor([2**-alpha,4**-alpha]))


@pytest.mark.parametrize("alpha",[0.,.5,1.])
def test_negative_bias_does_not_resurrect_masked_entries(alpha):
    q=torch.zeros(1,2,4,2,requires_grad=True)
    k=torch.zeros_like(q,requires_grad=True);v=torch.ones_like(q,requires_grad=True)
    bias=torch.tensor([-1.,-2.],requires_grad=True)
    out=ref(q,k,v,bias=bias,alpha=alpha,anchor=1,attention_mask=torch.zeros(1,4))
    assert torch.count_nonzero(out)==0
    out.sum().backward()
    assert all(torch.count_nonzero(t.grad)==0 for t in (q,k,v,bias))


@pytest.mark.parametrize("alpha",[0.,.5,1.])
def test_prefix_cache_and_padding_invariance(alpha):
    torch.manual_seed(3)
    q,k,v=[torch.randn(2,3,13,5,dtype=torch.float64) for _ in range(3)]
    tau=torch.tensor([-.25,.2,0.],dtype=torch.float64)
    mask=torch.ones(2,13,dtype=torch.bool);mask[0,:3]=False;mask[1,6]=False
    fn=lambda a,b,c,ma:ref(a,b,c,scale=-.8,bias=tau,alpha=alpha,anchor=4,attention_mask=ma)
    full=fn(q,k,v,mask)
    torch.testing.assert_close(fn(q[:,:,:7],k[:,:,:7],v[:,:,:7],mask[:,:7]),full[:,:,:7],atol=1e-14,rtol=1e-12)
    for start,end in [(0,4),(4,5),(5,9),(9,13)]:
        torch.testing.assert_close(fn(q[:,:,start:end],k[:,:,:end],v[:,:,:end],mask[:,:end]),full[:,:,start:end],atol=1e-14,rtol=1e-12)
    # Appending enormous future keys/values cannot affect earlier queries.
    k[:,:,10:]*=1e3;v[:,:,10:]*=1e3
    torch.testing.assert_close(fn(q,k,v,mask)[:,:,:10],full[:,:,:10],atol=1e-14,rtol=1e-12)


@pytest.mark.parametrize("variant",list(CONTEXT_ARMS))
def test_model_cache_roundtrip_and_checkpointed_gradients(variant,tmp_path):
    torch.manual_seed(1)
    model=tiny(variant,"capped").eval()
    if model.config.learned_subtractive_bias:
        with torch.no_grad():
            for layer in model.model.layers:layer.attn.subtractive_bias.copy_(torch.tensor([-.3,.1,.2]))
    x=torch.randint(0,32,(1,10));mask=torch.ones_like(x);mask[:,:2]=0
    full=model(x,attention_mask=mask,use_cache=True).logits
    prefix=model(x[:,:5],attention_mask=mask[:,:5],use_cache=True)
    tail=model(x[:,5:],attention_mask=mask,past_key_values=prefix.past_key_values,use_cache=True).logits
    torch.testing.assert_close(tail,full[:,5:],atol=2e-6,rtol=2e-5)
    model.save_pretrained(tmp_path)
    loaded=AutoModelForCausalLM.from_pretrained(tmp_path,trust_remote_code=True).eval()
    torch.testing.assert_close(loaded(x,attention_mask=mask).logits,full,atol=0,rtol=0)
    assert loaded.config.causal_length_alpha==CONTEXT_ARMS[variant][1]
    model.train();model.model.gradient_checkpointing=True
    token_loss(model,x,torch.roll(x,-1,-1),4).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_zero_initialized_bias_and_auxiliary_adamw_parameter_ownership():
    model=tiny()
    groups=partition(model,"muon")
    bias_names=[n for n,p in model.named_parameters() if n.endswith(".subtractive_bias")]
    assert len(bias_names)==2
    assert all(p.shape==(3,) and torch.count_nonzero(p)==0 for n,p in model.named_parameters() if n in bias_names)
    assert all(n in {name for name,p in groups["adamw_no_decay"]} for n in bias_names)
    recipe=read(ROOT/'configs/context_ablation_50m.json')["optimizer_recipes"]["muon"]
    opt=optimizer_for(model,"muon",recipe,"cpu")
    x=torch.arange(12).reshape(1,12)
    token_loss(model,x,torch.roll(x,-1,-1),4).backward();opt.step()
    assert any(torch.count_nonzero(layer.attn.subtractive_bias)>0 for layer in model.model.layers)


def test_shared_backbone_and_plan_have_every_factorial_treatment():
    plan=build(read(ROOT/'configs/context_ablation_50m.json'))
    assert plan['run_count']==48 and plan['tokens_per_run']==100007936
    assert plan['spec']['model']['hidden_size']==384 and plan['spec']['model']['num_attention_heads']==3
    smoke=build(read(ROOT/'configs/context_ablation_smoke.json'))
    hashes=[]
    for variant in plan['variants']:
        model,_=make_model(smoke['configs']['16_none'],variant,0,'cpu')
        hashes.append(state_digest(model,shared_only=True))
    assert len(set(hashes))==1
    assert set(CONTEXT_ARMS)<=set(plan['variants'])


def test_probe_accounts_for_bias_and_scaling_and_does_not_change_parameters():
    cfg=tiny().config
    cfg.causal_length_anchor=2
    q=torch.ones(1,3,4,1);bias=torch.tensor([0.,.5,1.5]);pos=torch.tensor([0,3])
    stats=weight_statistics(q,q,torch.tensor(2.),cfg,pos,bias)
    factors=torch.tensor([1.,2**-.5])
    expected=(2-bias)[:,None].square()*torch.tensor([1,4])[None,:]/256*factors
    torch.testing.assert_close(stats['row_mass'],expected)
    model=tiny().eval()
    before=state_digest(model)
    class Stream:
        def batch(self,cursor,batch_size,length,device):
            x=torch.arange(cursor,cursor+length,device=device)[None]%32
            return x,(x+1)%32
    result=evaluate(model,Stream(),8,16,'cpu','float32')
    assert result['attention'] and state_digest(model)==before
    assert all(r['length_alpha']==.5 for r in result['attention'])
    assert all(layer.attn.diagnostic_callback is None for layer in model.model.layers)


@pytest.mark.parametrize('kwargs',[{'causal_length_anchor':0},{'subtractive_bias_init':float('nan')},
    {'causal_length_alpha':1.},{'relu2max_divide_by_sequence_length':True}])
def test_invalid_or_mislabeled_config_is_rejected(kwargs):
    with pytest.raises(ValueError):ComparisonConfig(variant='relu2_bias',**kwargs)
