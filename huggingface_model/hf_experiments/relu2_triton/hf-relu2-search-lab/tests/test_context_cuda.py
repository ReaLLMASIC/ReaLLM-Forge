"""Mandatory-on-GPU tests; skipped explicitly on CPU-only installations."""
import itertools
import pytest
import torch
from hf_model.triton_context_attention import triton_context_attention as fused
from hf_model.context_attention import context_attention_reference as reference
from diagnostics.context_precision import precision_oracle, assert_precision

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA execution required')


def accurate(got,want):
    got,want=got.float(),want.float()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got,want,rtol=.035,atol=.015)
    relative_rms=(got-want).square().mean().sqrt()/(want.square().mean().sqrt()+.001)
    assert relative_rms<.015,float(relative_rms)


@pytest.mark.parametrize('alpha',[0.,.5,1.])
@pytest.mark.parametrize('has_bias',[False,True])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('shape',[(2,3,37,37,128),(1,3,3,1057,128)])
def test_fused_forward_and_every_gradient(alpha,has_bias,masked,shape):
    b,h,m,n,d=shape
    torch.manual_seed(928)
    q=torch.randn(b,h,m,d,device='cuda',dtype=torch.bfloat16).div(8).requires_grad_()
    k=torch.randn(b,h,n,d,device='cuda',dtype=torch.bfloat16).div(8).requires_grad_()
    v=torch.randn_like(k,requires_grad=True)
    scale=torch.tensor(-.7,device='cuda',requires_grad=True)
    bias=torch.tensor([-.2,.15,0.],device='cuda',requires_grad=True) if has_bias else None
    mask=torch.ones(b,n,device='cuda',dtype=torch.bool) if masked else None
    if masked:mask[:,::3]=False;mask[0,:min(4,n)]=False
    inputs=[q,k,v,scale]+([bias] if has_bias else [])
    # FP64 reference cannot inherit a TF32 setting from another test/model.
    clones=[x.detach().double().requires_grad_() for x in inputs]
    args=dict(divisor=2.,alpha=alpha,anchor=8.,attention_mask=mask)
    y=fused(q,k,v,scale=scale,bias=bias,**args)
    expected=reference(*clones[:3],scale=clones[3],bias=clones[4] if has_bias else None,**args)
    do=torch.randn_like(y)
    y.backward(do);expected.backward(do.double())
    oracle=precision_oracle(q,k,v,do,scale,bias,**args)
    names=['dq','dk','dv','dscale']+(['dbias'] if has_bias else [])
    expected_fields={'output':expected,**{name:t.grad for name,t in zip(names,clones)}}
    actual_fields={'output':y,**{name:t.grad for name,t in zip(names,inputs)}}
    for name,want in expected_fields.items():
        # Independently verify the oracle's ideal math against autograd.
        torch.testing.assert_close(oracle[name].ideal,want,rtol=1e-10,atol=1e-10)
        assert_precision(name,actual_fields[name],oracle[name])


def test_zero_scale_bias_only_backward_and_all_masked_rows():
    q=torch.zeros(1,3,7,32,device='cuda',dtype=torch.float32)
    k=torch.ones_like(q);v=torch.ones_like(q)
    bias=torch.tensor([-1.,.5,-.25],device='cuda',requires_grad=True)
    mask=torch.zeros(1,7,device='cuda',dtype=torch.bool)
    out=fused(q,k,v,scale=0.,bias=bias,alpha=1,anchor=2,attention_mask=mask)
    out.sum().backward()
    assert torch.count_nonzero(out)==0 and torch.count_nonzero(bias.grad)==0
    mask.fill_(True);bias.grad=None
    out=fused(q,k,v,scale=0.,bias=bias,alpha=.5,anchor=2,attention_mask=mask)
    expected=reference(q,k,v,scale=0.,bias=bias,alpha=.5,anchor=2,attention_mask=mask)
    expected_grad=torch.autograd.grad(expected.sum(),bias)[0]
    out.sum().backward()
    torch.testing.assert_close(out,expected,atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(bias.grad,expected_grad,atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_long_query_gradient_in_fp32_against_fp64_autograd(alpha):
    """Tight arithmetic check with the intermediate BF16 casts absent."""
    torch.manual_seed(928)
    q=torch.randn(1,3,3,128,device='cuda',dtype=torch.bfloat16).div(8).float().requires_grad_()
    k=torch.randn(1,3,1057,128,device='cuda',dtype=torch.bfloat16).div(8).float().requires_grad_()
    v=torch.randn(k.shape,device='cuda',dtype=torch.bfloat16).float().requires_grad_()
    scale=torch.tensor(-.7,device='cuda',requires_grad=True)
    bias=torch.tensor([-.2,.15,0.],device='cuda',requires_grad=True)
    inputs=[q,k,v,scale,bias]
    refs=[x.detach().double().requires_grad_() for x in inputs]
    args=dict(divisor=2.,alpha=alpha,anchor=8.)
    y=fused(q,k,v,scale=scale,bias=bias,**args)
    want=reference(*refs[:3],scale=refs[3],bias=refs[4],**args)
    do=torch.randn_like(y)
    y.backward(do);want.backward(do.double())
    for name,got,expected in [('output',y,want),*[(n,x.grad,r.grad) for n,x,r in zip(['dq','dk','dv','dscale','dbias'],inputs,refs)]]:
        torch.testing.assert_close(got.double(),expected,atol=2e-4,rtol=2e-4,msg=lambda detail:f'{name}: {detail}')


def test_prefix_and_chunked_cache_agree_with_full_sequence():
    torch.manual_seed(12)
    q,k,v=[torch.randn(1,3,529,64,device='cuda',dtype=torch.bfloat16)/4 for _ in range(3)]
    bias=torch.tensor([.1,-.2,0.],device='cuda')
    args=dict(scale=.8,bias=bias,alpha=.5,anchor=512.)
    full=fused(q,k,v,**args)
    for start,end in [(0,257),(257,512),(512,529)]:
        actual=fused(q[:,:,start:end],k[:,:,:end],v[:,:,:end],**args)
        accurate(actual,full[:,:,start:end])


@pytest.mark.parametrize('variant',['relu2_bias','relu2_scale_half','relu2_scale_one','relu2_bias_scale_half','relu2_bias_scale_one'])
@pytest.mark.parametrize('norm',['none','capped'])
def test_hf_fused_muon_training_step(variant,norm):
    from hf_model.modeling_comparison import ComparisonForCausalLM
    from hf_model.configuration_comparison import ComparisonConfig
    from experiment.optimizers import optimizer_for
    from experiment.common import token_loss, backend_labels
    from context_sweep.recipe import ROOT, read
    torch.manual_seed(92)
    model=ComparisonForCausalLM(ComparisonConfig(vocab_size=64,hidden_size=96,intermediate_size=192,
        num_hidden_layers=2,num_attention_heads=3,num_key_value_heads=3,variant=variant,output_norm=norm,
        causal_length_anchor=8.,use_qk_norm=True,use_qk_norm_scale=True,qk_norm_scale_init=6.,
        use_rotary_embeddings=True,use_absolute_position_embeddings=False,require_fused=True)).cuda().train()
    optimizer=optimizer_for(model,'muon',read(ROOT/'configs/context_ablation_50m.json')['optimizer_recipes']['muon'],'cuda')
    x=torch.randint(0,64,(2,37),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=token_loss(model,x,torch.roll(x,-1,-1),16)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert backend_labels(model)==['triton_fused']
    optimizer.step()
    if model.config.learned_subtractive_bias:
        assert any(torch.count_nonzero(layer.attn.subtractive_bias)>0 for layer in model.model.layers)
