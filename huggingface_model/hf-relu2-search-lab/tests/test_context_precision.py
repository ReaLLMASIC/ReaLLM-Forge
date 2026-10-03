"""The revised checker must accept expected rounding AND reject real corruption."""
import pytest
import torch
from diagnostics.context_precision import precision_oracle, assess, assert_precision, Prediction
from hf_model.context_attention import context_attention_reference


def fixture(seed=928,m=3,n=1057,d=128,dtype=torch.bfloat16,masked=False,bias=True):
    torch.manual_seed(seed)
    q=torch.randn(1,3,m,d,dtype=dtype)/8
    k=torch.randn(1,3,n,d,dtype=dtype)/8
    v=torch.randn_like(k);do=torch.randn_like(q)
    scale=torch.tensor(-.7)
    tau=torch.tensor([-.2,.15,0.]) if bias else None
    mask=torch.ones(1,n,dtype=torch.bool) if masked else None
    if mask is not None:mask[:,::3]=False
    return q,k,v,do,scale,tau,mask


def float32_emulation(q,k,v,do,scale,bias,alpha,mask,anchor=8.):
    """Separate dense implementation of the kernel's FP32 operations and casts."""
    Q,K,V,D=[x.float() for x in (q,k,v,do)]
    m,n=q.shape[-2],k.shape[-2]
    raw=Q@K.transpose(-1,-2)
    s=raw*scale
    if bias is not None:s=s-bias[None,:,None,None]
    allowed=(torch.arange(n)[None,:]<=torch.arange(m)[:,None]+n-m)[None,None]
    if mask is not None:allowed=allowed & mask[:,None,None,:]
    p=s.relu().masked_fill(~allowed,0)
    f=(allowed.sum(-1,keepdim=True).float()/anchor).clamp_min(1).pow(-alpha)
    w=(p*p*.5*f).to(q.dtype).float()
    ds=(D@V.transpose(-1,-2))*p*f
    dr=ds.to(q.dtype).float()
    out={'output':(w@V).to(q.dtype),'dq':((dr@K)*scale).to(q.dtype),
         'dk':((dr.transpose(-1,-2)@Q)*scale).to(q.dtype),
         'dv':(w.transpose(-1,-2)@D).to(q.dtype),'dscale':(ds*raw).sum()}
    if bias is not None:out['dbias']=-ds.sum((0,2,3))
    return out


@pytest.mark.parametrize('alpha',[0.,.5,1.])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_fp64_oracle_matches_independent_autograd(alpha,masked,bias):
    q,k,v,do,g,tau,mask=fixture(m=7,n=13,d=16,masked=masked,bias=bias)
    refs=[x.double().requires_grad_() for x in [q,k,v,g]+([tau] if bias else [])]
    y=context_attention_reference(*refs[:3],scale=refs[3],bias=refs[4] if bias else None,
        divisor=2.,alpha=alpha,anchor=8.,attention_mask=mask)
    y.backward(do.double())
    predictions=precision_oracle(q,k,v,do,g,tau,alpha=alpha,attention_mask=mask)
    fields={'output':y,**dict(zip(['dq','dk','dv','dscale','dbias'],[x.grad for x in refs]))}
    for name,want in fields.items():torch.testing.assert_close(predictions[name].ideal,want,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('alpha',[0.,.5,1.])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_fp32_arithmetic_and_casts_fall_inside_computed_bounds(alpha,masked,dtype):
    q,k,v,do,g,tau,mask=fixture(dtype=dtype,masked=masked)
    predictions=precision_oracle(q,k,v,do,g,tau,alpha=alpha,attention_mask=mask)
    observed=float32_emulation(q,k,v,do,g,tau,alpha,mask)
    for name,value in observed.items():assert_precision(name,value,predictions[name])


def test_original_near_cancelled_gradient_false_failure_is_reproduced():
    q,k,v,do,g,tau,mask=fixture()
    predictions=precision_oracle(q,k,v,do,g,tau)
    observed=float32_emulation(q,k,v,do,g,tau,0.,mask)
    report=assert_precision('dq',observed['dq'],predictions['dq'])
    assert report['strict_fp32_tolerance_violations']>=1
    assert report['relative_rms']<.005


@pytest.mark.parametrize('field',['output','dq','dk','dv','dscale','dbias'])
def test_isolated_corruption_is_rejected_even_with_small_global_error(field):
    q,k,v,do,g,tau,mask=fixture()
    p=precision_oracle(q,k,v,do,g,tau)[field]
    corrupted=p.rounded.clone()
    # Beyond the computed upper bound, independently of the input's magnitude.
    gap=(p.upper-p.lower).reshape(-1)
    index=int(gap.argmin())
    corrupted.reshape(-1)[index]=p.upper.reshape(-1)[index]+.25
    report=assess(field,corrupted,p)
    assert not report['passed'] and report['interval_violations']>=1
    if field in ('output','dq','dk','dv'):assert report['relative_rms']<.015


def test_aggregate_error_gate_is_not_replaced_by_rounding_bounds():
    ideal=torch.ones(20,dtype=torch.float64)
    deliberately_wide=Prediction(ideal,ideal,ideal-100,ideal+100)
    report=assess('dq',ideal*1.02,deliberately_wide)
    assert report['interval_violations']==0 and report['relative_rms']>.015 and not report['passed']


@pytest.mark.parametrize('fault',['nan','sign','bias_omitted','wrong_length_exponent'])
def test_nonfinite_and_structural_gradient_errors_are_rejected(fault):
    q,k,v,do,g,tau,mask=fixture()
    expected=precision_oracle(q,k,v,do,g,tau,alpha=.5)
    if fault=='nan':
        value=expected['dq'].rounded.clone();value.flatten()[0]=float('nan');field='dq'
    elif fault=='sign':value=-expected['dbias'].rounded;field='dbias'
    elif fault=='bias_omitted':value=precision_oracle(q,k,v,do,g,None,alpha=.5)['dq'].rounded;field='dq'
    else:value=precision_oracle(q,k,v,do,g,tau,alpha=1.)['dq'].rounded;field='dq'
    assert not assess(field,value,expected[field])['passed']


def test_negative_bias_all_masked_row_remains_exactly_zero():
    q,k,v,do,g,tau,mask=fixture(m=3,n=7,d=16)
    p=precision_oracle(q,k,v,do,g,tau,attention_mask=torch.zeros(1,7,dtype=torch.bool))
    for prediction in p.values():
        assert torch.count_nonzero(prediction.ideal)==0
        assert torch.count_nonzero(prediction.lower)==0
        assert torch.count_nonzero(prediction.upper)==0
