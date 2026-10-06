"""Independent recurrence, causality, cache, HF and optimizer checks."""
import copy
import math
import pytest
import torch
from transformers import AutoModelForCausalLM
from context_sweep.recipe import ROOT, build, read
from experiment.common import make_model, token_loss
from experiment.optimizers import partition
from hf_model.kda_attention import kda_reference, triton_ops


def config(norm="none"):
    return build(read(ROOT / "configs/smoke_sweep.json"))["configs"]["16_"+norm]


def test_recurrence_against_explicit_transition_products_and_gradcheck():
    torch.manual_seed(7)
    b,t,h,d=1,4,2,3
    q,k,v=[torch.randn(b,t,h,d,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    g=(-torch.rand_like(q)).requires_grad_()
    beta=torch.rand(b,t,h,dtype=torch.float64,requires_grad=True)
    initial=torch.randn(b,h,d,d,dtype=torch.float64,requires_grad=True)
    got,state=kda_reference(q,k,v,g,beta,initial)
    expected=[]
    for end in range(t):
        # Expand S_end = (product T) S_initial + sum (later T products) B.
        transitions=[];writes=[]
        for i in range(end+1):
            kk=k[:,i].unsqueeze(-1)@k[:,i].unsqueeze(-2)
            transitions.append((torch.eye(d)-beta[:,i,:,None,None]*kk)@torch.diag_embed(g[:,i].exp()))
            writes.append(beta[:,i,:,None,None]*(k[:,i].unsqueeze(-1)@v[:,i].unsqueeze(-2)))
        expanded=initial
        for tr in transitions:expanded=tr@expanded
        for j in range(end+1):
            term=writes[j]
            for tr in transitions[j+1:]:term=tr@term
            expanded=expanded+term
        expected.append((q[:,end].unsqueeze(-2)@expanded).squeeze(-2)/math.sqrt(d))
    torch.testing.assert_close(got,torch.stack(expected,1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(state,expanded,rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradcheck(lambda *args:kda_reference(*args),(q,k,v,g,beta,initial),fast_mode=True)


@pytest.mark.parametrize("norm",["none","capped"])
def test_kda_prefix_cache_padding_and_generation(norm):
    m,_=make_model(config(norm),'kda',3,'cpu');m.eval()
    torch.manual_seed(1);x=torch.randint(2,100,(2,13))
    with torch.no_grad():
        all_logits=m(x,use_cache=False).logits
        first=m(x[:,:7],use_cache=True)
        cache_copy=tuple(tuple(t.clone() for t in s) for s in first.past_key_values)
        later=m(x[:,7:],past_key_values=first.past_key_values,use_cache=True)
        torch.testing.assert_close(all_logits[:,:7],first.logits,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(all_logits[:,7:],later.logits,rtol=2e-5,atol=2e-6)
        for old,new in zip(cache_copy,first.past_key_values):
            for a,b in zip(old,new):torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Token-at-a-time decoding, including convolution history.
        state=None;pieces=[]
        for i in range(x.shape[1]):
            y=m(x[:,i:i+1],past_key_values=state,use_cache=True);state=y.past_key_values;pieces.append(y.logits)
        torch.testing.assert_close(torch.cat(pieces,1),all_logits,rtol=2e-5,atol=2e-6)
        assert state[0][0].shape==(2,3,8,8)
        assert state[0][1].shape==(2,24,4)
        # Skip left/right padding and holes, rather than advancing the state.
        mask=torch.tensor([[0,0,1,1,0,1,1,1,1,1,1,0,0],[1]*13])
        padded=m(x,attention_mask=mask,use_cache=True)
        for b in range(2):
            keep=mask[b].bool();reference=m(x[b:b+1,keep],use_cache=True)
            torch.testing.assert_close(padded.logits[b,keep],reference.logits[0],rtol=2e-5,atol=2e-6)
            for actual,want in zip(padded.past_key_values[0],reference.past_key_values[0]):
                torch.testing.assert_close(actual[b:b+1],want,rtol=2e-5,atol=2e-6)
        empty=m(x,attention_mask=torch.zeros_like(x),use_cache=True)
        assert all(torch.count_nonzero(t)==0 for layer in empty.past_key_values for t in layer)
        reordered=m._reorder_cache(state,torch.tensor([1,0]))
        for a,b in zip(reordered[0],state[0]):torch.testing.assert_close(a,b.flip(0))
        assert m.generate(x[:1,:3],max_new_tokens=3,eos_token_id=None).shape==(1,6)


def test_kda_hf_roundtrip_checkpointing_and_common_initial_weights(tmp_path):
    cfg=config('capped');m,_=make_model(cfg,'kda',0,'cpu');other,_=make_model(cfg,'relu2',0,'cpu')
    common=dict(other.named_parameters())
    for name,p in m.named_parameters():
        if name in common:torch.testing.assert_close(p,common[name],rtol=0,atol=0)
    assert not m.config.use_rotary_embeddings and not m.config.use_qk_norm_scale
    m.eval();x=torch.arange(16).reshape(2,8)
    expected=m(x).logits
    m.save_pretrained(tmp_path)
    assert (tmp_path/'kda_attention.py').is_file()
    restored=AutoModelForCausalLM.from_pretrained(tmp_path,trust_remote_code=True)
    torch.testing.assert_close(restored(x).logits,expected,rtol=0,atol=0)
    eager=copy.deepcopy(m).train();m.train();m.model.gradient_checkpointing=True
    for model in (eager,m):token_loss(model,x,torch.roll(x,-1,-1),8).backward()
    for (n,a),(n2,b) in zip(eager.named_parameters(),m.named_parameters()):
        assert n==n2 and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        assert torch.isfinite(b.grad).all()
    assigned={n:g for g,ps in partition(m,'muon').items() for n,_ in ps}
    for n in assigned:
        if n.endswith(('A_log','dt_bias','head_norm_weight','gain','.bias')):assert assigned[n]=='adamw_no_decay'
        if '_conv.weight' in n or '.f_proj.' in n or '.b_proj.' in n or '.g_proj.' in n and n.endswith('.weight'):
            assert assigned[n]=='muon_decoder'


def test_original_arms_match_previous_release():
    # Generated by running the previous, unchanged delivered project on CPU.
    gold=read(ROOT/'validation/baseline_golden.json')
    for variant in ('softmax_sdpa','relu2','linear16'):
        for norm in ('none','capped'):
            m,initial=make_model(config(norm),variant,0,'cpu');m.eval()
            with torch.no_grad():logits=m(torch.arange(16).reshape(2,8),use_cache=False).logits
            expected=gold[variant+'_'+norm]
            assert initial==expected['initial_hash']
            # CPU BLAS/thread/version choices can alter the last few FP32 bits.
            # Parameter initialization is exact; numerical outputs must agree.
            torch.testing.assert_close(logits,torch.tensor(expected['logits']),rtol=2e-6,atol=2e-7)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required for actual FLA kernels")
@pytest.mark.parametrize('length',[65,512,1024,2048])
def test_cuda_kda_forward_backward_and_recurrent_state_against_reference(length):
    from torch.nn import functional as F
    torch.manual_seed(12)
    b,h,d=1,3,128
    q,k,v,g=[(torch.randn(b,length,h,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_() for _ in range(4)]
    beta=torch.randn(b,length,h,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    a=torch.zeros(h,device='cuda',requires_grad=True)
    dt=torch.full((h*d,),-4.,device='cuda',requires_grad=True)
    state=torch.randn(b,h,d,d,device='cuda',requires_grad=True)*.01
    inputs=(q,k,v,g,beta,a,dt,state)
    ref=[x.detach().float().requires_grad_() for x in inputs]
    rq,rk,rv,rg,rb,ra,rd,rs=ref
    rq=rq*torch.rsqrt(rq.square().sum(-1,keepdim=True)+1e-6)
    rk=rk*torch.rsqrt(rk.square().sum(-1,keepdim=True)+1e-6)
    log_decay=-ra.exp()[None,None,:,None]*F.softplus(rg+rd.view(h,d))
    expected,expected_state=kda_reference(rq,rk,rv,log_decay,rb.sigmoid(),rs)
    kwargs=dict(q=q,k=k,v=v,g=g,beta=beta,A_log=a,dt_bias=dt,initial_state=state,
                output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,state_v_first=False)
    actual,final=triton_ops()[0](**kwargs)
    def close(a,b,tolerance):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        error=(a.float()-b.float()).square().mean().sqrt()
        rms=b.float().square().mean().sqrt()
        assert error<=1e-5+tolerance*rms, (float(error),float(rms))
    close(actual,expected,.02);close(final,expected_state,.02)
    upstream=torch.randn_like(actual);state_up=torch.randn_like(final)*.01
    grads=torch.autograd.grad((actual.float()*upstream).sum()+(final*state_up).sum(),inputs)
    ref_grads=torch.autograd.grad((expected*upstream).sum()+(expected_state*state_up).sum(),ref)
    for ga,gr in zip(grads,ref_grads):close(ga,gr,.05)
    with torch.no_grad():
        decoded,s=triton_ops()[1](**kwargs)
    close(decoded,expected,.02);close(s,expected_state,.02)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required for FLA module integration")
@pytest.mark.parametrize('norm',['none','capped'])
def test_cuda_full_kda_module_cache_and_cpu_reference(norm):
    cfg=config(norm)
    cfg['model'].update(hidden_size=384,intermediate_size=768,num_hidden_layers=1,max_position_embeddings=128)
    m,_=make_model(cfg,'kda',0,'cuda:0');m.eval()
    cpu=copy.deepcopy(m).cpu();cpu.config.require_fused=False
    x=torch.randint(2,100,(2,65),device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        full=m(x,use_cache=False).logits.float()
        first=m(x[:,:64],use_cache=True)
        last=m(x[:,64:],past_key_values=first.past_key_values,use_cache=True)
        cached=torch.cat((first.logits,last.logits),dim=1).float()
    with torch.no_grad():expected=cpu(x.cpu(),use_cache=False).logits.cuda()
    for actual in (full,cached):
        error=(actual-expected).square().mean().sqrt()
        assert error <= .002 + .03 * expected.square().mean().sqrt()
    assert m.model.layers[0].attn.last_backend=='fla_triton_kda_recurrent'
    # eval mode WITH gradients must select a backward-capable chunk operator.
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(x[:,:7],labels=x[:,:7],use_cache=False).loss
    loss.backward()
    assert m.model.layers[0].attn.last_backend=='fla_triton_kda_chunk'
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
