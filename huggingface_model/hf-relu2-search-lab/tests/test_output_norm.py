import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM
from context_sweep.recipe import ROOT, read, write, build, scheduled_cells
from context_sweep.report import collect, best
from experiment.common import make_model, parameter_counts, state_digest, token_loss
from experiment.optimizers import partition, describe_partition, optimizer_for
from hf_model.configuration_comparison import ComparisonConfig
from hf_model.modeling_comparison import ComparisonForCausalLM
from hf_model.output_norm import CappedHypersphereNorm


def preset(smoke=True):
    return build(read(ROOT / "configs" / ("smoke_sweep.json" if smoke else "sweep_50m.json")))


def invoke(module, *args, timeout=600):
    result = subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT,
        env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout[-16000:]
    return result.stdout


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_identity_inside_radius_and_projection_outside(dtype):
    norm = CappedHypersphereNorm(4)
    x = torch.tensor([[0,0,0,0],[.1,-.2,.3,-.4],[1,1,1,1],[3,4,0,0]], dtype=dtype)
    y = norm(x)
    assert y.dtype == dtype and y.shape == x.shape
    torch.testing.assert_close(y[:3], x[:3], rtol=0, atol=0)
    expected = torch.tensor([1.2,1.6,0,0], dtype=dtype)
    torch.testing.assert_close(y[-1], expected, rtol=0.005, atol=0.003)
    assert torch.all(norm.gain == 0)
    # The no-norm arm returns the original tensor and owns no gain parameters.
    off = CappedHypersphereNorm(4, False)
    assert off(x) is x and list(off.parameters()) == []


@pytest.mark.parametrize("values", [[.1,-.2,.3,.4], [3.,4.,1.,-2.]])
def test_jacobian_and_gain_gradient_against_closed_form(values):
    m = CappedHypersphereNorm(4).double()
    with torch.no_grad(): m.gain.copy_(torch.tensor([.1,-.2,.3,.4], dtype=torch.float64))
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    r = float(x.detach().norm())
    gamma = 1 + m.gain.detach()
    jac = torch.autograd.functional.jacobian(m, x)
    reference = torch.eye(4,dtype=x.dtype) if r < 2 else (2/r)*(torch.eye(4,dtype=x.dtype)-torch.outer(x.detach(),x.detach())/r**2)
    reference = gamma[:,None] * reference
    torch.testing.assert_close(jac, reference, atol=1e-12, rtol=1e-12)
    m(x).sum().backward()
    torch.testing.assert_close(m.gain.grad, x.detach()/max(1,r/2), atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(m,(x,),eps=1e-6,atol=1e-5)


def test_boundary_and_zero_gradients_are_finite_and_identity_side():
    m = CappedHypersphereNorm(4).double()
    for x in [torch.zeros(4,dtype=torch.float64),torch.ones(4,dtype=torch.float64)]:
        jac = torch.autograd.functional.jacobian(m,x)
        torch.testing.assert_close(jac,torch.eye(4,dtype=x.dtype),atol=0,rtol=0)


def test_gain_is_per_channel_applied_after_cap_and_can_exceed_radius():
    m = CappedHypersphereNorm(4)
    with torch.no_grad(): m.gain.copy_(torch.tensor([1.,0.,-.5,.25]))
    x = torch.tensor([[.1,.2,.3,.4],[4.,0,0,0]])
    y = m(x)
    torch.testing.assert_close(y[0],x[0]*torch.tensor([2.,1.,.5,1.25]))
    torch.testing.assert_close(y[1],torch.tensor([4.,0,0,0]))
    assert y[1].norm()>math.sqrt(4)


def test_diagnostics_count_tokens_do_not_change_outputs_or_keep_graphs():
    for enabled in [False,True]:
        m=CappedHypersphereNorm(4,enabled)
        x=torch.tensor([[[0.,0,0,0],[4.,0,0,0]],[[.1,.2,.3,.4],[3.,4.,0,0]]],requires_grad=True)
        expected=m(x)
        m.start_diagnostics(); got=m(x)
        assert all(not value.requires_grad for value in m._stats.values() if torch.is_tensor(value))
        row=m.stop_diagnostics()
        torch.testing.assert_close(got,expected,atol=0,rtol=0)
        assert row['vectors']==4 and row['fraction_above_radius']==.5
        assert row['max_input_rms']==2.5
        assert row['max_output_rms']==(1.0 if enabled else 2.5)
        assert m._record is False and m._stats is None


def test_full_plan_counts_and_optimizer_routing():
    plan=preset(False)
    assert (plan['run_count'],plan['steps'],plan['tokens_per_run'])==(24,3052,100007936)
    assert plan['total_training_tokens']==2400190464
    assert plan['contexts']==[512,1024,2048] and plan['optimizers']==['muon']
    assert plan['output_norms']==['none','capped'] and len(plan['series'])==8
    assert len({c['name'] for c in plan['cells']})==24
    for key,cfg in plan['configs'].items():
        t=cfg['train']
        assert t['sequence_length']*t['micro_batch_size']*t['gradient_accumulation']==32768
        assert t['eval_batches']*t['sequence_length']*t['micro_batch_size']==131072
        assert t['common_validation_context']==512
    for norm in plan['output_norms']:
        for variant in plan['variants']:
            with torch.device('meta'):
                m=ComparisonForCausalLM(ComparisonConfig(**plan['spec']['model'],variant=variant,output_norm=norm))
            assert parameter_counts(m)['total']==(52866739 if variant=='kda' else 49411217)+(13056 if norm=='capped' else 0)
            info=describe_partition(m,'muon',plan['spec']['optimizer_recipes']['muon'])
            groups={g['name']:g for g in info['groups']}
            assert groups['muon_decoder']['parameter_count']==(33521280 if variant=='kda' else 30081024)
            assert groups['adamw_no_decay']['parameter_count']==(28723 if variant=='kda' else 13457)+(13056 if norm=='capped' else 0)
            assigned={name:group for group,ps in partition(m,'muon').items() for name,_ in ps}
            assert all(assigned[n]=='adamw_no_decay' for n in assigned if n.endswith('output_norm.gain'))
    printed=json.loads(invoke('context_sweep.run','--data','unused','--dry-run'))
    assert printed['time_limit'] is None and printed['selected_run_count']==24
    selected=scheduled_cells(plan,contexts=[1024],variants=['linear16'],output_norms=['capped'])
    assert len(selected)==1 and selected[0]['config_key']=='1024_capped'


def test_shared_initialization_is_exactly_matched():
    plan=preset(); hashes={}; condition_hashes={}
    for cfg in plan['configs'].values():
        for variant in plan['variants']:
            m,h=make_model(cfg,variant,0,'cpu')
            family='kda' if variant=='kda' else 'standard'
            hashes.setdefault(family,set()).add(state_digest(m,shared_only=True))
            condition_hashes.setdefault((family,cfg['model']['output_norm']),set()).add(h)
    assert all(len(v)==1 for v in hashes.values()) and all(len(v)==1 for v in condition_hashes.values())
    assert all(condition_hashes[(f,'none')]!=condition_hashes[(f,'capped')] for f in hashes)


@pytest.mark.parametrize('variant',['softmax_sdpa','relu2','linear16','kda'])
def test_both_branches_are_normed_before_residual_and_off_is_exact(variant):
    for mode in ['none','capped']:
        cfg=preset()['configs']['16_'+mode]
        m,_=make_model(cfg,variant,0,'cpu'); b=m.model.layers[0]
        x=torch.randn(2,8,24)*2
        attn,_=b.attn(b.attn_norm(x))
        cap=lambda y:y if mode=='none' else y/y.square().mean(-1,keepdim=True).clamp_min(1).sqrt()
        residual=x+cap(attn)
        expected=residual+cap(b.mlp(b.mlp_norm(residual)))
        actual,_=b(x)
        torch.testing.assert_close(actual,expected,atol=2e-6 if mode=='capped' else 0,rtol=2e-6 if mode=='capped' else 0)


def test_hf_checkpoint_roundtrip_custom_gains_and_gradient_checkpointing(tmp_path):
    cfg=preset()['configs']['16_capped']
    model,_=make_model(cfg,'linear16',0,'cpu')
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.endswith('output_norm.gain'):p.fill_(.2)
    model.eval(); x=torch.arange(16).reshape(2,8)
    expected=model(x).logits
    model.save_pretrained(tmp_path)
    assert (tmp_path/'output_norm.py').is_file()
    restored=AutoModelForCausalLM.from_pretrained(tmp_path,trust_remote_code=True)
    assert restored.config.output_norm=='capped'
    torch.testing.assert_close(restored(x).logits,expected,atol=0,rtol=0)
    model.train();model.model.gradient_checkpointing=True
    loss=token_loss(model,x,torch.roll(x,-1,-1),8);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    from experiment.smoke import create_fixture
    root=tmp_path_factory.mktemp('output-norm-smoke')
    create_fixture(root)
    invoke('context_sweep.prepare','--local-jsonl',root/'documents.jsonl','--tokenizer',root/'tokenizer_source',
           '--output',root/'data','--train-tokens',10000,'--validation-tokens',3000,'--validation-permyriad',2500)
    return root


def compare_state(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:compare_state(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare_state(x,y)
    else:assert a==b


def test_twenty_four_run_smoke_and_exact_resume_for_both_conditions(prepared):
    out=prepared/'sweep'
    args=['--preset',ROOT/'configs/smoke_sweep.json','--data',prepared/'data','--output',out,'--device','cpu','--no-monitor']
    invoke('context_sweep.run',*args)
    state=read(out/'sweep_state.json')
    assert state['status']=='complete' and state['time_limit'] is None and len(state['jobs'])==48
    assert all(j['status']=='ok' for j in state['jobs'])
    data=collect(out)
    assert data['completed']==data['total']==24 and not data['initialization_mismatch_seeds']
    assert all(r['tokens']==384 and r['best_eligible_checks']==3 for r in data['runs'])
    assert len(data['paired_deltas'])==12 and all(r['complete_pair'] for r in data['paired_deltas'])
    assert len(data['output_norm_diagnostics'])==24*4*4  # 4 checks × 4 branch modules
    for name in ['best_validation_vs_context.png','common_context_validation.svg','output_norm_delta.pdf','output_norm_diagnostics.csv','paired_deltas.csv','index.html']:
        assert (out/'live'/name).stat().st_size>100
    invoke('context_sweep.run',*args,'--resume')
    assert len(read(out/'sweep_state.json')['jobs'])==48
    for norm in ['none','capped']:
        direct=prepared/('direct_'+norm)
        invoke('experiment.train','--config',out/f'configs/context_16_{norm}.json','--data',prepared/'data',
            '--output',direct,'--variant','kda','--optimizer','muon','--device','cpu')
        staged=out/f'ctx16_kda_muon_norm-{norm}_seed0'
        ds,ss=read(direct/'train_summary.json'),read(staged/'train_summary.json')
        for field in ['final_state_sha256','best_validation','best_common_validation']:
            assert ds[field]==ss[field]
        a=torch.load(direct/read(direct/'latest.json')['checkpoint']/'training_state.pt',weights_only=False)
        b=torch.load(staged/read(staged/'latest.json')['checkpoint']/'training_state.pt',weights_only=False)
        compare_state(a['optimizer'],b['optimizer'])
        assert set(a['optimizer']['states'])=={'muon','adamw'}
    (staged/'train_summary.json').unlink()
    invoke('context_sweep.run',*args,'--resume')
    assert read(staged/'train_summary.json')['status']=='complete'
    assert len(read(out/'sweep_state.json')['jobs'])==49


def test_missing_failed_and_incomplete_pairs_are_never_zero(tmp_path):
    plan=preset();write(tmp_path/'plan.json',plan)
    cell=plan['cells'][0];run=tmp_path/cell['name'];run.mkdir()
    (run/'metrics.jsonl').write_text(json.dumps(dict(kind='validation',tokens=128,step=2,loss=4.,common_loss=4.,eligible_for_best=True))+'\n')
    write(tmp_path/'sweep_state.json',dict(jobs=[dict(run=cell['name'],status='failed',error='injected')]))
    data=collect(tmp_path)
    assert data['completed']==0 and data['runs'][0]['status']=='failed'
    assert data['runs'][0]['best_validation_loss']==4.
    assert all(r['best_validation_loss'] is None for r in data['runs'][1:])
    assert all(not r['complete_pair'] and r['delta_best_validation_loss'] is None for r in data['paired_deltas'])
    assert best([dict(tokens=1,loss=0.,eligible_for_best=False)],'loss',384) is None


def test_attention_kernel_sources_remain_unchanged():
    hashes=read(ROOT/'validation/original_kernel_hashes.json')
    for name,sha in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
