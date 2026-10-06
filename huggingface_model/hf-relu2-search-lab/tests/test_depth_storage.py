from pathlib import Path
import shutil
import pytest
import torch
from safetensors.torch import save_file
from context_sweep.recipe import ROOT, read, write, build
from depth_sweep.planning import depth_spec
from depth_sweep.storage import (estimate,choose_grid,available_budget,check_job_space,
                                 retire_completed_checkpoint,GB)


def base():
    s=read(ROOT/'configs/context_ablation_50m.json')
    s['model']['num_hidden_layers']=16
    return s


def counter(spec,depth):
    return {(v,n):depth*10**6 for v in spec['variants'] for n in spec['output_norms']}


def test_disk_clamps_three_distinct_depths():
    spec=base()
    budget=estimate(spec,[16,23,30],counter)['estimated_peak_bytes']
    grid,report=choose_grid(spec,100,.7,3,budget,counter)
    assert grid==[16,23,30] and report['disk_limited']
    assert report['estimated_peak_bytes']<=budget
    assert report['vram_endpoint']==70


def test_minimum_cannot_fit_no_silent_pruning():
    with pytest.raises(ValueError,match='No models were deleted'):
        choose_grid(base(),100,.7,3,GB,counter)


def test_vram_limit_controls_when_disk_ample():
    grid,report=choose_grid(base(),100,.7,3,100*GB,counter)
    assert grid==[16,43,70] and not report['disk_limited']


def test_disk_checks_use_actual_free_and_decimal_gb(monkeypatch,tmp_path):
    monkeypatch.setattr(shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(100*GB,90*GB,10*GB))
    assert available_budget(tmp_path,30,2)==8*GB
    with pytest.raises(RuntimeError,match='Low disk'):
        check_job_space(tmp_path,400_000_000)


def fixture(root,complete=True):
    run=root/'run'
    cp=run/'checkpoint-00000001'
    final=run/'final'
    cp.mkdir(parents=True);final.mkdir()
    tensor={'w':torch.arange(4,dtype=torch.float32).reshape(2,2)}
    save_file(tensor,cp/'model.safetensors')
    save_file(tensor,final/'model.safetensors')
    write(final/'config.json',{})
    (cp/'training_state.pt').write_bytes(b'optimizer')
    write(run/'latest.json',dict(checkpoint=cp.name))
    write(run/'run.json',dict(actual_token_budget=100))
    write(cp/'progress.json',dict(cursor=100 if complete else 50))
    write(run/'train_summary.json',dict(status='complete' if complete else 'paused',tokens=100 if complete else 50))
    return run,cp,final


def test_final_weights_kept_optimizer_retired_idempotently(tmp_path):
    run,cp,final=fixture(tmp_path)
    content=(final/'model.safetensors').read_bytes()
    retire_completed_checkpoint(run)
    assert (final/'model.safetensors').read_bytes()==content
    assert (cp/'progress.json').exists()
    assert not (cp/'training_state.pt').exists()
    assert (cp/'model.safetensors').samefile(final/'model.safetensors')
    assert read(run/'storage_retention.json')['exact_optimizer_resume'] is False
    retire_completed_checkpoint(run)


@pytest.mark.parametrize('fault',['paused','missing','wrong_weights','corrupt','symlink'])
def test_retirement_refuses_bad_or_partial_exports(tmp_path,fault):
    run,cp,final=fixture(tmp_path,complete=fault!='paused')
    if fault=='missing':(final/'model.safetensors').unlink()
    if fault=='wrong_weights':save_file({'w':torch.ones(2,2)},final/'model.safetensors')
    if fault=='corrupt':(final/'model.safetensors').write_bytes(b'bad')
    if fault=='symlink':
        (final/'model.safetensors').unlink()
        (final/'model.safetensors').symlink_to(cp/'model.safetensors')
    with pytest.raises(Exception):retire_completed_checkpoint(run)
    assert (cp/'training_state.pt').exists() and (cp/'model.safetensors').exists()


def test_changed_export_after_retirement_is_detected(tmp_path):
    run,cp,final=fixture(tmp_path)
    retire_completed_checkpoint(run)
    save_file({'w':torch.zeros(2,2)},final/'model.safetensors')
    with pytest.raises(ValueError,match='changed'):retire_completed_checkpoint(run)


def test_failed_hardlink_keeps_optimizer(tmp_path,monkeypatch):
    import os
    run,cp,final=fixture(tmp_path)
    def fail(*args):raise OSError('Hardlinks not supported')
    monkeypatch.setattr(os,'link',fail)
    with pytest.raises(OSError,match='Hardlinks'):retire_completed_checkpoint(run)
    assert (cp/'training_state.pt').exists()


def test_three_depth_default_plan_cpu_only():
    import json,subprocess,sys
    result=subprocess.run([sys.executable,'-m','depth_sweep.launch','plan'],cwd=ROOT,capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert plan['start_layers']==16 and plan['points']==3
    assert plan['maximum_run_count']==96 and plan['disk_budget_gb']==30


def test_real_completed_muon_runs_retire_then_skip_on_resume(tmp_path):
    import subprocess,sys
    from context_sweep.report import collect
    spec=read(ROOT/'configs/context_ablation_smoke.json')
    spec.update(contexts=[8],variants=['relu2_bias_scale_half'])
    preset=tmp_path/'preset.json'
    write(preset,spec)
    fixture_root=tmp_path/'smoke'
    subprocess.run([sys.executable,'-m','experiment.smoke','--preset',str(preset),
                    '--output',str(fixture_root)],cwd=ROOT,check=True,capture_output=True,text=True)
    out=fixture_root/'comparison'
    before=collect(out)
    assert before['completed']==before['total']==2
    for row in before['runs']:
        retire_completed_checkpoint(out/row['name'])
        from hf_model.modeling_comparison import ComparisonForCausalLM
        cp=out/row['name']/read(out/row['name']/'latest.json')['checkpoint']
        restored=ComparisonForCausalLM.from_pretrained(cp)
        assert restored.config.variant=='relu2_bias_scale_half'
    subprocess.run([sys.executable,'-m','context_sweep.run','--preset',str(preset),
                    '--data',str(fixture_root/'data'),'--output',str(out),'--device','cpu',
                    '--resume','--no-monitor'],cwd=ROOT,check=True,capture_output=True,text=True)
    after=collect(out)
    assert after['completed']==2
    assert [r['best_validation_loss'] for r in before['runs']]==[r['best_validation_loss'] for r in after['runs']]
    assert not list(out.glob('ctx*/checkpoint-*/training_state.pt'))
    assert len(list(out.glob('ctx*/final/model.safetensors')))==2
