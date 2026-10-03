"""No GPU required for planner, capacity decisions, report honesty and isolation."""
import copy
import math
from decimal import Decimal
from pathlib import Path
import pytest
from context_sweep.recipe import ROOT, read, write, build
from depth_sweep.planning import depth_grid, depth_spec, find_limit, probe_cells
from depth_sweep.report import completed_mean, gather, refresh


@pytest.mark.parametrize('maximum',[25,32,63,90,100,173])
def test_grid_endpoints(maximum):
    grid=depth_grid(17,maximum)
    assert grid[0]==17 and grid[-1]==int(Decimal('.7')*maximum)
    assert grid==sorted(set(grid)) and len(grid)<=5


def test_too_small_or_unbounded_is_not_reported_as_maximum():
    with pytest.raises(ValueError,match='below'):
        depth_grid(17,20)
    with pytest.raises(ValueError,match='increase the cap'):
        find_limit(17,128,lambda _:True)
    with pytest.raises(ValueError,match='Starting depth'):
        find_limit(17,128,lambda _:False)


@pytest.mark.parametrize('limit',[17,18,31,32,33,67,127])
def test_search_finds_neighbor(limit):
    seen=[]
    def fits(n):
        seen.append(n)
        return n<=limit
    assert find_limit(17,128,fits)==(limit,limit+1)
    assert limit in seen and limit+1 in seen


def test_compiler_error_not_capacity_failure():
    def bad(n):
        if n==17:return True
        raise RuntimeError('compile error')
    with pytest.raises(RuntimeError,match='compile error'):
        find_limit(17,128,bad)


def test_only_depth_and_context_selection_change():
    base=read(ROOT/'configs/context_ablation_50m.json')
    before=copy.deepcopy(base)
    spec=depth_spec(base,40)
    assert base==before
    assert spec['contexts']==[512,1024]
    assert spec['model']['num_hidden_layers']==40
    for k in base:
        if k not in ('name','contexts','model'):
            assert spec[k]==base[k]
    assert {k:v for k,v in spec['model'].items() if k!='num_hidden_layers'}=={
        k:v for k,v in base['model'].items() if k!='num_hidden_layers'}
    cells=probe_cells(base,40)
    assert len(cells)==32
    assert {c['context'] for c,_ in cells}=={512,1024}
    assert {c['target_tokens'] for c,_ in cells}=={100007936}
    assert {cfg['train']['micro_batch_size'] for _,cfg in cells}=={4,8}
    assert {cfg['train']['gradient_accumulation'] for _,cfg in cells}=={8}


@pytest.mark.parametrize('bad',[{}, {'completed_budget':False},{'initialization_ok':False},
                                {'best_validation_loss':None},{'best_validation_loss':float('nan')}])
def test_missing_failed_or_partial_is_not_zero(bad):
    row=dict(seed=0,completed_budget=True,initialization_ok=True,best_validation_loss=3.)
    if not bad:
        assert math.isnan(completed_mean([], 'best_validation_loss',[0]))
    else:
        row.update(bad)
        assert math.isnan(completed_mean([row],'best_validation_loss',[0]))


def test_complete_all_seeds_only():
    rows=[dict(seed=s,completed_budget=True,initialization_ok=True,best_validation_loss=3.+s) for s in (0,1)]
    assert completed_mean(rows,'best_validation_loss',[0,1])==3.5
    assert math.isnan(completed_mean(rows[:1],'best_validation_loss',[0,1]))


def test_pending_file_report_does_not_need_http_or_torch(tmp_path):
    base=read(ROOT/'configs/context_ablation_50m.json')
    write(tmp_path/'depth_plan.json',dict(base=base,layers=[17,24],measured_common_limit=35,
          endpoint_fraction=.7,tokens_per_run=100007936))
    data=refresh(tmp_path,draw=False)
    assert len(data['runs'])==64
    assert all(r['status']=='pending' for r in data['runs'])
    assert (tmp_path/'live/report.md').is_file()
    assert (tmp_path/'live/depth_runs.csv').is_file()
    assert '0/64' in (tmp_path/'live/report.md').read_text()
    assert not list(tmp_path.glob('L*/plan.json'))  # Reporting never creates training cohorts.


def test_no_plan_report_during_probe(tmp_path):
    write(tmp_path/'depth_status.json',dict(phase='capacity_probe'))
    data=refresh(tmp_path,draw=False)
    assert data['runs']==[]
    assert 'capacity_probe' in (tmp_path/'live/report.md').read_text()


def test_data_cannot_change_between_depths(tmp_path):
    from depth_sweep.launch import bind_data
    data=tmp_path/'data'
    write(data/'manifest.json',{'tokens':100})
    bind_data(tmp_path,data)
    bind_data(tmp_path,data)
    write(data/'manifest.json',{'tokens':200})
    with pytest.raises(ValueError,match='data changed'):
        bind_data(tmp_path,data)


def test_calibration_freezes_and_checks_every_grid_cell(tmp_path,monkeypatch):
    import argparse
    import torch
    import depth_sweep.launch as launch
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(23*2**30,24*2**30))
    base=read(ROOT/'configs/context_ablation_50m.json')
    args=argparse.Namespace(device='cuda:0',reserve_gib=1.,max_search_layers=64,points=5,fraction=.7,disk_budget_gb=1000,disk_reserve_gb=0)
    monkeypatch.setattr(launch,'available_budget',lambda *a:1000*10**9)
    calls=[]
    def fake(command,log,cancelled,grace):
        request=read(command[command.index('--request')+1])
        cell=request['cell']
        depth=request['config']['model']['num_hidden_layers']
        # Simulate the worst arm limiting the common maximum to 40 layers.
        limit=40 if cell['context']==1024 and cell['variant']=='relu2_bias_scale_one' and cell['output_norm']=='capped' else 50
        write(command[command.index('--result')+1],dict(status='fit' if depth<=limit else 'budget_exceeded'))
        calls.append((depth,cell['name']))
        return dict(status='ok')
    monkeypatch.setattr(launch,'supervise',fake)
    plan=launch.calibrate(base,args,tmp_path,{'test':True},lambda:False)
    assert plan['measured_common_limit']==40 and plan['first_failing_depth']==41
    assert plan['layers']==[17,20,23,25,28]
    for depth in plan['layers']:
        assert {name for d,name in calls if d==depth}=={c['name'] for c,_ in probe_cells(base,depth)}
        assert read(tmp_path/'presets'/f'layers_{depth:04d}.json')==depth_spec(base,depth)
    count=len(calls)
    assert launch.calibrate(base,args,tmp_path,{'test':True},lambda:False)==plan
    assert len(calls)==count  # Frozen plan reused, not silently re-probed.
    with pytest.raises(ValueError,match='settings'):
        launch.calibrate(base,args,tmp_path,{'test':False},lambda:False)


def test_calibration_missing_worker_result_is_not_oom(tmp_path,monkeypatch):
    import argparse
    import torch
    import depth_sweep.launch as launch
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(23*2**30,24*2**30))
    monkeypatch.setattr(launch,'supervise',lambda *a,**k:dict(status='failed'))
    args=argparse.Namespace(device='cuda:0',reserve_gib=1.,max_search_layers=64,points=5,fraction=.7,disk_budget_gb=1000,disk_reserve_gb=0)
    with pytest.raises(RuntimeError,match='non-capacity'):
        launch.calibrate(read(ROOT/'configs/context_ablation_50m.json'),args,tmp_path,{},lambda:False)
    assert not (tmp_path/'depth_plan.json').exists()
