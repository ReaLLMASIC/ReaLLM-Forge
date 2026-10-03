"""Real optimizer resume and diagnostic recovery, using tiny offline data."""
import os
import socket
import subprocess
import sys
import time
import pytest
import torch
from context_sweep.recipe import ROOT, read, write, build, progress
from context_sweep.report import collect
from diagnostics.sweep import complete_result


def invoke(module,*args):
    result=subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=ROOT,
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    assert result.returncode==0,result.stdout[-10000:]
    return result.stdout


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    from experiment.smoke import create_fixture
    root=tmp_path_factory.mktemp('context-pipeline')
    create_fixture(root)
    invoke('context_sweep.prepare','--local-jsonl',root/'documents.jsonl','--tokenizer',root/'tokenizer_source',
        '--output',root/'data','--train-tokens',10000,'--validation-tokens',3000,'--validation-permyriad',2500)
    return root/'data'


def same_state(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same_state(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same_state(x,y)
    else:assert a==b


@pytest.mark.parametrize('norm',['none','capped'])
def test_bias_and_length_scaling_resume_exactly_with_muon(data,tmp_path,norm):
    plan=build(read(ROOT/'configs/context_ablation_smoke.json'))
    write(tmp_path/'plan.json',plan)
    cell=next(c for c in plan['cells'] if c['context']==16 and c['variant']=='relu2_bias_scale_half' and c['output_norm']==norm)
    config=tmp_path/'config.json';write(config,plan['configs'][cell['config_key']])
    base=['--config',config,'--data',data,'--variant',cell['variant'],'--optimizer','muon','--device','cpu']
    staged=tmp_path/cell['name'];direct=tmp_path/'direct'
    invoke('experiment.train',*base,'--output',staged,'--stop-after',2)
    invoke('experiment.train',*base,'--output',staged,'--resume')
    invoke('experiment.train',*base,'--output',direct)
    a,b=read(staged/'train_summary.json'),read(direct/'train_summary.json')
    assert a['status']==b['status']=='complete'
    for field in ('final_state_sha256','best_validation','best_common_validation'):assert a[field]==b[field]
    cp,_=progress(tmp_path,cell['name'])
    full=direct/read(direct/'latest.json')['checkpoint']
    left=torch.load(cp/'training_state.pt',weights_only=False)
    right=torch.load(full/'training_state.pt',weights_only=False)
    same_state(left['optimizer'],right['optimizer'])
    same_state(left['torch_rng'],right['torch_rng'])
    records=collect(tmp_path)
    biases=[r['subtractive_bias'] for r in records['attention_parameters'] if r['tokens']==384]
    assert biases and all(x is not None for x in biases) and any(x!=0 for x in biases)
    assert records['completed']==1 and all(r['final_validation_loss'] is None for r in records['runs'] if not r['completed_budget'])
    # Probe, preserve complete diagnostics on rerun, recover a truncated result.
    args=['--root',tmp_path,'--data',data,'--run',cell['name'],'--device','cpu','--tokens',32]
    invoke('diagnostics.sweep',*args)
    state=read(tmp_path/'diagnostics_state.json')['runs'][cell['name']]
    assert state['status']=='complete'
    result=tmp_path/state['result'];saved_text=result.read_text();saved_mtime=result.stat().st_mtime_ns
    invoke('diagnostics.sweep',*args)
    assert result.stat().st_mtime_ns==saved_mtime
    assert read(tmp_path/'diagnostics_state.json')['runs'][cell['name']]==state
    truncated=read(result);truncated['evaluations'].pop();write(result,truncated)
    invoke('diagnostics.sweep',*args)
    repaired=read(tmp_path/'diagnostics_state.json')['runs'][cell['name']]
    assert repaired['status']=='complete' and repaired['result']!=state['result']
    assert complete_result(tmp_path/repaired['result'],[8,16],32)
    assert read(result)==truncated  # Do not overwrite the incomplete attempt.
    for filename in ('attention_parameters.csv','diagnostic_attention.csv','diagnostic_loss_by_position.csv','diagnostic_branch_norms.csv',
        'final_validation_vs_context.png',f"diagnostics/{cell['name']}.png"):
        assert (tmp_path/'live'/filename).stat().st_size>100


def test_monitor_keeps_refreshing_files_when_port_is_busy(tmp_path):
    plan=build(read(ROOT/'configs/context_ablation_smoke.json'));write(tmp_path/'plan.json',plan)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));sock.listen()
        with (tmp_path/'monitor.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','context_sweep.monitor','--root',str(tmp_path),
                '--port',str(sock.getsockname()[1]),'--interval','.1','--plot-interval','60'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+35
                report=tmp_path/'live/live_data.json'
                while time.monotonic()<deadline:
                    if report.is_file() and 'Continuing file reports' in (tmp_path/'monitor.log').read_text():break
                    if proc.poll() is not None:pytest.fail((tmp_path/'monitor.log').read_text())
                    time.sleep(.1)
                else:pytest.fail('Monitor did not initialize')
                first=read(report)['updated_at']
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if read(report)['updated_at']>first:break
                    time.sleep(.1)
                else:pytest.fail('File monitor stopped when HTTP binding failed')
            finally:
                proc.terminate();proc.wait(timeout=20)


def test_launcher_plan_has_no_data_or_cuda_dependency():
    import json
    result=json.loads(invoke('context_sweep.launch','plan','--contexts',1024,'--variants','relu2_bias','--output-norms','none'))
    assert result['selected_run_count']==1 and result['time_limit'] is None
    assert result['tokens_per_run']==100007936
