import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from hf_search.planning import build
from context_sweep.recipe import ROOT, read, write


def preset():
    return read(ROOT/'configs/hf_search_smoke.json')


def test_environment_guard_explains_wrong_transformers(monkeypatch):
    from importlib import metadata
    from hf_search.common import check_environment
    original = metadata.version
    monkeypatch.setattr(metadata,'version',lambda name:'5.0.0' if name == 'transformers' else original(name))
    with pytest.raises(RuntimeError,match='tied-weight metadata'):
        check_environment()


def test_search_pairs_arms_at_identical_tokens_with_independent_heads():
    plan = build(preset())
    assert len(plan['jobs']) == 3
    assert len({j['target_tokens'] for j in plan['jobs']}) == 1
    configs = list(plan['configurations'].values())
    assert configs[0] == configs[1] == configs[2]
    model = configs[0]['model']
    assert model['hidden_size'] % model['num_attention_heads'] != 0
    assert model['n_qk_head_dim'] != model['n_v_head_dim']
    assert configs[0]['optimizer_recipes']['muon']['weight_decay'] == 0


def test_random_search_reproducible_bounded_and_grid_requires_explicit_budget():
    spec = preset()
    spec['search_space']['num_attention_heads'] = [3,5,7]
    with pytest.raises(ValueError,match='Grid has'):
        build(spec)
    spec.update(method='random',max_architectures=2)
    a,b = build(spec),build(spec)
    assert a == b
    assert len(a['architectures']) == 2 and len(a['jobs']) == 6
    assert len({v['choices']['num_attention_heads'] for v in a['architectures']}) == 2


@pytest.mark.parametrize('change', ['weight_decay','arms','infinite','odd_rope','backend'])
def test_reject_unmatched_or_unsupported_choices(change):
    spec = preset()
    if change == 'weight_decay': spec['base']['optimizer_recipes']['muon']['weight_decay'] = .1
    if change == 'arms': spec['base']['variants'] = ['relu2']
    if change == 'infinite': spec['search_space']['memory_size'] = [64]
    if change == 'odd_rope': spec['search_space']['n_qk_head_dim'] = [7]
    if change == 'backend': spec['base']['smoke_only'] = False
    with pytest.raises(ValueError): build(spec)


def test_report_does_not_turn_missing_failed_scores_into_zero(tmp_path):
    from hf_search.report import render
    plan = build(preset())
    write(tmp_path/'plan.json',plan)
    write(tmp_path/'state.json',{plan['jobs'][0]['name']:{'status':'failed'}})
    rows = render(tmp_path,plots=False)
    assert all(r['best_validation_loss'] is None and not r['completed_budget'] for r in rows)
    assert rows[0]['status'] == 'failed'
    assert 'failed' in (tmp_path/'report.md').read_text()


def test_tiny_muon_training_resume_is_exact_and_arms_share_initial_state(tmp_path):
    import torch
    if not hasattr(torch.optim,'Muon'): pytest.skip('Native Muon is required')
    from experiment.smoke import create_fixture
    from hf_search.common import state_digest
    from hf_model.modeling_search import SearchForCausalLM
    fixture = create_fixture(tmp_path/'fixture')
    def call(*args):
        subprocess.run([sys.executable,'-m',*map(str,args)],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    data = tmp_path/'data'
    call('context_sweep.prepare','--local-jsonl',fixture/'documents.jsonl','--tokenizer',fixture/'tokenizer_source',
         '--output',data,'--train-tokens','10000','--validation-tokens','3000','--validation-permyriad','2500')
    plan = build(preset())
    cfg = tmp_path/'config.json'
    write(cfg,next(iter(plan['configurations'].values())))
    hashes=[]
    for variant in ('softmax_sdpa','relu2','linear16'):
        full = tmp_path/('full_'+variant)
        arguments=('hf_search.train','--config',cfg,'--data',data,'--variant',variant,'--optimizer','muon','--device','cpu')
        call(*arguments,'--output',full)
        summary = read(full/'train_summary.json')
        assert summary['status'] == 'complete' and summary['tokens'] == 256
        hashes.append(summary['initial_state_sha256'])
        if variant == 'relu2':
            resumed = tmp_path/'resumed'
            call(*arguments,'--output',resumed,'--stop-after','2')
            assert read(resumed/'train_summary.json')['status'] == 'paused'
            call(*arguments,'--output',resumed,'--resume','--stop-after','2')
            assert read(resumed/'train_summary.json')['steps'] == 2
            call(*arguments,'--output',resumed,'--resume')
            continued = read(resumed/'train_summary.json')
            assert continued['final_state_sha256'] == summary['final_state_sha256']
            assert continued['best_validation'] == summary['best_validation']
            model = SearchForCausalLM.from_pretrained(resumed/'final')
            assert state_digest(model) == summary['final_state_sha256']
    assert len(set(hashes)) == 1
