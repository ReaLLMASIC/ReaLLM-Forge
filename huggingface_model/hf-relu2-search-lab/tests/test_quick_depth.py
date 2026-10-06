"""Fixed 2048 context, resource gates, and honest partial reporting."""
import math
import pytest
from context_sweep.recipe import ROOT, read, write, build, digest
from quick_depth.recipe import specifications, summary, cells, storage_estimate, parameter_count
from quick_depth.launch import probe_outcome
from quick_depth.report import refresh


def test_defaults_preserve_original_preset_and_comparison_budget():
    before = digest(ROOT/'configs/context_ablation_50m.json')
    specs = specifications()
    view = summary(specs)
    assert list(specs) == [16, 24, 32]
    assert view['run_count'] == 6 and view['tokens_per_run'] == 100007936
    assert view['micro_batch_size'] == 2 and view['gradient_accumulation'] == 8
    for depth, spec in specs.items():
        assert spec['contexts'] == [2048] and spec['output_norms'] == ['none']
        assert spec['model']['hidden_size'] == 512 and spec['model']['num_hidden_layers'] == depth
        assert spec['model']['num_attention_heads'] == spec['model']['num_key_value_heads'] == 4
        assert spec['optimizer_recipes']['muon']['weight_decay'] == 0.
        assert spec['optimizer_recipes']['muon']['aux_weight_decay'] == .1
        assert len(cells(spec)) == 2
        assert all(c['context'] == cfg['train']['sequence_length'] == 2048 for c, cfg in cells(spec))
    assert digest(ROOT/'configs/context_ablation_50m.json') == before


@pytest.mark.parametrize('layers,tokens,micro', [([], 10, 4096), ([16, 16], 10, 4096),
    ([0], 10, 4096), ([True], 10, 4096), ([16], 0, 4096), ([16], 10, 3072), ([16], 10, 6144)])
def test_invalid_plan(layers, tokens, micro):
    with pytest.raises(ValueError):
        specifications(layers, tokens, micro)


def test_smaller_batch_preserves_tokens_and_accumulation():
    spec = specifications([2], tokens=65536, microbatch_tokens=2048)[2]
    plan = build(spec)
    assert plan['tokens_per_run'] == 65536
    assert plan['cells'][0]['batch_size'] == 1 and plan['cells'][0]['accumulation'] == 16


def test_storage_counts_every_final_and_single_active_checkpoint():
    specs = specifications([1, 2])
    cost = storage_estimate(specs, lambda spec, variant: spec['model']['num_hidden_layers']*100)
    assert cost['final_weight_bytes'] == 4*(100*2+200*2)
    assert cost['transient_checkpoint_bytes'] == 32*200
    assert cost['estimated_peak_bytes'] == 3_000_000_000+2400+6400


def test_softmax_and_relu_have_equal_parameter_count():
    spec = specifications([1])[1]
    assert parameter_count(spec, 'softmax_sdpa') == parameter_count(spec, 'relu2')


@pytest.mark.parametrize('process,measured,message', [
    ({'status': 'ok'}, {'status': 'oom'}, 'did not fit'),
    ({'status': 'ok'}, {'status': 'budget_exceeded'}, 'did not fit'),
    ({'status': 'failed'}, {'status': 'fit'}, 'not an OOM'),
    ({'status': 'ok'}, {'status': 'error'}, 'not an OOM'),
    ({'status': 'ok'}, {}, 'not an OOM')])
def test_probe_rejects_failure_and_never_silently_skips_depth(process, measured, message):
    with pytest.raises(RuntimeError, match=message):
        probe_outcome(process, measured, 'worker.log')


def test_pending_file_report_is_2048_and_never_shows_fake_zero(tmp_path):
    specs = specifications([1, 2])
    cost = storage_estimate(specs, lambda *args: 100)
    write(tmp_path/'quick_plan.json', dict(**summary(specs), specs={str(k): v for k, v in specs.items()},
        storage=dict(cost, budget_bytes=28_000_000_000)))
    write(tmp_path/'quick_status.json', dict(phase='ready'))
    data = refresh(tmp_path, draw=True)
    assert len(data['runs']) == 4
    assert all(r['context'] == 2048 and r['status'] == 'pending' and r['best_validation_loss'] is None for r in data['runs'])
    report = (tmp_path/'live/report.md').read_text()
    assert 'Complete: 0/4' in report and 'context 2048' in report
    assert (tmp_path/'live/validation_progress.png').is_file()
    assert (tmp_path/'live/best_validation_vs_layers.pdf').is_file()
    assert not list(tmp_path.glob('L*/plan.json'))


def test_completed_mean_rejects_partial_points():
    from depth_sweep.report import completed_mean
    assert math.isnan(completed_mean([dict(seed=0, completed_budget=False, initialization_ok=True,
        best_validation_loss=3.2)], 'best_validation_loss', [0]))
