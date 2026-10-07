"""Exercise the reporter through real Trainer setup, training and resume."""

import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

import train
from train_args import parse_args


def _dataset(path, alphabet):
    path.mkdir()
    with (path / 'meta.pkl').open('wb') as handle:
        pickle.dump({'vocab_size': len(alphabet),
                     'stoi': {char: i for i, char in enumerate(alphabet)},
                     'itos': dict(enumerate(alphabet))}, handle)
    for split in ('train', 'val'):
        np.tile(np.arange(len(alphabet), dtype=np.uint16), 32).tofile(path / f'{split}.bin')
    return str(path)


def _cli(dataset, out_dir):
    return [
        '--dataset', dataset, '--out_dir', str(out_dir), '--device', 'cpu',
        '--dtype', 'float32', '--n_layer', '1', '--n_head', '1', '--n_kv_group', '1',
        '--n_embd', '8', '--block_size', '4', '--batch_size', '1',
        '--eval_iters', '1', '--eval_interval', '1', '--log_interval', '1',
        '--gradient_accumulation_steps', '2', '--max_iters', '0',
        '--lr_decay_iters', '10', '--warmup_iters', '0', '--dropout', '0',
        '--no-tensorboard_log', '--no-csv_log', '--no-wandb_log',
        '--no-print_model_info', '--log_per_token_metrics',
        '--only_save_checkpoint_at_end', '--sample_file', str(out_dir / 'sample.txt'),
    ]


def _trainer(monkeypatch, cli):
    monkeypatch.setattr(sys, 'argv', ['train.py', *cli])
    return train.Trainer(*parse_args())


@pytest.fixture
def fast_reports(monkeypatch):
    # The reporter tests and CLI smoke test exercise actual PNG rendering.
    # Retain real CSV and HTML generation in the training/resume tests.
    monkeypatch.setattr('utils.per_token_static.write_static_dashboards', lambda *args: [])
    monkeypatch.delenv('RANK', raising=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('mode', ['single', 'multicontext', 'multidataset'])
def test_training_resume_uses_checkpoint_counts_and_stream_labels(tmp_path, monkeypatch, fast_reports, mode):
    first = _dataset(tmp_path / 'first', 'abc')
    datasets = {first: 'abc'}
    cli = _cli(first, tmp_path / 'out')
    if mode != 'single':
        second = _dataset(tmp_path / 'second', 'VWXYZ')
        datasets[second] = 'VWXYZ'
        cli += ['--training_mode', mode]
        if mode == 'multicontext':
            cli += ['--multicontext_datasets', first, second]
        else:
            cli += ['--dataset_list', first, second, '--multidataset_wte']
    trainer = _trainer(monkeypatch, cli)
    assert trainer.per_token_metrics.vocab_sizes == {d: len(a) for d, a in datasets.items()}
    trainer.train()
    checkpoint = torch.load(tmp_path / 'out' / 'ckpt.pt', weights_only=True)
    saved = checkpoint['per_token_metrics']['seen']
    expected_total = 16 if mode == 'multicontext' else 8
    assert sum(int(counts.sum()) for counts in saved.values()) == expected_total
    assert checkpoint['metrics'] is not None  # Preserve existing checkpoint metadata.
    original_dir = Path(trainer.per_token_metrics.output_dir)
    before_resume = Path(trainer.per_token_metrics.detail_path).read_bytes()

    # Emulate reports written after the chosen checkpoint was saved.
    trainer.per_token_metrics.count_training_batch(first, torch.tensor([0, 0, 0]))
    trainer.per_token_metrics.export(99)
    old_history = Path(trainer.per_token_metrics.detail_path).read_bytes()
    assert old_history != before_resume

    resumed = _trainer(monkeypatch, cli + ['--init_from', 'resume', '--max_iters', '1'])
    report_dir = Path(resumed.per_token_metrics.output_dir)
    assert report_dir.parent == original_dir
    assert report_dir.name.startswith('resume_00000001_')
    for dataset in datasets:
        assert resumed.per_token_metrics.seen[dataset].tolist() == saved[dataset].tolist()
    resumed.train()
    assert Path(trainer.per_token_metrics.detail_path).read_bytes() == old_history
    with Path(resumed.per_token_metrics.detail_path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row['iteration']) for row in rows} == {1}
    for dataset, alphabet in datasets.items():
        stream_rows = [row for row in rows if row['dataset'] == dataset]
        assert [row['token_text_escaped'] for row in stream_rows] == list(alphabet)
        assert sum(int(row['val_eval_count']) for row in stream_rows) == 4
        assert [int(row['training_seen_count']) for row in stream_rows] == saved[dataset].tolist()
    after_resume = torch.load(tmp_path / 'out' / 'ckpt.pt', weights_only=True)
    assert sum(int(c.sum()) for c in after_resume['per_token_metrics']['seen'].values()) == expected_total * 2
    metadata = json.loads((report_dir / 'per_token_metadata.json').read_text())
    assert metadata['counts_restored_from_checkpoint'] is True
    assert metadata['counting_started_at_iteration'] == 0


def test_legacy_resume_warns_and_prev_run_resets_counts(tmp_path, monkeypatch, fast_reports):
    dataset = _dataset(tmp_path / 'data', 'abc')
    cli = _cli(dataset, tmp_path / 'out')
    trainer = _trainer(monkeypatch, cli)
    trainer.train()
    checkpoint_path = tmp_path / 'out' / 'ckpt.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    del checkpoint['per_token_metrics']
    torch.save(checkpoint, checkpoint_path)
    with pytest.warns(RuntimeWarning, match='counts restart at iteration 1'):
        resumed = _trainer(monkeypatch, cli + ['--init_from', 'resume', '--eval_only'])
    resumed.train()
    assert resumed.per_token_metrics.seen[dataset].sum() == 0
    assert resumed.per_token_metrics.counting_started_at == 1
    report_dir = Path(resumed.per_token_metrics.output_dir)
    assert 'counts start at iteration 1' in (report_dir / 'per_token_metrics.html').read_text()
    assert json.loads((report_dir / 'per_token_metadata.json').read_text())['counts_restored_from_checkpoint'] is False
    resumed.save_checkpoint('ckpt.pt')
    again = _trainer(monkeypatch, cli + ['--init_from', 'resume'])
    assert again.per_token_metrics.counting_started_at == 1
    assert again.per_token_metrics.output_dir != resumed.per_token_metrics.output_dir
    again.per_token_metrics.count_training_batch(dataset, torch.tensor([0, 1]))
    again.save_checkpoint('ckpt.pt')
    new_run = _trainer(monkeypatch, _cli(dataset, tmp_path / 'new-run') + [
        '--init_from', 'prev_run', '--prev_run_ckpt', str(tmp_path / 'out'),
    ])
    assert new_run.per_token_metrics.seen[dataset].sum() == 0
    assert new_run.per_token_metrics.counting_started_at == 0


@pytest.mark.parametrize('rank,extra,error', [
    ('0', [], 'single-process'),
    ('1', [], 'single-process'),
    (None, ['--numerical_multicontext'], 'categorical token logits'),
])
def test_unsupported_reporting_fails_before_ddp_or_output(tmp_path, monkeypatch, rank, extra, error):
    if rank is None:
        monkeypatch.delenv('RANK', raising=False)
    else:
        monkeypatch.setenv('RANK', rank)
    monkeypatch.setattr(train, 'init_process_group', lambda **kw: pytest.fail('DDP initialized'))
    with pytest.raises(ValueError, match=error):
        _trainer(monkeypatch, _cli('missing-dataset', tmp_path / 'out') + extra)
    assert not (tmp_path / 'out').exists()


def test_numerical_checkpoint_is_rejected_when_cli_uses_default(tmp_path, monkeypatch, fast_reports):
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    torch.save({'iter_num': 3, 'model_args': {'numerical_multicontext': True}}, out_dir / 'ckpt.pt')
    with pytest.raises(ValueError, match='categorical token logits'):
        _trainer(monkeypatch, _cli('missing-dataset', out_dir) + ['--init_from', 'resume'])
    assert not (out_dir / 'per_token_metrics').exists()


def test_reporting_disabled_keeps_checkpoint_free_of_report_state(tmp_path, monkeypatch, fast_reports):
    dataset = _dataset(tmp_path / 'data', 'abc')
    trainer = _trainer(monkeypatch, _cli(dataset, tmp_path / 'out') + ['--no-log_per_token_metrics'])
    trainer.train()
    assert trainer.per_token_metrics is None
    assert not (tmp_path / 'out' / 'per_token_metrics').exists()
    assert 'per_token_metrics' not in torch.load(tmp_path / 'out' / 'ckpt.pt', weights_only=True)


def test_cli_reporting_smoke_writes_all_graphs(tmp_path):
    dataset = _dataset(tmp_path / 'data', 'abc')
    out_dir = tmp_path / 'out'
    env = dict(os.environ, OMP_NUM_THREADS='1', MPLBACKEND='Agg')
    env.pop('RANK', None)
    result = subprocess.run(
        [sys.executable, 'train.py', *_cli(dataset, out_dir), '--eval_only'],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report_dir = out_dir / 'per_token_metrics'
    assert len(list(report_dir.glob('*.html'))) == 17
    assert len(list(report_dir.glob('*.png'))) == 8
    with (report_dir / 'per_token_metrics.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert sum(int(row['val_eval_count']) for row in rows) == 4
    assert sum(int(row['training_seen_count']) for row in rows) == 0
