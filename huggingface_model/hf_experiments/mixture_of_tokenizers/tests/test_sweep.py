"""Regression checks for a stopped sweep and bounded/resumed training ETA."""
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_sweep
from train import estimate_eta_seconds


def test_interrupted_child_stops_sweep_before_evaluation_or_next_size(tmp_path, monkeypatch):
    for size in (128, 256):
        (tmp_path / 'tokenizers' / f'k{size}').mkdir(parents=True)
    config = {
        'tokenizers': {'output': str(tmp_path / 'tokenizers'), 'sizes': [128, 256]},
        'training': {'output': str(tmp_path / 'runs'), 'device': 'cpu'},
        'data': {'output': str(tmp_path / 'data')},
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    calls = []

    def child(command, check):
        calls.append(command)
        output = Path(command[command.index('--output') + 1])
        output.mkdir(parents=True)
        (output / 'summary.json').write_text(json.dumps({'interrupted': True, 'step': 3}))

    monkeypatch.setattr(run_sweep.subprocess, 'run', child)
    monkeypatch.setattr(sys, 'argv', ['run_sweep.py', '--config', str(config_path), '--evaluate'])
    with pytest.raises(SystemExit) as exc:
        run_sweep.main()
    assert exc.value.code == 130
    assert len(calls) == 1
    assert Path(calls[0][1]).name == 'train.py'
    summary = json.loads((tmp_path / 'runs' / 'sweep-summary.json').read_text())
    assert summary['runs'][0]['training']['interrupted'] is True


def test_eta_uses_earliest_budget_and_only_resumed_process_progress():
    # This process completed 10 steps / 1000 tokens in 20 seconds, after resume.
    args = dict(tokens=6000, target=100000, step=60, initial_tokens=5000, initial_step=50, elapsed=20)
    assert estimate_eta_seconds(max_steps=65, **args) == pytest.approx(10)
    assert estimate_eta_seconds(max_steps=60, **args) == 0
    assert estimate_eta_seconds(max_steps=None, **args) == pytest.approx(1880)
    args['target'] = 6100
    assert estimate_eta_seconds(max_steps=65, **args) == pytest.approx(2)
