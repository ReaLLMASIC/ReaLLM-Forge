"""Search-specific model construction, with the original token-stream semantics."""
from experiment.common import (ROOT, read_json, write_json, digest, json_digest,
    state_digest, seed_all, environment, amp, sync, configure_device,
    parameter_counts, budget, TokenStream, token_loss, backend_labels)
from experiment.common import validate_config as validate_base


def check_environment():
    from importlib.metadata import version
    expected = {'transformers':'4.44.2', 'tokenizers':'0.19.1'}
    mismatches = {name:version(name) for name,wanted in expected.items() if version(name) != wanted}
    if mismatches:
        raise RuntimeError(f'Unsupported HF environment: {mismatches}; expected {expected}. '
                           'Use the package venv from bash scripts/setup.sh (or set LAB_PYTHON to that interpreter). '
                           'No training has started; newer Transformers changes tied-weight metadata.')


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for folder in ('hf_model', 'experiment', 'hf_search')
            for p in sorted((ROOT / folder).glob('*.py'))}


def validate_config(cfg):
    validate_base(cfg)
    if cfg['variants'] != ['softmax_sdpa', 'relu2', 'linear16']:
        raise ValueError('Each architecture must compare softmax_sdpa, relu2 and linear16')
    if cfg['optimizers'] != ['muon'] or cfg['optimizer_recipes']['muon']['weight_decay'] != 0:
        raise ValueError('This search requires Muon with zero decoder weight decay')
    return cfg


def make_model(cfg, variant, seed, device):
    check_environment()
    from hf_model.configuration_search import SearchConfig
    from hf_model.modeling_search import SearchForCausalLM
    seed_all(seed)
    args = dict(cfg['model'], variant=variant, linear_threshold=16.0)
    args['sdpa_backend'] = cfg['sdpa_backend'] if str(device).startswith('cuda') else 'math'
    args['require_fused'] = str(device).startswith('cuda')
    args['relu2max_accelerator'] = 'triton' if str(device).startswith('cuda') else 'torch'
    model = SearchForCausalLM(SearchConfig(**args))
    initial_hash = state_digest(model)
    return model.to(device), initial_hash


def count_parameters(cfg):
    check_environment()
    import torch
    from hf_model.configuration_search import SearchConfig
    from hf_model.modeling_search import SearchForCausalLM
    with torch.device('meta'):
        model = SearchForCausalLM(SearchConfig(**dict(cfg['model'], variant='relu2')))
    return parameter_counts(model)
