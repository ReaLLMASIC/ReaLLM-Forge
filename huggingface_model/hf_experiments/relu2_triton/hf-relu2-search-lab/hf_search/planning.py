"""Finite categorical grids and reproducible random samples, without CUDA."""
import copy
import itertools
import math
import random
from context_sweep.recipe import build as build_context, fingerprint

ARMS = ['softmax_sdpa', 'relu2', 'linear16']
MODEL_KEYS = {'hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'n_qk_head_dim', 'n_v_head_dim',
              'qk_norm_scale_init', 'relu2max_divisor', 'rope_theta'}
KEYS = MODEL_KEYS | {'context', 'muon_learning_rate', 'adamw_learning_rate'}


def candidates(spec):
    space = spec['search_space']
    if not space or not set(space) <= KEYS:
        raise ValueError(f'Only these finite categorical search keys are supported: {sorted(KEYS)}')
    for key, choices in space.items():
        if not isinstance(choices, list) or not choices:
            raise ValueError(f'{key} requires a nonempty list of finite numeric choices')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in choices):
            raise ValueError(f'{key}: values must be finite positive numbers')
        if len(set(choices)) != len(choices):
            raise ValueError(f'{key}: duplicate choices are not allowed')
    keys = sorted(space)
    count = math.prod(len(space[k]) for k in keys)
    maximum = spec.get('max_architectures', 1)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1 or maximum > 64:
        raise ValueError('max_architectures must be between 1 and 64')
    method = spec.get('method', 'grid')
    if method == 'grid':
        if count > maximum:
            raise ValueError(f'Grid has {count} architectures; explicitly increase max_architectures or choose random')
        indices = range(count)
    elif method == 'random':
        if count > 2**63 - 1:
            raise ValueError('Search space is too large')
        indices = sorted(random.Random(spec.get('search_seed', 0)).sample(range(count), min(count, maximum)))
    else:
        raise ValueError('method must be grid or random')
    result = []
    for index in indices:
        values = {}
        for key in reversed(keys):
            choices = space[key]
            values[key] = choices[index % len(choices)]
            index //= len(choices)
        result.append(values)
    return result


def build(spec):
    base = spec['base']
    if base.get('variants') != ARMS or base.get('optimizers') != ['muon']:
        raise ValueError('The matched search requires all three attention arms with Muon')
    if base['optimizer_recipes']['muon']['weight_decay'] != 0.0:
        raise ValueError('Muon weight_decay must be 0.0')
    if not base.get('smoke_only') and (base['dtype'] != 'bfloat16' or base['sdpa_backend'] != 'flash'):
        raise ValueError('GPU comparisons require BF16 and forced Flash SDPA, without silent backend fallback')
    if len(base.get('output_norms', [])) != 1:
        raise ValueError('Choose one fixed output norm for this architecture search')
    configurations, jobs, architectures = {}, [], []
    for index, choices in enumerate(candidates(spec)):
        trial = copy.deepcopy(base)
        context = choices.get('context', trial['contexts'][0])
        if not isinstance(context, int):
            raise ValueError('Context must be an integer')
        trial['contexts'] = [context]
        for key in MODEL_KEYS:
            if key in choices:
                trial['model'][key] = choices[key]
        model = trial['model']
        for key in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'n_qk_head_dim', 'n_v_head_dim'):
            if not isinstance(model[key], int) or isinstance(model[key], bool) or model[key] < 1:
                raise ValueError(f'model.{key} must be a positive integer')
        if model['n_qk_head_dim'] % 2 or max(model['n_qk_head_dim'], model['n_v_head_dim']) > 256:
            raise ValueError('RoPE requires even QK dimensions; the fused path supports QK/V dimensions <=256')
        model['num_key_value_heads'] = model['num_attention_heads']
        model['max_position_embeddings'] = max(context, model['max_position_embeddings'])
        muon = trial['optimizer_recipes']['muon']
        muon['learning_rate'] = choices.get('muon_learning_rate', muon['learning_rate'])
        muon['aux_learning_rate'] = choices.get('adamw_learning_rate', muon['aux_learning_rate'])
        context_plan = build_context(trial)
        name = f'arch{index:03d}_{fingerprint(choices)[:8]}'
        architecture = dict(name=name, choices=choices, model=model, context=context,
                            muon_learning_rate=muon['learning_rate'], adamw_learning_rate=muon['aux_learning_rate'])
        # Kernels zero-pad QK and V to a shared dimension; expose the actual work.
        padded = max(16, 1 << (max(model['n_qk_head_dim'], model['n_v_head_dim']) - 1).bit_length())
        stored = max(model['n_qk_head_dim'], model['n_v_head_dim'])
        architecture['padding'] = dict(kernel_dim=padded, stored_head_dim=stored,
                                       qk_ratio=padded/model['n_qk_head_dim'], value_ratio=padded/model['n_v_head_dim'],
                                       stored_qk_ratio=stored/model['n_qk_head_dim'], stored_value_ratio=stored/model['n_v_head_dim'])
        architectures.append(architecture)
        for cell in context_plan['cells']:
            job = dict(cell, name=f"{name}_{cell['variant']}_seed{cell['seed']}", architecture=name)
            configurations[job['name']] = context_plan['configs'][cell['config_key']]
            jobs.append(job)
    return dict(schema=1, spec=copy.deepcopy(spec), spec_sha256=fingerprint(spec),
                architectures=architectures, jobs=jobs, configurations=configurations,
                run_count=len(jobs), total_training_tokens=sum(j['target_tokens'] for j in jobs))
