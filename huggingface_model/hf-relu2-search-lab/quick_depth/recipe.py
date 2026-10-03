"""Planning without loading Torch; no changes to the original depth preset."""
import copy
from context_sweep.recipe import ROOT, read, build


def specifications(layers=(16, 24, 32), tokens=100_000_000, microbatch_tokens=4096):
    if not layers or len(set(layers)) != len(layers) or any(type(n) is not int or n < 1 for n in layers):
        raise ValueError('Choose unique positive integer layer counts')
    if type(tokens) is not int or tokens < 1:
        raise ValueError('tokens must be a positive integer')
    if type(microbatch_tokens) is not int or microbatch_tokens < 2048 or microbatch_tokens % 2048:
        raise ValueError('microbatch-tokens must be a positive multiple of 2048')
    base = read(ROOT/'configs/quick_depth_512_ctx2048.json')
    base['requested_tokens_per_run'] = tokens
    base['microbatch_token_target'] = microbatch_tokens
    base['milestone_tokens'] = sorted({max(1, tokens//4), max(1, tokens//2), tokens})
    result = {}
    for depth in sorted(layers):
        spec = copy.deepcopy(base)
        spec['name'] = f'quick_depth_d512_ctx2048_L{depth}'
        spec['model']['num_hidden_layers'] = depth
        build(spec)
        result[depth] = spec
    return result


def cells(spec):
    plan = build(spec)
    return [(cell, plan['configs'][cell['config_key']]) for cell in plan['cells']]


def parameter_count(spec, variant):
    import torch
    from hf_model.configuration_comparison import ComparisonConfig
    from hf_model.modeling_comparison import ComparisonForCausalLM
    with torch.device('meta'):
        model = ComparisonForCausalLM(ComparisonConfig(**dict(spec['model'], variant=variant, output_norm='none')))
    return sum(p.numel() for p in model.parameters())


def storage_estimate(specs, counter=parameter_count):
    counts = {str(depth): {v: counter(spec, v) for v in spec['variants']} for depth, spec in specs.items()}
    retained = sum(4*counts[str(depth)][c['variant']] for depth, spec in specs.items() for c, _ in cells(spec))
    largest = max(n for by_variant in counts.values() for n in by_variant.values())
    # Same conservative policy as the depth lab: all final FP32 weights, one
    # active checkpoint rotation (32 bytes/parameter), data/log/cache allowance.
    return dict(parameter_counts=counts, final_weight_bytes=retained,
                transient_checkpoint_bytes=32*largest, overhead_bytes=3_000_000_000,
                estimated_peak_bytes=retained+32*largest+3_000_000_000)


def summary(specs):
    plans = [build(spec) for spec in specs.values()]
    return dict(layers=list(specs), hidden_size=512, context=2048, heads=4, head_dim=128,
        intermediate_size=2048, variants=['softmax_sdpa', 'relu2'], output_norm='none',
        optimizer='muon', muon_weight_decay=0.0, adamw_matrix_weight_decay=0.1,
        adamw_scalar_norm_bias_weight_decay=0.0, time_limit=None,
        run_count=sum(p['run_count'] for p in plans), tokens_per_run=plans[0]['tokens_per_run'],
        requested_tokens_per_run=plans[0]['spec']['requested_tokens_per_run'],
        steps_per_run=plans[0]['steps'], tokens_per_step=plans[0]['tokens_per_step'],
        micro_batch_size=plans[0]['cells'][0]['batch_size'],
        gradient_accumulation=plans[0]['cells'][0]['accumulation'],
        note='Requested explicit depths; each attention/depth pair must pass a real CUDA capacity probe.')
