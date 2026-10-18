"""Pure planning and an auditable empirical capacity search."""
import copy
import math
from decimal import Decimal
from context_sweep.recipe import build


def depth_spec(base, layers):
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError('layers must be a positive integer')
    spec = copy.deepcopy(base)
    spec['name'] = base['name'] + f'_depth{layers}'
    spec['contexts'] = [512, 1024]
    spec['model']['num_hidden_layers'] = layers
    build(spec)  # Validate without importing Torch or initializing CUDA.
    return spec


def depth_grid(start, maximum, fraction=.7, points=5):
    if not 0 < fraction <= 1 or points < 2:
        raise ValueError('Use 0 < fraction <= 1 and at least two grid points')
    # Avoid .7 * 90 == 62.99999999999999 dropping a mathematically exact layer.
    end = int(Decimal(str(fraction)) * maximum)
    if end < start:
        raise ValueError(f'70%-style endpoint {end} is below starting depth {start}; '
                         'reduce the batch size in a copied preset and re-probe in a new output')
    return sorted({start + round(i * (end-start) / (points-1)) for i in range(points)})


def find_limit(start, cap, fits):
    """Return last fit and first failing neighbor. Never label the search cap a maximum.

    fits must raise on compilation, correctness, timeout, or environment errors.
    Only a CUDA OOM or measured memory-budget exceedance returns False.
    Assumes broadly monotone memory; every eventual grid cell is revalidated.
    """
    if cap <= start:
        raise ValueError('search cap must exceed starting depth')
    if not fits(start):
        raise ValueError('Starting depth does not fit the chosen memory reserve')
    low = start
    while True:
        high = min(cap, low * 2)
        if not fits(high):
            break
        low = high
        if low == cap:
            raise ValueError(f'All tested depths through --max-search-layers {cap} fit; '
                             'increase the cap to establish a failing bound. No maximum was inferred.')
    while high - low > 1:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid
    return low, high


def probe_cells(base, depth):
    plan = build(depth_spec(base, depth))
    # Seed affects values but not parameter/batch shapes. Probe every planned seed,
    # optimizer, context, attention variant and output-norm condition anyway.
    return [(cell, plan['configs'][cell['config_key']]) for cell in plan['cells']]
