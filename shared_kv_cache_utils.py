"""Utilities for building KV cache sharing maps.

This mirrors the layer sharing patterns available in
``shared_param_utils`` but instead of constructing modules it returns a
list describing which layer supplies the KV cache for each layer.
"""
from __future__ import annotations
from typing import List


def build_kv_cache_map(n_layer: int, shared_size: int = 1, *,
                        shared_seq: int = 1, shared_sym: bool = False) -> List[int]:
    """Create a layer → cache owner mapping.

    Parameters
    ----------
    n_layer: int
        Total number of layers in the model.
    shared_size: int, default 1
        Number of contiguous layers that share the same cache entry.
    shared_seq: int, default 1
        Length of the repeating pattern (e.g. ``4`` gives ``A B C D``
        style sharing; combining with ``shared_size`` yields
        ``A A B B C C D D`` patterns).
    shared_sym: bool, default False
        If ``True`` mirror the pattern to create symmetric layouts such as
        ``A B C B A``.

    Returns
    -------
    List[int]
        A list of length ``n_layer`` where each position indicates from
        which cache group that layer should draw its KV values.
    """
    if shared_seq < 1:
        raise ValueError("shared_seq must be >= 1")

    mapping: List[int] = []
    physical_layers = (n_layer + 1) // 2 if shared_sym else n_layer

    for i in range(physical_layers):
        group_idx = (i // shared_size) % shared_seq
        mapping.append(group_idx)

    if shared_sym:
        mirror = mapping[:-1] if n_layer % 2 else mapping
        mapping = mapping + mirror[::-1]

    return mapping[:n_layer]
