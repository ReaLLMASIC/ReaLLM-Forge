"""Optional recording boundaries for aligned multicontext streams (NumPy only)."""
from pathlib import Path
import json
import pickle
import numpy as np


class SequenceWindows:
    def __init__(self, ranges, length, block_size):
        if block_size < 1:
            raise ValueError('block_size must be positive')
        self.ranges = np.asarray(ranges, dtype=np.int64)
        if self.ranges.ndim != 2 or self.ranges.shape[1] != 2 or len(self.ranges) == 0:
            raise ValueError('Sequence ranges must be nonempty [start,stop] pairs')
        starts, stops = self.ranges.T
        if (starts[0] != 0 or stops[-1] != length or np.any(stops <= starts)
                or np.any(starts[1:] != stops[:-1])):
            raise ValueError('Sequence ranges must partition their binary stream exactly')
        counts = stops-starts-block_size
        if np.any(counts < 1):
            raise ValueError(f'Every recording needs at least {block_size+1} frames')
        self.cumulative = np.cumsum(counts)
        self.total = int(self.cumulative[-1])

    def starts(self, ranks):
        ranks = np.asarray(ranks, dtype=np.int64)
        if np.any(ranks < 0) or np.any(ranks >= self.total):
            raise ValueError('Window rank outside valid range')
        sequence = np.searchsorted(self.cumulative, ranks, side='right')
        previous = np.r_[0, self.cumulative[:-1]][sequence]
        return self.ranges[sequence, 0] + ranks-previous


def load_sequence_windows(data_dirs, lengths, block_size):
    """All channels must opt in together and share exactly the same boundaries."""
    descriptions = []
    for directory in data_dirs:
        with (Path(directory)/'meta.pkl').open('rb') as stream:
            metadata = pickle.load(stream)
        name = metadata.get('sequence_ranges_file')
        if name is None:
            descriptions.append(None)
        else:
            description = json.loads((Path(directory)/name).read_text())
            if description.get('version') != 1:
                raise ValueError('Unsupported sequence range version')
            descriptions.append(description['ranges'])
    if all(item is None for item in descriptions):
        return None  # Preserve legacy sampling for datasets that do not opt in.
    if any(item is not None for item in descriptions) and any(item is None for item in descriptions):
        raise ValueError('All aligned channels must provide recording boundaries')
    if any(item != descriptions[0] for item in descriptions):
        raise ValueError('Aligned channels have different recording boundaries')
    windows = {}
    for split in ('train', 'val'):
        sizes = lengths[split]
        if len(set(sizes)) != 1:
            raise ValueError(f'Multicontext {split} channels have different lengths')
        ranges = descriptions[0][split] if descriptions[0] is not None else [[0, sizes[0]]]
        windows[split] = SequenceWindows(ranges, sizes[0], block_size)
    return windows
