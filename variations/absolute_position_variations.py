import torch
import torch.nn as nn

from variations.position_encoding_variations import QuantizedEmbedding


class LearnedAbsolutePositionEmbedding(nn.Module):
    """Standard learned absolute position embeddings."""

    def __init__(self, config):
        super().__init__()
        if config.quantize_wpe:
            self.embedding = QuantizedEmbedding(
                config.block_size,
                config.n_embd,
                config.quantize_wpe_method,
                config.quantize_wpe_bits,
            )
        else:
            self.embedding = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, seq_len, device, training=False):
        del training  # unused, kept for interface compatibility
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        return self.embedding(pos)

    def update_block_size(self, new_block_size):
        old_weight = self.embedding.weight.data
        old_block_size, dim = old_weight.shape
        if new_block_size <= old_block_size:
            return

        if isinstance(self.embedding, QuantizedEmbedding):
            new_embedding = QuantizedEmbedding(
                new_block_size,
                dim,
                self.embedding.quantization_method,
                self.embedding.quantization_bits,
            )
        else:
            new_embedding = nn.Embedding(new_block_size, dim)

        with torch.no_grad():
            new_embedding.weight[:old_block_size] = old_weight
        self.embedding = new_embedding

    def crop_block_size(self, block_size):
        self.embedding.weight = nn.Parameter(self.embedding.weight[:block_size])


class CyclicAbsolutePositionEmbedding(nn.Module):
    """Sums learned embeddings from multiple cyclic periods.

    For cycle lengths [2, 3, 5], position i uses:
      emb2[i % 2] + emb3[i % 3] + emb5[i % 5].
    """

    def __init__(self, config):
        super().__init__()
        if config.quantize_wpe:
            raise ValueError("cyclic absolute position embeddings currently do not support quantized wpe")

        cycle_lengths = config.cyclic_abs_pos_cycle_lengths
        if cycle_lengths is None or len(cycle_lengths) == 0:
            raise ValueError("cyclic_abs_pos_cycle_lengths must be set for cyclic_abs_pos embedding variant")
        if any(length <= 0 for length in cycle_lengths):
            raise ValueError("All cyclic_abs_pos_cycle_lengths values must be positive")

        self.cycle_lengths = [int(length) for length in cycle_lengths]
        self.randomize_starts = config.cyclic_abs_pos_randomize_starts

        self.embeddings = nn.ModuleList([
            nn.Embedding(cycle_len, config.n_embd) for cycle_len in self.cycle_lengths
        ])

    def forward(self, seq_len, device, training=False):
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        out = None

        for cycle_len, emb in zip(self.cycle_lengths, self.embeddings):
            if training and self.randomize_starts:
                start = torch.randint(0, cycle_len, (1,), device=device).item()
            else:
                start = 0

            cycle_pos = (pos + start) % cycle_len
            cycle_emb = emb(cycle_pos)
            out = cycle_emb if out is None else out + cycle_emb

        return out

    def update_block_size(self, new_block_size):
        del new_block_size
        # No-op; these embeddings are cycle-length based.
        return

    def crop_block_size(self, block_size):
        del block_size
        return


absolute_position_embedding_dict = {
    "learned": LearnedAbsolutePositionEmbedding,
    "cyclic": CyclicAbsolutePositionEmbedding,
}
