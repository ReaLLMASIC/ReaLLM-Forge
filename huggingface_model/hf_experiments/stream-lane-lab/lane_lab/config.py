"""Small, serializable Hugging Face configuration for numeric streams."""

from copy import deepcopy

from transformers import PretrainedConfig


class LaneConfig(PretrainedConfig):
    model_type = "stream_lane"

    def __init__(self, columns=None, d_model=48, n_heads=2, n_layers=2,
                 d_ff=128, context_length=64, dropout=0.0,
                 auxiliary_mse_weight=0.0, **kwargs):
        # This model implements tying per column, not the LM word-embedding API.
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)
        if columns is None:
            columns = [{"name": "signal_1", "kind": "slerp", "knots": 5,
                        "tied": True, "levels": 128, "minimum": -1.0,
                        "maximum": 1.0, "backend": "auto"}]
        defaults = dict(kind="slerp", knots=5, tied=True, levels=128,
                        minimum=-1.0, maximum=1.0, backend="auto")
        self.columns = []
        for i, item in enumerate(columns):
            spec = {**defaults, **deepcopy(item)}
            spec.setdefault("name", f"signal_{i + 1}")
            if spec["kind"] not in {"independent", "slerp"}:
                raise ValueError("Column kind must be 'independent' or 'slerp'")
            if spec["backend"] not in {"auto", "torch", "triton"}:
                raise ValueError("Column backend must be auto, torch, or triton")
            if not isinstance(spec["levels"], int) or spec["levels"] < 2:
                raise ValueError("Each column needs at least two quantization levels")
            if not isinstance(spec["knots"], int) or spec["knots"] < 2:
                raise ValueError("SLERP knot count must be at least two")
            if not spec["maximum"] > spec["minimum"]:
                raise ValueError("Each column maximum must exceed its minimum")
            if not isinstance(spec["tied"], bool):
                raise ValueError("Column tied must be a Boolean")
            self.columns.append(spec)
        if not self.columns or len({c["name"] for c in self.columns}) != len(self.columns):
            raise ValueError("Provide one or more uniquely named columns")
        if d_model < 2 or n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be >= 2 and divisible by n_heads")
        if n_layers < 1 or d_ff < 1 or context_length < 1:
            raise ValueError("n_layers, d_ff, and context_length must be positive")
        if not 0 <= dropout < 1 or auxiliary_mse_weight < 0:
            raise ValueError("Invalid dropout or auxiliary loss weight")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.context_length = int(context_length)
        self.dropout = float(dropout)
        self.auxiliary_mse_weight = float(auxiliary_mse_weight)

    @property
    def num_columns(self):
        return len(self.columns)
