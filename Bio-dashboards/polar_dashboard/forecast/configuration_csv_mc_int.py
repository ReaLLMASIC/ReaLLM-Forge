"""Hugging Face `PretrainedConfig` for ReaLLM-Forge csv_mc_int multicontext models.

A csv_mc_int model reads K integer CSV columns ("lanes"), one token per row per lane
(token_id = value - int_min), sums the K lane embeddings into one residual stream, runs a
GPT-style transformer over the rows, and predicts each lane's next value with its own head
(tied to that lane's embedding). This config records the architecture plus the lanes
themselves (names, ranges, units), so a saved model directory is self-describing: a
forecaster can turn its tokens back into mV / mg / ... without any other file.

Supported: the ReaLLM-Forge GPT with `attention_variant=causal` (standard multi-head attention,
optional GQA), softmax or relu2max attention, optional QK-norm, RoPE, pre-RMSNorm blocks, an MLP
with gelu / relu / silu / squared_relu, no biases or biases, tied lane heads.
`from_reallm_model_args` refuses checkpoints that use anything else, rather than converting
them approximately.
"""

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig

ACTIVATIONS = ("gelu", "relu", "silu", "squared_relu", "relu2")

# model_args fields copied into the config (names as in ReaLLM-Forge's GPTConfig)
ARCH_KEYS = (
    "n_layer", "n_head", "n_kv_group", "n_embd", "block_size", "dropout", "bias",
    "use_rotary_embeddings", "rope_length", "rope_fp32", "softmax_variant_attn", "relu2max_divisor", "div_by_seq_len",
    "use_qk_norm", "use_qk_norm_scale", "activation_variant", "mlp_expansion_factor", "mlp_size",
    "mlp_x_offset", "mlp_y_offset", "learn_mlp_x_offset", "learn_mlp_y_offset",
)

# model_args fields the HF model does NOT implement: each must have one of these values (a field
# missing from an older checkpoint counts as its default). Anything else is refused.
REQUIRED = {
    "attention_variant": ("causal",),
    "softmax_variant_attn": ("softmax", "relu2max"),
    "norm_variant_attn": ("rmsnorm",),
    "norm_variant_output": ("rmsnorm",),
    "use_pre_ln": (True,),
    "use_post_ln": (False, None),
    "use_peri_ln": (False, None),
    "use_pre_ln_attn": (None, True),
    "use_pre_ln_mlp": (None, True),
    "use_post_ln_attn": (None, False),
    "use_post_ln_mlp": (None, False),
    "use_peri_ln_attn": (None, False),
    "use_peri_ln_mlp": (None, False),
    "mlp_variant": ("mlp",),
    "mlp_down_projs": (1, None),
    "use_parallel_mlp": (False,),
    "use_moe": (False,),
    "activation_variant": ACTIVATIONS,
    "use_gradual_activation": (False,),
    "use_abs_pos_embeddings": (False,),
    "use_embedding_scale": (False,),
    "norm_variant_wte": (None,),
    "norm_variant_lm_head": (None,),
    "rope_variant": ("rope",),
    "use_fire_embeddings": (False,),
    "window_size": (None,),
    "attn_logit_softcapping": (None,),
    "final_logit_softcapping": (None,),
    "use_v_norm": (False,),
    "gate": (False,),
    "use_flash_lobo": (False,),
    "use_flex_attn": (None, False),
    "n_embd_wte": (None,),
    "wte_weight_tying": (True,),
    "wte_fixed_norm": (False,),
    "numerical_multicontext": (False,),
    "multicontext": (True,),
    "multidataset_wte": (False,),
    "attention_residual_variant": ("standard",),
    "attn_residual_combination": ("add",),
    "mlp_residual_combination": ("add",),
    "use_attn_resid_scaling": (False,),
    "use_mlp_resid_scaling": (False,),
    "use_attn_resid_const": (False,),
    "use_mlp_resid_const": (False,),
    "use_ln_f_input_mixer": (False,),
    "use_lsv": (False,),
    "shared_attn_size": (1,),
    "shared_mlp_size": (1,),
    "linear_variant_attn": ("linear",),
    "linear_variant_mlp": ("linear",),
    "linear_variant_q": (None, "linear"),
    "linear_variant_k": (None, "linear"),
    "linear_variant_v": (None, "linear"),
    "linear_variant_attn_proj": (None, "linear"),
    "linear_variant_mlp_up": (None, "linear"),
    "linear_variant_mlp_down": (None, "linear"),
    "attn_cproj_scale": (1.0,),
    "mlp_cproj_scale": (1.0,),
    "attn_post_act_l2_norm": (False,),
    "mlp_post_act_l2_norm": (False,),
    "l2_norm_attn_q": (False,),
    "l2_norm_attn_k": (False,),
    "l2_norm_attn_v": (False,),
    "l2_norm_attn_cproj": (False,),
    "l2_norm_mlp_up": (False,),
    "l2_norm_mlp_down": (False,),
    "use_edgellm_asic": (False,),
    "mlp_up_bias": (None,),
    "mlp_down_bias": (None,),
    "n_qk_head_dim": (None,),
    "n_v_head_dim": (None,),
    "n_head_layerlist": (None, []),
    "n_kv_group_layerlist": (None, []),
    "n_qk_head_dim_layerlist": (None, []),
    "n_v_head_dim_layerlist": (None, []),
    "mlp_size_layerlist": (None, []),
    "n_cproj_layerlist": (None, []),
    "attention_variant_layerlist": (None, []),
    "use_rotary_embeddings_layerlist": (None, []),
    "window_size_layerlist": (None, []),
    "apply_vector_at_layer_idx": (None,),
    "obtain_vector_at_layer_idx": (None,),
}


def unsupported_reallm_args(model_args: Dict[str, Any]) -> Dict[str, Any]:
    """{field: value} for every model_args field this HF model can't reproduce exactly."""
    bad = {k: model_args[k] for k, ok in REQUIRED.items() if k in model_args and model_args[k] not in ok}
    if not model_args.get("use_rotary_embeddings", True):
        bad.pop("rope_variant", None)
    for k, v in model_args.items():   # fake-quantisation switches (the *_bits / *_method fields are inert)
        if k.startswith("quantize_") and not k.endswith(("_bits", "_method")) and v not in (False, None, 0):
            bad[k] = v
    return bad


class CsvMcIntConfig(PretrainedConfig):
    """Configuration of `CsvMcIntForCausalLM`.

    lanes: one dict per CSV column in model order, e.g.
        {"column": "ecg_uv", "int_min": -2500, "int_max": 2500, "vocab_size": 5001,
         "unit": "mV", "scale": 1000, "step": 1, "decode": "mV = value * 1 / 1000"}
      A real value is `value / scale` when the lane has a scale, else `value * step` (unit as given).
    vocab_sizes: per-lane vocabulary sizes (taken from lanes when not given).
    sample_rate_hz: rows per second, when the rows are evenly spaced in time (130 for the H10 ECG).
    source: free-form description of how the rows were made (e.g. Bio-dash waves settings), so a
      live forecaster can check it builds its rows the same way.
    rope_fp32: compute the rotary angles (and the rotation) in float32 even under bf16 autocast. Off for the
      models trained before it existed: autocast computed their angles in bf16 (whole radians past 128 rad,
      and positions above 256 not even whole numbers), so they must keep doing that to stay bit-identical.
    lane_loss_weights: optional per-lane loss weights (default: the plain mean over lanes, as train.py).
    """

    model_type = "csv_mc_int"

    def __init__(
        self,
        lanes: Optional[List[Dict[str, Any]]] = None,
        vocab_sizes: Optional[List[int]] = None,
        n_layer: int = 6,
        n_head: int = 6,
        n_kv_group: Optional[int] = None,
        n_embd: int = 384,
        block_size: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
        use_rotary_embeddings: bool = True,
        rope_length: Optional[int] = None,
        rope_fp32: bool = False,
        softmax_variant_attn: str = "softmax",
        relu2max_divisor: float = 256.0,
        div_by_seq_len: bool = False,
        use_qk_norm: bool = False,
        use_qk_norm_scale: bool = False,
        activation_variant: str = "gelu",
        mlp_expansion_factor: int = 4,
        mlp_size: Optional[int] = None,
        mlp_x_offset: float = 0.0,
        mlp_y_offset: float = 0.0,
        learn_mlp_x_offset: bool = False,
        learn_mlp_y_offset: bool = False,
        lane_loss_weights: Optional[List[float]] = None,
        sample_rate_hz: Optional[float] = None,
        source: Optional[Dict[str, Any]] = None,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.lanes = [dict(x) for x in lanes] if lanes else None
        if vocab_sizes is None and self.lanes:
            vocab_sizes = [int(x["vocab_size"]) for x in self.lanes]
        self.vocab_sizes = [int(v) for v in vocab_sizes] if vocab_sizes else [256]
        if self.lanes and len(self.lanes) != len(self.vocab_sizes):
            raise ValueError(f"{len(self.lanes)} lanes but {len(self.vocab_sizes)} vocab sizes")
        if self.lanes:
            for lane, v in zip(self.lanes, self.vocab_sizes):
                if int(lane["int_max"]) - int(lane["int_min"]) + 1 != v:
                    raise ValueError(f"lane {lane.get('column')}: range {lane['int_min']}..{lane['int_max']} "
                                     f"doesn't match vocab_size {v}")
        self.num_lanes = len(self.vocab_sizes)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_group = n_kv_group if n_kv_group is not None else n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rope_length = rope_length
        # rotary angles and the rotation in float32 even under bf16 autocast (False: as the models trained before
        # 2026-10-21, whose angles autocast computed in bf16, rounding them past position ~128)
        self.rope_fp32 = bool(rope_fp32)
        if softmax_variant_attn not in ("softmax", "relu2max"):
            raise ValueError(f"softmax_variant_attn {softmax_variant_attn!r}: softmax or relu2max")
        self.softmax_variant_attn = softmax_variant_attn
        self.relu2max_divisor = relu2max_divisor
        self.div_by_seq_len = div_by_seq_len
        self.use_qk_norm = use_qk_norm
        self.use_qk_norm_scale = use_qk_norm_scale
        if activation_variant not in ACTIVATIONS:
            raise ValueError(f"activation_variant {activation_variant!r}: one of {ACTIVATIONS}")
        self.activation_variant = activation_variant
        self.mlp_expansion_factor = mlp_expansion_factor
        self.mlp_size = mlp_size
        self.mlp_x_offset = mlp_x_offset
        self.mlp_y_offset = mlp_y_offset
        self.learn_mlp_x_offset = learn_mlp_x_offset
        self.learn_mlp_y_offset = learn_mlp_y_offset
        if lane_loss_weights is not None and len(lane_loss_weights) != self.num_lanes:
            raise ValueError(f"{len(lane_loss_weights)} lane_loss_weights for {self.num_lanes} lanes")
        self.lane_loss_weights = [float(w) for w in lane_loss_weights] if lane_loss_weights else None
        self.sample_rate_hz = sample_rate_hz
        self.source = source
        self.initializer_range = initializer_range
        if n_embd % n_head:
            raise ValueError("n_embd must be a multiple of n_head")
        if n_head % self.n_kv_group:
            raise ValueError("n_head must be a multiple of n_kv_group")
        if "auto_map" not in kwargs:
            kwargs["auto_map"] = {
                "AutoConfig": "configuration_csv_mc_int.CsvMcIntConfig",
                "AutoModelForCausalLM": "modeling_csv_mc_int.CsvMcIntForCausalLM",
            }
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def lane_names(self) -> List[str]:
        if self.lanes:
            return [str(x.get("column", f"lane_{i}")) for i, x in enumerate(self.lanes)]
        return [f"lane_{i}" for i in range(self.num_lanes)]

    @classmethod
    def from_reallm_model_args(cls, model_args: Dict[str, Any], lanes: Optional[List[Dict[str, Any]]] = None,
                               **extra) -> "CsvMcIntConfig":
        """The config of a ReaLLM-Forge checkpoint (its `model_args`), refusing features this model lacks."""
        bad = unsupported_reallm_args(model_args)
        if bad:
            raise ValueError(f"checkpoint uses features hf_csv_mc_int doesn't implement: {bad}")
        kw = {k: model_args[k] for k in ARCH_KEYS if k in model_args}
        if kw.get("mlp_x_offset") is None:
            kw["mlp_x_offset"] = 0.0
        if kw.get("mlp_y_offset") is None:
            kw["mlp_y_offset"] = 0.0
        sizes = model_args.get("vocab_sizes") or [model_args["vocab_size"]]
        if lanes is not None and [int(x["vocab_size"]) for x in lanes] != [int(v) for v in sizes]:
            raise ValueError(f"lanes' vocab sizes {[x['vocab_size'] for x in lanes]} != checkpoint's {sizes}")
        kw.update(extra)
        return cls(lanes=lanes, vocab_sizes=list(sizes), **kw)
