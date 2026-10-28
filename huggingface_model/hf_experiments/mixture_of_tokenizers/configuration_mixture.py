"""Hugging Face configuration and the shared global vocabulary contract."""

from transformers import AutoConfig, GPTNeoXConfig


ROUTES = ("byte", "zh", "ko", "ja", "en")


def route_index(route):
    """Resolve a public route name or integer, rejecting accidental aliases."""
    if isinstance(route, str):
        if route not in ROUTES:
            raise ValueError(f"Unknown route {route!r}; choose one of {ROUTES}")
        return ROUTES.index(route)
    if isinstance(route, bool) or int(route) != route or not 0 <= int(route) < 5:
        raise ValueError(f"Route must be an integer from 0 to 4; got {route!r}")
    return int(route)


class MixtureConfig(GPTNeoXConfig):
    """Four disjoint lexical banks, 256 shared bytes, three controls per route."""

    model_type = "mixture_tokenizer_neox"

    def __init__(
        self,
        lexical_size=10000,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        max_position_embeddings=512,
        initializer_seed=1337,
        use_tokenizer_embeddings=True,
        **kwargs,
    ):
        if not isinstance(lexical_size, int) or lexical_size < 10:
            raise ValueError("lexical_size must be an integer >= 10 (the ten native digits)")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        vocab_size = 256 + 4 * lexical_size + 15
        saved_vocab_size = kwargs.pop("vocab_size", vocab_size)
        if saved_vocab_size != vocab_size:
            raise ValueError("vocab_size conflicts with lexical_size and the global ID contract")
        kwargs.pop("tie_word_embeddings", None)
        controls = 256 + 4 * lexical_size
        kwargs.setdefault("bos_token_id", controls)
        kwargs.setdefault("eos_token_id", controls + 1)
        kwargs.setdefault("pad_token_id", controls + 2)
        kwargs.setdefault("hidden_dropout", 0.0)
        kwargs.setdefault("attention_dropout", 0.0)
        kwargs.setdefault("rotary_pct", 1.0)
        kwargs.setdefault("use_parallel_residual", False)
        kwargs.setdefault("hidden_act", "gelu")
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            tie_word_embeddings=True,
            **kwargs,
        )
        self.lexical_size = lexical_size
        self.initializer_seed = int(initializer_seed)
        self.use_tokenizer_embeddings = bool(use_tokenizer_embeddings)
        self.num_tokenizer_routes = len(ROUTES)

    def lexical_range(self, route):
        route = route_index(route)
        if route == 0:
            return range(0, 0)
        start = 256 + (route - 1) * self.lexical_size
        return range(start, start + self.lexical_size)

    def bos_id(self, route):
        return 256 + 4 * self.lexical_size + 3 * route_index(route)

    def eos_id(self, route):
        return self.bos_id(route) + 1

    def pad_id(self, route):
        return self.bos_id(route) + 2

    def allowed_ids(self, route):
        """Predictable IDs: bytes, this route's lexical bank, and its EOS."""
        return list(range(256)) + list(self.lexical_range(route)) + [self.eos_id(route)]


AutoConfig.register(MixtureConfig.model_type, MixtureConfig, exist_ok=True)
