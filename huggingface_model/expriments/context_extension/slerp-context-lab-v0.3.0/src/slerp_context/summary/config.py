from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
from ..config import Config


@dataclass
class Study:
    model: dict = field(default_factory=lambda: dict(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct", method="slerp",
        window=2048, chunk=256, slots=64, token_budget=2_000_000,
        tokens_per_update=8192, save_every=50, seed=17))
    data_dir: str = "data/summary-smol"
    dataset_id: str = "ccdv/govreport-summarization"
    dataset_config: str = "document"
    source_field: str = "report"
    target_field: str = "summary"
    dataset_revision: str = "4e21184e01ae8017e2c036e180fe5e541fef60a0"
    length_edges: list = field(default_factory=lambda: [512, 4096, 7168, 8193, 16384, 32768])
    train_per_bin: int = 32
    validation_per_bin: int = 3
    test_per_bin: int = 4
    max_scan: int = 20_000
    max_data_mb: int = 250
    train_source_limit: int = 16384
    train_target_limit: int = 1024
    max_epochs: int = 3
    max_new_tokens: int = 512
    intermediate_tokens: int = 192
    eval_weight_dtype: str = "float32"
    distill_weight: float = 0.0
    distill_every: int = 4
    distill_tokens: int = 16
    prompt: str = ("Write an accurate, concise executive summary of the complete document. "
                   "Explain its main purpose, major findings, conclusions, and recommendations where present. "
                   "Preserve important qualifications and causal relationships. Do not invent information. "
                   "Return only the summary.")

    def validate(self):
        cfg = Config(**self.model).validate()
        if sorted(set(self.length_edges)) != self.length_edges or len(self.length_edges)<2 or min(self.length_edges)<1:
            raise ValueError("length_edges must be strictly increasing positive token counts")
        if any(x<1 for x in [self.train_per_bin,self.validation_per_bin,self.test_per_bin,
                            self.max_scan,self.max_data_mb,self.max_epochs,self.max_new_tokens,
                            self.intermediate_tokens,self.train_source_limit,self.train_target_limit,
                            self.distill_every,self.distill_tokens]):
            raise ValueError("Summary counts/budgets must be positive")
        if self.max_new_tokens >= cfg.window or self.intermediate_tokens >= cfg.window:
            raise ValueError("Generation budget must be below the working context")
        if self.eval_weight_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unknown eval weight dtype")
        if self.distill_weight < 0 or cfg.detach_local:
            raise ValueError("Nonnegative distillation weight and differentiable local state required")
        if cfg.kl_weight:
            raise ValueError("Summarization uses distill_weight; retrieval kl_weight does not apply here")
        if cfg.tiny and cfg.tiny_arch=="gpt_neox":
            raise ValueError("Summary workflow currently supports Llama and Hunyuan, not Pythia")
        return self

    def base(self): return Config(**self.model).validate()

    def save(self,path): Path(path).write_text(json.dumps(asdict(self),indent=2)+"\n")

    @classmethod
    def load(cls,path): return cls(**json.loads(Path(path).read_text())).validate()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
