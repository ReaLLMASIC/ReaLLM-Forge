from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path


@dataclass
class Config:
    model_id: str = "tencent/Hunyuan-0.5B-Instruct"
    revision: str = "main"
    method: str = "slerp"
    window: int = 2048
    chunk: int = 256
    slots: int = 64
    pi_factor: float = 4.0
    fixed_alpha: float = 0.1
    backbone_lr: float = 1e-5
    memory_lr: float = 1e-4
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    token_budget: int = 5_000_000
    tokens_per_update: int = 8192
    lengths: tuple = (2048, 4096)
    length_weights: tuple = (0.3, 0.7)
    tasks: tuple = ("recall", "recall", "update", "trace", "multi")
    fact_counts: tuple = (2, 8, 16)
    positions: tuple = (0.05, 0.2, 0.5, 0.8)
    seed: int = 17
    save_every: int = 100
    keep_checkpoints: int = 1
    min_free_gb: float = 8.0
    max_project_gb: float = 40.0
    checkpoint: bool = True
    detach_local: bool = False
    warmup_fraction: float = 0.03
    natural_fraction: float = 0.0
    data_dir: str = "data/pg19-hunyuan"
    chat_template: bool = True
    loss_chunk_tokens: int = 32
    kl_weight: float = 0.0
    kl_every: int = 4
    kl_context: int = 512
    kl_tokens: int = 16
    tiny: bool = False
    tiny_arch: str = "hunyuan_v1_dense"

    def validate(self):
        if self.method not in {"slerp", "nlerp", "ema", "fixed", "local", "pi", "native", "delta", "rmt"}:
            raise ValueError("Unknown method")
        if not (self.window >= self.chunk > 0 and self.window % self.chunk == 0):
            raise ValueError("window must be a positive multiple of chunk")
        if self.slots <= 0 or self.loss_chunk_tokens <= 0:
            raise ValueError("slots and loss_chunk_tokens must be positive")
        if len(self.lengths) != len(self.length_weights) or not self.lengths or min(self.lengths) < 32:
            raise ValueError("Invalid length curriculum")
        if not all(w > 0 for w in self.length_weights):
            raise ValueError("Length weights must be positive")
        if self.token_budget <= 0 or self.tokens_per_update <= 0:
            raise ValueError("Token budgets must be positive")
        if self.pi_factor < 1 or not 0 < self.fixed_alpha < 1:
            raise ValueError("Invalid geometry settings")
        if not 0 <= self.natural_fraction <= 1 or not 0 < self.warmup_fraction <= 1:
            raise ValueError("Invalid mixture/warmup fraction")
        if self.optimizer not in {"adamw", "adamw8bit"}:
            raise ValueError("Choose adamw or adamw8bit (both update all weights)")
        if min(self.backbone_lr, self.memory_lr) <= 0 or self.save_every < 1 or self.keep_checkpoints < 1:
            raise ValueError("Learning rates and checkpoint intervals must be positive")
        if not self.fact_counts or min(self.fact_counts) < 1 or not self.tasks:
            raise ValueError("Need positive fact counts and at least one task")
        if not set(self.tasks) <= {"recall", "update", "trace", "multi", "capacity"}:
            raise ValueError("Unknown training task")
        if "trace" in self.tasks and min(self.fact_counts) < 2:
            raise ValueError("Trace training requires at least two facts")
        if not self.positions or not all(0 <= p <= 1 for p in self.positions):
            raise ValueError("Positions must be in [0, 1]")
        if self.kl_weight < 0 or self.kl_every < 1 or self.kl_tokens < 1 or self.kl_context < self.kl_tokens:
            raise ValueError("Invalid KL settings")
        if self.kl_weight and self.kl_context > self.window:
            raise ValueError("Retention KL context must fit the local window")
        if self.tiny_arch not in {"gpt_neox", "llama", "hunyuan_v1_dense"}:
            raise ValueError("Unsupported tiny architecture")
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if "rank" in data or "lora_lr" in data or "memory_only" in data:
            raise ValueError("v0.1 LoRA configs/checkpoints cannot be resumed in v0.2; use a new full-weight config")
        cfg = cls(**data).validate()
        lock = revision_lock(cfg.model_id)
        if cfg.revision == "main" and lock.exists():
            resolved = json.loads(lock.read_text())
            if resolved["model_id"] == cfg.model_id:
                cfg.revision = resolved["revision"]
        return cfg


def revision_lock(model_id):
    name = hashlib.sha256(model_id.encode()).hexdigest()[:16]
    return Path("reports/model-locks") / (name + ".json")
