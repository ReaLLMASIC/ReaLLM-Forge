from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LayerState:
    k: torch.Tensor
    v: torch.Tensor
    direction: torch.Tensor
    log_radius: torch.Tensor
    valid: torch.Tensor

    def tensors(self):
        return (self.k, self.v, self.direction, self.log_radius, self.valid)


@dataclass(frozen=True)
class StreamState:
    consumed: int
    layers: tuple

    def detached(self):
        return StreamState(self.consumed, tuple(LayerState(*(t.detach().clone() for t in x.tensors())) for x in self.layers))

    def nbytes(self):
        return sum(t.numel() * t.element_size() for x in self.layers for t in x.tensors())

    def save(self, path):
        # Local trusted session files; only tensors and primitive metadata.
        torch.save({"consumed": self.consumed,
                    "layers": [[t.detach().cpu() for t in x.tensors()] for x in self.layers]}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, map_location=device, weights_only=True)
        return cls(d["consumed"], tuple(LayerState(*ts) for ts in d["layers"]))

