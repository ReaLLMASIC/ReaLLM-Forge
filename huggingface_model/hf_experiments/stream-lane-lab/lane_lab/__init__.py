"""Stream Lane Lab: small transformers with numeric-column geometry."""

from transformers import AutoConfig, AutoModel

from .config import LaneConfig
from .model import LaneModel

AutoConfig.register(LaneConfig.model_type, LaneConfig, exist_ok=True)
AutoModel.register(LaneConfig, LaneModel, exist_ok=True)

__all__ = ["LaneConfig", "LaneModel"]
__version__ = "0.1.0"
