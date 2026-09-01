"""Hugging Face implementation of the nanoGPT QK-norm experiments."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_nanogpt import NanoGPTConfig
from .modeling_nanogpt import NanoGPTForCausalLM, NanoGPTModel

# Make the classes available to standard local Auto APIs. ``exist_ok`` also
# makes notebook reloads safe without hiding unrelated registration failures.
AutoConfig.register(NanoGPTConfig.model_type, NanoGPTConfig, exist_ok=True)
AutoModel.register(NanoGPTConfig, NanoGPTModel, exist_ok=True)
AutoModelForCausalLM.register(NanoGPTConfig, NanoGPTForCausalLM, exist_ok=True)

__all__ = ["NanoGPTConfig", "NanoGPTModel", "NanoGPTForCausalLM"]
