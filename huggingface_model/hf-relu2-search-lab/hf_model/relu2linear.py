"""Opt-in HF entry point. Importing hf_model alone retains the v4 API."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_nanogpt_relu2linear import NanoGPTReLU2LinearConfig
from .modeling_nanogpt_relu2linear import NanoGPTReLU2LinearModel, NanoGPTReLU2LinearForCausalLM
from .relu2linear_activation import relu2linear
from .triton_relu2linear_attention import (
    can_use_triton_relu2linear_attention,
    triton_relu2linear8_attention,
    triton_relu2linear_attention,
    make_triton_relu2linear_attention,
)

AutoConfig.register(NanoGPTReLU2LinearConfig.model_type, NanoGPTReLU2LinearConfig, exist_ok=True)
AutoModel.register(NanoGPTReLU2LinearConfig, NanoGPTReLU2LinearModel, exist_ok=True)
AutoModelForCausalLM.register(NanoGPTReLU2LinearConfig, NanoGPTReLU2LinearForCausalLM, exist_ok=True)

# Saved model directories carry their implementation for trust_remote_code=True.
NanoGPTReLU2LinearConfig.register_for_auto_class()
NanoGPTReLU2LinearModel.register_for_auto_class("AutoModel")
NanoGPTReLU2LinearForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = [
    "NanoGPTReLU2LinearConfig", "NanoGPTReLU2LinearModel", "NanoGPTReLU2LinearForCausalLM",
    "relu2linear", "can_use_triton_relu2linear_attention",
    "triton_relu2linear8_attention", "triton_relu2linear_attention",
    "make_triton_relu2linear_attention",
]
