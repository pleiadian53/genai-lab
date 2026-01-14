"""
Foundation Model Adaptation Framework

This package provides reusable components for adapting pretrained foundation models
to computational biology tasks, with emphasis on:
- Parameter-efficient fine-tuning (LoRA, adapters)
- Resource-aware model sizing (small/medium/large)
- Modular conditioning mechanisms
- End-to-end recipes for gene expression tasks
"""

from genailab.foundation.configs import ModelConfig, ResourceProfile
from genailab.foundation.tuning import LoRA, LoRALinear, apply_lora_to_model

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "ResourceProfile",
    "LoRA",
    "LoRALinear",
    "apply_lora_to_model",
]
