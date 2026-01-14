# Foundation Model Adaptation Framework

A modular framework for adapting pretrained foundation models to computational biology tasks with parameter-efficient fine-tuning.

## Overview

This package provides:

- **Resource-aware configurations** — Small/medium/large model presets for different hardware (M1 Mac, RunPod, Cloud)
- **Parameter-efficient tuning** — LoRA, adapters, freezing strategies
- **Conditioning mechanisms** — FiLM, cross-attention, classifier-free guidance
- **End-to-end recipes** — Complete pipelines for gene expression tasks

## Quick Start

### 1. Auto-detect Hardware and Get Recommended Config

```python
from genailab.foundation.configs import get_resource_profile, get_model_config

# Auto-detect hardware
profile = get_resource_profile()
print(f"Detected: {profile.name}")
print(f"Recommended model size: {profile.recommended_model_size}")

# Get corresponding model config
config = get_model_config(profile.recommended_model_size)
print(f"Model parameters: ~{config.total_params_millions}M")
print(f"Estimated memory: ~{config.memory_estimate_gb()}GB")
```

### 2. Apply LoRA to a Model

```python
from genailab.foundation.tuning import apply_lora_to_model

# Apply LoRA to attention layers
model = apply_lora_to_model(
    model,
    target_modules=["attention.query", "attention.key", "attention.value"],
    rank=8,
    alpha=16,
)

# Now only LoRA parameters are trainable (~1% of total parameters)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### 3. Use Adapters

```python
from genailab.foundation.tuning import Adapter, AdapterConfig

# Create adapter config
adapter_config = AdapterConfig(
    hidden_dim=512,
    bottleneck_dim=64,
    dropout=0.1,
)

# Add adapters to transformer blocks
for block in model.blocks:
    block.adapter = Adapter(adapter_config)
```

## Model Size Configurations

Three preset configurations optimized for different hardware:

| Config | Hardware | Params | Memory (fp16) | Batch Size | Use Case |
|--------|----------|--------|---------------|------------|----------|
| **Small** | M1 Mac 16GB | ~50M | ~8GB | 8 (×4 accum) | Development, prototyping |
| **Medium** | RunPod 24GB | ~200M | ~18GB | 32 | Training, experimentation |
| **Large** | Cloud 40GB+ | ~600M | ~35GB | 64 | Production, large-scale |

### Example: Using Small Config on M1 Mac

```python
from genailab.foundation.configs import SMALL_CONFIG

print(f"Embed dim: {SMALL_CONFIG.embed_dim}")  # 256
print(f"Depth: {SMALL_CONFIG.depth}")  # 6
print(f"Batch size: {SMALL_CONFIG.batch_size}")  # 8
print(f"Gradient accumulation: {SMALL_CONFIG.gradient_accumulation_steps}")  # 4
print(f"Effective batch: {SMALL_CONFIG.effective_batch_size}")  # 32
```

## Parameter-Efficient Fine-Tuning

### LoRA (Low-Rank Adaptation)

**Best for**: Maximum parameter efficiency, task-specific adaptation

```python
from genailab.foundation.tuning import LoRA

# Apply LoRA
model = LoRA.apply(
    model,
    target_modules=["attention.query", "attention.key", "attention.value"],
    rank=8,  # Typical: 4-16
    alpha=16,  # Typical: 2*rank
)

# Save only LoRA parameters (tiny file!)
LoRA.save_lora_only(model, "lora_weights.pt")

# Load LoRA parameters
LoRA.load_lora_only(model, "lora_weights.pt")
```

**Memory savings**: Train only ~1% of parameters!

### Adapters

**Best for**: Stable training, multiple tasks, modular design

```python
from genailab.foundation.tuning import Adapter, AdapterConfig

config = AdapterConfig(hidden_dim=512, bottleneck_dim=64)

# Add to each transformer block
for block in model.blocks:
    block.adapter = Adapter(config)
```

### Freezing Strategies

**Best for**: Transfer learning, low-data regimes

```python
from genailab.foundation.tuning import freeze_layers, freeze_all_except

# Freeze all except last 3 blocks
freeze_all_except(model, ["blocks.9", "blocks.10", "blocks.11"])

# Always unfreeze layer norms
for module in model.modules():
    if isinstance(module, nn.LayerNorm):
        for param in module.parameters():
            param.requires_grad = True
```

## Conditioning Mechanisms

### FiLM (Feature-wise Linear Modulation)

**Best for**: Perturbation modeling, multi-modal conditioning

```python
from genailab.foundation.conditioning import FiLM

# Add FiLM to each block
film = FiLM(condition_dim=128, hidden_dim=512)

# In forward pass
hidden = transformer_block(hidden)
hidden = film(hidden, condition_embedding)
```

### Cross-Attention

**Best for**: Complex conditioning, multi-modal inputs

```python
from genailab.foundation.conditioning import CrossAttention

cross_attn = CrossAttention(
    query_dim=512,
    context_dim=256,
    num_heads=8,
)

# Condition tokens attend into backbone tokens
output = cross_attn(
    query=backbone_tokens,
    context=condition_tokens,
)
```

## Complete Example: Gene Expression with LoRA

```python
import torch
from genailab.foundation.configs import get_resource_profile, get_model_config
from genailab.foundation.tuning import apply_lora_to_model

# 1. Detect hardware and get config
profile = get_resource_profile()
config = get_model_config(profile.recommended_model_size)

# 2. Create model (your architecture)
from genailab.model import GeneExpressionDiffusion

model = GeneExpressionDiffusion(
    num_genes=20000,
    embed_dim=config.embed_dim,
    depth=config.depth,
    num_heads=config.num_heads,
)

# 3. Apply LoRA
model = apply_lora_to_model(
    model,
    target_modules=["attention"],
    rank=8,
)

# 4. Setup training
device = torch.device(profile.device)
model = model.to(device)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
)

# 5. Train with gradient accumulation
accumulation_steps = config.gradient_accumulation_steps

for step, batch in enumerate(dataloader):
    x = batch['expression'].to(device)
    
    # Forward
    loss = model(x)
    loss = loss / accumulation_steps
    
    # Backward
    loss.backward()
    
    # Update every N steps
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Directory Structure

```
genailab/foundation/
├── __init__.py
├── configs/
│   ├── model_configs.py      # Small/medium/large presets
│   └── resource_profiles.py  # Hardware detection
├── tuning/
│   ├── lora.py               # LoRA implementation
│   ├── adapters.py           # Adapter modules
│   └── freeze.py             # Freezing utilities
├── conditioning/
│   ├── film.py               # FiLM layers
│   ├── cross_attention.py    # Cross-attention
│   └── cfg.py                # Classifier-free guidance
├── backbones/
│   ├── dit.py                # DiT backbone
│   └── base.py               # Abstract base
└── recipes/
    ├── latent_diffusion.py   # Complete latent diffusion
    └── perturbation.py       # Perturbation prediction
```

## Tutorials

Interactive Jupyter notebooks demonstrating each component:

1. **Model Sizes and Resources** — Understanding hardware constraints
2. **LoRA Basics** — Parameter-efficient fine-tuning
3. **Adapters vs LoRA** — Comparing strategies
4. **Freezing Strategies** — Transfer learning patterns
5. **Conditioning Patterns** — FiLM, cross-attention, CFG
6. **Mixture of Experts** — Advanced architectures
7. **End-to-End Gene Expression** — Complete pipeline

See `notebooks/foundation_models/` for tutorials.

## Design Principles

### 1. Resource-Aware by Default

All configurations include memory estimates and hardware recommendations:

```python
config = SMALL_CONFIG
print(f"Memory estimate: {config.memory_estimate_gb()}GB")
print(f"Gradient checkpointing: {config.use_checkpoint}")
```

### 2. Modular and Composable

Mix and match components:

```python
# Combine LoRA + FiLM + gradient checkpointing
model = apply_lora_to_model(model, rank=8)
model = add_film_conditioning(model, condition_dim=128)
model = enable_gradient_checkpointing(model)
```

### 3. Production-Ready

Save/load utilities for deployment:

```python
# Save only LoRA (tiny file)
LoRA.save_lora_only(model, "lora.pt")

# Merge for inference
LoRA.merge_and_save(model, "merged_model.pt")
```

## Testing

Run tests to verify your setup:

```bash
# Activate environment
mamba activate genailab

# Test resource detection
python -m genailab.foundation.configs.resource_profiles

# Test model configs
python -m genailab.foundation.configs.model_configs

# Test LoRA
python -m genailab.foundation.tuning.lora
```

## References

**LoRA**:
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"

**Adapters**:
- Houlsby et al. (2019): "Parameter-Efficient Transfer Learning for NLP"

**FiLM**:
- Perez et al. (2018): "FiLM: Visual Reasoning with a General Conditioning Layer"

**Latent Diffusion**:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"

## Related Documentation

- [Leveraging Foundation Models](../../../docs/foundation_models/leveraging_foundation_models_v2.md)
- [Understanding Transformer Data Shapes](../../../docs/foundation_models/data_shape_v2.md)
- [Latent Diffusion Series](../../../docs/latent_diffusion/)
- [DiT Architecture](../../../docs/DiT/)
