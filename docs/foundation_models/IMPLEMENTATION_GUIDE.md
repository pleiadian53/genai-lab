# Foundation Model Adaptation: Implementation Guide

Quick reference for implementing the foundation model adaptation framework in your projects.

## 🎯 Quick Start

### 1. Check Your Hardware

```bash
# Activate environment
mamba activate genailab

# Check detected hardware
python -m genailab.foundation.configs.resource_profiles
```

**Output example (M1 Mac)**:
```
Detected Profile: M1 MacBook Pro 16GB
  Device: mps
  Memory: 16.0 GB
  Recommended model size: small
  Max batch size: 8
```

### 2. Compare Model Configurations

```bash
python -m genailab.foundation.configs.model_configs
```

**Output**:
```
Model Configuration Comparison
================================================================================
Config                    Params (M)   Memory (GB)  Batch    Depth   
--------------------------------------------------------------------------------
Small (M1 16GB)           50.2         8.1          32       6       
Medium (RunPod 24GB)      201.3        18.4         32       12      
Large (Cloud 40GB+)       603.9        35.2         64       24      
```

### 3. Test LoRA Implementation

```bash
python -m genailab.foundation.tuning.lora
```

---

## 📦 Package Structure Created

```
src/genailab/foundation/
├── __init__.py                          ✅ Created
├── README.md                            ✅ Created
├── configs/
│   ├── __init__.py                      ✅ Created
│   ├── model_configs.py                 ✅ Created (SMALL/MEDIUM/LARGE)
│   └── resource_profiles.py             ✅ Created (M1/RunPod/Cloud)
└── tuning/
    ├── __init__.py                      ✅ Created
    └── lora.py                          ✅ Created (Full implementation)
```

**Still to create**:

- `tuning/adapters.py`
- `tuning/freeze.py`
- `conditioning/film.py`
- `conditioning/cross_attention.py`
- `conditioning/cfg.py`
- `backbones/dit.py`
- `recipes/latent_diffusion.py`

---

## 🔧 Usage Patterns

### Pattern 1: Auto-Configure for Your Hardware

```python
from genailab.foundation.configs import get_resource_profile, get_model_config

# Auto-detect
profile = get_resource_profile()
config = get_model_config(profile.recommended_model_size)

print(f"Using {config.embed_dim}d model with {config.depth} layers")
print(f"Batch size: {config.batch_size} (×{config.gradient_accumulation_steps} accum)")
print(f"Memory estimate: ~{config.memory_estimate_gb()}GB")
```

### Pattern 2: Apply LoRA to Any Model

```python
from genailab.foundation.tuning import apply_lora_to_model

# Your model
model = YourTransformer(embed_dim=256, depth=6)

# Apply LoRA (trains only ~1% of parameters!)
model = apply_lora_to_model(
    model,
    target_modules=["attention.query", "attention.key", "attention.value"],
    rank=8,
    alpha=16,
)

# Train only LoRA parameters
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
)
```

### Pattern 3: Resource-Aware Training Loop

```python
import torch
from genailab.foundation.configs import get_resource_profile, get_model_config

# Configure
profile = get_resource_profile()
config = get_model_config(profile.recommended_model_size)

# Setup
device = torch.device(profile.device)
model = model.to(device)

# Training with gradient accumulation
accumulation_steps = config.gradient_accumulation_steps

for step, batch in enumerate(dataloader):
    x = batch['data'].to(device)
    
    # Forward
    loss = model(x) / accumulation_steps
    
    # Backward
    loss.backward()
    
    # Update every N steps
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 💾 Saving and Loading

### Save Only LoRA Weights (Tiny File!)

```python
from genailab.foundation.tuning import LoRA

# After training
LoRA.save_lora_only(model, "lora_weights.pt")
# File size: ~1MB (vs ~200MB for full model)

# Load later
base_model = YourTransformer()
base_model = apply_lora_to_model(base_model, rank=8)
LoRA.load_lora_only(base_model, "lora_weights.pt")
```

### Merge for Inference

```python
# Merge LoRA into base weights for faster inference
LoRA.merge_and_save(model, "merged_model.pt")
```

---

## 🎓 Learning Path

### For M1 Mac Users (16GB)

**Start here**:
1. Run `python -m genailab.foundation.configs.resource_profiles`
2. Verify you get `SMALL_CONFIG` recommendation
3. Open `notebooks/foundation_models/01_model_sizes_and_resources.ipynb`
4. Try `notebooks/foundation_models/02_lora_basics.ipynb`

**Key settings for M1**:

- Batch size: 8
- Gradient accumulation: 4 (effective batch = 32)
- Gradient checkpointing: ON
- Mixed precision: ON (fp16)
- Device: `mps`

### For RunPod/Cloud Users (24GB+)

**Start here**:
1. Verify CUDA setup: `python -c "import torch; print(torch.cuda.is_available())"`
2. Run resource detection
3. Jump to `notebooks/foundation_models/03_adapters_vs_lora.ipynb`
4. Experiment with `MEDIUM_CONFIG` or `LARGE_CONFIG`

**Key settings for GPU**:

- Batch size: 32-64
- Gradient accumulation: 1
- Flash attention: ON
- Torch compile: ON

---

## 🔬 Next Steps

### Immediate (This Session)

1. **Test the framework**:
   ```bash
   python -m genailab.foundation.configs.resource_profiles
   python -m genailab.foundation.configs.model_configs
   python -m genailab.foundation.tuning.lora
   ```

2. **Create first notebook**: `01_model_sizes_and_resources.ipynb`
   - Interactive hardware detection
   - Model size comparison
   - Memory estimation examples

3. **Implement remaining tuning modules**:
   - `adapters.py` — Bottleneck adapter implementation
   - `freeze.py` — Layer freezing utilities

### Short-term (Next Sessions)

4. **Conditioning mechanisms**:
   - `film.py` — FiLM layers for perturbation conditioning
   - `cross_attention.py` — Multi-modal conditioning
   - `cfg.py` — Classifier-free guidance

5. **Complete notebooks**:
   - `02_lora_basics.ipynb`
   - `03_adapters_vs_lora.ipynb`
   - `07_end_to_end_gene_expression.ipynb`

6. **Recipes**:
   - `latent_diffusion.py` — Complete pipeline
   - `perturbation.py` — Perturbation prediction

### Long-term

7. **Advanced patterns**:
   - Mixture of Experts (MoE)
   - Progressive unfreezing
   - Multi-task learning

8. **Production deployment**:
   - Model serving
   - Batch inference
   - API endpoints

---

## 📊 Model Size Reference

### Small Config (M1 Mac 16GB)

```python
from genailab.foundation.configs import SMALL_CONFIG

print(SMALL_CONFIG.embed_dim)              # 256
print(SMALL_CONFIG.depth)                  # 6
print(SMALL_CONFIG.num_heads)              # 8
print(SMALL_CONFIG.batch_size)             # 8
print(SMALL_CONFIG.gradient_accumulation_steps)  # 4
print(SMALL_CONFIG.use_checkpoint)         # True
```

**Use for**: Development, prototyping, testing on M1 Mac

### Medium Config (RunPod 24GB)

```python
from genailab.foundation.configs import MEDIUM_CONFIG

print(MEDIUM_CONFIG.embed_dim)             # 512
print(MEDIUM_CONFIG.depth)                 # 12
print(MEDIUM_CONFIG.num_heads)             # 8
print(MEDIUM_CONFIG.batch_size)            # 32
print(MEDIUM_CONFIG.use_flash_attention)   # True
```

**Use for**: Training, experimentation, RunPod instances

### Large Config (Cloud 40GB+)

```python
from genailab.foundation.configs import LARGE_CONFIG

print(LARGE_CONFIG.embed_dim)              # 768
print(LARGE_CONFIG.depth)                  # 24
print(LARGE_CONFIG.num_heads)              # 12
print(LARGE_CONFIG.num_tokens)             # 128
print(LARGE_CONFIG.batch_size)             # 64
```

**Use for**: Production, large-scale training, cloud instances

---

## 🐛 Troubleshooting

### "Out of Memory" on M1 Mac

**Solution 1**: Reduce batch size
```python
config = SMALL_CONFIG
config.batch_size = 4  # Reduce from 8
config.gradient_accumulation_steps = 8  # Increase to maintain effective batch
```

**Solution 2**: Enable gradient checkpointing
```python
config.use_checkpoint = True
```

**Solution 3**: Reduce model size
```python
config.embed_dim = 128  # Reduce from 256
config.depth = 4  # Reduce from 6
```

### LoRA Not Reducing Parameters

**Check**: Verify LoRA was applied correctly
```python
from genailab.foundation.tuning import LoRA

LoRA.print_trainable_parameters(model)
# Should show ~1-2% trainable
```

**Fix**: Ensure target modules match your model
```python
# Print all module names
for name, _ in model.named_modules():
    print(name)

# Apply to correct modules
model = apply_lora_to_model(
    model,
    target_modules=["your.actual.module.names"],
    rank=8,
)
```

### MPS (M1) Performance Issues

**Tip 1**: Use mixed precision
```python
from torch.cuda.amp import autocast

with autocast(device_type='cpu', dtype=torch.float16):
    output = model(input)
```

**Tip 2**: Avoid frequent CPU-GPU transfers
```python
# Bad: Transfer every step
for x in data:
    x = x.to('mps')
    
# Good: Transfer batch once
batch = batch.to('mps')
for x in batch:
    ...
```

---

## 📚 Related Documentation

- [Foundation Models Overview](leveraging_foundation_models_v2.md)
- [Transformer Data Shapes](data_shape_v2.md)
- [Latent Diffusion Series](../latent_diffusion/)
- [Package README](../../src/genailab/foundation/README.md)
- [Notebook Tutorials](../../notebooks/foundation_models/README.md)

---

## ✅ Verification Checklist

Before moving to notebooks:

- [ ] Resource detection works: `python -m genailab.foundation.configs.resource_profiles`
- [ ] Model configs print correctly: `python -m genailab.foundation.configs.model_configs`
- [ ] LoRA test runs: `python -m genailab.foundation.tuning.lora`
- [ ] Can import in Python: `from genailab.foundation import *`
- [ ] Memory estimates are reasonable for your hardware

---

## 🎯 Success Criteria

You'll know the framework is working when:

1. **Auto-detection works**: Correctly identifies your hardware
2. **LoRA reduces parameters**: From 100% to ~1-2% trainable
3. **Memory fits**: Model + optimizer + activations < available memory
4. **Training runs**: Can complete one epoch without OOM
5. **Saves/loads work**: LoRA weights save and restore correctly

---

## Next Session Preview

In the next session, we'll create:

1. **First notebook**: Interactive hardware detection and model sizing
2. **Adapter implementation**: Alternative to LoRA
3. **Freeze utilities**: Layer freezing strategies
4. **Comparison notebook**: LoRA vs Adapters vs Full fine-tuning

This will give you a complete toolkit for parameter-efficient fine-tuning!
