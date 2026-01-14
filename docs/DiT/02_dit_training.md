# DiT Training: Rectified Flow with Transformers

This document explains how to train Diffusion Transformers (DiT) with rectified flow, covering the complete training pipeline from data preparation to optimization strategies.

**Prerequisites**: Understanding of [DiT architecture](01_dit_foundations.md) and [rectified flow](../flow_matching/02_flow_matching_training.md).

---

## Overview

Training DiT with rectified flow is remarkably simple compared to DDPM:

**Key advantages**:
- No noise schedules to tune
- No variance parameterization
- Direct regression on velocity
- Stable training dynamics

**Training loop**:
```python
for batch in dataloader:
    x_0 = batch  # Real data
    x_1 = torch.randn_like(x_0)  # Noise
    t = torch.rand(batch_size)  # Random time
    
    x_t = t * x_1 + (1 - t) * x_0  # Interpolate
    v_pred = model(x_t, t)  # Predict velocity
    
    target = x_1 - x_0  # True velocity
    loss = F.mse_loss(v_pred, target)
    
    loss.backward()
    optimizer.step()
```

---

## 1. Data Preparation

### 1.1 Image Data

**Standard preprocessing**:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),  # Resize to target resolution
    transforms.CenterCrop(256),  # Center crop
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

dataset = ImageFolder(root='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
```

**Key points**:
- Normalize to `[-1, 1]` (not `[0, 1]`)
- Use data augmentation (flips, crops)
- Batch size as large as GPU memory allows

### 1.2 Conditional Data

**Class-conditional** (e.g., ImageNet):

```python
class ConditionalDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = ImageFolder(root, transform)
        self.num_classes = len(self.dataset.classes)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
    
    def __len__(self):
        return len(self.dataset)
```

**Text-conditional** (e.g., text-to-image):

```python
class TextImageDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform, tokenizer):
        self.images = load_images(image_dir)
        self.captions = load_captions(caption_file)
        self.transform = transform
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        caption = self.captions[idx]
        tokens = self.tokenizer(caption, max_length=77, padding='max_length')
        return image, tokens
```

### 1.3 Gene Expression Data

**Preprocessing**:

```python
import scanpy as sc

# Load data
adata = sc.read_h5ad('data/expression.h5ad')

# Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Convert to tensor
expression = torch.tensor(adata.X.toarray(), dtype=torch.float32)

# Create dataset
class GeneExpressionDataset(Dataset):
    def __init__(self, expression, conditions=None):
        self.expression = expression
        self.conditions = conditions
    
    def __getitem__(self, idx):
        x = self.expression[idx]
        if self.conditions is not None:
            c = self.conditions[idx]
            return x, c
        return x
    
    def __len__(self):
        return len(self.expression)
```

---

## 2. Model Architecture

### 2.1 Instantiate DiT

```python
from dit import DiT

model = DiT(
    img_size=256,          # Image resolution
    patch_size=8,          # Patch size (8×8)
    in_channels=3,         # RGB
    embed_dim=1152,        # Hidden dimension (DiT-XL)
    depth=28,              # Number of transformer blocks
    num_heads=16,          # Attention heads
    mlp_ratio=4.0,         # MLP expansion ratio
    num_classes=1000       # For class conditioning (ImageNet)
)

# Move to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params / 1e6:.1f}M parameters")
```

### 2.2 Model Sizes

**Choose based on compute budget**:

| Model | Params | Patch Size | Training Time (ImageNet) |
|-------|--------|------------|--------------------------|
| DiT-S/8 | 33M | 8×8 | ~1 day (8 GPUs) |
| DiT-B/8 | 130M | 8×8 | ~2 days (8 GPUs) |
| DiT-L/8 | 458M | 8×8 | ~4 days (8 GPUs) |
| DiT-XL/8 | 675M | 8×8 | ~7 days (8 GPUs) |

**Recommendation**: Start with DiT-B for prototyping, scale to DiT-XL for best results.

---

## 3. Training Objective

### 3.1 Rectified Flow Loss

**Simple regression**:

$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

where:
- $x_0 \sim p_{\text{data}}$ (real data)
- $x_1 \sim \mathcal{N}(0, I)$ (noise)
- $x_t = t x_1 + (1-t) x_0$ (linear interpolation)
- $t \sim \mathcal{U}(0, 1)$ (uniform time)

### 3.2 Implementation

```python
def compute_loss(model, x_0, t=None, condition=None):
    """
    Compute rectified flow loss.
    
    Args:
        model: DiT model
        x_0: Real data (B, C, H, W)
        t: Timesteps (B,) - if None, sample uniformly
        condition: Optional conditioning (class labels, text, etc.)
    
    Returns:
        loss: Scalar loss
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # Sample timesteps
    if t is None:
        t = torch.rand(batch_size, device=device)
    
    # Sample noise
    x_1 = torch.randn_like(x_0)
    
    # Linear interpolation
    t_expanded = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    
    # Predict velocity
    v_pred = model(x_t, t, condition)
    
    # Compute target
    target = x_1 - x_0
    
    # MSE loss
    loss = F.mse_loss(v_pred, target)
    
    return loss
```

### 3.3 Conditional Training

**Class-conditional**:

```python
def compute_loss_conditional(model, x_0, labels):
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # Sample timesteps
    t = torch.rand(batch_size, device=device)
    
    # Sample noise
    x_1 = torch.randn_like(x_0)
    
    # Interpolate
    t_expanded = t.view(-1, 1, 1, 1)
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    
    # Predict with class conditioning
    v_pred = model(x_t, t, y=labels)
    
    # Target
    target = x_1 - x_0
    
    # Loss
    loss = F.mse_loss(v_pred, target)
    
    return loss
```

**Classifier-free guidance training**:

```python
def compute_loss_cfg(model, x_0, labels, p_uncond=0.1):
    """
    Train with classifier-free guidance.
    
    Args:
        p_uncond: Probability of dropping conditioning (typically 0.1)
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # Sample timesteps
    t = torch.rand(batch_size, device=device)
    
    # Sample noise
    x_1 = torch.randn_like(x_0)
    
    # Interpolate
    t_expanded = t.view(-1, 1, 1, 1)
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    
    # Randomly drop conditioning
    mask = torch.rand(batch_size, device=device) < p_uncond
    labels_masked = labels.clone()
    labels_masked[mask] = model.num_classes  # Use special "null" class
    
    # Predict
    v_pred = model(x_t, t, y=labels_masked)
    
    # Target
    target = x_1 - x_0
    
    # Loss
    loss = F.mse_loss(v_pred, target)
    
    return loss
```

---

## 4. Training Loop

### 4.1 Basic Training Loop

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Model
model = DiT(...).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

# Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Compute loss
        loss = compute_loss_conditional(model, images, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Scheduler step
    scheduler.step()
    
    # Epoch logging
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Save checkpoint
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/dit_epoch_{epoch}.pt')
```

### 4.2 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Scaler for mixed precision
scaler = GradScaler()

for epoch in range(num_epochs):
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward with autocast
        with autocast():
            loss = compute_loss_conditional(model, images, labels)
        
        # Backward with scaler
        scaler.scale(loss).backward()
        
        # Unscale and clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        scaler.step(optimizer)
        scaler.update()
```

**Benefits**:
- 2× faster training
- 2× less memory
- Minimal quality loss

### 4.3 Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = DiT(...).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        
        for images, labels in dataloader:
            images = images.to(rank)
            labels = labels.to(rank)
            
            loss = compute_loss_conditional(model, images, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    cleanup()

# Launch
import torch.multiprocessing as mp
world_size = torch.cuda.device_count()
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

---

## 5. Optimization Strategies

### 5.1 Learning Rate

**Recommended schedule**:

```python
# Base learning rate
base_lr = 1e-4

# Warmup
warmup_epochs = 5
warmup_lr_schedule = torch.linspace(0, base_lr, warmup_epochs * len(dataloader))

# Cosine decay
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

# Combined
for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        # Warmup
        for batch_idx in range(len(dataloader)):
            step = epoch * len(dataloader) + batch_idx
            lr = warmup_lr_schedule[step]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        # Cosine decay
        scheduler.step()
```

**Typical values**:
- Base LR: `1e-4` (DiT-B/L/XL)
- Warmup: 5-10 epochs
- Decay: Cosine to `1e-6`

### 5.2 Batch Size

**Scaling rule**: Larger batch = better, but diminishing returns

| Batch Size | GPUs | Memory per GPU | Training Speed |
|------------|------|----------------|----------------|
| 256 | 1 | 40GB | Baseline |
| 512 | 2 | 40GB | 1.8× |
| 1024 | 4 | 40GB | 3.2× |
| 2048 | 8 | 40GB | 5.5× |

**Effective batch size** with gradient accumulation:

```python
effective_batch_size = 2048
batch_size_per_gpu = 256
accumulation_steps = effective_batch_size // (batch_size_per_gpu * num_gpus)

for batch_idx, (images, labels) in enumerate(dataloader):
    loss = compute_loss_conditional(model, images, labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5.3 Weight Decay

**AdamW with weight decay**:

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.0  # No weight decay for DiT (empirically better)
)
```

**Note**: DiT works well without weight decay, unlike some other models.

### 5.4 Gradient Clipping

**Prevent gradient explosion**:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Typical value**: `max_norm=1.0`

---

## 6. Exponential Moving Average (EMA)

### 6.1 Why EMA?

**Problem**: Model weights fluctuate during training

**Solution**: Maintain moving average of weights

**Benefits**:
- Smoother convergence
- Better sample quality
- Minimal overhead

### 6.2 Implementation

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Usage
ema = EMA(model, decay=0.9999)

for epoch in range(num_epochs):
    for images, labels in dataloader:
        # Training step
        loss = compute_loss_conditional(model, images, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema.update()

# For sampling, use EMA weights
ema.apply_shadow()
samples = sample(model, ...)
ema.restore()
```

**Typical decay**: `0.9999` (slower) or `0.999` (faster)

---

## 7. Monitoring and Debugging

### 7.1 Metrics to Track

**During training**:
1. **Loss**: Should decrease steadily
2. **Learning rate**: Check schedule is correct
3. **Gradient norm**: Should be stable (not exploding)
4. **Sample quality**: Generate samples periodically

```python
import wandb

wandb.init(project="dit-training")

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(dataloader):
        loss = compute_loss_conditional(model, images, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log
        wandb.log({
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
        })
```

### 7.2 Validation

**Generate samples periodically**:

```python
@torch.no_grad()
def validate(model, num_samples=16, num_steps=50):
    model.eval()
    
    # Generate samples
    x = torch.randn(num_samples, 3, 256, 256, device=device)
    dt = 1.0 / num_steps
    
    for k in range(num_steps):
        t = torch.full((num_samples,), k * dt, device=device)
        v = model(x, t)
        x = x + v * dt
    
    # Denormalize
    x = (x + 1) / 2  # [-1, 1] → [0, 1]
    x = torch.clamp(x, 0, 1)
    
    model.train()
    return x

# During training
if epoch % 10 == 0:
    samples = validate(model)
    wandb.log({'samples': [wandb.Image(img) for img in samples]})
```

### 7.3 Common Issues

**Loss not decreasing**:
- Check data normalization (should be [-1, 1])
- Verify learning rate (try 1e-4)
- Check model initialization

**Gradient explosion**:
- Use gradient clipping (max_norm=1.0)
- Reduce learning rate
- Check for NaN in data

**Poor sample quality**:
- Train longer (DiT needs 400K+ steps)
- Use EMA
- Try smaller patch size (better quality, slower)

---

## 8. Training Hyperparameters

### 8.1 Recommended Settings

**For ImageNet (256×256)**:

```python
config = {
    # Model
    'model': 'DiT-XL/8',
    'img_size': 256,
    'patch_size': 8,
    'embed_dim': 1152,
    'depth': 28,
    'num_heads': 16,
    
    # Training
    'batch_size': 256,  # Per GPU
    'num_gpus': 8,
    'effective_batch_size': 2048,
    'num_epochs': 1400,  # ~400K steps
    
    # Optimization
    'lr': 1e-4,
    'weight_decay': 0.0,
    'warmup_epochs': 5,
    'grad_clip': 1.0,
    
    # EMA
    'ema_decay': 0.9999,
    
    # Mixed precision
    'use_amp': True,
    
    # Logging
    'log_every': 100,
    'sample_every': 1000,
    'save_every': 10000,
}
```

### 8.2 Scaling to Different Resolutions

| Resolution | Patch Size | Tokens | Batch Size | Training Time |
|------------|------------|--------|------------|---------------|
| 64×64 | 4×4 | 256 | 512 | 1 day |
| 128×128 | 8×8 | 256 | 256 | 2 days |
| 256×256 | 8×8 | 1024 | 256 | 7 days |
| 512×512 | 16×16 | 1024 | 128 | 14 days |

**Rule of thumb**: Larger resolution = more tokens = more memory = smaller batch size

---

## 9. Advanced Techniques

### 9.1 Progressive Growing

**Start with low resolution, gradually increase**:

```python
# Stage 1: Train at 64×64
model_64 = DiT(img_size=64, patch_size=4, ...)
train(model_64, resolution=64, epochs=100)

# Stage 2: Upsample to 128×128
model_128 = DiT(img_size=128, patch_size=8, ...)
model_128.load_state_dict(model_64.state_dict(), strict=False)
train(model_128, resolution=128, epochs=100)

# Stage 3: Upsample to 256×256
model_256 = DiT(img_size=256, patch_size=8, ...)
model_256.load_state_dict(model_128.state_dict(), strict=False)
train(model_256, resolution=256, epochs=200)
```

**Benefits**: Faster convergence, better quality

### 9.2 Latent Diffusion

**Train in latent space** (like Stable Diffusion):

```python
# Pretrained VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Encode images to latent
with torch.no_grad():
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * 0.18215  # Scaling factor

# Train DiT on latents
model = DiT(in_channels=4, ...)  # VAE latent has 4 channels
loss = compute_loss(model, latents)
```

**Benefits**:
- 4-8× faster training
- 4-8× less memory
- Similar quality

### 9.3 Multi-Scale Training

**Train on multiple resolutions simultaneously**:

```python
resolutions = [128, 192, 256]

for images, labels in dataloader:
    # Random resolution
    res = random.choice(resolutions)
    images_resized = F.interpolate(images, size=(res, res))
    
    # Train
    loss = compute_loss_conditional(model, images_resized, labels)
    loss.backward()
    optimizer.step()
```

**Benefits**: Better generalization, flexible inference

---

## 10. Comparison with DDPM Training

| Aspect | DDPM | DiT + Rectified Flow |
|--------|------|----------------------|
| **Objective** | Noise prediction | Velocity prediction |
| **Noise schedule** | Required (β_t) | Not needed |
| **Variance** | Parameterized | Not needed |
| **Loss weighting** | SNR-based | Uniform |
| **Training stability** | Moderate | High |
| **Hyperparameters** | Many | Few |

**Key advantage**: Rectified flow is simpler and more stable.

---

## Key Takeaways

### Training Process

1. **Data**: Normalize to [-1, 1], augment
2. **Model**: Choose size based on compute
3. **Loss**: Simple MSE on velocity
4. **Optimization**: AdamW, cosine schedule, gradient clipping
5. **EMA**: Use for better sample quality

### Hyperparameters

1. **Learning rate**: 1e-4 with warmup
2. **Batch size**: As large as possible (2048 effective)
3. **Training steps**: 400K+ for ImageNet
4. **EMA decay**: 0.9999
5. **Gradient clip**: 1.0

### Best Practices

1. **Use mixed precision** (2× speedup)
2. **Use EMA** (better quality)
3. **Monitor gradients** (catch instabilities)
4. **Generate samples** (visual feedback)
5. **Save checkpoints** (resume training)

---

## Related Documents

- [01_dit_foundations.md](01_dit_foundations.md) — Architecture details
- [03_dit_sampling.md](03_dit_sampling.md) — Sampling strategies
- [Flow Matching Training](../flow_matching/02_flow_matching_training.md) — Rectified flow theory

---

## References

- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers"
- Liu et al. (2022): "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
