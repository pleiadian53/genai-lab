# Flow Matching Training

This document covers the practical aspects of training flow matching models: loss functions, network architectures, implementation details, training strategies, and best practices.

---

## Training Overview

### The Training Loop

Flow matching training is remarkably simple compared to diffusion models:

**Algorithm**:
```python
for batch in dataloader:
    # 1. Sample data and noise
    x0 = batch  # data
    x1 = sample_noise()  # noise
    
    # 2. Sample time uniformly
    t = uniform(0, 1)
    
    # 3. Interpolate
    xt = (1 - t) * x0 + t * x1
    
    # 4. Compute target velocity
    target = x1 - x0
    
    # 5. Predict and compute loss
    pred = model(xt, t)
    loss = mse_loss(pred, target)
    
    # 6. Update
    loss.backward()
    optimizer.step()
```

**Key simplicity**: Direct regression with MSE loss, no complex score matching objectives.

---

## Loss Functions

### Conditional Flow Matching Loss

The standard loss for flow matching:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - u_t(x_0, x_1) \right\|^2 \right]
$$

**Components**:

- $t \sim \text{Uniform}[0, 1]$: Random time
- $x_0 \sim p_{\text{data}}$: Data sample
- $x_1 \sim p_{\text{noise}}$: Noise sample
- $x_t = \psi_t(x_0, x_1)$: Interpolated point
- $u_t(x_0, x_1) = \frac{d}{dt}\psi_t(x_0, x_1)$: Target velocity

### Rectified Flow Loss

For linear interpolation $x_t = (1-t)x_0 + tx_1$:

$$
\mathcal{L}_{\text{RF}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

**Simplification**: Target velocity is constant: $u_t = x_1 - x_0$

**PyTorch implementation**:
```python
def rectified_flow_loss(model, x0, x1, t):
    """
    Compute rectified flow loss.
    
    Args:
        model: Neural network v_theta(x, t)
        x0: Data samples [batch_size, ...]
        x1: Noise samples [batch_size, ...]
        t: Time values [batch_size]
    
    Returns:
        loss: Scalar loss value
    """
    # Interpolate
    t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
    xt = (1 - t_expanded) * x0 + t_expanded * x1
    
    # Target velocity
    target = x1 - x0
    
    # Predict velocity
    pred = model(xt, t)
    
    # MSE loss
    loss = F.mse_loss(pred, target)
    
    return loss
```

### Variance-Preserving Loss

For VP interpolation $x_t = \sqrt{1-\sigma_t^2} \, x_0 + \sigma_t \, x_1$:

$$
\mathcal{L}_{\text{VP}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - \frac{d}{dt}\left(\sqrt{1-\sigma_t^2} \, x_0 + \sigma_t \, x_1\right) \right\|^2 \right]
$$

**Target velocity**:

$$
u_t = -\frac{\sigma_t \sigma_t'}{\sqrt{1-\sigma_t^2}} x_0 + \sigma_t' x_1
$$

where $\sigma_t' = \frac{d\sigma_t}{dt}$.

**Common choice**: $\sigma_t = t$, so $\sigma_t' = 1$:

$$
u_t = -\frac{t}{\sqrt{1-t^2}} x_0 + x_1
$$

### Weighted Loss

Add time-dependent weighting:

$$
\mathcal{L}_{\text{weighted}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ w(t) \left\| v_\theta(x_t, t) - u_t \right\|^2 \right]
$$

**Common weights**:

**1. Uniform**: $w(t) = 1$ (standard)

**2. SNR-based**: $w(t) = \frac{1}{\text{SNR}(t)}$ (from diffusion)

**3. Endpoint emphasis**: $w(t) = t^2$ or $w(t) = (1-t)^2$

**4. Min-SNR**: $w(t) = \min(\text{SNR}(t), \gamma)$ (clip large weights)

**When to use**:

- Uniform works well for most cases
- Endpoint emphasis if sampling quality at $t=0$ is critical
- SNR-based for VP flows to match diffusion performance

---

## Network Architectures

### Architecture Requirements

Flow matching networks $v_\theta(x, t)$ must:

1. **Input**: Accept data $x$ and time $t$
2. **Output**: Velocity vector same shape as $x$
3. **Time conditioning**: Incorporate $t$ throughout the network
4. **Expressiveness**: Capture complex velocity fields

### U-Net Architecture

**Standard choice for images**:

```python
class FlowMatchingUNet(nn.Module):
    def __init__(self, channels=3, dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4)
        )
        
        # Encoder
        self.downs = nn.ModuleList([])
        for mult in dim_mults:
            self.downs.append(
                ResnetBlock(channels, dim * mult, time_emb_dim=dim * 4)
            )
        
        # Bottleneck
        self.mid = ResnetBlock(dim * dim_mults[-1], dim * dim_mults[-1])
        
        # Decoder
        self.ups = nn.ModuleList([])
        for mult in reversed(dim_mults):
            self.ups.append(
                ResnetBlock(dim * mult * 2, dim * mult, time_emb_dim=dim * 4)
            )
        
        # Output
        self.final = nn.Conv2d(dim, channels, 1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        h = []
        for down in self.downs:
            x = down(x, t_emb)
            h.append(x)
        
        # Bottleneck
        x = self.mid(x, t_emb)
        
        # Decoder
        for up in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = up(x, t_emb)
        
        # Output velocity
        return self.final(x)
```

**Key components**:

- **Sinusoidal time embedding**: Encodes $t \in [0, 1]$
- **ResNet blocks**: With time conditioning via FiLM
- **Skip connections**: Preserve spatial information
- **U-Net structure**: Encoder-decoder with bottleneck

### Diffusion Transformer (DiT)

**Modern architecture for images**:

```python
class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
    
    def forward(self, x, c):
        # c is time + condition embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # MLP with AdaLN
        x = x + gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        
        return x

class FlowMatchingDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=2, dim=512, depth=12, num_heads=8):
        super().__init__()
        
        # Patchify
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        
        # Time + condition embedding
        self.time_embed = TimestepEmbedder(dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(depth)
        ])
        
        # Output
        self.final_layer = FinalLayer(dim, patch_size, 3)
    
    def forward(self, x, t):
        # Patchify
        x = self.patch_embed(x)
        
        # Time embedding
        c = self.time_embed(t)
        
        # Transformer
        for block in self.blocks:
            x = block(x, c)
        
        # Unpatchify and output velocity
        return self.final_layer(x)
```

**Advantages**:

- **Scalability**: Scales better to large models
- **Flexibility**: Handles variable resolutions
- **Attention**: Captures long-range dependencies
- **Modern**: State-of-the-art for image generation

### Time Conditioning

**Sinusoidal embedding** (standard):

```python
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
```

**FiLM conditioning** (Feature-wise Linear Modulation):

```python
class FiLM(nn.Module):
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.scale_shift = nn.Linear(time_emb_dim, dim * 2)
    
    def forward(self, x, time_emb):
        scale, shift = self.scale_shift(time_emb).chunk(2, dim=-1)
        return x * (1 + scale) + shift
```

**AdaLN conditioning** (Adaptive Layer Normalization):

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

---

## Training Strategies

### Data Preprocessing

**Images**:
```python
# Normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
```

**Gene expression**:
```python
# Log-normalize and standardize
def preprocess_gene_expression(X):
    # Log1p transform
    X = np.log1p(X)
    
    # Standardize per gene
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return X
```

### Noise Distribution

**Standard Gaussian** (most common):
```python
x1 = torch.randn_like(x0)
```

**Matched variance**:
```python
# Match data variance
data_std = x0.std()
x1 = torch.randn_like(x0) * data_std
```

**Domain-specific** (for gene expression):
```python
# Sparse noise matching dropout structure
def sparse_noise(x0, dropout_rate=0.1):
    noise = torch.randn_like(x0)
    mask = torch.rand_like(x0) > dropout_rate
    return noise * mask
```

### Time Sampling

**Uniform** (standard):
```python
t = torch.rand(batch_size, device=device)
```

**Stratified** (better coverage):
```python
# Divide [0,1] into bins
n_bins = batch_size
bins = torch.linspace(0, 1, n_bins + 1, device=device)
t = bins[:-1] + torch.rand(n_bins, device=device) / n_bins
```

**Importance sampling** (emphasize difficult times):
```python
# More samples near t=0 (data)
t = torch.rand(batch_size, device=device) ** 2
```

### Batch Size and Learning Rate

**Batch size**:

- **Images**: 128-512 (larger is better for diverse pairs)
- **Gene expression**: 64-256 (depends on dataset size)
- **Rule of thumb**: As large as GPU memory allows

**Learning rate**:

- **AdamW**: 1e-4 to 5e-4 (standard)
- **Warmup**: 1000-5000 steps
- **Schedule**: Cosine decay or constant

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### EMA (Exponential Moving Average)

**Use EMA for better sampling quality**:

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Usage
ema = EMA(model, decay=0.9999)

for batch in dataloader:
    loss = train_step(model, batch)
    optimizer.step()
    ema.update()  # Update EMA after each step

# For sampling, use EMA weights
ema.apply_shadow()
samples = sample(model, ...)
ema.restore()
```

---

## Complete Training Script

### Full Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_flow_matching(
    model,
    train_loader,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
    use_ema=True,
    save_every=10
):
    """
    Complete training loop for flow matching.
    """
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # EMA
    if use_ema:
        ema = EMA(model, decay=0.9999)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, x0 in enumerate(pbar):
            x0 = x0.to(device)
            batch_size = x0.shape[0]
            
            # Sample noise
            x1 = torch.randn_like(x0)
            
            # Sample time
            t = torch.rand(batch_size, device=device)
            
            # Interpolate
            t_expanded = t.view(-1, *([1] * (x0.ndim - 1)))
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            
            # Target velocity
            target = x1 - x0
            
            # Predict velocity
            pred = model(xt, t)
            
            # Compute loss
            loss = F.mse_loss(pred, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update
            optimizer.step()
            
            # Update EMA
            if use_ema:
                ema.update()
            
            # Logging
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Scheduler step
        scheduler.step()
        
        # Log epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            if use_ema:
                checkpoint['ema_state_dict'] = ema.shadow
            
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
    
    return model
```

### Training Example

```python
# Load data
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Create model
model = FlowMatchingUNet(channels=3, dim=64).cuda()

# Train
trained_model = train_flow_matching(
    model,
    train_loader,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
    use_ema=True
)
```

---

## Reflow: Iterative Refinement

### The Reflow Algorithm

Reflow iteratively straightens flow paths for faster sampling.

**Algorithm**:
```python
def reflow(model, train_loader, num_iterations=3):
    """
    Iterative reflow to straighten paths.
    """
    models = [model]
    
    for iteration in range(1, num_iterations):
        print(f'Reflow iteration {iteration}')
        
        # Generate synthetic data using current model
        synthetic_data = []
        for x1 in tqdm(train_loader, desc='Generating synthetic data'):
            x1 = x1.to(device)
            # Sample from current model
            x0_synthetic = sample_ode(models[-1], x1)
            synthetic_data.append((x0_synthetic, x1))
        
        # Train new model on synthetic data
        new_model = FlowMatchingUNet(channels=3, dim=64).cuda()
        
        for epoch in range(num_epochs):
            for x0_syn, x1 in synthetic_data:
                # Standard flow matching training
                t = torch.rand(x0_syn.shape[0], device=device)
                t_exp = t.view(-1, *([1] * (x0_syn.ndim - 1)))
                xt = (1 - t_exp) * x0_syn + t_exp * x1
                
                target = x1 - x0_syn
                pred = new_model(xt, t)
                loss = F.mse_loss(pred, target)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        models.append(new_model)
    
    return models
```

**Effect**:

- **Iteration 1**: 50-100 steps needed
- **Iteration 2**: 20-30 steps needed
- **Iteration 3**: 10-15 steps needed

**Trade-off**: More training time for faster sampling.

---

## Monitoring and Debugging

### Training Metrics

**Track during training**:

1. **Loss**: Should decrease steadily
2. **Gradient norm**: Should be stable (clip if exploding)
3. **Learning rate**: Follow schedule
4. **Sample quality**: Generate samples periodically

```python
# Log metrics
wandb.log({
    'loss': loss.item(),
    'grad_norm': grad_norm,
    'lr': scheduler.get_last_lr()[0],
    'epoch': epoch
})

# Generate samples every N epochs
if epoch % 10 == 0:
    with torch.no_grad():
        samples = sample_ode(model, num_samples=64)
        wandb.log({'samples': wandb.Image(samples)})
```

### Common Issues

**1. Loss not decreasing**:

- Check learning rate (try 1e-4)
- Check data normalization
- Verify target velocity computation
- Increase batch size

**2. NaN loss**:

- Gradient clipping (clip_grad_norm)
- Lower learning rate
- Check for inf/nan in data
- Use mixed precision carefully

**3. Poor sample quality**:

- Train longer
- Use EMA
- Increase model capacity
- Try more sampling steps
- Check noise distribution

**4. Mode collapse**:

- Increase batch size
- Use diverse noise samples
- Check data augmentation
- Verify loss computation

---

## Best Practices

### Do's

✅ **Use EMA** for better sampling quality
✅ **Clip gradients** to prevent instability
✅ **Normalize data** to [-1, 1] or standardize
✅ **Use large batch sizes** for diverse pairs
✅ **Monitor samples** during training
✅ **Save checkpoints** regularly
✅ **Use mixed precision** for faster training (with caution)

### Don'ts

❌ **Don't skip EMA** (significant quality improvement)
❌ **Don't use tiny batch sizes** (<32)
❌ **Don't ignore gradient norms** (clip if >1.0)
❌ **Don't overtrain** (diminishing returns after convergence)
❌ **Don't forget data normalization**

---

## Summary

### Key Training Steps

1. **Sample** data $x_0$ and noise $x_1$
2. **Sample** time $t \sim U[0, 1]$
3. **Interpolate** $x_t = (1-t)x_0 + tx_1$
4. **Compute target** $u_t = x_1 - x_0$
5. **Predict** $v_\theta(x_t, t)$
6. **Optimize** MSE loss

### Key Hyperparameters

- **Batch size**: 128-512 (larger is better)
- **Learning rate**: 1e-4 to 5e-4
- **EMA decay**: 0.9999
- **Gradient clip**: 1.0
- **Epochs**: 100-500 (depends on dataset)

### Architecture Choices

- **Images**: U-Net or DiT
- **Sequences**: Transformer
- **Time conditioning**: Sinusoidal + FiLM/AdaLN

---

## Related Documents

- [Flow Matching Foundations](01_flow_matching_foundations.md) — Theory and mathematics
- [Flow Matching Sampling](03_flow_matching_sampling.md) — ODE solvers and sampling
- [DDPM Training](../DDPM/02_ddpm_training.md) — Comparison with diffusion training
- [Diffusion Transformers](../diffusion/DiT/diffusion_transformer.md) — DiT architecture

---

## References

1. **Lipman, Y., et al. (2023)**. Flow Matching for Generative Modeling. *ICLR*.
2. **Liu, X., et al. (2023)**. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR*.
3. **Peebles, W., & Xie, S. (2023)**. Scalable Diffusion Models with Transformers. *ICCV*.
4. **Karras, T., et al. (2022)**. Elucidating the Design Space of Diffusion-Based Generative Models. *NeurIPS*.
