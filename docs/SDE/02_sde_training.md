# Training Diffusion Models: The SDE Perspective

This document explains how diffusion models are trained from the SDE perspective, clarifying a common source of confusion: **SDE solvers are NOT used during training**.

Training uses closed-form solutions of the forward SDE, making it computationally simple and efficient.

---

## Overview

### The Key Insight

**Common misconception**: Since diffusion models are based on SDEs, training must involve solving SDEs.

**Reality**: Training uses **closed-form marginals** from the forward SDE solution. No numerical SDE solver is needed.

### Training vs Sampling

| Phase | SDE Solver? | What's Used |
|-------|-------------|-------------|
| **Training** | ❌ No | Closed-form $q(x_t \mid x_0)$ |
| **Sampling** | ✅ Yes | Numerical discretization of reverse SDE/ODE |

**This document focuses on training** — see [03_sde_sampling.md](03_sde_sampling.md) for sampling.

---

## The Forward SDE Solution

### VP-SDE (Variance-Preserving)

The forward SDE is:

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

**Closed-form solution** (see [03_solving_vpsde.md](03_solving_vpsde.md) for derivation):

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon
$$

where:

- $\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)$
- $\epsilon \sim \mathcal{N}(0, I)$

**Key property**: We can sample $x_t$ directly from $x_0$ without simulating the SDE!

### Marginal Distribution

The marginal distribution at time $t$ is:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)
$$

**This is all we need for training** — no SDE solver required.

---

## Training Objective

### Score Matching

The goal is to learn the **score function** $\nabla_x \log p_t(x)$.

**Theoretical objective** (intractable):

$$
\mathbb{E}_{t, x_t} \left[ \left\| s_\theta(x_t, t) - \nabla_x \log p_t(x_t) \right\|^2 \right]
$$

**Practical objective** (tractable via denoising score matching):

$$
\mathbb{E}_{t, x_0, \epsilon} \left[ \lambda(t) \left\| s_\theta(x_t, t) - \nabla_x \log q(x_t \mid x_0) \right\|^2 \right]
$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.

### Connection to Noise Prediction

The conditional score has a simple form:

$$
\nabla_x \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

**In practice**, we predict noise instead of score:

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$

**Relationship**:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}
$$

### DDPM Training Loss

The standard DDPM loss is:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2 \right]
$$

**Why this works**:
1. Equivalent to score matching with specific weighting $\lambda(t) = 1$
2. Simpler to implement (predict noise, not score)
3. Better empirical performance

---

## Training Algorithm

### Pseudocode

```python
# Training loop for VP-SDE diffusion model

for epoch in range(num_epochs):
    for batch in dataloader:
        x_0 = batch  # Clean data samples
        
        # 1. Sample random timesteps
        t = torch.rand(batch_size) * T  # t ∈ [0, T]
        
        # 2. Sample noise
        epsilon = torch.randn_like(x_0)
        
        # 3. Compute α̅_t (cumulative product)
        alpha_bar_t = compute_alpha_bar(t)  # exp(-∫₀ᵗ β(s)ds)
        
        # 4. Create noisy samples (closed-form!)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        # 5. Predict noise
        epsilon_pred = model(x_t, t)
        
        # 6. Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon)
        
        # 7. Update model
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Key observation**: Step 4 uses the closed-form solution — no SDE solver!

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionTrainer:
    def __init__(self, model, beta_schedule, T=1.0):
        self.model = model
        self.T = T
        
        # Precompute α̅_t for efficiency
        # In continuous time: α̅(t) = exp(-∫₀ᵗ β(s)ds)
        # For linear schedule: β(t) = β_min + t(β_max - β_min)
        self.beta_min = beta_schedule['beta_min']
        self.beta_max = beta_schedule['beta_max']
    
    def compute_alpha_bar(self, t):
        """Compute α̅_t = exp(-∫₀ᵗ β(s)ds) for linear schedule."""
        # For β(s) = β_min + s(β_max - β_min):
        # ∫₀ᵗ β(s)ds = β_min*t + (β_max - β_min)*t²/2
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        return torch.exp(-integral)
    
    def training_step(self, x_0):
        """Single training step."""
        batch_size = x_0.shape[0]
        
        # Sample timesteps uniformly
        t = torch.rand(batch_size, device=x_0.device) * self.T
        
        # Sample noise
        epsilon = torch.randn_like(x_0)
        
        # Compute α̅_t
        alpha_bar_t = self.compute_alpha_bar(t)
        
        # Reshape for broadcasting
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)  # For images
        
        # Create noisy samples (closed-form marginal)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
        
        # Predict noise
        epsilon_pred = self.model(x_t, t)
        
        # Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon)
        
        return loss

# Usage
model = UNet(...)  # Your noise prediction network
trainer = DiffusionTrainer(
    model=model,
    beta_schedule={'beta_min': 0.1, 'beta_max': 20.0},
    T=1.0
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = trainer.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Why No SDE Solver in Training?

### Mathematical Reason

The forward SDE has a **closed-form solution**:

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon
$$

**This means**:

- We can sample $x_t$ directly from $x_0$
- No need to simulate the SDE step-by-step
- Training is fast and exact

### Computational Advantage

**With SDE solver** (hypothetical, not used):
- Start with $x_0$
- Simulate many small steps: $x_0 \to x_{\Delta t} \to x_{2\Delta t} \to \cdots \to x_t$
- Slow and introduces discretization error

**With closed-form** (actual approach):
- Directly compute $x_t$ from $x_0$ in one step
- Fast and exact
- This is what makes training practical!

---

## Connection to DDPM

### Discrete DDPM Training

DDPM uses discrete timesteps $k = 1, 2, \ldots, N$:

```python
# DDPM training
t = torch.randint(0, N, (batch_size,))
alpha_bar_t = alpha_bar[t]  # Precomputed cumulative product
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

### Continuous SDE Training

SDE uses continuous time $t \in [0, T]$:

```python
# SDE training
t = torch.rand(batch_size) * T
alpha_bar_t = compute_alpha_bar(t)  # exp(-∫₀ᵗ β(s)ds)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

**Key difference**: How $\bar{\alpha}_t$ is computed
- **DDPM**: Discrete products $\prod_{i=1}^t \alpha_i$
- **SDE**: Continuous integral $\exp(-\int_0^t \beta(s)ds)$

**Training loop**: Identical structure!

---

## Noise Schedules

### Linear Schedule (Most Common)

$$
\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min}), \quad t \in [0, 1]
$$

**Cumulative**:

$$
\bar{\alpha}_t = \exp\left(-\beta_{\min} t - \frac{1}{2}(\beta_{\max} - \beta_{\min}) t^2\right)
$$

**Typical values**: $\beta_{\min} = 0.1$, $\beta_{\max} = 20$

### Cosine Schedule

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
$$

where $s = 0.008$ is a small offset.

**Advantage**: More uniform SNR across timesteps.

### Implementation

```python
def linear_beta_schedule(t, beta_min=0.1, beta_max=20.0):
    """Linear noise schedule."""
    return beta_min + t * (beta_max - beta_min)

def compute_alpha_bar_linear(t, beta_min=0.1, beta_max=20.0):
    """Compute α̅_t for linear schedule."""
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    return torch.exp(-integral)

def compute_alpha_bar_cosine(t, s=0.008):
    """Compute α̅_t for cosine schedule."""
    def f(t):
        return torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    return f(t) / f(torch.zeros_like(t))
```

---

## Training Strategies

### Time Sampling

**Uniform sampling** (standard):
```python
t = torch.rand(batch_size) * T
```

**Importance sampling** (advanced):
- Sample more frequently at difficult timesteps
- Weight loss by inverse sampling probability

### Weighting

**Simple weighting** (DDPM):

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2 \right]
$$

**SNR weighting** (improved):

$$

\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \frac{1}{\text{SNR}(t)} \left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2 \right]
$$

where $\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$.

### Exponential Moving Average (EMA)

Maintain EMA of model parameters for better sample quality:

```python
ema_model = copy.deepcopy(model)
ema_decay = 0.9999

# After each training step
for param_ema, param in zip(ema_model.parameters(), model.parameters()):
    param_ema.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

# Use ema_model for sampling
```

---

## Network Architecture

### U-Net (Standard for Images)

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, 
                 base_channels=128, time_emb_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder
        self.encoder = nn.ModuleList([
            ResBlock(in_channels, base_channels, time_emb_dim),
            ResBlock(base_channels, base_channels * 2, time_emb_dim),
            ResBlock(base_channels * 2, base_channels * 4, time_emb_dim),
        ])
        
        # Bottleneck
        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # Decoder
        self.decoder = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels * 2, time_emb_dim),
            ResBlock(base_channels * 4, base_channels, time_emb_dim),
            ResBlock(base_channels * 2, base_channels, time_emb_dim),
        ])
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)
    
    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t)
        
        # Encoder with skip connections
        skips = []
        for block in self.encoder:
            x = block(x, t_emb)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        
        # Bottleneck
        x = self.bottleneck(x, t_emb)
        
        # Decoder with skip connections
        for block in self.decoder:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t_emb)
        
        return self.out(x)
```

### Time Embedding

**Sinusoidal embeddings** (like Transformer positional encoding):

```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

---

## Training Tips

### Hyperparameters

**Learning rate**: 

- Start: $1 \times 10^{-4}$
- Use cosine annealing or constant

**Batch size**:

- Larger is better (256-2048)
- Use gradient accumulation if memory limited

**Training steps**:

- Images: 500K - 1M steps
- High resolution: 1M+ steps

### Monitoring

**Track during training**:
1. **Loss**: Should decrease steadily
2. **Sample quality**: Generate samples every N steps
3. **EMA decay**: Use EMA model for evaluation

**Early stopping**:

- Monitor FID or other metrics on validation set
- Diffusion models benefit from long training

### Debugging

**Common issues**:

1. **Loss not decreasing**:
   - Check data normalization (typically [-1, 1])
   - Verify $\bar{\alpha}_t$ computation
   - Check time embedding

2. **Poor sample quality**:
   - Train longer
   - Use EMA
   - Try different noise schedule

3. **Mode collapse**:
   - Rare in diffusion models
   - Check data diversity

---

## Comparison: Training vs Sampling

### Training (This Document)

**What**: Learn to predict noise $\epsilon_\theta(x_t, t)$

**How**:
1. Sample $x_0$, $t$, $\epsilon$
2. Compute $x_t$ using closed-form
3. Predict $\epsilon_\theta(x_t, t)$
4. Minimize $\|\epsilon_\theta - \epsilon\|^2$

**SDE solver**: ❌ Not used

**Speed**: Fast (single forward pass per sample)

### Sampling (See 03_sde_sampling.md)

**What**: Generate samples from learned model

**How**:
1. Start with $x_T \sim \mathcal{N}(0, I)$
2. Iteratively denoise using reverse SDE/ODE
3. End with $x_0 \approx$ data sample

**SDE solver**: ✅ Used (Euler, RK4, adaptive)

**Speed**: Slow (many steps required)

---

## Key Takeaways

### Conceptual

1. **Training uses closed-form marginals** — no SDE solver needed
2. **The forward SDE solution is exact** — we can sample $x_t$ directly from $x_0$
3. **Score matching = denoising** — predicting noise is equivalent to learning scores
4. **Training is the same as DDPM** — just continuous time instead of discrete

### Practical

1. **Implementation is simple**: Sample $t$, add noise, predict, compute loss
2. **No numerical integration** during training
3. **Fast and stable** — closed-form solution avoids discretization errors
4. **SDE solvers only used for sampling** (generation)

### Mathematical

1. **Forward SDE**: $dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw$
2. **Closed-form**: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
3. **Training loss**: $\mathbb{E}[\|\epsilon_\theta(x_t, t) - \epsilon\|^2]$
4. **No solver needed**: Direct sampling from $q(x_t | x_0)$

---

## Related Documents

### SDE Documentation
- [00_sde_overview.md](00_sde_overview.md) — High-level SDE introduction
- [01_diffusion_sde_view.md](01_diffusion_sde_view.md) — Detailed SDE formulation
- [03_solving_vpsde.md](03_solving_vpsde.md) — Derivation of closed-form solution
- [03_sde_sampling.md](03_sde_sampling.md) — How to sample (where SDE solvers are used)

### Related Topics
- [DDPM Training](../DDPM/02_ddpm_training.md) — Discrete version
- [Flow Matching Training](../flow_matching/02_flow_matching_training.md) — Alternative approach

---

## Summary

**Training diffusion models from the SDE perspective is simple**:

1. **Use closed-form marginals** from the forward SDE solution
2. **No SDE solver required** — direct sampling of $x_t$ from $x_0$
3. **Same as DDPM training** — just continuous time
4. **Fast and exact** — no discretization errors

**The confusion arises because**:

- The SDE formulation seems complex
- But training only uses the closed-form solution
- SDE solvers are only needed for sampling (generation)

**Bottom line**: Training is straightforward regression on noise prediction, using the exact solution of the forward SDE.
