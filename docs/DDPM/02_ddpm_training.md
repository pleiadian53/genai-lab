# DDPM Training: From Theory to Practice

This document bridges the mathematical foundations of DDPM to practical training considerations, covering loss functions, architectures, conditioning strategies, and training tips.

---

## Overview

Training a DDPM involves:
1. **Loss function**: Simple MSE vs. weighted ELBO
2. **Architecture**: Choosing the right network for your data
3. **Conditioning**: How to incorporate additional information
4. **Optimization**: Hyperparameters and training strategies

**Goal**: Understand the practical decisions that make DDPM training successful.

---

## Training Objective

### Simple Loss (What You Actually Use)

The **simple loss** from Ho et al. (2020):

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

**Algorithm**:
```python
1. Sample x_0 ~ q(x_0)           # Real data
2. Sample t ~ Uniform({1,...,T})  # Random timestep
3. Sample ε ~ N(0, I)             # Noise
4. Compute x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
5. Predict ε_θ(x_t, t)
6. Loss = ||ε - ε_θ(x_t, t)||²
7. Update θ via gradient descent
```

### Why Simple Loss Works

The simple loss **ignores the time-dependent weighting** in the full ELBO:

$$
L_{\text{ELBO}} = \mathbb{E}_{t, x_0, \epsilon} \left[\frac{1}{2\sigma_t^2} \|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

**Empirical finding** (Ho et al., 2020): The simple loss produces **better sample quality** despite being theoretically less justified.

**Intuition**: The simple loss gives equal weight to all timesteps, preventing the model from over-focusing on high-noise timesteps.

---

## Loss Function Variants

### 1. Noise Prediction (Standard)

$$
L = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

**Pros**:

- Most common formulation
- Works well empirically
- Easy to implement

**When to use**: Default choice for most applications

### 2. Score Prediction

$$
L = \mathbb{E}_{t, x_0, \epsilon} \left[\left\|\nabla_{x_t} \log q(x_t) - s_\theta(x_t, t)\right\|^2\right]
$$

Equivalent to noise prediction with scaling:

$$
s_\theta(x_t, t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)
$$

**When to use**: When connecting to score matching literature

### 3. $x_0$ Prediction

$$
L = \mathbb{E}_{t, x_0, \epsilon} \left[\|x_0 - \hat{x}_0(x_t, t)\|^2\right]
$$

where:

$$
\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

**Pros**:

- Direct prediction of clean data
- Can be easier to interpret

**Cons**:

- Can be less stable (predicting data vs. noise)

**When to use**: When you want direct $x_0$ estimates (e.g., for visualization)

### 4. Velocity Prediction (Rectified Flow)

$$
L = \mathbb{E}_{t, x_0, x_1} \left[\|v_t - v_\theta(x_t, t)\|^2\right]
$$

where $v_t = x_1 - x_0$ is the "velocity" from noise to data.

**When to use**: Rectified flow models, ODE-based sampling

---

## Architecture Choices

### For Images: U-Net

**Standard architecture** for image diffusion models.

**Key components**:

- **Encoder-decoder structure**: Downsampling → bottleneck → upsampling
- **Skip connections**: Preserve spatial information
- **Attention blocks**: Capture long-range dependencies
- **Time conditioning**: Via AdaGN (Adaptive Group Normalization)

**Example structure**:
```
Input (3, 256, 256)
  ↓ Conv + ResBlock + Attention
(64, 128, 128)
  ↓ Downsample
(128, 64, 64)
  ↓ Downsample
(256, 32, 32)
  ↓ Bottleneck + Attention
(256, 32, 32)
  ↓ Upsample + Skip
(128, 64, 64)
  ↓ Upsample + Skip
(64, 128, 128)
  ↓ Conv
Output (3, 256, 256)
```

**When to use**: Images, spatial data

### For Tabular Data: MLP

**Simple architecture** for non-spatial data (gene expression, tabular features).

**Key components**:

- **Residual MLP blocks**: Prevent vanishing gradients
- **Layer normalization**: Stabilize training
- **Time embeddings**: Sinusoidal positional encodings
- **Conditional embeddings**: Concatenate or cross-attention

**Example from notebook**:
```python
class ConditionalScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=4):
        # Time embedding
        self.time_mlp = SinusoidalPositionEmbeddings(64)
        
        # Condition embedding
        self.condition_embed = nn.Embedding(n_conditions, 32)
        
        # MLP blocks with residual connections
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim) for _ in range(n_layers)
        ])
```

**When to use**: Gene expression, tabular data, point clouds

### For Sequences: Diffusion Transformers (DiT)

**Transformer-based architecture** for sequences and non-grid data.

**Key components**:

- **Token embeddings**: Convert data to tokens
- **Self-attention**: Capture dependencies
- **Time conditioning**: Via AdaLN (Adaptive Layer Normalization)
- **Positional encodings**: For sequential data

**When to use**: Sequences, non-grid structured data, biological sequences

---

## Time Conditioning

### Sinusoidal Embeddings (Standard)

$$
\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$

**Pros**:

- No learnable parameters
- Smooth interpolation
- Works well empirically

**Implementation**:
```python
def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

### Learned Embeddings

$$
t_{\text{emb}} = \text{Embedding}(t)
$$

**Pros**:

- Can learn task-specific representations
- More flexible

**Cons**:

- Requires more parameters
- May overfit with limited data

**When to use**: Large-scale models with lots of data

### Time Conditioning Mechanisms

**1. Concatenation** (Simple):
```python
h = torch.cat([x, time_emb], dim=-1)
```

**2. Additive** (U-Net style):
```python
h = x + time_emb
```

**3. Adaptive Normalization** (AdaGN, AdaLN):
```python
scale, shift = time_mlp(time_emb).chunk(2, dim=-1)
h = scale * normalize(x) + shift
```

**Best practice**: AdaGN/AdaLN for images, concatenation for tabular data

---

## Conditional Generation

### Types of Conditioning

1. **Class-conditional**: Generate specific categories (e.g., cell types)
2. **Text-conditional**: Generate from text descriptions
3. **Image-conditional**: Inpainting, super-resolution
4. **Continuous-conditional**: Drug dose, physical parameters

### Conditioning Strategies

#### 1. Concatenation (Simple)

```python
condition_emb = embedding(condition)
h = torch.cat([x, time_emb, condition_emb], dim=-1)
```

**Pros**: Simple, works well for discrete conditions
**Cons**: Limited flexibility

#### 2. Cross-Attention (Text-to-Image)

```python
# Query from noisy image
Q = linear_q(x)

# Key, Value from text embedding
K = linear_k(text_emb)
V = linear_v(text_emb)

# Attention
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

**Pros**: Flexible, captures complex relationships
**Cons**: More parameters, slower

**When to use**: Text-to-image, complex conditioning

#### 3. Adaptive Normalization (AdaGN)

```python
scale, shift = condition_mlp(condition).chunk(2, dim=-1)
h = scale * group_norm(x) + shift
```

**Pros**: Efficient, modulates features directly
**Cons**: Less flexible than cross-attention

**When to use**: Class-conditional, continuous conditioning

### Classifier-Free Guidance

**Key idea**: Train both conditional and unconditional models simultaneously.

**Training**:
```python
# Randomly drop condition with probability p (e.g., 0.1)
if random() < p:
    condition = None  # Unconditional
```

**Sampling**:
```python
# Interpolate between conditional and unconditional predictions
ε_pred = ε_uncond + w * (ε_cond - ε_uncond)
```

where $w$ is the guidance scale (typically 1-10).

**Effect**: Higher $w$ → stronger conditioning, less diversity

---

## Hyperparameters

### Noise Schedule

**Linear schedule** (original DDPM):
```python
betas = torch.linspace(1e-4, 0.02, T)
```

**Cosine schedule** (improved DDPM):
```python
def cosine_schedule(t, T, s=0.008):
    f_t = np.cos((t/T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar_t = f_t / f(0)
    return alpha_bar_t
```

**Best practice**: Cosine schedule for better sample quality

### Number of Timesteps

- **Training**: $T = 1000$ (standard)
- **Sampling**: Can use fewer steps with DDIM (e.g., 50-100)

**Trade-off**: More steps → better quality, slower sampling

### Learning Rate

- **Images**: $1 \times 10^{-4}$ to $2 \times 10^{-4}$
- **Tabular**: $1 \times 10^{-4}$ to $5 \times 10^{-4}$

**Best practice**: Use AdamW with weight decay $0.01$

### Batch Size

- **Images**: 128-256 (depends on GPU memory)
- **Tabular**: 128-512

**Best practice**: Larger batch size → more stable training

---

## Training Tips

### 1. EMA (Exponential Moving Average)

Maintain a moving average of model weights:

```python
ema_model = copy.deepcopy(model)

# After each training step
for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    ema_param.data.mul_(0.999).add_(param.data, alpha=0.001)
```

**Effect**: Smoother samples, better quality

### 2. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Effect**: Prevents exploding gradients, stabilizes training

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Effect**: Faster training, lower memory usage

### 4. Monitoring

**Key metrics to track**:

- Training loss (should decrease steadily)
- Sample quality (visual inspection or FID)
- Gradient norms (should be stable)

**Best practice**: Generate samples every N epochs to monitor quality

---

## Common Issues

### Issue 1: Model Predicts Constant Noise

**Symptom**: Generated samples are pure noise
**Cause**: Model hasn't learned the score function
**Solution**:

- Train longer
- Check learning rate (may be too high or too low)
- Verify data preprocessing

### Issue 2: Mode Collapse

**Symptom**: Model generates similar samples
**Cause**: Insufficient model capacity or training
**Solution**:

- Increase model size
- Train longer
- Use classifier-free guidance

### Issue 3: Slow Convergence

**Symptom**: Loss decreases very slowly
**Cause**: Poor hyperparameters or architecture
**Solution**:

- Increase learning rate
- Use cosine schedule instead of linear
- Add more layers or hidden dimensions

---

## Summary

**Key training decisions**:

1. **Loss**: Use simple MSE on noise prediction
2. **Architecture**: U-Net for images, MLP for tabular, DiT for sequences
3. **Time conditioning**: Sinusoidal embeddings with AdaGN/concatenation
4. **Conditioning**: Concatenation for simple, cross-attention for complex
5. **Hyperparameters**: Cosine schedule, $T=1000$, lr=$10^{-4}$
6. **Training tips**: Use EMA, gradient clipping, mixed precision

**Best practices**:

- Start with simple loss and standard architecture
- Use cosine schedule for better quality
- Monitor sample quality during training
- Use EMA for final model

---

## Related Documents

- [DDPM Foundations](01_ddpm_foundations.md) — Mathematical theory
- [DDPM Sampling](03_ddpm_sampling.md) — Sampling algorithms
- [DDPM Basics Notebook](../../notebooks/diffusion/01_ddpm/01_ddpm_basics.ipynb) — Implementation
- [SDE View](../SDE/01_diffusion_sde_view.md) — Continuous-time perspective

---

## References

1. **Ho, J., Jain, A., & Abbeel, P. (2020)**. Denoising Diffusion Probabilistic Models. *NeurIPS*.
2. **Nichol, A., & Dhariwal, P. (2021)**. Improved Denoising Diffusion Probabilistic Models. *ICML*.
3. **Dhariwal, P., & Nichol, A. (2021)**. Diffusion Models Beat GANs on Image Synthesis. *NeurIPS*.
4. **Ho, J., & Salimans, T. (2022)**. Classifier-Free Diffusion Guidance. *NeurIPS Workshop*.
5. **Peebles, W., & Xie, S. (2023)**. Scalable Diffusion Models with Transformers. *ICCV*.
