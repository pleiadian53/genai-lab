# Score Network Architecture: Activation Functions and Model Design

## Overview

This document explains the `SimpleScoreNetwork` architecture from the SDE formulation notebook, focusing on the SiLU activation function and how architecture choices differ for toy 2D data vs. real images.

---

## Referenced From

- **Notebook**: `notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb`
- **Module**: The notebook imports from `genailab.diffusion` module:
  ```python
  from genailab.diffusion import SimpleScoreNetwork as ScoreNet
  from genailab.diffusion import VPSDE as VPSDE_Module
  from genailab.diffusion import train_score_network, sample_reverse_sde
  ```

---

## What is SiLU (Swish)?

### Definition

**SiLU** = **S**igmoid **L**inear **U**nit, also known as **Swish**

$$
\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}} = \frac{x}{1 + e^{-x}}
$$

where $\sigma(x)$ is the sigmoid function.

**In PyTorch**: `nn.SiLU()` or `F.silu()`

### Visual Comparison with ReLU

```
ReLU(x):                    SiLU(x):
    |                           |
    |    /                      |      /
    |   /                       |     /
____|__/_______ x            ___|____/_______ x
    |                          /|
    |                        /  |
                           /    |
```

**ReLU**: Sharp corner at 0, completely kills negative values  
**SiLU**: Smooth everywhere, allows small negative values through

### Mathematical Properties

| Property | ReLU | SiLU |
|----------|------|------|
| **Formula** | $\max(0, x)$ | $x \cdot \sigma(x)$ |
| **Range** | $[0, \infty)$ | $\approx[-0.28, \infty)$ |
| **For $x > 0$** | $x$ | $\approx x$ (slightly less) |
| **For $x = 0$** | $0$ | $0$ |
| **For $x < 0$** | $0$ (dead) | Small negative value (alive) |
| **Smoothness** | Not smooth at 0 | Smooth everywhere |
| **Derivative** | $\begin{cases}0 & x < 0 \\ 1 & x > 0\end{cases}$ | $\sigma(x) + x \cdot \sigma(x)(1-\sigma(x))$ |

### Why SiLU is Better for Negative Values

#### The "Dying ReLU" Problem

**ReLU issue**: When $x < 0$, ReLU outputs exactly 0, and gradient is also 0:
- Neuron can get "stuck" with negative input
- No gradient flows back → can't recover
- Known as "dying ReLU" problem

**Example**:
```python
x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

relu_output = F.relu(x)
# tensor([0., 0., 0., 0., 0.5, 1., 2.])
# All negative values become 0 - information lost!

silu_output = F.silu(x)
# tensor([-0.2384, -0.2689, -0.1887, 0., 0.3113, 0.7311, 1.7616])
# Negative values still contribute (small but non-zero)
```

#### SiLU Advantages

1. **No dead neurons**: Even for negative inputs, SiLU outputs non-zero values and has non-zero gradients
2. **Smooth gradients**: Better for optimization (no sharp corners)
3. **Self-gating**: The $x \cdot \sigma(x)$ structure allows the network to learn adaptive gating
4. **Empirically better**: Often improves performance in deep networks

### SiLU Behavior at Different Values

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-5, 5, 11)
silu_x = F.silu(x)

for xi, si in zip(x, silu_x):
    print(f"x={xi:5.1f} → SiLU(x)={si:7.4f}")

# Output:
# x= -5.0 → SiLU(x)=-0.0337  (small negative)
# x= -4.0 → SiLU(x)=-0.0719  (slightly larger)
# x= -3.0 → SiLU(x)=-0.1423  (getting bigger)
# x= -2.0 → SiLU(x)=-0.2384  (peak negativity around here)
# x= -1.0 → SiLU(x)=-0.2689  (maximum negative value)
# x=  0.0 → SiLU(x)= 0.0000  (zero)
# x=  1.0 → SiLU(x)= 0.7311  (positive)
# x=  2.0 → SiLU(x)= 1.7616  (approaching x)
# x=  3.0 → SiLU(x)= 2.8577  (close to x)
# x=  4.0 → SiLU(x)= 3.9281  (almost x)
# x=  5.0 → SiLU(x)= 4.9665  (≈ x)
```

**Key observation**: 
- For large positive $x$: $\text{SiLU}(x) \approx x$ (acts like identity)
- For large negative $x$: $\text{SiLU}(x) \approx 0$ (but not exactly 0)
- Minimum value: $\approx -0.278$ (at $x \approx -1.278$)

### Derivative (Gradient)

$$
\frac{d}{dx}\text{SiLU}(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))
$$

**Key point**: Derivative is **non-zero** for all $x$, even negative values!

```python
x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
y = F.silu(x)
y.sum().backward()

print(x.grad)
# tensor([0.1050, 0.5000, 1.0762])
# Even x=-2.0 has non-zero gradient (0.1050)
```

**Why `y.sum().backward()` instead of `y.backward()`?**

In PyTorch, `backward()` can only be called on a **scalar** (single value), not on a tensor with multiple elements.

**The issue**:
```python
x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
y = F.silu(x)  # y has 3 elements: tensor([-0.2384, 0., 1.7616])

# This would fail:
# y.backward()  # RuntimeError: grad can be implicitly created only for scalar outputs
```

**Why?** Because `backward()` computes $\frac{\partial L}{\partial x}$ where $L$ is the loss (a scalar). If `y` is a vector, PyTorch doesn't know which scalar loss you want to differentiate with respect to.

**Solutions**:

1. **Sum to scalar** (most common for demos):
   ```python
   y.sum().backward()  # Computes gradient of sum(y) w.r.t. x
   ```
   This treats the "loss" as $L = \sum_i y_i$, so $\frac{\partial L}{\partial x_i} = \frac{\partial y_i}{\partial x_i}$.

2. **Mean to scalar**:
   ```python
   y.mean().backward()  # Computes gradient of mean(y) w.r.t. x
   ```

3. **Select one element**:
   ```python
   y[0].backward()  # Only compute gradient for first element
   ```

4. **Provide gradient weights explicitly**:
   ```python
   y.backward(torch.ones_like(y))  # Equivalent to y.sum().backward()
   ```
   The argument to `backward()` specifies $\frac{\partial L}{\partial y}$ (gradient of loss w.r.t. output).

**In neural network training**, you don't need `.sum()` because:
- Loss functions already return scalars
- `loss.backward()` works directly

```python
# Real training example
loss = F.mse_loss(pred, target)  # loss is already scalar
loss.backward()  # Works!
```

**For our demo**, we use `.sum()` just to show that gradients exist and are non-zero, even for negative inputs.

---

## The `SimpleScoreNetwork` Architecture

### Code Review

**Note**: The notebook now imports `SimpleScoreNetwork` from `genailab.diffusion` module:
```python
from genailab.diffusion import SimpleScoreNetwork as ScoreNet
```

The architecture shown below is the same as the module implementation, shown here for educational purposes:

```python
class SimpleScoreNetwork(nn.Module):
    """Simple MLP score network for 2D data."""
    
    def __init__(self, data_dim=2, hidden_dim=128, time_dim=32):
        super().__init__()
        
        # Time embedding (sinusoidal)
        self.time_dim = time_dim
        
        # Network
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
```

### Why This Architecture for 2D Data

**Design choices**:
1. **MLP (Multi-Layer Perceptron)**: Simple fully connected layers
2. **3 hidden layers**: Deep enough to model complex score functions
3. **SiLU activations**: Smooth, non-linear transformations
4. **Time embedding**: Sinusoidal encoding of time (like Transformer positional encoding)

**Why it works for 2D toy data**:
- 2D data has no spatial structure (just points in plane)
- MLPs are sufficient for learning score functions of 2D distributions
- No need for convolutions (no spatial locality to exploit)

---

## Architecture for Real Images

### Why MLPs Don't Work for Images

**Problem**: Images have:
1. **Spatial structure**: Nearby pixels are correlated
2. **Local patterns**: Edges, textures, shapes
3. **High dimensionality**: 256×256×3 = 196,608 dimensions
4. **Translation invariance**: A cat is a cat regardless of position

**MLP issues**:
- Treats every pixel independently
- Ignores spatial relationships
- Requires massive number of parameters
- Can't generalize across positions

### Solution 1: U-Net (Most Common)

**U-Net** is the standard architecture for image diffusion models (DDPM, Stable Diffusion, etc.)

#### Architecture

```
Input image (+ time embedding)
    ↓
[Encoder - Downsampling]
    Conv → Conv → ↓ (downsample)
    Conv → Conv → ↓
    Conv → Conv → ↓
           ↓
[Bottleneck - Latent Representation]
           ↓
[Decoder - Upsampling]
    Conv → Conv → ↑ (upsample) ← skip connection
    Conv → Conv → ↑            ← skip connection
    Conv → Conv → ↑            ← skip connection
    ↓
Output (predicted noise or score)
```

**Key features**:
1. **Convolutional layers**: Exploit spatial locality
2. **Downsampling**: Capture features at multiple scales
3. **Skip connections**: Preserve fine details from encoder
4. **Time conditioning**: Time embedding injected at each resolution

#### Example (Simplified)

```python
class UNetScoreNetwork(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, base_channels)
        )
        
        # Encoder (downsampling)
        self.enc1 = ConvBlock(in_channels, base_channels)       # 256×256
        self.enc2 = ConvBlock(base_channels, base_channels*2)   # 128×128
        self.enc3 = ConvBlock(base_channels*2, base_channels*4) # 64×64
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels*4, base_channels*8) # 32×32
        
        # Decoder (upsampling)
        self.dec3 = ConvBlock(base_channels*8, base_channels*4) # 64×64
        self.dec2 = ConvBlock(base_channels*4, base_channels*2) # 128×128
        self.dec1 = ConvBlock(base_channels*2, base_channels)   # 256×256
        
        # Output
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        # Encoder with skip connections
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(F.avg_pool2d(e1, 2), t_emb)
        e3 = self.enc3(F.avg_pool2d(e2, 2), t_emb)
        
        # Bottleneck
        b = self.bottleneck(F.avg_pool2d(e3, 2), t_emb)
        
        # Decoder with skip connections
        d3 = self.dec3(F.interpolate(b, scale_factor=2) + e3, t_emb)
        d2 = self.dec2(F.interpolate(d3, scale_factor=2) + e2, t_emb)
        d1 = self.dec1(F.interpolate(d2, scale_factor=2) + e1, t_emb)
        
        return self.out(d1)
```

**Why U-Net?**
- Preserves spatial structure
- Multi-scale feature extraction
- Efficient (reuses features via skip connections)
- Proven effective for image generation

#### Activation Functions in U-Net

Modern U-Nets often use:
- **SiLU** (or Swish): For most layers
- **GroupNorm**: Instead of BatchNorm (works better for small batches)
- **Attention**: At lower resolutions (e.g., 16×16) for global context

### Solution 2: Vision Transformers (DiT)

**DiT** = **Di**ffusion **T**ransformer (Peebles & Xie, 2023)

#### Architecture

```
Input image → Patch Embedding
    ↓
[Transformer Blocks × N]
    Self-Attention (with time conditioning)
    Feed-Forward (with time conditioning)
    ↓
Unpatch → Output image
```

**Key features**:
1. **Patch-based**: Treats image as sequence of patches
2. **Self-attention**: Captures global relationships
3. **Adaptive LayerNorm**: Conditions on time via AdaLN
4. **Scalability**: Larger models → better quality

#### Example (Simplified)

```python
class DiTScoreNetwork(nn.Module):
    def __init__(self, img_size=256, patch_size=16, dim=512, depth=12):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim*2)  # For AdaLN (scale & shift)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads=8) for _ in range(depth)
        ])
        
        # Unpatch
        self.unpatch = nn.Linear(dim, 3 * patch_size ** 2)
    
    def forward(self, x, t):
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H, W) → (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # → (B, num_patches, dim)
        x = x + self.pos_embed
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, dim*2)
        
        # Transformer blocks with time conditioning
        for block in self.blocks:
            x = block(x, t_emb)
        
        # Unpatch
        x = self.unpatch(x)  # (B, num_patches, 3*P*P)
        # Reshape to image...
        return x
```

**Why DiT?**
- **Scalability**: Can scale to billions of parameters
- **Global context**: Self-attention captures long-range dependencies
- **State-of-the-art**: Achieves best FID scores on ImageNet
- **Flexible**: Easy to condition on additional inputs

**Trade-offs**:
- **Compute**: More expensive than U-Net
- **Data**: Requires more data to train effectively
- **Memory**: Attention is $O(n^2)$ in sequence length

---

## Architecture Comparison

| Aspect | MLP (2D data) | U-Net (Images) | DiT (Transformers) |
|--------|---------------|----------------|-------------------|
| **Data type** | Low-dim (e.g., 2D points) | Images, medical scans | Images (large-scale) |
| **Spatial structure** | None | Local (convolutions) | Global (attention) |
| **Parameters** | ~100K | ~100M | ~100M-1B |
| **Compute** | Low | Medium | High |
| **Training data** | Small (10K samples) | Medium (100K-1M) | Large (1M-100M) |
| **Use case** | Toy problems, demos | Practical image generation | State-of-the-art generation |
| **Activation** | SiLU | SiLU + GroupNorm | SiLU (in FFN) |

---

## Medical Imaging: Special Considerations

For **medical images** (X-rays, MRIs, CT scans), you would typically use:

### U-Net (Most Common)

**Why U-Net is popular for medical imaging**:
1. **Proven effective**: Originally designed for medical image segmentation
2. **Works with small datasets**: Medical data is often limited
3. **Preserves details**: Skip connections maintain fine structures
4. **3D extensions**: Easy to adapt for 3D volumes (MRI, CT)

**Modifications for medical imaging**:
```python
class Medical3DUNet(nn.Module):
    """3D U-Net for medical volumes."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Use 3D convolutions instead of 2D
        self.enc1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        # ... encoder blocks ...
        
        # Same U-Net structure but in 3D
        # ...
```

**Special considerations**:
- **Normalization**: GroupNorm or InstanceNorm (works better than BatchNorm for small medical datasets)
- **Activation**: SiLU or GELU
- **Conditioning**: Can condition on patient metadata (age, sex, etc.)
- **Domain-specific**: May need to preserve certain properties (Hounsfield units for CT)

### Hybrid Architectures

Modern approaches combine best of both:
- **U-ViT**: U-Net with Vision Transformer in bottleneck
- **Swin Transformer**: Hierarchical transformer (like U-Net structure)
- **CoAtNet**: Convolutions early, attention later

---

## Summary

### Activation Functions

**SiLU advantages**:
- Smooth (no sharp corners) → better gradients
- Non-zero for negative values → no dying neurons
- Self-gating mechanism → adaptive feature selection
- Empirically better than ReLU in deep networks

**When to use**:
- ✅ Deep networks (diffusion models, transformers)
- ✅ When gradient flow is critical
- ✅ Modern architectures (post-2017)

### Architecture Choices

| Task | Architecture | Activation | Why |
|------|--------------|------------|-----|
| **2D toy data** | MLP | SiLU | No spatial structure, simple |
| **Images (general)** | U-Net | SiLU + GroupNorm | Spatial structure, proven effective |
| **Images (SOTA)** | DiT / U-ViT | SiLU (+ GELU in attention) | Scale, global context |
| **Medical 2D** | U-Net | SiLU + InstanceNorm | Small datasets, detail preservation |
| **Medical 3D** | 3D U-Net | SiLU + GroupNorm | Volumetric data, isotropic structure |

---

## Code Examples

### Simple SiLU vs ReLU Test

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Compare activations
x = torch.linspace(-5, 5, 200)
relu = F.relu(x)
silu = F.silu(x)

plt.figure(figsize=(10, 4))
plt.plot(x, relu, label='ReLU', linewidth=2)
plt.plot(x, silu, label='SiLU', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel('x')
plt.ylabel('Activation(x)')
plt.title('ReLU vs SiLU')
plt.show()
```

### Minimal U-Net for Images

See the simplified example above, or check out:
- **DDPM paper**: [github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)
- **Hugging Face**: [github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)

---

## References

### Activation Functions
- **SiLU/Swish**: Ramachandran et al. (2017), "Searching for Activation Functions"
- **GELU**: Hendrycks & Gimpel (2016), "Gaussian Error Linear Units"

### Architectures
- **U-Net**: Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **DDPM**: Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
- **DiT**: Peebles & Xie (2023), "Scalable Diffusion Models with Transformers"
- **Stable Diffusion**: Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models"

### Related Documents
- **Advanced Architectures**: [advanced_architectures.md](./advanced_architectures.md)
- **Time Embedding and FiLM**: [time_embedding_and_film.md](./time_embedding_and_film.md)
- **Training Loss and Denoising**: [../02_sde_formulation/supplements/04_training_loss_and_denoising.md](../02_sde_formulation/supplements/04_training_loss_and_denoising.md)
- **Notebook**: `notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb`

