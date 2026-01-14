# DiT Sampling: Generating with Rectified Flow

This document explains how to generate samples from trained DiT models using rectified flow, covering ODE solvers, conditional generation, and practical strategies.

**Prerequisites**: Understanding of [DiT architecture](01_dit_foundations.md), [training](02_dit_training.md), and [flow matching sampling](../flow_matching/03_flow_matching_sampling.md).

---

## Overview

Sampling from DiT + rectified flow is **deterministic ODE integration**:

**Forward ODE** (noise → data):
$$
\frac{dx}{dt} = v_\theta(x, t)
$$

**Key properties**:
- Deterministic (same noise → same output)
- Fast (20-50 steps typical)
- Straight paths (rectified flow)
- No stochasticity (unlike DDPM)

**Basic sampling**:
```python
# Start from noise
x = torch.randn(batch_size, 3, 256, 256)

# Integrate ODE
dt = 1.0 / num_steps
for k in range(num_steps):
    t = k * dt
    v = model(x, t)
    x = x + v * dt

# Result: generated image
```

---

## 1. ODE Solvers

### 1.1 Euler Method (Simplest)

**First-order method**:

$$
x_{k+1} = x_k + v_\theta(x_k, t_k) \cdot \Delta t
$$

**Implementation**:

```python
@torch.no_grad()
def sample_euler(model, shape, num_steps=50, device='cuda'):
    """
    Sample using Euler method.
    
    Args:
        model: Trained DiT model
        shape: Output shape (B, C, H, W)
        num_steps: Number of integration steps
        device: Device to run on
    
    Returns:
        samples: Generated images
    """
    # Start from noise
    x = torch.randn(shape, device=device)
    
    # Time step
    dt = 1.0 / num_steps
    
    # Integrate
    for k in range(num_steps):
        t = torch.full((shape[0],), k * dt, device=device)
        
        # Predict velocity
        v = model(x, t)
        
        # Euler step
        x = x + v * dt
    
    # Denormalize from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    return x
```

**Pros**: Simple, fast
**Cons**: Lower accuracy, needs more steps

### 1.2 Heun's Method (2nd Order)

**Predictor-corrector approach**:

$$
\begin{align}
\tilde{x}_{k+1} &= x_k + v_\theta(x_k, t_k) \cdot \Delta t \quad \text{(predictor)} \\
x_{k+1} &= x_k + \frac{1}{2}[v_\theta(x_k, t_k) + v_\theta(\tilde{x}_{k+1}, t_{k+1})] \cdot \Delta t \quad \text{(corrector)}
\end{align}
$$

**Implementation**:

```python
@torch.no_grad()
def sample_heun(model, shape, num_steps=25, device='cuda'):
    """Sample using Heun's method (2nd order)."""
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    
    for k in range(num_steps):
        t_k = torch.full((shape[0],), k * dt, device=device)
        t_k1 = torch.full((shape[0],), (k + 1) * dt, device=device)
        
        # Predictor
        v_k = model(x, t_k)
        x_pred = x + v_k * dt
        
        # Corrector
        v_k1 = model(x_pred, t_k1)
        x = x + 0.5 * (v_k + v_k1) * dt
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x
```

**Pros**: Better accuracy, fewer steps needed
**Cons**: 2× model evaluations per step

### 1.3 Runge-Kutta 4th Order (RK4)

**Fourth-order method** (most accurate):

$$
\begin{align}
k_1 &= v_\theta(x_k, t_k) \\
k_2 &= v_\theta(x_k + \frac{\Delta t}{2} k_1, t_k + \frac{\Delta t}{2}) \\
k_3 &= v_\theta(x_k + \frac{\Delta t}{2} k_2, t_k + \frac{\Delta t}{2}) \\
k_4 &= v_\theta(x_k + \Delta t \, k_3, t_k + \Delta t) \\
x_{k+1} &= x_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{align}
$$

**Implementation**:

```python
@torch.no_grad()
def sample_rk4(model, shape, num_steps=20, device='cuda'):
    """Sample using RK4 (4th order)."""
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    
    for k in range(num_steps):
        t = k * dt
        batch_size = shape[0]
        
        # k1
        t_tensor = torch.full((batch_size,), t, device=device)
        k1 = model(x, t_tensor)
        
        # k2
        t_tensor = torch.full((batch_size,), t + 0.5 * dt, device=device)
        k2 = model(x + 0.5 * dt * k1, t_tensor)
        
        # k3
        k3 = model(x + 0.5 * dt * k2, t_tensor)
        
        # k4
        t_tensor = torch.full((batch_size,), t + dt, device=device)
        k4 = model(x + dt * k3, t_tensor)
        
        # Update
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x
```

**Pros**: Highest accuracy, fewest steps
**Cons**: 4× model evaluations per step

### 1.4 Solver Comparison

| Solver | Order | Steps for Quality | Model Evals | Speed | Accuracy |
|--------|-------|-------------------|-------------|-------|----------|
| Euler | 1st | 50-100 | 50-100 | Fastest | Lowest |
| Heun | 2nd | 25-50 | 50-100 | Moderate | Good |
| RK4 | 4th | 10-20 | 40-80 | Slowest | Best |

**Recommendation**: 
- Fast generation: Euler with 50 steps
- Balanced: Heun with 25 steps
- Best quality: RK4 with 20 steps

---

## 2. Conditional Generation

### 2.1 Class-Conditional Sampling

**For class-conditional models** (e.g., ImageNet):

```python
@torch.no_grad()
def sample_conditional(model, class_labels, num_steps=50, device='cuda'):
    """
    Generate samples conditioned on class labels.
    
    Args:
        model: Trained conditional DiT
        class_labels: Class indices (B,)
        num_steps: Number of steps
        device: Device
    
    Returns:
        samples: Generated images
    """
    batch_size = len(class_labels)
    shape = (batch_size, 3, 256, 256)
    
    # Start from noise
    x = torch.randn(shape, device=device)
    class_labels = class_labels.to(device)
    
    # Integrate
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((batch_size,), k * dt, device=device)
        
        # Predict with class conditioning
        v = model(x, t, y=class_labels)
        
        # Update
        x = x + v * dt
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x

# Usage
class_labels = torch.tensor([207, 360, 387, 974])  # ImageNet classes
samples = sample_conditional(model, class_labels, num_steps=50)
```

### 2.2 Classifier-Free Guidance

**Interpolate between conditional and unconditional**:

$$
\tilde{v}_\theta(x, t, c) = (1 + w) v_\theta(x, t, c) - w v_\theta(x, t, \emptyset)
$$

where $w$ is the guidance weight.

**Implementation**:

```python
@torch.no_grad()
def sample_cfg(model, class_labels, guidance_weight=2.0, num_steps=50, device='cuda'):
    """
    Sample with classifier-free guidance.
    
    Args:
        model: Trained model with CFG
        class_labels: Class indices (B,)
        guidance_weight: Guidance strength (typically 1.5-4.0)
        num_steps: Number of steps
        device: Device
    
    Returns:
        samples: Generated images
    """
    batch_size = len(class_labels)
    shape = (batch_size, 3, 256, 256)
    
    x = torch.randn(shape, device=device)
    class_labels = class_labels.to(device)
    
    # Null class for unconditional
    null_class = torch.full((batch_size,), model.num_classes, device=device)
    
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((batch_size,), k * dt, device=device)
        
        # Conditional prediction
        v_cond = model(x, t, y=class_labels)
        
        # Unconditional prediction
        v_uncond = model(x, t, y=null_class)
        
        # Guided prediction
        v = (1 + guidance_weight) * v_cond - guidance_weight * v_uncond
        
        # Update
        x = x + v * dt
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x

# Usage
samples = sample_cfg(model, class_labels, guidance_weight=2.0, num_steps=50)
```

**Guidance weight effects**:
- $w = 0$: Pure conditional (no guidance)
- $w = 1$: Moderate guidance
- $w = 2-4$: Strong guidance (better quality, less diversity)
- $w > 5$: Too strong (artifacts)

### 2.3 Text-Conditional Sampling

**For text-to-image models**:

```python
@torch.no_grad()
def sample_text_conditional(model, text_prompts, tokenizer, guidance_weight=7.5, num_steps=50):
    """
    Generate images from text prompts.
    
    Args:
        model: Text-conditional DiT
        text_prompts: List of text strings
        tokenizer: Text tokenizer
        guidance_weight: CFG weight
        num_steps: Number of steps
    
    Returns:
        samples: Generated images
    """
    batch_size = len(text_prompts)
    shape = (batch_size, 3, 256, 256)
    device = next(model.parameters()).device
    
    # Tokenize text
    text_tokens = tokenizer(text_prompts, max_length=77, padding='max_length')
    text_tokens = text_tokens.to(device)
    
    # Empty prompt for unconditional
    empty_tokens = tokenizer([""] * batch_size, max_length=77, padding='max_length')
    empty_tokens = empty_tokens.to(device)
    
    x = torch.randn(shape, device=device)
    
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((batch_size,), k * dt, device=device)
        
        # Conditional
        v_cond = model(x, t, context=text_tokens)
        
        # Unconditional
        v_uncond = model(x, t, context=empty_tokens)
        
        # Guided
        v = (1 + guidance_weight) * v_cond - guidance_weight * v_uncond
        
        x = x + v * dt
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x
```

---

## 3. Sampling Strategies

### 3.1 Number of Steps

**Quality vs speed trade-off**:

| Steps | Quality | Time (relative) | Use Case |
|-------|---------|-----------------|----------|
| 10-15 | Low | 1× | Quick preview |
| 20-25 | Good | 2× | Fast generation |
| 50 | High | 5× | Standard |
| 100 | Very high | 10× | Best quality |

**Recommendation**: 50 steps for most applications.

### 3.2 Non-Uniform Time Discretization

**Uniform spacing** (standard):
```python
times = torch.linspace(0, 1, num_steps)
```

**Quadratic spacing** (more steps at high noise):
```python
times = torch.linspace(0, 1, num_steps) ** 2
```

**Cosine spacing**:
```python
s = 0.008
times = torch.cos((torch.linspace(0, 1, num_steps) + s) / (1 + s) * math.pi / 2) ** 2
```

**Implementation**:

```python
@torch.no_grad()
def sample_nonuniform(model, shape, num_steps=50, spacing='cosine', device='cuda'):
    """Sample with non-uniform time steps."""
    x = torch.randn(shape, device=device)
    
    # Generate time steps
    if spacing == 'uniform':
        times = torch.linspace(0, 1, num_steps + 1)
    elif spacing == 'quadratic':
        times = torch.linspace(0, 1, num_steps + 1) ** 2
    elif spacing == 'cosine':
        s = 0.008
        times = torch.cos((torch.linspace(0, 1, num_steps + 1) + s) / (1 + s) * math.pi / 2) ** 2
    
    # Integrate
    for k in range(num_steps):
        t = times[k]
        dt = times[k + 1] - times[k]
        
        t_tensor = torch.full((shape[0],), t, device=device)
        v = model(x, t_tensor)
        x = x + v * dt
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x
```

**Empirical finding**: Cosine spacing slightly better for DiT.

### 3.3 Adaptive Step Sizes

**Idea**: Use larger steps when error is small, smaller when large.

```python
@torch.no_grad()
def sample_adaptive(model, shape, target_error=1e-3, max_steps=100, device='cuda'):
    """
    Sample with adaptive step sizes.
    
    Uses local error estimation to adjust step size.
    """
    x = torch.randn(shape, device=device)
    t = 0.0
    step_count = 0
    dt = 0.1  # Initial step size
    
    while t < 1.0 and step_count < max_steps:
        # Full step
        t_tensor = torch.full((shape[0],), t, device=device)
        v = model(x, t_tensor)
        x_full = x + v * dt
        
        # Two half steps
        v1 = model(x, t_tensor)
        x_half = x + v1 * (dt / 2)
        
        t_mid_tensor = torch.full((shape[0],), t + dt/2, device=device)
        v2 = model(x_half, t_mid_tensor)
        x_double = x_half + v2 * (dt / 2)
        
        # Estimate error
        error = torch.abs(x_full - x_double).mean()
        
        # Adjust step size
        if error < target_error:
            # Accept step
            x = x_double
            t += dt
            step_count += 1
            # Increase step size
            dt = min(dt * 1.5, 1.0 - t)
        else:
            # Reject step, decrease step size
            dt = dt * 0.5
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x
```

**Note**: Adaptive methods can be slower due to error estimation overhead.

---

## 4. Practical Considerations

### 4.1 Batch Generation

**Generate multiple samples efficiently**:

```python
@torch.no_grad()
def generate_batch(model, num_samples, batch_size=16, num_steps=50, device='cuda'):
    """
    Generate many samples in batches.
    
    Args:
        model: Trained model
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        num_steps: ODE steps
        device: Device
    
    Returns:
        all_samples: All generated samples
    """
    all_samples = []
    
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        shape = (current_batch_size, 3, 256, 256)
        
        samples = sample_euler(model, shape, num_steps, device)
        all_samples.append(samples.cpu())
    
    return torch.cat(all_samples, dim=0)

# Usage
samples = generate_batch(model, num_samples=1000, batch_size=16)
```

### 4.2 Memory Optimization

**For large models or high resolution**:

```python
@torch.no_grad()
def sample_memory_efficient(model, shape, num_steps=50, device='cuda'):
    """Sample with reduced memory usage."""
    # Use torch.cuda.amp for mixed precision
    from torch.cuda.amp import autocast
    
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    
    for k in range(num_steps):
        t = torch.full((shape[0],), k * dt, device=device)
        
        # Use autocast for forward pass
        with autocast():
            v = model(x, t)
        
        x = x + v * dt
        
        # Clear cache periodically
        if k % 10 == 0:
            torch.cuda.empty_cache()
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x
```

### 4.3 Deterministic Sampling

**For reproducibility**:

```python
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Usage
set_seed(42)
samples = sample_euler(model, shape, num_steps=50)
# Same seed → same samples
```

### 4.4 Progressive Generation

**Generate at low resolution, then upsample**:

```python
@torch.no_grad()
def sample_progressive(model, target_size=256, num_steps=50, device='cuda'):
    """
    Generate progressively from low to high resolution.
    
    Faster than direct high-res generation.
    """
    # Stage 1: Generate at 64×64
    shape_64 = (1, 3, 64, 64)
    x_64 = sample_euler(model, shape_64, num_steps=num_steps//2, device=device)
    
    # Stage 2: Upsample to 128×128 and refine
    x_128 = F.interpolate(x_64, size=(128, 128), mode='bilinear')
    # Optional: Run a few more steps at 128×128
    
    # Stage 3: Upsample to 256×256 and refine
    x_256 = F.interpolate(x_128, size=(target_size, target_size), mode='bilinear')
    # Optional: Run final steps at full resolution
    
    return x_256
```

---

## 5. Advanced Techniques

### 5.1 Interpolation in Latent Space

**Smooth transitions between samples**:

```python
@torch.no_grad()
def interpolate_samples(model, num_frames=10, num_steps=50, device='cuda'):
    """
    Generate interpolation between two random samples.
    
    Args:
        model: Trained model
        num_frames: Number of interpolation frames
        num_steps: ODE steps
        device: Device
    
    Returns:
        frames: Interpolated samples
    """
    shape = (1, 3, 256, 256)
    
    # Two random starting points
    z1 = torch.randn(shape, device=device)
    z2 = torch.randn(shape, device=device)
    
    frames = []
    alphas = torch.linspace(0, 1, num_frames)
    
    for alpha in alphas:
        # Spherical interpolation (slerp)
        z = slerp(z1, z2, alpha)
        
        # Generate from interpolated noise
        x = z.clone()
        dt = 1.0 / num_steps
        
        for k in range(num_steps):
            t = torch.full((1,), k * dt, device=device)
            v = model(x, t)
            x = x + v * dt
        
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        frames.append(x)
    
    return torch.cat(frames, dim=0)

def slerp(z1, z2, alpha):
    """Spherical linear interpolation."""
    z1_norm = z1 / z1.norm(dim=1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=1, keepdim=True)
    
    omega = torch.acos((z1_norm * z2_norm).sum(dim=1, keepdim=True))
    so = torch.sin(omega)
    
    return (torch.sin((1.0 - alpha) * omega) / so) * z1 + (torch.sin(alpha * omega) / so) * z2
```

### 5.2 Inpainting

**Fill in masked regions**:

```python
@torch.no_grad()
def sample_inpainting(model, image, mask, num_steps=50, device='cuda'):
    """
    Inpaint masked regions of an image.
    
    Args:
        model: Trained model
        image: Input image with known regions (B, C, H, W)
        mask: Binary mask (1 = known, 0 = unknown) (B, 1, H, W)
        num_steps: ODE steps
        device: Device
    
    Returns:
        inpainted: Image with filled regions
    """
    image = image.to(device)
    mask = mask.to(device)
    
    # Start from noise
    x = torch.randn_like(image)
    
    # Replace known regions
    x = mask * image + (1 - mask) * x
    
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((image.shape[0],), k * dt, device=device)
        
        # Predict velocity
        v = model(x, t)
        
        # Update
        x = x + v * dt
        
        # Project onto constraint (keep known regions fixed)
        x = mask * image + (1 - mask) * x
    
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    return x
```

### 5.3 Image Editing

**Edit images by manipulating latent codes**:

```python
@torch.no_grad()
def edit_image(model, image, direction, strength=1.0, num_steps=50, device='cuda'):
    """
    Edit image in a semantic direction.
    
    Args:
        model: Trained model
        image: Input image (B, C, H, W)
        direction: Edit direction in latent space
        strength: Edit strength
        num_steps: ODE steps
        device: Device
    
    Returns:
        edited: Edited image
    """
    image = image.to(device)
    
    # Encode to noise (reverse ODE)
    x = image.clone()
    dt = -1.0 / num_steps  # Negative for reverse
    
    for k in range(num_steps):
        t = torch.full((image.shape[0],), 1.0 - k * abs(dt), device=device)
        v = model(x, t)
        x = x + v * dt
    
    # Apply edit in latent space
    x_edited = x + strength * direction
    
    # Decode back to image (forward ODE)
    dt = 1.0 / num_steps
    for k in range(num_steps):
        t = torch.full((image.shape[0],), k * dt, device=device)
        v = model(x_edited, t)
        x_edited = x_edited + v * dt
    
    x_edited = (x_edited + 1) / 2
    x_edited = torch.clamp(x_edited, 0, 1)
    return x_edited
```

---

## 6. Quality vs Speed Trade-offs

### 6.1 Speed Optimizations

**Techniques to speed up sampling**:

1. **Fewer steps**: 20-25 instead of 50
2. **Better solver**: RK4 instead of Euler
3. **Compiled model**: `torch.compile(model)`
4. **Mixed precision**: Use FP16
5. **Batch generation**: Generate multiple samples at once

```python
# Fast sampling configuration
model_compiled = torch.compile(model)

@torch.no_grad()
def sample_fast(model, shape, device='cuda'):
    """Fast sampling with all optimizations."""
    from torch.cuda.amp import autocast
    
    with autocast():
        samples = sample_rk4(model, shape, num_steps=20, device=device)
    
    return samples
```

### 6.2 Quality Optimizations

**Techniques to improve quality**:

1. **More steps**: 100 instead of 50
2. **Better solver**: RK4 with adaptive steps
3. **Classifier-free guidance**: Increase guidance weight
4. **EMA weights**: Use EMA model for sampling
5. **Non-uniform spacing**: Cosine schedule

```python
# High-quality sampling configuration
@torch.no_grad()
def sample_high_quality(model_ema, shape, class_labels, device='cuda'):
    """High-quality sampling with all optimizations."""
    samples = sample_cfg(
        model_ema,
        class_labels,
        guidance_weight=3.0,
        num_steps=100,
        device=device
    )
    return samples
```

### 6.3 Comparison Table

| Configuration | Steps | Solver | Guidance | Time | Quality |
|---------------|-------|--------|----------|------|---------|
| **Fast** | 20 | Euler | None | 1× | Good |
| **Balanced** | 50 | Heun | 2.0 | 3× | High |
| **Best** | 100 | RK4 | 3.0 | 10× | Excellent |

---

## 7. Evaluation Metrics

### 7.1 FID (Fréchet Inception Distance)

**Measure distribution similarity**:

```python
from pytorch_fid import fid_score

# Generate samples
samples = generate_batch(model, num_samples=50000)

# Compute FID
fid = fid_score.calculate_fid_given_paths(
    [real_images_path, generated_images_path],
    batch_size=50,
    device='cuda',
    dims=2048
)

print(f"FID: {fid:.2f}")
```

**Lower is better**: FID < 10 is excellent for ImageNet.

### 7.2 Inception Score (IS)

**Measure quality and diversity**:

```python
from pytorch_fid import inception

def compute_inception_score(samples, splits=10):
    """Compute Inception Score."""
    # Get Inception predictions
    preds = inception.get_predictions(samples)
    
    # Compute IS
    is_mean, is_std = inception.calculate_inception_score(preds, splits=splits)
    
    return is_mean, is_std
```

**Higher is better**: IS > 100 is good for ImageNet.

### 7.3 Precision and Recall

**Measure quality vs diversity trade-off**:

```python
def compute_precision_recall(real_features, fake_features, k=3):
    """
    Compute precision and recall.
    
    Precision: Quality (fake samples look real)
    Recall: Diversity (cover real distribution)
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Fit on real features
    nn_real = NearestNeighbors(n_neighbors=k).fit(real_features)
    nn_fake = NearestNeighbors(n_neighbors=k).fit(fake_features)
    
    # Precision: fraction of fake samples close to real
    distances_fake_to_real, _ = nn_real.kneighbors(fake_features)
    precision = (distances_fake_to_real[:, 0] < threshold).mean()
    
    # Recall: fraction of real samples close to fake
    distances_real_to_fake, _ = nn_fake.kneighbors(real_features)
    recall = (distances_real_to_fake[:, 0] < threshold).mean()
    
    return precision, recall
```

---

## Key Takeaways

### Sampling Process

1. **ODE integration**: Deterministic, fast, straight paths
2. **Solvers**: Euler (simple), Heun (balanced), RK4 (best)
3. **Steps**: 20-50 typical, 100 for best quality
4. **Conditioning**: Class, text, or other modalities

### Practical Tips

1. **Use EMA weights** for sampling
2. **Classifier-free guidance** improves quality
3. **RK4 with 20 steps** ≈ Euler with 50 steps
4. **Batch generation** for efficiency
5. **Set seed** for reproducibility

### Quality vs Speed

1. **Fast**: Euler, 20 steps, no guidance
2. **Balanced**: Heun, 50 steps, CFG 2.0
3. **Best**: RK4, 100 steps, CFG 3.0

---

## Related Documents

- [01_dit_foundations.md](01_dit_foundations.md) — Architecture details
- [02_dit_training.md](02_dit_training.md) — Training pipeline
- [Flow Matching Sampling](../flow_matching/03_flow_matching_sampling.md) — ODE theory

---

## References

- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers"
- Liu et al. (2022): "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- Ho & Salimans (2022): "Classifier-Free Diffusion Guidance"
