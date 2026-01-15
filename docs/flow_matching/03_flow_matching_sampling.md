# Flow Matching Sampling

This document covers sampling from flow matching models: ODE solvers, sampling strategies, quality-speed tradeoffs, and practical considerations.

---

## Sampling Overview

### The Sampling Problem

After training a flow matching model $v_\theta(x, t)$, we generate samples by solving an **ordinary differential equation (ODE)** backward in time.

**Setup**:

- **Start**: $x(1) \sim p_{\text{noise}}$ (e.g., $\mathcal{N}(0, I)$)
- **Goal**: $x(0) \sim p_{\text{data}}$
- **Method**: Integrate the ODE from $t=1$ to $t=0$

**Flow ODE**:

$$
\frac{dx}{dt} = v_\theta(x(t), t)
$$

**Key property**: This is a **deterministic** process—same initial noise always produces the same output.

---

## ODE Solvers

### Euler Method

The simplest ODE solver uses first-order approximation.

**Discrete update**:

$$
x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t)
$$

**Algorithm**:
```python
def euler_sample(model, x_init, num_steps=50):
    """
    Sample using Euler method.
    
    Args:
        model: Trained flow matching model v_theta(x, t)
        x_init: Initial noise [batch_size, ...]
        num_steps: Number of discretization steps
    
    Returns:
        x_final: Generated samples [batch_size, ...]
    """
    x = x_init
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = 1.0 - i * dt  # Time going backward from 1 to 0
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        
        # Compute velocity
        with torch.no_grad():
            v = model(x, t_tensor)
        
        # Euler step
        x = x - dt * v
    
    return x
```

**Properties**:

- **Simple**: Easy to implement
- **Fast**: One function evaluation per step
- **Accuracy**: $O(\Delta t)$ local error, $O(\Delta t)$ global error
- **Typical steps**: 50-100 for good quality

### Runge-Kutta 4 (RK4)

Fourth-order Runge-Kutta provides better accuracy with fewer steps.

**Update formula**:

$$
\begin{align}
k_1 &= v_\theta(x_t, t) \\
k_2 &= v_\theta(x_t - \frac{\Delta t}{2} k_1, t - \frac{\Delta t}{2}) \\
k_3 &= v_\theta(x_t - \frac{\Delta t}{2} k_2, t - \frac{\Delta t}{2}) \\
k_4 &= v_\theta(x_t - \Delta t \cdot k_3, t - \Delta t) \\
x_{t-\Delta t} &= x_t - \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{align}
$$

**Algorithm**:
```python
def rk4_sample(model, x_init, num_steps=20):
    """
    Sample using RK4 method.
    
    Args:
        model: Trained flow matching model
        x_init: Initial noise
        num_steps: Number of steps
    
    Returns:
        x_final: Generated samples
    """
    x = x_init
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = 1.0 - i * dt
        
        # k1
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        with torch.no_grad():
            k1 = model(x, t_tensor)
        
        # k2
        t_half = t - dt / 2
        t_tensor = torch.full((x.shape[0],), t_half, device=x.device)
        with torch.no_grad():
            k2 = model(x - dt / 2 * k1, t_tensor)
        
        # k3
        with torch.no_grad():
            k3 = model(x - dt / 2 * k2, t_tensor)
        
        # k4
        t_next = t - dt
        t_tensor = torch.full((x.shape[0],), t_next, device=x.device)
        with torch.no_grad():
            k4 = model(x - dt * k3, t_tensor)
        
        # Weighted average
        x = x - dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return x
```

**Properties**:

- **Accurate**: $O(\Delta t^4)$ local error, $O(\Delta t^4)$ global error
- **Efficient**: 4× function evaluations per step, but needs 3-5× fewer steps
- **Typical steps**: 10-20 for good quality
- **Trade-off**: More computation per step, but fewer total steps

### Adaptive Solvers

Adaptive solvers automatically adjust step size based on local error estimates.

**Dormand-Prince (dopri5)**:
```python
from torchdiffeq import odeint

def adaptive_sample(model, x_init, rtol=1e-5, atol=1e-5):
    """
    Sample using adaptive ODE solver.
    
    Args:
        model: Trained flow matching model
        x_init: Initial noise
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        x_final: Generated samples
    """
    # Define ODE function
    def ode_func(t, x):
        t_tensor = torch.full((x.shape[0],), t.item(), device=x.device)
        return model(x, t_tensor)
    
    # Time points (backward from 1 to 0)
    t_span = torch.tensor([1.0, 0.0], device=x_init.device)
    
    # Solve ODE
    trajectory = odeint(
        ode_func,
        x_init,
        t_span,
        rtol=rtol,
        atol=atol,
        method='dopri5'
    )
    
    return trajectory[-1]  # Return final point
```

**Properties**:

- **Automatic**: No need to choose number of steps
- **Efficient**: Uses more steps where needed, fewer where possible
- **Accurate**: Error control via tolerances
- **Typical NFE**: 15-30 (number of function evaluations)

**When to use**:

- When you want guaranteed accuracy
- When sampling budget is flexible
- For complex, non-smooth velocity fields

---

## Sampling Strategies

### Standard Sampling

**Basic procedure**:
```python
def sample_flow_matching(model, batch_size=64, num_steps=20, device='cuda'):
    """
    Standard sampling from flow matching model.
    """
    # Sample initial noise
    x_init = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # Integrate ODE
    samples = rk4_sample(model, x_init, num_steps=num_steps)
    
    # Denormalize if needed
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    
    return samples
```

### Conditional Sampling

For conditional generation:

```python
def conditional_sample(model, condition, batch_size=64, num_steps=20):
    """
    Conditional sampling with class or text conditioning.
    
    Args:
        model: Conditional flow matching model v_theta(x, t, c)
        condition: Conditioning variable (class label, text embedding, etc.)
        batch_size: Number of samples
        num_steps: ODE steps
    
    Returns:
        samples: Generated samples conditioned on c
    """
    # Initial noise
    x = torch.randn(batch_size, *data_shape, device=device)
    dt = 1.0 / num_steps
    
    # Repeat condition for batch
    if condition.ndim == 1:
        condition = condition.repeat(batch_size, 1)
    
    # Integrate ODE with conditioning
    for i in range(num_steps):
        t = 1.0 - i * dt
        t_tensor = torch.full((batch_size,), t, device=device)
        
        with torch.no_grad():
            v = model(x, t_tensor, condition)
        
        x = x - dt * v
    
    return x
```

### Classifier-Free Guidance

Enhance conditioning strength:

```python
def guided_sample(model, condition, guidance_weight=7.5, num_steps=20):
    """
    Sampling with classifier-free guidance.
    
    Args:
        model: Conditional model trained with dropout
        condition: Conditioning variable
        guidance_weight: Guidance strength (w)
        num_steps: ODE steps
    
    Returns:
        samples: Guided samples
    """
    x = torch.randn(batch_size, *data_shape, device=device)
    dt = 1.0 / num_steps
    
    # Null condition (empty)
    null_condition = torch.zeros_like(condition)
    
    for i in range(num_steps):
        t = 1.0 - i * dt
        t_tensor = torch.full((batch_size,), t, device=device)
        
        with torch.no_grad():
            # Conditional velocity
            v_cond = model(x, t_tensor, condition)
            
            # Unconditional velocity
            v_uncond = model(x, t_tensor, null_condition)
            
            # Guided velocity
            v_guided = v_uncond + guidance_weight * (v_cond - v_uncond)
        
        x = x - dt * v_guided
    
    return x
```

**Effect**:

- **w = 0**: Unconditional generation
- **w = 1**: Standard conditional generation
- **w > 1**: Stronger conditioning (sharper, less diverse)
- **Typical**: w = 5-10 for images

---

## Quality-Speed Tradeoffs

### Number of Steps vs. Quality

**Empirical relationship**:

| Solver | Steps | NFE | Quality (FID) | Time |
|--------|-------|-----|---------------|------|
| Euler | 100 | 100 | Excellent | 100× |
| Euler | 50 | 50 | Good | 50× |
| RK4 | 20 | 80 | Excellent | 80× |
| RK4 | 10 | 40 | Good | 40× |
| Adaptive | Auto | 15-30 | Excellent | 15-30× |

**Key insight**: RK4 with 10-20 steps often matches Euler with 50-100 steps.

### Choosing Number of Steps

**Guidelines**:

**For images**:

- **High quality**: 20-50 steps (RK4) or 100-200 steps (Euler)
- **Balanced**: 10-20 steps (RK4) or 50-100 steps (Euler)
- **Fast**: 5-10 steps (RK4) or 20-50 steps (Euler)

**For gene expression**:

- **High quality**: 50-100 steps (Euler) or 20-30 steps (RK4)
- **Balanced**: 30-50 steps (Euler) or 10-20 steps (RK4)
- **Fast**: 10-20 steps (Euler) or 5-10 steps (RK4)

**Rule of thumb**: Start with RK4 + 20 steps, adjust based on quality needs.

### Reflow for Faster Sampling

After reflow iterations, fewer steps are needed:

**Iteration 0** (base model):
- Euler: 100 steps
- RK4: 20 steps

**Iteration 1** (1st reflow):
- Euler: 50 steps
- RK4: 10 steps

**Iteration 2** (2nd reflow):
- Euler: 20 steps
- RK4: 5 steps

**Trade-off**: More training time for faster sampling.

---

## Practical Considerations

### Batch Sampling

Generate multiple samples efficiently:

```python
def batch_sample(model, num_samples=1000, batch_size=100, num_steps=20):
    """
    Generate many samples in batches.
    """
    all_samples = []
    
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        
        # Sample batch
        x_init = torch.randn(current_batch_size, *data_shape, device=device)
        samples = rk4_sample(model, x_init, num_steps=num_steps)
        
        all_samples.append(samples.cpu())
    
    return torch.cat(all_samples, dim=0)
```

### Memory Optimization

For large models or high-resolution images:

```python
@torch.no_grad()
def memory_efficient_sample(model, x_init, num_steps=20):
    """
    Memory-efficient sampling with gradient checkpointing disabled.
    """
    model.eval()
    
    x = x_init
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = 1.0 - i * dt
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        
        # Ensure no gradients
        v = model(x, t_tensor)
        x = x - dt * v
        
        # Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    return x
```

### Deterministic Sampling

For reproducibility:

```python
def deterministic_sample(model, seed=42, batch_size=64, num_steps=20):
    """
    Deterministic sampling with fixed seed.
    """
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Sample
    x_init = torch.randn(batch_size, *data_shape, device=device)
    samples = rk4_sample(model, x_init, num_steps=num_steps)
    
    return samples
```

---

## Advanced Sampling Techniques

### Trajectory Visualization

Visualize the generation process:

```python
def sample_with_trajectory(model, x_init, num_steps=20, save_every=5):
    """
    Sample and save intermediate states.
    """
    x = x_init
    dt = 1.0 / num_steps
    trajectory = [x.clone()]
    
    for i in range(num_steps):
        t = 1.0 - i * dt
        t_tensor = torch.full((x.shape[0],), t, device=x.device)
        
        with torch.no_grad():
            v = model(x, t_tensor)
        
        x = x - dt * v
        
        # Save intermediate states
        if (i + 1) % save_every == 0:
            trajectory.append(x.clone())
    
    return x, trajectory
```

### Interpolation in Latent Space

Interpolate between samples:

```python
def interpolate_samples(model, x1_init, x2_init, num_interp=10, num_steps=20):
    """
    Interpolate between two noise samples.
    """
    # Interpolation weights
    alphas = torch.linspace(0, 1, num_interp, device=x1_init.device)
    
    interpolated_samples = []
    for alpha in alphas:
        # Interpolate in noise space
        x_init = (1 - alpha) * x1_init + alpha * x2_init
        
        # Generate sample
        sample = rk4_sample(model, x_init, num_steps=num_steps)
        interpolated_samples.append(sample)
    
    return torch.stack(interpolated_samples)
```

### Inpainting

Fill in missing regions:

```python
def inpaint(model, x_observed, mask, num_steps=20, num_iterations=5):
    """
    Inpainting with flow matching.
    
    Args:
        model: Trained flow matching model
        x_observed: Observed data (with missing regions)
        mask: Binary mask (1 = observed, 0 = missing)
        num_steps: ODE steps per iteration
        num_iterations: Number of refinement iterations
    
    Returns:
        x_inpainted: Completed image
    """
    x = torch.randn_like(x_observed)
    
    for _ in range(num_iterations):
        # Sample from model
        x_sampled = rk4_sample(model, x, num_steps=num_steps)
        
        # Replace observed regions
        x = mask * x_observed + (1 - mask) * x_sampled
    
    return x
```

---

## Comparison with Diffusion Sampling

### Conceptual Differences

| Aspect | Diffusion (DDPM) | Diffusion (DDIM) | Flow Matching |
|--------|------------------|------------------|---------------|
| **Process** | Stochastic SDE | Deterministic ODE | Deterministic ODE |
| **Noise injection** | Yes (ancestral) | No | No |
| **Steps** | 1000 (original) | 50-100 | 10-50 |
| **Solver** | Langevin dynamics | Euler/RK | Euler/RK/Adaptive |
| **Determinism** | No | Yes | Yes |
| **Speed** | Slowest | Fast | Fastest |

### Sampling Speed

**Typical performance** (ImageNet 256×256):

**DDPM**:

- 1000 steps: ~10 seconds per image (GPU)
- High quality, stochastic

**DDIM**:

- 50 steps: ~0.5 seconds per image
- Good quality, deterministic

**Flow Matching**:

- 20 steps (RK4): ~0.2 seconds per image
- Good quality, deterministic

**Flow Matching (reflow)**:

- 10 steps (RK4): ~0.1 seconds per image
- Good quality, deterministic

**Key advantage**: Flow matching is 2-5× faster than DDIM for similar quality.

---

## Evaluation Metrics

### Sample Quality

**FID (Fréchet Inception Distance)**:
```python
from pytorch_fid import fid_score

# Generate samples
samples = batch_sample(model, num_samples=50000)

# Compute FID
fid = fid_score.calculate_fid_given_paths(
    [real_data_path, samples_path],
    batch_size=50,
    device='cuda',
    dims=2048
)
print(f'FID: {fid:.2f}')
```

**Inception Score**:
```python
from torchmetrics.image.inception import InceptionScore

inception = InceptionScore(normalize=True)
inception.update(samples)
is_mean, is_std = inception.compute()
print(f'IS: {is_mean:.2f} ± {is_std:.2f}')
```

### Sampling Efficiency

**Number of Function Evaluations (NFE)**:

- Euler: NFE = num_steps
- RK4: NFE = 4 × num_steps
- Adaptive: NFE varies (typically 15-30)

**Wall-clock time**:
```python
import time

start = time.time()
samples = batch_sample(model, num_samples=1000, num_steps=20)
elapsed = time.time() - start

print(f'Time: {elapsed:.2f}s')
print(f'Samples/sec: {1000/elapsed:.2f}')
```

---

## Troubleshooting

### Common Issues

**1. Poor sample quality**:

- **Increase steps**: Try 2× more steps
- **Use RK4**: More accurate than Euler
- **Check model**: Ensure training converged
- **Use EMA weights**: Significant quality improvement

**2. Numerical instability**:

- **Reduce step size**: More steps, smaller dt
- **Clip values**: Prevent overflow
- **Use mixed precision carefully**: Can cause instability
- **Check velocity magnitudes**: Should be reasonable

**3. Slow sampling**:

- **Use fewer steps**: Start with 10-20 (RK4)
- **Batch samples**: Generate multiple at once
- **Use reflow**: Iteratively straighten paths
- **Optimize model**: TorchScript, ONNX, quantization

**4. Out of memory**:

- **Reduce batch size**: Sample in smaller batches
- **Use gradient checkpointing**: During sampling (if needed)
- **Clear cache**: torch.cuda.empty_cache()
- **Lower resolution**: If applicable

### Debugging Tips

**Visualize trajectory**:
```python
# Sample with intermediate states
final, trajectory = sample_with_trajectory(model, x_init, num_steps=20, save_every=5)

# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(trajectory), figsize=(20, 4))
for i, x in enumerate(trajectory):
    axes[i].imshow(x[0].permute(1, 2, 0).cpu())
    axes[i].set_title(f'Step {i * 5}')
plt.show()
```

**Check velocity field**:
```python
# Visualize velocity magnitudes
t = torch.tensor([0.5])
v = model(x, t)
v_norm = v.norm(dim=1).mean()
print(f'Average velocity magnitude at t=0.5: {v_norm:.4f}')
```

---

## Best Practices

### Do's

✅ **Use RK4** for better accuracy with fewer steps
✅ **Start with 20 steps** and adjust based on quality
✅ **Use EMA weights** for sampling (not training weights)
✅ **Batch samples** for efficiency
✅ **Set seeds** for reproducibility
✅ **Monitor NFE** (number of function evaluations)
✅ **Use adaptive solvers** when quality is critical

### Don'ts

❌ **Don't use too few steps** (<5 for RK4, <20 for Euler)
❌ **Don't forget EMA** (significant quality loss)
❌ **Don't sample during training** (use eval mode)
❌ **Don't ignore numerical stability** (clip if needed)
❌ **Don't use training weights** (use EMA for sampling)

---

## Summary

### Key Sampling Steps

1. **Initialize**: Sample $x(1) \sim \mathcal{N}(0, I)$
2. **Choose solver**: RK4 recommended (10-20 steps)
3. **Integrate ODE**: $\frac{dx}{dt} = v_\theta(x, t)$ from $t=1$ to $t=0$
4. **Output**: $x(0) \approx x_{\text{data}}$

### Solver Recommendations

**For quality**:

- RK4 with 20-50 steps
- Adaptive solver with tight tolerances

**For speed**:

- RK4 with 10-15 steps
- Reflow model with 5-10 steps

**For balance**:

- RK4 with 15-20 steps (recommended default)

### Typical Performance

**ImageNet 256×256**:

- **FID < 5**: 20-30 steps (RK4)
- **FID < 10**: 10-20 steps (RK4)
- **FID < 20**: 5-10 steps (RK4)

**Sampling speed**:

- ~0.2 seconds per image (20 steps, RK4, GPU)
- 2-5× faster than DDIM
- 10-50× faster than DDPM

---

## Related Documents

- [Flow Matching Foundations](01_flow_matching_foundations.md) — Theory and mathematics
- [Flow Matching Training](02_flow_matching_training.md) — Training strategies
- [DDPM Sampling](../DDPM/03_ddpm_sampling.md) — Comparison with diffusion sampling
- [Rectifying Flow Tutorial](rectifying_flow.md) — Detailed walkthrough

---

## References

### ODE Solvers

1. **Chen, R. T. Q., et al. (2018)**. Neural Ordinary Differential Equations. *NeurIPS*. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

2. **Dormand, J. R., & Prince, P. J. (1980)**. A family of embedded Runge-Kutta formulae. *Journal of Computational and Applied Mathematics*.

### Flow Matching Sampling

3. **Liu, X., et al. (2023)**. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR*.

4. **Lipman, Y., et al. (2023)**. Flow Matching for Generative Modeling. *ICLR*.

### Comparison with Diffusion

5. **Song, J., Meng, C., & Ermon, S. (2021)**. Denoising Diffusion Implicit Models. *ICLR*. (DDIM)

6. **Karras, T., et al. (2022)**. Elucidating the Design Space of Diffusion-Based Generative Models. *NeurIPS*.

### Guidance

7. **Ho, J., & Salimans, T. (2022)**. Classifier-Free Diffusion Guidance. *NeurIPS Workshop*.
