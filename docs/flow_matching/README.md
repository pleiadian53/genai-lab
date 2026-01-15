# Flow Matching: A Comprehensive Guide

This directory contains comprehensive documentation on **flow matching** methods for generative modeling — an alternative to score-based diffusion that learns velocity fields via simple regression.

Flow matching offers faster sampling (10-50 steps vs 100-1000 for diffusion), simpler training (direct regression), and flexible paths, making it particularly promising for biological data applications.

---

## Core Documentation Series

This series mirrors the structure of the DDPM documentation, providing a complete foundation for understanding and implementing flow matching models.

| Document | Description |
|----------|-------------|
| [01_flow_matching_foundations.md](01_flow_matching_foundations.md) | **Foundations**: Mathematical theory, forward/backward processes, theoretical properties |
| [02_flow_matching_training.md](02_flow_matching_training.md) | **Training**: Loss functions, network architectures, training strategies, reflow |
| [03_flow_matching_sampling.md](03_flow_matching_sampling.md) | **Sampling**: ODE solvers, sampling strategies, quality-speed tradeoffs |
| [04_flow_matching_landscape.md](04_flow_matching_landscape.md) | **Landscape**: Normalizing flows vs flow matching, variants comparison, historical context |
| [rectifying_flow.md](rectifying_flow.md) | **Tutorial**: Rectified flow from first principles (original tutorial) |

---

## Quick Navigation

### For Beginners
1. Start with [Rectifying Flow Tutorial](rectifying_flow.md) for intuitive introduction
2. Read [Foundations](01_flow_matching_foundations.md) for mathematical details
3. Review [Training](02_flow_matching_training.md) for implementation

### For Implementation
1. [Training Guide](02_flow_matching_training.md) — Complete training loop with code
2. [Sampling Guide](03_flow_matching_sampling.md) — ODE solvers and sampling strategies
3. [Foundations](01_flow_matching_foundations.md) — Reference for equations

### For Comparison with Diffusion
1. [Foundations](01_flow_matching_foundations.md#comparison-with-diffusion-models) — Conceptual differences
2. [Sampling](03_flow_matching_sampling.md#comparison-with-diffusion-sampling) — Speed and quality comparison
3. See [DDPM Documentation](../DDPM/README.md) for diffusion model details

---

## Key Concepts

### Flow Matching Overview

**Flow matching** learns a velocity field $v_\theta(x, t)$ that transports samples from a noise distribution to a data distribution:

$$
\frac{dx}{dt} = v_\theta(x, t)
$$

**Key advantages**:

- **Simpler training**: Direct regression instead of score matching
- **Faster sampling**: 10-50 steps (vs 100-1000 for diffusion)
- **Deterministic**: Same noise → same output
- **Flexible**: Not restricted to Gaussian noise schedules

### Rectified Flow

**Rectified flow** is the simplest and most practical instantiation:

- **Path**: Linear interpolation $x_t = (1-t) x_0 + t x_1$
- **Velocity**: Constant $v = x_1 - x_0$
- **Loss**: Simple MSE $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$
- **Sampling**: Deterministic ODE integration

**Reflow**: Iteratively straighten paths for even faster sampling (5-10 steps).

---

## Comparison with Diffusion Models

| Aspect | Score Matching (Diffusion) | Flow Matching |
|--------|---------------------------|---------------|
| **What's learned** | Score: $\nabla_x \log p_t(x)$ | Velocity: $v_\theta(x, t)$ |
| **Forward process** | Stochastic (add noise) | Deterministic (interpolate) |
| **Reverse process** | Stochastic SDE or ODE | Deterministic ODE |
| **Training** | Score matching (complex) | Simple regression |
| **Sampling steps** | 100-1000 (SDE), 50-100 (ODE) | 10-50 (ODE) |
| **Speed** | Slower | **2-5× faster** |
| **Noise schedule** | Critical design choice | Less critical |

**When to use flow matching**:

- Faster sampling is critical
- Simpler training preferred
- Exploring new domains (biology, molecules)
- Need deterministic generation

---

## Learning Path

### Conceptual Understanding
1. **[Rectifying Flow Tutorial](rectifying_flow.md)** — Intuitive introduction
   - Linear interpolation paths
   - Velocity fields
   - Why "rectified"?

2. **[Foundations](01_flow_matching_foundations.md)** — Mathematical theory
   - Probability flows
   - Conditional flow matching
   - Optimal transport connection

### Practical Implementation
3. **[Training](02_flow_matching_training.md)** — How to train
   - Loss functions
   - Network architectures (U-Net, DiT)
   - Training strategies and best practices
   - Reflow for faster sampling

4. **[Sampling](03_flow_matching_sampling.md)** — How to sample
   - ODE solvers (Euler, RK4, adaptive)
   - Quality-speed tradeoffs
   - Conditional generation and guidance

### Advanced Topics
5. **Reflow** — Iterative path straightening
6. **Conditional generation** — Class/text conditioning
7. **Classifier-free guidance** — Enhanced conditioning
8. **Domain adaptation** — Biological data applications

---

## Code Examples

### Training
```python
# Simple rectified flow training
for batch in dataloader:
    x0 = batch  # data
    x1 = torch.randn_like(x0)  # noise
    t = torch.rand(batch_size)
    
    xt = (1 - t) * x0 + t * x1
    target = x1 - x0
    
    pred = model(xt, t)
    loss = F.mse_loss(pred, target)
    loss.backward()
```

### Sampling
```python
# RK4 sampling (10-20 steps)
x = torch.randn(batch_size, *data_shape)
dt = 1.0 / num_steps

for i in range(num_steps):
    t = 1.0 - i * dt
    k1 = model(x, t)
    k2 = model(x - dt/2 * k1, t - dt/2)
    k3 = model(x - dt/2 * k2, t - dt/2)
    k4 = model(x - dt * k3, t - dt)
    x = x - dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

---

## Related Documentation

### Within GenAI Lab
- [DDPM Documentation](../DDPM/README.md) — Comparison with diffusion models
- [SDE Documentation](../SDE/README.md) — SDE perspective on diffusion
- [Evaluation Metrics](../eval/README.md) — How to evaluate generated samples
- [DiT Architecture](../diffusion/DiT/diffusion_transformer.md) — Transformer for flow matching

### External Resources
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747) — Lipman et al., 2023
- [Rectified Flow Paper](https://arxiv.org/abs/2209.03003) — Liu et al., 2023
- [Optimal Transport](https://arxiv.org/abs/2209.15571) — Albergo & Vanden-Eijnden, 2023

---

## Summary

**Flow matching** provides a simpler, faster alternative to diffusion models:

- **Training**: Direct regression on velocity fields
- **Sampling**: Deterministic ODE (10-50 steps)
- **Quality**: Comparable to diffusion models
- **Speed**: 2-5× faster than DDIM, 10-50× faster than DDPM

**Rectified flow** is the simplest instantiation, using linear interpolation paths and constant velocities. With **reflow**, sampling can be reduced to 5-10 steps while maintaining quality.

This makes flow matching particularly attractive for:
- Real-time applications
- Resource-constrained environments
- Biological data generation (gene expression, molecules)
- Domains requiring deterministic generation
