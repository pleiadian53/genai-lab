# Sampling from Diffusion Models: The SDE Perspective

This document explains how to generate samples from trained diffusion models using the SDE perspective. Unlike training (which uses closed-form solutions), **sampling requires numerical SDE/ODE solvers**.

---

## Overview

### The Sampling Problem

**Given**: A trained score network $s_\theta(x, t) \approx \nabla_x \log p_t(x)$

**Goal**: Generate samples from the data distribution $p_0(x)$

**Approach**: Start with noise $x_T \sim \mathcal{N}(0, I)$ and run the reverse process to get $x_0$

### Two Sampling Strategies

**1. Reverse SDE (Stochastic)**

- Uses the reverse-time SDE
- Injects noise at each step (like DDPM)
- Multiple samples from same initial noise give different outputs

**2. Probability Flow ODE (Deterministic)**

- Uses a deterministic ODE with same marginals
- No noise injection (like DDIM)
- Same initial noise always gives same output

**Key difference**: SDE is stochastic, ODE is deterministic, but both have the same marginal distributions.

---

## The Reverse-Time SDE

### Mathematical Form

Given the forward SDE:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

The **reverse-time SDE** is:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

where $\bar{w}$ is reverse-time Brownian motion.

### For VP-SDE

Forward VP-SDE:

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

Reverse VP-SDE:

$$
dx = \left[-\frac{1}{2}\beta(t) x - \beta(t) \nabla_x \log p_t(x)\right]dt + \sqrt{\beta(t)}\,d\bar{w}
$$

**Using noise prediction** $\epsilon_\theta(x, t)$:

$$
\nabla_x \log p_t(x) \approx -\frac{\epsilon_\theta(x, t)}{\sqrt{1-\bar{\alpha}_t}}
$$

**Reverse SDE becomes**:

$$
dx = \left[-\frac{1}{2}\beta(t) x + \frac{\beta(t)}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x, t)\right]dt + \sqrt{\beta(t)}\,d\bar{w}
$$

---

## The Probability Flow ODE

### Mathematical Form

For any SDE with forward process:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

There exists a **probability flow ODE**:

$$
dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

**Key property**: This ODE has the **same marginal distributions** $p_t(x)$ as the SDE, but follows deterministic paths.

### For VP-SDE

Probability flow ODE:

$$
dx = \left[-\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) \nabla_x \log p_t(x)\right]dt
$$

**Using noise prediction**:

$$
dx = \left[-\frac{1}{2}\beta(t) x + \frac{\beta(t)}{2\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x, t)\right]dt
$$

**Note**: Factor of $\frac{1}{2}$ compared to reverse SDE (no noise term).

---

## Numerical Discretization

### Why We Need Solvers

Unlike training, we **cannot** use closed-form solutions for sampling because:
1. We don't know the true score $\nabla_x \log p_t(x)$ — only an approximation $s_\theta(x, t)$
2. The reverse process depends on the learned network at each step
3. We must simulate the process step-by-step

### Euler-Maruyama (EM) Method

**For SDEs**: The simplest discretization scheme.

**General form**:

$$
x_{k-1} = x_k + f(x_k, t_k)\Delta t + g(t_k)\sqrt{|\Delta t|}\,z_k
$$

where $z_k \sim \mathcal{N}(0, I)$ and $\Delta t < 0$ (going backward in time).

**For VP-SDE reverse process**:

$$
x_{k-1} = x_k + \left[-\frac{1}{2}\beta(t_k) x_k + \frac{\beta(t_k)}{\sqrt{1-\bar{\alpha}_{t_k}}} \epsilon_\theta(x_k, t_k)\right]\Delta t + \sqrt{\beta(t_k)|\Delta t|}\,z_k
$$

**This is ancestral sampling** (DDPM-style).

### Euler Method for ODE

**For ODEs**: No noise term.

**General form**:

$$
x_{k-1} = x_k + f(x_k, t_k)\Delta t
$$

**For probability flow ODE**:

$$
x_{k-1} = x_k + \left[-\frac{1}{2}\beta(t_k) x_k + \frac{\beta(t_k)}{2\sqrt{1-\bar{\alpha}_{t_k}}} \epsilon_\theta(x_k, t_k)\right]\Delta t
$$

**This is DDIM-style sampling**.

---

## Sampling Algorithms

### Ancestral Sampling (Reverse SDE)

**Pseudocode**:

```python
def ancestral_sampling(model, shape, num_steps=1000, T=1.0):
    """
    Sample using reverse SDE (stochastic).
    Corresponds to DDPM sampling.
    """
    # Start from pure noise
    x = torch.randn(shape)
    
    # Time discretization
    dt = -T / num_steps  # Negative (going backward)
    
    for k in range(num_steps):
        t = T - k * abs(dt)  # Current time
        
        # Compute β(t) and α̅_t
        beta_t = compute_beta(t)
        alpha_bar_t = compute_alpha_bar(t)
        
        # Predict noise
        epsilon_pred = model(x, t)
        
        # Compute drift
        drift = -0.5 * beta_t * x + beta_t / sqrt(1 - alpha_bar_t) * epsilon_pred
        
        # Compute diffusion (noise term)
        if k < num_steps - 1:  # No noise at final step
            noise = torch.randn_like(x)
            diffusion = sqrt(beta_t * abs(dt)) * noise
        else:
            diffusion = 0
        
        # Update
        x = x + drift * dt + diffusion
    
    return x
```

### DDIM Sampling (Probability Flow ODE)

**Pseudocode**:

```python
def ddim_sampling(model, shape, num_steps=50, T=1.0):
    """
    Sample using probability flow ODE (deterministic).
    Corresponds to DDIM sampling.
    """
    # Start from pure noise
    x = torch.randn(shape)
    
    # Time discretization
    dt = -T / num_steps  # Negative (going backward)
    
    for k in range(num_steps):
        t = T - k * abs(dt)  # Current time
        
        # Compute β(t) and α̅_t
        beta_t = compute_beta(t)
        alpha_bar_t = compute_alpha_bar(t)
        
        # Predict noise
        epsilon_pred = model(x, t)
        
        # Compute ODE drift (factor of 1/2 compared to SDE)
        drift = -0.5 * beta_t * x + 0.5 * beta_t / sqrt(1 - alpha_bar_t) * epsilon_pred
        
        # Update (no noise term)
        x = x + drift * dt
    
    return x
```

### PyTorch Implementation

```python
import torch
import math

class SDESampler:
    def __init__(self, model, beta_schedule, T=1.0):
        self.model = model
        self.T = T
        self.beta_min = beta_schedule['beta_min']
        self.beta_max = beta_schedule['beta_max']
    
    def compute_beta(self, t):
        """Linear schedule: β(t) = β_min + t(β_max - β_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def compute_alpha_bar(self, t):
        """α̅(t) = exp(-∫₀ᵗ β(s)ds)"""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        return torch.exp(-integral)
    
    @torch.no_grad()
    def sample_sde(self, shape, num_steps=1000, device='cuda'):
        """Ancestral sampling (stochastic)."""
        x = torch.randn(shape, device=device)
        dt = -self.T / num_steps
        
        for k in range(num_steps):
            t = self.T - k * abs(dt)
            t_tensor = torch.full((shape[0],), t, device=device)
            
            # Compute schedule values
            beta_t = self.compute_beta(t_tensor)
            alpha_bar_t = self.compute_alpha_bar(t_tensor)
            
            # Reshape for broadcasting
            beta_t = beta_t.view(-1, 1, 1, 1)
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
            
            # Predict noise
            epsilon_pred = self.model(x, t_tensor)
            
            # Drift term
            drift = -0.5 * beta_t * x + beta_t / torch.sqrt(1 - alpha_bar_t) * epsilon_pred
            
            # Diffusion term
            if k < num_steps - 1:
                noise = torch.randn_like(x)
                diffusion = torch.sqrt(beta_t * abs(dt)) * noise
            else:
                diffusion = 0
            
            # Update
            x = x + drift * dt + diffusion
        
        return x
    
    @torch.no_grad()
    def sample_ode(self, shape, num_steps=50, device='cuda'):
        """DDIM sampling (deterministic)."""
        x = torch.randn(shape, device=device)
        dt = -self.T / num_steps
        
        for k in range(num_steps):
            t = self.T - k * abs(dt)
            t_tensor = torch.full((shape[0],), t, device=device)
            
            # Compute schedule values
            beta_t = self.compute_beta(t_tensor)
            alpha_bar_t = self.compute_alpha_bar(t_tensor)
            
            # Reshape for broadcasting
            beta_t = beta_t.view(-1, 1, 1, 1)
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
            
            # Predict noise
            epsilon_pred = self.model(x, t_tensor)
            
            # ODE drift (factor of 1/2)
            drift = -0.5 * beta_t * x + 0.5 * beta_t / torch.sqrt(1 - alpha_bar_t) * epsilon_pred
            
            # Update (no noise)
            x = x + drift * dt
        
        return x

# Usage
sampler = SDESampler(
    model=trained_model,
    beta_schedule={'beta_min': 0.1, 'beta_max': 20.0},
    T=1.0
)

# Stochastic sampling (1000 steps)
samples_sde = sampler.sample_sde(shape=(16, 3, 32, 32), num_steps=1000)

# Deterministic sampling (50 steps)
samples_ode = sampler.sample_ode(shape=(16, 3, 32, 32), num_steps=50)
```

---

## Higher-Order Solvers

### Runge-Kutta 4th Order (RK4)

**For ODEs**: More accurate than Euler method.

**Algorithm**:

```python
@torch.no_grad()
def sample_ode_rk4(self, shape, num_steps=50, device='cuda'):
    """DDIM sampling with RK4 solver."""
    x = torch.randn(shape, device=device)
    dt = -self.T / num_steps
    
    for k in range(num_steps):
        t = self.T - k * abs(dt)
        
        # RK4 stages
        k1 = self.ode_drift(x, t)
        k2 = self.ode_drift(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.ode_drift(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.ode_drift(x + dt * k3, t + dt)
        
        # Update
        x = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return x

def ode_drift(self, x, t):
    """Compute ODE drift at (x, t)."""
    t_tensor = torch.full((x.shape[0],), t, device=x.device)
    
    beta_t = self.compute_beta(t_tensor).view(-1, 1, 1, 1)
    alpha_bar_t = self.compute_alpha_bar(t_tensor).view(-1, 1, 1, 1)
    
    epsilon_pred = self.model(x, t_tensor)
    
    return -0.5 * beta_t * x + 0.5 * beta_t / torch.sqrt(1 - alpha_bar_t) * epsilon_pred
```

**Advantage**: Fewer steps needed for same quality (20-30 steps vs 50-100 for Euler).

### Heun's Method (2nd Order)

**For SDEs**: Improved accuracy over Euler-Maruyama.

**Algorithm**: Predictor-corrector approach
1. **Predictor**: Euler step
2. **Corrector**: Average drift at current and predicted points

```python
@torch.no_grad()
def sample_sde_heun(self, shape, num_steps=500, device='cuda'):
    """Ancestral sampling with Heun's method."""
    x = torch.randn(shape, device=device)
    dt = -self.T / num_steps
    
    for k in range(num_steps):
        t = self.T - k * abs(dt)
        
        # Predictor step
        drift_1 = self.sde_drift(x, t)
        noise = torch.randn_like(x)
        diffusion = self.sde_diffusion(t) * torch.sqrt(torch.abs(dt)) * noise
        x_pred = x + drift_1 * dt + diffusion
        
        # Corrector step
        drift_2 = self.sde_drift(x_pred, t + dt)
        x = x + 0.5 * (drift_1 + drift_2) * dt + diffusion
    
    return x
```

---

## Connection to DDPM and DDIM

### DDPM as Discretized Reverse SDE

**DDPM update rule**:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z
$$

**This is equivalent to** Euler-Maruyama discretization of the reverse VP-SDE with specific time discretization.

### DDIM as Discretized Probability Flow ODE

**DDIM update rule**:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left(\frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)
$$

**This is equivalent to** Euler discretization of the probability flow ODE.

### The η Parameter

DDIM can interpolate between deterministic and stochastic:

$$
x_{t-1} = \text{deterministic part} + \eta \sigma_t z
$$

- $\eta = 0$: Pure ODE (deterministic)
- $\eta = 1$: Full SDE (stochastic, like DDPM)
- $0 < \eta < 1$: Hybrid

---

## Sampling Strategies

### Number of Steps

**Trade-off**: Quality vs speed

**Reverse SDE (Ancestral)**:

- High quality: 1000 steps
- Medium quality: 500 steps
- Fast: 250 steps

**Probability Flow ODE (DDIM)**:

- High quality: 100 steps
- Medium quality: 50 steps
- Fast: 20-25 steps

**Rule of thumb**: ODE needs 5-10× fewer steps than SDE for similar quality.

### Non-Uniform Time Discretization

**Uniform spacing** (simple):
```python
times = torch.linspace(T, 0, num_steps)
```

**Quadratic spacing** (more steps at high noise):
```python
times = T * (1 - torch.linspace(0, 1, num_steps)**2)
```

**Cosine spacing** (from improved DDPM):
```python
s = 0.008
times = T * torch.cos((torch.linspace(0, 1, num_steps) + s) / (1 + s) * math.pi / 2)**2
```

### Adaptive Step Sizes

**Idea**: Use larger steps when error is small, smaller when error is large.

**Simple adaptive scheme**:
1. Take a full step
2. Take two half steps
3. Compare results
4. If difference is small, accept; otherwise, reduce step size

---

## Conditional Generation

### Classifier Guidance

**Modify the score** to incorporate class information:

$$
\nabla_x \log p(x_t \mid y) = \nabla_x \log p(x_t) + s \nabla_x \log p(y \mid x_t)
$$

where $s$ is the guidance scale.

**Implementation**:
```python
# Unconditional score
epsilon_uncond = model(x, t)
score_uncond = -epsilon_uncond / sqrt(1 - alpha_bar_t)

# Classifier gradient
with torch.enable_grad():
    x_in = x.detach().requires_grad_(True)
    logits = classifier(x_in, t)
    log_prob = F.log_softmax(logits, dim=-1)[..., class_label]
    classifier_grad = torch.autograd.grad(log_prob.sum(), x_in)[0]

# Guided score
score_guided = score_uncond + s * classifier_grad
```

### Classifier-Free Guidance

**Train a conditional model** $\epsilon_\theta(x_t, t, c)$ where $c$ is the condition.

**During sampling**, interpolate between conditional and unconditional:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = (1 + w) \epsilon_\theta(x_t, t, c) - w \epsilon_\theta(x_t, t, \emptyset)
$$

where $w$ is the guidance weight.

**Implementation**:
```python
# Conditional prediction
epsilon_cond = model(x, t, condition)

# Unconditional prediction (null condition)
epsilon_uncond = model(x, t, null_condition)

# Guided prediction
epsilon_guided = (1 + w) * epsilon_cond - w * epsilon_uncond
```

---

## Practical Considerations

### Memory Optimization

**Gradient checkpointing** during sampling:
```python
from torch.utils.checkpoint import checkpoint

def model_forward(x, t):
    return checkpoint(model, x, t, use_reentrant=False)
```

**Batch sampling**:
```python
# Sample in batches to avoid OOM
all_samples = []
for i in range(0, total_samples, batch_size):
    batch_samples = sampler.sample_ode(shape=(batch_size, C, H, W))
    all_samples.append(batch_samples)
samples = torch.cat(all_samples, dim=0)
```

### Deterministic Sampling

**For reproducibility**, set random seed:
```python
torch.manual_seed(42)
samples = sampler.sample_ode(shape=(16, 3, 32, 32))
```

**ODE sampling is deterministic** given the same initial noise.

### Interpolation

**Linear interpolation in latent space**:
```python
# Start from two different noise samples
z1 = torch.randn(1, C, H, W)
z2 = torch.randn(1, C, H, W)

# Interpolate
alphas = torch.linspace(0, 1, 10)
interpolated_samples = []

for alpha in alphas:
    z_interp = (1 - alpha) * z1 + alpha * z2
    sample = sampler.sample_ode_from_noise(z_interp)
    interpolated_samples.append(sample)
```

---

## Comparison: SDE vs ODE Sampling

| Aspect | Reverse SDE | Probability Flow ODE |
|--------|-------------|----------------------|
| **Stochasticity** | Stochastic | Deterministic |
| **Steps needed** | 500-1000 | 20-100 |
| **Speed** | Slower | **Faster** |
| **Diversity** | Higher | Lower |
| **Reproducibility** | Different each time | Same given same noise |
| **Interpolation** | Harder | **Easier** |
| **Likelihood** | Cannot compute | **Can compute** |
| **Corresponds to** | DDPM | DDIM |

**When to use SDE**:

- Need maximum sample diversity
- Quality is critical, speed is not

**When to use ODE**:

- Need fast sampling
- Want deterministic generation
- Need to compute likelihoods
- Want smooth interpolations

---

## Advanced Topics

### Exact Likelihood Computation

**Probability flow ODE** allows exact likelihood via change of variables:

$$
\log p_0(x_0) = \log p_T(x_T) - \int_0^T \nabla \cdot f_\theta(x_t, t)\,dt
$$

where $f_\theta$ is the ODE drift.

**Implementation** (expensive):
```python
from torchdiffeq import odeint

def compute_likelihood(x_0):
    # Encode to noise
    x_T = odeint(lambda t, x: -ode_drift(x, t), x_0, torch.tensor([0., T]))[-1]
    
    # Compute divergence integral
    def augmented_dynamics(t, state):
        x, logp = state[0], state[1]
        with torch.enable_grad():
            x = x.requires_grad_(True)
            drift = ode_drift(x, t)
            divergence = compute_divergence(drift, x)
        return drift, -divergence
    
    _, neg_log_likelihood = odeint(augmented_dynamics, (x_0, 0.), torch.tensor([0., T]))
    
    return -neg_log_likelihood + log_p_T(x_T)
```

### Inpainting

**Idea**: Constrain known pixels during sampling.

```python
def sample_with_inpainting(known_pixels, mask, num_steps=50):
    x = torch.randn(shape)
    dt = -T / num_steps
    
    for k in range(num_steps):
        t = T - k * abs(dt)
        
        # Normal ODE step
        epsilon_pred = model(x, t)
        drift = compute_ode_drift(x, t, epsilon_pred)
        x = x + drift * dt
        
        # Project onto constraint
        x = mask * known_pixels + (1 - mask) * x
    
    return x
```

---

## Key Takeaways

### Conceptual

1. **Sampling requires SDE/ODE solvers** — unlike training which uses closed-form
2. **Two strategies**: Stochastic (SDE) vs deterministic (ODE)
3. **Same marginals**: SDE and ODE produce same distributions
4. **DDPM = discretized reverse SDE**, **DDIM = discretized probability flow ODE**

### Practical

1. **Use ODE for speed** — 5-10× fewer steps than SDE
2. **Use SDE for diversity** — when quality matters more than speed
3. **Higher-order solvers help** — RK4 can reduce steps by 2-3×
4. **Non-uniform spacing** — allocate more steps to high-noise regions

### Mathematical

1. **Reverse SDE**: $dx = [f - g^2 \nabla \log p]dt + g\,d\bar{w}$
2. **Probability flow ODE**: $dx = [f - \frac{1}{2}g^2 \nabla \log p]dt$
3. **Discretization**: Euler-Maruyama (SDE), Euler/RK4 (ODE)
4. **Connection**: DDPM/DDIM are specific discretizations

---

## Related Documents

### SDE Documentation
- [00_sde_overview.md](00_sde_overview.md) — High-level SDE introduction
- [01_diffusion_sde_view.md](01_diffusion_sde_view.md) — Detailed SDE formulation
- [02_sde_training.md](02_sde_training.md) — How training works (no solvers!)
- [03a_reverse_time_sde_and_proba_flow_ode.md](03a_reverse_time_sde_and_proba_flow_ode.md) — Theoretical derivations

### Related Topics
- [DDPM Sampling](../DDPM/03_ddpm_sampling.md) — Discrete version
- [Flow Matching Sampling](../flow_matching/03_flow_matching_sampling.md) — Alternative approach

---

## Summary

**Sampling from diffusion models requires numerical solvers**:

1. **Reverse SDE**: Stochastic, 500-1000 steps, high diversity
2. **Probability Flow ODE**: Deterministic, 20-100 steps, fast
3. **Discretization methods**: Euler, Heun, RK4, adaptive
4. **DDPM/DDIM**: Specific discretizations of SDE/ODE

**Key distinction from training**:

- **Training**: Uses closed-form marginals, no solver needed
- **Sampling**: Requires numerical integration of reverse process

**Practical recommendation**: Use probability flow ODE (DDIM) with 50 steps for good balance of quality and speed.
