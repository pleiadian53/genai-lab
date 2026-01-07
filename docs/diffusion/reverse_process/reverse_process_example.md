# Concrete Example: Reversing a 1D Gaussian Diffusion

## Overview

This document provides a detailed worked example of the reverse-time SDE for a simple 1D case. This makes the abstract mathematics of reverse diffusion concrete and intuitive.

---

## Referenced From

- **Main Document**: [`docs/diffusion/reverse_process/reverse_process_derivation.md`](./reverse_process_derivation.md) — Full derivation of reverse SDE

---

## The Setup

Consider the simplest possible diffusion: a 1D random walk that becomes a Gaussian.

### Forward Process

Starting from a point at the origin $x_0 = 0$, particles undergo Brownian motion:

$$
dx = 0 \cdot dt + \sqrt{2D}\,dw
$$

**Parameters**:
- Drift: $f(x,t) = 0$ (no preferred direction)
- Diffusion coefficient: $g(t) = \sqrt{2D}$ (constant diffusion)

**Solution**: At time $t$, the probability distribution is:

$$
p_t(x) = \mathcal{N}(0, 2Dt) = \frac{1}{\sqrt{4\pi Dt}} \exp\left(-\frac{x^2}{4Dt}\right)
$$

**Properties**:
- Mean: $\mathbb{E}[x(t)] = 0$ (stays centered at origin)
- Variance: $\text{Var}(x(t)) = 2Dt$ (spreads linearly with time)

---

## Computing the Score Function

The score function is the gradient of the log probability:

$$
\nabla_x \log p_t(x) = \frac{\partial}{\partial x} \log p_t(x)
$$

### Step 1: Write the Log Probability

$$
\log p_t(x) = \log\left(\frac{1}{\sqrt{4\pi Dt}}\right) + \log\left(\exp\left(-\frac{x^2}{4Dt}\right)\right)
$$

$$
= -\frac{1}{2}\log(4\pi Dt) - \frac{x^2}{4Dt}
$$

### Step 2: Take the Derivative

$$
\frac{\partial}{\partial x} \log p_t(x) = \frac{\partial}{\partial x}\left(-\frac{1}{2}\log(4\pi Dt) - \frac{x^2}{4Dt}\right)
$$

The first term is constant (doesn't depend on $x$):

$$
\frac{\partial}{\partial x}\left(-\frac{1}{2}\log(4\pi Dt)\right) = 0
$$

The second term:

$$
\frac{\partial}{\partial x}\left(-\frac{x^2}{4Dt}\right) = -\frac{2x}{4Dt} = -\frac{x}{2Dt}
$$

### Result

$$
\boxed{\nabla_x \log p_t(x) = -\frac{x}{2Dt}}
$$

---

## Interpreting the Score

### Physical Meaning

The score $\nabla_x \log p_t(x) = -\frac{x}{2Dt}$ has a clear interpretation:

**Sign**: Always points toward $x = 0$ (the origin)
- If $x > 0$: score is negative → points left (toward origin)
- If $x < 0$: score is positive → points right (toward origin)

**Magnitude**: $|\text{score}| = \frac{|x|}{2Dt}$
- Proportional to distance from origin
- Inversely proportional to time (and diffusion coefficient)

**Intuition**: "The further you are from the center of the probability distribution, the stronger the pull back toward it."

### Visual Representation

```
Probability:           Score:
    
    │  ___                │    ╱
    │ /   \               │   ╱
p   │/     \              │  ╱
    │       \             │ ╱────── x=0
    └────────x            │╱
    -2  0  +2            -2  0  +2
    (Gaussian)           (linear, points to origin)
```

---

## The Reverse-Time SDE

### Forward SDE (for reference)

$$
dx = 0 \cdot dt + \sqrt{2D}\,dw
$$

### Reverse SDE (using Anderson's theorem)

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\,d\bar{w}(t)
$$

Substitute our values:
- $f(x,t) = 0$
- $g(t) = \sqrt{2D}$
- $\nabla_x \log p_t(x) = -\frac{x}{2Dt}$

$$
dx = \left[0 - (\sqrt{2D})^2 \cdot \left(-\frac{x}{2Dt}\right)\right] dt + \sqrt{2D}\,d\bar{w}(t)
$$

$$
dx = \left[2D \cdot \frac{x}{2Dt}\right] dt + \sqrt{2D}\,d\bar{w}(t)
$$

$$
\boxed{dx = \frac{x}{t}\,dt + \sqrt{2D}\,d\bar{w}(t)}
$$

---

## Understanding the Reverse Drift

The reverse SDE has drift:

$$
\text{drift} = \frac{x}{t}
$$

### What This Means

**Sign**: Points away from origin!
- If $x > 0$: drift is positive → pushes right (away from origin)
- If $x < 0$: drift is negative → pushes left (away from origin)

**Wait, that seems wrong!** Shouldn't we be pulling toward the origin to reverse the diffusion?

### The Key Insight

**The drift alone would push particles outward**, but the **noise term** is also present: $+\sqrt{2D}\,d\bar{w}(t)$.

When running in **reverse time**, the combination of:
1. Outward drift: $\frac{x}{t}$
2. Random noise: $\sqrt{2D}\,d\bar{w}(t)$

actually **brings the distribution back** from $\mathcal{N}(0, 2Dt)$ to $\mathcal{N}(0, 0)$ (point mass at origin).

**Mathematical fact**: This is not intuitive from looking at the drift alone. You need to analyze the Fokker-Planck equation to see that the marginal distributions correctly evolve backward.

---

## Numerical Verification

Let's verify this numerically. Starting from $p_T(x) = \mathcal{N}(0, 2DT)$ and running the reverse SDE backward, we should approach a point mass at the origin.

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 0.5
T = 1.0
num_steps = 1000
num_particles = 1000
dt = -T / num_steps  # Negative because going backward

# Initial condition: particles from N(0, 2DT)
x = np.random.normal(0, np.sqrt(2*D*T), num_particles)

# Reverse SDE: dx = (x/t)dt + sqrt(2D)dw
x_history = [x.copy()]

for i in range(num_steps):
    t = T + i * dt  # Current time (decreasing)
    
    # Drift term
    drift = x / t
    
    # Diffusion term
    noise = np.sqrt(2*D * abs(dt)) * np.random.randn(num_particles)
    
    # Update
    x = x + drift * dt + noise
    
    if i % 100 == 0:
        x_history.append(x.copy())

# Plot evolution
fig, axes = plt.subplots(1, len(x_history), figsize=(15, 3))
for i, (ax, x_snap) in enumerate(zip(axes, x_history)):
    ax.hist(x_snap, bins=30, density=True, alpha=0.7)
    ax.set_title(f't = {T - i*100*abs(dt):.2f}')
    ax.set_xlim(-3, 3)
plt.tight_layout()
plt.show()
```

**Expected result**: Distribution starts wide (Gaussian with large variance) and shrinks toward a point mass at $x=0$.

---

## Comparison: Forward vs Reverse

### Forward Process

$$
dx = 0 \cdot dt + \sqrt{2D}\,dw
$$

- **Drift**: None
- **Effect**: Pure diffusion, spreads outward
- **Variance**: Increases linearly with time

### Reverse Process

$$
dx = \frac{x}{t}\,dt + \sqrt{2D}\,d\bar{w}(t)
$$

- **Drift**: $\frac{x}{t}$ (proportional to position and inversely to time)
- **Effect**: Combines drift and diffusion to contract the distribution
- **Variance**: Decreases as $t \to 0$

### Key Difference

**Forward**: No drift needed—pure random motion spreads things out.

**Reverse**: Need drift $\frac{x}{t}$ to counteract the spreading and guide particles back to origin.

**The score term made this drift possible**: $-g^2 \nabla \log p = \frac{x}{t}$

---

## Why the Drift Points Outward (Paradox Explained)

It seems paradoxical that the reverse drift points **away from** the origin, yet the process brings particles **back to** the origin.

### Resolution

The key is understanding what "running in reverse time" means:

1. **In forward time** ($t$ increasing): 
   - Drift $\frac{x}{t}$ would push particles outward
   - But the drift coefficient decreases as $t$ increases
   - This is **not** the physical forward process (which has no drift)

2. **In reverse time** ($t$ decreasing, moving backward from $T$ to $0$):
   - We start at large $t$ (small drift coefficient)
   - As we move backward, $t$ decreases (drift coefficient increases)
   - The drift $\frac{x}{t}$ combined with noise $d\bar{w}$ (which is also reversed) produces the correct backward evolution

**Bottom line**: You cannot understand the reverse process by just looking at the drift sign. The full stochastic dynamics, including the noise term and time direction, determine the behavior.

---

## Generalizing to Diffusion Models

### The Pattern

In our example:
- Forward: $p_t(x) = \mathcal{N}(0, 2Dt)$ → distribution spreads
- Score: $\nabla \log p_t = -\frac{x}{2Dt}$ → points to center
- Reverse: Uses score to guide particles back

In diffusion models:
- Forward: $p_t(x)$ evolves from data to noise
- Score: $\nabla \log p_t(x)$ points toward data-like regions
- Reverse: Uses learned score $s_\theta(x,t)$ to guide from noise back to data

### Why We Need to Learn the Score

In our simple example, $p_t(x) = \mathcal{N}(0, 2Dt)$ is known, so we can compute $\nabla \log p_t$ analytically.

In diffusion models:
- $p_t(x)$ is complex (the distribution of partially noised images)
- We can't write it down or compute its gradient directly
- **Solution**: Train a neural network $s_\theta(x,t)$ to approximate $\nabla \log p_t(x)$

---

## Summary

| Aspect | Forward | Reverse |
|--------|---------|---------|
| **SDE** | $dx = \sqrt{2D}\,dw$ | $dx = \frac{x}{t}\,dt + \sqrt{2D}\,d\bar{w}$ |
| **Drift** | None | $\frac{x}{t}$ (from score) |
| **Score** | N/A | $\nabla \log p_t = -\frac{x}{2Dt}$ |
| **Effect** | Spread outward | Contract inward |
| **Variance** | Increases: $2Dt$ | Decreases: $2Dt \to 0$ |

**Key takeaway**: The score term $-g^2 \nabla \log p_t(x) = \frac{x}{t}$ provides the drift needed to reverse the diffusion process. Without it, we cannot bring particles back to the origin.

---

## References

- **Main Derivation**: [`docs/diffusion/reverse_process/reverse_process_derivation.md`](./reverse_process_derivation.md)
- **Anderson (1982)**: "Reverse-time diffusion equation models"
- **Song et al. (2021)**: "Score-Based Generative Modeling through SDEs"

