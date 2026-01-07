# Noise Schedules for Diffusion Models

## Overview

The noise schedule $\beta(t)$ controls how quickly noise is added during the forward diffusion process. It's a crucial design choice that affects:
- Training speed and stability
- Sample quality
- The balance between preserving signal and reaching pure noise

This document covers common noise schedule choices, their properties, and when to use each.

---

## Referenced From

This document is referenced in:
- [`docs/diffusion/forward_process_derivation.md`](./forward_process_derivation.md) — Forward SDE derivation

---

## Mathematical Background

Before diving into specific schedules, let's clarify the key quantities and their relationships.

### Definitions

**Noise schedule** $\beta(t)$:
- Controls the rate of noise addition at time $t$
- This is what you **design/choose**
- Appears in the VP-SDE: $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$

**Signal coefficient** $\alpha_t$ or $\alpha(t)$:
- Related to how much of the original signal remains
- Defined as: $\alpha(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right)$
- Sometimes written as: $\sqrt{\bar{\alpha}_t} = \alpha(t)$

**Cumulative signal coefficient** $\bar{\alpha}_t$:
- The square of the signal coefficient: $\bar{\alpha}_t = \alpha(t)^2$
- More commonly used in formulas
- Defined as: $\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)$

**Why these specific forms?** These definitions emerge from solving the VP-SDE using the integrating factor technique. See **[`alpha_definitions_derivation.md`](./alpha_definitions_derivation.md)** for the complete derivation showing how $\alpha(t) = 1/\mu(t)$ where $\mu(t)$ is the integrating factor.

### The Forward Process

The clean data $x_0$ is corrupted into noisy data $x_t$:

$$
x_t = \underbrace{\sqrt{\bar{\alpha}_t}}_{\text{signal scale}} x_0 + \underbrace{\sqrt{1-\bar{\alpha}_t}}_{\text{noise scale}} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**Interpretation**:
- When $\bar{\alpha}_t = 1$: Pure signal ($x_t = x_0$)
- When $\bar{\alpha}_t = 0$: Pure noise ($x_t = \varepsilon$)
- The schedule $\beta(t)$ determines how $\bar{\alpha}_t$ decays from 1 to 0

### Relationship Summary

$$
\beta(t) \quad \xrightarrow{\text{integrate}} \quad \bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

**In practice**:
- You **choose** $\beta(t)$ (the noise schedule)
- This **determines** $\bar{\alpha}_t$ via integration
- Alternatively, you can **choose** $\bar{\alpha}_t$ directly and derive $\beta(t)$ from it

**Example**: For linear schedule $\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})t$:

$$
\bar{\alpha}_t = \exp\left(-\int_0^t [\beta_{\min} + (\beta_{\max} - \beta_{\min})s]\,ds\right) = \exp\left(-\beta_{\min}t - \frac{1}{2}(\beta_{\max} - \beta_{\min})t^2\right)
$$

---

## Common Noise Schedule Choices

### 1. Linear Schedule

**Formula**:
$$
\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \cdot t
$$

**Properties**:
- Simple and interpretable
- Noise increases linearly from $\beta_{\min}$ to $\beta_{\max}$
- Used in early DDPM papers (Ho et al., 2020)

**Typical values**: $\beta_{\min} = 0.0001$, $\beta_{\max} = 0.02$

**Cumulative**:
$$
\bar{\alpha}_t = \exp\left(-\frac{1}{2}(\beta_{\min} t + \frac{1}{2}(\beta_{\max} - \beta_{\min}) t^2)\right)
$$

**When to use**: Good default for initial experiments and simple datasets.

---

### 2. Cosine Schedule

**Formula**:
$$
\beta(t) = 1 - \cos\left(\frac{\pi t}{2}\right)
$$

Or in terms of $\bar{\alpha}_t$ directly:
$$
\bar{\alpha}_t = \cos\left(\frac{\pi t}{2}\right)^2
$$

**Properties**:
- Noise increases slowly at first, then accelerates
- Better preserves signal at early timesteps
- Often produces higher quality samples
- Popular in modern diffusion models (Nichol & Dhariwal, 2021)

**Intuition**: The cosine function starts flat (slow noise addition) and becomes steeper (faster noise addition) as $t \to 1$.

**When to use**: Preferred for high-quality image generation and when training stability is important.

---

### 3. Polynomial Schedule

**Formula**:
$$
\beta(t) = t^n, \quad n > 0
$$

**Properties**:
- $n < 1$: Noise added faster at the beginning
- $n = 1$: Linear schedule
- $n > 1$: Noise added faster at the end

**Cumulative**:
$$
\bar{\alpha}_t = \exp\left(-\frac{t^{n+1}}{2(n+1)}\right)
$$

**When to use**: When you want to experiment with different temporal profiles. Useful for ablation studies.

---

### 4. Sigmoid Schedule

**Formula**:
$$
\beta(t) = \frac{\beta_{\max}}{1 + \exp(-k(t - t_0))}
$$

**Properties**:
- S-shaped curve
- Slow at beginning and end, fast in the middle
- Less commonly used

**Parameters**:
- $k$: Controls steepness of transition
- $t_0$: Center point of transition

**When to use**: When you want a specific transition region where noise is added most rapidly.

---

### 5. Learned Schedule

Some recent work learns $\beta(t)$ as a neural network parameter, but this is still experimental.

**Advantages**:
- Potentially optimal for specific datasets
- Can adapt to data characteristics

**Disadvantages**:
- Adds complexity to training
- May overfit
- Less interpretable

---

## Comparison of Common Schedules

| Schedule | Early Noise | Late Noise | Quality | Complexity | Best For |
|----------|-------------|------------|---------|------------|----------|
| **Linear** | Moderate | Moderate | Good | Simple | Initial experiments |
| **Cosine** | Slow | Fast | Better | Simple | High-quality generation |
| **Polynomial** | Varies | Varies | Good | Moderate | Ablation studies |
| **Sigmoid** | Slow | Slow | Good | Moderate | Specific use cases |

---

## Visual Comparison

For $t \in [0, 1]$:

### $\beta(t)$ Profiles

- **Linear**: Increases steadily from $\beta_{\min}$ to $\beta_{\max}$
- **Cosine**: Starts near 0, increases slowly, then accelerates
- **Polynomial ($n=2$)**: Starts slow, accelerates quadratically

### $\bar{\alpha}_t$ Decay

All schedules aim to drive $\bar{\alpha}_t$ (signal retention) from 1 to near 0:

- **Linear**: Exponential decay with constant rate
- **Cosine**: Slower decay initially, faster later
- **Polynomial**: Decay rate depends on $n$

### Signal-to-Noise Ratio Over Time

The SNR is $\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$:

- **Linear**: Decreases exponentially
- **Cosine**: Maintains higher SNR longer at the start
- This affects which timesteps contribute most to training

---

## Why Cosine Often Works Better

### 1. Preserves Signal Early

**Problem with linear**: Too much noise added early can destroy fine details.

**Cosine solution**: Slow noise addition at $t \approx 0$ keeps more information, helping the network learn to denoise subtle features.

### 2. Efficient Corruption

**Problem**: Need to reach pure noise by $t = T$.

**Cosine solution**: Fast noise addition at $t \approx 1$ quickly reaches $\mathcal{N}(0, I)$, ensuring the reverse process starts from a well-defined distribution.

### 3. Better Training Dynamics

**Problem with linear**: Some timesteps may be over-represented or under-represented in training.

**Cosine solution**: The network sees more diverse noise levels during training because:
- More training samples at moderate noise levels
- Better gradient signal across all timesteps

### 4. Empirical Results

Nichol & Dhariwal (2021) showed that cosine schedules improve:
- FID scores on ImageNet
- Sample quality on various datasets
- Training stability

---

## Discrete-Time Equivalents

In discrete-time DDPM, the noise schedule is typically:

$$
\beta_t = \text{linear or cosine interpolation between } \beta_1 \text{ and } \beta_T
$$

The continuous-time $\beta(t)$ is the limit as the number of steps $T \to \infty$.

### Example: DDPM Linear Schedule

```python
beta_min = 0.0001
beta_max = 0.02
num_steps = 1000

# Linear interpolation
beta = np.linspace(beta_min, beta_max, num_steps)

# Compute alpha_bar
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)
```

### Example: Cosine Schedule (Nichol & Dhariwal)

```python
def cosine_beta_schedule(num_steps, s=0.008):
    """
    Cosine schedule as proposed in Nichol & Dhariwal (2021).
    """
    steps = num_steps + 1
    t = np.linspace(0, num_steps, steps)
    
    # Alpha bar from cosine
    alpha_bar = np.cos((t / num_steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # Normalize
    
    # Derive beta from alpha_bar
    alpha = alpha_bar[1:] / alpha_bar[:-1]
    beta = 1 - alpha
    
    # Clip to reasonable range
    return np.clip(beta, 0, 0.999)
```

---

## Choosing a Schedule

### Guidelines

1. **Start simple**: Use linear schedule for initial experiments
2. **For better quality**: Try cosine schedule
3. **For specific needs**: Adjust based on your data distribution
4. **Monitor**: Check that $\bar{\alpha}_T \approx 0$ (data becomes pure noise)

### Key Principle

The schedule should ensure that:
- **Early timesteps**: Preserve enough structure for the network to learn meaningful features
- **Final timestep**: Data is corrupted to approximately pure Gaussian noise $\mathcal{N}(0, I)$
- **Middle timesteps**: Smooth transition with good gradient signal

### Validation

Plot $\bar{\alpha}_t$ for your schedule and check:
- Does it start near 1? (✓)
- Does it end near 0? (✓)
- Is the transition smooth? (✓)
- Are there any abrupt changes? (✗)

---

## Advanced Topics

### Adaptive Schedules

Some recent work adjusts the schedule during training based on:
- Current loss values
- Dataset statistics
- Per-sample difficulty

### Schedule Optimization

Treating $\beta(t)$ as a hyperparameter that can be optimized:
- Grid search over schedule parameters
- Bayesian optimization
- Neural architecture search

### Data-Dependent Schedules

Adjusting the schedule based on data properties:
- Image resolution
- Complexity of structure
- Presence of fine details

---

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Default choice** | Cosine schedule |
| **Simplest** | Linear schedule |
| **Most flexible** | Polynomial schedule |
| **Best quality** | Cosine schedule (empirically) |
| **Experimental** | Learned schedule |

**Key takeaway**: The cosine schedule is preferred for most modern diffusion models due to its better signal preservation early and efficient corruption late, leading to improved sample quality.

---

## References

- **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" — Original DDPM with linear schedule
- **Nichol & Dhariwal (2021)**: "Improved Denoising Diffusion Probabilistic Models" — Introduced cosine schedule
- **Song et al. (2021)**: "Score-Based Generative Modeling through SDEs" — Continuous-time perspective
- **Karras et al. (2022)**: "Elucidating the Design Space of Diffusion-Based Generative Models" — Systematic analysis of schedules

