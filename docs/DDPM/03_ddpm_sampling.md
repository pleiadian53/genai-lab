# DDPM Sampling: From Noise to Data

This document covers sampling algorithms for diffusion models, including DDPM ancestral sampling, DDIM deterministic sampling, fast sampling techniques, and guidance methods.

---

## Overview

Sampling from a trained DDPM means **reversing the diffusion process**: starting from pure noise and iteratively denoising to generate data.

**Key sampling methods**:
1. **DDPM (Ancestral)**: Stochastic, adds noise at each step
2. **DDIM**: Deterministic, no added noise
3. **Fast sampling**: Fewer steps via step skipping
4. **Guidance**: Conditional generation with control

**Goal**: Understand the trade-offs between quality, speed, and diversity.

---

## DDPM Ancestral Sampling

### The Algorithm

**DDPM sampling** (Ho et al., 2020) is the original stochastic sampling procedure.

**Algorithm**:
```
1. Sample x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. Predict noise: ε_θ(x_t, t)
   b. Compute mean: μ_θ
   c. Compute variance: σ_t²
   d. Sample: x_{t-1} = μ_θ + σ_t * z
3. Return x_0
```

### The Update Formula

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z
$$

where $z \sim \mathcal{N}(0, I)$ is fresh noise at each step.

### Properties

**Pros**:
- Theoretically grounded
- High sample quality
- Diverse samples

**Cons**:
- Slow (1000 steps)
- Stochastic

**When to use**: Highest quality and diversity needed

---

## DDIM: Deterministic Sampling

### The Key Insight

**DDIM** (Song et al., 2021) follows a **deterministic ODE** instead of a stochastic SDE.

### The Update Formula

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)
$$

where $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$

### Properties

**Pros**:
- Deterministic
- Fast (can skip steps)
- Smooth interpolation

**Cons**:
- Slightly lower diversity

**When to use**: Speed, reproducibility, or interpolation needed

---

## Fast Sampling

### Step Skipping

Use a subsequence of timesteps: $\{t_S, t_{S-1}, \ldots, t_0\}$ where $S \ll T$.

**Quality vs. Speed**:
- 1000 steps: Best quality
- 250 steps: Excellent
- 50 steps: Very good
- 10 steps: Good for previews

**Best practice**: Start with 50 steps for DDIM

---

## Guidance Methods

### Classifier-Free Guidance

**Training**: Randomly drop conditioning with probability $p$ (e.g., 0.1)

**Sampling**: Interpolate predictions:

$$
\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

where $w$ is the guidance scale (typically 3-10).

**Effect**:
- $w = 1$: Standard conditional
- $w > 1$: Stronger conditioning, less diversity

**Best practice**: $w = 7.5$ for text-to-image, $w = 3-5$ for class-conditional

---

## Summary

**Key sampling methods**:

1. **DDPM**: Stochastic, 1000 steps, highest quality
2. **DDIM**: Deterministic, 50-250 steps, very good quality
3. **Fast DDIM**: 10-50 steps, good quality
4. **With guidance**: Better conditioning control

**Recommendations**:
- Default: DDIM with 50 steps
- High quality: DDPM with 1000 steps
- Fast: DDIM with 10-20 steps
- Conditional: Add classifier-free guidance

---

## Related Documents

- [DDPM Foundations](01_ddpm_foundations.md) — Mathematical theory
- [DDPM Training](02_ddpm_training.md) — Training details
- [DDIM Update Coefficients](../SDE/03b_ddim_update_coeff.md) — Exact formulas
- [Reverse SDE & Probability Flow ODE](../SDE/03a_reverse_time_sde_and_proba_flow_ode.md) — Theoretical foundation

---

## References

1. **Ho, J., Jain, A., & Abbeel, P. (2020)**. Denoising Diffusion Probabilistic Models. *NeurIPS*.
2. **Song, J., Meng, C., & Ermon, S. (2021)**. Denoising Diffusion Implicit Models. *ICLR*.
3. **Ho, J., & Salimans, T. (2022)**. Classifier-Free Diffusion Guidance. *NeurIPS Workshop*.
4. **Lu, C., et al. (2022)**. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling. *NeurIPS*.
5. **Salimans, T., & Ho, J. (2022)**. Progressive Distillation for Fast Sampling of Diffusion Models. *ICLR*.
