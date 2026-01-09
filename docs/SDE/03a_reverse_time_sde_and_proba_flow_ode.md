# Reverse-Time SDE and Probability Flow ODE

This document connects the complete story of diffusion model sampling, showing how DDPM and DDIM emerge as discretizations of continuous-time processes.

---

## Overview

Now we connect the whole story:

- **VP-SDE forward**: Continuous-time noising process
- **Reverse-time SDE**: Stochastic sampler (DDPM-like)
- **Probability flow ODE**: Deterministic sampler (DDIM-like)
- **DDIM $\eta$ parameter**: Interpolates between ODE and SDE

**Key insight**: The same learned score function can be used for both stochastic and deterministic sampling.

---

## Notation

### Time Convention

- Continuous time: $t \in [0, T]$
- Convention: $t = 0$ is clean data, $t = T$ is pure noise

### VP-SDE Forward Process

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

### Signal Coefficient

$$
\bar{\alpha}(t) := \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

Forward marginal:

$$
q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}(t)} x_0, (1 - \bar{\alpha}(t)) I\right)
$$

### Score Function

$$
s(x, t) := \nabla_x \log p_t(x)
$$

Learned network: $s_\theta(x, t) \approx s(x, t)$

**Note**: Equivalently, you can predict noise $\epsilon_\theta$; we'll connect these later.

---

## Step 1: Reverse-Time SDE for VP-SDE

### General Theorem (Anderson, 1982)

For a forward Itô SDE:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

the **reverse-time SDE** (running from $T \to 0$) is:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

where $\bar{w}$ is reverse-time Brownian motion.

### Apply to VP-SDE

For the VP-SDE:

- $f(x, t) = -\frac{1}{2}\beta(t) x$
- $g(t) = \sqrt{\beta(t)}$
- $g(t)^2 = \beta(t)$

Therefore, the **reverse VP-SDE** is:

$$
\boxed{dx = \left[-\frac{1}{2}\beta(t) x - \beta(t) s(x, t)\right]dt + \sqrt{\beta(t)}\,d\bar{w}}
$$

### Interpretation

**Drift terms**:

1. $-\frac{1}{2}\beta(t) x$: Same "shrink toward zero" as forward process
2. $-\beta(t) s(x, t)$: **Score correction** that pushes toward high-density regions

**Diffusion term**: $\sqrt{\beta(t)}\,d\bar{w}$ (same magnitude as forward, but reverse-time)

**This is the continuous-time object that corresponds to DDPM sampling (stochastic).**

---

## Step 2: Probability Flow ODE

### The Surprising Result (Song et al., 2021)

There exists a **deterministic ODE** whose solution has **exactly the same marginal distributions** $p_t(x)$ as the reverse SDE.

That ODE is:

$$
\boxed{\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2 s(x, t)}
$$

### Apply to VP-SDE

For the VP-SDE:

$$
\boxed{\frac{dx}{dt} = -\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) s(x, t)}
$$

### Key Distinction

| Property | Reverse SDE | Probability Flow ODE |
|----------|-------------|----------------------|
| **Trajectories** | Stochastic | Deterministic |
| **Marginals** | $p_t(x)$ | $p_t(x)$ (same!) |
| **Noise** | Yes ($\sqrt{\beta(t)}\,d\bar{w}$) | No |
| **Sampling** | DDPM-like | DDIM-like |

**This ODE is the continuous-time conceptual ancestor of DDIM.**

### Why This Matters

- **Same score function**: Both use $s(x, t)$
- **Different dynamics**: ODE has half the score correction, no noise
- **Same marginals**: Generate from the same distribution
- **Different paths**: Individual trajectories differ, but statistics match

---

## Step 3: DDPM Sampling as SDE Discretization

### Discretize the Reverse SDE

To generate samples, discretize time: $t_N = T > t_{N-1} > \cdots > t_0 = 0$

Let $\Delta t_k = t_{k-1} - t_k$ (negative, since we go backward).

### Euler–Maruyama Step

A simple **Euler–Maruyama discretization** (backward in time):

$$
x_{k-1} = x_k + \left[-\frac{1}{2}\beta(t_k) x_k - \beta(t_k) s_\theta(x_k, t_k)\right]\Delta t_k + \sqrt{\beta(t_k) |\Delta t_k|}\, z_k
$$

where $z_k \sim \mathcal{N}(0, I)$.

### The DDPM Connection

The last term is the distinctive **"DDPM-ness"**: **fresh Gaussian noise at every step**.

DDPM typically presents this as a Gaussian transition:

$$
p_\theta(x_{k-1} \mid x_k) = \mathcal{N}(x_{k-1}; \mu_\theta(x_k, t_k), \sigma_k^2 I)
$$

But mathematically, a one-step Euler–Maruyama update **is exactly** a Gaussian transition:

- **Mean**: $x_k + \text{drift} \cdot \Delta t$
- **Variance**: $\text{diffusion}^2 \cdot |\Delta t|$

### Key Insight

$$
\boxed{\text{DDPM sampling} \approx \text{Euler–Maruyama discretization of reverse SDE}}
$$

---

## Step 4: DDIM Sampling as ODE Discretization

### Discretize the Probability Flow ODE

Now discretize the **probability flow ODE**:

$$
\frac{dx}{dt} = -\frac{1}{2}\beta(t) x - \frac{1}{2}\beta(t) s_\theta(x, t)
$$

### Euler Step

A simple **Euler discretization**:

$$
x_{k-1} = x_k + \left[-\frac{1}{2}\beta(t_k) x_k - \frac{1}{2}\beta(t_k) s_\theta(x_k, t_k)\right]\Delta t_k
$$

**Notice**: **No randomness term!**

### The DDIM Connection

$$
\boxed{\text{DDIM sampling} \approx \text{Euler discretization of probability flow ODE}}
$$

**Key difference from DDPM**: Deterministic trajectories, no added noise.

---

## Step 5: DDIM Update in $\bar{\alpha}$ Notation

This bridges back to the discrete DDPM/DDIM formulas you see in code.

### Discrete-Time Formulation

In discrete-time DDPM notation (steps $t \in \{1, \ldots, T\}$):

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

### Predict $x_0$

A network predicts $\epsilon_\theta(x_t, t)$. Form an estimate of $x_0$:

$$
\boxed{\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}}
$$

### DDIM Deterministic Update ($\eta = 0$)

$$
\boxed{x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_\theta(x_t, t)}
$$

### Why This Form is Natural

1. **Keeps the noise direction**: Uses $\epsilon_\theta(x_t, t)$ predicted at time $t$
2. **Changes the noise scale**: From $\sqrt{1 - \bar{\alpha}_t}$ to $\sqrt{1 - \bar{\alpha}_{t-1}}$
3. **No new randomness**: Deterministic update

**Interpretation**: Follow one consistent flow line rather than resampling noise at each step.

This is exactly what a deterministic probability flow ODE discretization does.

---

## Step 6: The $\eta$ Parameter (Interpolating ODE and SDE)

DDIM is often written with a parameter $\eta \in [0, 1]$ that controls extra noise:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\, \epsilon_\theta(x_t, t) + \sigma_t z
$$

where $z \sim \mathcal{N}(0, I)$ and $\sigma_t$ is chosen based on $\eta$:

### The Spectrum

| $\eta$ | $\sigma_t$ | Behavior | Corresponds to |
|--------|------------|----------|----------------|
| $0$ | $0$ | Deterministic | Probability flow ODE |
| $1$ | DDPM variance | Stochastic | Reverse SDE |
| $(0, 1)$ | Intermediate | Hybrid | Interpolation |

### Intuitive Mapping

- **Probability flow ODE** $\Leftrightarrow$ **DDIM** ($\eta = 0$)
- **Reverse SDE** $\Leftrightarrow$ **DDPM** (stochastic, "full noise")

**Key insight**: The $\eta$ parameter lets you trade off between:

- **Determinism** (faster, reproducible, good for interpolation)
- **Stochasticity** (more diverse samples, better mode coverage)

---

## Summary: The Conceptual Triangle

Here's the clean mental model:

### The Learned Object

**Score field**: $s_\theta(x, t) \approx \nabla_x \log p_t(x)$

### Two Ways to Sample

You can generate samples by evolving $x$ backward using either:

1. **Reverse SDE** (stochastic)
   - Adds noise every step
   - DDPM-like sampling
   - Equation: $dx = [f - g^2 s]\,dt + g\,d\bar{w}$

2. **Probability Flow ODE** (deterministic)
   - No added noise
   - DDIM-like sampling
   - Equation: $\frac{dx}{dt} = f - \frac{1}{2}g^2 s$

### Key Insights

- **Same score**: Both use $s_\theta(x, t)$
- **Different dynamics**: SDE adds noise, ODE doesn't
- **Same marginals**: Generate from the same distribution $p_t(x)$
- **Different trajectories**: Individual paths differ

### The Complete Picture

```
Forward SDE → Marginals p_t(x) → Score s(x,t) → Reverse SDE (DDPM)
                                              ↘
                                                Probability Flow ODE (DDIM)
```

---

## Related Documents

- [Solving the VP-SDE](03_solving_vpsde.md) — Forward process solution
- [DDPM from VP-SDE](02_sde_and_ddpm.md) — Discrete-time derivation
- [DDPM Foundations](../DDPM/01_ddpm_foundations.md) — Variational perspective
- [SDE View Overview](01_diffusion_sde_view.md) — Conceptual introduction

---

## References

1. **Anderson, B. D. O. (1982)**. Reverse-time diffusion equation models. *Stochastic Processes and their Applications*.
2. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021)**. Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*.
3. **Song, J., Meng, C., & Ermon, S. (2021)**. Denoising Diffusion Implicit Models. *ICLR*.
