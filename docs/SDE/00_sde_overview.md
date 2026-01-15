# SDE Formulation of Diffusion Models: Overview

This document provides a high-level overview of the Stochastic Differential Equation (SDE) perspective on diffusion models, explaining why this view is valuable and how it connects to discrete formulations like DDPM.

---

## What is the SDE View?

### The Core Idea

Diffusion models can be understood as **continuous-time stochastic processes** that gradually transform data into noise (forward) and noise back into data (reverse).

**Key insight**: Instead of thinking about discrete timesteps (DDPM), we can view diffusion as a **continuous flow** governed by stochastic differential equations.

### Why the SDE View Matters

**1. Unified Framework**

- DDPM, NCSN, and other variants are all **discretizations** of the same continuous process
- Different sampling methods (ancestral, DDIM) correspond to different ODE/SDE solvers
- Provides theoretical foundation for understanding what diffusion models do

**2. Flexible Sampling**

- Can use any number of steps (not fixed to training schedule)
- Adaptive step sizes based on error estimates
- Trade-off between quality and speed

**3. Theoretical Clarity**

- Connects to classical stochastic calculus
- Enables rigorous analysis (Fokker-Planck equation, probability flow ODE)
- Explains why certain design choices work

**4. Generalization**

- Extends to manifolds and non-Euclidean spaces
- Connects to optimal transport and flow matching
- Provides path to new model designs

---

## The Forward SDE

### General Form

A diffusion process is defined by a **forward SDE**:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

where:

- $x(t) \in \mathbb{R}^d$ is the data state at time $t$
- $f(x, t)$ is the **drift** (deterministic component)
- $g(t)$ is the **diffusion coefficient** (noise strength)
- $w(t)$ is **Brownian motion** (random component)

### Interpretation

**Drift term** $f(x, t)\,dt$:
- Average direction of movement
- Designed by model creator (not learned)
- Controls how data is corrupted

**Diffusion term** $g(t)\,dw$:
- Random fluctuations
- Noise strength varies with time
- Brownian motion provides randomness

### Key Property

The forward SDE is **completely specified** — nothing is learned. It's a design choice that determines how data is gradually destroyed.

---

## The Reverse SDE

### The Fundamental Result

Given a forward SDE, there exists a **reverse-time SDE** that reconstructs the data:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

where:

- $\nabla_x \log p_t(x)$ is the **score function** (gradient of log density)
- $\bar{w}$ is reverse-time Brownian motion

**Critical observation**: The only unknown is the score function $\nabla_x \log p_t(x)$.

### What Diffusion Models Learn

Diffusion models train a neural network to approximate the score:

$$
s_\theta(x, t) \approx \nabla_x \log p_t(x)
$$

Once we have the score, we can:
1. **Sample via reverse SDE**: Stochastic sampling (like DDPM)
2. **Sample via probability flow ODE**: Deterministic sampling (like DDIM)

---

## Three Main SDE Formulations

Different choices of $f(x, t)$ and $g(t)$ lead to different diffusion models.

### 1. Variance Preserving (VP-SDE)

**SDE**:

$$

dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

**Properties**:

- Preserves variance: $\mathbb{E}[\|x(t)\|^2] \approx \text{const}$
- **Corresponds to DDPM** when discretized
- Most common in practice

**Noise schedule**: $\beta(t)$ controls corruption rate

### 2. Variance Exploding (VE-SDE)

**SDE**:

$$

dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,dw
$$

**Properties**:

- No drift term ($f = 0$)
- Variance grows: $\mathbb{E}[\|x(t)\|^2]$ increases
- **Corresponds to NCSN** (Noise Conditional Score Networks)

**Noise schedule**: $\sigma(t)$ controls noise level

### 3. Sub-VP SDE

**SDE**:

$$

dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)(1 - e^{-2\int_0^t \beta(s)ds})}\,dw
$$

**Properties**:

- Interpolates between VP and VE
- Better numerical stability
- Less common in practice

---

## Connection to DDPM

### DDPM as Discretization

DDPM is the **Euler-Maruyama discretization** of the VP-SDE:

**Continuous (VP-SDE)**:

$$

dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

**Discrete (DDPM)**:

$$

x_{t+1} = \sqrt{1 - \beta_t} \, x_t + \sqrt{\beta_t} \, \epsilon_t
$$

where $\beta_t = \beta(t) \Delta t$ for small $\Delta t$.

### Key Notation Mapping

| DDPM (Discrete) | SDE (Continuous) |
|-----------------|------------------|
| $\beta_t$ | $\beta(t) \Delta t$ |
| $\alpha_t = 1 - \beta_t$ | $1 - \frac{1}{2}\beta(t)\Delta t$ |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | $\exp(-\int_0^t \beta(s)ds)$ |
| Noise prediction $\epsilon_\theta$ | Score $s_\theta = -\epsilon_\theta / \sigma_t$ |

**Key insight**: Products in DDPM become integrals in continuous time.

---

## Sampling Methods

### Ancestral Sampling (SDE)

**Use the reverse SDE**:

$$
dx = \left[f(x, t) - g(t)^2 s_\theta(x, t)\right]dt + g(t)\,d\bar{w}
$$

**Properties**:

- Stochastic (injects noise at each step)
- Corresponds to DDPM sampling
- Multiple samples from same noise give different outputs

### Probability Flow ODE

**Use the deterministic ODE**:

$$

dx = \left[f(x, t) - \frac{1}{2}g(t)^2 s_\theta(x, t)\right]dt
$$

**Properties**:

- Deterministic (no noise injection)
- Corresponds to DDIM sampling
- Same noise always gives same output
- Faster (fewer steps needed)

**Key result**: The ODE has the **same marginals** as the SDE but follows deterministic paths.

---

## Why Score Functions?

### What is a Score?

The **score function** is the gradient of the log probability density:

$$
s(x, t) = \nabla_x \log p_t(x)
$$

**Interpretation**: Points in the direction of increasing probability density.

### Why Learn Scores Instead of Densities?

**1. Tractability**

- Computing $p_t(x)$ requires normalization (intractable)
- Computing $\nabla_x \log p_t(x)$ avoids normalization

**2. Denoising Connection**

- Score matching ≈ denoising
- Training objective is simple MSE on noise prediction

**3. Sampling Efficiency**

- Only need score to run reverse SDE
- Don't need to evaluate likelihood

### Score vs Noise Prediction

In DDPM, we predict noise $\epsilon_\theta$ instead of score directly:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sigma_t}
$$

**Why predict noise?**

- Numerically more stable
- Simpler training objective
- Better empirical performance

---

## Mathematical Foundations

### Brownian Motion

**Definition**: $w(t)$ is Brownian motion if:
1. $w(0) = 0$
2. $w(t) - w(s) \sim \mathcal{N}(0, t-s)$ for $t > s$
3. Independent increments
4. Continuous paths

**Key property**: Infinitesimal increment has variance $dt$:

$$
dw \sim \mathcal{N}(0, dt) \quad \Rightarrow \quad dw = \sqrt{dt}\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1)
$$

### Itô's Lemma

For a function $f(x, t)$ where $x$ follows an SDE:

$$
df = \left(\frac{\partial f}{\partial t} + \frac{\partial f}{\partial x} \mu + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}\sigma^2\right)dt + \frac{\partial f}{\partial x}\sigma\,dw
$$

**Key difference from ordinary calculus**: The $\frac{1}{2}\sigma^2$ term appears due to stochastic nature.

### Fokker-Planck Equation

The probability density $p_t(x)$ evolves according to:

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g^2 \nabla^2 p_t
$$

**Interpretation**: Describes how probability mass flows and diffuses over time.

---

## Advantages of SDE View

### Theoretical Benefits

✅ **Unified framework**: All diffusion variants in one formulation
✅ **Rigorous analysis**: Connects to classical stochastic calculus
✅ **Probability flow ODE**: Deterministic sampling with same marginals
✅ **Flexible discretization**: Any number of steps, adaptive solvers

### Practical Benefits

✅ **Fast sampling**: DDIM uses 10-50 steps vs 1000 for DDPM
✅ **Exact likelihood**: Can compute via probability flow ODE
✅ **Interpolation**: Smooth paths in latent space
✅ **Controllability**: Easier to design new corruption processes

### Limitations

❌ **Complexity**: Requires stochastic calculus background
❌ **Abstraction**: Less intuitive than discrete view
❌ **Implementation**: Need to choose discretization carefully

---

## Comparison: Discrete vs Continuous

| Aspect | Discrete (DDPM) | Continuous (SDE) |
|--------|----------------|------------------|
| **Time** | Fixed steps $t = 0, 1, \ldots, T$ | Continuous $t \in [0, T]$ |
| **Notation** | $\alpha_t$, $\bar{\alpha}_t$, products | $\beta(t)$, integrals |
| **Forward** | Markov chain | Stochastic process |
| **Reverse** | Learned transitions | Reverse SDE/ODE |
| **Sampling** | Fixed schedule | Flexible steps |
| **Theory** | Discrete probability | Stochastic calculus |
| **Intuition** | More concrete | More abstract |
| **Flexibility** | Less | More |

**Key insight**: They describe the **same underlying process**, just in different mathematical languages.

---

## Learning Path

### For Beginners

1. **Start here**: Understand the basic idea (this document)
2. **DDPM connection**: See how discrete and continuous relate
3. **VP-SDE solution**: Closed-form marginals
4. **Sampling**: Reverse SDE vs probability flow ODE

### For Deep Dive

1. **Mathematical foundations**: Brownian motion, Itô calculus
2. **Fokker-Planck equation**: How probability densities evolve
3. **Score matching**: Why and how we learn scores
4. **Advanced topics**: Manifolds, optimal transport, flow matching

---

## Document Organization

This SDE documentation is organized into focused topics:

### Core Theory
- **[01_diffusion_sde_view.md](01_diffusion_sde_view.md)** — Detailed SDE formulation
- **[01a_diffusion_sde_view_QA.md](01a_diffusion_sde_view_QA.md)** — Design principles Q&A

### DDPM Connection
- **[02_sde_and_ddpm.md](02_sde_and_ddpm.md)** — Deriving DDPM from VP-SDE
- **[02c_ddpm_to_vpsde.md](02c_ddpm_to_vpsde.md)** — From DDPM to VP-SDE (reverse direction)

### Mathematical Foundations
- **[02a_taylor_expansion.md](02a_taylor_expansion.md)** — Taylor expansions in diffusion
- **[02b_fokker_plank_eq.md](02b_fokker_plank_eq.md)** — Fokker-Planck equation derivation

### Solutions and Sampling
- **[03_solving_vpsde.md](03_solving_vpsde.md)** — Exact VP-SDE solution
- **[03a_reverse_time_sde_and_proba_flow_ode.md](03a_reverse_time_sde_and_proba_flow_ode.md)** — Reverse SDE and probability flow ODE
- **[03b_ddim_update_coeff.md](03b_ddim_update_coeff.md)** — DDIM coefficients from theory

---

## Key Takeaways

### Conceptual

1. **Diffusion models are continuous-time processes** discretized for computation
2. **The forward process is designed, not learned** (key simplification)
3. **Only the score function is learned** (via denoising/noise prediction)
4. **Reverse SDE and probability flow ODE give same marginals** (stochastic vs deterministic)

### Practical

1. **DDPM is Euler-Maruyama discretization of VP-SDE**
2. **DDIM is ODE solver for probability flow ODE**
3. **Fewer steps possible with better solvers** (adaptive, higher-order)
4. **Score = negative noise / noise level** (connection to DDPM)

### Mathematical

1. **Brownian motion provides randomness** with $\sqrt{dt}$ scaling
2. **Fokker-Planck describes probability evolution** (forward equation)
3. **Reverse SDE requires score function** (Anderson's theorem)
4. **Probability flow ODE is deterministic** (same marginals as SDE)

---

## Related Documentation

### Within GenAI Lab
- [DDPM Documentation](../DDPM/README.md) — Discrete diffusion formulation
- [Flow Matching](../flow_matching/README.md) — Modern alternative to diffusion
- [Evaluation Metrics](../eval/README.md) — How to evaluate generated samples

### External Resources
- **Song et al. (2021)**: Score-Based Generative Modeling through SDEs
- **Anderson (1982)**: Reverse-time diffusion equation models
- **Särkkä & Solin (2019)**: Applied Stochastic Differential Equations

---

## Summary

The **SDE view** provides a unified, continuous-time perspective on diffusion models:

- **Forward SDE**: Gradually destroys data structure (designed, not learned)
- **Reverse SDE**: Reconstructs data by reversing the process (requires score)
- **Score function**: Gradient of log density (learned via denoising)
- **Probability flow ODE**: Deterministic alternative with same marginals

**Key advantage**: Flexibility in sampling (any number of steps, adaptive solvers) while maintaining theoretical rigor.

**Connection to practice**: DDPM and DDIM are specific discretizations of the continuous SDE/ODE framework.
