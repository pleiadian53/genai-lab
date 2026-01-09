# The SDE View of Diffusion Models

This document provides a unified mathematical perspective on diffusion models through the lens of **stochastic differential equations (SDEs)**. Understanding this view clarifies what diffusion models fundamentally do and why different variants (DDPM, NCSN, flow matching) are all solving the same core problem.

---

## 1. What Diffusion Models Really Are

At their core, diffusion models are **continuous-time stochastic processes** that transform a simple distribution (Gaussian noise) into a complex data distribution by **reversing a corruption process**.

Key insight: The corruption process is **not learned**. It is deliberately chosen to be:

- **Simple**: Analytically tractable
- **Stable**: No explosions or pathologies
- **Analyzable**: Closed-form marginals

This design choice is what makes diffusion models practical.

---

## 2. The SDE Formulation

### The Forward SDE

A diffusion model can be written as a **stochastic differential equation (SDE)**:

$$
dx(t) = f(x(t), t)\,dt + g(t)\,dw(t)
$$

This equation has two components:

**Deterministic drift**: $f(x, t)\,dt$

- The average direction the system moves
- Chosen by the model designer
- Examples: linear drift, zero drift

**Stochastic diffusion**: $g(t)\,dw(t)$

- Random fluctuations
- Noise strength controlled by $g(t)$
- Randomness from Brownian motion $w(t)$

### Brownian Motion

The randomness comes from **Brownian motion** $w(t)$, whose defining property is:

$$
w(t+dt) - w(t) \sim \mathcal{N}(0, dt)
$$

This implies the differential form:

$$
dw(t) = \sqrt{dt}\,\varepsilon \quad \text{where } \varepsilon \sim \mathcal{N}(0, 1)
$$

The $\sqrt{dt}$ scaling is **not arbitrary** — it is the only scaling that yields finite, non-trivial randomness in continuous time. This is a fundamental result from stochastic calculus.

---

## 3. Notation and Terminology

Let's be precise about what each symbol means:

| Symbol | Meaning | Type |
|--------|---------|------|
| $x(t)$ | State at time $t$ | Random variable |
| $p_t(x)$ | Probability density at time $t$ | Distribution |
| $f(x, t)$ | Drift function | Deterministic, chosen |
| $g(t)$ | Diffusion coefficient | Scalar function, chosen |
| $w(t)$ | Wiener process (Brownian motion) | Stochastic process |

**Critical point**: Nothing in the forward SDE is learned. The entire forward process is **designed**, not trained.

---

## 4. Forward vs Reverse Processes

### The Forward Process

The forward SDE gradually destroys structure:

$$
\text{clean data} \rightarrow \text{noisy data} \rightarrow \text{pure noise}
$$

This defines a family of distributions $\{p_t(x)\}_{t \in [0, T]}$. While we can sample from this process, we typically **do not know** $p_t(x)$ analytically.

### The Reverse-Time SDE

The **reverse-time SDE** reconstructs structure by running the process backward:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

where $\bar{w}$ is a reverse-time Brownian motion.

**The correction term** involves:

$$
\nabla_x \log p_t(x)
$$

This is the **score** — the gradient of the log probability density.

**Key insight**: The score is the **only unknown object** in the entire framework. Everything else is specified by the forward process design.

---

## 5. What Is Learned (and What Is Not)

### Not Learned

- The SDE form
- The drift function $f(x, t)$
- The noise schedule $g(t)$
- Brownian motion $w(t)$

### Learned

A neural network $s_\theta(x, t)$ approximates the score:

$$
s_\theta(x, t) \approx \nabla_x \log p_t(x)
$$

Interpretation: The network learns a **time-dependent vector field** that points toward regions of higher data density.

---

## 6. Training: Score Matching

Training **never solves an SDE**. Instead, it uses a simple supervised learning approach:

### Training Algorithm

1. Sample clean data $x_0 \sim p_{\text{data}}$
2. Sample time $t \sim \text{Uniform}(0, T)$
3. Corrupt $x_0$ using the known forward process to get $x_t$
4. Compute the **analytic score** of the corruption (closed-form)
5. Train neural network to match that score: $\min_\theta \|s_\theta(x_t, t) - \nabla_x \log p(x_t \mid x_0)\|^2$

### Equivalent Parameterizations

These are all mathematically equivalent:

- **Predicting noise**: $\epsilon_\theta(x_t, t) \approx \epsilon$
- **Predicting denoised data**: $\hat{x}_\theta(x_t, t) \approx x_0$
- **Predicting score**: $s_\theta(x_t, t) \approx \nabla_x \log p_t(x)$

Different papers use different parameterizations, but they're learning the same object.

---

## 7. Sampling: Numerical Integration

Generation **does** solve an equation. We have two options:

### Option 1: Reverse SDE (Stochastic)

$$
dx = \left[f(x, t) - g(t)^2 s_\theta(x, t)\right]dt + g(t)\,d\bar{w}
$$

- Stochastic sampling
- Produces diverse samples
- Requires more steps

### Option 2: Probability-Flow ODE (Deterministic)

$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2 s_\theta(x, t)
$$

- Deterministic sampling
- Same marginals as SDE
- Faster, exact likelihoods

### Numerical Solvers

Both require discretization:

| Solver | Type | Examples |
|--------|------|----------|
| Euler–Maruyama | SDE | DDPM |
| Predictor–Corrector | SDE | PC sampler |
| Runge–Kutta | ODE | DDIM, DPM-Solver |

**Key insight**: DDPM, DDIM, and modern samplers are all **discretizations** of these continuous equations.

---

## 8. Why Brownian Motion?

Brownian motion is not the only possible noise model in SDEs (finance uses jumps, stochastic volatility, etc.), but it is used in diffusion models because it guarantees:

- **Gaussian increments**: Tractable distributions
- **Markov structure**: Memoryless dynamics
- **Tractable reverse-time dynamics**: Closed-form score targets
- **Clean score matching objectives**: Simple training

This mathematical control is what makes diffusion models practical for high-dimensional data.

---

## 9. The Unifying Principle

> **Diffusion models learn a time-dependent score field that tells you how to move probability mass uphill from noise back to data.**

Everything else — DDPM, NCSN, SDEs, ODEs, samplers — is just a different coordinate system for expressing this core idea.

### Three Views, One Object

| View | Focus | Natural Formulation |
|------|-------|---------------------|
| Variational | ELBO, likelihood | DDPM (discrete) |
| Score-based | $\nabla_x \log p_t(x)$ | NCSN, reverse SDE |
| Flow-based | Deterministic transport | Probability-flow ODE |
| **SDE** | **Continuous-time umbrella** | **Unifies all three** |

---

## Next Steps

To deepen understanding:

1. **Derive DDPM from VP-SDE**: See how discrete diffusion emerges from continuous time
2. **Compare VP-SDE vs VE-SDE**: Understand different noise schedules
3. **Implement a simple sampler**: Connect theory to code

---

## References

- Song et al. (2021) - "Score-Based Generative Modeling through Stochastic Differential Equations"
- Ho et al. (2020) - "Denoising Diffusion Probabilistic Models" (DDPM)
- Song & Ermon (2019) - "Generative Modeling by Estimating Gradients of the Data Distribution" (NCSN)

---

## Related Documents

### Core Theory
- [DDPM Foundations](../DDPM/01_ddpm_foundations.md) — Discrete-time perspective
- [DDPM to VP-SDE](02c_ddpm_to_vpsde.md) — How DDPM becomes an SDE
- [Fokker-Planck Equation](02b_fokker_plank_eq.md) — How probability densities evolve

### Extensions
- [Classifier-Free Guidance](../diffusion/classifier_free_guidance.md) — Conditional generation (works in both DDPM and SDE views)

### Background
- [Historical Development](../diffusion/history/diffusion_models_development.md) — How DDPM, score-based, and SDE views unified
- [SDE Q&A](01a_diffusion_sde_view_QA.md) — Common questions
