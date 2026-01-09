# DDIM Update Coefficients: From Continuous SDE to Code

This document pins the continuous-time VP-SDE all the way down to the **exact coefficients you see in DDIM code**: the $\sqrt{\bar{\alpha}_t}$, $\sqrt{1-\bar{\alpha}_t}$, and the step-to-step update from $t$ to $t'$.

---

## Overview

We'll derive the **exact DDIM update formula** through a clean "dictionary derivation":

1. **Choose a schedule**: Concrete $\beta(t)$ and define $\bar{\alpha}(t)$
2. **Discretize time**: Get discrete $\bar{\alpha}_k$ values
3. **Forward conditional**: Show how $q(x_{t'} \mid x_t, x_0)$ gives the DDIM form
4. **DDIM update**: Derive the deterministic update and the $\eta$ noise term

**Goal**: Understand why DDIM code has those specific square root coefficients and how they emerge from the continuous SDE.

---

## Notation

### Continuous Time

- Time: $t \in [0, T]$
- **VP-SDE forward**:
  $$
  dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
  $$

### Signal Survival Coefficient

$$
\boxed{\bar{\alpha}(t) := \exp\left(-\int_0^t \beta(s)\,ds\right)}
$$

### Forward Marginal (Exact)

$$
\boxed{q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}(t)} x_0, (1 - \bar{\alpha}(t)) I\right)}
$$

### Discrete Sampling Times

Pick any decreasing sequence:

$$
t_N = T > t_{N-1} > \cdots > t_0 = 0
$$

In code, these are often integer indices, but conceptually they're just a grid in $[0, T]$.

### Shorthand

$$
\bar{\alpha}_k := \bar{\alpha}(t_k), \quad x_k := x(t_k)
$$

---

## Step 1: Concrete Schedule Examples

To make $\bar{\alpha}(t)$ tangible, let's look at two common schedules.

### Example A: Constant $\beta(t) = \beta_0$

Then:

$$
\bar{\alpha}(t) = \exp(-\beta_0 t)
$$

Between two times $t$ and $t'$:

$$
\frac{\bar{\alpha}(t')}{\bar{\alpha}(t)} = e^{-\beta_0 (t' - t)}
$$

**Simple exponential decay** of signal.

### Example B: Linear $\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\frac{t}{T}$

Then:

$$
\int_0^t \beta(s)\,ds = \beta_{\min} t + \frac{\beta_{\max} - \beta_{\min}}{2T} t^2
$$

So:

$$
\bar{\alpha}(t) = \exp\left(-\beta_{\min} t - \frac{\beta_{\max} - \beta_{\min}}{2T} t^2\right)
$$

**Quadratic in the exponent** (common in DDPM).

### Connection to Code

In practice, implementations often **directly specify** discrete $\beta_k$ and compute $\bar{\alpha}_k$ by products:

$$
\bar{\alpha}_k = \prod_{i=1}^k (1 - \beta_i)
$$

The continuous view tells you these products are **approximating the integral**.

---

## Step 2: Discretization (How $\bar{\alpha}_k$ Becomes Code Constants)

### Continuous Formula

On your chosen time grid, compute:

$$
\bar{\alpha}_k = \exp\left(-\int_0^{t_k} \beta(s)\,ds\right)
$$

### Discrete Approximation

If you only have discrete $\beta_k$ at steps, approximate the integral:

$$
\int_0^{t_k} \beta(s)\,ds \approx \sum_{i=1}^k \beta(t_i) \Delta t_i
$$

Therefore:

$$
\bar{\alpha}_k \approx \exp\left(-\sum_{i=1}^k \beta(t_i) \Delta t_i\right)
$$

### DDPM Product Form

In DDPM notation with small per-step $\beta_i$:

$$
\bar{\alpha}_k = \prod_{i=1}^k (1 - \beta_i) \approx \exp\left(-\sum_{i=1}^k \beta_i\right)
$$

This matches the exponential-of-integral story!

### Why Code Has `alphas_cumprod`

$$
\boxed{\text{alphas\_cumprod}[k] = \bar{\alpha}_k}
$$

**This array is the discretized signal survival coefficient.**

---

## Step 3: The Key Conditional (Heart of DDIM)

This is where DDIM's deterministic nature emerges.

### Forward Marginal at Time $t$

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### Solve for $\epsilon$

$$
\boxed{\epsilon = \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}}}
$$

### Same Expression at Earlier Time $t' < t$

$$
x_{t'} = \sqrt{\bar{\alpha}_{t'}} x_0 + \sqrt{1 - \bar{\alpha}_{t'}} \epsilon
$$

**Key insight**: Use the **same $\epsilon$** at both times!

### Substitute $\epsilon$

$$
x_{t'} = \sqrt{\bar{\alpha}_{t'}} x_0 + \sqrt{1 - \bar{\alpha}_{t'}} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}}
$$

### Group Terms

Rearranging:

$$
\boxed{x_{t'} = \underbrace{\left(\sqrt{\bar{\alpha}_{t'}} - \sqrt{1 - \bar{\alpha}_{t'}} \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{1 - \bar{\alpha}_t}}\right)}_{\text{coefficient on } x_0} x_0 + \underbrace{\frac{\sqrt{1 - \bar{\alpha}_{t'}}}{\sqrt{1 - \bar{\alpha}_t}}}_{\text{coefficient on } x_t} x_t}
$$

**This is a deterministic mapping** from $x_t$ to $x_{t'}$ given $x_0$.

But in generation, we don't have $x_0$—so we estimate it!

---

## Step 4: Enter the Model (Estimate $x_0$ or $\epsilon$)

Most code predicts $\epsilon_\theta(x_t, t)$. Define the **estimated clean image**:

$$
\boxed{\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}}
$$

Equivalently, view it as defining a "consistent" noise:

$$
\hat{\epsilon}_\theta(x_t, t) := \epsilon_\theta(x_t, t)
$$

### The DDIM Idea

> **Use the same "direction" $\epsilon_\theta$ to move between noise levels, without injecting fresh noise.**

### Deterministic DDIM Update

The update from $t$ to $t'$ is:

$$
\boxed{x_{t'} = \sqrt{\bar{\alpha}_{t'}}\, \hat{x}_0(x_t, t) + \sqrt{1 - \bar{\alpha}_{t'}}\, \epsilon_\theta(x_t, t)}
$$

This is the **classic DDIM update** (deterministic, $\eta = 0$).

### Why This is Clean

1. **Replace signal amplitude**: $\sqrt{\bar{\alpha}_t} \to \sqrt{\bar{\alpha}_{t'}}$
2. **Replace noise amplitude**: $\sqrt{1 - \bar{\alpha}_t} \to \sqrt{1 - \bar{\alpha}_{t'}}$
3. **Keep the same predicted noise**: $\epsilon_\theta$

**This is exactly "ODE-like"**: Follow one consistent path.

### Fast Sampling

- If $t' = t - 1$ (one step): Familiar single-step form
- If $t'$ skips many steps: **Fast DDIM sampler** (e.g., 50 steps instead of 1000)

---

## Step 5: Add the $\eta$ Noise Knob (Interpolating Toward DDPM)

DDPM-style sampling injects new Gaussian noise each step. DDIM introduces parameter $\eta$ to add **controlled** randomness.

### DDIM with $\eta$ Parameter

$$
\boxed{x_{t'} = \sqrt{\bar{\alpha}_{t'}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t'} - \sigma^2}\, \epsilon_\theta + \sigma z}
$$

where $z \sim \mathcal{N}(0, I)$.

### Noise Variance $\sigma$

A common choice for $\sigma$ as a function of $\eta$ and the pair $(t, t')$:

$$
\boxed{\sigma = \eta \sqrt{\frac{1 - \bar{\alpha}_{t'}}{1 - \bar{\alpha}_t} \left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t'}}\right)}}
$$

### The Spectrum

| $\eta$ | $\sigma$ | Behavior | Sampling Type |
|--------|----------|----------|---------------|
| $0$ | $0$ | Deterministic | Probability flow ODE |
| $(0, 1)$ | Intermediate | Hybrid | Interpolation |
| $1$ | DDPM-like | Stochastic | Close to reverse SDE |

### What This Does

- **$\eta = 0 \Rightarrow \sigma = 0$**: Deterministic DDIM (probability flow ODE discretization)
- **$\eta > 0$**: Injects some noise each step (more "SDE-like")
- **$\eta = 1$**: Lands close to DDPM stochasticity (though exact DDPM reverse variance has its own specific form)

---

## Step 6: The Continuous-Time Interpretation

Now we see the complete picture:

### The Chain

1. **Continuous-time VP-SDE** gives:
   $$
   \bar{\alpha}(t) = \exp\left(-\int_0^t \beta(s)\,ds\right)
   $$

2. **Discrete sampler** chooses a grid $t_k$ and uses:
   $$
   \bar{\alpha}_k = \bar{\alpha}(t_k)
   $$

3. **DDIM** is essentially:
   - Use the model's score/noise direction
   - Transport from $\bar{\alpha}_t$ to $\bar{\alpha}_{t'}$
   - Without adding new randomness (ODE)

4. **DDPM** corresponds to:
   - Stochastic discretization (SDE)
   - Injecting noise consistent with the diffusion coefficient

### The Complete Connection

```
Continuous VP-SDE → ̄α(t) = exp(-∫β) → Discrete ̄α_k → DDIM/DDPM coefficients
```

---

## Mental Model: Why the Coefficients Feel Inevitable

### Signal Survival Interpretation

$\bar{\alpha}(t)$ represents **"how much of $x_0$ survives"** at noise level $t$.

Therefore:

- $\sqrt{\bar{\alpha}(t)}$: **Signal amplitude**
- $\sqrt{1 - \bar{\alpha}(t)}$: **Noise amplitude**

### The DDIM Insight

DDIM says: Keep the same estimated $x_0$ and $\epsilon$, but **change the amplitudes** to match the new time.

$$
\text{Old time } t: \quad x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon
$$

$$
\text{New time } t': \quad x_{t'} = \sqrt{\bar{\alpha}_{t'}}\, x_0 + \sqrt{1 - \bar{\alpha}_{t'}}\, \epsilon
$$

**That's why those square roots look so clean—they're literally amplitudes.**

---

## Summary

We derived the exact DDIM update coefficients:

1. **Schedule**: $\bar{\alpha}(t) = \exp(-\int_0^t \beta(s)\,ds)$
2. **Discretization**: $\bar{\alpha}_k = \bar{\alpha}(t_k)$ (the `alphas_cumprod` array)
3. **Key conditional**: Same $\epsilon$ at different times
4. **DDIM update**: $x_{t'} = \sqrt{\bar{\alpha}_{t'}}\, \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t'}}\, \epsilon_\theta$
5. **$\eta$ parameter**: Interpolates between ODE (deterministic) and SDE (stochastic)

**Key insight**: The continuous VP-SDE theory directly determines the exact coefficients you see in DDIM code.

---

## Related Documents

- [Solving VP-SDE](03_solving_vpsde.md) — Derivation of $\bar{\alpha}(t)$
- [Reverse SDE & Probability Flow ODE](03a_reverse_time_sde_and_proba_flow_ode.md) — Conceptual sampling framework
- [DDPM from VP-SDE](02_sde_and_ddpm.md) — Discrete-time perspective
- [DDPM Foundations](../DDPM/01_ddpm_foundations.md) — Variational formulation

---

## References

1. **Song, J., Meng, C., & Ermon, S. (2021)**. Denoising Diffusion Implicit Models. *ICLR*.
2. **Ho, J., Jain, A., & Abbeel, P. (2020)**. Denoising Diffusion Probabilistic Models. *NeurIPS*.
3. **Song, Y., et al. (2021)**. Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*.
