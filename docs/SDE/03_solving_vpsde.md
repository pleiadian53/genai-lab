# Solving the VP-SDE: Closed-Form Marginals

This document provides the **exact solution** to the variance-preserving SDE (VP-SDE), showing how the continuous-time formulation connects to DDPM's discrete notation.

---

## Overview

This is the final "dictionary page" that makes DDPM notation ($\alpha_t$, $\bar{\alpha}_t$) line up **exactly** with the VP-SDE notation ($\beta(t)$, integrals).

### What We'll Show

We'll derive the closed-form marginal:

$$
x(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right) x(0) + \text{Gaussian noise}
$$

and show how $\bar{\alpha}_t$ in DDPM corresponds to $\exp\left(-\int_0^t \beta(s)\,ds\right)$ in continuous time.

**This is the key connection** that makes discrete and continuous notations line up perfectly.

### Roadmap

1. **Solve the VP-SDE**: Get the closed-form marginal $q(x_t \mid x_0)$
2. **Identify $\bar{\alpha}(t)$**: Continuous-time analogue of DDPM's $\bar{\alpha}_t$
3. **Connect discrete and continuous**: Show how products become integrals

---

## Notation

### Continuous Time

- Time: $t \in [0, T]$
- **VP-SDE**:

  $$
  dx(t) = -\frac{1}{2}\beta(t) x(t)\,dt + \sqrt{\beta(t)}\,dw(t)
  $$

  where $w(t)$ is Brownian motion and $\beta(t) \geq 0$

### DDPM Discrete Time

- Time steps: $k = 0, 1, \ldots, N$ with step size $\Delta t$
- **Notation**:
  - $\beta_k$: discrete noise amount
  - $\alpha_k := 1 - \beta_k$: signal retention
  - $\bar{\alpha}_k := \prod_{i=1}^k \alpha_i$: cumulative signal retention

---

## Step 1: Solve the VP-SDE

The VP-SDE is a **linear SDE** (similar to an Ornstein–Uhlenbeck process with time-varying rate). The standard technique is an **integrating factor**.

### Define the Integrating Factor

Let:

$$
\gamma(t) := \frac{1}{2}\int_0^t \beta(s)\,ds
$$

Define the transformed variable:

$$
y(t) := e^{\gamma(t)} x(t)
$$

**Strategy**: We'll compute $dy(t)$ and show that the drift term cancels, leaving only a pure diffusion.

---

### Differentiate $y(t)$

Using the product rule (since $e^{\gamma(t)}$ is deterministic):

$$
dy(t) = e^{\gamma(t)}\,dx(t) + x(t)\,d(e^{\gamma(t)})
$$

The differential of the exponential is:

$$
d(e^{\gamma(t)}) = e^{\gamma(t)}\,d\gamma(t) = e^{\gamma(t)} \cdot \frac{1}{2}\beta(t)\,dt
$$

Substitute $dx(t)$ from the VP-SDE:

$$
dx(t) = -\frac{1}{2}\beta(t) x(t)\,dt + \sqrt{\beta(t)}\,dw(t)
$$

Therefore:

$$
\begin{align}
dy(t) &= e^{\gamma(t)}\left(-\frac{1}{2}\beta(t) x(t)\,dt + \sqrt{\beta(t)}\,dw(t)\right) + x(t) e^{\gamma(t)} \frac{1}{2}\beta(t)\,dt \\
&= e^{\gamma(t)}\sqrt{\beta(t)}\,dw(t)
\end{align}
$$

**Key observation**: The drift terms cancel perfectly! This is why we chose this particular integrating factor.

---

### Integrate from 0 to $t$

Integrating both sides:

$$
y(t) - y(0) = \int_0^t e^{\gamma(s)}\sqrt{\beta(s)}\,dw(s)
$$

Since $y(0) = e^{\gamma(0)} x(0) = x(0)$ (because $\gamma(0) = 0$):

$$
y(t) = x(0) + \int_0^t e^{\gamma(s)}\sqrt{\beta(s)}\,dw(s)
$$

Substitute back $x(t) = e^{-\gamma(t)} y(t)$:

$$
\boxed{x(t) = e^{-\gamma(t)} x(0) + e^{-\gamma(t)} \int_0^t e^{\gamma(s)}\sqrt{\beta(s)}\,dw(s)}
$$

**This is the exact solution to the VP-SDE.**

---

## Step 2: The Marginal Distribution is Gaussian

The second term in our solution is an **Itô integral** of deterministic coefficients against Brownian motion, so it's Gaussian with mean zero.

### Mean

$$
\mathbb{E}[x(t) \mid x(0)] = e^{-\gamma(t)} x(0)
$$

### Variance

Define the noise term:

$$
\eta(t) := e^{-\gamma(t)} \int_0^t e^{\gamma(s)}\sqrt{\beta(s)}\,dw(s)
$$

The covariance of $\eta(t)$ is:

$$
\text{Cov}[\eta(t)] = e^{-2\gamma(t)} \int_0^t e^{2\gamma(s)} \beta(s)\,ds \cdot I
$$

### Beautiful Simplification

Notice that:

$$
\frac{d}{ds}\left(e^{2\gamma(s)}\right) = e^{2\gamma(s)} \cdot 2\gamma'(s) = e^{2\gamma(s)} \beta(s)
$$

Therefore:

$$
\int_0^t e^{2\gamma(s)} \beta(s)\,ds = e^{2\gamma(t)} - 1
$$

Substituting:

$$
\text{Cov}[\eta(t)] = e^{-2\gamma(t)}(e^{2\gamma(t)} - 1) I = (1 - e^{-2\gamma(t)}) I
$$

### Result

$$
\boxed{q(x_t \mid x_0) = \mathcal{N}\left(e^{-\gamma(t)} x_0, (1 - e^{-2\gamma(t)}) I\right)}
$$

---

## Step 3: Identify the Continuous-Time $\bar{\alpha}(t)$

### Compare with DDPM

Recall the DDPM marginal:

$$
q(x_k \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_k} x_0, (1 - \bar{\alpha}_k) I\right)
$$

From the SDE solution, we have:

- **Mean coefficient**: $e^{-\gamma(t)}$
- **Variance**: $1 - e^{-2\gamma(t)}$

### Define $\bar{\alpha}(t)$

To match the DDPM form, define:

$$
\boxed{\bar{\alpha}(t) := e^{-2\gamma(t)} = \exp\left(-\int_0^t \beta(s)\,ds\right)}
$$

Then:

$$
e^{-\gamma(t)} = \sqrt{\bar{\alpha}(t)}, \qquad 1 - e^{-2\gamma(t)} = 1 - \bar{\alpha}(t)
$$

### Unified Form

The SDE marginal becomes:

$$
\boxed{q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}(t)} x_0, (1 - \bar{\alpha}(t)) I\right)}
$$

where:

$$
\bar{\alpha}(t) = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

**This is the exact continuous-time version of DDPM's closed-form marginal.**

---

## Step 4: Discrete Products Become Integrals

Now we connect the discrete $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$ to the continuous form.

### Setup

Recall:

- $\alpha_i = 1 - \beta_i$
- In the SDE scaling: $\beta_i = \beta(t_i) \Delta t$ (small)

### Take Logarithms

$$
\log \bar{\alpha}_k = \sum_{i=1}^k \log(1 - \beta_i)
$$

### Taylor Expansion

For small $\beta_i$:

$$
\log(1 - \beta_i) \approx -\beta_i
$$

Therefore:

$$
\log \bar{\alpha}_k \approx -\sum_{i=1}^k \beta_i = -\sum_{i=1}^k \beta(t_i) \Delta t
$$

### Riemann Sum → Integral

As $\Delta t \to 0$, the Riemann sum becomes an integral:

$$
\log \bar{\alpha}_k \approx -\int_0^{t_k} \beta(s)\,ds
$$

### Exponentiate

$$
\boxed{\bar{\alpha}_k \approx \exp\left(-\int_0^{t_k} \beta(s)\,ds\right) = \bar{\alpha}(t_k)}
$$

**This is the precise "product → integral" dictionary.**

---

## The Complete DDPM ↔ VP-SDE Dictionary

### Discrete (DDPM)

$$
\bar{\alpha}_k = \prod_{i=1}^k (1 - \beta_i)
$$

### Continuous (VP-SDE)

$$
\bar{\alpha}(t) = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

### Marginal (Both Cases)

$$
x \approx \sqrt{\bar{\alpha}} x_0 + \sqrt{1 - \bar{\alpha}} \epsilon
$$

### Interpretation

$\bar{\alpha}$ represents the **"survival of signal"** coefficient:

- **Discrete time**: Multiplicative (product of retention factors)
- **Continuous time**: Exponential of integral (accumulated decay)

As $\Delta t \to 0$, products converge to exponentials of integrals—this is a fundamental connection in stochastic calculus.

---

## Summary

We derived the exact solution to the VP-SDE:

1. **Integrating factor method**: Transforms the SDE into pure diffusion
2. **Closed-form marginal**: Gaussian with explicit mean and variance
3. **Connection to DDPM**: $\bar{\alpha}(t) = \exp\left(-\int_0^t \beta(s)\,ds\right)$
4. **Discrete-continuous bridge**: Products become integrals in the limit

**Key insight**: The continuous VP-SDE and discrete DDPM describe the same underlying process, just in different time parameterizations.

---

## Related Documents

- [Reverse-Time SDE and Probability Flow ODE](03a_reverse_time_sde_and_proba_flow_ode.md) — How to sample (next)
- [DDPM to VP-SDE](02c_ddpm_to_vpsde.md) — Deriving the SDE from DDPM
- [VP-SDE to DDPM](02_sde_and_ddpm.md) — Deriving DDPM from the SDE
- [SDE View Overview](01_diffusion_sde_view.md) — Conceptual introduction
