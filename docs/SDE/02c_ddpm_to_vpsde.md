# From DDPM to VP-SDE: The Continuous Limit

This "identity check" is one of the most satisfying derivations in diffusion theory. We'll start from the **DDPM forward discrete step**, take the **small-step limit**, and recover the **VP-SDE**:

$$
dx(t) = -\frac{1}{2}\beta(t) x(t)\,dt + \sqrt{\beta(t)}\,dw(t)
$$

We'll derive this by matching **conditional mean** and **conditional variance** of increments—the cleanest way to pass from discrete Markov chains to continuous SDEs.

---

## Overview

This derivation shows that DDPM is not an arbitrary discrete algorithm—it's a **discretization of a continuous stochastic process**. Understanding this connection:

- Unifies DDPM with the SDE framework
- Explains the variance-preserving structure
- Justifies the $\sqrt{1-\beta_k}$ coefficient
- Enables continuous-time analysis and samplers

---

## Notation

### Time Discretization

| Symbol | Meaning |
|--------|----------|
| $k = 0, 1, \ldots, N$ | Discrete DDPM time index |
| $t \in [0, T]$ | Continuous time |
| $t_k = k\,\Delta t$ | Time grid (uniform steps) |
| $\Delta t = T/N$ | Step size |

### DDPM Forward Step

The standard DDPM forward step is:

$$
x_{k+1} = \sqrt{\alpha_k}\,x_k + \sqrt{1-\alpha_k}\,\varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, I)
$$

with $\alpha_k = 1 - \beta_k$. Equivalently:

$$
x_{k+1} = \sqrt{1-\beta_k}\,x_k + \sqrt{\beta_k}\,\varepsilon_k
$$

where:

- $\beta_k$ is the discrete "noise amount" at step $k$
- In the continuous limit, we'll set $\beta_k \propto \Delta t$

---

## Step 1: Continuous-Time Scaling

To obtain an SDE limit, the per-step noise must shrink as the step size shrinks. The **standard scaling** is:

$$
\boxed{\beta_k = \beta(t_k)\,\Delta t}
$$

where $\beta(t)$ is a smooth nonnegative function called the **noise rate per unit time**.

**Intuition**: As we take finer time steps ($\Delta t \to 0$), the noise added per step must also shrink proportionally.

Substituting into the DDPM step:

$$
x_{k+1} = \sqrt{1 - \beta(t_k)\,\Delta t}\,x_k + \sqrt{\beta(t_k)\,\Delta t}\,\varepsilon_k
$$

---

## Step 2: Write the Increment

Define the increment:

$$
\Delta x_k := x_{k+1} - x_k
$$

Substituting:

$$
\Delta x_k = \left(\sqrt{1 - \beta(t_k)\,\Delta t} - 1\right) x_k + \sqrt{\beta(t_k)\,\Delta t}\,\varepsilon_k
$$

Now we'll Taylor-expand the square root term.

---

## Step 3: Taylor Expand the Square Root

For small $u$, the Taylor expansion of $\sqrt{1-u}$ is:

$$
\sqrt{1-u} = 1 - \frac{1}{2}u - \frac{1}{8}u^2 + O(u^3)
$$

With $u = \beta(t_k)\,\Delta t$:

$$
\sqrt{1 - \beta(t_k)\,\Delta t} - 1 = -\frac{1}{2}\beta(t_k)\,\Delta t + O(\Delta t^2)
$$

Substituting back:

$$
\Delta x_k = \left(-\frac{1}{2}\beta(t_k)\,\Delta t + O(\Delta t^2)\right) x_k + \sqrt{\beta(t_k)\,\Delta t}\,\varepsilon_k
$$

**Observation**: This already looks like an SDE increment!

---

## Step 4: Match Moments (The Key Move)

For an Itô SDE:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

the increment over $\Delta t$ satisfies:

**Conditional mean**:

$$
\mathbb{E}[\Delta x \mid x(t)=x] \approx f(x, t)\,\Delta t
$$

**Conditional covariance** (for isotropic noise):

$$
\text{Cov}[\Delta x \mid x(t)=x] \approx g(t)^2\,\Delta t\,I
$$

We'll compute these for DDPM and match them to identify $f$ and $g$.

### Conditional Mean

Since $\mathbb{E}[\varepsilon_k] = 0$:

$$
\mathbb{E}[\Delta x_k \mid x_k] = \left(-\frac{1}{2}\beta(t_k)\,\Delta t + O(\Delta t^2)\right) x_k
$$

Divide by $\Delta t$ and take $\Delta t \to 0$:

$$
\lim_{\Delta t \to 0} \frac{1}{\Delta t}\mathbb{E}[\Delta x_k \mid x_k] = -\frac{1}{2}\beta(t)\,x(t)
$$

**Therefore, the drift is**:

$$
\boxed{f(x, t) = -\frac{1}{2}\beta(t)\,x}
$$

### Conditional Covariance

The only random part is $\sqrt{\beta(t_k)\,\Delta t}\,\varepsilon_k$. Since $\varepsilon_k \sim \mathcal{N}(0, I)$:

$$
\text{Cov}[\Delta x_k \mid x_k] = \text{Cov}\left[\sqrt{\beta(t_k)\,\Delta t}\,\varepsilon_k\right] = \beta(t_k)\,\Delta t\,I
$$

Comparing with $g(t)^2\,\Delta t\,I$:

$$
g(t)^2 = \beta(t) \quad \Rightarrow \quad \boxed{g(t) = \sqrt{\beta(t)}}
$$

---

## Step 5: Recognize Brownian Scaling

We can rewrite the noise term:

$$
\sqrt{\beta(t_k)\,\Delta t}\,\varepsilon_k = \sqrt{\beta(t_k)} \cdot \underbrace{\left(\sqrt{\Delta t}\,\varepsilon_k\right)}_{\Delta w_k}
$$

But $\sqrt{\Delta t}\,\varepsilon_k$ is exactly a **discretized Brownian increment**:

$$
\Delta w_k \sim \mathcal{N}(0, \Delta t\,I)
$$

### The DDPM Increment

Combining everything:

$$
\Delta x_k = -\frac{1}{2}\beta(t_k)\,x_k\,\Delta t + \sqrt{\beta(t_k)}\,\Delta w_k + O(\Delta t^2)\,x_k
$$

### The Continuous Limit

In the limit $\Delta t \to 0$, this converges to the Itô SDE:

$$
\boxed{dx(t) = -\frac{1}{2}\beta(t)\,x(t)\,dt + \sqrt{\beta(t)}\,dw(t)}
$$

This is exactly the **variance-preserving SDE (VP-SDE)**.

---

## What We Proved

We can now state precisely:

> **DDPM's forward Markov chain is a discrete-time process whose small-step continuous-time limit is the VP-SDE, with drift $-\frac{1}{2}\beta(t) x$ and diffusion $\sqrt{\beta(t)}$.**

### Key Insights

1. **The $\sqrt{1-\beta_k}$ coefficient** is a variance-preserving discretization whose Taylor expansion agrees with the SDE drift to first order

2. **DDPM is not arbitrary**—it's a principled discretization of a continuous stochastic process

3. **The connection is exact**—matching moments uniquely determines both drift and diffusion

---

## Why "Variance-Preserving"?

The VP-SDE has a special property:

- The **linear drift** $-\frac{1}{2}\beta(t) x$ shrinks $x$ toward zero
- The **noise** $\sqrt{\beta(t)}\,dw$ injects variance
- The coefficients are **tuned** so the overall variance stays controlled

Under typical schedules, the process smoothly approaches a standard Gaussian at $t = T$, without variance explosion.

---

## Connection to Closed-Form Marginals

The VP-SDE has a closed-form solution:

$$
x(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right) x(0) + \text{Gaussian noise}
$$

In DDPM notation, $\bar{\alpha}_t$ corresponds to:

$$
\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

This is the **last piece** that makes discrete and continuous notations line up perfectly.

---

## Summary

We derived the VP-SDE from DDPM through:

1. **Continuous-time scaling**: $\beta_k = \beta(t_k)\,\Delta t$
2. **Taylor expansion**: $\sqrt{1-\beta_k} \approx 1 - \frac{1}{2}\beta_k$
3. **Moment matching**: Identify drift and diffusion from mean and covariance
4. **Brownian scaling**: Recognize $\sqrt{\Delta t}\,\varepsilon$ as Brownian increment

**The result**: DDPM is the Euler–Maruyama discretization of the VP-SDE, with variance-preserving modifications.

---

## Related Documents

- [Deriving DDPM from VP-SDE](02_sde_and_ddpm.md) (the reverse direction)
- [Taylor Expansions in Diffusion](02a_taylor_expansion.md)
- [Fokker–Planck Equation](02b_fokker_plank_eq.md)
- [SDE View Overview](01_diffusion_sde_view.md)
