# Deriving DDPM from the VP-SDE

This document shows how the discrete DDPM algorithm emerges naturally from the continuous VP-SDE through Euler–Maruyama discretization. We'll derive both the forward noising process and the reverse denoising process, making explicit why DDPM predicts noise rather than the score directly.

---

## Overview

We'll proceed in two parts:

1. **Forward VP-SDE → DDPM forward Markov chain** (Euler–Maruyama discretization)
2. **Reverse VP-SDE → DDPM reverse step** (why the network predicts noise/score)

---

## Notation

Let's establish precise notation to avoid confusion:

| Symbol | Meaning |
|--------|----------|
| $x(t) \in \mathbb{R}^d$ | Data state at continuous time $t \in [0, T]$ |
| $0 = t_0 < t_1 < \cdots < t_N = T$ | Discrete time grid |
| $\Delta t_k := t_{k+1} - t_k$ | Time step size |
| $w(t)$ | Brownian motion (Wiener process) |
| $\Delta w_k := w(t_{k+1}) - w(t_k)$ | Brownian increment, $\sim \mathcal{N}(0, \Delta t_k I)$ |
| $dw(t)$ | Infinitesimal increment: $dw \approx \sqrt{dt}\,\varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$ |
| $\beta(t) \geq 0$ | Noise rate schedule (chosen by designer) |

---

## Part A: Forward VP-SDE → DDPM Forward Noising

### Step 1: The Variance-Preserving SDE

The **variance-preserving SDE** (VP-SDE) is:

$$
dx(t) = -\frac{1}{2}\beta(t) x(t)\,dt + \sqrt{\beta(t)}\,dw(t)
$$

This SDE has:

- **Drift**: $f(x, t) = -\frac{1}{2}\beta(t) x$
- **Diffusion coefficient**: $g(t) = \sqrt{\beta(t)}$

### Step 2: Apply Euler–Maruyama Discretization

**Euler–Maruyama** is a numerical method for discretizing SDEs. For a general SDE:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

The discrete update from $t_k \to t_{k+1}$ is:

$$
x_{k+1} = x_k + f(x_k, t_k)\,\Delta t_k + g(t_k)\,\Delta w_k
$$

where $\Delta w_k := w(t_{k+1}) - w(t_k) \sim \mathcal{N}(0, \Delta t_k I)$.

**Rewrite the Brownian increment**:

$$
\Delta w_k = \sqrt{\Delta t_k}\,\varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, I)
$$

Then:

$$
x_{k+1} = x_k + f(x_k, t_k)\,\Delta t_k + g(t_k)\sqrt{\Delta t_k}\,\varepsilon_k
$$

### Step 3: Plug in VP-SDE Components

Substitute $f(x, t) = -\frac{1}{2}\beta(t) x$ and $g(t) = \sqrt{\beta(t)}$:

$$
x_{k+1} = x_k - \frac{1}{2}\beta(t_k) x_k\,\Delta t_k + \sqrt{\beta(t_k)}\sqrt{\Delta t_k}\,\varepsilon_k
$$

**Factor out $x_k$**:

$$
x_{k+1} = \left(1 - \frac{1}{2}\beta(t_k)\Delta t_k\right) x_k + \sqrt{\beta(t_k)\Delta t_k}\,\varepsilon_k
$$

**Define discrete noise parameter**:

$$
\beta_k := \beta(t_k)\,\Delta t_k
$$

Then:

$$
\boxed{x_{k+1} = \left(1 - \frac{1}{2}\beta_k\right) x_k + \sqrt{\beta_k}\,\varepsilon_k} \quad \text{(Euler–Maruyama form)}
$$

This is already a "diffusion-like" forward step!

### Step 4: Why DDPM Uses $\sqrt{1-\beta_k}$ Instead

The **actual DDPM forward step** is written as:

$$
\boxed{x_{k+1} = \sqrt{1-\beta_k}\,x_k + \sqrt{\beta_k}\,\varepsilon_k}
$$

Where did $\sqrt{1-\beta_k}$ come from instead of $1 - \frac{1}{2}\beta_k$?

**Answer**: It's a **variance-preserving tweak** that matches the first-order Taylor expansion:

$$
\sqrt{1-\beta_k} = 1 - \frac{1}{2}\beta_k + O(\beta_k^2)
$$

**Comparison**:

| Form | Accuracy | Variance Control |
|------|----------|------------------|
| $1 - \frac{1}{2}\beta_k$ | First-order accurate | Approximate |
| $\sqrt{1-\beta_k}$ | First-order accurate | **Exact** |

DDPM uses $\sqrt{1-\beta_k}$ because:

- It agrees with Euler–Maruyama to first order
- It **exactly controls variance** in discrete time
- It keeps the process well-behaved when $\beta_k$ isn't infinitesimal

**DDPM notation**: Define $\alpha_k := 1 - \beta_k$. Then:

$$
q(x_{k+1} \mid x_k) = \mathcal{N}\left(\sqrt{\alpha_k} x_k, (1-\alpha_k) I\right)
$$

Sampling form:

$$
x_{k+1} = \sqrt{\alpha_k}\,x_k + \sqrt{1-\alpha_k}\,\varepsilon_k
$$

**Key insight**: DDPM's forward chain is a **renormalized Euler step** that preserves variance exactly.

---

## Part B: Reverse VP-SDE → DDPM Reverse Denoising

### Step 1: The Reverse-Time SDE Formula

For a forward SDE:

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

The **reverse-time SDE** (running from $T \to 0$) is:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

where:

- $p_t(x)$ is the marginal density of $x(t)$
- $\nabla_x \log p_t(x)$ is the **score** (gradient of log density)
- $d\bar{w}$ is Brownian noise in reverse time

### Step 2: Apply to VP-SDE

Substitute $f = -\frac{1}{2}\beta(t) x$ and $g^2 = \beta(t)$:

$$
\boxed{dx = \left[-\frac{1}{2}\beta(t) x - \beta(t)\nabla_x \log p_t(x)\right]dt + \sqrt{\beta(t)}\,d\bar{w}}
$$

This is the **reverse diffusion equation**.

**Key observation**: The only unknown term is the score $\nabla_x \log p_t(x)$.

**Solution**: Learn a neural network:

$$
s_\theta(x, t) \approx \nabla_x \log p_t(x)
$$

### Step 3: Discretize the Reverse SDE

Apply Euler–Maruyama again, but stepping **backward** in time:

$$
x_{k-1} \approx x_k + \left[-\frac{1}{2}\beta_k x_k - \beta_k s_\theta(x_k, t_k)\right] + \sqrt{\beta_k}\,z_k
$$

where $z_k \sim \mathcal{N}(0, I)$.

(Note: We've absorbed $\Delta t$ factors into $\beta_k$ to match DDPM convention.)

This is the **SDE-solver view** of reverse sampling.

### Step 4: Connection to DDPM's Learned Gaussian

DDPM doesn't present sampling as "Euler–Maruyama on the reverse SDE." Instead, it presents it as a **learned Gaussian transition**:

$$
p_\theta(x_{k-1} \mid x_k) = \mathcal{N}\left(\mu_\theta(x_k, k), \Sigma_k\right)
$$

**These are consistent**: An Euler step of an SDE is a Gaussian update where:

- **Mean**: current state + drift term
- **Variance**: diffusion strength

### Step 5: Why Predict Noise Instead of Score?

The remaining question: Why parameterize via **noise prediction** $\varepsilon_\theta$ instead of score $s_\theta$?

**Answer**: Under the forward marginal:

$$
x_k = \sqrt{\bar{\alpha}_k} x_0 + \sqrt{1 - \bar{\alpha}_k}\,\varepsilon
$$

The conditional score has a clean identity:

$$
\nabla_{x_k} \log q(x_k \mid x_0) = -\frac{1}{\sqrt{1 - \bar{\alpha}_k}}\,\varepsilon
$$

**Therefore**: If a network predicts $\varepsilon$, it is (up to scaling) predicting the score!

### The Bridge Between Views

| View | What to Learn | Relationship |
|------|---------------|-------------|
| **SDE view** | Score $s_\theta(x, t)$ | Direct |
| **DDPM view** | Noise $\varepsilon_\theta(x_t, t)$ | $s_\theta = -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_\theta$ |

These are **equivalent up to a known scale factor**.

---

## Summary

### Forward DDPM

The forward process is the VP-SDE discretized via Euler–Maruyama, with a variance-preserving square-root coefficient:

$$
x_{k+1} = \sqrt{1-\beta_k}\,x_k + \sqrt{\beta_k}\,\varepsilon_k
$$

**Key points**:

- Emerges from continuous-time SDE
- $\sqrt{1-\beta_k}$ preserves variance exactly
- Agrees with Euler–Maruyama to first order

### Reverse DDPM

The reverse process discretizes the reverse-time SDE:

$$
x_{k-1} = x_k + \left[-\frac{1}{2}\beta_k x_k - \beta_k s_\theta(x_k, t_k)\right] + \sqrt{\beta_k}\,z_k
$$

**Key points**:

- Only unknown: the score $\nabla_x \log p_t(x)$
- Noise prediction $\varepsilon_\theta$ is a reparameterization of score
- Equivalent to learning the score up to scaling

---

## Next Steps

To deepen understanding:

1. **Continuous limit**: Start from DDPM and recover VP-SDE as $\Delta t \to 0$
2. **Variance analysis**: Verify variance preservation in the forward chain
3. **Sampling algorithms**: Derive DDPM sampler, DDIM, and other variants

---

## Related Documents

- [SDE View Overview](01_diffusion_sde_view.md)
- [Taylor Expansions in Diffusion](02a_taylor_expansion.md)
- [Historical Development](../diffusion/history/diffusion_models_development.md)
