# Deriving the Forward Process: From SDE to Closed-Form Marginal

## The Goal

We want to derive the closed-form relationship between clean data $x_0$ and noisy data $x_t$:

$$
\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)}
$$

This formula lets us generate noisy samples at any time $t$ directly, without simulating the SDE step-by-step.

---

## Starting Point: The VP-SDE

The Variance-Preserving SDE is:

$$
dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw
$$

where:

- $\beta(t) > 0$ is the noise schedule (a design choice)
- $dw$ is the Brownian motion increment

**Goal**: Solve this SDE to find $x_t$ in terms of $x_0$.

---

## Noise Schedule Choices

The noise schedule $\beta(t)$ controls how quickly noise is added over time. It's a crucial design choice that affects training stability and sample quality.

### Common Schedules

| Schedule | When to Use | Key Property |
|----------|-------------|--------------|
| **Linear** | Initial experiments | Simple, uniform noise addition |
| **Cosine** | High-quality generation | Preserves signal early, efficient corruption |
| **Polynomial** | Ablation studies | Flexible temporal profile |

### Recommended: Cosine Schedule

The cosine schedule often produces better results because:
1. Preserves signal at early timesteps (slow noise addition)
2. Efficiently corrupts to pure noise at late timesteps
3. Better training dynamics across all noise levels

### For Details

See **[`noise_schedules.md`](./noise_schedules.md)** for:
- Complete formulas for all schedules
- Derivations of cumulative noise $\bar{\alpha}_t$
- Visual comparisons and intuitions
- Implementation examples
- Guidelines for choosing and tuning schedules

---

## Method: Solving Linear SDEs

The VP-SDE is a **linear SDE** (drift is linear in $x$). For linear SDEs, we can find closed-form solutions using integrating factors.

### General Linear SDE

A linear SDE has the form:

$$
dx = a(t)\,x\,dt + b(t)\,dw
$$

For VP-SDE: $a(t) = -\frac{1}{2}\beta(t)$ and $b(t) = \sqrt{\beta(t)}$.

### Step 1: Define the Integrating Factor

Define:

$$
\mu(t) = \exp\left(-\int_0^t a(s)\,ds\right) = \exp\left(\frac{1}{2}\int_0^t \beta(s)\,ds\right)
$$

This is chosen so that $\frac{d\mu}{dt} = -a(t)\mu(t)$.

### Step 2: Apply the Product Rule (Itô's Lemma)

Consider the product $\mu(t) x(t)$. By Itô's lemma:

$$
d(\mu x) = \mu\,dx + x\,d\mu + \underbrace{d\mu \cdot dx}_{=0 \text{ (since } d\mu \text{ is deterministic)}}
$$

Since $d\mu = -a(t)\mu\,dt$:

$$
d(\mu x) = \mu\,dx - a(t)\mu x\,dt
$$

Substitute $dx = a(t)x\,dt + b(t)\,dw$:

$$
d(\mu x) = \mu(a(t)x\,dt + b(t)\,dw) - a(t)\mu x\,dt
$$

$$
d(\mu x) = \mu a(t)x\,dt + \mu b(t)\,dw - a(t)\mu x\,dt
$$

The drift terms cancel:

$$
d(\mu x) = \mu(t) b(t)\,dw
$$

### Step 3: Integrate Both Sides

Integrate from $0$ to $t$:

$$
\mu(t) x(t) - \mu(0) x(0) = \int_0^t \mu(s) b(s)\,dw(s)
$$

Since $\mu(0) = 1$:

$$
\mu(t) x(t) = x(0) + \int_0^t \mu(s) b(s)\,dw(s)
$$

### Step 4: Solve for $x(t)$

$$
x(t) = \frac{1}{\mu(t)} x(0) + \frac{1}{\mu(t)} \int_0^t \mu(s) b(s)\,dw(s)
$$

Define:

$$
\alpha(t) = \frac{1}{\mu(t)} = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right)
$$

Then:

$$
x(t) = \alpha(t) x(0) + \alpha(t) \int_0^t \mu(s) b(s)\,dw(s)
$$

---

## Simplifying the Stochastic Integral

### The Integral is Gaussian

The stochastic integral $\int_0^t \mu(s) b(s)\,dw(s)$ is a sum of Gaussian increments, so it's Gaussian with:

- **Mean**: $0$ (Itô integrals have zero mean)
- **Variance**: $\int_0^t \mu(s)^2 b(s)^2\,ds$ (Itô isometry)

### Computing the Variance

Recall:
- $\mu(s) = \exp\left(\frac{1}{2}\int_0^s \beta(u)\,du\right)$
- $b(s) = \sqrt{\beta(s)}$

So:

$$
\mu(s)^2 b(s)^2 = \exp\left(\int_0^s \beta(u)\,du\right) \cdot \beta(s)
$$

The variance of the stochastic integral is:

$$
\text{Var} = \int_0^t \exp\left(\int_0^s \beta(u)\,du\right) \beta(s)\,ds
$$

**Trick**: Let $\Phi(s) = \int_0^s \beta(u)\,du$. Then $\frac{d\Phi}{ds} = \beta(s)$, so:

$$
\text{Var} = \int_0^t e^{\Phi(s)}\,d\Phi(s) = e^{\Phi(t)} - e^{\Phi(0)} = e^{\Phi(t)} - 1
$$

Since $\Phi(t) = \int_0^t \beta(s)\,ds$ and $\mu(t) = e^{\Phi(t)/2}$:

$$
\text{Var} = \mu(t)^2 - 1
$$

### Variance of the Noise Term in $x(t)$

The noise term in $x(t)$ is:

$$
\alpha(t) \int_0^t \mu(s) b(s)\,dw(s)
$$

Its variance is:

$$
\alpha(t)^2 \cdot (\mu(t)^2 - 1) = \frac{1}{\mu(t)^2} \cdot (\mu(t)^2 - 1) = 1 - \frac{1}{\mu(t)^2} = 1 - \alpha(t)^2
$$

---

## The Final Result

### Distribution of $x_t$

We've shown that $x_t$ given $x_0$ is Gaussian:

$$
x_t \mid x_0 \sim \mathcal{N}\left(\alpha(t) x_0, (1 - \alpha(t)^2) I\right)
$$

### Reparameterization

We can write this as:

$$
x_t = \alpha(t) x_0 + \sqrt{1 - \alpha(t)^2}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

### Standard Notation

Using $\bar{\alpha}_t = \alpha(t)^2 = \exp\left(-\int_0^t \beta(s)\,ds\right)$:

$$
\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon}
$$

where:

- $\sqrt{\bar{\alpha}_t}$ = signal coefficient (how much of $x_0$ remains)
- $\sqrt{1-\bar{\alpha}_t}$ = noise coefficient (how much noise is added)

---

## Interpreting the Formula

### At Different Times

| Time $t$ | $\bar{\alpha}_t$ | Signal | Noise | Interpretation |
|----------|------------------|--------|-------|----------------|
| $t = 0$ | $1$ | $x_0$ | $0$ | Clean data |
| Small $t$ | $\approx 1$ | $\approx x_0$ | Small | Slightly noisy |
| Large $t$ | $\approx 0$ | $\approx 0$ | $\approx \varepsilon$ | Almost pure noise |
| $t \to \infty$ | $0$ | $0$ | $\varepsilon$ | Pure Gaussian noise |

### Why "Variance-Preserving"?

The variance of $x_t$ is:

$$
\text{Var}(x_t) = \bar{\alpha}_t \cdot \text{Var}(x_0) + (1 - \bar{\alpha}_t) \cdot I
$$

If $\text{Var}(x_0) = I$ (data is pre-normalized), then:

$$
\text{Var}(x_t) = \bar{\alpha}_t \cdot I + (1 - \bar{\alpha}_t) \cdot I = I
$$

The variance is preserved at all times! This is why it's called the **Variance-Preserving SDE**.

---

## Connection to DDPM

In discrete-time DDPM, the noise schedule is:

$$
\alpha_t = \sqrt{1 - \beta_t}, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s^2 = \prod_{s=1}^t (1 - \beta_s)
$$

The continuous-time limit ($\Delta t \to 0$) gives:

$$
\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

which matches our derivation. DDPM is the discretized version of the VP-SDE.

---

## Summary

| Quantity | Formula | Meaning |
|----------|---------|---------|
| VP-SDE | $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$ | Continuous-time forward process |
| Signal coefficient | $\sqrt{\bar{\alpha}_t} = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right)$ | How much of $x_0$ survives |
| Noise coefficient | $\sqrt{1-\bar{\alpha}_t}$ | How much noise is added |
| Closed-form marginal | $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$ | Sample noisy data directly |

---

## Why This Matters for Training

During training, we need to sample $(x_t, t)$ pairs efficiently. Without the closed-form marginal, we'd have to simulate the SDE step-by-step from $t=0$ to some random $t$—expensive!

With the closed form:
1. Sample $x_0$ from data
2. Sample $t$ uniformly
3. Sample $\varepsilon \sim \mathcal{N}(0, I)$
4. Compute $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$ directly

This is fast, exact, and enables efficient training of diffusion models.

---

## References

- **Song et al. (2021)**: [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
- **Ho et al. (2020)**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Øksendal (2003)**: Stochastic Differential Equations — Chapter 5 on linear SDEs
- **Särkkä & Solin (2019)**: Applied Stochastic Differential Equations — Practical derivations

