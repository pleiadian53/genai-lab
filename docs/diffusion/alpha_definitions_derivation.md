# Where Do $\alpha(t)$ and $\bar{\alpha}_t$ Come From?

## Overview

The coefficients $\alpha(t)$ and $\bar{\alpha}_t$ appear throughout diffusion model theory, but their definitions with integrals in the exponent can seem mysterious:

$$
\alpha(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right), \quad \bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

**Key insight**: These are **not arbitrary definitions**. They emerge naturally from solving the forward SDE using the integrating factor technique.

This document explains where these definitions come from and why they have this specific form.

---

## Referenced From

- [`docs/diffusion/noise_schedules.md`](./noise_schedules.md) — Uses these definitions extensively
- [`docs/diffusion/forward_process_derivation.md`](./forward_process_derivation.md) — Full derivation of the forward process

---

## The Starting Point: The VP-SDE

The Variance-Preserving SDE describes how clean data $x_0$ is corrupted over time:

$$
dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw
$$

where:

- $x(t)$ is the state at time $t$ (starts at $x_0$)
- $\beta(t) > 0$ is the noise schedule
- $dw$ is Brownian motion

**Question**: What is the relationship between $x_t$ and $x_0$?

To answer this, we need to solve this SDE.

---

## Solving the SDE via Integrating Factor

### Step 1: Identify the SDE Structure

The VP-SDE has the form:

$$
dx = a(t) x\,dt + b(t)\,dw
$$

with $a(t) = -\frac{1}{2}\beta(t)$ and $b(t) = \sqrt{\beta(t)}$.

This is a **linear SDE** (the drift is linear in $x$), which can be solved using an integrating factor.

### Step 2: Define the Integrating Factor

For a linear SDE with drift coefficient $a(t)$, the integrating factor is:

$$
\mu(t) = \exp\left(-\int_0^t a(s)\,ds\right)
$$

In our case, $a(t) = -\frac{1}{2}\beta(t)$, so:

$$
\mu(t) = \exp\left(-\int_0^t \left(-\frac{1}{2}\beta(s)\right)\,ds\right) = \exp\left(\frac{1}{2}\int_0^t \beta(s)\,ds\right)
$$

**Why this choice?** The integrating factor is designed so that $\frac{d\mu}{dt} = -a(t)\mu(t)$, which allows the drift term to cancel when we multiply through.

### Step 3: Apply the Integrating Factor

Multiply both sides of the SDE by $\mu(t)$:

$$
\mu(t)\,dx = \mu(t) \cdot \left[-\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw\right]
$$

Using Itô's lemma on $\mu(t)x(t)$, we get:

$$
d(\mu x) = \mu\,dx + x\,d\mu
$$

Since $d\mu = -a(t)\mu\,dt = \frac{1}{2}\beta(t)\mu\,dt$:

$$
d(\mu x) = \mu\left[-\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw\right] + x \cdot \frac{1}{2}\beta(t)\mu\,dt
$$

The drift terms cancel:

$$
d(\mu x) = \mu(t)\sqrt{\beta(t)}\,dw
$$

### Step 4: Integrate

Integrate from $0$ to $t$:

$$
\mu(t)x(t) - \mu(0)x(0) = \int_0^t \mu(s)\sqrt{\beta(s)}\,dw(s)
$$

Since $\mu(0) = 1$:

$$
\mu(t)x(t) = x_0 + \int_0^t \mu(s)\sqrt{\beta(s)}\,dw(s)
$$

### Step 5: Solve for $x(t)$

$$
x(t) = \frac{1}{\mu(t)} x_0 + \frac{1}{\mu(t)} \int_0^t \mu(s)\sqrt{\beta(s)}\,dw(s)
$$

---

## The Emergence of $\alpha(t)$

Define:

$$
\alpha(t) = \frac{1}{\mu(t)} = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right)
$$

**This is where $\alpha(t)$ comes from!** It's the inverse of the integrating factor.

Now the solution becomes:

$$
x(t) = \alpha(t) x_0 + \alpha(t) \int_0^t \mu(s)\sqrt{\beta(s)}\,dw(s)
$$

**Physical meaning**: 

- $\alpha(t)$ is the **signal decay coefficient**
- The term $\alpha(t) x_0$ shows how the original signal scales over time
- As $t$ increases and $\beta(s) > 0$, $\alpha(t)$ decreases toward 0

---

## The Stochastic Integral: Computing the Variance

The stochastic integral $\int_0^t \mu(s)\sqrt{\beta(s)}\,dw(s)$ is Gaussian with:
- Mean: 0
- Variance: $\int_0^t \mu(s)^2 \beta(s)\,ds$ (by Itô isometry)

### Computing the Variance

$$
\text{Var}\left(\int_0^t \mu(s)\sqrt{\beta(s)}\,dw(s)\right) = \int_0^t \mu(s)^2 \beta(s)\,ds
$$

Since $\mu(s) = \exp\left(\frac{1}{2}\int_0^s \beta(u)\,du\right)$:

$$
\mu(s)^2 = \exp\left(\int_0^s \beta(u)\,du\right)
$$

So:

$$
\text{Var} = \int_0^t \exp\left(\int_0^s \beta(u)\,du\right) \beta(s)\,ds
$$

**Trick**: Let $\Phi(s) = \int_0^s \beta(u)\,du$. Then $\frac{d\Phi}{ds} = \beta(s)$:

$$
\text{Var} = \int_0^t e^{\Phi(s)}\,d\Phi(s) = e^{\Phi(t)} - e^{\Phi(0)} = e^{\Phi(t)} - 1
$$

Since $\Phi(t) = \int_0^t \beta(s)\,ds$ and $\mu(t) = e^{\Phi(t)/2}$:

$$
\text{Var} = \mu(t)^2 - 1
$$

### Variance of $x(t)$

The noise term in $x(t)$ is:

$$
\alpha(t) \int_0^t \mu(s)\sqrt{\beta(s)}\,dw(s)
$$

Its variance is:

$$
\alpha(t)^2 \cdot (\mu(t)^2 - 1) = \frac{1}{\mu(t)^2} \cdot (\mu(t)^2 - 1) = 1 - \frac{1}{\mu(t)^2}
$$

Since $\alpha(t) = 1/\mu(t)$:

$$
\text{Variance of noise term} = 1 - \alpha(t)^2
$$

---

## The Emergence of $\bar{\alpha}_t$

Define:

$$
\bar{\alpha}_t = \alpha(t)^2 = \left(\frac{1}{\mu(t)}\right)^2 = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

**This is where $\bar{\alpha}_t$ comes from!** It's the square of the signal coefficient.

### The Final Form

The solution becomes:

$$
x_t = \alpha(t) x_0 + \sqrt{1 - \alpha(t)^2} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

Or equivalently:

$$
\boxed{x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon}
$$

where:

- $\sqrt{\bar{\alpha}_t} = \alpha(t)$ is the signal coefficient
- $\sqrt{1-\bar{\alpha}_t}$ is the noise coefficient

---

## Why These Definitions?

### Not Arbitrary!

The definitions of $\alpha(t)$ and $\bar{\alpha}_t$ are **not chosen arbitrarily**. They emerge naturally from:

1. **The SDE structure**: The VP-SDE $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$
2. **The integrating factor technique**: $\mu(t) = \exp\left(\frac{1}{2}\int_0^t \beta(s)\,ds\right)$
3. **The solution process**: $\alpha(t) = 1/\mu(t)$ and $\bar{\alpha}_t = \alpha(t)^2$

### The Integrating Factor Connection

| Quantity | Definition | Origin |
|----------|-----------|--------|
| $\mu(t)$ | $\exp\left(\frac{1}{2}\int_0^t \beta(s)\,ds\right)$ | Integrating factor for SDE |
| $\alpha(t)$ | $1/\mu(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right)$ | Inverse of integrating factor (signal coefficient) |
| $\bar{\alpha}_t$ | $\alpha(t)^2 = \exp\left(-\int_0^t \beta(s)\,ds\right)$ | Square of signal coefficient |

**Key insight**: The exponential with an integral in the exponent is **exactly** the integrating factor form from ODE/SDE theory.

---

## Alternative: Starting from $\bar{\alpha}_t$

Some papers define $\bar{\alpha}_t$ directly and derive $\beta(t)$ from it.

### Forward Approach (This Document)

$$
\beta(t) \quad \xrightarrow{\text{solve SDE}} \quad \alpha(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right) \quad \xrightarrow{\text{square}} \quad \bar{\alpha}_t = \alpha(t)^2
$$

### Inverse Approach (Also Valid)

$$
\bar{\alpha}_t \quad \xrightarrow{\text{differentiate}} \quad \beta(t) = -\frac{d \log \bar{\alpha}_t}{dt}
$$

**Derivation**: From $\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)$, take the log:

$$
\log \bar{\alpha}_t = -\int_0^t \beta(s)\,ds
$$

Differentiate:

$$
\frac{d \log \bar{\alpha}_t}{dt} = -\beta(t)
$$

So:

$$
\beta(t) = -\frac{d \log \bar{\alpha}_t}{dt}
$$

**Both approaches are equivalent**—you can start with $\beta(t)$ or $\bar{\alpha}_t$.

---

## Summary

| Definition | Formula | Origin |
|------------|---------|--------|
| **Integrating factor** | $\mu(t) = \exp\left(\frac{1}{2}\int_0^t \beta(s)\,ds\right)$ | Standard technique for linear SDEs |
| **Signal coefficient** | $\alpha(t) = 1/\mu(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right)$ | Inverse of integrating factor |
| **Cumulative coefficient** | $\bar{\alpha}_t = \alpha(t)^2 = \exp\left(-\int_0^t \beta(s)\,ds\right)$ | Square of signal coefficient |

**The key point**: These definitions are not ad hoc. They arise naturally from solving the VP-SDE using the integrating factor technique, which is why they have integrals in the exponent.

---

## References

- **Forward Process Derivation**: [`docs/diffusion/forward_process_derivation.md`](./forward_process_derivation.md) — Complete derivation
- **Integrating Factor Technique**: [`docs/diffusion/integrating_factor.md`](./integrating_factor.md) — General method
- **Øksendal (2003)**: "Stochastic Differential Equations" — Chapter 5 on linear SDEs
- **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" — Uses these coefficients throughout

