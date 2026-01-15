# Fokker-Planck Equation: Derivation and Intuition

## Overview

The **Fokker-Planck equation** (also called the **forward Kolmogorov equation**) describes how the probability distribution of a stochastic process evolves over time. It is the bridge between individual particle dynamics (described by SDEs) and collective probability evolution.

This document derives the Fokker-Planck equation from first principles and explains why probability distributions must obey it.

---

## Referenced From

- **Main Document**: [`docs/diffusion/reverse_process/reverse_process_derivation.md`](./reverse_process_derivation.md) — Uses the Fokker-Planck equation to derive the reverse SDE
- **Related**: [`notebooks/diffusion/02_sde_formulation/supplements/07_fokker_planck_equation.md`](../../../notebooks/diffusion/02_sde_formulation/supplements/07_fokker_planck_equation.md)

---

## Table of Contents

1. [The Setting](#the-setting)
2. [Intuitive Picture](#intuitive-picture)
3. [Derivation via Infinitesimal Evolution](#derivation-via-infinitesimal-evolution)
4. [Kramers-Moyal Expansion (Rigorous)](#kramers-moyal-expansion-rigorous)
5. [Physical Interpretation](#physical-interpretation)
6. [Connection to Conservation Laws](#connection-to-conservation-laws)
7. [Example: Simple Diffusion](#example-simple-diffusion)
8. [Why This Matters for Reverse SDEs](#why-this-matters-for-reverse-sdes)

---

## The Setting

### Stochastic Differential Equation

Consider a $d$-dimensional stochastic process $x(t) \in \mathbb{R}^d$ governed by:

$$
dx = f(x,t)\,dt + g(t)\,dw
$$

where:

- $f(x,t) \in \mathbb{R}^d$ is the **drift** (deterministic force)
- $g(t) \in \mathbb{R}$ is the **diffusion coefficient** (noise amplitude)
- $w(t) \in \mathbb{R}^d$ is standard Brownian motion

### Probability Distribution

At each time $t$, the random variable $x(t)$ has a probability distribution:

$$
p_t(x) = p(x, t)
$$

where $\int p_t(x)\,dx = 1$.

### The Question

**How does $p_t(x)$ evolve over time?**

The Fokker-Planck equation answers this:

$$
\boxed{\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t}
$$

---

## Intuitive Picture

Before diving into the math, let's build intuition.

### What Changes Probability at a Point?

Imagine a point $x$ in space. The probability density $p_t(x)$ at this point can change due to:

1. **Drift (advection)**: Particles drift from nearby points to $x$, or from $x$ to elsewhere
2. **Diffusion (spreading)**: Particles randomly wander in and out of $x$

These two mechanisms give rise to the two terms in the Fokker-Planck equation.

### Analogy: Heat Equation

You can think of probability like heat:
- **Drift term**: Like a wind blowing heat from one place to another
- **Diffusion term**: Like heat spreading from hot to cold regions

The Fokker-Planck equation is essentially a heat equation with an additional drift/advection term.

---

## Derivation via Infinitesimal Evolution

We'll derive the equation by considering how probability evolves over an infinitesimal time step $\Delta t$.

### Step 1: Chapman-Kolmogorov Equation

The probability at time $t + \Delta t$ is related to the probability at time $t$ by:

$$
p_{t+\Delta t}(x) = \int p_t(x') \, p(x, t+\Delta t \mid x', t) \, dx'
$$

where $p(x, t+\Delta t \mid x', t)$ is the **transition probability** from $x'$ at time $t$ to $x$ at time $t + \Delta t$.

**Interpretation**: To find the probability of being at $x$ at time $t + \Delta t$, we sum over all possible starting points $x'$ at time $t$, weighted by their probability.

### Step 2: Taylor Expand the Left Side

$$
p_{t+\Delta t}(x) = p_t(x) + \frac{\partial p_t}{\partial t} \Delta t + O(\Delta t^2)
$$

### Step 3: Understand the Transition Probability

For the SDE $dx = f(x,t)\,dt + g(t)\,dw$, the change over $\Delta t$ is:

$$
\Delta x = x(t + \Delta t) - x(t) = f(x,t) \Delta t + g(t) \sqrt{\Delta t} \, \xi
$$

where $\xi \sim \mathcal{N}(0, I)$ is a standard normal random variable.

This means:
- **Mean displacement**: $\mathbb{E}[\Delta x \mid x] = f(x,t) \Delta t$
- **Covariance**: $\text{Cov}[\Delta x \mid x] = g(t)^2 \Delta t \, I$

### Step 4: Express Transition Probability

For small $\Delta t$, the transition from $x'$ to $x$ is approximately:

$$
p(x \mid x', \Delta t) \approx \delta(x - x' - f(x',t)\Delta t) * \mathcal{N}(0, g(t)^2 \Delta t \, I)
$$

where $*$ denotes convolution.

More precisely, using Gaussian approximation:

$$
p(x \mid x', \Delta t) \approx \frac{1}{(2\pi g^2 \Delta t)^{d/2}} \exp\left(-\frac{|x - x' - f(x',t)\Delta t|^2}{2g^2 \Delta t}\right)
$$

### Step 5: Substitute into Chapman-Kolmogorov

$$
p_{t+\Delta t}(x) = \int p_t(x') \, p(x \mid x', \Delta t) \, dx'
$$

Now we use a clever trick: expand around $x' = x$ (nearby points contribute most).

Let $\delta x = x - x'$, so $x' = x - \delta x$:

$$
p_{t+\Delta t}(x) = \int p_t(x - \delta x) \, p(x \mid x - \delta x, \Delta t) \, d(\delta x)
$$

### Step 6: Taylor Expand $p_t(x - \delta x)$

$$
p_t(x - \delta x) = p_t(x) - \delta x \cdot \nabla p_t(x) + \frac{1}{2} \sum_{i,j} \delta x_i \delta x_j \frac{\partial^2 p_t}{\partial x_i \partial x_j} + \ldots
$$

Using index notation:

$$
p_t(x - \delta x) \approx p_t(x) - \sum_i \delta x_i \frac{\partial p_t}{\partial x_i} + \frac{1}{2} \sum_{i,j} \delta x_i \delta x_j \frac{\partial^2 p_t}{\partial x_i \partial x_j}
$$

### Step 7: Compute Moments of $\delta x$

From the transition probability, conditioned on starting at $x' = x - \delta x \approx x$:

**First moment**:

$$

\mathbb{E}[\delta x] = f(x,t) \Delta t + O(\Delta t^2)
$$

**Second moment** (each component):

$$

\mathbb{E}[\delta x_i \delta x_j] = \begin{cases}
g(t)^2 \Delta t + f_i f_j \Delta t^2 & \text{if } i = j \\
f_i f_j \Delta t^2 & \text{if } i \neq j
\end{cases}
$$

For small $\Delta t$, the $\Delta t^2$ terms are negligible, so:

$$
\mathbb{E}[\delta x_i \delta x_j] \approx g(t)^2 \Delta t \, \delta_{ij}
$$

where $\delta_{ij}$ is the Kronecker delta.

### Step 8: Substitute Moments

$$
p_{t+\Delta t}(x) = \int \left[p_t(x) - \sum_i \delta x_i \frac{\partial p_t}{\partial x_i} + \frac{1}{2} \sum_{i,j} \delta x_i \delta x_j \frac{\partial^2 p_t}{\partial x_i \partial x_j}\right] p(\delta x \mid x, \Delta t) \, d(\delta x)
$$

Taking expectations over $\delta x$:

$$
p_{t+\Delta t}(x) = p_t(x) - \sum_i \mathbb{E}[\delta x_i] \frac{\partial p_t}{\partial x_i} + \frac{1}{2} \sum_{i,j} \mathbb{E}[\delta x_i \delta x_j] \frac{\partial^2 p_t}{\partial x_i \partial x_j}
$$

Substitute the moments:

$$
p_{t+\Delta t}(x) = p_t(x) - \sum_i f_i(x,t) \Delta t \frac{\partial p_t}{\partial x_i} + \frac{1}{2} \sum_i g(t)^2 \Delta t \frac{\partial^2 p_t}{\partial x_i^2}
$$

### Step 9: Rearrange and Take Limit

$$
p_{t+\Delta t}(x) - p_t(x) = -\sum_i f_i(x,t) \frac{\partial p_t}{\partial x_i} \Delta t + \frac{1}{2} g(t)^2 \sum_i \frac{\partial^2 p_t}{\partial x_i^2} \Delta t
$$

Divide by $\Delta t$ and take $\Delta t \to 0$:

$$
\frac{\partial p_t}{\partial t} = -\sum_i \frac{\partial}{\partial x_i} \left[f_i(x,t) p_t(x)\right] + \frac{1}{2} g(t)^2 \sum_i \frac{\partial^2 p_t}{\partial x_i^2}
$$

### Step 10: Vector Notation

Using $\nabla \cdot$ for divergence and $\nabla^2$ for Laplacian:

$$
\boxed{\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t}
$$

**This is the Fokker-Planck equation!**

---

## Kramers-Moyal Expansion (Rigorous)

The derivation above can be made rigorous using the **Kramers-Moyal expansion**.

### Jump Moments

Define the **jump moments**:

$$
M^{(n)}_i(x, t) = \lim_{\Delta t \to 0} \frac{1}{\Delta t} \mathbb{E}\left[(\Delta x_i)^n \mid x(t) = x\right]
$$

### Kramers-Moyal Theorem

The evolution of the probability distribution is:

$$
\frac{\partial p_t}{\partial t} = \sum_{n=1}^\infty \frac{(-1)^n}{n!} \sum_{i_1, \ldots, i_n} \frac{\partial^n}{\partial x_{i_1} \cdots \partial x_{i_n}} \left[M^{(n)}_{i_1 \cdots i_n} p_t\right]
$$

### Fokker-Planck Approximation

For **continuous diffusion processes** (like those generated by SDEs with smooth coefficients), the Kramers-Moyal expansion terminates at $n=2$:

$$
M^{(n)} = 0 \quad \text{for } n \geq 3
$$

This gives:

$$
\frac{\partial p_t}{\partial t} = -\sum_i \frac{\partial}{\partial x_i} [M^{(1)}_i p_t] + \frac{1}{2} \sum_{i,j} \frac{\partial^2}{\partial x_i \partial x_j} [M^{(2)}_{ij} p_t]
$$

### For Our SDE

From $dx = f(x,t)\,dt + g(t)\,dw$:

$$
M^{(1)}_i = f_i(x,t)
$$

$$
M^{(2)}_{ij} = g(t)^2 \delta_{ij}
$$

Substituting:

$$
\frac{\partial p_t}{\partial t} = -\sum_i \frac{\partial}{\partial x_i} [f_i p_t] + \frac{1}{2} g(t)^2 \sum_i \frac{\partial^2 p_t}{\partial x_i^2}
$$

This is the Fokker-Planck equation.

---

## Physical Interpretation

Let's break down each term:

### The Full Equation

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t
$$

### Term 1: Rate of Change

$$
\frac{\partial p_t}{\partial t}
$$

**Meaning**: How fast probability density is changing at point $x$ at time $t$.

### Term 2: Drift (Advection)

$$
-\nabla \cdot (f p_t) = -\sum_i \frac{\partial}{\partial x_i} [f_i(x,t) p_t(x)]
$$

**Meaning**: Probability flux due to deterministic drift.

**Expanded form**:

$$
-\nabla \cdot (f p_t) = -f \cdot \nabla p_t - p_t \nabla \cdot f
$$

- $-f \cdot \nabla p_t$: Advection of probability along the drift field
- $-p_t \nabla \cdot f$: Change in probability due to compression/expansion of the drift field

**Physical picture**: Probability "flows" along the drift field $f(x,t)$, like wind blowing particles.

### Term 3: Diffusion (Spreading)

$$
\frac{1}{2}g(t)^2 \nabla^2 p_t = \frac{1}{2}g(t)^2 \sum_i \frac{\partial^2 p_t}{\partial x_i^2}
$$

**Meaning**: Probability spreads from high-density to low-density regions.

**Sign convention**:

- $\nabla^2 p_t > 0$: Concave up → probability flows *in* (increases)
- $\nabla^2 p_t < 0$: Concave down → probability flows *out* (decreases)

**Physical picture**: Random motion causes probability to diffuse, like heat spreading in a metal rod.

---

## Connection to Conservation Laws

### Continuity Equation

The Fokker-Planck equation can be written as a **continuity equation**:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot J = 0
$$

where $J$ is the **probability current** (flux):

$$
J = f p_t - \frac{1}{2}g(t)^2 \nabla p_t
$$

**Components**:

- $f p_t$: Drift current (probability flowing along drift)
- $-\frac{1}{2}g(t)^2 \nabla p_t$: Diffusion current (Fick's law, probability flowing down gradients)

### Conservation of Probability

Integrating over all space:

$$
\frac{d}{dt} \int p_t(x)\,dx = \int \frac{\partial p_t}{\partial t}\,dx = -\int \nabla \cdot J\,dx = 0
$$

(using divergence theorem, assuming $J \to 0$ at infinity).

**Result**: Total probability is conserved, as it must be!

---

## Example: Simple Diffusion

### Setup

Consider pure diffusion with no drift:

$$
dx = \sigma \, dw
$$

where $\sigma$ is constant.

### Fokker-Planck Equation

$$
\frac{\partial p_t}{\partial t} = \frac{\sigma^2}{2} \nabla^2 p_t
$$

This is the **heat equation**!

### Solution

Starting from a point mass $p_0(x) = \delta(x - x_0)$, the solution is:

$$
p_t(x) = \frac{1}{(2\pi \sigma^2 t)^{d/2}} \exp\left(-\frac{|x - x_0|^2}{2\sigma^2 t}\right)
$$

This is a Gaussian with:
- Mean: $x_0$ (stays at starting point)
- Variance: $\sigma^2 t$ (spreads linearly with time)

**Verification**: Substitute this solution into the heat equation — it works!

---

## Example: Ornstein-Uhlenbeck Process

### Setup

Consider the SDE:

$$
dx = -\theta x \, dt + \sigma \, dw
$$

where:

- $-\theta x$: Drift toward origin (like a spring)
- $\sigma$: Constant diffusion

### Fokker-Planck Equation

$$
\frac{\partial p_t}{\partial t} = \nabla \cdot (\theta x p_t) + \frac{\sigma^2}{2} \nabla^2 p_t
$$

### Stationary Distribution

At equilibrium ($\partial p / \partial t = 0$):

$$
0 = \nabla \cdot (\theta x p) + \frac{\sigma^2}{2} \nabla^2 p
$$

Solving this ODE:

$$
p_\infty(x) = \mathcal{N}\left(0, \frac{\sigma^2}{2\theta}\right)
$$

**Interpretation**: The drift pulls particles toward the origin, while diffusion spreads them out. The equilibrium is a balance between these forces.

---

## Why This Matters for Reverse SDEs

### Forward Process

The forward SDE:

$$
dx = f(x,t)\,dt + g(t)\,dw
$$

generates a probability evolution governed by:

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t
$$

### Reverse Process

To reverse the process, we need to find an SDE whose Fokker-Planck equation gives **backward** evolution:

$$
\frac{\partial p_t}{\partial t} = +\nabla \cdot (f p_t) - \frac{1}{2}g(t)^2 \nabla^2 p_t
$$

(note the sign flips).

### Key Insight

The Fokker-Planck equation can be rewritten using the **score function** $\nabla \log p_t$:

$$
\nabla^2 p_t = \nabla \cdot (\nabla p_t) = \nabla \cdot (p_t \nabla \log p_t)
$$

This allows us to express the diffusion term in terms of the score, leading to the **effective drift**:

$$
\tilde{f} = f - \frac{1}{2}g^2 \nabla \log p_t
$$

The reverse SDE uses this effective drift to reverse the probability evolution.

**See**: [`reverse_process_derivation.md`](./reverse_process_derivation.md) for the full derivation.

---

## Summary

### What We Learned

1. **Fokker-Planck Equation**: Describes how probability distributions evolve for stochastic processes
2. **Two Mechanisms**: Drift (advection) and diffusion (spreading)
3. **Derivation**: From infinitesimal evolution using Chapman-Kolmogorov and Taylor expansion
4. **Physical Meaning**: Continuity equation for probability flux
5. **Connection to Reverse SDEs**: The score function appears when rewriting the diffusion term

### The Equation

$$
\boxed{\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t}
$$

**Drift term**: $-\nabla \cdot (f p_t)$ — Probability flows along $f$

**Diffusion term**: $\frac{1}{2}g^2 \nabla^2 p_t$ — Probability spreads

---

## References

### Classic Texts
- **Risken (1989)**: "The Fokker-Planck Equation: Methods of Solution and Applications"
- **Gardiner (2009)**: "Stochastic Methods: A Handbook for the Natural and Social Sciences"
- **Øksendal (2003)**: "Stochastic Differential Equations: An Introduction with Applications"

### Papers
- **Fokker (1914)**: "Die mittlere Energie rotierender elektrischer Dipole im Strahlungsfeld" — Original work
- **Planck (1917)**: "Über einen Satz der statistischen Dynamik und seine Erweiterung in der Quantentheorie"
- **Kolmogorov (1931)**: "Über die analytischen Methoden in der Wahrscheinlichkeitsrechnung" — Forward equation

### Related Documents
- **Reverse Process**: [`reverse_process_derivation.md`](./reverse_process_derivation.md)
- **Forward Process**: [`forward_process_derivation.md`](../forward_process_derivation.md)
- **Supplement in Notebook**: [`notebooks/diffusion/02_sde_formulation/supplements/07_fokker_planck_equation.md`](../../../notebooks/diffusion/02_sde_formulation/supplements/07_fokker_planck_equation.md)

