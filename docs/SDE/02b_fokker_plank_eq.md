# Deriving the Fokker–Planck Equation
=====================================

This document derives the **Fokker–Planck equation** (also known as the **Kolmogorov forward equation**) line by line from an SDE using Taylor expansion. This is one of the cleanest places to see Taylor expansion doing real mathematical work in diffusion theory.

We'll derive it for the general Itô SDE with isotropic noise, then note the vector/matrix generalization.

## Overview
------------

We'll derive the PDE that governs how probability densities evolve under stochastic dynamics. This is the continuous-time analog of the Chapman-Kolmogorov equation for Markov chains.

## Notation and Goal
--------------------

### Variables

| Symbol | Meaning |
|--------|----------|
| $x(t) \in \mathbb{R}^d$ | Random state at time $t$ |
| $p(x, t)$ | Probability density of $x(t)$ |
| $f(x, t) \in \mathbb{R}^d$ | Drift function |
| $g(t) \in \mathbb{R}$ | Diffusion strength (scalar for simplicity) |
| $w(t) \in \mathbb{R}^d$ | Brownian motion (independent components) |
| $\Delta w := w(t+\Delta t) - w(t)$ | Brownian increment, $\sim \mathcal{N}(0, \Delta t I)$ |

### Goal

Show that $p(x, t)$ satisfies:

$$
\boxed{\frac{\partial p}{\partial t}(x, t) = -\nabla \cdot \left(f(x, t) p(x, t)\right) + \frac{1}{2}g(t)^2 \Delta p(x, t)}
$$

where:

- $\nabla \cdot$ is the divergence operator
- $\Delta = \nabla^2 = \sum_{i=1}^d \frac{\partial^2}{\partial x_i^2}$ is the Laplacian

## Step 1: Use a Test Function
---------------------------

Instead of manipulating densities directly (which gets messy), we use a smooth **test function** $\varphi: \mathbb{R}^d \to \mathbb{R}$ (think of it as a smooth probe).

### What is a Test Function?

A **test function** is a smooth function used to "probe" or "measure" properties of distributions and generalized functions. Think of it as a mathematical instrument that lets us study objects that are too rough to handle directly.

**Formal definition**: $\varphi \in C_c^\infty(\mathbb{R}^d)$ means:
- $\varphi$ is infinitely differentiable (smooth)
- $\varphi$ has compact support (zero outside some bounded region)

**Practical requirements** (we can relax compact support):
- $\varphi$ is at least twice continuously differentiable
- $\varphi$ and its derivatives decay rapidly at infinity
- Think: $\varphi(x) = e^{-\|x\|^2}$ or any smooth bump function

### Why Use Test Functions?

**1. Avoid Direct Density Manipulation**

Probability densities $p(x, t)$ can be:
- Discontinuous (e.g., uniform distributions)
- Non-differentiable (e.g., at boundaries)
- Singular (e.g., delta functions)

Working with them directly in differential equations is mathematically treacherous.

**2. Weak Formulation**

Test functions allow us to work with **weak solutions**—solutions that satisfy the PDE "on average" rather than pointwise. This is the standard approach in modern PDE theory.

**3. Integration by Parts**

Test functions have nice decay properties, so boundary terms vanish when integrating by parts. This lets us transfer derivatives freely.

**4. Distribution Theory Connection**

This is how Laurent Schwartz formalized distributions (generalized functions). A distribution is defined by how it acts on test functions.

### The Test Function Strategy

**Expected Value**:

$$
\mathbb{E}[\varphi(x(t))] = \int_{\mathbb{R}^d} \varphi(x) p(x, t)\,dx
$$

**Strategy**: 
1. Compute how $\mathbb{E}[\varphi(x(t))]$ changes with time
2. Express this in terms of $p(x, t)$
3. Use integration by parts to transfer derivatives from $\varphi$ to $p$
4. Conclude that $p$ satisfies the Fokker-Planck equation

**Key insight**: If we know how $\mathbb{E}[\varphi(x(t))]$ evolves for *all* test functions $\varphi$, we know everything about $p(x, t)$.

## Step 2: SDE Increment
----------------------

For the SDE:

$$
dx(t) = f(x(t), t)\,dt + g(t)\,dw(t)
$$

Over a small interval $[t, t+\Delta t]$, the increment is:

$$
\Delta x := x(t+\Delta t) - x(t) = f(x(t), t)\,\Delta t + g(t)\,\Delta w
$$

### Key Scaling Facts

- $\Delta t$ is small
- $\Delta w \sim \mathcal{N}(0, \Delta t I)$, so typical size $|\Delta w| \sim \sqrt{\Delta t}$

## Step 3: Taylor Expand the Test Function
--------------------------------------

We write:

$$
\varphi(x(t+\Delta t)) = \varphi(x(t) + \Delta x)
$$

Now perform a **second-order Taylor expansion** around $x(t)$:

$$
\varphi(x + \Delta x) = \varphi(x) + \nabla\varphi(x)^\top \Delta x + \frac{1}{2}\Delta x^\top H_\varphi(x) \Delta x + \text{higher-order terms}
$$

where:

- $\nabla\varphi$ is the gradient (a $d$-vector)
- $H_\varphi$ is the Hessian matrix of second derivatives

### Why Second Order?

Because $\Delta x$ has a $\sqrt{\Delta t}$ piece (from the noise), squaring it produces order-$\Delta t$ terms that **survive** in the limit.

## Step 4: Take Conditional Expectation
--------------------------------------

Condition on the current state $x(t) = x$:

$$
\mathbb{E}[\varphi(x(t+\Delta t)) \mid x(t)=x] \approx \varphi(x) + \nabla\varphi(x)^\top \mathbb{E}[\Delta x \mid x] + \frac{1}{2}\mathbb{E}[\Delta x^\top H_\varphi(x) \Delta x \mid x]
$$

Now we compute the needed moments.

### First Moment of $\Delta x$

$$
\mathbb{E}[\Delta x \mid x] = f(x, t)\,\Delta t + g(t)\,\mathbb{E}[\Delta w] = f(x, t)\,\Delta t
$$

since $\mathbb{E}[\Delta w] = 0$.

**First-order contribution**:

$$
\nabla\varphi(x)^\top f(x, t)\,\Delta t
$$

### Second Moment Term

Substitute $\Delta x = f\,\Delta t + g\,\Delta w$:

$$
\Delta x^\top H \Delta x = (f\,\Delta t + g\,\Delta w)^\top H (f\,\Delta t + g\,\Delta w)
$$

Expanding:

$$
= (f\,\Delta t)^\top H (f\,\Delta t) + 2(f\,\Delta t)^\top H (g\,\Delta w) + (g\,\Delta w)^\top H (g\,\Delta w)
$$

**Take conditional expectation**:

| Term | Order | Survives? |
|------|-------|----------|
| $(f\,\Delta t)^\top H (f\,\Delta t)$ | $O(\Delta t^2)$ | No |
| $2(f\,\Delta t)^\top H (g\,\Delta w)$ | $\mathbb{E}[\Delta w] = 0$ | No |
| $(g\,\Delta w)^\top H (g\,\Delta w)$ | $O(\Delta t)$ | **Yes** |

**For the surviving term**:

$$
\mathbb{E}[(g\,\Delta w)^\top H (g\,\Delta w) \mid x] = g(t)^2\,\mathbb{E}[\Delta w^\top H \Delta w]
$$

**Key identity for Gaussian increments**:

$$
\mathbb{E}[\Delta w \Delta w^\top] = \Delta t\,I
$$

For any symmetric matrix $H$:

$$
\mathbb{E}[\Delta w^\top H \Delta w] = \text{tr}(H\,\mathbb{E}[\Delta w \Delta w^\top]) = \text{tr}(H\,\Delta t I) = \Delta t\,\text{tr}(H)
$$

**Therefore**:

$$
\mathbb{E}[\Delta x^\top H \Delta x \mid x] = g(t)^2\,\Delta t\,\text{tr}(H_\varphi(x))
$$

But $\text{tr}(H_\varphi) = \sum_i \frac{\partial^2 \varphi}{\partial x_i^2} = \Delta \varphi$ (the Laplacian).

**Second-order contribution**:

$$
\frac{1}{2}g(t)^2\,\Delta t\,\Delta\varphi(x)
$$

## Step 5: The Infinitesimal Generator
--------------------------------------

Combining both contributions:

$$
\mathbb{E}[\varphi(x(t+\Delta t)) \mid x(t)=x] = \varphi(x) + \Delta t\left(f(x, t) \cdot \nabla\varphi(x) + \frac{1}{2}g(t)^2 \Delta\varphi(x)\right) + o(\Delta t)
$$

Subtract $\varphi(x)$, divide by $\Delta t$, and take $\Delta t \to 0$:

$$
\frac{d}{dt}\mathbb{E}[\varphi(x(t)) \mid x(t)=x] = f(x, t) \cdot \nabla\varphi(x) + \frac{1}{2}g(t)^2 \Delta\varphi(x)
$$

### Define the Generator

The **infinitesimal generator** $\mathcal{L}$ is:

$$
(\mathcal{L}\varphi)(x, t) := f(x, t) \cdot \nabla\varphi(x) + \frac{1}{2}g(t)^2 \Delta\varphi(x)
$$

Then:

$$
\frac{d}{dt}\mathbb{E}[\varphi(x(t))] = \mathbb{E}[(\mathcal{L}\varphi)(x(t), t)]
$$

### Understanding the Infinitesimal Generator

**What is it?**

The generator $\mathcal{L}$ is an operator that acts on functions, telling us how their expected values change when evaluated along the stochastic process.

**Interpretation**:

$$
\mathbb{E}[\varphi(x(t+dt))] \approx \mathbb{E}[\varphi(x(t))] + dt\,\mathbb{E}[(\mathcal{L}\varphi)(x(t))]
$$

So $\mathcal{L}\varphi$ is the "instantaneous rate of change" of $\mathbb{E}[\varphi(x(t))]$.

**Components**:

1. **Drift term** $f \cdot \nabla\varphi$: How $\varphi$ changes along the flow
   - If $f$ points toward regions where $\varphi$ is larger, this is positive
   - Pure directional derivative along the drift

2. **Diffusion term** $\frac{1}{2}g^2 \Delta\varphi$: How $\varphi$ changes due to spreading
   - Laplacian measures local curvature
   - Positive $\Delta\varphi$ means $\varphi$ is locally concave (bowl-shaped)
   - Diffusion tends to move probability toward regions of positive curvature

**Example**: Consider $\varphi(x) = x^2$ (measuring second moment) with drift $f = -\beta x$ (OU process) and constant diffusion $g$:

$$
\mathcal{L}(x^2) = -\beta x \cdot 2x + \frac{1}{2}g^2 \cdot 2 = -2\beta x^2 + g^2
$$

So:

$$
\frac{d}{dt}\mathbb{E}[x(t)^2] = -2\beta\mathbb{E}[x^2] + g^2
$$

This shows variance decays toward equilibrium $\mathbb{E}[x^2] = g^2/(2\beta)$.

## Step 6: Transfer to the Density
---------------------------------

Recall:

$$
\mathbb{E}[\varphi(x(t))] = \int \varphi(x) p(x, t)\,dx
$$

Differentiate in time:

$$
\frac{d}{dt}\mathbb{E}[\varphi(x(t))] = \int \varphi(x)\,\frac{\partial p}{\partial t}(x, t)\,dx
$$

But we also have:

$$
\frac{d}{dt}\mathbb{E}[\varphi(x(t))] = \int (\mathcal{L}\varphi)(x, t)\,p(x, t)\,dx
$$

Therefore, for all smooth $\varphi$:

$$
\int \varphi(x)\,\partial_t p(x, t)\,dx = \int \left(f \cdot \nabla\varphi + \frac{1}{2}g^2 \Delta\varphi\right) p\,dx
$$

### Integration by Parts

Now we move derivatives from $\varphi$ to $p$. This is the crucial step where test functions earn their keep.

#### Drift Term

Start with:

$$
\int (f \cdot \nabla\varphi)\,p\,dx = \int \sum_i f_i(x, t)\,\frac{\partial \varphi}{\partial x_i}\,p(x, t)\,dx
$$

Integrate by parts (product rule for derivatives):

$$
\int f_i\,\frac{\partial \varphi}{\partial x_i}\,p\,dx = \underbrace{\left[f_i\,\varphi\,p\right]_{-\infty}^{+\infty}}_{\text{boundary term} = 0} - \int \varphi\,\frac{\partial}{\partial x_i}(f_i\,p)\,dx
$$

**Why boundary term vanishes**:
- Test function $\varphi$ has compact support (or decays rapidly)
- Probability density $p$ decays at infinity
- Their product $\to 0$ as $|x| \to \infty$

Summing over all components:

$$
\int (f \cdot \nabla\varphi)\,p\,dx = -\int \varphi\,\nabla \cdot (fp)\,dx
$$

where $\nabla \cdot (fp) = \sum_i \frac{\partial}{\partial x_i}(f_i p)$ is the divergence.

#### Diffusion Term

The Laplacian $\Delta\varphi = \sum_i \frac{\partial^2 \varphi}{\partial x_i^2}$.

Integrate by parts **twice** (once for each derivative):

**First integration**:

$$
\int \frac{\partial^2 \varphi}{\partial x_i^2}\,p\,dx = \underbrace{\left[\frac{\partial \varphi}{\partial x_i}\,p\right]_{-\infty}^{+\infty}}_{=0} - \int \frac{\partial \varphi}{\partial x_i}\,\frac{\partial p}{\partial x_i}\,dx
$$

**Second integration**:

$$
-\int \frac{\partial \varphi}{\partial x_i}\,\frac{\partial p}{\partial x_i}\,dx = -\underbrace{\left[\varphi\,\frac{\partial p}{\partial x_i}\right]_{-\infty}^{+\infty}}_{=0} + \int \varphi\,\frac{\partial^2 p}{\partial x_i^2}\,dx
$$

Summing over all components:

$$
\int (\Delta\varphi)\,p\,dx = \int \varphi\,(\Delta p)\,dx
$$

**Key point**: We've transferred two derivatives from $\varphi$ to $p$.

### The Result

Combining both terms:

$$
\int \varphi\,\partial_t p\,dx = \int \varphi\left(-\nabla \cdot (fp) + \frac{1}{2}g^2 \Delta p\right)dx
$$

Rearranging:

$$
\int \varphi(x)\left[\partial_t p(x, t) + \nabla \cdot (fp) - \frac{1}{2}g^2 \Delta p\right]dx = 0
$$

### The Fundamental Lemma (du Bois-Reymond)

**Statement**: If $\int \varphi(x)\,h(x)\,dx = 0$ for **all** smooth test functions $\varphi$, then $h(x) = 0$ almost everywhere.

**Proof sketch**: 
- Suppose $h(x_0) > 0$ at some point $x_0$
- By continuity, $h > 0$ in some neighborhood $U$ of $x_0$
- Choose $\varphi$ to be a smooth bump function: $\varphi > 0$ on $U$, $\varphi = 0$ outside $U$
- Then $\int \varphi\,h\,dx > 0$, contradiction!

**Application**: Since our equation holds for all $\varphi$, we conclude:

$$
\boxed{\frac{\partial p}{\partial t}(x, t) = -\nabla \cdot \left(f(x, t) p(x, t)\right) + \frac{1}{2}g(t)^2 \Delta p(x, t)}
$$

**This is the Fokker–Planck equation.**

### What Does This Mean?

The Fokker-Planck equation says that the probability density evolves according to:

1. **Transport term** $-\nabla \cdot (fp)$: Probability flows along the drift $f$
2. **Diffusion term** $\frac{1}{2}g^2 \Delta p$: Probability spreads out due to randomness

**Physical interpretation**:
- If you have a swarm of particles following the SDE
- The density $p(x, t)$ describes their distribution
- The drift $f$ creates a flow (like a river current)
- The diffusion $g^2 \Delta p$ creates spreading (like heat diffusion)

## Generalization: Matrix-Valued Diffusion
-----------------------------------------

For a more general SDE:

$$
dx = f(x, t)\,dt + G(x, t)\,dw
$$

with $G \in \mathbb{R}^{d \times d}$, define the **diffusion matrix**:

$$
D(x, t) = G(x, t) G(x, t)^\top
$$

Then the Fokker–Planck equation becomes:

$$
\frac{\partial p}{\partial t} = -\nabla \cdot (fp) + \frac{1}{2}\sum_{i,j} \frac{\partial^2}{\partial x_i \partial x_j}\left(D_{ij}(x, t)\,p\right)
$$

**In diffusion models**: We often choose isotropic noise so $D \propto I$, which collapses back to the Laplacian form.

## Weak vs. Strong Solutions (Advanced Note)
------------------------------------------

### Strong Solutions

A **strong solution** to the Fokker-Planck equation is a function $p(x, t)$ that:
- Is differentiable in both $x$ and $t$
- Satisfies the equation pointwise everywhere

**Problem**: Many important probability densities are not this smooth (e.g., boundaries, discontinuities).

### Weak Solutions

A **weak solution** satisfies:

$$
\int \varphi\,\partial_t p\,dx = \int \left(-\varphi\,\nabla \cdot (fp) + \varphi\,\frac{1}{2}g^2 \Delta p\right)dx
$$

for all test functions $\varphi$.

**Advantages**:
- Exists even when $p$ is not differentiable
- Includes generalized functions (distributions)
- More physically relevant (measurements are always averaged)

**Our derivation**: We worked in the weak formulation from the start, which is why we never needed $p$ to be differentiable!

### Physical Interpretation

Weak solutions capture the idea that:
- We never measure a probability density at a point
- We always measure averages over regions: $\int_A p\,dx$
- Test functions represent these measurements

This is how nature actually works—you can't measure an infinitesimal probability, only finite probabilities over finite regions.

## Connection to Diffusion Models
---------------------------------

In diffusion models:

- The **forward SDE** defines how $p_t(x)$ evolves toward noise via the Fokker–Planck equation
- The **reverse-time SDE** introduces the **score** $\nabla_x \log p_t(x)$
- The score is the gradient of the log of the very density governed by Fokker–Planck
- The **score network** learns the "missing information" needed to run the probability flow backward

This creates a beautiful closed loop:

```
Forward SDE → Fokker–Planck → Density p_t(x) → Score ∇log p_t → Reverse SDE
```

### Why This Matters for Score-Based Models

The Fokker-Planck equation guarantees that:
1. The forward diffusion process has a well-defined probability flow
2. The density $p_t(x)$ evolves smoothly (in the weak sense)
3. The score $\nabla_x \log p_t(x)$ exists (under mild conditions)
4. The reverse process can be constructed using this score

**Key insight**: Score-based diffusion models work because the Fokker-Planck equation ensures the forward process creates a smooth probability path that can be reversed.

## Examples and Intuition
------------------------

### Example 1: Pure Diffusion (Brownian Motion)

For $dx = dw$ (no drift, $g=1$):

$$
\frac{\partial p}{\partial t} = \frac{1}{2}\Delta p
$$

This is the **heat equation**! Probability diffuses like heat, spreading out over time.

**Solution**: If $p(x, 0) = \delta(x)$ (point mass at origin), then:

$$
p(x, t) = \frac{1}{\sqrt{2\pi t}}e^{-x^2/(2t)}
$$

The variance grows linearly: $\text{Var}(x(t)) = t$.

### Example 2: Ornstein-Uhlenbeck Process

For $dx = -\beta x\,dt + \sigma\,dw$ (mean-reverting):

$$
\frac{\partial p}{\partial t} = \beta\,\nabla \cdot (xp) + \frac{1}{2}\sigma^2 \Delta p
$$

Expanding the divergence:

$$
\frac{\partial p}{\partial t} = \beta\,\nabla \cdot (xp) + \frac{1}{2}\sigma^2 \Delta p = \beta(p + x \cdot \nabla p) + \frac{1}{2}\sigma^2 \Delta p
$$

**Equilibrium**: Set $\partial_t p = 0$, solve to get:

$$
p_{\infty}(x) = \mathcal{N}\left(0, \frac{\sigma^2}{2\beta}\right)
$$

The system equilibrates to a Gaussian!

### Example 3: Forward Diffusion in Generative Models

For the VP-SDE: $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$:

$$
\frac{\partial p_t}{\partial t} = \frac{1}{2}\beta(t)\,\nabla \cdot (xp_t) + \frac{1}{2}\beta(t)\,\Delta p_t
$$

This drives any initial $p_0(x)$ toward a Gaussian as $t \to T$.

### Intuition: Conservation of Probability

The Fokker-Planck equation is a **conservation law**. Rewrite it as:

$$
\frac{\partial p}{\partial t} + \nabla \cdot J = 0
$$

where the **probability current** is:

$$
J = fp - \frac{1}{2}g^2 \nabla p
$$

**Interpretation**:
- $fp$: Probability flows along the drift
- $-\frac{1}{2}g^2 \nabla p$: Probability flows down gradients (from high to low density)

Total probability is conserved: $\frac{d}{dt}\int p\,dx = 0$.

## Summary
----------

We derived the Fokker–Planck equation through:

1. **Test function approach** (cleaner than direct density manipulation)
   - Avoids issues with non-smooth densities
   - Leads to weak solutions naturally
   - Based on fundamental distribution theory

2. **Second-order Taylor expansion** (captures both drift and diffusion)
   - First order → drift term
   - Second order → diffusion term
   - Higher orders vanish in the limit $\Delta t \to 0$

3. **Moment matching** (first moment → drift, second moment → diffusion)
   - $\mathbb{E}[\Delta x] \sim O(\Delta t)$ → drift
   - $\mathbb{E}[\Delta x^2] \sim O(\Delta t)$ → diffusion
   - Cross terms vanish

4. **Integration by parts** (transfers derivatives to the density)
   - Boundary terms vanish due to test function properties
   - Fundamental lemma ensures pointwise equality

**Key insights**:
- **Stochastic dynamics at infinitesimal scales are governed by just two terms—drift and diffusion—both emerging from Taylor expansion**
- **Test functions provide a rigorous framework for working with probability densities**
- **The Fokker-Planck equation is a conservation law for probability**
- **This equation is the foundation for understanding diffusion models**

## Frequently Asked Questions
----------------------------

### Q1: Why do we need second-order Taylor expansion?

**A**: Because the stochastic term $g\,dw$ has size $O(\sqrt{dt})$. When squared, it produces $O(dt)$ terms that survive in the limit. First-order expansion would miss the diffusion term entirely.

### Q2: What if the test function doesn't have compact support?

**A**: We can relax this requirement. We only need $\varphi$ and $p$ to decay fast enough at infinity that boundary terms vanish. In practice, $p$ is a probability density (decays at infinity) and we choose $\varphi$ to decay at least as fast.

### Q3: How does this relate to the Chapman-Kolmogorov equation?

**A**: The Fokker-Planck equation is the continuous-time limit of Chapman-Kolmogorov. In discrete time, Chapman-Kolmogorov describes how transition probabilities compose. Taking the limit gives Fokker-Planck.

### Q4: What about non-Markovian processes?

**A**: The Fokker-Planck equation assumes the process is Markovian (memoryless). For non-Markovian processes, you need generalized forms like the generalized Langevin equation or path integral methods.

### Q5: Can I solve the Fokker-Planck equation analytically?

**A**: Rarely. Analytical solutions exist for:
- Constant coefficients (Gaussian processes)
- Ornstein-Uhlenbeck process
- Some special 1D cases

Generally, you need numerical methods or approximations.

### Q6: Why is it called "Fokker-Planck"?

**A**: Named after Adriaan Fokker and Max Planck who derived it (independently) in the 1910s for studying Brownian motion. It's also called the Kolmogorov forward equation.

## Further Reading
-----------------

### Classical References
- Risken, H. (1996). *The Fokker-Planck Equation*. Springer. (The definitive reference)
- Gardiner, C. W. (2009). *Stochastic Methods*. Springer. (More accessible)

### Modern Perspectives
- Pavliotis, G. A. (2014). *Stochastic Processes and Applications*. Springer. (Rigorous treatment)
- Øksendal, B. (2013). *Stochastic Differential Equations*. Springer. (Standard textbook)

### For Diffusion Models
- Song et al. (2021). "Score-Based Generative Modeling through SDEs"
- See: [SDE View Overview](01_diffusion_sde_view.md) in this repository

## Related Documents
--------------------

- [Taylor Expansions in Diffusion](02a_taylor_expansion.md) - Mathematical background for the derivation
- [Deriving DDPM from VP-SDE](02_sde_and_ddpm.md) - Connecting discrete and continuous views
- [DDPM to VP-SDE (Continuous Limit)](02c_ddpm_to_vpsde.md) - How DDPM becomes an SDE
- [SDE View Overview](01_diffusion_sde_view.md) - Big picture of SDEs in diffusion models
