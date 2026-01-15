# The Fokker-Planck Equation: From SDEs to Probability Evolution

**Understanding how stochastic processes evolve probability distributions**

The Fokker-Planck equation (FPE) is the bridge between individual particle motion (described by SDEs) and collective probability evolution (described by PDEs). It's the reason we can train diffusion models by learning score functions and the foundation for understanding reverse-time sampling.

This supplement builds intuition from first principles, explains why divergence and the Laplacian must appear, and connects everything to diffusion models.

---

## Table of Contents

1. [The Central Question](#1-the-central-question)
2. [From Particles to Densities](#2-from-particles-to-densities)
3. [Conservation Laws and Divergence](#3-conservation-laws-and-divergence)
4. [The Laplacian and Diffusion](#4-the-laplacian-and-diffusion)
5. [The Complete Fokker-Planck Equation](#5-the-complete-fokker-planck-equation)
6. [Geometric Intuition](#6-geometric-intuition)
7. [Connection to Diffusion Models](#7-connection-to-diffusion-models)

---

## 1. The Central Question

### The Setup

You have a stochastic differential equation:

$$
dX_t = f(X_t, t)\,dt + g(t)\,dW_t
$$

This describes how a **single particle** moves randomly through space.

But in diffusion models, we don't care about individual trajectories. We care about **probability distributions**:

$$
p_t(x) = \text{probability density of } X_t \text{ at position } x
$$

**The question**: How does $p_t(x)$ evolve over time?

**The answer**: The Fokker-Planck equation.

---

## 2. From Particles to Densities

### The Conceptual Shift

**Particle view**: Track individual random trajectories $X_t(\omega)$ for each random outcome $\omega$.

**Density view**: Track the probability density $p_t(x)$ describing where particles are likely to be.

**Key insight**: Even though individual particles move randomly, the **density** evolves deterministically according to a PDE.

### Why This Matters

In diffusion models:
- **Forward process**: We know how to corrupt data (SDE)
- **Training**: We learn the score function $\nabla_x \log p_t(x)$
- **Sampling**: We need to evolve probability backwards

The FPE is what connects these three pieces.

---

## 3. Conservation Laws and Divergence

### The Physical Principle

Imagine a huge swarm of particles. At time $t$, they're spread out with density $p(x,t)$.

Take any region $V$ in space. The total probability inside is:

$$
\mathbb{P}(X_t \in V) = \int_V p(x,t)\,dx
$$

**Fundamental assumption**: Probability doesn't teleport. It can't be created or destroyed. The only way it changes is by **flowing across the boundary**.

### Probability Current

To describe flow, introduce a **probability current** (or flux) $J(x,t)$:

$$
J(x,t) = \text{probability flow per unit area per unit time}
$$

The total probability leaving $V$ per unit time is:

$$
\text{outflow} = \int_{\partial V} J(x,t) \cdot n(x)\,dS
$$

where $n(x)$ is the outward unit normal on the boundary $\partial V$.

### Conservation Equation

Conservation states:

$$
\frac{d}{dt}\int_V p(x,t)\,dx = -\int_{\partial V} J \cdot n\,dS
$$

The minus sign: outward flow decreases the amount inside.

### Enter Divergence

By the **divergence theorem** (Gauss's theorem):

$$
\int_{\partial V} J \cdot n\,dS = \int_V \nabla \cdot J\,dx
$$

Substituting:

$$
\int_V \partial_t p\,dx = -\int_V \nabla \cdot J\,dx
$$

Since this holds for **every region** $V$, the integrands must match:

$$
\boxed{\partial_t p(x,t) + \nabla \cdot J(x,t) = 0}
$$

This is the **continuity equation**.

### Why Divergence Appears

**Divergence is not a choice**—it's the unique local operator that measures "net outflow from an infinitesimal volume."

**Physical meaning**: $\nabla \cdot J$ tells you whether probability is accumulating (negative) or escaping (positive) at a point.

**Geometric meaning**: Divergence measures how a flow expands or contracts volume.

---

## 4. The Laplacian and Diffusion

### What Is the Laplacian?

The Laplacian of a scalar field $u(x)$ is:

$$
\Delta u(x) = \nabla^2 u(x) = \sum_{i=1}^d \frac{\partial^2 u}{\partial x_i^2}
$$

But what does it **mean**?

### The Key Intuition

At a point $x$, the Laplacian answers:

> **Is the value here higher or lower than the average value nearby?**

More precisely:

$$
\Delta u(x) \propto \lim_{r \to 0} \frac{\text{Avg}_{|y-x|=r} u(y) - u(x)}{r^2}
$$

**Interpretation**:

- $\Delta u(x) > 0$: $u(x)$ is **below** its local average (valley)
- $\Delta u(x) < 0$: $u(x)$ is **above** its local average (peak)
- $\Delta u(x) = 0$: $u(x)$ matches its local average (locally flat)

### Derivation via Taylor Expansion

Take a sphere of radius $r$ centered at $x_0$. The average value on the sphere is:

$$
\langle u \rangle_{S_r} = \frac{1}{|S_r|} \int_{S_r} u(x_0 + h)\,dS
$$

Taylor expand $u(x_0 + h)$:

$$
u(x_0 + h) = u(x_0) + \nabla u(x_0) \cdot h + \frac{1}{2} h^\top H(x_0) h + o(|h|^2)
$$

where $H$ is the Hessian matrix.

**Key observations**:
1. Constant term: $u(x_0)$ survives
2. Linear term: $\nabla u(x_0) \cdot h$ vanishes by symmetry (sphere is symmetric)
3. Quadratic term: Only diagonal terms survive, giving $\frac{r^2}{2d} \Delta u(x_0)$

**Result**:

$$
\langle u \rangle_{S_r} - u(x_0) = \frac{r^2}{2d} \Delta u(x_0) + o(r^2)
$$

So the Laplacian literally measures the difference between a point and its neighborhood average!

### Why Diffusion Equals the Laplacian

Consider pure Brownian motion:

$$
dX_t = \sigma\,dW_t
$$

Particles spread isotropically. The density evolves as:

$$
\partial_t p = \frac{\sigma^2}{2} \Delta p
$$

This is the **heat equation**.

**Why this form?**

1. **Local**: Depends only on infinitesimal neighborhood
2. **Isotropic**: Doesn't prefer any direction
3. **Mass-conserving**: Total probability stays 1
4. **Gaussian**: Produces exactly the Gaussian spreading of Brownian motion

**Physical interpretation**: Diffusion is nature's way of eliminating curvature. Peaks decrease, valleys fill.

---

## 5. The Complete Fokker-Planck Equation

### Two Mechanisms of Motion

An SDE has two parts:

$$
dX_t = \underbrace{f(X_t,t)\,dt}_{\text{drift}} + \underbrace{g(t)\,dW_t}_{\text{diffusion}}
$$

**Drift**: Deterministic transport along a vector field  
**Diffusion**: Random spreading due to noise

### The Probability Current

The total current has two contributions:

$$
J(x,t) = \underbrace{f(x,t)\,p(x,t)}_{\text{advection}} - \underbrace{\frac{g(t)^2}{2}\nabla p(x,t)}_{\text{diffusion}}
$$

**Advection term**: $J_{\text{drift}} = f \cdot p$
- Density times velocity (like fluid mechanics)
- Probability is transported by the drift field

**Diffusion term**: $J_{\text{diff}} = -D \nabla p$ (Fick's law)
- Flow goes from high to low concentration
- $D = g(t)^2/2$ is the diffusion coefficient

### The Full Equation

Plug into the continuity equation $\partial_t p + \nabla \cdot J = 0$:

$$
\partial_t p = -\nabla \cdot (f p) + \frac{g(t)^2}{2} \Delta p
$$

Expanding the divergence:

$$
\boxed{\partial_t p = -\nabla \cdot (f p) + \frac{g(t)^2}{2} \Delta p}
$$

Or equivalently:

$$
\boxed{\partial_t p = -f \cdot \nabla p - p\,\nabla \cdot f + \frac{g(t)^2}{2} \Delta p}
$$

This is the **Fokker-Planck equation** (also called the **forward Kolmogorov equation**).

### Reading the Equation

**Term by term**:

1. **$-\nabla \cdot (f p)$**: Probability transported by drift
2. **$\frac{g(t)^2}{2} \Delta p$**: Probability spread by noise

**As a sentence**: "Probability changes because it's pushed by drift and smoothed by diffusion."

---

## 6. Geometric Intuition

### Divergence: Where Does Probability Go?

**Question**: Is probability accumulating or escaping at this point?

**Answer**: $\nabla \cdot J$

- **Positive divergence**: Net outflow → density decreases
- **Negative divergence**: Net inflow → density increases
- **Zero divergence**: Incompressible flow → density unchanged by transport

**Mental image**: Stand at a point with a microscopic balloon. Divergence tells you whether the balloon inflates (source), deflates (sink), or stays constant (incompressible).

### Laplacian: How Curved Is the Density?

**Question**: Is this point higher or lower than its surroundings?

**Answer**: $\Delta p$

- **Positive Laplacian**: Valley → fills in
- **Negative Laplacian**: Peak → flattens out
- **Zero Laplacian**: Harmonic → locally balanced

**Mental image**: Diffusion punishes disagreement with neighbors. The Laplacian measures that disagreement.

### Together: Flow + Smoothing

The FPE describes two fundamental motions:

1. **Divergence**: Rearranges probability (transport)
2. **Laplacian**: Smooths probability (diffusion)

These are the **only two local operations** that respect conservation, isotropy, and locality.

---

## 7. Connection to Diffusion Models

### The Forward Process

In diffusion models, the forward SDE is:

$$
dx = f(x,t)\,dt + g(t)\,dw
$$

Common choices:
- **VP-SDE**: $f(x,t) = -\frac{1}{2}\beta(t) x$, $g(t) = \sqrt{\beta(t)}$
- **VE-SDE**: $f(x,t) = 0$, $g(t) = \sqrt{\frac{d\sigma^2(t)}{dt}}$

The density $p_t(x)$ evolves via FPE:

$$
\partial_t p_t = -\nabla \cdot (f p_t) + \frac{g(t)^2}{2} \Delta p_t
$$

### The Score Function

Rewrite the FPE using the **score function** $s_t(x) = \nabla_x \log p_t(x)$:

$$
\nabla p_t = p_t \cdot s_t
$$

Substitute into the Laplacian term:

$$
\Delta p_t = \nabla \cdot (\nabla p_t) = \nabla \cdot (p_t s_t) = p_t |s_t|^2 + p_t \nabla \cdot s_t
$$

The FPE becomes:

$$
\partial_t p_t = -\nabla \cdot \left(p_t \left[f - \frac{g^2}{2} s_t - \frac{g^2}{2} \nabla \cdot s_t\right]\right)
$$

This form shows how the score appears in the probability current.

### The Reverse Process

The **reverse-time SDE** (Anderson, 1982) is:

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

**Key insight**: The score function $\nabla_x \log p_t(x)$ is what you need to reverse the diffusion.

**Training**: Learn $s_\theta(x,t) \approx \nabla_x \log p_t(x)$ using denoising score matching.

**Sampling**: Use the learned score in the reverse SDE to generate data.

### Why the FPE Matters

1. **Connects SDEs to PDEs**: Particle motion → density evolution
2. **Justifies score matching**: The score appears naturally in the reverse SDE
3. **Explains training**: We're learning the gradient of the log-density
4. **Enables sampling**: The reverse SDE uses the learned score

---

## Summary: The Big Picture

### What We've Learned

1. **Conservation forces divergence**: Probability flow requires $\partial_t p + \nabla \cdot J = 0$
2. **Drift gives advection**: $J_{\text{drift}} = f \cdot p$ (density times velocity)
3. **Noise gives diffusion**: $J_{\text{diff}} = -D \nabla p$ (Fick's law)
4. **FPE combines both**: $\partial_t p = -\nabla \cdot (f p) + D \Delta p$
5. **Score function bridges**: $\nabla \log p$ connects forward and reverse processes

### The Operators Are Not Arbitrary

- **Divergence**: The unique local measure of net outflow
- **Laplacian**: The unique isotropic local smoothing operator

They appear because they **must**—any other choice would violate conservation, locality, or isotropy.

### For Diffusion Models

The FPE explains:
- Why we can learn score functions
- How reverse-time sampling works
- Why the Laplacian appears in generators
- How SDEs and PDEs are two views of the same process

---

## Further Reading

### Primary Sources

- **Fokker (1914)**: Original work on Brownian motion
- **Planck (1917)**: Generalization to arbitrary drift
- **Kolmogorov (1931)**: Forward and backward equations
- **Anderson (1982)**: Reverse-time SDE

### Modern Treatments

- **Øksendal (2003)**: *Stochastic Differential Equations*
  - Chapter 7: The Fokker-Planck equation
  
- **Pavliotis (2014)**: *Stochastic Processes and Applications*
  - Comprehensive treatment of FPE and applications

- **Song et al. (2021)**: [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
  - Application to diffusion models

### Related Supplements

- [`01_forward_sde_design_choices.md`](./01_forward_sde_design_choices.md): How to choose $f$ and $g$
- [`05_reverse_sde_and_probability_flow_ode.md`](./05_reverse_sde_and_probability_flow_ode.md): Reverse-time sampling
- [`06_fokker_planck_and_effective_drift.md`](./06_fokker_planck_and_effective_drift.md): Advanced FPE topics

---

## Exercises

### Conceptual

1. **Why must divergence appear?** Explain in your own words why conservation of probability forces the continuity equation.

2. **Laplacian intuition**: For a 1D function $u(x)$, explain why $u''(x) > 0$ means $u(x)$ is below its local average.

3. **Score connection**: Show that $\nabla p = p \cdot \nabla \log p$ and explain why this is useful.

### Computational

4. **Verify FPE for Ornstein-Uhlenbeck**: For $dx = -\theta x\,dt + \sigma\,dw$, verify that the Gaussian density $p_t(x) = \mathcal{N}(x; \mu_t, \sigma_t^2)$ satisfies the FPE.

5. **Simulate and compare**: Simulate the SDE and solve the FPE numerically. Compare the histogram of particles to the PDE solution.

---

**Next**: Now that you understand the FPE, you're ready to see how it connects to the generator of an SDE and how Itô's formula makes everything rigorous. See the main SDE tutorial for the complete picture!
