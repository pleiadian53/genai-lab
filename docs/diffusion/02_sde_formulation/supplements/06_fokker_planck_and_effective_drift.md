# The Fokker-Planck Equation and the Effective Drift

## Overview

The **Fokker-Planck equation** (FPE) is the bridge between:
- **SDEs**: Describe individual particle/sample trajectories
- **PDEs**: Describe how the probability density of all particles evolves

This is the key to understanding why the probability flow ODE exists and why its effective drift has a specific form.

---

## The Fokker-Planck Equation

### The Formula

For the SDE:

$$
dx = f(x,t)\,dt + g(t)\,dw
$$

The corresponding Fokker-Planck equation is:

$$
\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot \left(f(x,t) \cdot p_t(x)\right) + \frac{1}{2}g(t)^2 \nabla^2 p_t(x)
$$

### What This Equation Describes

The FPE tells us **how the probability density $p_t(x)$ changes over time**.

- **Left side**: Rate of change of probability density at location $x$
- **Right side**: Two effects that cause probability to flow

---

## Term-by-Term Interpretation

### Term 1: Drift (Advection)

$$
-\nabla \cdot \left(f(x,t) \cdot p_t(x)\right)
$$

**What it means**: Probability is carried along by the drift $f(x,t)$.

**Physical analogy**: Imagine probability as a fluid. The drift $f$ is like a current that pushes the fluid. Where the current converges, probability accumulates; where it diverges, probability spreads out.

**Mathematical structure**:

- $f(x,t) \cdot p_t(x)$ is the **probability flux** (probability per unit time per unit area flowing due to drift)
- $\nabla \cdot (\cdot)$ is the divergence operator
- The negative sign means: if flux flows out of a region (positive divergence), probability decreases there

**Expansion** (for scalar case):

$$
-\nabla \cdot (f \cdot p) = -\frac{\partial}{\partial x}(f \cdot p) = -f \frac{\partial p}{\partial x} - p \frac{\partial f}{\partial x}
$$

The first term is advection (probability carried by flow), the second accounts for varying drift.

---

### Clarification: The Drift $f(x,t)$

#### Notation: Is $f \cdot p$ a Dot Product?

**No.** The notation $f(x,t) \cdot p_t(x)$ is **scalar multiplication**, not a dot product:

- $f(x,t) \in \mathbb{R}^d$ is a **vector** (the drift field)
- $p_t(x) \in \mathbb{R}$ is a **scalar** (probability density)
- $f \cdot p$ means: multiply each component of the vector $f$ by the scalar $p$

The result is a vector in $\mathbb{R}^d$, which is then acted on by the divergence operator $\nabla \cdot$.

#### Units of Drift

$$
\text{Units of } f(x,t) = \frac{[\text{position}]}{[\text{time}]} = \text{velocity}
$$

In the SDE $dx = f(x,t)\,dt + g(t)\,dw$:
- $dx$ has units of position
- $dt$ has units of time
- Therefore $f(x,t)$ must have units of position/time for dimensional consistency

#### Physical Meaning of Drift

The drift $f(x,t)$ is the **deterministic velocity field**. It answers:

> "If there were no noise, which direction and how fast would $x$ move?"

**Examples**:

| System | Drift $f(x,t)$ | Meaning |
|--------|---------------|---------|
| Particle in gravity | $f = -g$ (constant) | Falls at constant rate |
| Spring (Hooke's law) | $f = -kx$ | Pulled toward origin |
| VP-SDE (diffusion models) | $f = -\frac{1}{2}\beta(t)x$ | Shrinks toward origin |
| VE-SDE | $f = 0$ | No deterministic motion |

**In diffusion models (VP-SDE)**: The drift $f(x,t) = -\frac{1}{2}\beta(t)x$ means the state $x$ (e.g., an image) is deterministically pulled toward the origin at a rate proportional to $\beta(t)$. This "shrinking" combines with the noise term to gradually destroy the signal while keeping variance bounded.

#### Units of Probability Flux

The probability flux due to drift is:

$$
J_{\text{drift}} = f(x,t) \cdot p_t(x)
$$

**Units**:

$$

\frac{[\text{position}]}{[\text{time}]} \times \frac{[\text{probability}]}{[\text{position}]^d} = \frac{[\text{probability}]}{[\text{time}] \cdot [\text{position}]^{d-1}}
$$

**Meaning**: Rate of probability flowing per unit (hyper)surface area due to deterministic motion. Think of it as: "How much probability passes through a surface per unit time because of the drift?"

---

### Term 2: Diffusion

$$
\frac{1}{2}g(t)^2 \nabla^2 p_t(x)
$$

**What it means**: Probability spreads out due to random noise.

**Physical analogy**: Think of dropping ink in water. Even without currents, the ink spreads due to molecular collisions. This is diffusion—probability smooths out, moving from high-concentration to low-concentration regions.

**Mathematical structure**:

- $\nabla^2 = \sum_i \frac{\partial^2}{\partial x_i^2}$ is the Laplacian
- $g(t)^2$ controls the diffusion strength
- The factor $\frac{1}{2}$ comes from Itô calculus conventions

**Key property**: Diffusion always smooths. Peaks get lower, valleys get filled. This is why pure diffusion eventually leads to a uniform distribution (in the absence of boundaries).

---

## The FPE as a Conservation Law

The FPE can be written as a **continuity equation**:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot J = 0
$$

where $J$ is the **probability current** (flux):

$$
J = f \cdot p_t - \frac{1}{2}g(t)^2 \nabla p_t
$$

**Interpretation**: Probability is conserved—it doesn't appear or disappear, only flows from one location to another.

The total probability current has two parts:
1. **Drift current**: $f \cdot p_t$ (probability carried by deterministic flow)
2. **Diffusion current**: $-\frac{1}{2}g(t)^2 \nabla p_t$ (probability flowing from high to low density)

---

## Interpreting the FPE for Diffusion Models

### Forward Process

For the VP-SDE: $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw$

The FPE becomes:

$$

\frac{\partial p_t}{\partial t} = \frac{1}{2}\beta(t) \nabla \cdot (x \cdot p_t) + \frac{1}{2}\beta(t) \nabla^2 p_t
$$

**What happens**:

- **Drift term**: Pushes probability toward the origin ($x \to 0$)
- **Diffusion term**: Spreads probability out
- **Combined effect**: Probability converges to $\mathcal{N}(0, I)$ as $t \to T$

### Reverse Process

For the reverse-time SDE, the FPE runs backward, describing how probability flows from noise back to data.

---

## Why the Effective Drift Takes This Form

Now we derive why the probability flow ODE has:

$$
\text{Effective drift} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)
$$

### The Goal

Find an ODE (no noise) that produces the **same** probability evolution as the SDE.

### Step 1: Start with the FPE

The SDE:

$$

dx = f(x,t)\,dt + g(t)\,dw
$$

has FPE:

$$

\frac{\partial p_t}{\partial t} = -\nabla \cdot (f \cdot p_t) + \frac{1}{2}g(t)^2 \nabla^2 p_t
$$

### Step 2: Rewrite the Diffusion Term

The key trick is to express $\nabla^2 p$ using the score function.

**Identity**: For any smooth density $p$:

$$

\nabla^2 p = \nabla \cdot (\nabla p)
$$

**Now use the score**: Since $\nabla \log p = \frac{\nabla p}{p}$, we have $\nabla p = p \cdot \nabla \log p$.

Therefore:

$$

\nabla^2 p = \nabla \cdot (\nabla p) = \nabla \cdot (p \cdot \nabla \log p)
$$

### Step 3: Substitute into the FPE

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot (f \cdot p_t) + \frac{1}{2}g(t)^2 \nabla \cdot (p_t \cdot \nabla \log p_t)
$$

### Step 4: Combine the Divergence Terms

Factor out the divergence:

$$

\frac{\partial p_t}{\partial t} = -\nabla \cdot \left(f \cdot p_t - \frac{1}{2}g(t)^2 p_t \cdot \nabla \log p_t\right)
$$

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot \left(\left[f - \frac{1}{2}g(t)^2 \nabla \log p_t\right] \cdot p_t\right)
$$

### Step 5: Recognize This as a Pure Advection Equation

The equation now has the form:

$$

\frac{\partial p_t}{\partial t} = -\nabla \cdot (\tilde{f} \cdot p_t)
$$

where:

$$

\boxed{\tilde{f}(x,t) = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)}
$$

**This is the FPE of an ODE** (no diffusion term). The ODE:

$$

dx = \tilde{f}(x,t)\,dt = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

produces the **exact same** probability evolution as the original SDE!

---

## Intuitive Understanding of the Effective Drift

### Why $-\frac{1}{2}g(t)^2 \nabla \log p_t$?

The diffusion term in the SDE ($g(t)dw$) causes probability to spread out. In the absence of this noise, we need something to compensate.

**The score term does exactly this**:

- $\nabla \log p_t$ points toward higher density
- The term $-\frac{1}{2}g(t)^2 \nabla \log p_t$ pulls samples toward high-density regions
- This "anti-diffusion" counteracts the spreading that would have happened from noise

### Physical Analogy

Imagine two ways to move particles from point A to point B:

1. **SDE way**: Random walk with a drift. Particles take wiggly paths, but on average they move from A to B.

2. **ODE way**: Deterministic flow with enhanced drift. Particles take smooth paths, but the drift is adjusted to account for the "missing" randomness.

Both methods move the same amount of probability mass from A to B, but via different particle trajectories.

### Why Factor of 1/2?

In the SDE, the diffusion term contributes to probability spreading with coefficient $\frac{1}{2}g(t)^2$ (from Itô calculus). To compensate for this exactly, the effective drift correction must also have coefficient $\frac{1}{2}g(t)^2$.

---

## Summary Table

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| **Fokker-Planck equation** | $\frac{\partial p_t}{\partial t} = -\nabla \cdot (fp_t) + \frac{1}{2}g^2 \nabla^2 p_t$ | How probability density evolves |
| **Drift term** | $-\nabla \cdot (fp_t)$ | Probability carried by deterministic flow |
| **Diffusion term** | $\frac{1}{2}g^2 \nabla^2 p_t$ | Probability spreading from noise |
| **Probability current** | $J = fp_t - \frac{1}{2}g^2 \nabla p_t$ | Total probability flow |
| **Effective drift** | $f - \frac{1}{2}g^2 \nabla \log p_t$ | ODE drift that matches SDE marginals |

---

## Connection to Diffusion Models

In diffusion models:

1. **Training**: Learn the score $s_\theta(x,t) \approx \nabla_x \log p_t(x)$

2. **SDE sampling**: Use the reverse SDE with learned score
   $$
   dx = [f - g^2 s_\theta]dt + g\,dw
   $$

3. **ODE sampling (DDIM)**: Use the probability flow ODE with learned score
   $$

   dx = [f - \tfrac{1}{2}g^2 s_\theta]dt
   $$

Both use the same learned score $s_\theta$, but:
- SDE has coefficient $g^2$ on the score (plus noise term)
- ODE has coefficient $\frac{1}{2}g^2$ on the score (no noise)

The Fokker-Planck analysis proves they generate from the same distribution.

---

## References

- **Risken (1989)**: "The Fokker-Planck Equation" — Comprehensive textbook
- **Øksendal (2003)**: "Stochastic Differential Equations" — Mathematical foundations
- **Song et al. (2021)**: "Score-Based Generative Modeling through SDEs" — Application to diffusion models
- **Maoutsa et al. (2020)**: "Interacting Particle Solutions of Fokker-Planck Equations" — Numerical methods

