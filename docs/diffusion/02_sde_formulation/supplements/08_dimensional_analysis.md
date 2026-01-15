# Dimensional Analysis in SDEs and Diffusion Models

**Why units matter: from physical intuition to diffusion model design**

Dimensional analysis is one of the most powerful sanity checks in physics and mathematics. In SDEs and diffusion models, tracking units reveals why certain formulas must have specific forms, why the $\sqrt{dt}$ scaling is inevitable, and why score functions have inverse data units.

This supplement builds unit intuition from classical diffusion, translates it to abstract data spaces, and shows how dimensional consistency constrains diffusion model design.

---

## Table of Contents

1. [Why Units Matter](#1-why-units-matter)
2. [Units in Classical Diffusion](#2-units-in-classical-diffusion)
3. [Units in Abstract Data Spaces](#3-units-in-abstract-data-spaces)
4. [The Score Function's Units](#4-the-score-functions-units)
5. [Dimensional Consistency in SDEs](#5-dimensional-consistency-in-sdes)
6. [Applications to Diffusion Models](#6-applications-to-diffusion-models)

---

## 1. Why Units Matter

### The Power of Dimensional Analysis

**Dimensional analysis** is the practice of tracking physical units through equations. It's powerful because:

1. **Error detection**: Dimensionally inconsistent equations are wrong
2. **Structure revelation**: Units force certain mathematical forms
3. **Scaling laws**: Dimensional reasoning predicts how systems behave
4. **Sanity checks**: Quick verification without detailed calculation

**Example**: If someone claims $\text{velocity} = \text{acceleration} \times \text{time}^2$, you can immediately reject it:

$$
[L/T] \neq [L/T^2] \times [T^2] = [L]
$$

The units don't match, so the equation is wrong.

### Units in Diffusion Models

Even though diffusion models work in abstract "data space" (pixels, latent vectors, gene expression), **units still exist and still constrain the mathematics**.

Tracking units explains:
- Why noise variance grows linearly in time
- Why score functions have units of inverse data
- Why $\sqrt{dt}$ appears everywhere
- Why certain noise schedules are valid

---

## 2. Units in Classical Diffusion

### The Setup

Start with ordinary physical space:
- Space variable $x$ has units of **length**: $[x] = [L]$
- Time $t$ has units of **time**: $[t] = [T]$

Everything else follows from these choices.

### Units of Probability Density

A probability density $p(x,t)$ is **not dimensionless**.

The probability of finding a particle in region $V$ is:

$$
\mathbb{P}(X_t \in V) = \int_V p(x,t)\,dx
$$

The left side is dimensionless (it's a probability). The volume element $dx$ has units $[L^d]$ in $d$ dimensions.

Therefore:

$$
[p] \cdot [L^d] = 1 \quad \Rightarrow \quad [p] = [L^{-d}]
$$

**Key insight**: Density means "per unit volume."

### Units of the Drift

In an SDE:

$$
dX_t = f(X_t)\,dt + \sigma\,dW_t
$$

The term $f(X_t)\,dt$ must have the same units as $dX_t$, which is length.

Since $[dt] = [T]$, we must have:

$$
[f] = [L/T]
$$

So $f$ is a **velocity field**—not metaphorically, but dimensionally.

### Units of the Probability Current

The probability current is:

$$
J(x,t) = f(x)\,p(x,t)
$$

Multiply the units:

$$
[J] = [f] \cdot [p] = [L/T] \cdot [L^{-d}] = [L^{-(d-1)}/T]
$$

**Interpretation**: $J$ measures "probability crossing a unit area per unit time."

This is exactly what a flux should measure. The equation $J = (\text{density}) \times (\text{velocity})$ is not an analogy—it's dimensional necessity.

### Units of Divergence

The divergence operator $\nabla \cdot$ involves spatial derivatives:

$$
\nabla \cdot J = \sum_{i=1}^d \frac{\partial J_i}{\partial x_i}
$$

Each derivative contributes a factor of $[1/L]$:

$$
[\nabla \cdot J] = [J] \cdot [1/L] = [L^{-(d-1)}/T] \cdot [1/L] = [L^{-d}/T]
$$

Check the continuity equation:

$$
\partial_t p + \nabla \cdot J = 0
$$

Both terms have units $[L^{-d}/T]$. ✓

**Why divergence?** It's the **only** operator with the right units to convert flux into density change.

### Units of Diffusion

For the diffusion equation:

$$
\partial_t p = D\,\Delta p
$$

The Laplacian contributes two spatial derivatives:

$$
[\Delta] = [1/L^2]
$$

So:

$$
[D] \cdot [p] \cdot [1/L^2] = [p]/[T]
$$

This forces:

$$
[D] = [L^2/T]
$$

**Physical meaning**: Diffusion coefficient measures "how much area a random walker explores per unit time."

This is why Brownian motion spreads as variance $\sim Dt$.

### Units of the Noise Term

Brownian motion satisfies:

$$
[dW_t] = [\sqrt{T}]
$$

For $\sigma\,dW_t$ to have units of length:

$$
[\sigma] = [L/\sqrt{T}]
$$

Then:

$$
D = \frac{\sigma^2}{2} \quad \Rightarrow \quad [D] = [L^2/T]
$$

Everything is consistent. ✓

---

## 3. Units in Abstract Data Spaces

### The Translation

In diffusion models, space is no longer physical space. Instead:

- $x \in \mathbb{R}^d$ is **data space**
- Coordinates are pixel intensities, audio amplitudes, latent features, etc.

There's no meter stick—but **units still exist**.

### The Abstract Unit

Call the unit of data $[X]$. This could be:
- Pixel intensity (0-255 or normalized)
- Audio amplitude
- Gene expression level
- Latent coordinate

The key is that $[X]$ is **not dimensionless**—it's the unit of your data.

### Reinterpreting All Units

Now translate everything from physical space to data space:

| Quantity | Physical Space | Data Space |
|----------|---------------|------------|
| State | $[L]$ | $[X]$ |
| Density | $[L^{-d}]$ | $[X^{-d}]$ |
| Drift | $[L/T]$ | $[X/T]$ |
| Current | $[L^{-(d-1)}/T]$ | $[X^{-(d-1)}/T]$ |
| Noise scale | $[L/\sqrt{T}]$ | $[X/\sqrt{T}]$ |
| Diffusion coeff | $[L^2/T]$ | $[X^2/T]$ |

**The dimensional structure is identical.**

### Why This Matters

Even though "meters" are gone, the **relationships between units** remain:

- Gradients have units $[1/X]$
- Divergence has units $[1/X]$
- Laplacians have units $[1/X^2]$

These constraints apply whether $X$ is meters or pixels.

---

## 4. The Score Function's Units

### Definition

The score function is:

$$
s(x,t) = \nabla_x \log p(x,t)
$$

What are its units?

### Careful Derivation

**Step 1**: Units of $p(x,t)$

$$
[p] = [X^{-d}]
$$

**Step 2**: The logarithm issue

Strictly speaking, $\log$ requires a dimensionless argument. The proper interpretation is:

$$
\log p \equiv \log(p/p_0)
$$

where $p_0$ is a reference density with the same units as $p$. Since $p_0$ is constant, it doesn't affect gradients:

$$
\nabla_x \log p = \nabla_x \log(p/p_0) = \nabla_x \log p
$$

**Step 3**: Apply the gradient

Use the identity:

$$
\nabla_x \log p = \frac{\nabla_x p}{p}
$$

The gradient contributes $[1/X]$:

$$
[\nabla_x p] = [\nabla_x] \cdot [p] = [1/X] \cdot [X^{-d}] = [X^{-(d+1)}]
$$

Divide by $p$:

$$
\left[\frac{\nabla_x p}{p}\right] = \frac{[X^{-(d+1)}]}{[X^{-d}]} = [X^{-1}]
$$

Therefore:

$$
\boxed{[s] = [\nabla_x \log p] = [X^{-1}]}
$$

**The score has units of inverse data.**

### Sanity Check: 1D Gaussian

For a 1D Gaussian:

$$
p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

The score is:

$$
\frac{d}{dx}\log p(x) = -\frac{x-\mu}{\sigma^2}
$$

Units:
- Numerator: $[x-\mu] = [X]$
- Denominator: $[\sigma^2] = [X^2]$

So:

$$
\left[-\frac{x-\mu}{\sigma^2}\right] = [X]/[X^2] = [X^{-1}]
$$

Exactly as predicted. ✓

---

## 5. Dimensional Consistency in SDEs

### The Forward SDE

$$
dx = f(x,t)\,dt + g(t)\,dw
$$

**Check units**:

Left side: $[dx] = [X]$

Right side, drift term: $[f] \cdot [dt] = [X/T] \cdot [T] = [X]$ ✓

Right side, diffusion term: $[g] \cdot [dw] = [X/\sqrt{T}] \cdot [\sqrt{T}] = [X]$ ✓

Everything matches.

### The Reverse SDE

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

**Check the score correction term**:

$$
[g^2 \cdot \nabla_x \log p] = [X^2/T] \cdot [1/X] = [X/T]
$$

This matches the drift $[f] = [X/T]$. ✓

**Key insight**: The reverse SDE is dimensionally consistent because the score has units $[1/X]$.

### The Fokker-Planck Equation

$$
\partial_t p = -\nabla \cdot (f p) + \frac{g^2}{2} \Delta p
$$

**Check each term**:

Left side: $[\partial_t p] = [X^{-d}/T]$

Right side, drift term: $[\nabla \cdot (f p)] = [1/X] \cdot [X/T] \cdot [X^{-d}] = [X^{-d}/T]$ ✓

Right side, diffusion term: $[g^2 \Delta p] = [X^2/T] \cdot [1/X^2] \cdot [X^{-d}] = [X^{-d}/T]$ ✓

All terms have the same units.

---

## 6. Applications to Diffusion Models

### Why Noise Variance Grows Linearly

Variance has units $[X^2]$. Time has units $[T]$.

For dimensional consistency:

$$
\text{Var}(x_t) \sim [X^2/T] \cdot [T] = [X^2]
$$

The diffusion coefficient $D = [X^2/T]$ must multiply time to give variance.

**Conclusion**: Variance **must** grow linearly in time for dimensional consistency.

### Why $\sqrt{dt}$ Appears

Noise increments must have units $[X]$. Time steps have units $[T]$.

For $\sigma \sqrt{dt} \cdot \varepsilon$ to have units $[X]$:

$$
[\sigma] \cdot [\sqrt{T}] = [X] \quad \Rightarrow \quad [\sigma] = [X/\sqrt{T}]
$$

**Conclusion**: Noise **must** scale as $\sqrt{dt}$ for dimensional consistency.

### Why Score Functions Are Learned

The reverse SDE requires $\nabla_x \log p_t(x)$ with units $[1/X]$.

A neural network with:
- **Input**: $x \in \mathbb{R}^d$ (units $[X]$)
- **Output**: $s_\theta(x,t) \in \mathbb{R}^d$ (units $[1/X]$)

naturally has the right dimensional structure.

### Time Reparameterization

Different noise schedules are valid if they preserve dimensional consistency.

For example, reparameterizing time $t \to \tau(t)$ requires:

$$
\frac{d\tau}{dt} = \text{dimensionless}
$$

So $\tau$ has the same units as $t$, and all formulas remain dimensionally consistent.

### Noise Schedule Constraints

A noise schedule $\beta(t)$ must have units:

$$
[\beta(t)] = [1/T]
$$

So that $\beta(t)\,dt$ is dimensionless (it's integrated to give $\bar{\alpha}_t$).

Similarly, $g(t) = \sqrt{\beta(t)}$ has units:

$$
[g(t)] = [1/\sqrt{T}]
$$

Wait, this seems wrong! Let me recalculate...

Actually, for VP-SDE: $g(t) = \sqrt{\beta(t)}$ where $\beta(t)$ is a **rate** with units $[1/T]$.

But we need $[g] = [X/\sqrt{T}]$ for the noise term to work.

**Resolution**: The data is implicitly normalized so that $[X] = 1$ (dimensionless in the normalized space). Then:

$$
[g] = [1/\sqrt{T}]
$$

which is consistent with $g(t) = \sqrt{\beta(t)}$.

**Lesson**: Normalization affects dimensional analysis, but the **structure** remains.

---

## Summary: The Power of Units

### Key Takeaways

1. **Units constrain structure**: Dimensional consistency forces certain mathematical forms
2. **Density has units**: $[p] = [X^{-d}]$ (per unit volume)
3. **Drift is velocity**: $[f] = [X/T]$
4. **Score is inverse data**: $[s] = [1/X]$
5. **Noise scales as $\sqrt{dt}$**: Only scaling that gives finite variance
6. **Divergence and Laplacian have fixed units**: $[1/X]$ and $[1/X^2]$

### Why This Matters

Dimensional analysis:
- **Detects errors**: Inconsistent units mean wrong equations
- **Guides design**: Units constrain valid noise schedules
- **Builds intuition**: Physical reasoning transfers to abstract spaces
- **Enables sanity checks**: Quick verification without detailed math

### For Diffusion Models

Even in abstract data spaces:
- Probability behaves like a compressible fluid
- Density, velocity, and flux have well-defined units
- Score functions naturally have inverse data units
- Time reparameterization must preserve dimensional structure

**The units don't let you cheat.** They force the mathematics whether the "space" is meters or pixels.

---

## Further Reading

### Dimensional Analysis

- **Barenblatt (1996)**: *Scaling, Self-similarity, and Intermediate Asymptotics*
  - Comprehensive treatment of dimensional analysis

- **Bridgman (1922)**: *Dimensional Analysis*
  - Classic text on the method

### Applications to SDEs

- **Pavliotis (2014)**: *Stochastic Processes and Applications*
  - Section on scaling and dimensional analysis

- **Gardiner (2009)**: *Stochastic Methods*
  - Physical intuition for SDEs

### Diffusion Models

- **Song et al. (2021)**: [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
  - Implicit dimensional structure in noise schedules

---

## Exercises

### Conceptual

1. **Why inverse data?** Explain in your own words why the score function must have units $[1/X]$.

2. **Variance scaling**: Show that if noise scaled as $dt$ instead of $\sqrt{dt}$, variance would have the wrong units.

3. **Gradient units**: Verify that $[\nabla_x] = [1/X]$ by considering the definition of a derivative.

### Computational

4. **Check VP-SDE**: For VP-SDE with $f(x,t) = -\frac{1}{2}\beta(t) x$ and $g(t) = \sqrt{\beta(t)}$, verify dimensional consistency assuming normalized data.

5. **Score network**: Design a neural network architecture that naturally outputs a score function with the correct units.

---

**Next**: With dimensional intuition in place, you can now confidently design noise schedules, verify equations, and understand why certain formulas are inevitable rather than arbitrary!
