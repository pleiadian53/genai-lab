# Taylor Expansions in Diffusion Models

Taylor expansions are the **quiet workhorse** behind diffusion models, SDE discretization, and the Fokker–Planck equation. They're rarely foregrounded, but almost every "natural-looking" formula in diffusion comes from a first- or second-order expansion where higher-order terms are deliberately thrown away.

This document explains why Taylor expansions appear everywhere in diffusion models and how they enable the bridge between continuous theory and discrete algorithms.

---

## 1. The Continuous-Discrete Boundary

Diffusion models live at an awkward boundary:

- The **true theory** is continuous in time (SDEs, PDEs)
- The **algorithms** are discrete (finite steps, finite networks)

**Taylor expansions bridge these worlds.**

### Signs You're Seeing Taylor Expansion

Any time you encounter:

- "small time step"
- "as $dt \to 0$"
- "ignore higher-order terms"
- "first-order accurate"

you are seeing Taylor expansion at work.

---

## 2. Taylor Expansion for Dynamics

For a smooth function $f(t)$, the Taylor expansion is:

$$
f(t + \Delta t) = f(t) + f'(t)\,\Delta t + \frac{1}{2}f''(t)\,\Delta t^2 + \cdots
$$

### The Core Philosophy

> Over a small enough time step, the future is approximately linear in the present.

Diffusion models lean heavily on this idea, but applied to:

- **Random processes** (SDEs)
- **Probability densities** (Fokker–Planck equation)

---

## 3. Warm-Up: Deterministic Dynamics

Consider an ordinary differential equation (ODE):

$$
\frac{dx(t)}{dt} = a(x(t), t)
$$

By Taylor expansion:

$$
x(t + \Delta t) = x(t) + a(x(t), t)\,\Delta t + O(\Delta t^2)
$$

This is **Euler's method** for numerical integration.

Nothing fancy yet—just Taylor applied to deterministic dynamics.

---

## 4. Enter Randomness: Why SDEs Change the Rules

Now consider a stochastic differential equation (SDE):

$$
dx(t) = f(x, t)\,dt + g(t)\,dw(t)
$$

### The Key Twist

- $dt$ is "small"
- But $dw(t)$ is **not** order $dt$
- It is order $\sqrt{dt}$

### Two Scales

We have **two different scales**:

| Term Type | Scaling |
|-----------|----------|
| Deterministic | $\sim dt$ |
| Stochastic | $\sim \sqrt{dt}$ |

This single fact **reshapes Taylor expansion** for stochastic processes.

---

## 5. Taylor Logic Behind Euler–Maruyama

When discretizing the SDE, we write:

$$
x(t + \Delta t) - x(t) = f(x, t)\,\Delta t + g(t)\,\Delta w
$$

with:

$$
\Delta w \sim \mathcal{N}(0, \Delta t)
$$

### Hierarchy of Terms

Observe the **order of magnitude**:

| Term | Order |
|------|-------|
| $f\,\Delta t$ | $\Delta t$ |
| $g\,\Delta w$ | $\sqrt{\Delta t}$ |
| $(\Delta w)^2$ | $\Delta t$ |
| $(\Delta w)^3$ | $\Delta t^{3/2}$ |

### Truncation Rule

- **Keep** terms up to order $\Delta t$
- **Drop** terms like $\Delta t^{3/2}$

This truncation **is** a Taylor expansion, adapted to stochastic scaling.

---

## 6. Example: Why $\sqrt{1-\beta}$ Becomes $1 - \frac{1}{2}\beta$

In DDPM, the forward step is:

$$
x_{k+1} = \sqrt{1-\beta_k}\,x_k + \sqrt{\beta_k}\,\varepsilon
$$

Why not just $1 - \frac{1}{2}\beta_k$?

### Taylor Expansion of the Square Root

$$
\sqrt{1-\beta_k} = 1 - \frac{1}{2}\beta_k - \frac{1}{8}\beta_k^2 + \cdots
$$

**To first order** in $\beta_k$:

$$
\sqrt{1-\beta_k} \approx 1 - \frac{1}{2}\beta_k
$$

Higher-order terms are deliberately ignored.

### Connection to VP-SDE

This is exactly the same approximation that appears when discretizing:

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

### Why Keep the Square Root?

DDPM keeps $\sqrt{1-\beta_k}$ because:

- It matches the first-order Taylor expansion
- It **behaves better at finite step sizes**
- It **exactly preserves variance**

This is a **numerical stabilization choice**, not an accident.

---

## 7. Taylor Expansion Behind the Fokker–Planck Equation

Now we reach the deeper place where Taylor expansions do heavy conceptual lifting.

### Setup

Let $p(x, t)$ be the probability density of $x(t)$. We want an equation for how $p$ evolves over time.

Start from:

$$
p(x, t + \Delta t) = \mathbb{E}\left[p(x - \Delta x, t)\right]
$$

### Taylor Expand the Density

Expand **the density itself** in space:

$$
p(x - \Delta x, t) \approx p(x, t) - \Delta x \cdot \nabla p(x, t) + \frac{1}{2}(\Delta x \Delta x^\top) : \nabla^2 p(x, t)
$$

### Substitute SDE Increment

From the SDE:

$$
\Delta x = f\,\Delta t + g\,\Delta w
$$

### Take Expectations

Key facts:

- $\mathbb{E}[\Delta w] = 0$
- $\mathbb{E}[\Delta w \Delta w^\top] = \Delta t\,I$

All higher-order terms vanish or are $o(\Delta t)$.

### The Result: Fokker–Planck Equation

$$
\frac{\partial p}{\partial t} = -\nabla \cdot (f p) + \frac{1}{2}\nabla^2 (g^2 p)
$$

**Key insight**: This entire PDE is literally a **second-order Taylor expansion of the density under random motion**.

---

## 8. Why Stop at Second Order?

You might ask: why stop at second order in the Fokker–Planck derivation?

### The Scaling Argument

| Order | Physical Meaning | Scaling |
|-------|------------------|----------|
| First | Drift (deterministic flow) | $\Delta t$ |
| Second | Diffusion (spreading) | $\Delta t$ |
| Third+ | Higher moments | $\Delta t^{3/2}$ or higher |

**Key fact**: $(\Delta w)^n \sim (\Delta t)^{n/2}$

For $n \geq 3$, these terms **vanish** in the $\Delta t \to 0$ limit.

### This Is Not Hand-Waving

It's a rigorous scaling argument. The Fokker–Planck equation is the **exact continuous-time limit**.

---

## 9. The Hidden Pattern Across Diffusion Models

You can now recognize a repeating structure:

| Component | What Taylor Expansion Does |
|-----------|----------------------------|
| DDPM forward step | Linearize SDE over small time |
| Noise schedule | Match first-order decay |
| Reverse SDE | Drop higher-order stochastic terms |
| Score matching | Linearize log-density gradients |
| Fokker–Planck | Second-order expansion of density |
| Probability-flow ODE | Remove stochastic second-order term |

**Philosophy**: Diffusion models are **Taylor expansions with taste**—you keep just enough terms to stay correct, stable, and learnable.

---

## 10. The Big Picture

> **Diffusion models work because, over infinitesimal time, random dynamics are simple—and Taylor expansions let us exploit that simplicity repeatedly.**

### Everything "Magical" Comes from Taylor

Everything that feels magical in diffusion models:

- Gaussian noise
- Linear drift
- Quadratic variance
- Score as gradient
- Clean discretizations

comes from **discarding higher-order terms in a controlled way**.

---

## Summary

Taylor expansions are the **mathematical glue** that turns continuous stochastic dynamics into tractable learning rules in diffusion models, governing:

1. **SDE discretization** (Euler–Maruyama)
2. **Fokker–Planck equation** (density evolution)
3. **Variance-preserving structure** (DDPM forward chain)
4. **Score matching** (gradient approximations)

---

## Next Steps

To deepen understanding:

1. **Derive Fokker–Planck line by line**: Spell out every expectation, no black boxes
2. **Verify variance preservation**: Show $\sqrt{1-\beta_k}$ exactly preserves variance
3. **Study Itô calculus**: Understand $(dw)^2 = dt$ rigorously

---

## Related Documents

- [Deriving DDPM from VP-SDE](02_sde_and_ddpm.md)
- [SDE View Overview](01_diffusion_sde_view.md)
- [Historical Development](../diffusion/history/diffusion_models_development.md)
