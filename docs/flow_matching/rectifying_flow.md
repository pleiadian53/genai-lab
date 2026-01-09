# Rectified Flow: A Flow Matching Tutorial

This tutorial introduces **rectified flow**, one of the simplest and most elegant approaches to generative modeling. We build from first principles with precise notation.

---

## 1. The Problem: Transforming Noise into Data

Generative models solve a fundamental problem: transform samples from a simple distribution (noise) into samples from a complex distribution (data).

**Setup:**

- **Data distribution**: $p_{\text{data}}(x)$ — the distribution we want to sample from
- **Noise distribution**: $p_{\text{noise}}(x)$ — a simple distribution we can easily sample (often Gaussian)
- **Goal**: Learn a continuous transformation that maps noise → data

Rectified flow achieves this by learning a **velocity field** that transports points along straight paths.

---

## 2. Notation and Objects

We work in continuous space $x \in \mathbb{R}^d$ with the following definitions:

| Symbol | Meaning |
|--------|---------|
| $x_0 \sim p_{\text{data}}$ | Sample from data distribution |
| $x_1 \sim p_{\text{noise}}$ | Sample from noise distribution |
| $t \in [0, 1]$ | Continuous time parameter |
| $x_t$ | Interpolated point at time $t$ |

The key insight: we construct a **path** $x_t$ connecting data ($x_0$) to noise ($x_1$).

---

## 3. The Linear Interpolation Path

Rectified flow makes a deliberate, simple choice for the path:

$$
\boxed{x_t = (1 - t) \cdot x_0 + t \cdot x_1}
$$

This is **linear interpolation** between a data point and a noise point.

**Interpretation:**

- At $t = 0$: $x_t = x_0$ (pure data)
- At $t = 1$: $x_t = x_1$ (pure noise)
- For $t \in (0, 1)$: a point along the straight line between them

This is purely geometric — no stochasticity in the path definition.

---

## 4. Velocity: The Central Object

Differentiating the path with respect to time:

$$
\frac{dx_t}{dt} = x_1 - x_0
$$

This derivative is the **velocity** — the direction and speed of movement along the path.

**Key observations:**

- **Constant**: The velocity doesn't change along the path
- **Direction**: Points from data toward noise
- **Deterministic**: Depends only on the pair $(x_0, x_1)$

Rectified flow trains a neural network to **predict this velocity** given only the current position and time.

---

## 5. The Neural Network

We introduce a neural network $v_\theta(x, t)$ that learns the velocity field:

- **Input**: Position $x \in \mathbb{R}^d$ and time $t \in [0, 1]$
- **Output**: Velocity vector in $\mathbb{R}^d$
- **Meaning**: "How should a point at position $x$ at time $t$ be moving?"

**Training target:**

$$
v_\theta(x_t, t) \approx x_1 - x_0
$$

The model learns **how points should move**, not probabilities or likelihoods.

---

## 6. The Rectified Flow Loss

Training uses simple mean-squared-error regression:

$$
\mathcal{L}_{\text{RF}} = \mathbb{E}_{x_0, x_1, t} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

**Sampling procedure during training:**

1. Sample $x_0 \sim p_{\text{data}}$ (data point)
2. Sample $x_1 \sim p_{\text{noise}}$ (noise point)
3. Sample $t \sim \text{Uniform}[0, 1]$ (random time)
4. Compute $x_t = (1-t) x_0 + t x_1$ (interpolated point)
5. Predict velocity and compute loss

This is **flow matching** in its simplest form.

---

## 7. Why "Rectified"?

The term *rectified* comes from geometry, meaning "straightened."

If we had access to the true **optimal transport** between $p_{\text{data}}$ and $p_{\text{noise}}$, the trajectories would generally be curved (following geodesics in probability space).

Rectified flow instead:

> **Straightens** (rectifies) the paths into lines, letting the neural network implicitly compensate for any distortion.

The straight-line paths are not the true optimal transport — but they're simple to define and work remarkably well in practice.

---

## 8. Sampling: Generating New Data

After training, we generate samples by **solving an ODE backward in time**.

**Procedure:**

1. Start from noise: $x(1) \sim p_{\text{noise}}$
2. Integrate backward: $\frac{dx}{dt} = -v_\theta(x(t), t)$ from $t=1$ to $t=0$
3. Result: $x(0) \approx x_{\text{data}}$

**Properties:**

- **Deterministic**: No noise injected during sampling
- **Standard solvers**: Can use Euler, RK4, or adaptive ODE solvers
- **Efficient**: Often needs far fewer steps than diffusion (10-50 vs 100-1000)

---

## 9. Flow Matching: The General Framework

Rectified flow is a specific instance of the broader **flow matching** framework:

| Component | General Flow Matching | Rectified Flow |
|-----------|----------------------|----------------|
| Path $x_t$ | Any differentiable path | Linear interpolation |
| Target velocity | $\dot{x}_t$ | Constant: $x_1 - x_0$ |
| Stochasticity | Optional | None |

**Hierarchy:**

$$
\text{Rectified Flow} \subset \text{Flow Matching} \subset \text{Continuous Generative Models}
$$

---

## 10. Comparison: Rectified Flow vs Score Matching

These two approaches answer fundamentally different questions:

| Aspect | Score Matching (Diffusion) | Rectified Flow |
|--------|---------------------------|----------------|
| Forward process | Stochastic (add noise) | Deterministic (interpolate) |
| What's learned | Score: $\nabla_x \log p_t(x)$ | Velocity: $v_\theta(x, t)$ |
| Reverse process | Stochastic SDE | Deterministic ODE |
| Assumptions | Gaussian noise schedules | Minimal |
| Question answered | "Where is probability increasing?" | "How should this point move?" |

**Mental model:**

- **Score matching**: Learn forces on a probability landscape
- **Rectified flow**: Learn motion through state space

---

## 11. Why Gaussian Noise (and When It's Not Required)

Rectified flow does **not** require Gaussian noise for $x_1$.

**Why Gaussian is common:**

- Easy to sample
- Isotropic (no preferred direction)
- Numerically stable
- Enables comparison with diffusion baselines

**Alternatives (important for biology):**

- Domain-specific priors
- Learned noise distributions
- Structured noise matching data characteristics

This flexibility is one reason rectified flow generalizes well beyond images.

---

## 12. Connection to Transformers

Rectified flow pairs naturally with Transformer architectures because it requires networks that can:

- Model **global dependencies** (attention excels here)
- **Condition on time** (via embeddings and modulation)
- Handle **flexible input structures** (tokens, not grids)

This combination — rectified flow + Transformer — forms the backbone of modern generative models like those used in Stable Diffusion 3 and video generation.

See the companion tutorial on **Diffusion Transformers (DiT)** for architectural details.

---

## Summary

> **Rectified flow learns a deterministic velocity field that transports noise to data along straightened paths, using simple regression instead of probability gradients.**

**Key equations:**

- Path: $x_t = (1-t) x_0 + t x_1$
- Velocity: $\frac{dx_t}{dt} = x_1 - x_0$
- Loss: $\mathcal{L} = \mathbb{E}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$
- Sampling: Solve $\frac{dx}{dt} = -v_\theta(x, t)$ from $t=1$ to $t=0$

---

## References

- Liu et al. (2022) - "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- Lipman et al. (2023) - "Flow Matching for Generative Modeling"
- Albergo & Vanden-Eijnden (2023) - "Building Normalizing Flows with Stochastic Interpolants"
