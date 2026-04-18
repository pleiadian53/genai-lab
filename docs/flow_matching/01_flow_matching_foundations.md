# Flow Matching Foundations

How do you turn Gaussian noise into a photorealistic face, a protein sequence, or a musical phrase? The answer requires learning a transformation that moves probability mass from one distribution to another. Flow matching is a principled, elegant framework for doing exactly that.

At its core, flow matching treats the transformation as a continuous-time *flow*: every point in the noise distribution traces a smooth trajectory to a corresponding point in the data distribution. The model's job is to learn the *velocity field* that drives these trajectories — and, crucially, this can be done with a simple regression loss, without the complex score-matching objectives or carefully tuned noise schedules that diffusion models require.

This document builds the mathematical foundations of flow matching from the ground up: the continuity equation that governs how distributions evolve, the conditional flow matching objective that makes training tractable, the ODE-based sampling procedure, and the connection to optimal transport that explains why straight paths are so attractive in practice.

---

## Overview

### The Core Idea

The generative modeling problem can be framed as a *transport* problem: we have a simple source distribution $p_0$ (easy to sample from, e.g. Gaussian noise) and a complex target distribution $p_1$ (the data distribution we care about). We want to find a continuous map that moves samples from one to the other.

Flow matching parameterizes this map as the solution to an ordinary differential equation (ODE):

**Goal**: Transform samples from a simple distribution $p_0$ (noise) into samples from a complex distribution $p_1$ (data).

**Approach**: Learn a time-dependent velocity field $v_\theta(x, t)$ that defines a continuous transformation:

$$
\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]
$$

Starting from a noise sample $x(0) \sim p_0$, integrating this ODE to $t = 1$ produces a data sample $x(1) \sim p_1$.

**Key insight**: Instead of learning probability gradients (scores), we directly learn how points should move through space. This shift — from asking "how likely is this point?" to asking "which direction should this point move?" — is what makes flow matching both mathematically clean and practically efficient.

### Why Flow Matching?

**Advantages over diffusion models**:

- **Simpler training**: Direct regression instead of score matching
- **Faster sampling**: Deterministic ODE, fewer steps needed (10-50 vs 100-1000)
- **Flexible paths**: Not restricted to Gaussian noise schedules
- **Theoretical clarity**: Optimal transport interpretation
- **Better for non-Euclidean data**: Natural extension to manifolds

**Trade-offs**:

- Less mature than diffusion (fewer architectural tricks)
- Requires careful ODE solver selection
- May need more training data for complex distributions

---

## Mathematical Framework

To understand how flow matching works, we need to formalize what it means to "move a distribution" via a vector field. This section builds the mathematical language: probability flows, the continuity equation that links velocity fields to distribution evolution, and the forward process that defines *which* trajectories we want the model to learn.

### Probability Flows

A **probability flow** is a time-dependent vector field that transports probability mass smoothly from one distribution to another. The key insight is that specifying a velocity field $v(x, t)$ is equivalent to specifying a family of distributions $\{p_t\}_{t \in [0,1]}$ — the flow *induces* the distributions.

**Setup**:

- Start distribution: $p_0(x)$ (e.g., Gaussian noise)
- End distribution: $p_1(x)$ (data distribution)
- Time: $t \in [0, 1]$

**Flow equation**:

$$
\frac{dx}{dt} = v(x, t)
$$

This defines a trajectory $x(t)$ for each starting point $x(0)$.

**Probability evolution**: The distribution at time $t$, denoted $p_t(x)$, evolves according to the **continuity equation**:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v) = 0
$$

The continuity equation is the PDE counterpart of the particle ODE $\frac{dx}{dt} = v(x, t)$. While the ODE tracks individual particles, the continuity equation describes how the *entire distribution* evolves in time. Understanding this connection is central to flow matching.

#### Physical Intuition: Probability as a Fluid

Think of $p_t(x)$ as the density of an incompressible fluid at position $x$ and time $t$, and $v(x, t)$ as the local fluid velocity. The product $p_t(x)\, v(x, t)$ is then the *probability flux* — a vector field describing how much probability mass flows through a surface element per unit time. The divergence $\nabla \cdot (p_t v)$ measures how much of that flux is *leaving* a small region per unit volume.

**Dimensional analysis** (working in $\mathbb{R}^d$, using $L$ for length):

| Quantity | Units | Reasoning |
|----------|-------|-----------|
| $p_t(x)$ | $\text{prob} \cdot L^{-d}$ | Probability per unit $d$-volume (integrates to 1 over $\mathbb{R}^d$) |
| $v(x, t)$ | $L \cdot T^{-1}$ | Velocity |
| $p_t v$ (flux) | $\text{prob} \cdot L^{-(d-1)} \cdot T^{-1}$ | Density × velocity; reduces the spatial dimension by one |
| $\nabla \cdot (p_t v)$ | $\text{prob} \cdot L^{-d} \cdot T^{-1}$ | Divergence applies $\partial/\partial x_i$, adding one inverse length |
| $\partial p_t / \partial t$ | $\text{prob} \cdot L^{-d} \cdot T^{-1}$ | Time derivative of density; units match ✓ |

The flux $p_t v$ carries units of probability per $(d-1)$-dimensional surface element per unit time. When integrated over the boundary $\partial\Omega$ — a $(d-1)$-dimensional hypersurface — the result has units of probability per unit time: the rate at which probability mass crosses that boundary. In three dimensions this boundary is literally an *area* ($L^2$), which is where the "per unit area" phrasing originates; in general it is a $(d-1)$-dimensional hypersurface element.

> **Rate of local density change = −(net outflow of probability flux)**

If more probability is flowing out of a region than flowing in, the local density decreases — exactly the physics of a conserved fluid. No probability mass is ever created or destroyed; it merely moves.

#### Term-by-Term Breakdown

| Term | Interpretation |
|------|---------------|
| $\frac{\partial p_t}{\partial t}$ | Local rate of change of probability density at position $x$ and time $t$ |
| $p_t(x)\, v(x, t)$ | Probability flux: the "current" of probability mass flowing at $(x, t)$ |
| $\nabla \cdot (p_t v)$ | Divergence of that flux — net outflow of probability per unit volume |
| $= 0$ | Conservation: total probability is preserved (the distribution integrates to 1 for all $t$) |

#### Derivation Sketch

For any test region $\Omega \subset \mathbb{R}^d$, the total probability mass inside $\Omega$ at time $t$ is $\int_\Omega p_t(x)\, dx$. Since particles move deterministically under $\frac{dx}{dt} = v(x, t)$, mass can only enter or leave $\Omega$ by crossing its boundary $\partial \Omega$:

$$
\frac{d}{dt} \int_\Omega p_t(x)\, dx = -\oint_{\partial \Omega} p_t(x)\, v(x, t) \cdot \hat{n}\, dS
$$

Applying the divergence theorem to the right-hand side converts the surface integral into a volume integral over $\nabla \cdot (p_t v)$. Because $\Omega$ is arbitrary, the integrands must match pointwise, giving the continuity equation.

#### Connection to the ODE and the Pushforward

The continuity equation reveals the deep link between particle trajectories and distribution evolution. If every particle starting at $x_0$ follows the ODE $\frac{dx}{dt} = v(x, t)$, it traces a path $x(t) = \phi_t(x_0)$, where $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$ is called the **flow map**. The distribution at time $t$ is simply the distribution you get by applying $\phi_t$ to every sample from $p_0$, called the *pushforward* of $p_0$ under $\phi_t$:

$$
p_t = {\phi_t}_{\,\#}\, p_0
\quad \Longleftrightarrow \quad
\text{if } X_0 \sim p_0 \text{, then } \phi_t(X_0) \sim p_t
$$

The $\#$ subscript is notation for "push forward through": ${\phi_t}_{\,\#}\, p_0$ means "apply $\phi_t$ to samples drawn from $p_0$." The continuity equation is the differential form of this statement — instead of asking where particles end up after a finite time $t$, it tracks how density changes over an infinitesimal step $dt$. The goal of flow matching is to learn a velocity field $v_\theta$ whose induced flow map $\phi_1$ transports $p_{\text{noise}}$ to $p_{\text{data}}$.

### The Forward Process

Unlike diffusion models, which define the forward process by *adding noise* stochastically (making it harder to reason about the exact trajectory), flow matching uses **deterministic interpolation**. We simply connect each data point $x_0$ to a noise point $x_1$ with a smooth path, then train the model to follow that path.

This design choice is deliberate: by decoupling path construction from learning, we can analyze trajectories analytically and choose path families that are convenient for training (e.g., straight lines) or that satisfy constraints (e.g., variance preservation). The model does not need to know which pair $(x_0, x_1)$ generated a given $x_t$ — it only needs to know the *velocity* at that $x_t$.

**Conditional flow**: Given a data-noise pair $(x_0, x_1)$ where $x_0 \sim p_{\text{data}}$ and $x_1 \sim p_{\text{noise}}$, define a smooth path connecting them:

$$
x_t = \psi_t(x_0, x_1), \qquad t \in [0, 1]
$$

Here $\psi_t : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$ is the **interpolant** — a family of functions indexed by time $t$ that smoothly interpolates between the endpoints. It must satisfy the boundary conditions:

$$
\psi_0(x_0, x_1) = x_0 \quad \text{(starts at data)} \qquad \psi_1(x_0, x_1) = x_1 \quad \text{(ends at noise)}
$$

Beyond these endpoints, the shape of the path is a design choice. The conditional velocity along any such path is the time derivative of $\psi_t$:

$$
u_t(x_0, x_1) = \frac{d}{dt}\,\psi_t(x_0, x_1)
$$

This is the quantity the model is trained to predict. Different choices of $\psi_t$ lead to different velocity targets and different training dynamics.

**Common choices**:

**1. Linear interpolation (Rectified Flow)**:

$$
\psi_t(x_0, x_1) = (1-t)\, x_0 + t\, x_1
$$

The path is a straight line from $x_0$ to $x_1$, giving a constant conditional velocity $u_t = x_1 - x_0$ that does not depend on $t$. This is the simplest and most widely used choice.

**2. Geodesic interpolation**:

$$
\psi_t(x_0, x_1) = \exp_{x_0}\!\left(t \log_{x_0}(x_1)\right)
$$

The path follows the shortest curve on a Riemannian manifold between $x_0$ and $x_1$. Reduces to linear interpolation in Euclidean space; useful for manifold-valued data such as rotations, shapes, or hyperbolic embeddings.

**3. Variance-preserving interpolation**:

$$
\psi_t(x_0, x_1) = \sqrt{1-t}\, x_0 + \sqrt{t}\, x_1
$$

The coefficients are chosen so that $\mathbb{E}[\|\psi_t\|^2] \approx \text{const}$ when $x_0$ and $x_1$ have unit variance. This mimics the signal-to-noise schedule of diffusion models and is useful when comparing against DDPM-style baselines.

### The Marginal Flow

There is a subtlety in moving from conditional paths to a trainable model. At any given position $x$ and time $t$, many different data-noise pairs $(x_0, x_1)$ may have generated a trajectory passing through that point. Each pair implies a different conditional velocity $u_t(x_0, x_1)$. The **marginal velocity field** resolves this ambiguity by averaging:

$$
v_t(x) = \mathbb{E}_{x_0, x_1 \mid x_t = x} [u_t(x_0, x_1)]
$$

This is the expected velocity at position $x$ and time $t$, averaged over *all* data-noise pairs whose trajectories pass through $x$ at time $t$. It is the unique velocity field that generates the correct marginal distribution $p_t$ at every time step.

**Key property**: If we know $v_t(x)$ exactly, solving the ODE:

$$
\frac{dx}{dt} = v_t(x), \quad x(0) = x_0 \sim p_{\text{noise}}
$$

will transport $x(0)$ to $x(1) \sim p_{\text{data}}$.

---

## Flow Matching Objective

The marginal velocity field $v_t(x)$ is the ideal target, but it requires integrating over all data-noise pairs that pass through a given $x$ at time $t$ — an intractable computation. The central insight of flow matching is that we do not need to compute this integral explicitly. Instead, we can regress directly on the *conditional* velocities from individual pairs, and the objective still converges to the correct marginal field.

### The Training Loss

Flow matching trains a neural network $v_\theta(x, t)$ to approximate the marginal velocity field $v_t(x)$ by supervising on conditional velocities along sampled trajectories.

**Conditional Flow Matching (CFM) loss**:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1, x_t} \left[ \left\| v_\theta(x_t, t) - u_t(x_0, x_1) \right\|^2 \right]
$$

where:

- $t \sim \text{Uniform}[0, 1]$
- $x_0 \sim p_{\text{data}}$
- $x_1 \sim p_{\text{noise}}$
- $x_t = \psi_t(x_0, x_1)$

#### What a Training Example Looks Like

Each gradient step consumes a batch of independently sampled tuples. One tuple is constructed as follows (using linear interpolation throughout):

**Step 1 — Sample a data point and a noise point**:

$$
x_0 \sim p_{\text{data}}, \qquad x_1 \sim \mathcal{N}(0, I)
$$

For example, $x_0$ might be a flattened image vector and $x_1$ a random Gaussian sample of the same dimension.

**Step 2 — Sample a time**:

$$
t \sim \text{Uniform}[0, 1], \quad \text{e.g. } t = 0.3
$$

**Step 3 — Interpolate to get the noisy input**:

$$
x_t = (1 - t)\,x_0 + t\,x_1 = 0.7\,x_0 + 0.3\,x_1
$$

This is a point 30% of the way along the straight-line path from data to noise. It is what the network *sees* as input, together with $t$.

**Step 4 — Compute the velocity target**:

$$
u_t = x_1 - x_0
$$

This is the direction and magnitude of the straight-line step from $x_0$ to $x_1$. It is the same for all $t$ along this path (constant velocity), so no additional computation is needed.

**Step 5 — Evaluate the loss for this example**:

$$
\ell = \left\| v_\theta(x_t,\, t) - u_t \right\|^2 = \left\| v_\theta(x_t,\, t) - (x_1 - x_0) \right\|^2
$$

The network takes $(x_t, t)$ as input and should predict the direction from the original data point to the noise point. The loss penalizes any deviation from that direction.

**Summary — the five objects in one training example**:

| Object | What it is | Role |
|--------|-----------|------|
| $x_0$ | A real data sample | Anchor: the path starts here |
| $x_1$ | A Gaussian noise sample | Anchor: the path ends here |
| $t$ | A uniform random time in $[0,1]$ | Where along the path to evaluate |
| $x_t = (1-t)x_0 + tx_1$ | The interpolated point | Network input |
| $u_t = x_1 - x_0$ | The conditional velocity | Regression target |

A minibatch is formed by independently drawing $B$ such tuples and averaging the per-example losses. In pseudocode:

```python
# One training step (linear / rectified flow)
x0 = sample_data(batch_size)           # (B, d) — real data
x1 = torch.randn_like(x0)             # (B, d) — Gaussian noise
t  = torch.rand(batch_size, 1)        # (B, 1) — uniform time

x_t    = (1 - t) * x0 + t * x1       # (B, d) — interpolated input
target = x1 - x0                      # (B, d) — constant velocity target

pred = model(x_t, t)                  # (B, d) — network prediction
loss = ((pred - target) ** 2).mean()

loss.backward()
optimizer.step()
```

Note how little machinery is required: no score functions, no importance weights, no carefully calibrated noise schedules. The entire forward pass is three lines.

**Why this works**: Although we train on conditional velocities $u_t(x_0, x_1)$ for specific pairs, the loss is equivalent to matching the marginal field. This follows from a key identity called the **conditional expectation property**:

$$
\mathbb{E}_{x_0, x_1 \mid x_t = x} \bigl[u_t(x_0, x_1)\bigr] = v_t(x)
$$

#### Reading the Notation

The left-hand side is a **conditional expectation**. Written in full as an integral:

$$
\mathbb{E}_{x_0, x_1 \mid x_t = x} \bigl[u_t(x_0, x_1)\bigr]
= \int u_t(x_0, x_1)\; p(x_0, x_1 \mid x_t = x)\; dx_0\, dx_1
$$

The subscript $x_0, x_1 \mid x_t = x$ specifies the distribution being averaged over: not all pairs $(x_0, x_1)$, but only those *consistent with the observed interpolated point $x_t = x$*. In other words, we weight each pair by the posterior probability $p(x_0, x_1 \mid x_t = x)$ — how likely is it that this particular pair generated the point we are currently standing at?

#### Why the Equality Holds

The derivation follows from combining two ingredients: the conditional continuity equation and Bayes' theorem.

**Step 1 — Each conditional path satisfies its own continuity equation.**

For a fixed pair $(x_0, x_1)$, the path $x_t = \psi_t(x_0, x_1)$ is a moving delta mass. Its "conditional distribution" $p_t(x \mid x_0, x_1) = \delta(x - \psi_t(x_0, x_1))$ satisfies a conditional continuity equation driven by the conditional velocity $u_t(x_0, x_1)$:

$$
\frac{\partial\, p_t(x \mid x_0, x_1)}{\partial t} + \nabla \cdot \bigl(p_t(x \mid x_0, x_1)\; u_t(x_0, x_1)\bigr) = 0
$$

**Step 2 — The marginal distribution is a mixture of conditional distributions.**

Averaging over all pairs (weighted by $p(x_0)\,p(x_1)$):

$$
p_t(x) = \int p_t(x \mid x_0, x_1)\; p(x_0)\, p(x_1)\; dx_0\, dx_1
$$

**Step 3 — Integrate Step 1 over all pairs to get the marginal continuity equation.**

Applying the same average to the conditional continuity equation:

$$
\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \underbrace{\int p_t(x \mid x_0, x_1)\; u_t(x_0, x_1)\; p(x_0)\, p(x_1)\; dx_0\, dx_1}_{\text{probability-flux term}} = 0
$$

But the marginal distribution must itself satisfy *its own* continuity equation with some velocity $v_t(x)$:

$$
\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \bigl(p_t(x)\; v_t(x)\bigr) = 0
$$

Matching the two expressions for the flux term:

$$
p_t(x)\; v_t(x) = \int p_t(x \mid x_0, x_1)\; u_t(x_0, x_1)\; p(x_0)\, p(x_1)\; dx_0\, dx_1
$$

**Step 4 — Divide both sides by $p_t(x)$ and apply Bayes' theorem.**

$$
v_t(x) = \int u_t(x_0, x_1)\; \underbrace{\frac{p_t(x \mid x_0, x_1)\; p(x_0)\, p(x_1)}{p_t(x)}}_{=\; p(x_0, x_1 \mid x_t = x)}\; dx_0\, dx_1
= \mathbb{E}_{x_0, x_1 \mid x_t = x}\bigl[u_t(x_0, x_1)\bigr]
$$

The Bayes inversion in the last step is the key move: the weighting $p_t(x \mid x_0, x_1) \cdot p(x_0)\,p(x_1)\,/\,p_t(x)$ is exactly the posterior over which pairs could have produced the point $x$ at time $t$.

#### What This Means for Training

The equality says that $u_t(x_0, x_1)$ — computed from a single sampled pair — is an *unbiased estimator* of the target $v_t(x_t)$. Minimizing the MSE loss over many randomly drawn pairs therefore drives the network toward the true marginal field, without ever computing the intractable integral explicitly. This is directly analogous to how stochastic gradient descent minimizes a population loss using single-sample gradients: noise in each estimate cancels out in expectation, and the average over training converges to the correct target. The tractability of flow matching hinges entirely on this property.

### Comparison with Score Matching

| Aspect | Score Matching (Diffusion) | Flow Matching |
|--------|---------------------------|---------------|
| **Target** | Score: $\nabla_x \log p_t(x)$ | Velocity: $v_t(x)$ |
| **Loss** | Score matching loss (complex) | Simple MSE regression |
| **Forward process** | Stochastic (add noise) | Deterministic (interpolate) |
| **Training complexity** | Requires careful noise schedule | Direct regression |
| **Interpretation** | Probability gradient | Motion direction |

---

## The Backward Process (Sampling)

Training gives us a neural network $v_\theta(x, t)$ that approximates the true velocity field. At inference time, we exploit the fact that the flow is deterministic: to draw a sample from $p_1$ (the data distribution), we simply *run the flow forward* from a noise sample. There is no stochastic component — the same noise input always produces the same output.

### ODE Integration

After training, we generate samples by solving the **flow ODE** forward in time (from $t=0$, noise, to $t=1$, data). In practice, "backward" refers to the generative direction in the diffusion literature, where time runs from noise to data.

**Sampling procedure**:

1. **Initialize**: Sample $x(1) \sim p_{\text{noise}}$ (e.g., $\mathcal{N}(0, I)$)

2. **Integrate backward**: Solve the ODE from $t=1$ to $t=0$:

$$
\frac{dx}{dt} = v_\theta(x(t), t)
$$

3. **Output**: $x(0) \approx x_{\text{data}}$

#### Why Backward, and What Does That Mean?

During training, every interpolation path was defined to run from data at $t=0$ to noise at $t=1$. The learned velocity field $v_\theta(x, t)$ therefore points in the *data-to-noise* direction at every point in space. To generate a sample, we want to travel the opposite way — from noise back to data — which means traversing time in the decreasing direction: $t: 1 \to 0$.

The ODE itself does not change. We run the same equation $\frac{dx}{dt} = v_\theta(x(t), t)$, but we integrate it with *negative* time steps. Concretely, if we divide the interval $[0, 1]$ into $N$ steps of size $\Delta t = 1/N$, each Euler update subtracts $\Delta t$ from the current time:

$$
x_{t - \Delta t} = x_t - \Delta t \cdot v_\theta(x_t,\; t), \qquad t = 1,\; 1-\Delta t,\; \ldots,\; \Delta t
$$

Because $v_\theta$ points from data toward noise, multiplying it by $-\Delta t$ reverses the direction: each step nudges $x$ back toward the data distribution.

#### A Concrete Walkthrough ($N = 4$ steps)

Start with a Gaussian noise sample $x_1 \sim \mathcal{N}(0, I)$ and take four equal steps of $\Delta t = 0.25$:

| Step | Current time $t$ | Update | Next time |
|------|-----------------|--------|-----------|
| 1 | $t = 1.00$ | $x_{0.75} = x_{1.00} - 0.25 \cdot v_\theta(x_{1.00},\; 1.00)$ | $t = 0.75$ |
| 2 | $t = 0.75$ | $x_{0.50} = x_{0.75} - 0.25 \cdot v_\theta(x_{0.75},\; 0.75)$ | $t = 0.50$ |
| 3 | $t = 0.50$ | $x_{0.25} = x_{0.50} - 0.25 \cdot v_\theta(x_{0.50},\; 0.50)$ | $t = 0.25$ |
| 4 | $t = 0.25$ | $x_{0.00} = x_{0.25} - 0.25 \cdot v_\theta(x_{0.25},\; 0.25)$ | $t = 0.00$ |

At each step, the network is queried at the *current* position and time. The time argument $t$ tells the network where along the trajectory it is being evaluated — the velocity field is different at $t=1.0$ (near noise) than at $t=0.25$ (near data), so conditioning on $t$ is essential.

After four steps, $x_0$ is the generated sample. In pseudocode:

```python
x = torch.randn(batch_size, d)   # x_1 ~ N(0, I)
N = 50                            # number of integration steps
dt = 1.0 / N

for step in range(N):
    t = 1.0 - step * dt           # t decreases: 1.0, 1-dt, 1-2dt, ...
    t_tensor = torch.full((batch_size, 1), t)
    x = x - dt * model(x, t_tensor)   # Euler step backward

# x is now x_0 ≈ data sample
```

**Key properties**:

- **Deterministic**: The same noise input $x_1$ always produces the same output $x_0$, since the ODE has a unique solution
- **Reversible**: The forward pass (data → noise) and backward pass (noise → data) use the same network and ODE; only the direction of integration changes
- **Step-count tradeoff**: More steps give a more accurate trajectory but cost more network evaluations; with straight paths (rectified flow), even $N = 10$–$20$ steps often suffices

### ODE Solvers

Flow matching uses standard numerical ODE solvers:

**1. Euler method** (simplest):

$$
x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t)
$$

**2. Runge-Kutta 4 (RK4)** (more accurate):

$$
\begin{align}
k_1 &= v_\theta(x_t, t) \\
k_2 &= v_\theta(x_t - \frac{\Delta t}{2} k_1, t - \frac{\Delta t}{2}) \\
k_3 &= v_\theta(x_t - \frac{\Delta t}{2} k_2, t - \frac{\Delta t}{2}) \\
k_4 &= v_\theta(x_t - \Delta t \cdot k_3, t - \Delta t) \\
x_{t-\Delta t} &= x_t - \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{align}
$$

**3. Adaptive solvers** (e.g., Dormand-Prince):
- Automatically adjust step size
- Balance accuracy and computation
- Useful for complex trajectories

**Typical performance**:

- **Euler**: 50-100 steps for good quality
- **RK4**: 10-20 steps for good quality
- **Adaptive**: 5-15 steps with error control

---

## Rectified Flow: The Simplest Instance

Flow matching is a general framework that admits many choices of interpolation path. Rectified flow makes the most aggressive simplification possible: connect every data-noise pair with a *straight line*. This turns out to be not just mathematically convenient but practically powerful — straight paths are easy for the ODE solver to follow, require fewer integration steps, and connect naturally to optimal transport theory.

### Definition

**Rectified flow** uses the simplest possible choices:

**Path**: Linear interpolation

$$
x_t = (1-t) x_0 + t x_1
$$

**Velocity**: Constant

$$
u_t(x_0, x_1) = x_1 - x_0
$$

**Loss**: Direct MSE

$$
\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

### Why "Rectified"?

The term comes from geometry: **rectification** means straightening.

**Intuition**: The ideal transport between two distributions generally follows curved trajectories through probability space (Wasserstein geodesics). Rectified flow *straightens* these into straight lines. The neural network then learns to follow straight-line trajectories — a much simpler task than navigating curved paths.

Straight paths have a critical practical advantage: an ODE solver integrating along a straight line can take large steps without accumulating error. Curved paths, by contrast, require many small steps to track accurately. This is why rectified flow can generate high-quality samples with 10–20 function evaluations, whereas score-based diffusion models may require hundreds.

A key subtlety is that while individual $(x_0, x_1)$ paths are straight, the *marginal* velocity field $v_t(x)$ is not constant in $x$ — it must reconcile the fact that different trajectories pass through the same $x_t$ from different directions. The network learns this aggregated field, which is smooth but not trivially linear.

**Advantage**: Straight paths are:
- Simple to define and compute targets for
- Easy to integrate (fewer ODE steps, larger step sizes)
- Numerically stable
- Amenable to analysis via optimal transport theory

### Reflow: Iterative Straightening

Rectified flow can be **iteratively refined** through a process called **reflow**:

**Algorithm**:
1. Train initial model $v_\theta^{(1)}$
2. Generate samples using $v_\theta^{(1)}$
3. Use these samples as new training data
4. Train refined model $v_\theta^{(2)}$
5. Repeat

**Effect**: Each iteration makes paths straighter, requiring fewer sampling steps.

**Typical results**:

- Iteration 1: 50 steps needed
- Iteration 2: 20 steps needed
- Iteration 3: 10 steps needed

---

## Theoretical Properties

The design choices in flow matching — the conditional flow objective, the connection to marginal flows via the conditional expectation property, the relation to optimal transport — are not arbitrary. This section makes those connections precise and shows why the framework has strong theoretical guarantees.

### Optimality

**Theorem** (Lipman et al., 2023): The flow matching objective is equivalent to minimizing:

$$
\mathbb{E}_t \left[ \mathbb{E}_{x \sim p_t} \left[ \|v_\theta(x, t) - v_t(x)\|^2 \right] \right]
$$

This is the $L^2$ distance between the learned and true velocity fields.

**Consequence**: Flow matching directly optimizes the sampling quality.

### Connection to Optimal Transport

Flow matching has a deep connection to **optimal transport** (OT) theory, which provides a normative framework for asking: among all possible ways to move probability mass from $p_0$ to $p_1$, which one is most "efficient"?

**Optimal transport problem**: Find the transport map $T: p_0 \to p_1$ that minimizes the expected cost of moving each particle:

$$
\min_T \, \mathbb{E}_{x_0 \sim p_0} \left[ c(x_0, T(x_0)) \right]
$$

where $c(x, y)$ is a cost function (typically $\|x - y\|^2$, the squared Euclidean distance).

**Connection**:

- Rectified flow with linear interpolation approximates the OT map under the $L^2$ cost: straight-line paths are the shortest possible paths, so they minimize total transport cost
- The learned velocity field $v_\theta$ implicitly defines a transport map $\phi_1: x_0 \mapsto x_1$
- Reflow iterations progressively straighten trajectories toward the true OT map — each round of reflow further reduces crossing paths, which are suboptimal under any cost that penalizes movement

The OT perspective explains why reflow is so effective at reducing the number of sampling steps: the OT map produces the *straightest possible* trajectories, which are trivial for any ODE solver to integrate accurately.

### Probability Flow ODE

Flow matching defines a **probability flow ODE** that transports distributions:

$$
\frac{d p_t}{dt} + \nabla \cdot (p_t v_t) = 0
$$

**Key property**: If $x(0) \sim p_0$, then $x(t) \sim p_t$ when following the flow.

**Comparison with diffusion**:

- Diffusion: Stochastic SDE with deterministic probability flow ODE
- Flow matching: Directly learns the probability flow ODE

---

## Variance-Preserving vs. Non-Variance-Preserving

The choice of interpolation path affects more than geometry — it determines the *statistics* of the intermediate distributions $\{p_t\}$. Two families are common in practice: variance-preserving (VP) flows, which maintain a roughly constant signal-to-noise ratio throughout the trajectory, and non-variance-preserving (NVP) flows, which allow variance to vary freely. Understanding this tradeoff matters when adapting flow matching to different data modalities or when comparing against diffusion baselines that use VP-type noise schedules.

### Variance-Preserving (VP) Flows

**Interpolation**:

$$
x_t = \sqrt{1-\sigma_t^2} \, x_0 + \sigma_t \, x_1
$$

where $\sigma_t$ is a noise schedule (e.g., $\sigma_t = t$).

**Properties**:

- Maintains $\mathbb{E}[\|x_t\|^2] \approx \text{const}$ (if $x_0, x_1$ have similar norms)
- Similar to diffusion models
- Useful when data has specific scale

**Velocity**:

$$
u_t = \frac{d}{dt}\left(\sqrt{1-\sigma_t^2} \, x_0 + \sigma_t \, x_1\right)
$$

### Non-Variance-Preserving (NVP) Flows

**Interpolation** (Rectified Flow):

$$
x_t = (1-t) x_0 + t x_1
$$

**Properties**:

- Does not preserve variance
- Simpler mathematics
- Often works better in practice

**Velocity**:

$$
u_t = x_1 - x_0
$$

### Which to Use?

**Use VP flows when**:

- Data has specific scale requirements
- Comparing with diffusion baselines
- Working with normalized data (e.g., images in $[-1, 1]$)

**Use NVP flows when**:

- Simplicity is preferred
- Data scale is flexible
- Focusing on rectified flow

---

## Conditional Generation

So far we have described unconditional generation: sampling from $p_1$ without any control over which sample is produced. In practice, most applications require *conditional* generation — images matching a text prompt, protein sequences satisfying a functional constraint, or molecules with a desired property. Flow matching extends to conditioning in a straightforward way, and the same guidance techniques developed for diffusion models apply directly.

### Conditioning Mechanisms

Flow matching naturally supports conditional generation by augmenting the velocity field with a conditioning signal.

**Conditional velocity field**:

$$
v_\theta(x, t, c)
$$

where $c$ is a conditioning variable (e.g., class label, text embedding).

**Training**: Sample $(x_0, c)$ pairs from data, then:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1, c} \left[ \left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|^2 \right]
$$

**Sampling**: Integrate ODE conditioned on $c$:

$$
\frac{dx}{dt} = v_\theta(x(t), t, c)
$$

### Classifier-Free Guidance

Adapt classifier-free guidance from diffusion to flow matching:

**Training**: Randomly drop conditioning (set $c = \emptyset$ with probability $p_{\text{uncond}}$)

**Sampling**: Use guided velocity:

$$
\tilde{v}_\theta(x, t, c) = v_\theta(x, t, \emptyset) + w \cdot (v_\theta(x, t, c) - v_\theta(x, t, \emptyset))
$$

where $w$ is the guidance weight.

**Effect**: Stronger conditioning, sharper samples (at cost of diversity).

---

## Comparison with Diffusion Models

### Conceptual Differences

| Aspect | Diffusion Models | Flow Matching |
|--------|-----------------|---------------|
| **Forward process** | Stochastic noise addition | Deterministic interpolation |
| **What's learned** | Score: $\nabla_x \log p_t(x)$ | Velocity: $v_t(x)$ |
| **Training objective** | Score matching (complex) | Simple regression |
| **Reverse process** | Stochastic SDE or ODE | Deterministic ODE |
| **Sampling** | 100-1000 steps (SDE), 50-100 (ODE) | 10-50 steps (ODE) |
| **Noise schedule** | Critical design choice | Less critical |
| **Theoretical foundation** | Score-based models, SDEs | Optimal transport, ODEs |

### When to Use Each

**Use diffusion when**:

- Mature architectures and tricks are important
- Stochastic sampling is desired
- Extensive baselines exist for comparison
- Working with images (well-established)

**Use flow matching when**:

- Faster sampling is critical
- Simpler training is preferred
- Working with non-Euclidean data
- Exploring new domains (e.g., biology, molecules)

---

## Practical Considerations

The mathematical framework places few constraints on how the velocity network $v_\theta$ is implemented — any architecture that can take $(x, t)$ as input and produce a vector field output is in principle valid. In practice, the architecture choice is guided by the data modality and the importance of incorporating time information effectively.

### Network Architecture

Flow matching networks $v_\theta(x, t)$ typically use:

**For images**:

- **U-Net**: Convolutional architecture with skip connections
- **DiT (Diffusion Transformer)**: Transformer with patch embeddings
- **Time conditioning**: Via sinusoidal embeddings + FiLM layers

**For sequences** (text, DNA, proteins):
- **Transformers**: Self-attention over sequence
- **Time conditioning**: Added to token embeddings

**For graphs** (molecules):
- **GNNs**: Message passing with time conditioning
- **Equivariance**: Preserve symmetries

### Time Embedding

Time $t \in [0, 1]$ is typically embedded via:

**Sinusoidal embedding**:

$$
\text{emb}(t) = [\sin(2\pi k t), \cos(2\pi k t)]_{k=1}^K
$$

**Learned embedding**:

$$
\text{emb}(t) = \text{MLP}(t)
$$

**Conditioning**: Via FiLM (Feature-wise Linear Modulation):

$$
\text{FiLM}(h, t) = \gamma(t) \cdot h + \beta(t)
$$

where $h$ is a hidden representation.

### Training Tips

**1. Noise distribution**: Match data characteristics
- Images: Standard Gaussian
- Gene expression: Consider sparsity structure
- Molecules: Respect physical constraints

**2. Time sampling**: Uniform $t \sim U[0, 1]$ works well, but can weight:
- More weight near $t=0$ (data) for quality
- More weight near $t=1$ (noise) for coverage

**3. Batch size**: Larger is better (more diverse pairs)

**4. Learning rate**: Standard schedules work (cosine, constant)

---

## Summary

### Key Concepts

1. **Flow matching learns velocity fields** that transport noise to data
2. **Training is simple regression** on conditional velocities
3. **Sampling is deterministic ODE integration** (fast, few steps)
4. **Rectified flow uses straight paths** (simplest, most practical)
5. **Reflow iteratively straightens paths** (fewer steps needed)

### Key Equations

**Path** (rectified flow):

$$
x_t = (1-t) x_0 + t x_1
$$

**Velocity**:

$$
u_t = x_1 - x_0
$$

**Loss**:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

**Sampling ODE**:

$$
\frac{dx}{dt} = v_\theta(x(t), t), \quad x(1) \sim p_{\text{noise}}
$$

---

## Related Documents

- [Flow Map and Pushforward](01b_flowmap.md) — Deep dive on the flow map $\phi_t$, pushforward notation, and the ODE → distribution chain
- [Bayes' Theorem with Three Variables](01c_bayes_three_variables.md) — How Bayes is applied to $(x_0, x_1, x_t)$ in the conditional expectation derivation
- [Flow Matching Training](02_flow_matching_training.md) — Training strategies and implementation
- [Flow Matching Sampling](03_flow_matching_sampling.md) — ODE solvers and sampling efficiency
- [Rectifying Flow Tutorial](rectifying_flow.md) — Detailed walkthrough
- [DDPM Foundations](../DDPM/01_ddpm_foundations.md) — Comparison with diffusion
- [Diffusion Transformers](../diffusion/DiT/diffusion_transformer.md) — Architecture for flow matching

---

## References

### Foundational Papers

1. **Lipman, Y., et al. (2023)**. Flow Matching for Generative Modeling. *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

2. **Liu, X., et al. (2023)**. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

3. **Albergo, M. S., & Vanden-Eijnden, E. (2023)**. Building Normalizing Flows with Stochastic Interpolants. *ICLR*. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)

### Optimal Transport

4. **Pooladian, A., et al. (2023)**. Multisample Flow Matching: Straightening Flows with Minibatch Couplings. *ICML*.

5. **Tong, A., et al. (2024)**. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. *TMLR*.

### Applications

6. **Esser, P., et al. (2024)**. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *ICML*.

7. **Pooladian, A., et al. (2024)**. Flow Matching on General Geometries. *ICLR*.
