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

Think of $p_t(x)$ as the density of an incompressible fluid at position $x$ and time $t$, and $v(x, t)$ as the local fluid velocity. The product $p_t(x)\, v(x, t)$ is then the *probability flux* — the rate at which probability mass flows past a given point per unit area. The divergence $\nabla \cdot (p_t v)$ measures how much of that flux is *leaving* a small region per unit volume:

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

The continuity equation reveals the deep link between particle trajectories and distribution evolution. If every particle starting at $x(0)$ follows the ODE $\frac{dx}{dt} = v(x, t)$, it traces a path $x(t) = \phi_t(x(0))$, where $\phi_t$ is called the **flow map**. The distribution evolves as the *pushforward* of $p_0$ under this map:

$$
p_t = (\phi_t)_\# \, p_0
$$

In words: to find $p_t$, take every sample from $p_0$ and push it forward along the flow. The continuity equation is simply the differential form of this statement. The goal of flow matching is to learn a velocity field $v_\theta$ whose induced flow map $\phi_1$ transports $p_{\text{noise}}$ to $p_{\text{data}}$.

### The Forward Process

Unlike diffusion models, which define the forward process by *adding noise* stochastically (making it harder to reason about the exact trajectory), flow matching uses **deterministic interpolation**. We simply connect each data point $x_0$ to a noise point $x_1$ with a smooth path, then train the model to follow that path.

This design choice is deliberate: by decoupling path construction from learning, we can analyze trajectories analytically and choose path families that are convenient for training (e.g., straight lines) or that satisfy constraints (e.g., variance preservation). The model does not need to know which pair $(x_0, x_1)$ generated a given $x_t$ — it only needs to know the *velocity* at that $x_t$.

**Conditional flow**: Given a data-noise pair $(x_0, x_1)$ where $x_0 \sim p_{\text{data}}$ and $x_1 \sim p_{\text{noise}}$, define a path:

$$
x_t = \psi_t(x_0, x_1)
$$

**Common choices**:

**1. Linear interpolation (Rectified Flow)**:

$$
x_t = (1-t) x_0 + t x_1
$$

**2. Geodesic interpolation**:

$$
x_t = \exp_{x_0}(t \log_{x_0}(x_1))
$$

(useful for manifold-valued data)

**3. Variance-preserving interpolation**:

$$
x_t = \sqrt{1-t} \, x_0 + \sqrt{t} \, x_1
$$

(maintains variance like diffusion)

**Conditional velocity**: The velocity along this path is:

$$
u_t(x_0, x_1) = \frac{d}{dt} \psi_t(x_0, x_1)
$$

For linear interpolation: $u_t(x_0, x_1) = x_1 - x_0$ (constant velocity).

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

**Why this works**: Although we train on conditional velocities $u_t(x_0, x_1)$ for specific pairs, the loss is equivalent to matching the marginal field. This follows from the **conditional expectation property**:

$$
\mathbb{E}_{x_0, x_1 \mid x_t} [u_t(x_0, x_1)] = v_t(x_t)
$$

The conditional velocity is an *unbiased estimator* of the marginal velocity at $x_t$. Minimizing the MSE loss in expectation over $(x_0, x_1, t)$ therefore trains the network toward the true marginal field — without ever computing it directly. This is analogous to how stochastic gradient descent minimizes a full-batch loss using minibatch gradients. The tractability of flow matching hinges entirely on this property.

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

**Key properties**:

- **Deterministic**: Same noise input always produces same output
- **Reversible**: Can go forward and backward
- **Continuous**: Smooth trajectories through state space

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
