# Flow Matching Foundations

Flow matching is a family of generative modeling methods that learn to transport samples from a simple noise distribution to a complex data distribution via continuous-time flows. Unlike diffusion models that learn score functions, flow matching directly learns velocity fields through simple regression.

This document covers the mathematical foundations, forward and backward processes, and theoretical properties of flow matching.

---

## Overview

### The Core Idea

**Goal**: Transform samples from a simple distribution $p_0$ (noise) into samples from a complex distribution $p_1$ (data).

**Approach**: Learn a time-dependent velocity field $v_\theta(x, t)$ that defines a continuous transformation:

$$
\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]
$$

**Key insight**: Instead of learning probability gradients (scores), we directly learn how points should move through space.

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

### Probability Flows

A **probability flow** is a time-dependent vector field that transports probability mass.

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

This ensures probability mass is conserved as it flows.

### The Forward Process

Unlike diffusion models with stochastic forward processes, flow matching uses **deterministic interpolation**.

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

The **marginal velocity field** at time $t$ is:

$$
v_t(x) = \mathbb{E}_{x_0, x_1 | x_t = x} [u_t(x_0, x_1)]
$$

This is the expected velocity at position $x$ and time $t$, averaged over all data-noise pairs that pass through $x$ at time $t$.

**Key property**: If we know $v_t(x)$ exactly, solving the ODE:

$$
\frac{dx}{dt} = v_t(x), \quad x(0) = x_0 \sim p_{\text{noise}}
$$

will transport $x(0)$ to $x(1) \sim p_{\text{data}}$.

---

## Flow Matching Objective

### The Training Loss

Flow matching trains a neural network $v_\theta(x, t)$ to approximate the marginal velocity field $v_t(x)$.

**Conditional Flow Matching (CFM) loss**:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1, x_t} \left[ \left\| v_\theta(x_t, t) - u_t(x_0, x_1) \right\|^2 \right]
$$

where:

- $t \sim \text{Uniform}[0, 1]$
- $x_0 \sim p_{\text{data}}$
- $x_1 \sim p_{\text{noise}}$
- $x_t = \psi_t(x_0, x_1)$

**Why this works**: The conditional velocity $u_t(x_0, x_1)$ is a valid target because:

$$
\mathbb{E}_{x_0, x_1 | x_t} [u_t(x_0, x_1)] = v_t(x_t)
$$

This is the **conditional expectation property** that makes flow matching tractable.

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

### ODE Integration

After training, we generate samples by solving the **flow ODE** backward in time.

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

**Intuition**:

- Optimal transport between distributions typically follows curved paths (geodesics in probability space)
- Rectified flow **straightens** these paths into lines
- The neural network compensates for any distortion from this simplification

**Advantage**: Straight paths are:
- Simple to define
- Easy to sample (fewer ODE steps)
- Numerically stable

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

### Optimality

**Theorem** (Lipman et al., 2023): The flow matching objective is equivalent to minimizing:

$$
\mathbb{E}_t \left[ \mathbb{E}_{x \sim p_t} \left[ \|v_\theta(x, t) - v_t(x)\|^2 \right] \right]
$$

This is the $L^2$ distance between the learned and true velocity fields.

**Consequence**: Flow matching directly optimizes the sampling quality.

### Connection to Optimal Transport

Flow matching is related to **optimal transport** (OT):

**Optimal transport problem**: Find the transport map $T: p_0 \to p_1$ that minimizes:

$$
\min_T \mathbb{E}_{x_0 \sim p_0} \left[ c(x_0, T(x_0)) \right]
$$

where $c(x, y)$ is a cost function (e.g., $\|x - y\|^2$).

**Connection**:

- Rectified flow with linear paths approximates OT with quadratic cost
- The learned velocity field defines a transport map
- Reflow iterations improve the OT approximation

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

### Conditioning Mechanisms

Flow matching naturally supports conditional generation.

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
