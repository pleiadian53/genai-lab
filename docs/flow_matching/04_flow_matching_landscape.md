# The Flow Matching Landscape: Methods, Comparisons, and History

This document provides context on the broader landscape of flow-based generative models, comparing normalizing flows, flow matching variants, and providing guidance on which methods to use for different applications.

---

## Overview

The field of flow-based generative models has evolved significantly over the past decade. This document clarifies the relationships between different approaches and helps you choose the right method for your application.

**Key distinction**:

- **Normalizing flows** (2015-2020): Invertible transformations with tractable Jacobians
- **Flow matching** (2022-present): Learned velocity fields via regression

Flow matching has largely superseded normalizing flows for most applications due to simpler training and fewer architectural constraints.

---

## Normalizing Flows

### Core Concept

**Normalizing flows** learn an **invertible transformation** $f_\theta: \mathbb{R}^d \to \mathbb{R}^d$ that maps a simple distribution (noise) to a complex distribution (data):

$$
x_{\text{data}} = f_\theta(z_{\text{noise}})
$$

**Critical requirement**: The transformation must be:
1. **Invertible**: $z = f_\theta^{-1}(x)$ must exist
2. **Tractable Jacobian**: $\det \frac{\partial f_\theta}{\partial z}$ must be computable

### Training Objective

Normalizing flows are trained by **maximizing likelihood** using the change of variables formula:

$$
\log p_\theta(x) = \log p(z) + \log \left| \det \frac{\partial f_\theta}{\partial z} \right|
$$

where $z = f_\theta^{-1}(x)$.

**Training procedure**:
1. Sample data $x \sim p_{\text{data}}$
2. Compute inverse: $z = f_\theta^{-1}(x)$
3. Compute log-likelihood using change of variables
4. Maximize likelihood via gradient ascent

### Architectural Constraints

To ensure invertibility and tractable Jacobians, normalizing flows use specialized architectures:

**1. Coupling Layers** (RealNVP, Glow):
- Split input: $x = [x_a, x_b]$
- Transform one part conditioned on the other:

  $$
  \begin{align}
  y_a &= x_a \\
  y_b &= x_b \odot \exp(s(x_a)) + t(x_a)
  \end{align}
  $$
- Jacobian is triangular (easy to compute)

**2. Autoregressive Flows** (MAF, IAF):
- Transform each dimension conditioned on previous:

  $$

  x_i = z_i \cdot \exp(s_i(z_{<i})) + t_i(z_{<i})
  $$
- Jacobian is triangular

**3. Continuous Normalizing Flows (CNFs)**:

- Define transformation via ODE:

  $$

  \frac{dx}{dt} = f_\theta(x, t)
  $$
- Compute log-likelihood via instantaneous change of variables:

  $$

  \frac{d \log p(x)}{dt} = -\text{Tr}\left(\frac{\partial f_\theta}{\partial x}\right)
  $$

### Major Methods

**RealNVP** (Dinh et al., 2017):
- Coupling layers with affine transformations
- Simple, stable training
- Limited expressiveness

**Glow** (Kingma & Dhariwal, 2018):
- Adds invertible 1×1 convolutions
- Activation normalization (ActNorm)
- Better for high-resolution images

**Neural Spline Flows** (Durkan et al., 2019):
- Monotonic spline transformations
- More expressive than affine
- Still tractable Jacobian

**FFJORD** (Grathwohl et al., 2019):
- Continuous normalizing flow
- Uses ODE solver + trace estimator
- Expensive but flexible

### Advantages

✅ **Single-step sampling**: $x = f_\theta(z)$ (no iterative process)
✅ **Exact likelihood**: Can compute $p(x)$ exactly
✅ **Bidirectional**: Can go noise → data and data → noise
✅ **Theoretical elegance**: Clean probabilistic interpretation

### Limitations

❌ **Architectural constraints**: Must design invertible networks
❌ **Jacobian computation**: Expensive for high dimensions
❌ **Training complexity**: Likelihood-based training can be unstable
❌ **Limited expressiveness**: Invertibility restricts model capacity
❌ **Scaling issues**: Difficult to scale to very high dimensions

---

## Flow Matching

### Core Concept

**Flow matching** learns a **velocity field** $v_\theta(x, t)$ that defines how to transport samples from noise to data:

$$
\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]
$$

**Key difference**: No invertibility requirement, no Jacobian computation.

### Training Objective

Flow matching uses **simple regression** on conditional velocities:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - u_t(x_0, x_1) \right\|^2 \right]
$$

where:

- $x_0 \sim p_{\text{data}}$ (data)
- $x_1 \sim p_{\text{noise}}$ (noise)
- $x_t = \psi_t(x_0, x_1)$ (interpolated point)
- $u_t(x_0, x_1) = \frac{d}{dt}\psi_t(x_0, x_1)$ (target velocity)

**Training procedure**:
1. Sample data $x_0$ and noise $x_1$
2. Sample time $t \sim U[0, 1]$
3. Compute interpolated point $x_t$
4. Compute target velocity $u_t$
5. Predict velocity and minimize MSE

### No Architectural Constraints

**Any neural network** can be used for $v_\theta(x, t)$:
- U-Net for images
- Transformer for sequences
- GNN for graphs
- MLP for low-dimensional data

No need for invertibility or special Jacobian structures.

### Advantages over Normalizing Flows

✅ **Simpler training**: Direct regression (no likelihood computation)
✅ **No constraints**: Any architecture works
✅ **Faster training**: No Jacobian determinant computation
✅ **More flexible**: Not restricted to invertible transformations
✅ **Better scaling**: Easier to scale to high dimensions
✅ **Comparable quality**: Matches or exceeds normalizing flows

### Trade-off

❌ **Multi-step sampling**: Requires ODE solver (10-50 steps)
❌ **No exact likelihood**: Cannot compute $p(x)$ exactly (but rarely needed)

---

## Flow Matching Variants

The flow matching framework has spawned several variants, each with different trade-offs.

### 1. Rectified Flow

**Paper**: Liu et al. (2023) - "Flow Straight and Fast"

**Key idea**: Use the simplest possible path—linear interpolation.

**Path**:

$$
x_t = (1-t) x_0 + t x_1
$$

**Velocity**:

$$

u_t = x_1 - x_0 \quad \text{(constant)}
$$

**Loss**:

$$

\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

**Reflow**: Iteratively straighten paths by training on synthetic data:
- Iteration 1: Train on real data
- Iteration 2: Generate synthetic data, train new model
- Iteration 3+: Repeat

**Effect**: Each reflow iteration reduces required sampling steps:
- Base model: 50-100 steps
- After 1 reflow: 20-30 steps
- After 2 reflows: 10-15 steps
- After 3 reflows: 5-10 steps

**Advantages**:

- ⭐⭐⭐⭐⭐ Simplest to implement
- ⭐⭐⭐⭐⭐ Fastest training
- ⭐⭐⭐⭐ Good sample quality
- ⭐⭐⭐⭐ Fast sampling (after reflow)

**When to use**: Default choice for most applications.

### 2. Flow Matching (General)

**Paper**: Lipman et al. (2023) - "Flow Matching for Generative Modeling"

**Key idea**: General framework allowing flexible path choices.

**Path**: Any differentiable $x_t = \psi_t(x_0, x_1)$

**Velocity**: $u_t = \frac{d}{dt}\psi_t(x_0, x_1)$

**Examples**:

- Linear: $x_t = (1-t)x_0 + tx_1$ (rectified flow)
- Variance-preserving: $x_t = \sqrt{1-\sigma_t^2} x_0 + \sigma_t x_1$
- Geodesic: $x_t = \exp_{x_0}(t \log_{x_0}(x_1))$ (for manifolds)

**Advantages**:

- ⭐⭐⭐⭐⭐ Theoretical foundation
- ⭐⭐⭐⭐⭐ Flexibility
- ⭐⭐⭐⭐ Sample quality

**When to use**: When you need custom path designs (e.g., manifold-valued data).

### 3. Stochastic Interpolants

**Paper**: Albergo & Vanden-Eijnden (2023) - "Building Normalizing Flows with Stochastic Interpolants"

**Key idea**: Generalize to include stochasticity in the path.

**Path**: Can include Brownian motion:

$$
x_t = \alpha_t x_0 + \beta_t x_1 + \gamma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Advantage**: Bridges flow matching and diffusion models.

**When to use**: When you want to interpolate between deterministic and stochastic generation.

### 4. Optimal Transport Flow Matching

**Paper**: Tong et al. (2024) - "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"

**Key idea**: Use optimal transport to find better data-noise couplings.

**Approach**:
1. Solve minibatch OT problem:

   $$

   \min_{\pi} \sum_{i,j} \pi_{ij} \|x_0^{(i)} - x_1^{(j)}\|^2
   $$

2. Use OT couplings for training instead of independent pairing

**Effect**: Straighter paths, fewer sampling steps needed.

**Advantages**:

- ⭐⭐⭐⭐⭐ Better sample quality
- ⭐⭐⭐⭐⭐ Straighter paths
- ⭐⭐⭐⭐⭐ Fewer sampling steps

**Trade-off**:

- ⭐⭐⭐ Slower training (OT computation)
- ⭐⭐⭐ More complex implementation

**When to use**: When sample quality is critical and you have computational budget.

### 5. Multisample Flow Matching

**Paper**: Pooladian et al. (2023) - "Multisample Flow Matching: Straightening Flows with Minibatch Couplings"

**Key idea**: Use multiple samples to learn better couplings without explicit OT.

**Approach**: Instead of pairing $x_0^{(i)}$ with $x_1^{(i)}$, use minibatch to find better pairings.

**Advantages**:

- ⭐⭐⭐⭐⭐ Better quality than rectified flow
- ⭐⭐⭐⭐ Faster than OT flow matching
- ⭐⭐⭐⭐ Straighter paths without reflow

**When to use**: When you want better quality without reflow iterations.

---

## Detailed Comparison

### Normalizing Flows vs Flow Matching

| Aspect | Normalizing Flows | Flow Matching |
|--------|-------------------|---------------|
| **What's learned** | Invertible transformation $f_\theta$ | Velocity field $v_\theta(x, t)$ |
| **Training objective** | Maximize likelihood | Minimize MSE |
| **Architectural constraints** | Must be invertible | None |
| **Jacobian computation** | Required | Not required |
| **Training complexity** | High (likelihood) | Low (regression) |
| **Sampling** | 1 step | 10-50 steps (ODE) |
| **Exact likelihood** | Yes | No |
| **Expressiveness** | Limited by invertibility | High |
| **Scaling** | Difficult | Easy |
| **Sample quality** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Flow Matching Variants Comparison

| Method | Training Speed | Sample Quality | Sampling Steps | Implementation |
|--------|---------------|----------------|----------------|----------------|
| **Rectified Flow** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 20-50 (10-15 after reflow) | Easy |
| **OT Flow Matching** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 10-20 | Moderate |
| **Multisample FM** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 10-20 | Moderate |
| **Stochastic Interpolants** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 20-50 | Moderate |

### Flow Matching vs Diffusion Models

| Aspect | Diffusion (DDPM) | Diffusion (DDIM) | Flow Matching |
|--------|------------------|------------------|---------------|
| **Training** | Score matching | Score matching | Simple regression |
| **Forward process** | Stochastic | Stochastic | Deterministic |
| **Reverse process** | Stochastic SDE | Deterministic ODE | Deterministic ODE |
| **Sampling steps** | 1000 | 50-100 | 10-50 |
| **Speed** | Slowest | Fast | **Fastest** |
| **Determinism** | No | Yes | Yes |
| **Maturity** | High | High | Growing |

---

## Historical Evolution

### Timeline

**2015-2016: Early Normalizing Flows**

- NICE (Dinh et al., 2015)
- RealNVP (Dinh et al., 2017)
- Focus: Invertible transformations

**2017-2019: Refinements**

- Glow (Kingma & Dhariwal, 2018)
- Neural Spline Flows (Durkan et al., 2019)
- FFJORD (Grathwohl et al., 2019) - Continuous normalizing flows
- Focus: Better architectures, continuous time

**2020-2021: Diffusion Dominance**

- DDPM (Ho et al., 2020)
- Score-based models (Song et al., 2021)
- Normalizing flows fade in popularity
- Focus: Sample quality over speed

**2022: Flow Matching Emerges**

- Flow Matching (Lipman et al., 2022)
- Stochastic Interpolants (Albergo & Vanden-Eijnden, 2022)
- Focus: Simpler training than diffusion

**2023: Rectified Flow & Variants**

- Rectified Flow (Liu et al., 2023)
- Multisample Flow Matching (Pooladian et al., 2023)
- OT Flow Matching (Tong et al., 2023)
- Focus: Faster sampling, straighter paths

**2024-Present: Maturation**

- Integration with Transformers (DiT)
- Applications to video, 3D, biology
- Focus: Scaling and applications

### Key Insights from History

**Why normalizing flows declined**:
1. Architectural constraints limited expressiveness
2. Diffusion models achieved better quality
3. Training was more complex than alternatives

**Why flow matching succeeded**:
1. Learned from normalizing flows (ODE-based)
2. Removed constraints (no invertibility)
3. Simpler than diffusion (regression vs score matching)
4. Faster than diffusion (fewer sampling steps)

**Current state**: Flow matching is the **modern successor** to normalizing flows, combining their speed advantages with diffusion-level quality.

---

## Choosing the Right Method

### Decision Tree

```
Do you need exact likelihood computation?
├─ Yes → Normalizing Flows (RealNVP, Glow)
└─ No → Continue

Do you need single-step sampling?
├─ Yes → Normalizing Flows
└─ No → Continue

Are you exploring a new domain?
├─ Yes → Start with Rectified Flow (simplest)
└─ No → Continue

Is sample quality critical?
├─ Yes → OT Flow Matching or Multisample FM
└─ No → Rectified Flow

Do you have computational budget for reflow?
├─ Yes → Rectified Flow + Reflow (fastest sampling)
└─ No → Multisample FM (good without reflow)
```

### Recommendations by Application

**For computational biology (gene expression, molecules)**:

- **Start**: Rectified Flow
- **Upgrade**: Multisample FM if quality needs improvement
- **Avoid**: Normalizing flows (too constrained)

**For images (high resolution)**:

- **Start**: Rectified Flow + DiT architecture
- **Upgrade**: OT Flow Matching for best quality
- **Consider**: Reflow for production (5-10 steps)

**For sequences (text, DNA, proteins)**:

- **Start**: Rectified Flow + Transformer
- **Upgrade**: Multisample FM
- **Consider**: Stochastic Interpolants for flexibility

**For low-dimensional data (<100D)**:

- **Consider**: Normalizing Flows (single-step sampling valuable)
- **Alternative**: Rectified Flow (if multi-step OK)

**For research/exploration**:

- **Start**: Rectified Flow (fastest iteration)
- **Experiment**: Try variants once baseline established

---

## Practical Guidelines

### For Your GenAI Lab Project

**Phase 1: Establish Baselines**
1. Implement **Rectified Flow** (simplest)
2. Implement **DDPM** (already documented)
3. Compare on gene expression data

**Phase 2: Optimize**
1. Apply **Reflow** to rectified flow (1-2 iterations)
2. Compare sampling steps: DDPM (100) vs RF (20) vs Reflow (10)
3. Evaluate quality with your metrics (FID, prediction consistency, epiplexity)

**Phase 3: Advanced (if needed)**
1. Try **Multisample Flow Matching** if quality insufficient
2. Experiment with **OT Flow Matching** if computational budget allows
3. Compare with VAE, JEPA, other methods

**Skip**:

- Normalizing flows (superseded by flow matching)
- Stochastic interpolants (unless you need stochastic generation)

### Implementation Priority

**High priority**:
1. ✅ Rectified Flow (already documented)
2. ⏳ DDPM experiments (next step)
3. ⏳ Reflow (after baseline established)

**Medium priority**:
4. Multisample Flow Matching (if quality needs improvement)
5. DiT architecture (for scaling)
6. Latent diffusion (for high-dimensional data)

**Low priority**:
7. OT Flow Matching (expensive, marginal improvement)
8. Normalizing flows (outdated)
9. Stochastic interpolants (niche use case)

---

## Summary

### Key Takeaways

**Normalizing Flows**:

- Older approach (2015-2020)
- Invertible transformations with tractable Jacobians
- Single-step sampling but limited expressiveness
- Largely superseded by flow matching

**Flow Matching**:

- Modern approach (2022-present)
- Learns velocity fields via simple regression
- Multi-step sampling but high quality
- Current state-of-the-art for flow-based models

**Rectified Flow**:

- Simplest flow matching variant
- Linear interpolation paths
- Best starting point for most applications
- Can be improved via reflow

**Advanced Variants**:

- OT Flow Matching: Best quality, expensive
- Multisample FM: Good quality, moderate cost
- Stochastic Interpolants: Flexible, niche

### Recommendation

**For your combio project**: Start with **Rectified Flow**. It's the sweet spot of simplicity, performance, and flexibility. You can always upgrade to advanced variants if needed, but rectified flow will give you a strong baseline quickly.

---

## Related Documents

- [Flow Matching Foundations](01_flow_matching_foundations.md) — Mathematical theory
- [Flow Matching Training](02_flow_matching_training.md) — Implementation guide
- [Flow Matching Sampling](03_flow_matching_sampling.md) — ODE solvers and sampling
- [Rectifying Flow Tutorial](rectifying_flow.md) — Intuitive introduction
- [DDPM Documentation](../DDPM/README.md) — Comparison with diffusion

---

## References

### Normalizing Flows

1. **Dinh, L., et al. (2017)**. Density estimation using Real NVP. *ICLR*. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)

2. **Kingma, D. P., & Dhariwal, P. (2018)**. Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*. [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)

3. **Durkan, C., et al. (2019)**. Neural Spline Flows. *NeurIPS*. [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)

4. **Grathwohl, W., et al. (2019)**. FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)

### Flow Matching

5. **Lipman, Y., et al. (2023)**. Flow Matching for Generative Modeling. *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

6. **Liu, X., et al. (2023)**. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

7. **Albergo, M. S., & Vanden-Eijnden, E. (2023)**. Building Normalizing Flows with Stochastic Interpolants. *ICLR*. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)

### Advanced Variants

8. **Pooladian, A., et al. (2023)**. Multisample Flow Matching: Straightening Flows with Minibatch Couplings. *ICML*. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)

9. **Tong, A., et al. (2024)**. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. *TMLR*. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)

### Reviews

10. **Papamakarios, G., et al. (2021)**. Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*. [arXiv:1912.02762](https://arxiv.org/abs/1912.02762)
