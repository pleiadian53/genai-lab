# Geodesic Interpolation in Flow Matching

Flow matching transforms the generative modeling paradigm from "adding noise and learning to denoise" to "walking deterministic paths through probability space." At the heart of this shift lies a fundamental question: **what kind of path should connect our noise distribution to our data distribution?**

This document explores the concept of **geodesic interpolation** — a geometric perspective on choosing paths that respect the structure of your data. We'll build intuition from the ground up, connect to practical implementations, and explore why this matters for biological and manifold-valued data.

---

## Table of Contents

1. [From Stochastic Diffusion to Deterministic Flows](#from-stochastic-diffusion-to-deterministic-flows)
2. [What is a Geodesic?](#what-is-a-geodesic)
3. [Linear Interpolation: The Euclidean Geodesic](#linear-interpolation-the-euclidean-geodesic)
4. [Beyond Flat Spaces: True Geodesics on Manifolds](#beyond-flat-spaces-true-geodesics-on-manifolds)
5. [Geodesics in Probability Space](#geodesics-in-probability-space)
6. [Practical Implementations](#practical-implementations)
7. [Why Geodesics Matter for Biological Data](#why-geodesics-matter-for-biological-data)
8. [The Unspoken Dream: Learned Geometry](#the-unspoken-dream-learned-geometry)
9. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## From Stochastic Diffusion to Deterministic Flows

### The Diffusion Paradigm

In diffusion models, the forward process is inherently stochastic. Given a data point $x_0$, we gradually corrupt it with Gaussian noise:

$$
x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1-\alpha_t} \, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

Each data point "fans out" into a probabilistic cloud. The reverse model must learn to navigate backward through this noisy landscape, swimming upstream against entropy.

**Characteristics of diffusion**:
- Stochastic forward process (every trajectory is different)
- Score-based reverse process (learning gradients of log-probabilities)
- Thermodynamic interpretation (gradual equilibration)
- Requires many steps (100-1000) for high-quality sampling

### The Flow Matching Paradigm

Flow matching takes a fundamentally different approach. The forward process is **deterministic**:

$$
x_t = \psi_t(x_0, x_1)
$$

where:
- $x_0 \sim p_{\text{data}}$ (data distribution)
- $x_1 \sim p_{\text{noise}}$ (base distribution, e.g., Gaussian)
- $t \in [0, 1]$ (time parameter)
- $\psi_t$ is an **interpolation rule**

**Key insight**: No noise, no branching, no stochasticity. Just a deterministic path connecting two points.

**Characteristics of flow matching**:
- Deterministic forward process (repeatable trajectories)
- Velocity-based reverse process (learning direction of motion)
- Geometric interpretation (paths through space)
- Requires fewer steps (10-50) for high-quality sampling

### The Central Question

Once we commit to deterministic paths, we face an immediate design choice:

> **What kind of path should connect two probability distributions?**

This is where **geodesics** enter the picture — not necessarily as strict geometric objects, but as a guiding principle for choosing "natural" paths.

---

## What is a Geodesic?

### Geometric Definition

In differential geometry, a **geodesic** is the shortest (or extremal) path between two points given a notion of distance.

**Familiar examples**:

| Space | Geodesic | Visualization |
|-------|----------|---------------|
| **Flat plane** ($\mathbb{R}^2$) | Straight lines | The shortest distance between two points |
| **Sphere** ($S^2$) | Great circles | Flight paths on Earth |
| **Curved surface** | Paths that "feel straight" locally | Walking on a saddle |

### Mathematical Formulation

For a space equipped with a metric $g$ (which defines distances and angles), a geodesic $\gamma(t)$ satisfies:

**Variational principle**: Minimizes arc length

$$
\text{minimize} \quad \int_0^1 \sqrt{\dot{\gamma}(t)^\top g(\gamma(t)) \dot{\gamma}(t)} \, dt
$$

**Differential equation**: Parallel transport of velocity

$$
\nabla_{\dot{\gamma}} \dot{\gamma} = 0
$$

This is the **geodesic equation** — a second-order ODE that says "don't turn unless the space curves."

**Intuition**: A geodesic is a path where your velocity vector doesn't change direction unless forced by the geometry of the space itself.

### Geodesics in Flow Matching Context

Here's the crucial distinction: **Flow matching typically does not solve true geodesic equations**.

Instead, it chooses **proxy paths** that behave "as if" they were geodesics under some implicit geometry.

When we say "geodesic interpolation" in flow matching, we really mean:

> "A natural, structure-respecting interpolation that captures the essence of shortest paths"

The question is: shortest with respect to **what metric**?

---

## Linear Interpolation: The Euclidean Geodesic

### The Standard Choice

The most common interpolation in flow matching is **linear interpolation**:

$$
x_t = (1 - t) x_0 + t x_1
$$

This interpolates linearly between data $x_0$ and noise $x_1$.

**Velocity field**: Taking the derivative with respect to $t$:

$$
\frac{dx_t}{dt} = x_1 - x_0
$$

Notice that the velocity is **constant** — it doesn't depend on time or position. This is the hallmark of straight-line motion.

### Why is This a Geodesic?

Linear interpolation is a **true geodesic** when:
1. The space is $\mathbb{R}^d$ (Euclidean space)
2. The metric is the standard Euclidean metric $g = I$
3. The data "lives" in flat space (or we pretend it does)

**Geometric interpretation**:
- In flat Euclidean space, straight lines are the shortest paths
- The geodesic equation $\nabla_{\dot{x}} \dot{x} = 0$ is automatically satisfied
- No "turning" occurs because space has no curvature

### Visual Example: Image Interpolation

Consider interpolating between two images in pixel space:

```
Image 0: Cat photo        Image 1: Random noise
   ↓                            ↓
   x_0 ∈ ℝ^(3×256×256)         x_1 ∈ ℝ^(3×256×256)
```

Linear path at $t = 0.5$:

$$
x_{0.5} = 0.5 \cdot x_{\text{cat}} + 0.5 \cdot x_{\text{noise}}
$$

**Result**: A "ghost cat" — 50% cat structure, 50% noise. The path walks straight through pixel space.

**Assumption**: We're treating pixel space as flat Euclidean space, which may or may not be appropriate.

### The Flow Matching Loss

Flow matching trains a neural network $v_\theta(x, t)$ to match this constant velocity:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

**Training procedure**:
1. Sample data point $x_0$ and noise point $x_1$
2. Sample random time $t \sim \text{Uniform}[0, 1]$
3. Compute interpolated point $x_t = (1-t) x_0 + t x_1$
4. Compute target velocity $u = x_1 - x_0$
5. Train network to predict this velocity: $v_\theta(x_t, t) \approx u$

This is beautifully simple — just regression toward a target velocity.

### When is Linear Interpolation Appropriate?

**Linear interpolation works well when**:
- Data lives in latent spaces (e.g., VAE latents, embeddings)
- Scale is normalized (e.g., images in $[-1, 1]$)
- Euclidean distances are meaningful
- Simplicity and speed are priorities

**Linear interpolation may struggle when**:
- Data lives on manifolds (e.g., spheres, rotations)
- Euclidean distance distorts structure
- Interpolated points leave the valid data space

---

## Beyond Flat Spaces: True Geodesics on Manifolds

### The Manifold Challenge

Many real-world data types naturally live on **manifolds** — geometric spaces that are locally Euclidean but globally curved.

**Examples in science**:

| Domain | Manifold | Why? |
|--------|----------|------|
| **Directional data** | Sphere $S^{n-1}$ | Unit vectors (e.g., gene expression directions) |
| **Rotations** | $SO(3)$ | 3D molecular orientations |
| **Positive matrices** | $SPD(n)$ | Covariance matrices, diffusion tensors |
| **Probability simplices** | $\Delta^{n-1}$ | Compositional data (e.g., cell type proportions) |
| **Latent spaces** | Implicit manifold | Learned data structure |

### Why Linear Interpolation Fails on Manifolds

**Problem**: Linear interpolation can leave the manifold!

**Example: Unit sphere**

Consider two points on the unit sphere: $x_0, x_1 \in S^2$ with $\|x_0\| = \|x_1\| = 1$.

Linear interpolation:

$$
x_t = (1-t) x_0 + t x_1
$$

**Issue**: Generally, $\|x_t\| \neq 1$. The interpolated point "cuts through" the sphere rather than walking along its surface.

```
      x₁
      /|\
     / | \
    /  |  \     Linear path cuts
   /   |   \    through interior!
  /    |    \
 /     ●     \   ← x_t (inside sphere)
x₀-----------x₁
      
Geodesic path
curves along surface
```

**Consequence**: 
- Interpolated points may be invalid (e.g., not unit norm)
- Structure is not preserved during interpolation
- Learned velocity fields may struggle to stay on manifold

### Spherical Linear Interpolation (SLERP)

For points on a sphere, the true geodesic is a **great circle arc**. This is computed via **SLERP**:

$$
x_t = \frac{\sin((1-t)\theta)}{\sin(\theta)} x_0 + \frac{\sin(t\theta)}{\sin(\theta)} x_1
$$

where $\theta = \arccos(x_0^\top x_1)$ is the angle between the points.

**Properties**:
- $\|x_t\| = 1$ for all $t$ (stays on sphere)
- Constant angular velocity
- Shortest path on the sphere

**Velocity field**: More complex than linear case:

$$
\frac{dx_t}{dt} = \frac{\theta}{\sin(\theta)} \left[ \cos(t\theta) x_1 - \cos((1-t)\theta) x_0 \right]
$$

### General Manifold Geodesics: Exponential Map

For arbitrary manifolds, geodesics are computed using the **exponential map** and **logarithmic map**.

**Exponential map** $\exp_x: T_x\mathcal{M} \to \mathcal{M}$:
- Starts at point $x$ on the manifold
- Takes a tangent vector $v$ (direction and magnitude)
- Shoots a geodesic in that direction for "time" 1
- Returns the endpoint on the manifold

**Logarithmic map** $\log_x: \mathcal{M} \to T_x\mathcal{M}$:
- Inverse of exponential map
- Finds the tangent vector that would geodesically connect $x$ to another point $y$

**Geodesic interpolation**:

$$
x_t = \exp_{x_0}(t \cdot \log_{x_0}(x_1))
$$

**Interpretation**:
1. Compute tangent vector from $x_0$ pointing toward $x_1$: $v = \log_{x_0}(x_1)$
2. Scale by $t$ to get partial vector: $t \cdot v$
3. Shoot geodesic from $x_0$ along $t \cdot v$: $x_t = \exp_{x_0}(t \cdot v)$

**Properties**:
- $x_t$ stays on the manifold for all $t$
- Shortest path with respect to manifold metric
- Respects geometric structure

**Velocity field**: Via pushforward of exponential map (requires manifold-specific computation).

### Practical Example: Spherical Data

**Setup**: Gene expression directions normalized to unit sphere.

**Linear interpolation** (WRONG):
```python
def linear_interp(x0, x1, t):
    xt = (1 - t) * x0 + t * x1
    # Issue: ||xt|| ≠ 1
    return xt
```

**SLERP** (CORRECT):
```python
def slerp(x0, x1, t):
    theta = torch.acos((x0 * x1).sum(-1))
    xt = (torch.sin((1-t)*theta)/torch.sin(theta))[:, None] * x0 \
       + (torch.sin(t*theta)/torch.sin(theta))[:, None] * x1
    # Guarantee: ||xt|| = 1
    return xt
```

**Result**: Interpolation respects the spherical structure of the data.

---

## Geodesics in Probability Space

### Wasserstein Geodesics: Optimal Transport

There's another sense in which "geodesic" appears in flow matching: **geodesics in the space of probability distributions**.

### The Wasserstein Distance

The **Wasserstein-2 distance** between distributions $p_0$ and $p_1$ is:

$$
W_2(p_0, p_1) = \inf_{T: T_\# p_0 = p_1} \mathbb{E}_{x \sim p_0}[\|x - T(x)\|^2]^{1/2}
$$

where $T$ is a transport map pushing $p_0$ to $p_1$.

**Interpretation**: The minimum cost of transporting probability mass from $p_0$ to $p_1$, where cost is quadratic distance.

### Displacement Interpolation

Given the **optimal transport map** $T^*: p_0 \to p_1$, the **displacement interpolation** is:

$$
x_t = (1-t) x_0 + t \cdot T^*(x_0)
$$

This defines a path in **Wasserstein space** — the space of probability distributions equipped with the Wasserstein metric.

**Key property**: This path is a **geodesic in Wasserstein space**, even though individual points follow straight lines in $\mathbb{R}^d$.

**Geometric hierarchy**:
- **Pointwise**: Linear paths in $\mathbb{R}^d$
- **Distribution-level**: Geodesic in Wasserstein space

### Connection to Flow Matching

Standard flow matching with linear interpolation **approximates** but doesn't exactly match optimal transport paths:

$$
x_t = (1-t) x_0 + t x_1, \quad x_0 \sim p_{\text{data}}, \, x_1 \sim p_{\text{noise}}
$$

**Issue**: Pairing $(x_0, x_1)$ is typically random (independent sampling), not optimal.

**Recent advances**: Methods like **Multisample Flow Matching** and **OT-CFM** use minibatch optimal transport to find better pairings, improving the OT approximation.

**Result**: The learned flow becomes closer to a true Wasserstein geodesic.

---

## Practical Implementations

### Implementation 1: Linear Interpolation (Standard)

**Path**:

$$
x_t = (1-t) x_0 + t x_1
$$

**Velocity**:

$$
u_t = x_1 - x_0
$$

**Code**:
```python
def linear_interpolation(x0, x1, t):
    """Linear geodesic (Euclidean)."""
    return (1 - t) * x0 + t * x1

def linear_velocity(x0, x1, t):
    """Constant velocity."""
    return x1 - x0
```

**Use when**: Data in Euclidean space, latent spaces, embeddings.

### Implementation 2: Spherical Interpolation (SLERP)

**Path**:

$$
x_t = \frac{\sin((1-t)\theta)}{\sin(\theta)} x_0 + \frac{\sin(t\theta)}{\sin(\theta)} x_1
$$

where $\theta = \arccos(x_0^\top x_1)$.

**Code**:
```python
def slerp(x0, x1, t, eps=1e-7):
    """Spherical linear interpolation (geodesic on sphere)."""
    # Normalize to ensure unit norm
    x0 = F.normalize(x0, dim=-1)
    x1 = F.normalize(x1, dim=-1)
    
    # Compute angle
    dot = (x0 * x1).sum(-1, keepdim=True)
    theta = torch.acos(dot.clamp(-1, 1))
    
    # Handle near-parallel vectors
    sin_theta = torch.sin(theta)
    mask = sin_theta.abs() < eps
    
    # SLERP formula
    w0 = torch.sin((1 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)
    
    # Fall back to linear for near-parallel
    w0 = torch.where(mask, 1 - t, w0)
    w1 = torch.where(mask, t, w1)
    
    return w0 * x0 + w1 * x1

def slerp_velocity(x0, x1, t, eps=1e-7):
    """Velocity field for SLERP."""
    x0 = F.normalize(x0, dim=-1)
    x1 = F.normalize(x1, dim=-1)
    
    dot = (x0 * x1).sum(-1, keepdim=True)
    theta = torch.acos(dot.clamp(-1, 1))
    sin_theta = torch.sin(theta)
    
    # d/dt [sin((1-t)θ)/sin(θ) x₀ + sin(tθ)/sin(θ) x₁]
    coeff0 = -theta * torch.cos((1-t) * theta) / (sin_theta + eps)
    coeff1 = theta * torch.cos(t * theta) / (sin_theta + eps)
    
    return coeff0 * x0 + coeff1 * x1
```

**Use when**: Directional data, normalized embeddings, angular spaces.

### Implementation 3: Learned Manifold (Encoder-Decoder)

**Strategy**: Use a learned latent space where linear interpolation is appropriate.

**Architecture**:
```python
class ManifoldFlowMatching(nn.Module):
    def __init__(self):
        self.encoder = Encoder()  # x → z (to latent manifold)
        self.decoder = Decoder()  # z → x (from latent)
        self.velocity_net = VelocityField()  # v(z, t)
    
    def forward(self, x0, x1, t):
        # Encode to latent space
        z0 = self.encoder(x0)
        z1 = self.encoder(x1)
        
        # Linear interpolation in latent space
        zt = (1 - t) * z0 + t * z1
        
        # Predict velocity in latent space
        vt = self.velocity_net(zt, t)
        return vt
```

**Rationale**: If the encoder learns to map data onto a region where Euclidean geometry is appropriate, linear interpolation becomes a good geodesic approximation.

**Use when**: Complex manifold structure, no closed-form geodesics available.

### Implementation 4: Variance-Preserving Interpolation

**Path** (maintains variance like diffusion):

$$
x_t = \sqrt{1 - \sigma_t^2} \, x_0 + \sigma_t \, x_1, \quad \sigma_t = t
$$

**Velocity**:

$$
u_t = \frac{d}{dt}\left[\sqrt{1-t^2} \, x_0 + t \, x_1\right] = \frac{-t}{\sqrt{1-t^2}} x_0 + x_1
$$

**Code**:
```python
def vp_interpolation(x0, x1, t, eps=1e-7):
    """Variance-preserving interpolation."""
    sigma_t = t
    alpha_t = torch.sqrt(1 - sigma_t**2 + eps)
    return alpha_t * x0 + sigma_t * x1

def vp_velocity(x0, x1, t, eps=1e-7):
    """VP velocity field."""
    sigma_t = t
    alpha_t = torch.sqrt(1 - sigma_t**2 + eps)
    d_alpha = -sigma_t / (alpha_t + eps)
    d_sigma = 1.0
    return d_alpha * x0 + d_sigma * x1
```

**Use when**: Want to maintain variance structure similar to diffusion models.

---

## Why Geodesics Matter for Biological Data

### Biological Data Often Lives on Manifolds

**Examples**:

1. **Gene expression manifolds**
   - Cells don't randomly occupy gene expression space
   - Valid states form lower-dimensional manifolds
   - Cell type trajectories follow manifold structure

2. **Protein conformations**
   - 3D rotations: $SO(3)$ manifold
   - Torsion angles: Toroidal manifolds
   - Valid conformations: Complex manifolds

3. **Single-cell trajectories**
   - Differentiation paths follow manifold geodesics
   - Branching points are manifold features
   - Pseudotime corresponds to geodesic distance

4. **Molecular latent spaces**
   - Drug-like molecules form manifolds
   - Chemical validity imposes geometric constraints
   - Property optimization follows manifold paths

### Consequences of Ignoring Manifold Structure

**Using linear interpolation when data lives on a manifold**:

1. **Invalid intermediate states**
   - Interpolated points may violate biological constraints
   - Generated samples may be "off-manifold" (biologically impossible)

2. **Distorted distances**
   - Euclidean distance ≠ manifold distance
   - Nearby cells in Euclidean space may be far on manifold
   - Affects neighborhood structure, clustering

3. **Suboptimal generation**
   - Model learns to "correct" for off-manifold paths
   - More complex velocity fields needed
   - Slower convergence, more training data required

### Benefits of Geodesic-Aware Interpolation

**Using manifold-aware interpolation**:

1. **Biologically valid paths**
   - Interpolated points stay on valid cell states
   - Transitions respect biological plausibility

2. **Improved sample quality**
   - Generated cells more realistic
   - Better preservation of biological correlations

3. **Interpretable trajectories**
   - Learned paths correspond to real biological processes
   - Can study intermediate states along trajectories

4. **Data efficiency**
   - Simpler velocity fields (follow natural structure)
   - Less training data needed

### Example: scPPDM Application

For your **scPPDM** (single-cell Protein Perturbation Diffusion Model), consider:

**Challenge**: Single-cell data lives on a low-dimensional manifold in high-dimensional gene expression space.

**Geodesic perspective**:
- Perturbation effects should follow manifold structure
- Control → perturbed paths should be manifold geodesics
- Linear interpolation in high-D space may cut through invalid states

**Potential approach**:
1. Learn manifold via encoder (e.g., VAE, autoencoder)
2. Perform flow matching in latent space
3. Use linear geodesics where appropriate, learned geometry where complex
4. Ensure interpolated states remain biologically plausible

---

## The Unspoken Dream: Learned Geometry

### The Bold Idea

The most exciting possibility hidden in flow matching:

> **What if the model itself induces the geometry?**

Rather than pre-specifying geodesics, let the learned velocity field **define** the geodesics.

### How It Works

**Standard view**:
1. Choose interpolation $\psi_t$ (defines geodesics)
2. Train velocity field $v_\theta$ to match
3. Velocity field is subordinate to choice

**Learned geometry view**:
1. Train velocity field $v_\theta$ with weak inductive bias
2. The flow lines $\frac{dx}{dt} = v_\theta(x,t)$ **are** the geodesics
3. Geometry emerges from the learned field

**Philosophical shift**:
- Geodesics are **outputs**, not inputs
- The model discovers natural paths through data space
- Geometry, dynamics, and generation become one

### Connection to Energy-Based Models

This connects to your **EBM** work:

**Energy functional**: Suppose the velocity field derives from an energy landscape:

$$
v_\theta(x, t) = -\nabla_x E_\theta(x, t)
$$

**Geodesics**: Flow lines follow gradient descent on $E$.

**Learned metric**: The energy landscape induces a Riemannian metric:

$$
g_{ij}(x) = \frac{\partial^2 E}{\partial x_i \partial x_j}
$$

**Result**: The learned flow respects a **learned geometry** encoded in $E$.

### Practical Realization

**Unrestricted flow matching**:
- Minimal assumptions on $\psi_t$
- Let network learn complex, data-adaptive paths
- More expressive but requires more data

**Architecture inductive biases**:
- Equivariant networks preserve symmetries
- Attention mechanisms capture long-range structure
- These biases shape the learned geometry

### Future Directions

**Open questions**:
1. Can we explicitly extract the learned metric from $v_\theta$?
2. How do architectural choices constrain the learnable geometries?
3. Can we regularize toward known geometric structures (e.g., local isometry)?
4. How does this connect to optimal transport and Wasserstein gradient flows?

This is where flow matching, differential geometry, optimal transport, and deep learning converge — fertile ground for research.

---

## Summary and Key Takeaways

### Core Concepts

1. **Geodesics are shortest paths** with respect to a metric
2. **Linear interpolation is the Euclidean geodesic** (flat space)
3. **Manifolds require manifold-aware geodesics** (curved spaces)
4. **Wasserstein geodesics** operate in probability space
5. **Learned geometry** allows data-adaptive paths

### Practical Guidance

**Choose your interpolation based on your data**:

| Data Type | Recommended Interpolation | Why |
|-----------|--------------------------|-----|
| **Images (pixels)** | Linear | Euclidean space, well-normalized |
| **Embeddings (learned)** | Linear | Latent spaces often Euclidean-like |
| **Directional data** | SLERP | Natural sphere structure |
| **Rotations** | Lie group geodesics | SO(3) manifold structure |
| **Biological states** | Learned manifold + linear | Complex, data-dependent |
| **Molecules** | Graph-aware + learned | Discrete + continuous |

### Theoretical Insights

1. **"Geodesic" in flow matching is aspirational**
   - Not usually true Riemannian geodesics
   - Rather, "natural, structure-respecting paths"

2. **Three levels of geometry**:
   - **Data space**: $\mathbb{R}^d$ or manifold $\mathcal{M}$
   - **Probability space**: Wasserstein geometry
   - **Learned space**: Induced by velocity field

3. **Trade-offs**:
   - **Simple geodesics** (linear): Fast, easy, may distort
   - **True geodesics** (manifold): Accurate, complex, expensive
   - **Learned geodesics**: Flexible, data-driven, requires more data

### Connection to Your Research

**For scPPDM and biological generative models**:

1. Single-cell data has **intrinsic manifold structure** — consider geodesic-aware methods
2. Perturbation effects should follow **biologically plausible paths**
3. Linear interpolation in high-D may be suboptimal — explore **latent manifold approaches**
4. Flow matching provides a clean framework for **geometry-aware generation**

### Key Equations Reference

**Linear interpolation (Rectified Flow)**:

$$
x_t = (1-t) x_0 + t x_1, \quad u_t = x_1 - x_0
$$

**Spherical interpolation (SLERP)**:

$$
x_t = \frac{\sin((1-t)\theta)}{\sin(\theta)} x_0 + \frac{\sin(t\theta)}{\sin(\theta)} x_1, \quad \theta = \arccos(x_0^\top x_1)
$$

**General manifold geodesic**:

$$
x_t = \exp_{x_0}(t \cdot \log_{x_0}(x_1))
$$

**Variance-preserving**:

$$
x_t = \sqrt{1-\sigma_t^2} \, x_0 + \sigma_t \, x_1
$$

---

## References

### Foundational Papers

1. **Lipman, Y., et al. (2023)**. Flow Matching for Generative Modeling. *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

2. **Liu, X., et al. (2023)**. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

### Manifold Geometry

3. **Pennec, X. (2006)**. Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements. *Journal of Mathematical Imaging and Vision*.

4. **Fletcher, P. T., et al. (2004)**. Principal Geodesic Analysis for the Study of Nonlinear Statistics of Shape. *IEEE TMI*.

### Optimal Transport

5. **Pooladian, A., et al. (2023)**. Multisample Flow Matching: Straightening Flows with Minibatch Couplings. *ICML*.

6. **Tong, A., et al. (2024)**. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. *TMLR*. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)

### Manifold Learning for Biology

7. **Hashimoto, T., et al. (2016)**. Learning Population-Level Diffusions with Generative RNNs. *ICML*.

8. **Lopez, R., et al. (2018)**. Deep Generative Modeling for Single-cell Transcriptomics. *Nature Methods*.

### SLERP and Interpolation

9. **Shoemake, K. (1985)**. Animating Rotation with Quaternion Curves. *SIGGRAPH*.

10. **White, T. (2016)**. Sampling Generative Networks. [arXiv:1609.04468](https://arxiv.org/abs/1609.04468)

---

## Related Documents

- [Flow Matching Foundations](01_flow_matching_foundations.md) — Mathematical foundations of flow matching
- [Rectifying Flow Tutorial](rectifying_flow.md) — Practical tutorial on rectified flow
- [Flow Matching README](README.md) — Overview of flow matching documentation

---

*This document provides a comprehensive tutorial on geodesic interpolation in flow matching. For questions or clarifications, please refer to the related documents or open an issue in the repository.*
