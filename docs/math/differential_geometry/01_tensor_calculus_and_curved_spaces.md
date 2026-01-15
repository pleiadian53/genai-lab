# Tensor Calculus and Curved Spaces: A Tutorial for Machine Learning

## Overview

This tutorial introduces tensor calculus and differential geometry on curved spaces, motivated by their emerging applications in machine learning. We'll build from first principles to modern applications in diffusion models, geometric deep learning, and beyond.

**Why this matters for ML**:

- Protein structures live on curved spaces (torsion angles on tori)
- Molecular conformations are constrained to manifolds
- Robotic joint configurations form non-Euclidean spaces
- Many ML problems have natural geometric structure that Euclidean methods ignore

---

## Part 1: From Euclidean to Curved Spaces

### 1.1 Euclidean Space: What We're Used To

In **Euclidean space** $\mathbb{R}^d$ with **Cartesian coordinates** $(x_1, x_2, \ldots, x_d)$:

**Distance** (Pythagorean theorem):

$$
ds^2 = dx_1^2 + dx_2^2 + \cdots + dx_d^2
$$

**Gradient** (just partial derivatives):

$$

\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\vdots \\
\frac{\partial f}{\partial x_d}
\end{bmatrix}
$$

**Dot product**:

$$

\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i
$$

**Key property**: All these operations are **coordinate-independent** in Cartesian coordinates.

### 1.2 Curvilinear Coordinates: First Step Beyond Cartesian

Even in **flat** (Euclidean) space, using different coordinates changes how we compute distances and gradients.

**Example: Polar Coordinates in 2D**

Cartesian: $(x, y)$  
Polar: $(r, \theta)$ where $x = r\cos\theta$, $y = r\sin\theta$

**Distance element**:

$$

ds^2 = dr^2 + r^2 d\theta^2
$$

Notice: Not $dr^2 + d\theta^2$ because angles and radii have different units!

**Gradient**:

$$

\nabla f = \frac{\partial f}{\partial r} \hat{e}_r + \frac{1}{r} \frac{\partial f}{\partial \theta} \hat{e}_\theta
$$

Notice: The $1/r$ factor! This comes from the coordinate system.

### 1.3 Curved Spaces: Intrinsic Curvature

A **curved space** (manifold) is one where you **cannot** find coordinates that make distances Pythagorean everywhere.

**Example: Sphere $S^2$**

Using spherical coordinates $(\theta, \phi)$:

$$
ds^2 = R^2(d\theta^2 + \sin^2\theta \, d\phi^2)
$$

**Key difference**: The coefficient $\sin^2\theta$ depends on position—this is **intrinsic curvature**.

**Physical intuition**: 

- Near the equator, longitude lines are far apart
- Near the poles, they converge
- This affects distances and gradients

---

## Part 2: The Metric Tensor

### 2.1 Definition

The **metric tensor** $g_{ij}$ encodes how to measure distances in arbitrary coordinates.

**Distance element**:

$$

ds^2 = \sum_{i,j=1}^d g_{ij} \, dx^i \, dx^j = g_{ij} \, dx^i \, dx^j
$$

(Using Einstein summation: repeated indices are summed)

**Matrix form**:

$$

ds^2 = (dx)^T G \, dx
$$

where $G = [g_{ij}]$ is the **metric tensor matrix**.

### 2.2 Examples

**Euclidean space** (Cartesian):

$$

G = I = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

**Polar coordinates** (in 2D):

$$

G = \begin{bmatrix}
1 & 0 \\
0 & r^2
\end{bmatrix}
$$

**Sphere** $S^2$ (radius $R$):

$$

G = \begin{bmatrix}
R^2 & 0 \\
0 & R^2\sin^2\theta
\end{bmatrix}
$$

**Minkowski spacetime** (special relativity):

$$

G = \begin{bmatrix}
-c^2 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

### 2.3 Raising and Lowering Indices

The metric tensor relates **contravariant** (upper index) and **covariant** (lower index) vectors.

**Lowering an index**:

$$

v_i = g_{ij} v^j
$$

**Raising an index**:

$$

v^i = g^{ij} v_j
$$

where $g^{ij}$ is the **inverse metric tensor**: $g^{ik} g_{kj} = \delta^i_j$.

**In Euclidean space**: $g_{ij} = g^{ij} = \delta_{ij}$, so there's no difference between upper and lower indices.

**In curved space**: Upper and lower indices are genuinely different!

---

## Part 3: Covariant Derivatives and Christoffel Symbols

### 3.1 The Problem with Ordinary Derivatives

In curved space, **partial derivatives of vectors are not tensors**!

**Why?** Because basis vectors $\hat{e}_i$ themselves change from point to point.

**Example on a sphere**: 

- The "north" direction at one point is different from "north" at another point
- Taking $\frac{\partial \mathbf{v}}{\partial \theta}$ mixes the change in vector components with the change in basis vectors

### 3.2 Covariant Derivative

The **covariant derivative** $\nabla_i$ is the correct way to differentiate in curved spaces.

**For a scalar** (easy):

$$
\nabla_i f = \frac{\partial f}{\partial x^i}
$$

**For a vector** (requires correction):

$$

\nabla_i v^j = \frac{\partial v^j}{\partial x^i} + \Gamma^j_{ik} v^k
$$

where $\Gamma^j_{ik}$ are the **Christoffel symbols** (connection coefficients).

### 3.3 Christoffel Symbols

The Christoffel symbols encode how basis vectors change:

$$
\frac{\partial \hat{e}_i}{\partial x^j} = \Gamma^k_{ij} \hat{e}_k
$$

**Formula in terms of metric**:

$$

\Gamma^k_{ij} = \frac{1}{2} g^{k\ell} \left(\frac{\partial g_{\ell i}}{\partial x^j} + \frac{\partial g_{\ell j}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^\ell}\right)
$$

**Key properties**:

- $\Gamma^k_{ij} = \Gamma^k_{ji}$ (symmetric in lower indices)
- $\Gamma^k_{ij} = 0$ in Cartesian coordinates on flat space
- Nonzero in curvilinear coordinates or curved spaces

### 3.4 Example: Sphere

For a sphere $S^2$ with metric $ds^2 = R^2(d\theta^2 + \sin^2\theta \, d\phi^2)$:

**Non-zero Christoffel symbols**:

$$

\Gamma^\theta_{\phi\phi} = -\sin\theta \cos\theta
$$
$$

\Gamma^\phi_{\theta\phi} = \Gamma^\phi_{\phi\theta} = \frac{\cos\theta}{\sin\theta} = \cot\theta
$$

**Interpretation**: 

- Moving in $\phi$ (longitude) changes your $\theta$ (latitude) direction
- This is the geometric "correction" needed on a curved surface

---

## Part 4: Gradient on Curved Spaces

### 4.1 Definition

The **gradient** of a scalar function $f$ is the vector that satisfies:

$$
df = \nabla f \cdot d\mathbf{x}
$$

for any displacement $d\mathbf{x}$.

### 4.2 Formula

In arbitrary coordinates:

$$
(\nabla f)^i = g^{ij} \frac{\partial f}{\partial x^j}
$$

**Key insight**: The **inverse metric** $g^{ij}$ couples dimensions!

### 4.3 Example: Polar Coordinates

Metric: $g_{ij} = \text{diag}(1, r^2)$  
Inverse metric: $g^{ij} = \text{diag}(1, 1/r^2)$

$$
\nabla f = \begin{bmatrix}
g^{rr} \frac{\partial f}{\partial r} \\
g^{\phi\phi} \frac{\partial f}{\partial \phi}
\end{bmatrix} = \begin{bmatrix}
\frac{\partial f}{\partial r} \\
\frac{1}{r^2} \frac{\partial f}{\partial \phi}
\end{bmatrix}
$$

**In orthonormal basis** $(\hat{e}_r, \hat{e}_\phi)$:

$$

\nabla f = \frac{\partial f}{\partial r} \hat{e}_r + \frac{1}{r} \frac{\partial f}{\partial \phi} \hat{e}_\phi
$$

(The $1/r$ factor accounts for the stretched basis vector)

---

## Part 5: Connection to Probability and Score Functions

### 5.1 Score Function on Euclidean Space

In standard diffusion models:

$$
s(x) = \nabla_x \log p(x) = \begin{bmatrix}
\frac{\partial \log p}{\partial x_1} \\
\vdots \\
\frac{\partial \log p}{\partial x_d}
\end{bmatrix}
$$

**This is straightforward because we're in Euclidean space.**

### 5.2 Score Function on Curved Spaces

On a **Riemannian manifold** $\mathcal{M}$ with metric $g_{ij}$:

$$
s(x)^i = g^{ij} \frac{\partial \log p(x)}{\partial x^j}
$$

**Key difference**: The inverse metric $g^{ij}$ can couple dimensions!

**Example on a sphere**: The score in the $\theta$ direction depends on both $\frac{\partial \log p}{\partial \theta}$ and $\frac{\partial \log p}{\partial \phi}$ if the metric has off-diagonal terms.

### 5.3 Why This Matters

Many real-world problems involve data on manifolds:

1. **Protein backbone angles**: Torsion angles $(\phi, \psi)$ live on a **torus** $T^2 = S^1 \times S^1$
2. **Molecular conformations**: Constrained to energy manifolds
3. **Rotations**: SO(3) is a 3D manifold (not $\mathbb{R}^3$!)
4. **Directional data**: Unit vectors live on a sphere $S^{d-1}$

**Standard Euclidean diffusion models** can violate these constraints or produce invalid samples.

**Riemannian diffusion models** respect the geometry.

---

## Part 6: Diffusion on Manifolds

### 6.1 Heat Equation on Manifolds

The **heat equation** (diffusion) on a Riemannian manifold:

$$
\frac{\partial p(x,t)}{\partial t} = \Delta_g p(x,t)
$$

where $\Delta_g$ is the **Laplace-Beltrami operator**:

$$
\Delta_g f = \frac{1}{\sqrt{\det g}} \frac{\partial}{\partial x^i} \left(\sqrt{\det g} \, g^{ij} \frac{\partial f}{\partial x^j}\right)
$$

**In Euclidean space**: $g_{ij} = \delta_{ij}$, $\det g = 1$, so $\Delta_g = \nabla^2$ (standard Laplacian).

**In curved space**: The metric and its determinant affect how diffusion spreads.

### 6.2 Brownian Motion on Manifolds

**Euclidean Brownian motion**:

$$

dx = \sigma \, dw
$$

where $dw \sim \mathcal{N}(0, dt \cdot I)$.

**Riemannian Brownian motion**:

$$

dx^i = \sqrt{g^{ij}} \sigma \, dw_j - \frac{1}{2} \Gamma^i_{jk} g^{jk} \sigma^2 \, dt
$$

**Key differences**:
1. The **metric** $g^{ij}$ scales the noise
2. The **Christoffel symbols** $\Gamma^i_{jk}$ add a drift term

**Physical interpretation**: The drift term prevents the process from "bunching up" in regions where the coordinate system is compressed.

### 6.3 Score-Based Diffusion on Manifolds

**Forward SDE on manifold $\mathcal{M}$**:

$$

dx = f(x,t) \, dt + g(t) \, dw_{\mathcal{M}}
$$

where $dw_{\mathcal{M}}$ is Brownian motion on the manifold.

**Reverse SDE**:

$$

dx = \left[f(x,t) - g(t)^2 \nabla_{\mathcal{M}} \log p_t(x)\right] dt + g(t) \, d\bar{w}
$$

where $\nabla_{\mathcal{M}}$ is the **Riemannian gradient**.

**Neural network**: Learn $s_\theta(x,t) \approx \nabla_{\mathcal{M}} \log p_t(x)$ (the score on the manifold).

---

## Part 7: Current Research and Applications

### 7.1 Riemannian Score-Based Generative Models

**Key papers**:

- De Bortoli et al. (2022): "Riemannian Score-Based Generative Modeling"
- Mathieu & Nickel (2020): "Continuous Hierarchical Representations with Poincaré VAEs"

**Approach**:
1. Define data as living on a manifold $\mathcal{M}$ with metric $g$
2. Construct forward diffusion using Brownian motion on $\mathcal{M}$
3. Learn the Riemannian score function $\nabla_{\mathcal{M}} \log p_t(x)$
4. Sample via reverse SDE using the learned score

**Applications**:

- Protein structure generation (torsion angles on torus)
- Molecular conformations (constrained manifolds)
- Earth science data (spherical domains)

### 7.2 Geometric Deep Learning

**Key idea**: Design neural networks that respect the geometry of the data.

**Examples**:
1. **Graph neural networks**: Data on graphs (non-Euclidean)
2. **Mesh CNNs**: Operate on 3D meshes (curved surfaces)
3. **Equivariant networks**: Respect symmetries (rotations, translations)

**Connection to diffusion**: 

- Can we build score networks that are equivariant to geometric transformations?
- How to parameterize networks on manifolds?

### 7.3 Proteins and Molecular Generation

**Problem**: Protein backbones are parameterized by torsion angles $(\phi, \psi, \omega)$.

**Naive approach**: Treat angles as $\mathbb{R}^3$
- **Issue**: Doesn't respect periodicity ($\phi + 2\pi = \phi$)
- **Result**: Invalid structures, poor generalization

**Geometric approach**: Treat each angle as living on $S^1$ (circle)
- **Manifold**: $T^3 = S^1 \times S^1 \times S^1$ (3-torus)
- **Diffusion**: Brownian motion on the torus
- **Result**: Always generates valid structures

**State-of-the-art**:

- **FoldFlow** (NeurIPS 2023): Diffusion on SE(3) for protein backbone generation
- **DiffDock** (ICLR 2023): SE(3)-equivariant diffusion for molecular docking

### 7.4 Robotics and Motion Planning

**Problem**: Robot joint configurations live on **configuration space** $\mathcal{C}$, often a manifold.

**Example**: Robot arm with revolute joints
- Each joint angle is on $S^1$
- Configuration space: $\mathcal{C} = (S^1)^n$ (n-dimensional torus)

**Geometric approach**:

- Learn motion distribution on $\mathcal{C}$ using Riemannian diffusion
- Generate collision-free paths respecting joint limits
- Naturally handles periodicity and constraints

---

## Part 8: Practical Considerations

### 8.1 Choosing Coordinates

**For manifolds embedded in $\mathbb{R}^D$** (like $S^2 \subset \mathbb{R}^3$):

**Option 1: Intrinsic coordinates** ($\theta, \phi$ on sphere)
- ✅ Lower dimensional (d < D)
- ❌ Singularities (e.g., poles on sphere)
- ❌ Requires explicit metric, Christoffel symbols

**Option 2: Ambient coordinates** ($x, y, z$ in $\mathbb{R}^3$ with constraint $x^2+y^2+z^2=1$)
- ✅ No singularities
- ✅ Simple Euclidean gradient
- ❌ Higher dimensional
- ❌ Need to project to manifold

**Hybrid approach**: Use ambient coordinates but project gradients to tangent space.

### 8.2 Computing Christoffel Symbols

**Symbolic computation** (small problems):
```python
import sympy as sp
# Define metric
g = sp.Matrix([[1, 0], [0, r**2]])
# Compute Christoffel symbols automatically
```

**Numerical computation** (large problems):
- Approximate using finite differences
- Learn from data (e.g., neural network parameterization)

**Automatic differentiation**:

- PyTorch/JAX can compute gradients on manifolds
- Libraries: `geomstats`, `pymanopt`, `geoopt`

### 8.3 Neural Network Architectures

**Challenge**: Standard MLPs and CNNs assume Euclidean structure.

**Solutions**:

1. **Tangent space networks**: Map to tangent space at each point (locally Euclidean)
2. **Gauge equivariant networks**: Use parallel transport to move between tangent spaces
3. **Ambient space projection**: Operate in $\mathbb{R}^D$, project outputs to manifold

**Example**: For $S^2$ (sphere)
```python
# Ambient space projection
z = mlp(x)  # z in R^3
z_normalized = z / torch.norm(z, dim=-1, keepdim=True)  # Project to S^2
```

---

## Part 9: Code Example - Diffusion on a Circle

### 9.1 Setup

Circle: $S^1 = \{(\cos\theta, \sin\theta) : \theta \in [0, 2\pi)\}$

**Parametrization**: Use angle $\theta$ (intrinsic) or $(x, y)$ with $x^2+y^2=1$ (ambient).

### 9.2 Ambient Space Approach

```python
import torch
import torch.nn as nn

class CircleDataset:
    """Data on a circle (e.g., von Mises distribution)"""
    def __init__(self, mu=0, kappa=10):
        self.mu = mu  # Mean direction
        self.kappa = kappa  # Concentration
    
    def sample(self, n):
        """Sample from von Mises distribution"""
        theta = torch.distributions.VonMises(self.mu, self.kappa).sample((n,))
        x = torch.cos(theta)
        y = torch.sin(theta)
        return torch.stack([x, y], dim=-1)  # Shape: (n, 2)

def project_to_circle(x):
    """Project points to unit circle"""
    return x / torch.norm(x, dim=-1, keepdim=True)

class CircleScoreNetwork(nn.Module):
    """Score network for circle using ambient space"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # [x, y, t]
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)  # Output in R^2
        )
    
    def forward(self, x, t):
        """
        x: (batch, 2) points on circle
        t: (batch,) time steps
        """
        t = t.unsqueeze(-1)  # (batch, 1)
        inp = torch.cat([x, t], dim=-1)  # (batch, 3)
        score = self.net(inp)  # (batch, 2)
        
        # Project score to tangent space (orthogonal to x)
        # Tangent space: {v : v · x = 0}
        score = score - (score * x).sum(dim=-1, keepdim=True) * x
        return score

def forward_diffusion_circle(x0, t, noise_schedule):
    """
    Forward diffusion on circle using wrapped normal
    x0: (batch, 2) points on circle
    t: scalar time
    """
    alpha_bar_t = noise_schedule(t)
    
    # Add noise in ambient space
    noise = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    # Project back to circle
    xt = project_to_circle(xt)
    return xt, noise

def train_circle_score(model, dataset, num_epochs=1000):
    """Train score network on circle"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        x0 = dataset.sample(128)  # Sample from data distribution
        t = torch.rand(128)  # Random timesteps
        xt, noise = forward_diffusion_circle(x0, t, lambda t: 1 - t)
        
        # Predict score
        score_pred = model(xt, t)
        
        # True score (for wrapped Gaussian)
        score_true = -(xt - torch.sqrt(1 - t.unsqueeze(-1)) * x0) / (1 - t.unsqueeze(-1))
        score_true = score_true - (score_true * xt).sum(dim=-1, keepdim=True) * xt
        
        loss = ((score_pred - score_true) ** 2).sum(dim=-1).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Usage
dataset = CircleDataset(mu=0, kappa=10)
model = CircleScoreNetwork(hidden_dim=128)
train_circle_score(model, dataset)
```

### 9.3 Intrinsic Coordinates Approach

```python
class CircleScoreNetworkIntrinsic(nn.Module):
    """Score network using angle θ directly"""
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Use periodic encoding for angle
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # [cos(θ), sin(θ), t]
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)  # Output: dθ/dt (scalar)
        )
    
    def forward(self, theta, t):
        """
        theta: (batch,) angles
        t: (batch,) time
        """
        # Periodic encoding
        x = torch.cos(theta)
        y = torch.sin(theta)
        inp = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        return self.net(inp).squeeze(-1)  # (batch,)
```

---

## Part 10: Resources and Next Steps

### 10.1 Essential Textbooks

**Differential Geometry**:
1. **do Carmo**: "Riemannian Geometry" - Classic, very readable
2. **Lee**: "Introduction to Smooth Manifolds" - Comprehensive, modern
3. **Spivak**: "Calculus on Manifolds" - Concise intro

**Geometric ML**:
1. **Bronstein et al.**: "Geometric Deep Learning" (2021) - Free online
2. **Bekkers**: "An Introduction to Geometric Deep Learning" (2023)

**Diffusion on Manifolds**:
1. **Grigoryan**: "Heat Kernel and Analysis on Manifolds"
2. **De Bortoli et al.**: "Riemannian Score-Based Generative Modeling" (ICML 2022)

### 10.2 Software Libraries

**Manifold optimization**:

- `geoopt` (PyTorch): Riemannian optimization
- `pymanopt` (NumPy/PyTorch/JAX): General manifold optimization
- `geomstats` (NumPy/PyTorch/JAX): Comprehensive Riemannian geometry

**Geometric deep learning**:

- `PyG` (PyTorch Geometric): Graph neural networks
- `DGL`: Deep Graph Library
- `e3nn`: Equivariant neural networks (SO(3)/E(3))

**Protein structure**:

- `AlphaFold`: Structure prediction
- `ESM` (Evolutionary Scale Modeling): Protein language models
- `OpenFold`: Open-source AlphaFold implementation

### 10.3 Paper Reading List

**Foundation**:
1. De Bortoli et al. (2022): "Riemannian Score-Based Generative Modeling"
2. Song et al. (2021): "Score-Based Generative Modeling through SDEs"

**Applications**:
3. Yim et al. (2023): "SE(3) Diffusion Model with Application to Protein Backbone Generation"
4. Watson et al. (2023): "De novo design of protein structure and function with RFdiffusion"
5. Corso et al. (2023): "DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking"

**Theory**:
6. Chen et al. (2023): "Riemannian Flow Matching on General Geometries"
7. Huang et al. (2022): "Riemannian Diffusion Models"

### 10.4 Future Directions

**Open research questions**:

1. **Optimal parametrizations**: When to use ambient vs. intrinsic coordinates?
2. **Scalability**: Efficient algorithms for high-dimensional manifolds
3. **Learning the geometry**: Can we learn the manifold structure from data?
4. **Hybrid models**: Combining Euclidean and Riemannian components
5. **Discrete manifolds**: Diffusion on graphs and combinatorial structures

**Emerging applications**:
1. **Drug discovery**: Molecular generation on conformational manifolds
2. **Protein design**: Backbone and sequence co-design
3. **Materials science**: Crystal structure generation (lattices are manifolds)
4. **Climate modeling**: Weather prediction on spherical domains
5. **Robotics**: Motion planning on configuration spaces

---

## Summary: Key Concepts

| Concept | Euclidean Space | Curved Space (Manifold) |
|---------|----------------|------------------------|
| **Metric** | $g_{ij} = \delta_{ij}$ (identity) | General $g_{ij}(x)$ (position-dependent) |
| **Distance** | $ds^2 = \sum_i dx_i^2$ | $ds^2 = g_{ij} dx^i dx^j$ |
| **Gradient** | $\nabla f = \frac{\partial f}{\partial x}$ | $(\nabla f)^i = g^{ij} \frac{\partial f}{\partial x^j}$ |
| **Covariant derivative** | $\nabla_i v^j = \frac{\partial v^j}{\partial x^i}$ | $\nabla_i v^j = \frac{\partial v^j}{\partial x^i} + \Gamma^j_{ik} v^k$ |
| **Laplacian** | $\Delta f = \sum_i \frac{\partial^2 f}{\partial x_i^2}$ | $\Delta_g f = \frac{1}{\sqrt{\det g}} \partial_i(\sqrt{\det g} \, g^{ij} \partial_j f)$ |
| **Brownian motion** | $dx = \sigma dw$ | $dx^i = \sqrt{g^{ij}} \sigma dw_j - \frac{1}{2}\Gamma^i_{jk} g^{jk} \sigma^2 dt$ |

**The main message**: Many of the formulas you know from Euclidean space have natural generalizations to curved spaces, but you need to account for the metric tensor and Christoffel symbols.

---

## Related Documents

- [`../../dev/ddpm_learning_process/03_gradient_operator_and_independence.md`](../../dev/ddpm_learning_process/03_gradient_operator_and_independence.md): Gradient operator in different coordinate systems
- [`../diffusion/brownian_motion_tutorial.md`](../diffusion/brownian_motion_tutorial.md): Brownian motion in Euclidean space
- [`../SDE/01_diffusion_sde_view.md`](../SDE/01_diffusion_sde_view.md): SDE formulation of diffusion models
