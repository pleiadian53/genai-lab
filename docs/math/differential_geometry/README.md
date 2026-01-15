# Differential Geometry for Machine Learning

This directory contains tutorials and resources on differential geometry, tensor calculus, and their applications to machine learning.

---

## Documents

### Core Tutorial

1. **[`01_tensor_calculus_and_curved_spaces.md`](./01_tensor_calculus_and_curved_spaces.md)** - Comprehensive tutorial
   - Part 1-2: From Euclidean to curved spaces, metric tensor
   - Part 3-4: Covariant derivatives, gradient on manifolds
   - Part 5-6: Connection to probability and diffusion on manifolds
   - Part 7-8: Current research and practical considerations
   - Part 9-10: Code examples and resources

### Supplementary Materials

2. **[`02_riemannian_diffusion_models.md`](./02_riemannian_diffusion_models.md)** - Recent advances (coming soon)
   - State-of-the-art research
   - Detailed case studies (proteins, molecules, robotics)
   - Implementation patterns

---

## Why Study This?

### Current Limitations of Euclidean ML

Most ML models assume data lives in **Euclidean space** $\mathbb{R}^d$. But many real-world problems have **geometric structure**:

- **Proteins**: Torsion angles on tori $T^n$
- **Molecules**: Conformations on energy manifolds
- **Rotations**: SO(3), not $\mathbb{R}^3$
- **Directions**: Sphere $S^{d-1}$
- **Hierarchies**: Hyperbolic space $\mathbb{H}^n$

Using Euclidean methods on these problems:
- ❌ Violates constraints (e.g., generates invalid bond angles)
- ❌ Wastes capacity (learning to stay on manifold)
- ❌ Poor generalization (doesn't respect geometry)

### Geometric ML Advantages

Using **Riemannian diffusion** and **geometric deep learning**:
- ✅ Respects structure (always valid outputs)
- ✅ More efficient (fewer parameters)
- ✅ Better generalization (bakes in inductive biases)
- ✅ Interpretable (geometry has physical meaning)

---

## Learning Path

### For ML Practitioners

If you're coming from standard ML and want to understand geometric methods:

1. **Start here**: [`01_tensor_calculus_and_curved_spaces.md`](./01_tensor_calculus_and_curved_spaces.md)
   - Read Part 1-2 (intuition for curved spaces)
   - Skim Part 3-4 (covariant derivatives—don't need to master)
   - Focus on Part 5-7 (connection to ML)
   - Try Part 9 (code example on circle)

2. **Then**: Read Bronstein et al. "Geometric Deep Learning" book (free online)

3. **Finally**: Pick an application area and dive into recent papers

### For Math/Physics Background

If you know differential geometry and want ML applications:

1. **Quick review**: Part 1-4 of [`01_tensor_calculus_and_curved_spaces.md`](./01_tensor_calculus_and_curved_spaces.md)
2. **Focus on**: Part 5-9 (how geometry connects to diffusion models)
3. **Dive into**: Recent papers in Part 10

---

## Key Applications (2024-2026)

### 1. Protein Structure Generation

**Problem**: Design novel proteins with desired functions

**Geometry**: 

- Backbone: Torsion angles $(\phi, \psi, \omega)$ on torus $T^3$
- Full structure: SE(3) transformations (rigid motions)

**Recent work**:

- **RFdiffusion** (2023): De novo protein design using SE(3)-equivariant diffusion
- **Chroma** (2023): Generative model for protein design
- **FoldFlow** (2023): Flow matching on SE(3)

**Impact**: Designed proteins for therapeutics, enzymes, biosensors

### 2. Molecular Docking

**Problem**: Predict how small molecules bind to proteins

**Geometry**:

- Translation: $\mathbb{R}^3$
- Rotation: SO(3) (3D rotations)
- Combined: SE(3) (Euclidean group)

**Recent work**:

- **DiffDock** (2023): SE(3)-equivariant diffusion for docking
- 38% improvement over prior methods

**Impact**: Drug discovery, understanding protein-ligand interactions

### 3. Climate and Earth Science

**Problem**: Weather prediction, climate modeling

**Geometry**: 

- Earth surface: Sphere $S^2$
- Need to respect spherical geometry (no "edges")

**Recent work**:

- Spherical CNNs for climate data
- Diffusion models on $S^2$ for weather forecasting

**Impact**: More accurate predictions, especially near poles

### 4. Robotics

**Problem**: Motion planning, manipulation

**Geometry**:

- Configuration space: Product of circles $(S^1)^n$ for revolute joints
- End-effector poses: SE(3)

**Recent work**:

- Diffusion policies on configuration manifolds
- SE(3)-equivariant networks for grasping

**Impact**: More natural motion, respects joint limits automatically

---

## Prerequisites

### Minimal Background

To get started:
- **Linear algebra**: Vectors, matrices, eigenvalues
- **Multivariable calculus**: Partial derivatives, chain rule
- **Probability**: Gaussian distributions, density functions
- **Python/PyTorch**: Basic neural network training

You do **not** need:
- Prior differential geometry knowledge
- Tensor calculus background
- Physics training

### Recommended Background

For deeper understanding:
- **Vector calculus**: Gradient, divergence, curl
- **Linear algebra**: Positive definite matrices, quadratic forms
- **PDEs**: Heat equation, diffusion
- **Probability**: Stochastic processes, SDEs

---

## Software Tools

### Python Libraries

**Riemannian geometry**:
```bash
pip install geoopt geomstats pymanopt
```

**Geometric deep learning**:
```bash
pip install torch-geometric e3nn
```

**Protein structure**:
```bash
pip install biopython py3Dmol
```

### Quick Start

```python
# Example: Optimization on sphere using geoopt
import torch
import geoopt

# Define manifold
sphere = geoopt.Sphere()

# Create point on sphere
x = sphere.random(5, 10)  # 5 samples, 10-dimensional sphere

# Define parameter on manifold
param = geoopt.ManifoldParameter(x, manifold=sphere)

# Use in optimization (automatically projects gradients)
optimizer = geoopt.optim.RiemannianAdam([param], lr=0.01)
```

---

## Recent Advances (2024-2026)

### Theoretical Breakthroughs

1. **Riemannian Flow Matching** (Chen et al. 2023)
   - Extends flow matching to general manifolds
   - Computationally efficient (no score function)

2. **Gauge Equivariant Networks** (Cohen et al. 2024)
   - Build networks that respect local symmetries
   - Application to curved spaces

3. **Learned Metrics** (Zhang et al. 2024)
   - Learn manifold geometry from data
   - Don't need to specify metric a priori

### Practical Tools

1. **`geomstats` 2.0** (2024): Unified API for Riemannian geometry
2. **`e3nn` for biology**: SE(3)-equivariant networks for proteins
3. **`geoopt` integration**: PyTorch Lightning support

### Benchmark Datasets

1. **RIEMANNIAN-BENCH** (2024): Standardized benchmarks for Riemannian ML
2. **Protein structure datasets**: PDB, AlphaFold DB
3. **Earth science**: Climate modeling datasets on $S^2$

---

## Open Research Questions

### Fundamental Theory

1. **Optimal parametrizations**: Ambient vs. intrinsic coordinates?
2. **Approximation theory**: How well can neural networks approximate on manifolds?
3. **Generalization bounds**: Sample complexity on manifolds
4. **Tractable inference**: Fast sampling on high-dimensional manifolds

### Practical Challenges

1. **Scalability**: Efficient algorithms for large manifolds (proteins: 1000+ atoms)
2. **Numerical stability**: Avoiding coordinate singularities
3. **Architecture design**: Best practices for network design on manifolds
4. **Training efficiency**: Slow convergence with geometric constraints

### New Applications

1. **Single-cell genomics**: Cell differentiation as flow on manifolds
2. **Financial time series**: Returns on positive definite cone
3. **Neuroscience**: Brain connectivity on manifold of correlation matrices
4. **Quantum chemistry**: Molecular orbitals on Grassmannian manifolds

---

## Contributing

Found an error or have suggestions? This is a living document!

- Corrections: Please point out any mathematical errors
- Clarifications: If something is unclear, let us know
- Extensions: Suggestions for additional topics welcome

---

## Related Documentation

### Within This Project

- [`../../diffusion/`](../../diffusion/): Standard Euclidean diffusion models
- [`../../SDE/`](../../SDE/): SDE formulation background
- [`../../dev/ddpm_learning_process/`](../../dev/ddpm_learning_process/): DDPM mechanics

### External Resources

- **Geometric Deep Learning Book**: https://geometricdeeplearning.com/
- **Riemannian Score-Based Models Paper**: https://arxiv.org/abs/2202.02763
- **geomstats Tutorials**: https://geomstats.github.io/tutorials/
