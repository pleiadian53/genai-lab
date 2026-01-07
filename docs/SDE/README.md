# Stochastic Differential Equations (SDEs) for Diffusion Models

This directory contains reference materials on the SDE formulation of diffusion models.

---

## 🎯 **Main SDE Resources Are in Notebooks!**

**All comprehensive SDE materials are located at**: [`notebooks/diffusion/02_sde_formulation/`](../../notebooks/diffusion/02_sde_formulation/)

### What You'll Find There

**Core Materials**:
- [`README.md`](../../notebooks/diffusion/02_sde_formulation/README.md) — Complete SDE theory guide
- [`02_sde_formulation.ipynb`](../../notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb) — Interactive code with visualizations
- [`sde_QA.md`](../../notebooks/diffusion/02_sde_formulation/sde_QA.md) — Common questions answered

**8 Focused Supplements** ([`supplements/`](../../notebooks/diffusion/02_sde_formulation/supplements/)):
1. **Forward SDE Design Choices** — How to choose $f(x,t)$ and $g(t)$ (VP/VE/sub-VP)
2. **Brownian Motion Dimensionality** — Why $w(t) \in \mathbb{R}^d$
3. **Equivalent Parameterizations** — Score ↔ noise ↔ $x_0$ conversions
4. **Training Loss and Denoising** — Score matching derivation
5. **Reverse SDE and Probability Flow ODE** — Sampling mechanics
6. **Fokker-Planck and Effective Drift** — Advanced PDE connections
7. **Fokker-Planck Equation** — From conservation laws to probability evolution ⭐ NEW
8. **Dimensional Analysis** — Units, scaling, and sanity checks ⭐ NEW

**Start here**: [`notebooks/diffusion/02_sde_formulation/README.md`](../../notebooks/diffusion/02_sde_formulation/README.md)

---

## Primary Resources

**For learning**: See the interactive tutorial with code:
- **Location**: [`notebooks/diffusion/02_sde_formulation/`](../../notebooks/diffusion/02_sde_formulation/)
- **Contents**:
  - `README.md` — Comprehensive theory document
  - `02_sde_formulation.ipynb` — Interactive notebook with visualizations

## What's Here

This `docs/SDE/` directory serves as a reference location for:
- Additional mathematical derivations
- Advanced topics not covered in notebooks
- Research paper notes

**Current status**: The main SDE tutorials are kept under the notebooks directory for better co-location with code.

## Quick Links

- [SDE Tutorial (Theory)](../../notebooks/diffusion/02_sde_formulation/README.md)
- [SDE Tutorial (Code)](../../notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb)
- [DDPM Basics](../../notebooks/diffusion/01_ddpm_basics.ipynb)

## Topics Covered

1. **Brownian Motion**: Properties and visualization
2. **Forward SDE**: Data corruption process
3. **Score Functions**: What is learned
4. **Reverse SDE**: Sampling process
5. **VP-SDE**: Variance-preserving formulation (DDPM)
6. **Probability Flow ODE**: Deterministic sampling

## Archive

**Note**: Original draft files have been moved to `dev/diffusion/sde/` (private development area).

These drafts have been superseded by comprehensive tutorials in [`notebooks/diffusion/02_sde_formulation/supplements/`](../../notebooks/diffusion/02_sde_formulation/supplements/):
- Supplement 07: Fokker-Planck Equation (replaces div_and_laplace content)
- Supplement 08: Dimensional Analysis (replaces unit_analysis content)
- Main sde_QA.md: Canonical Q&A version

**For historical reference**: See `dev/diffusion/sde/README.md` (not tracked in git)
