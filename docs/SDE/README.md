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

This `docs/SDE/` directory contains **tutorial-style mathematical derivations** that complement the interactive notebooks. These documents provide:

- Detailed line-by-line derivations
- Mathematical foundations and proofs
- Historical context and development
- Advanced theoretical connections

### Documents in This Directory

#### Core SDE Theory

1. **[01_diffusion_sde_view.md](01_diffusion_sde_view.md)** — SDE Formulation Overview
   - Forward and reverse SDEs
   - Score functions and their role
   - Brownian motion fundamentals
   - Connection to diffusion models

2. **[01a_diffusion_sde_view_QA.md](01a_diffusion_sde_view_QA.md)** — Design Principles Q&A
   - Why specific drift functions?
   - Score vs. noise prediction
   - High-dimensional intuition
   - Practical design choices

#### DDPM ↔ SDE Connections

3. **[02_sde_and_ddpm.md](02_sde_and_ddpm.md)** — Deriving DDPM from VP-SDE
   - Euler–Maruyama discretization
   - Variance-preserving structure
   - Why DDPM predicts noise
   - Forward and reverse processes

4. **[02c_ddpm_to_vpsde.md](02c_ddpm_to_vpsde.md)** — From DDPM to VP-SDE (Continuous Limit)
   - Moment matching approach
   - Taylor expansion of discrete steps
   - Recovering the continuous SDE
   - Identity check derivation

#### Mathematical Foundations

5. **[02a_taylor_expansion.md](02a_taylor_expansion.md)** — Taylor Expansions in Diffusion
   - Role in Euler–Maruyama
   - Square root approximation in DDPM
   - Fokker–Planck equation derivation
   - Continuous vs. discrete connections

6. **[02b_fokker_plank_eq.md](02b_fokker_plank_eq.md)** — Fokker–Planck Equation Derivation ⭐
   - Line-by-line derivation from SDEs
   - Test function approach
   - Integration by parts
   - Weak vs. strong solutions
   - Examples and intuition

#### Solving and Sampling

7. **[03_solving_vpsde.md](03_solving_vpsde.md)** — Solving the VP-SDE ⭐ NEW
   - Exact solution via integrating factor
   - Closed-form marginal distribution
   - Connection: $\bar{\alpha}(t) = \exp(-\int_0^t \beta(s)\,ds)$
   - Discrete products → continuous integrals

8. **[03a_reverse_time_sde_and_proba_flow_ode.md](03a_reverse_time_sde_and_proba_flow_ode.md)** — Reverse SDE and Probability Flow ODE ⭐ NEW
   - Reverse-time SDE (Anderson, 1982)
   - Probability flow ODE (Song et al., 2021)
   - DDPM as SDE discretization
   - DDIM as ODE discretization
   - The $\eta$ parameter

9. **[03b_ddim_update_coeff.md](03b_ddim_update_coeff.md)** — DDIM Update Coefficients ⭐ NEW
   - Exact DDIM update formula derivation
   - From $\bar{\alpha}(t)$ to `alphas_cumprod` array
   - Why $\sqrt{\bar{\alpha}_t}$ coefficients appear
   - Fast sampling (skipping steps)
   - Complete theory-to-code connection

**Current status**: These documents provide rigorous mathematical foundations that complement the code-focused notebooks.

## Quick Links

### Notebooks (Interactive + Code)
- [SDE Tutorial (Theory)](../../notebooks/diffusion/02_sde_formulation/README.md)
- [SDE Tutorial (Code)](../../notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb)
- [DDPM Basics](../../notebooks/diffusion/01_ddpm_basics.ipynb)

### Docs (Mathematical Derivations)
- [SDE View Overview](01_diffusion_sde_view.md)
- [DDPM from VP-SDE](02_sde_and_ddpm.md)
- [VP-SDE from DDPM](02c_ddpm_to_vpsde.md)
- [Solving VP-SDE](03_solving_vpsde.md) ⭐ NEW
- [Reverse SDE & Probability Flow ODE](03a_reverse_time_sde_and_proba_flow_ode.md) ⭐ NEW
- [DDIM Update Coefficients](03b_ddim_update_coeff.md) ⭐ NEW
- [Fokker–Planck Equation](02b_fokker_plank_eq.md)
- [Taylor Expansions](02a_taylor_expansion.md)

## Learning Path

### For Beginners

1. Start with [SDE View Overview](01_diffusion_sde_view.md) for conceptual understanding
2. Read [DDPM from VP-SDE](02_sde_and_ddpm.md) to see discrete-continuous connection
3. Understand [Solving VP-SDE](03_solving_vpsde.md) for the forward solution
4. See [DDIM Update Coefficients](03b_ddim_update_coeff.md) for exact code formulas
5. Work through the [interactive notebook](../../notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb)

### For Deep Dive

1. Study [Taylor Expansions](02a_taylor_expansion.md) for mathematical foundations
2. Work through [Fokker–Planck derivation](02b_fokker_plank_eq.md) for PDE theory
3. Complete the identity check with [VP-SDE from DDPM](02c_ddpm_to_vpsde.md)
4. Master [Solving VP-SDE](03_solving_vpsde.md) for closed-form solutions
5. Understand sampling with [Reverse SDE & Probability Flow ODE](03a_reverse_time_sde_and_proba_flow_ode.md)
6. Connect theory to code with [DDIM Update Coefficients](03b_ddim_update_coeff.md)
7. Explore [Design Principles Q&A](01a_diffusion_sde_view_QA.md) for practical insights

## Topics Covered

1. **Brownian Motion**: Properties, scaling, and dimensionality
2. **Forward SDE**: Data corruption as a stochastic process
3. **Score Functions**: Gradient of log density and its role
4. **Reverse SDE**: Time-reversal and sampling
5. **VP-SDE**: Variance-preserving formulation (DDPM connection)
6. **Fokker–Planck Equation**: Probability density evolution
7. **Discretization**: Euler–Maruyama and variance preservation
8. **Moment Matching**: Connecting discrete and continuous views

## Archive

**Note**: Original draft files have been moved to `dev/diffusion/sde/` (private development area).

These drafts have been superseded by comprehensive tutorials in [`notebooks/diffusion/02_sde_formulation/supplements/`](../../notebooks/diffusion/02_sde_formulation/supplements/):
- Supplement 07: Fokker-Planck Equation (replaces div_and_laplace content)
- Supplement 08: Dimensional Analysis (replaces unit_analysis content)
- Main sde_QA.md: Canonical Q&A version

**For historical reference**: See `dev/diffusion/sde/README.md` (not tracked in git)
