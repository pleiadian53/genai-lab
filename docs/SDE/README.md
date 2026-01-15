# SDE Formulation of Diffusion Models

This directory contains comprehensive documentation on the **Stochastic Differential Equation (SDE)** perspective on diffusion models, providing a continuous-time view that unifies DDPM, DDIM, and other variants.

The SDE formulation reveals diffusion models as continuous stochastic processes, enabling flexible sampling, theoretical analysis, and connections to classical stochastic calculus.

---

## Core Documentation Series

This series mirrors the structure of DDPM and flow matching documentation, separating theory, training, and sampling.

| Document | Description |
|----------|-------------|
| [00_sde_overview.md](00_sde_overview.md) | **Overview**: High-level introduction, why SDE view matters, key concepts |
| [01_diffusion_sde_view.md](01_diffusion_sde_view.md) | **Foundations**: Detailed SDE formulation, forward/reverse processes |
| [02_sde_training.md](02_sde_training.md) | **Training**: How training works — NO SDE solvers needed! |
| [03_sde_sampling.md](03_sde_sampling.md) | **Sampling**: How to generate samples — SDE/ODE solvers used here |

### Supplementary Documents

| Document | Description |
|----------|-------------|
| [01a_diffusion_sde_view_QA.md](01a_diffusion_sde_view_QA.md) | Design principles Q&A |
| [02a_taylor_expansion.md](02a_taylor_expansion.md) | Taylor expansions in diffusion |
| [02b_fokker_plank_eq.md](02b_fokker_plank_eq.md) | Fokker-Planck equation derivation |
| [02c_ddpm_to_vpsde.md](02c_ddpm_to_vpsde.md) | From DDPM to VP-SDE (reverse direction) |
| [03b_ddim_update_coeff.md](03b_ddim_update_coeff.md) | DDIM coefficients from theory |

---

## Quick Navigation

### For Beginners
1. Start with [SDE Overview](00_sde_overview.md) for high-level understanding
2. Read [SDE Foundations](01_diffusion_sde_view.md) for detailed formulation
3. See [DDPM from SDE](02_sde_and_ddpm.md) for discrete-continuous connection
4. Understand [Solving VP-SDE](03_solving_vpsde.md) for exact solutions

### For Implementation
1. [DDPM Connection](02_sde_and_ddpm.md) — How DDPM emerges from VP-SDE
2. [DDIM Coefficients](03b_ddim_update_coeff.md) — Exact formulas for code
3. [Reverse SDE & ODE](03a_reverse_time_sde_and_proba_flow_ode.md) — Sampling methods

### For Theory Deep Dive
1. [Fokker-Planck Equation](02b_fokker_plank_eq.md) — Probability evolution PDE
2. [Taylor Expansions](02a_taylor_expansion.md) — Mathematical foundations
3. [VP-SDE from DDPM](02c_ddpm_to_vpsde.md) — Continuous limit derivation

---

## Key Concepts

### The SDE Formulation

**Forward SDE** (data → noise):

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

**Reverse SDE** (noise → data):

$$

dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

**Key insight**: Only the score function $\nabla_x \log p_t(x)$ needs to be learned.

### Variance-Preserving SDE (VP-SDE)

The most common formulation, corresponding to DDPM:

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

**Properties**:

- Preserves variance over time
- Discretizes to DDPM forward process
- Closed-form marginals: $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$

### Probability Flow ODE

Deterministic alternative to reverse SDE with **same marginals**:

$$
dx = \left[f(x, t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]dt
$$

**Key property**: DDIM is the discretization of this ODE.

---

## Comparison: Discrete vs Continuous

| Aspect | DDPM (Discrete) | SDE (Continuous) |
|--------|----------------|------------------|
| **Time** | Steps $t = 0, 1, \ldots, T$ | Continuous $t \in [0, T]$ |
| **Forward** | Markov chain | Stochastic process |
| **Notation** | $\alpha_t$, $\bar{\alpha}_t$ (products) | $\beta(t)$, integrals |
| **Sampling** | Fixed schedule | Flexible steps |
| **Theory** | Discrete probability | Stochastic calculus |
| **Flexibility** | Limited | High |

**Key connection**: DDPM is the Euler-Maruyama discretization of VP-SDE.

---

## Learning Path

### Conceptual Understanding
1. **[SDE Overview](00_sde_overview.md)** — Why SDE view matters
   - Unified framework for diffusion variants
   - Flexible sampling strategies
   - Theoretical foundations

2. **[SDE Foundations](01_diffusion_sde_view.md)** — Detailed formulation
   - Forward and reverse SDEs
   - Score functions
   - Brownian motion

### Practical Connection
3. **[DDPM from SDE](02_sde_and_ddpm.md)** — Discretization
   - Euler-Maruyama method
   - Why DDPM predicts noise
   - Forward and reverse processes

4. **[Solving VP-SDE](03_solving_vpsde.md)** — Exact solutions
   - Closed-form marginals
   - Connection to $\bar{\alpha}_t$
   - Products → integrals

### Advanced Topics
5. **[Reverse SDE & Probability Flow ODE](03a_reverse_time_sde_and_proba_flow_ode.md)** — Sampling
   - Stochastic vs deterministic
   - DDPM vs DDIM
   - The $\eta$ parameter

6. **[DDIM Coefficients](03b_ddim_update_coeff.md)** — Theory to code
   - Exact update formulas
   - Fast sampling
   - Implementation details

7. **[Fokker-Planck Equation](02b_fokker_plank_eq.md)** — Advanced theory
   - Probability density evolution
   - PDE perspective
   - Weak vs strong solutions

---

## Interactive Resources

**For hands-on learning with code**:

- **Location**: [`notebooks/diffusion/02_sde_formulation/`](../../notebooks/diffusion/02_sde_formulation/)
- **Contents**:
  - `README.md` — Comprehensive theory document
  - `02_sde_formulation.ipynb` — Interactive notebook with visualizations
  - `sde_QA.md` — Common questions answered

**8 Focused Supplements**:
1. Forward SDE Design Choices (VP/VE/sub-VP)
2. Brownian Motion Dimensionality
3. Equivalent Parameterizations (score ↔ noise ↔ $x_0$)
4. Training Loss and Denoising
5. Reverse SDE and Probability Flow ODE
6. Fokker-Planck and Effective Drift
7. Fokker-Planck Equation (detailed derivation)
8. Dimensional Analysis

---

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
