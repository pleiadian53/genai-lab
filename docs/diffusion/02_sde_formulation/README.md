# SDE Formulation for Diffusion Models

A comprehensive tutorial on understanding diffusion models through the lens of Stochastic Differential Equations (SDEs).

---

## Overview

This directory contains a complete learning path for understanding the SDE formulation of diffusion models, from basic concepts to advanced theory. The materials progress from interactive code tutorials to theoretical deep-dives.

**Why SDEs?** The SDE perspective unifies discrete-time DDPM, score-based models, and DDIM into a single continuous-time framework. It provides:
- **Mathematical clarity**: Clean separation between design choices and learning
- **Flexibility**: Easy to design custom diffusion processes
- **Generality**: Discrete DDPM is a special case
- **Interpretability**: Clear connection to probability theory and stochastic processes

---

## Learning Path

### 1. Start Here: Core Materials

Read/work through these in order:

#### **[sde_formulation.md](./sde_formulation.md)** — Comprehensive Theory
- What is an SDE? (ODEs vs SDEs)
- Understanding each symbol: $x(t)$, $f(x,t)$, $g(t)$, $w(t)$
- What is chosen vs what is learned
- Training workflow step-by-step
- Sampling workflow
- Connection to DDPM
- Concrete example: VP-SDE

**Start here** if you want a complete theoretical foundation.

#### **[02_sde_formulation.ipynb](./02_sde_formulation.ipynb)** — Interactive Tutorial
- Visualize Brownian motion
- Simulate forward SDEs (data → noise)
- Implement score matching training
- Sample from reverse SDEs (noise → data)
- Compare VP-SDE and probability flow ODE

**Start here** if you prefer learning by coding and visualization.

### 2. Common Questions

#### **[sde_QA.md](./sde_QA.md)** — FAQ and Conceptual Clarifications

Addresses frequently asked questions:
- How is an SDE system solved?
- What models are learned in the SDE formulation?
- What do we use the learned score for?
- Is Brownian motion the only way to model randomness?
- Why don't diffusion models use jump processes, stochastic volatility, etc.?

**Read this** after going through the core materials to solidify understanding.

---

## 3. Deep Dives: Supplementary Materials

These documents provide focused deep-dives into specific topics. Read them in order for systematic understanding, or jump to specific topics as needed.

### **[01. Forward SDE Design Choices](./supplements/01_forward_sde_design_choices.md)** ⭐ NEW

**Topic**: Understanding $f(x,t)$ and $g(t)$ — what they are and how to choose them

**Key insights**:

- **Core principle**: $f(x,t)$ and $g(t)$ are design choices, not learned
- Three standard SDEs: VP-SDE, VE-SDE, sub-VP-SDE
- Why these specific functions? (mathematical tractability, variance behavior)
- Design considerations: closed-form marginals, SNR decay, connection to DDPM/NCSN
- Practical recommendations for choosing your forward SDE

**When to read**: **Start here** — This is foundational for understanding training and sampling

---

### **[02. Brownian Motion Dimensionality](./supplements/02_brownian_motion_dimensionality.md)**

**Topic**: Why $w(t)$ and $x(t)$ have the same dimension

**Key insights**:

- Brownian motion is $d$-dimensional, not scalar
- Each pixel/feature has its own independent Brownian path
- Noise term $g(t)dw(t)$ must match $x(t)$ dimensionality

**When to read**: After understanding basic SDE notation (clarifies a common confusion)

---

### **[03. Equivalent Parameterizations](./supplements/03_equivalent_parameterizations.md)**

**Topic**: Score vs noise vs clean data prediction

**Key insights**:

- Three ways to parameterize the neural network output
- Mathematical equivalence: $s_\theta \leftrightarrow \varepsilon_\theta \leftrightarrow \hat{x}_0$
- Conversion formulas between parameterizations
- Why DDPM predicts noise but score-based models predict score

**When to read**: When understanding what the neural network learns

---

### **[04. Training Loss and Denoising](./supplements/04_training_loss_and_denoising.md)**

**Topic**: Why predicting score = predicting noise = learning to denoise

**Key insights**:

- Derivation of the score matching loss
- Why $-\varepsilon/\sigma_t$ is the target score
- Connection between denoising and score estimation
- How $g(t)$ from the forward SDE appears in the loss
- Intuition: score points toward cleaner versions of data

**When to read**: When understanding the training objective (requires supplement 01)

---

### **[05. Reverse SDE and Probability Flow ODE](./supplements/05_reverse_sde_and_probability_flow_ode.md)**

**Topic**: Sampling mechanics and stochastic vs deterministic generation

**Key insights**:

- Term-by-term interpretation of reverse SDE
- How $f(x,t)$ and $g(t)$ from forward SDE appear in reverse SDE
- Why the score term reverses diffusion
- Probability flow ODE: deterministic alternative
- Trade-offs: SDE (diverse) vs ODE (fast, deterministic)
- Connection to DDIM

**When to read**: When understanding sampling/generation (requires supplement 01)

---

### **[06. Fokker-Planck Equation and Effective Drift](./supplements/06_fokker_planck_and_effective_drift.md)**

**Topic**: Advanced theory connecting SDEs to PDEs

**Key insights**:

- Fokker-Planck equation: from particle trajectories to probability density evolution
- Why probability flow ODE has the same marginals as the SDE
- Effective drift interpretation
- Connection to transport theory

**When to read**: For advanced theoretical understanding (optional for practitioners)

---

## Quick Reference

### File Organization

```
02_sde_formulation/
├── README.md                          # This file (index)
├── sde_formulation.md                 # Core theory document
├── sde_QA.md                          # Common questions
├── 02_sde_formulation.ipynb          # Interactive code tutorial
│
└── supplements/                       # Deep-dive documents
    ├── 01_forward_sde_design_choices.md
    ├── 02_brownian_motion_dimensionality.md
    ├── 03_equivalent_parameterizations.md
    ├── 04_training_loss_and_denoising.md
    ├── 05_reverse_sde_and_probability_flow_ode.md
    ├── 06_fokker_planck_and_effective_drift.md
    ├── 07_fokker_planck_equation.md            ⭐ NEW
    └── 08_dimensional_analysis.md              ⭐ NEW
```

### Suggested Reading Orders

#### **For Practitioners** (focus on implementation):
1. `02_sde_formulation.ipynb` (code first)
2. `sde_formulation.md` (theory)
3. `supplements/01_forward_sde_design_choices.md` (understand $f$ and $g$)
4. `supplements/03_equivalent_parameterizations.md`
5. `supplements/04_training_loss_and_denoising.md`
6. `supplements/05_reverse_sde_and_probability_flow_ode.md`
7. `supplements/07_fokker_planck_equation.md` (optional: deeper PDE connection)

#### **For Theorists** (focus on mathematics):
1. `sde_formulation.md` (theory first)
2. `sde_QA.md` (clarifications)
3. All supplements in order (01 → 08)
4. `02_sde_formulation.ipynb` (see theory in action)

#### **For Building Intuition**:
- `supplements/08_dimensional_analysis.md` (anytime - powerful sanity checks)
- `supplements/07_fokker_planck_equation.md` (understand probability evolution)
- `supplements/02_brownian_motion_dimensionality.md` (clarify vector dimensions)

#### **For Quick Reference**:
- Jump to `sde_QA.md` for specific questions
- Use supplements as needed for deep-dives
- **Start with supplement 01** if confused about what's fixed vs learned
- **Use supplement 08** for dimensional sanity checks

---

## Key Concepts Summary

### What You'll Learn

1. **SDEs describe continuous-time random processes**
   - $dx = f(x,t)dt + g(t)dw(t)$
   - Drift $f$ (deterministic) + diffusion $g$ (random)

2. **Only the score function is learned**
   - $s_\theta(x,t) \approx \nabla_x \log p_t(x)$
   - Everything else ($f$, $g$, forward process) is fixed

3. **Training = denoising score matching**
   - Learn to predict noise (or equivalently, the score)
   - No SDE solving during training (use closed-form marginals)

4. **Sampling = solving reverse SDE**
   - Numerically integrate from noise to data
   - Can use stochastic (SDE) or deterministic (ODE) sampling

5. **Brownian motion enables tractability**
   - Exact reverse-time equations (Anderson's theorem)
   - Closed-form marginals for many SDEs
   - Stable training and sampling

---

## Prerequisites

- **Mathematics**: Calculus, probability (Gaussian distributions), basic differential equations
- **Programming**: Python, PyTorch/NumPy basics
- **Machine Learning**: Neural networks, gradient descent, loss functions
- **Diffusion Models**: Helpful to know DDPM basics (see `../01_ddpm_basics.ipynb`)

---

## Next Steps

After mastering the SDE formulation:

1. **Apply to real problems**: See `../03_scPPDM_tutorial.ipynb` for drug-response prediction
2. **Implement custom SDEs**: Design diffusion processes for your domain
3. **Explore variants**: VE-SDE, sub-VP-SDE, conditional generation
4. **Read research papers**: Song et al. (2021), Ho et al. (2020), Karras et al. (2022)

---

## References

### Primary Papers

- **Song et al. (2021)**: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
  - The definitive paper on SDE formulation
  - Introduces VP-SDE, VE-SDE, and probability flow ODE

- **Ho et al. (2020)**: [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
  - Original discrete-time formulation
  - Shows connection to score matching

- **Anderson (1982)**: Reverse-time diffusion equation models
  - Original theorem on reverse-time SDEs
  - Foundation for all modern diffusion models

### Textbooks

- **Øksendal (2003)**: *Stochastic Differential Equations: An Introduction with Applications*
  - Comprehensive SDE theory
  - Rigorous mathematical treatment

- **Karatzas & Shreve (1991)**: *Brownian Motion and Stochastic Calculus*
  - Advanced reference
  - Detailed proofs and theory

### Related Topics

- **Score matching**: Hyvärinen (2005)
- **Langevin dynamics**: Neal (2011)
- **Flow matching**: Lipman et al. (2023)
- **Rectified flows**: Liu et al. (2022)

---

## Contributing

These materials are part of the `genai-lab` project. For questions or suggestions:
- See main project: [`../../README.md`](../../README.md)
- Theory documents: [`../../../docs/`](../../../docs/)
- Production examples: [`../../../examples/`](../../../examples/)

---

**Happy learning!** 🎓
