# Diffusion Models: Theory and Background

Reference materials and deep-dive documents for understanding diffusion models from first principles.

---

## 🎯 Looking for SDE Supplements?

**Most comprehensive SDE materials are now in**: [`notebooks/diffusion/02_sde_formulation/supplements/`](../../notebooks/diffusion/02_sde_formulation/supplements/)

**8 focused supplements covering**:
- Forward SDE design choices (VP/VE/sub-VP)
- Fokker-Planck equation and probability evolution
- Dimensional analysis and units
- Training loss, reverse SDEs, and more

**Start here**: [`notebooks/diffusion/02_sde_formulation/README.md`](../../notebooks/diffusion/02_sde_formulation/README.md)

---

## Purpose

This directory contains **theoretical background** and **mathematical foundations** for diffusion models. These documents complement the interactive tutorials in `notebooks/diffusion/` with deeper mathematical rigor and historical context.

**For learning**: Start with `notebooks/diffusion/` for hands-on tutorials, then return here for deeper understanding.

---

## Documents

### **[brownian_motion_tutorial.md](./brownian_motion_tutorial.md)** — Comprehensive Introduction

**Complete tutorial on Brownian motion from physical origins to diffusion models**

**Topics covered**:
- Physical origin story (Robert Brown, Einstein, Wiener)
- Random walks vs Brownian motion
- The four defining properties (and what they really mean)
- The mysterious $\sqrt{dt}$ scaling explained
- Scaling limit: from discrete to continuous (Donsker's theorem)
- Why Brownian motion powers diffusion models

**When to read**: After understanding basic SDEs, before diving deep into score matching

**Key takeaway**: Brownian motion is the universal limit of random walks and the mathematical foundation that makes diffusion models work.

---

### **[brownian_motion.md](./brownian_motion.md)** — Original Notes (Archive)

**Original Q&A-style notes on Brownian motion**

**Status**: Archived. See `brownian_motion_tutorial.md` for the comprehensive rewrite.

**Contents**:
- Random walk vs Brownian motion comparison
- Scaling limit derivation
- Origin story and physical interpretation

---

### **[brownian_motion_QA.md](./brownian_motion_QA.md)** — Original Q&A (Archive)

**Original Q&A focusing on the $\sqrt{dt}$ scaling**

**Status**: Archived. Content integrated into `brownian_motion_tutorial.md`.

**Contents**:
- Why $dw(t) \propto \sqrt{dt} \cdot \varepsilon$
- Step-by-step explanation of variance scaling
- Connection to Euler-Maruyama numerics

---

### **[sde_QA.md](./sde_QA.md)** — SDE Questions (Draft)

**Draft Q&A on SDE concepts**

**Status**: Draft/archive. For comprehensive SDE coverage, see `notebooks/diffusion/02_sde_formulation/`.

**Contents**:
- How SDEs are solved
- What models are learned
- Wiener process alternatives

**Note**: This content overlaps with `notebooks/diffusion/02_sde_formulation/sde_QA.md` which is the canonical version.

---

### **reverse_process/** — Reverse-Time SDE Theory

**Complete mathematical treatment of reversing diffusion processes**

This directory contains the theoretical foundation for generating samples from noise in diffusion models—the reverse-time SDE and its derivation.

#### **[reverse_process/reverse_process_derivation.md](./reverse_process/reverse_process_derivation.md)** — Main Derivation

**Complete derivation of the reverse-time SDE from first principles**

**Topics covered**:
- Anderson's theorem (1982) — the mathematical key to reversing SDEs
- Derivation via Fokker-Planck equation
- Why the score function $\nabla_x \log p_t(x)$ appears
- Physical intuition: drift, diffusion, and effective drift
- Connection to generative modeling

**When to read**: After understanding forward SDEs and Fokker-Planck equations

**Key result**: $dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{w}(t)$

#### **[reverse_process/reverse_process_example.md](./reverse_process/reverse_process_example.md)** — Concrete Example

**Detailed worked example: reversing a 1D Gaussian diffusion**

**Topics covered**:
- Complete setup: forward Brownian motion
- Step-by-step score calculation for Gaussian
- Physical interpretation of the score
- Reverse SDE with explicit coefficients
- Numerical verification with Python code
- Why the drift points outward (paradox explained)

**When to read**: After reading the main derivation, for concrete intuition

**Key insight**: For $p_t(x) = \mathcal{N}(0, 2Dt)$, the score is $\nabla \log p_t = -x/(2Dt)$, which guides particles back to the origin.

#### **[reverse_process/fokker_planck_derivation.md](./reverse_process/fokker_planck_derivation.md)** — Fokker-Planck Equation

**Derivation of the probability evolution equation from first principles**

**Topics covered**:
- Chapman-Kolmogorov equation
- Kramers-Moyal expansion
- Physical interpretation: drift vs. diffusion
- Conservation laws and probability current
- Examples: pure diffusion, Ornstein-Uhlenbeck
- Connection to reverse SDEs

**When to read**: Before the reverse process derivation, as foundational background

**Key result**: $\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g^2 \nabla^2 p_t$

---

### **score_network/** — Score Network Architecture Components

**Deep-dive into the architectural components of modern score networks**

This directory explains how to build neural networks that estimate $\nabla_x \log p_t(x)$ across different noise levels—the core of diffusion models.

#### **[score_network/advanced_architectures.md](./score_network/advanced_architectures.md)** — Architectures for Realistic Data

**Advanced neural network architectures for complex, real-world data**

**Topics covered**:
- **U-Net**: Dominant architecture for images and medical imaging
  - Multi-scale processing, skip connections
  - Residual blocks with time conditioning
  - Attention layers for long-range dependencies
- **Vision Transformer (DiT)**: Scalable transformer-based architecture
  - Adaptive Layer Normalization (AdaLN)
  - When to use DiT vs U-Net
- **Networks for Biological Data**: Gene expression, scRNA-seq
  - Deep residual MLPs for tabular data
  - Graph Neural Networks for pathway structure
  - Handling sparsity in single-cell data

**When to read**: When moving beyond toy examples to realistic data

**Key insight**: Match the architecture's inductive biases to your data's structure.

---

#### **[score_network/time_embedding_and_film.md](./score_network/time_embedding_and_film.md)** — Time Conditioning Components

**Deep-dive into time conditioning mechanisms used in score networks**

**Topics covered**:
- **Time Embedding**: Transform scalar $t$ to high-dimensional representation
  - Sinusoidal embeddings (multiple frequencies)
  - Why networks struggle with raw scalar inputs
  - Connection to Fourier basis
- **FiLM (Feature-wise Linear Modulation)**: Condition layers on time
  - Affine transformations: $\gamma_{\text{scale}} \odot h + \gamma_{\text{shift}}$
  - Why FiLM is more effective than concatenation
  - Implementation patterns for MLPs and CNNs
- **Comparison**: FiLM vs. concatenation vs. attention
- **Advanced topics**: Adaptive Group Normalization, multi-scale embeddings

**When to read**: When implementing score networks and needing to understand components

**Key insight**: Time embedding provides multiple frequencies for the network to understand time at different scales, while FiLM allows layer-wise adaptation to noise levels.

---

### **[forward_process_derivation.md](./forward_process_derivation.md)** — Forward SDE Solution

**Deriving the forward diffusion process: from clean data to noise**

**Topics covered**:
- VP-SDE (Variance Preserving) as the canonical example
- Solution using integrating factors
- Derivation of $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$
- Connection to noise schedules $\beta(t)$

**When to read**: After understanding basic SDEs, before reverse processes

**Supplements**:
- [`noise_schedules.md`](./noise_schedules.md) — Common noise schedule choices
- [`integrating_factor.md`](./integrating_factor.md) — Integrating factor technique
- [`alpha_definitions_derivation.md`](./alpha_definitions_derivation.md) — Origin of $\alpha(t)$ and $\bar{\alpha}_t$

---

### **[noise_schedules.md](./noise_schedules.md)** — Noise Schedule Design

**Comprehensive guide to noise schedules in diffusion models**

**Topics covered**:
- Mathematical background: $\beta(t)$, $\alpha(t)$, $\bar{\alpha}_t$
- Common schedules: linear, cosine, polynomial, sigmoid
- Properties and trade-offs
- Why cosine often works better

**When to read**: When implementing diffusion models or tuning performance

---

### **[integrating_factor.md](./integrating_factor.md)** — Integrating Factor Technique

**Explanation of integrating factors for solving linear differential equations**

**Topics covered**:
- What is an integrating factor
- How it simplifies differential equations
- Derivation of $\frac{d\mu}{dt}$ for exponential integrating factors
- Application to the forward SDE

**When to read**: When studying the forward process derivation

---

### **[alpha_definitions_derivation.md](./alpha_definitions_derivation.md)** — Origin of Alpha Definitions

**How $\alpha(t)$ and $\bar{\alpha}_t$ definitions arise from integrating factors**

**Topics covered**:
- Connection between integrating factor and signal coefficients
- Why $\bar{\alpha}_t = \exp(-\int_0^t \beta(s)\,ds)$
- Physical meaning of alpha decay

**When to read**: After forward process derivation, when seeking deeper understanding

---

### **[classifier_free_guidance.md](./classifier_free_guidance.md)** — Conditional Generation

**Comprehensive guide to classifier-free guidance for conditional diffusion models**

**Topics covered**:
- Problem: Why naive conditioning doesn't work well
- Classifier guidance (original approach with separate classifier)
- Classifier-free guidance (elegant solution without classifier)
- Training procedure (condition dropping)
- Guidance scale and fidelity vs. diversity trade-off
- Implementation in both DDPM and SDE views
- Variants: dynamic guidance, multi-conditional, negative prompting

**When to read**: When building conditional diffusion models (text-to-image, class-conditional, etc.)

**Key insight**: Train one model for both conditional and unconditional generation by randomly dropping conditions during training, then amplify the difference at sampling time.

---

### **history/** — Historical Development

**How diffusion models evolved and unified**

#### **[history/diffusion_models_development.md](./history/diffusion_models_development.md)** — Complete Historical Timeline

**Traces the development of diffusion models from multiple perspectives**

**Topics covered**:
- Timeline: Score matching (2005) → DDPM (2020) → SDE view (2021)
- Was DDPM derived from SDEs? (No—retrospective unification)
- Three views: Variational, Score-Based, Flow-Based
- How the SDE view unified them all
- Why understanding history clarifies the "multiple views" confusion

**When to read**: After understanding both DDPM and SDE views, when seeking big-picture understanding

**Key insight**: DDPM was developed independently as a discrete-time model. The SDE view came later and revealed that DDPM, NCSN, and flow-based models are all the same underlying process.

**Includes**: Diagram showing the convergence of three perspectives to continuous-time formulation

---

## Organization Strategy

### `docs/diffusion/` vs `notebooks/diffusion/`

**This directory (`docs/diffusion/`)**:
- **Purpose**: Reference materials, mathematical deep-dives, historical context
- **Format**: Markdown documents with rigorous derivations
- **Style**: Tutorial/blog style but with more mathematical detail
- **Audience**: Researchers, those seeking deeper understanding

**`notebooks/diffusion/`**:
- **Purpose**: Interactive learning, hands-on coding, visualization
- **Format**: Jupyter notebooks with executable code
- **Style**: Step-by-step tutorials with examples
- **Audience**: Practitioners, those learning by doing

**Relationship**: These directories complement each other. Start with notebooks for intuition, come here for rigor.

---

## Recommended Reading Order

### For Understanding Brownian Motion

1. **Start**: `brownian_motion_tutorial.md` (this directory)
   - Complete introduction from physics to mathematics
   - Explains the $\sqrt{dt}$ scaling thoroughly
   - Connects to diffusion models

2. **Practice**: `notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb`
   - Visualize Brownian motion paths
   - See the scaling in action with code

3. **Deep dive**: `notebooks/diffusion/02_sde_formulation/supplements/02_brownian_motion_dimensionality.md`
   - Why $w(t) \in \mathbb{R}^d$, not scalar

### For Understanding SDEs

1. **Start**: `notebooks/diffusion/02_sde_formulation/sde_formulation.md`
   - Core SDE theory for diffusion models

2. **Clarify**: `notebooks/diffusion/02_sde_formulation/sde_QA.md`
   - Common questions answered

3. **Supplements**: `notebooks/diffusion/02_sde_formulation/supplements/`
   - Eight focused deep-dives on specific topics

### For Understanding Forward Process (Data → Noise)

1. **Start**: `forward_process_derivation.md` (this directory)
   - Complete derivation of $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$

2. **Background**: `integrating_factor.md` (this directory)
   - Mathematical technique used in the derivation

3. **Deep dive**: `alpha_definitions_derivation.md` (this directory)
   - Why the definitions look the way they do

4. **Practice**: `noise_schedules.md` (this directory)
   - Choosing and tuning noise schedules

### For Understanding Reverse Process (Noise → Data)

1. **Foundation**: `reverse_process/fokker_planck_derivation.md`
   - How probability distributions evolve under SDEs
   - Essential background for understanding reverse processes

2. **Main theory**: `reverse_process/reverse_process_derivation.md`
   - Complete derivation of the reverse-time SDE
   - Anderson's theorem and the score function

3. **Concrete example**: `reverse_process/reverse_process_example.md`
   - Worked example with explicit calculations
   - Numerical verification code

4. **Practice**: `notebooks/diffusion/02_sde_formulation/supplements/`
   - Supplement 03: Training loss and denoising
   - Supplement 04: Score matching
   - Supplement 05: Reverse SDE implementation

### For Implementing Score Networks

1. **Architecture basics**: `dev/notebooks/diffusion/02_sde_formulation/score_network_architecture.md`
   - Activation functions (SiLU)
   - Basic MLP implementation
   - Why `y.sum().backward()`

2. **Advanced architectures**: `score_network/advanced_architectures.md` (this directory)
   - U-Net for images and medical imaging
   - Vision Transformer (DiT) for large-scale training
   - Architectures for gene expression and scRNA-seq
   - When to use which architecture

3. **Component deep-dive**: `score_network/time_embedding_and_film.md` (this directory)
   - Time embedding: sinusoidal representations
   - FiLM: feature-wise linear modulation
   - Implementation patterns and debugging tips

4. **Practice**: `notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb`
   - Train a score network on toy 2D data
   - See time conditioning in action
   - Sample from the reverse SDE

---

## Related Resources

### In This Repository

- **Interactive tutorials**: `notebooks/diffusion/`
  - `01_ddpm_basics.ipynb`: DDPM on gene expression
  - `02_sde_formulation/`: Complete SDE tutorial package

- **Theory documents**: `docs/`
  - `score_matching/`: Score functions and Fisher/Stein scores
  - `EBM/`: Energy-based models

### External References

- **Einstein (1905)**: Investigations on the Theory of the Brownian Movement
- **Donsker (1951)**: Invariance principle for random walks
- **Song et al. (2021)**: Score-Based Generative Modeling through SDEs
- **Øksendal (2003)**: Stochastic Differential Equations textbook

---

## Contributing

When adding new documents to this directory:

1. **Check for overlap**: See if content fits better in `notebooks/diffusion/`
2. **Maintain style**: Tutorial/blog style with rigorous mathematics
3. **Add to this README**: Update the documents list and reading order
4. **Cross-reference**: Link to related notebooks and docs

---

## Status

### Core Theory Documents
- ✅ **brownian_motion_tutorial.md**: Complete comprehensive tutorial
- ✅ **forward_process_derivation.md**: Complete with supplements
- ✅ **reverse_process/**: Complete directory with all three documents
  - ✅ reverse_process_derivation.md: Full reverse SDE derivation
  - ✅ reverse_process_example.md: Worked 1D Gaussian example
  - ✅ fokker_planck_derivation.md: Probability evolution equation
- ✅ **score_network/**: Architecture components directory
  - ✅ advanced_architectures.md: U-Net, DiT, and architectures for realistic data
  - ✅ time_embedding_and_film.md: Time embedding and FiLM conditioning components

### Supplement Documents
- ✅ **noise_schedules.md**: Complete guide to noise schedules
- ✅ **integrating_factor.md**: Mathematical technique explained
- ✅ **alpha_definitions_derivation.md**: Origin of alpha definitions

### Archived Documents
- 📦 **brownian_motion.md**: Archived (superseded by tutorial)
- 📦 **brownian_motion_QA.md**: Archived (content integrated into tutorial)
- 📝 **sde_QA.md**: Draft (see notebooks version for canonical content)

---

**Next Steps**:
- After understanding Brownian motion → `notebooks/diffusion/02_sde_formulation/` for SDE basics
- After understanding forward/reverse processes → `notebooks/diffusion/02_sde_formulation/supplements/` for implementation details
- For practical implementation → Start with DDPM notebook, then move to SDE formulation
