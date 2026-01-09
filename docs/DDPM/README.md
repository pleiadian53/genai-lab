# Denoising Diffusion Probabilistic Models (DDPM)

This directory contains comprehensive documentation on **Denoising Diffusion Probabilistic Models (DDPM)**, the foundational discrete-time diffusion model introduced by Ho et al. (2020).

---

## Overview

DDPM is a class of generative models that learns to generate data by reversing a gradual noising process. It achieves state-of-the-art results across multiple domains and provides a principled probabilistic framework for generation.

### Why Study DDPM?

1. **Foundational model**: Understanding DDPM is essential for modern diffusion models
2. **Theoretical depth**: Connects variational inference, score matching, and SDEs
3. **Practical success**: State-of-the-art image generation, protein design, molecular generation
4. **Training simplicity**: Simple MSE loss, no adversarial training
5. **Interpretability**: Clear probabilistic interpretation via ELBO

---

## Documents in This Directory

### Core Theory

1. **[01_ddpm_foundations.md](01_ddpm_foundations.md)** — Mathematical Foundations ⭐
   - Forward and reverse processes
   - Closed-form marginals
   - Variational lower bound (ELBO)
   - Noise prediction parameterization
   - Connection to score matching
   - Training and sampling algorithms

### Practical Implementation

2. **[02_ddpm_training.md](02_ddpm_training.md)** — Training Details ⭐ NEW
   - Loss function variants (simple vs. weighted ELBO)
   - Architecture choices (U-Net, MLP, DiT)
   - Time embeddings and conditioning strategies
   - Conditional generation methods
   - Hyperparameter tuning
   - Training tips and common issues

3. **[03_ddpm_sampling.md](03_ddpm_sampling.md)** — Sampling Methods ⭐ NEW
   - DDPM ancestral sampling (stochastic)
   - DDIM deterministic sampling
   - Fast sampling via step skipping
   - Classifier-free guidance (see detailed guide below)
   - Quality vs. speed trade-offs

### Extensions and Advanced Topics

4. **[Classifier-Free Guidance](../diffusion/classifier_free_guidance.md)** — Comprehensive Guide ⭐
   - Conditional generation without classifiers
   - Training procedure (condition dropping)
   - Guidance scale and its effects
   - Implementation in both DDPM and SDE views
   - Variants: dynamic guidance, negative prompting

### Coming Soon

5. **04_ddpm_extensions.md** — Extensions and Variants (Planned)
   - Improved DDPM (learned variance)
   - Latent diffusion models
   - Discrete diffusion
   - Domain-specific adaptations

---

## Quick Navigation

### For Beginners

**Start here**: [DDPM Foundations](01_ddpm_foundations.md)

This document provides a complete mathematical introduction from first principles, covering:
- The forward noising process
- The reverse denoising process
- Training via ELBO
- The simple noise prediction loss
- Sampling algorithms

**Then**: 
1. Read [DDPM Training](02_ddpm_training.md) for practical training details
2. Read [DDPM Sampling](03_ddpm_sampling.md) for sampling algorithms
3. Work through the [DDPM Basics Notebook](../../notebooks/diffusion/01_ddpm/01_ddpm_basics.ipynb) for hands-on implementation

### For Deep Dive

After understanding the foundations:
1. Study the [SDE perspective](../SDE/02_sde_and_ddpm.md) to see DDPM as a discretization
2. Review the [continuous limit](../SDE/02c_ddpm_to_vpsde.md) to understand VP-SDE
3. Explore [DDIM update coefficients](../SDE/03b_ddim_update_coeff.md) for exact formulas
4. Read [Reverse SDE & Probability Flow ODE](../SDE/03a_reverse_time_sde_and_proba_flow_ode.md) for sampling theory

### For Implementation

- **Documentation**: 
  - [Training Details](02_ddpm_training.md) — Architectures, loss functions, hyperparameters
  - [Sampling Methods](03_ddpm_sampling.md) — DDPM, DDIM, fast sampling, guidance

- **Notebook**: [01_ddpm_basics.ipynb](../../notebooks/diffusion/01_ddpm/01_ddpm_basics.ipynb)
  - Complete PyTorch implementation
  - Gene expression application
  - Conditional generation
  - Training and sampling code

- **Source code**: `src/genailab/diffusion/`
  - Production-ready implementations
  - Modular components
  - Reusable utilities

---

## Key Concepts

### Forward Process (Data → Noise)

Gradually add Gaussian noise over $T$ steps:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

**Closed-form**: Jump directly to any timestep:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### Reverse Process (Noise → Data)

Learn to denoise step by step:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$

### Training Objective

Simple MSE loss on noise prediction:

$$
L = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

### Sampling Algorithm

```
1. Start with x_T ~ N(0, I)
2. For t = T, ..., 1:
   - Predict noise: ε_θ(x_t, t)
   - Compute mean: μ_θ
   - Add noise: x_{t-1} = μ_θ + σ_t * z
3. Return x_0
```

---

## Connections to Other Frameworks

### Score Matching

DDPM implicitly learns the score function:

$$
\nabla_{x_t} \log q(x_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon
$$

Predicting noise ≈ predicting score (up to scaling).

### Stochastic Differential Equations (SDEs)

DDPM is a discretization of the VP-SDE:

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw
$$

See: [VP-SDE to DDPM](../SDE/02_sde_and_ddpm.md)

### Variational Autoencoders (VAEs)

DDPM can be viewed as a hierarchical VAE with:
- Markovian latent structure
- Fixed encoder (forward process)
- Learned decoder (reverse process)

---

## Learning Path

### Recommended Order

1. **Foundations** (this directory)
   - Start with [DDPM Foundations](01_ddpm_foundations.md)
   - Understand forward/reverse processes
   - Learn the training objective

2. **Practical Training** (this directory)
   - Read [DDPM Training](02_ddpm_training.md)
   - Learn architecture choices
   - Understand hyperparameters and conditioning

3. **Sampling Methods** (this directory)
   - Read [DDPM Sampling](03_ddpm_sampling.md)
   - Compare DDPM vs. DDIM
   - Learn fast sampling and guidance

4. **Implementation** (notebooks)
   - Work through [DDPM Basics](../../notebooks/diffusion/01_ddpm/01_ddpm_basics.ipynb)
   - Implement training loop
   - Generate samples

5. **Theory** (SDE directory)
   - Study [SDE View](../SDE/01_diffusion_sde_view.md)
   - Understand continuous-time perspective
   - Connect discrete and continuous

6. **Advanced Topics**
   - Latent diffusion models
   - Domain-specific applications
   - State-of-the-art techniques

---

## Applications

### Image Generation

- **Unconditional**: Generate diverse images from noise
- **Conditional**: Text-to-image, class-conditional
- **Inpainting**: Fill missing regions
- **Super-resolution**: Upscale low-resolution images

### Biological Data

- **Gene expression**: Generate cell states (see [DDPM Basics](../../notebooks/diffusion/01_ddpm_basics.ipynb))
- **Protein design**: Generate protein sequences and structures
- **Drug response**: Predict perturbation effects (scPPDM)
- **Single-cell data**: Generate realistic cell populations

### Other Domains

- **Audio**: Speech synthesis, music generation
- **Video**: Frame prediction, video synthesis
- **Molecular**: Drug design, molecular generation
- **3D**: Point clouds, meshes, NeRF

---

## Comparison with Other Generative Models

| Model | Training | Sampling | Quality | Diversity | Likelihood |
|-------|----------|----------|---------|-----------|------------|
| **DDPM** | Stable | Slow | Excellent | High | Approximate |
| **GAN** | Unstable | Fast | Excellent | Medium | No |
| **VAE** | Stable | Fast | Good | High | Exact |
| **Flow** | Stable | Fast | Good | High | Exact |

**DDPM advantages**:
- Training stability (no adversarial training)
- High sample quality
- Flexible conditioning
- Theoretical foundations

**DDPM disadvantages**:
- Slow sampling (1000 steps)
- Approximate likelihood
- High computational cost

---

## Historical Context

### Timeline

- **2015**: Sohl-Dickstein et al. introduce diffusion models (ICML)
- **2019**: Song & Ermon connect to score matching (NeurIPS)
- **2020**: Ho et al. introduce DDPM (NeurIPS) — breakthrough results
- **2021**: Nichol & Dhariwal improve DDPM (ICML)
- **2021**: Song et al. introduce SDE framework (ICLR 2021)
- **2021**: Dhariwal & Nichol beat GANs (NeurIPS)
- **2022**: Rombach et al. introduce Latent Diffusion (CVPR)
- **2022**: Ramesh et al. introduce DALL-E 2 (arXiv)

### Key Papers

1. **Sohl-Dickstein et al. (2015)**: Deep Unsupervised Learning using Nonequilibrium Thermodynamics
2. **Ho et al. (2020)**: Denoising Diffusion Probabilistic Models
3. **Song et al. (2021)**: Score-Based Generative Modeling through SDEs
4. **Nichol & Dhariwal (2021)**: Improved Denoising Diffusion Probabilistic Models
5. **Dhariwal & Nichol (2021)**: Diffusion Models Beat GANs on Image Synthesis

---

## Related Documentation

### In This Repository

- **SDE Formulation**: [docs/SDE/](../SDE/)
  - Continuous-time perspective
  - Fokker-Planck equation
  - Score matching connections
  - [DDPM to VP-SDE](../SDE/02c_ddpm_to_vpsde.md)

- **Diffusion Models**: [docs/diffusion/](../diffusion/)
  - [Historical Development](../diffusion/history/diffusion_models_development.md) — How DDPM, score-based, and flow-based models unified
  - [Classifier-Free Guidance](../diffusion/classifier_free_guidance.md) — Conditional generation
  - [Brownian Motion Tutorial](../diffusion/brownian_motion_tutorial.md)
  - General diffusion theory

- **Notebooks**: [notebooks/diffusion/](../../notebooks/diffusion/)
  - DDPM basics implementation
  - SDE formulation with code
  - Advanced topics

### External Resources

- **Original papers**: See references in [01_ddpm_foundations.md](01_ddpm_foundations.md)
- **Tutorials**: 
  - [Lilian Weng's blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - [Yang Song's blog](https://yang-song.net/blog/2021/score/)
- **Code**:
  - [Official DDPM repo](https://github.com/hojonathanho/diffusion)
  - [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

---

## Contributing

This documentation is part of the `genai-lab` project. To contribute:

1. Follow the tutorial/blog style established in existing documents
2. Use proper LaTeX notation (`$$...$$` for blocks, `$...$` for inline)
3. Include intuition alongside mathematics
4. Add examples and visualizations where helpful
5. Link to related documents

---

## Summary

DDPM is a foundational generative model that:
- Learns to reverse a gradual noising process
- Trains via simple MSE loss on noise prediction
- Achieves state-of-the-art sample quality
- Connects to score matching and SDEs
- Provides a principled probabilistic framework

**Start with**: [DDPM Foundations](01_ddpm_foundations.md) for a complete mathematical introduction.
