# Generative AI Learning Roadmap

A systematic progression from classical latent-variable models to modern generative architectures, with implementation milestones for computational biology applications.

---

## Overview

```
VAE → β-VAE → Score Matching → DDPM → Flow Matching → EBMs → JEPA → World Models
 │                                │
 └── cVAE (conditional)           └── Classifier-Free Guidance
```

---

## Stage 1: Variational Autoencoders (VAE)

**Status**: ✅ Implemented

### Key Concepts

- **ELBO**: Evidence Lower Bound as training objective
- **Reparameterization trick**: Making sampling differentiable
- **KL regularization**: Balancing reconstruction vs latent structure
- **Amortized inference**: Encoder learns approximate posterior

### Core Equations

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \| p(z))
$$

### Implementation

- `src/genailab/model/vae.py` — VAE architecture
- `src/genailab/model/encoders.py` — Encoder networks
- `src/genailab/model/decoders.py` — Decoder networks
- `src/genailab/objectives/losses.py` — ELBO loss

### Milestones

- [x] Basic VAE with MLP encoder/decoder
- [x] Toy bulk expression dataset
- [x] Training loop with validation
- [ ] Latent space visualization (UMAP/t-SNE)
- [ ] Interpolation demos

### Documentation

- [VAE Theory](VAE/VAE.md)

---

## Stage 2: Conditional VAE (cVAE)

**Status**: ✅ Implemented

### Key Concepts

- **Conditional generation**: $p_\theta(x | z, y)$
- **Label injection**: Concatenating conditions to encoder/decoder
- **Disentanglement**: Separating content from style/condition

### Core Equations

$$
\mathcal{L}_{\text{cVAE}} = \mathbb{E}_{q_\phi(z|x,y)}[\log p_\theta(x|z,y)] - \mathrm{KL}(q_\phi(z|x,y) \| p(z))
$$

### Implementation

- `src/genailab/model/conditioning.py` — Condition embedding
- `src/genailab/model/vae.py` — CVAE class

### Milestones

- [x] Condition embedding (tissue, disease, batch)
- [x] Conditional encoder/decoder
- [ ] Counterfactual generation (change condition, keep latent)
- [ ] Condition interpolation

---

## Stage 3: β-VAE and Disentanglement

**Status**: 🔲 Planned

### Key Concepts

- **β parameter**: Trade-off between reconstruction and disentanglement
- **Information bottleneck**: Limiting mutual information $I(x; z)$
- **Factor VAE**: Encouraging factorial posterior

### Core Equations

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot \mathrm{KL}(q(z|x) \| p(z))
$$

### Milestones

- [ ] Implement β-VAE with configurable β
- [ ] Disentanglement metrics (DCI, MIG)
- [ ] Latent traversal visualization
- [ ] Compare β values on gene expression

### References

- Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)

---

## Stage 4: Importance Weighted Autoencoders (IWAE)

**Status**: 🔲 Planned

### Key Concepts

- **Tighter bound**: Multiple samples for better gradient estimates
- **Importance weighting**: Reweighting samples by likelihood ratio
- **Signal-to-noise**: Trade-off with number of samples

### Core Equations

$$
\mathcal{L}_{\text{IWAE}} = \mathbb{E}_{z_1, \ldots, z_K \sim q(z|x)} \left[ \log \frac{1}{K} \sum_{k=1}^{K} \frac{p(x, z_k)}{q(z_k | x)} \right]
$$

### Milestones

- [ ] Implement IWAE with K samples
- [ ] Compare ELBO tightness vs VAE
- [ ] Analyze gradient variance

### References

- Burda et al., "Importance Weighted Autoencoders" (2016)

---

## Stage 5: Score Matching & Denoising

**Status**: ✅ Implemented

### Key Concepts

- **Score function**: $\nabla_x \log p(x)$
- **Denoising score matching**: Learn score from noisy samples
- **Langevin dynamics**: Sampling via score-guided MCMC

### Core Equations

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{x, \tilde{x}} \left[ \| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x} | x) \|^2 \right]
$$

### Implementation

- `src/genailab/diffusion/sde.py` — VP-SDE, VE-SDE with noise schedules
- `src/genailab/diffusion/architectures.py` — Score networks (MLP, TabularScoreNetwork, UNet2D, UNet3D)
- `src/genailab/diffusion/training.py` — Score matching training loops

### Milestones

- [x] Implement score network (MLP, attention-based)
- [x] Denoising score matching loss
- [x] VP-SDE and VE-SDE formulations
- [x] Noise schedules (linear, cosine)
- [ ] Langevin sampling (basic implementation exists)
- [ ] Noise-conditional score networks (NCSN)

### References

- Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (2019)
- `dev/references/Principles of diffusion models.pdf` — Section 2

---

## Stage 6: Denoising Diffusion (DDPM) & SDE Framework

**Status**: ✅ Implemented

### Key Concepts

- **Forward process**: Gradually add noise $q(x_t | x_{t-1})$
- **Reverse process**: Learn to denoise $p_\theta(x_{t-1} | x_t)$
- **Noise schedule**: Linear, cosine, learned
- **SDE formulation**: Continuous-time diffusion via SDEs

### Core Equations

**Forward:**

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

**Reverse:**

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$

### Implementation

- `src/genailab/diffusion/` — Complete diffusion module
  - `sde.py` — VP-SDE, VE-SDE base classes
  - `architectures.py` — UNet2D, UNet3D, TabularScoreNetwork
  - `training.py` — `train_score_network`, `train_image_diffusion`
  - `sampling.py` — Reverse SDE, probability flow ODE
- `notebooks/diffusion/` — Educational tutorials (01-04)
- `scripts/diffusion/` — Production training scripts

### Milestones

- [x] Forward diffusion process (VP-SDE, VE-SDE)
- [x] Score prediction network (U-Net for images, MLP+attention for tabular)
- [x] Training loop with checkpointing
- [x] Reverse SDE sampling
- [x] Probability flow ODE sampling
- [x] Medical imaging diffusion (synthetic X-rays)
- [ ] DDIM fast sampling
- [ ] Conditional diffusion (classifier-free guidance)

### References

- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (2021)
- `dev/references/Principles of diffusion models.pdf` — Sections 3-4

---

## Stage 7: Flow Matching

**Status**: 🔲 Planned

### Key Concepts

- **Continuous normalizing flows**: ODE-based density transformation
- **Optimal transport**: Straight-line interpolation paths
- **Simulation-free training**: No ODE solver during training

### Core Equations

$$
\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim p_0, \quad x(1) \sim p_1
$$

### Milestones

- [ ] Implement flow matching loss
- [ ] Conditional flow matching
- [ ] Compare with DDPM on gene expression

### References

- Lipman et al., "Flow Matching for Generative Modeling" (2023)

---

## Stage 8: Energy-Based Models (EBMs)

**Status**: 🔲 Planned

### Key Concepts

- **Energy function**: $E_\theta(x)$ defines unnormalized density
- **Contrastive divergence**: Approximate gradient of log-partition
- **MCMC sampling**: Langevin, HMC for generation

### Core Equations

$$
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}
$$

### Milestones

- [ ] Energy network architecture
- [ ] Contrastive divergence training
- [ ] Langevin sampling
- [ ] Noise contrastive estimation (NCE)

### References

- LeCun et al., "A Tutorial on Energy-Based Learning" (2006)

---

## Stage 9: Joint Embedding Predictive Architecture (JEPA)

**Status**: 🔲 Planned

### Key Concepts

- **Latent prediction**: Predict in embedding space, not pixel space
- **Self-supervised**: No reconstruction, no contrastive negatives
- **World models**: Learn dynamics without generation

### Architecture

```
x → Encoder → z_x
              ↓
         Predictor → ẑ_y
              ↑
y → Encoder → z_y (target)
```

### Milestones

- [ ] Joint embedding architecture
- [ ] Predictor network
- [ ] Variance-Invariance-Covariance (VICReg) regularization
- [ ] Apply to gene expression time series

### References

- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023)

---

## Stage 10: World Models

**Status**: 🔲 Planned

### Key Concepts

- **Latent dynamics**: $z_{t+1} = f_\theta(z_t, a_t)$
- **Imagination**: Plan in latent space without environment
- **Model-based RL**: Use world model for policy learning

### Applications in Biology

- Perturbation prediction (action = drug/knockdown)
- Trajectory modeling (action = time step)
- Counterfactual reasoning

### Milestones

- [ ] Latent dynamics model
- [ ] Recurrent state-space model (RSSM)
- [ ] Dreamer-style imagination
- [ ] Apply to perturbation biology

### References

- Ha & Schmidhuber, "World Models" (2018)
- Hafner et al., "Dream to Control" (2020)

---

## Cross-Cutting Themes

### Evaluation Metrics

| Model Type | Metrics |
|------------|---------|
| VAE/cVAE | ELBO, reconstruction MSE, KL, FID |
| Diffusion | FID, IS, likelihood bounds |
| EBM | Energy histograms, sample quality |
| JEPA | Downstream task performance |

### Computational Biology Applications

1. **Gene expression generation**: Conditional on cell type, disease
2. **Perturbation prediction**: What happens if we knock out gene X?
3. **Trajectory inference**: Developmental or disease progression
4. **Data augmentation**: Generate synthetic training data
5. **Representation learning**: Embeddings for downstream tasks

---

## Reading List

### Foundational

1. Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
2. Rezende et al., "Stochastic Backpropagation" (2014)
3. `dev/references/Principles of diffusion models.pdf`

### Surveys

1. `dev/references/Diffusion Models- A Comprehensive Survey of Methods and Applications.pdf`
2. Bond-Taylor et al., "Deep Generative Modelling: A Comparative Review" (2022)

### Biology-Specific

1. Lopez et al., "Deep generative modeling for single-cell transcriptomics" (scVI, 2018)
2. Lotfollahi et al., "scGen: Predicting single-cell perturbation responses" (2019)
3. Bunne et al., "Learning Single-Cell Perturbation Responses using Neural Optimal Transport" (2023)
4. **scPPDM**: "Single-cell Perturbation Prediction via Diffusion Models" — `dev/references/scPPDM.pdf`
   - Applies diffusion models to predict single-cell perturbation responses
   - Key target for implementation in this project

---

## Implementation Strategy

### Phase 1: Foundations (Current)
- VAE, cVAE on toy data
- Establish training infrastructure
- Evaluation metrics

### Phase 2: Diffusion
- Score matching basics
- DDPM implementation
- Conditional generation

### Phase 3: Advanced
- Flow matching
- EBMs
- JEPA for biology

### Phase 4: Applications
- Real single-cell data
- Perturbation prediction
- Benchmarking against published methods

---

## Next Steps

1. **Immediate**: Test diffusion training on RunPod with large preset
2. **This week**: Add conditional diffusion (classifier-free guidance)
3. **Next**: Flow matching implementation
4. **Ongoing**: Apply diffusion to gene expression data (scPPDM-style)
