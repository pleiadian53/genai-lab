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

**Status**: 🔲 Planned (see `docs/incubation/joint_latent_space_and_JEPA.md`)

### Key Concepts

- **Latent prediction**: Predict in embedding space, not pixel space
- **Self-supervised**: No reconstruction, no contrastive negatives
- **World models**: Learn dynamics without generation
- **Joint latent spaces**: Static and dynamic data share the same manifold

### Architecture

```
x → Encoder → z_x
              ↓
         Predictor → ẑ_y
              ↑
y → Encoder → z_y (target)
```

### Key Innovations from Goku/V-JEPA 2

- **Joint VAE**: Images (static) and videos (dynamic) share one latent space
- **Rectified Flow**: Direct velocity field instead of noise-based diffusion
- **Patch n' Pack**: Variable-length batching without padding
- **Full Attention**: No factorized spatial/temporal attention

### Biological Parallels

| Vision Domain | Biology Domain |
|---------------|----------------|
| Image (static) | Bulk RNA-seq, baseline expression |
| Video (dynamic) | Time-series, Perturb-seq, lineage tracing |
| Variable-length clips | Single-cell snapshots across conditions |

### Milestones

- [ ] Joint embedding architecture
- [ ] Predictor network with perturbation conditioning
- [ ] Variance-Invariance-Covariance (VICReg) regularization
- [ ] Apply to gene expression time series
- [ ] JEPA for Perturb-seq (predict perturbed state from baseline + perturbation)

### References

- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023)
- Meta AI, "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning" (2025)
- ByteDance & HKU, "Goku: Native Joint Image-Video Generation" (2024)

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

## Ideas Under Incubation

The `docs/incubation/` directory contains exploratory ideas and architectural proposals that may inform future development. These are not yet implemented but represent promising directions.

### Current Incubation Documents

| Document | Focus | Key Ideas |
|----------|-------|-----------|
| `joint_latent_space_and_JEPA.md` | Architecture | Joint latent spaces for static/dynamic data, JEPA for Perturb-seq |
| `generative-ai-for-gene-expression-prediction.md` | Application | Diffusion/VAE/Flow for gene expression with uncertainty |
| `generative-ai-for-perturbation-modeling.md` | Application | Generative approaches for scPerturb, beyond GEM-1 |

### Key Architectural Insights

1. **Joint Latent Spaces**: Static (bulk RNA-seq) and dynamic (time-series, Perturb-seq) data can share the same latent manifold, enabling mutual training
2. **JEPA over Reconstruction**: Predicting embeddings (not pixels/counts) is more robust for biology where reconstruction is rarely the goal
3. **Hybrid Predictive-Generative**: GEM-1-style predictive models + generative wrappers for uncertainty quantification
4. **Rectified Flow**: May be preferable to diffusion for biology where "noise semantics" are unclear

---

## Target Applications

### Application 1: Gene Expression Prediction

**Goal**: Predict gene expression from metadata with uncertainty quantification

**Current State**: GEM-1 (Synthesize Bio) demonstrates supervised prediction at scale

**Generative AI Value-Add**:
- Model full distribution $p(x \mid \text{metadata})$, not just $\mathbb{E}[x]$
- Uncertainty quantification for experimental planning
- Diverse synthetic data for augmentation

**Proposed Approach**: Hybrid model (predictive foundation + diffusion on residuals)

**See**: `docs/incubation/generative-ai-for-gene-expression-prediction.md`

### Application 2: Perturbation Prediction (scPerturb)

**Goal**: Predict cellular response to genetic/chemical perturbations

**Current State**: scGen, CPA, GEARS use VAE/GNN approaches

**Generative AI Value-Add**:
- Compositional generalization (unseen perturbation combinations)
- Cell-level heterogeneity modeling
- Counterfactual reasoning

**Proposed Approaches**:
1. Conditional diffusion on scPerturb
2. Causal VAE with perturbation operators
3. JEPA for Perturb-seq (predict perturbed latent from baseline + perturbation)

**See**: `docs/incubation/generative-ai-for-perturbation-modeling.md`, `docs/incubation/joint_latent_space_and_JEPA.md`

### Application 3: Synthetic Biological Datasets

**Goal**: Generate realistic synthetic datasets for drug/target discovery

**Use Cases**:
- Data augmentation for rare conditions
- Privacy-preserving data sharing
- Benchmarking computational methods
- Training downstream classifiers

**Generative AI Value-Add**:
- Diverse, realistic samples (not just mean predictions)
- Controllable generation (condition on disease, tissue, perturbation)
- Validation via biological consistency checks

**Proposed Approach**: Conditional diffusion with metadata conditioning

### Application 4: scPPDM (Single-cell Perturbation Prediction via Diffusion Models)

**Goal**: Implement and extend scPPDM methodology

**Status**: Deferred (see `dev/references/scPPDM.pdf`)

**Key Ideas**:
- Diffusion models for single-cell perturbation response prediction
- Conditional generation on perturbation identity
- Comparison with VAE-based methods (scGen, CPA)

---

## Efficient Paths to Applications

### Path A: Gene Expression Prediction (Fastest)

```
Current State → Conditional VAE → Conditional Diffusion → Hybrid Model
     ↓              (2 weeks)        (3 weeks)           (2 weeks)
  Stage 2          Add metadata     Add uncertainty      Combine with
  (cVAE)           conditioning     quantification       predictive model
```

**Dataset**: GTEx or harmonized bulk RNA-seq

### Path B: Perturbation Prediction (Most Impactful)

```
Current State → JEPA Prototype → Perturb-seq JEPA → Generative Wrapper
     ↓            (3 weeks)         (4 weeks)          (3 weeks)
  Stage 6        Basic JEPA on     Add perturbation    Add diffusion
  (Diffusion)    toy data          conditioning        for uncertainty
```

**Dataset**: Norman et al. 2019 (Perturb-seq, K562 cells)

### Path C: Synthetic Data Generation (Most General)

```
Current State → Conditional Diffusion → Multi-modal → Validation Pipeline
     ↓              (3 weeks)            (4 weeks)       (2 weeks)
  Stage 6         Gene expression      Add metadata,    Biological
  (Diffusion)     generation           perturbations    consistency checks
```

**Dataset**: scPerturb or CellxGene

---

## Cross-Cutting Themes

### Evaluation Metrics

| Model Type | Metrics |
|------------|---------|
| VAE/cVAE | ELBO, reconstruction MSE, KL, FID |
| Diffusion | FID, IS, likelihood bounds |
| EBM | Energy histograms, sample quality |
| JEPA | Downstream task performance |
| **Perturbation** | DEG recovery, pathway consistency, held-out perturbation accuracy |
| **Gene Expression** | Sample diversity, biological consistency, downstream task improvement |

### Computational Biology Applications

1. **Gene expression generation**: Conditional on cell type, disease
2. **Perturbation prediction**: What happens if we knock out gene X?
3. **Trajectory inference**: Developmental or disease progression
4. **Data augmentation**: Generate synthetic training data
5. **Representation learning**: Embeddings for downstream tasks
6. **Uncertainty quantification**: Confidence intervals for predictions
7. **Counterfactual reasoning**: "What if" scenarios for drug discovery

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

### Immediate (This Week)

1. Test diffusion training on RunPod with large preset
2. Add conditional diffusion (classifier-free guidance)

### Short-term (2-4 Weeks)

3. Implement conditional VAE for gene expression prediction
4. Begin JEPA prototype on toy biological data
5. Download and preprocess Norman et al. Perturb-seq dataset

### Medium-term (1-2 Months)

6. Flow matching implementation
7. JEPA for Perturb-seq with perturbation conditioning
8. Benchmark against scGen/CPA on perturbation prediction

### Long-term (3+ Months)

9. Hybrid predictive-generative model for gene expression
10. Synthetic biological dataset generation pipeline
11. Integration with real drug discovery workflows
