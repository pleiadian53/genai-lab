# Generative AI Learning Roadmap

A systematic progression from classical latent-variable models to modern generative architectures, with implementation milestones for computational biology applications.

---

## Overview

**Learning Path (Theory):**
```
VAE → β-VAE → Score Matching → DDPM → Flow Matching → EBMs → JEPA → World Models
 │                                │
 └── cVAE (conditional)           └── Classifier-Free Guidance
```

**Current Implementation Focus:**
```
Perturbation Prediction (Perturb-seq)
    ├── Phase 1: VAE baseline (CVAE_NB on Norman et al. dataset)
    ├── Phase 2: JEPA predictor (latent space prediction)
    └── Phase 3: Diffusion wrapper (uncertainty quantification)
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

## Stage 7: Flow Matching & Rectified Flow

**Status**: 📝 Documented

### Key Concepts

- **Flow matching**: Learn velocity field via regression, not score matching
- **Rectified flow**: Linear interpolation paths (simplest flow matching)
- **Deterministic sampling**: ODE-based generation (no stochastic noise)
- **Simulation-free training**: No ODE solver during training

### Core Equations

**Path (rectified flow):**

$$
x_t = (1 - t) \cdot x_0 + t \cdot x_1
$$

**Velocity:**

$$
\frac{dx_t}{dt} = x_1 - x_0
$$

**Loss:**

$$
\mathcal{L}_{\text{RF}} = \mathbb{E}_{x_0, x_1, t} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

### Comparison: Score Matching vs Flow Matching

| Aspect | Score Matching | Flow Matching |
|--------|---------------|---------------|
| What's learned | Score: $\nabla_x \log p_t(x)$ | Velocity: $v_\theta(x, t)$ |
| Forward process | Stochastic (add noise) | Deterministic (interpolate) |
| Reverse process | Stochastic SDE | Deterministic ODE |
| Sampling steps | 100-1000 | 10-50 |

### Documentation

- [Rectified Flow Tutorial](flow_matching/rectifying_flow.md) — From first principles

### Milestones

- [x] Rectified flow theory documentation
- [ ] Implement flow matching loss
- [ ] Conditional flow matching
- [ ] Compare with DDPM on gene expression

### References

- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022)
- Lipman et al., "Flow Matching for Generative Modeling" (2023)

---

## Stage 7b: Diffusion Transformers (DiT)

**Status**: ✅ Documented

### Key Concepts

- **Transformer backbone**: Replace U-Net with Transformer for diffusion/flow models
- **Patch tokenization**: Convert images/data to token sequences
- **Adaptive LayerNorm (AdaLN)**: Time and condition modulation via FiLM
- **Architecture-objective separation**: DiT works with score matching, noise prediction, or rectified flow

### Architecture

```text
Input → Patch Embed → [Transformer Blocks with AdaLN] → Output Projection
                              ↑
                    Time Embed + Condition Embed
```

### Why DiT Over U-Net

| Aspect | U-Net | DiT |
|--------|-------|-----|
| Context | Local → global via downsampling | Global via attention |
| Conditioning | Architectural changes needed | Add tokens or modulation |
| Input shapes | Fixed grid | Variable (with masking) |
| Scalability | Limited | Scales with compute |

### Documentation

- [DiT Series](DiT/README.md) — Complete documentation series
  - [00_dit_overview.md](DiT/00_dit_overview.md) — Introduction
  - [01_dit_foundations.md](DiT/01_dit_foundations.md) — Architecture details
  - [02_dit_training.md](DiT/02_dit_training.md) — Training with rectified flow
  - [03_dit_sampling.md](DiT/03_dit_sampling.md) — Sampling strategies
  - [open_research_tokenization.md](DiT/open_research_tokenization.md) — Tokenization for biology

### Milestones

- [x] DiT theory documentation (complete series)
- [x] AdaLN/FiLM conditioning explanation
- [x] Tokenization strategies for gene expression
- [ ] Minimal DiT implementation
- [ ] DiT + rectified flow for gene expression

### References

- Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)

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

## Stage 8b: Latent Diffusion Models

**Status**: ✅ Documented

### Key Concepts

- **Two-stage training**: VAE for compression + Diffusion in latent space
- **Count-aware decoders**: NB/ZINB decoders for gene expression
- **Efficiency**: Train diffusion on compressed representations
- **Biological constraints**: Preserve count structure and sparsity

### Documentation

- [Latent Diffusion Series](latent_diffusion/README.md) — Complete documentation
  - [00_latent_diffusion_overview.md](latent_diffusion/00_latent_diffusion_overview.md)
  - [01_latent_diffusion_foundations.md](latent_diffusion/01_latent_diffusion_foundations.md)
  - [02_latent_diffusion_training.md](latent_diffusion/02_latent_diffusion_training.md)
  - [03_latent_diffusion_applications.md](latent_diffusion/03_latent_diffusion_applications.md)
  - [04_latent_diffusion_combio.md](latent_diffusion/04_latent_diffusion_combio.md)

### Milestones

- [x] Latent diffusion theory documentation
- [x] VAE with NB/ZINB decoders
- [x] DiT backbone for latent diffusion
- [x] Conditioning mechanisms (FiLM, cross-attention)
- [ ] Implementation for gene expression
- [ ] End-to-end training pipeline

---

## Stage 9: Joint Embedding Predictive Architecture (JEPA)

**Status**: ✅ Documented

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

### Documentation

- [JEPA Series](JEPA/README.md) — Complete documentation
  - [00_jepa_overview.md](JEPA/00_jepa_overview.md)
  - [01_jepa_foundations.md](JEPA/01_jepa_foundations.md)
  - [02_jepa_training.md](JEPA/02_jepa_training.md)
  - [03_jepa_applications.md](JEPA/03_jepa_applications.md)
  - [04_jepa_perturbseq.md](JEPA/04_jepa_perturbseq.md) — Complete Perturb-seq implementation
  - [open_research_joint_latent.md](JEPA/open_research_joint_latent.md)

### Milestones

- [x] JEPA theory documentation (complete series)
- [x] VICReg regularization explanation
- [x] Perturb-seq architecture and training
- [ ] Joint embedding architecture implementation
- [ ] Predictor network with perturbation conditioning
- [ ] Apply to gene expression time series

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

## Stage 11: Foundation Model Adaptation

**Status**: ✅ Framework Implemented

### Key Concepts

- **Resource-aware configurations**: Small/medium/large model presets
- **Parameter-efficient fine-tuning**: LoRA, adapters, freezing strategies
- **Hardware auto-detection**: M1 Mac, RunPod, Cloud GPUs
- **Modular design**: Composable components for adaptation

### Implementation

- `src/genailab/foundation/` — Complete package
  - `configs/model_configs.py` — Small/medium/large presets
  - `configs/resource_profiles.py` — Hardware detection
  - `tuning/lora.py` — LoRA implementation

### Documentation

- [Foundation Models Series](foundation_models/) — Complete guides
  - [leveraging_foundation_models_v2.md](foundation_models/leveraging_foundation_models_v2.md)
  - [data_shape_v2.md](foundation_models/data_shape_v2.md)
  - [IMPLEMENTATION_GUIDE.md](foundation_models/IMPLEMENTATION_GUIDE.md)
- [Package README](../src/genailab/foundation/README.md)
- [Tutorial Roadmap](../notebooks/foundation_models/README.md)

### Milestones

- [x] Resource-aware model configurations
- [x] Auto-detection of hardware resources
- [x] LoRA implementation with save/load utilities
- [x] Comprehensive documentation (foundation models, DiT, JEPA, latent diffusion)
- [ ] Adapters and freezing strategies
- [ ] Conditioning modules (FiLM, cross-attention, CFG)
- [ ] Tutorial notebooks
- [ ] End-to-end recipes for gene expression

### References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019)
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)

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

## Implementation Status by Stage

### ✅ Completed (Validated & Documented)

**Stages 1-3: Foundation**
- VAE family (CVAE, CVAE_NB, CVAE_ZINB) with comprehensive documentation
- Data pipeline (preprocessing, path management, environment setup)
- Score matching theory (Fisher/Stein scores, energy functions, denoising score matching)

**Stage 4: Diffusion (Core Infrastructure)**
- Forward/reverse diffusion (VP-SDE, VE-SDE)
- Score networks (MLP, TabularScoreNetwork, UNet2D, UNet3D)
- Training/sampling infrastructure
- Proof-of-concept validated on medical imaging

### 🎯 Active Development

**Flagship Application: Perturbation Prediction (Perturb-seq)**

Current milestone: End-to-end pipeline for Norman et al. 2019 dataset

**Week 1-2: Data + VAE Baseline**
- [ ] Download and preprocess Norman et al. Perturb-seq dataset
- [ ] Establish data loaders and quality control
- [ ] Train CVAE_NB baseline with perturbation conditioning
- [ ] Establish evaluation metrics (DEG recovery, perturbation accuracy)

**Week 3-4: JEPA Implementation**
- [ ] Implement JEPA encoder-predictor architecture
- [ ] VICReg regularization for collapse prevention
- [ ] Train with perturbation conditioning
- [ ] Compare latent space quality vs. CVAE baseline

**Week 5-6: Diffusion Wrapper + Benchmarking**
- [ ] Add diffusion in latent space for uncertainty quantification
- [ ] Implement sampling for diverse cellular responses
- [ ] Benchmark against scGen, CPA, scPPDM
- [ ] Document results and create example notebook

**Success Criteria:**
- ✅ End-to-end notebook: `examples/perturbation/01_perturbseq_jepa_diffusion.ipynb`
- ✅ Benchmark table comparing with published methods
- ✅ Validation: DEG recovery, held-out perturbation accuracy, compositional generalization
- ✅ Documentation: Application guide in `docs/applications/perturbation_prediction.md`

### 📝 Research Prototypes (Theory Complete, Awaiting Implementation)

**Advanced Architectures:**
- DiT (Diffusion Transformers) - 5-part documentation series complete
- Latent Diffusion with NB/ZINB decoders - 5-part series complete
- Flow Matching & Rectified Flow - theory documented
- Foundation model adaptation framework - LoRA implemented, adapters/conditioning pending

**Stage 5: Foundation Model Adaptation** - Partially implemented:
- [x] Resource-aware configs, hardware auto-detection, LoRA
- [ ] Adapters and freezing strategies
- [ ] Conditioning modules (FiLM, cross-attention, CFG)
- [ ] Tutorial notebooks

### 🔮 Planned (After Current Focus)

**Next Applications** (one at a time after Perturb-seq):
1. Gene expression prediction (GTEx, harmonized bulk RNA-seq)
2. Synthetic biological dataset generation with validation pipeline

**Architectural Extensions:**
- Flow matching implementation (after JEPA + diffusion validated)
- Classifier-free guidance for conditional generation
- World models for trajectory prediction

**Integration with Causal Methods:**
- Counterfactual generation pipeline
- Causal regularization via invariance
- Integration with [causal-bio-lab](../../causal-bio-lab/) for causal validation

---

## Target Applications

### 🎯 Application 1: Perturbation Prediction (scPerturb) — ACTIVE

**Goal**: Predict cellular response to genetic/chemical perturbations at single-cell resolution

**Scientific Impact**: Central problem in computational biology; enables in silico perturbation screening

**Why This Application First**:
- Clear benchmarks (scGen, CPA, GEARS, scPPDM)
- Leverages existing strengths (VAE with NB/ZINB decoders, JEPA theory, latent diffusion)
- Natural progression: VAE → JEPA → Diffusion wrapper

**Generative AI Value-Add**:
- Compositional generalization (unseen perturbation combinations)
- Cell-level heterogeneity modeling (not just population means)
- Uncertainty quantification for experimental planning
- Counterfactual reasoning ("what if we perturb X instead?")

**Implementation Approach**:
1. **Phase 1**: CVAE_NB baseline on Norman et al. 2019 (K562 Perturb-seq)
2. **Phase 2**: JEPA predictor for latent space prediction
3. **Phase 3**: Diffusion in latent space for uncertainty quantification

**Evaluation Metrics**:
- DEG recovery (differential expression gene prediction)
- Held-out perturbation accuracy
- Compositional generalization (double knockouts from single)
- Pathway consistency (biological validation)

**See**:
- [docs/applications/perturbation_prediction.md](applications/perturbation_prediction.md) — Complete implementation guide
- [docs/JEPA/04_jepa_perturbseq.md](JEPA/04_jepa_perturbseq.md) — Architecture details

### 📋 Application 2: Gene Expression Prediction — NEXT

**Goal**: Predict gene expression from metadata with uncertainty quantification

**Current State**: GEM-1 (Synthesize Bio) demonstrates supervised prediction at scale

**Generative AI Value-Add**:
- Model full distribution $p(x \mid \text{metadata})$, not just $\mathbb{E}[x]$
- Uncertainty quantification for experimental planning
- Diverse synthetic data for augmentation

**Proposed Approach**: Hybrid model (GEM-1-style predictor + diffusion on residuals)

**When**: After Perturb-seq application is benchmarked and documented

**See**: [docs/incubation/generative-ai-for-gene-expression-prediction.md](incubation/generative-ai-for-gene-expression-prediction.md)

### 📋 Application 3: Synthetic Biological Datasets — NEXT

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

**When**: After at least one prediction-focused application (Perturb-seq or Gene Expression) is complete

---

## Implementation Strategy

### Principle: Depth Over Breadth

The project has completed extensive theory exploration. The current phase focuses on **consolidating one complete application** as a flagship demonstration before expanding.

**Chosen Path: Perturbation Prediction (Perturb-seq)**

Rationale:
- **Scientific impact**: Central problem in computational biology
- **Clear benchmarks**: Published methods (scGen, CPA, GEARS, scPPDM) for comparison
- **Leverages strengths**: VAE infrastructure, JEPA theory, latent diffusion documentation
- **Natural progression**: Demonstrates VAE → JEPA → Diffusion wrapper integration

**Success = One Complete Vertical**

A complete application means:
1. ✅ End-to-end implementation (data → model → evaluation)
2. ✅ Benchmarked against published methods
3. ✅ Documented with reproducible examples
4. ✅ Biologically validated (DEG recovery, pathway consistency)

**After Perturbation Prediction:**
- Decision point: Extend current application OR start next application
- Options: Gene expression prediction, Synthetic data generation, or deeper Perturb-seq extensions

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

### Week 1-2: Data + VAE Baseline (Immediate Focus)

1. **Dataset acquisition**:
   - Download Norman et al. 2019 Perturb-seq dataset (K562 cells, CRISPR knockouts)
   - Implement data loaders with quality control
   - Document data statistics and preprocessing choices

2. **VAE baseline**:
   - Train CVAE_NB with perturbation conditioning
   - Implement evaluation metrics (DEG recovery, perturbation classification accuracy)
   - Establish baseline performance for comparison

3. **Infrastructure**:
   - Create `examples/perturbation/` directory structure
   - Set up experiment tracking (wandb or similar)
   - Document training procedures

### Week 3-4: JEPA Implementation

4. **Core JEPA architecture**:
   - Implement context encoder + target encoder (EMA)
   - Perturbation encoder (embedding lookup + set encoder for multi-perturbations)
   - Predictor network with perturbation conditioning

5. **Collapse prevention**:
   - VICReg regularization (variance/covariance losses)
   - Multi-view augmentation strategies for gene expression
   - Diagnostic tools for monitoring latent space quality

6. **Evaluation**:
   - Compare JEPA vs. CVAE latent spaces
   - Held-out perturbation prediction accuracy
   - Latent space visualization (UMAP/t-SNE)

### Week 5-6: Diffusion Wrapper + Benchmarking

7. **Latent diffusion**:
   - Diffusion model in JEPA latent space
   - Sampling for diverse cellular responses
   - Uncertainty quantification metrics

8. **Comprehensive benchmarking**:
   - Compare against scGen, CPA, scPPDM (if code available)
   - Evaluation: DEG recovery, compositional generalization, pathway consistency
   - Create benchmark table for documentation

9. **Documentation & examples**:
   - Complete `examples/perturbation/01_perturbseq_jepa_diffusion.ipynb`
   - Update `docs/applications/perturbation_prediction.md` with results
   - Create reproducible training scripts

### After Current Milestone: Decision Point

**Option A: Extend Perturb-seq Application**
- Multi-dataset validation (other Perturb-seq datasets)
- Compositional perturbations (double/triple knockouts)
- Transfer learning across cell types
- Integration with causal-bio-lab for counterfactual validation

**Option B: Start Next Application**
- Gene expression prediction (GTEx, hybrid predictive-generative)
- Synthetic data generation pipeline

**Decision criteria**: Based on impact assessment and scientific priorities after completing flagship application
