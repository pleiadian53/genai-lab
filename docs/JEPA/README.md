# JEPA Documentation

**Joint Embedding Predictive Architecture (JEPA)** — A self-supervised learning paradigm that learns by predicting in embedding space rather than reconstructing in data space.

This documentation series covers JEPA from first principles through computational biology applications, with a focus on perturbation prediction and trajectory modeling.

---

## Core Documentation Series

### 1. Overview
**[00_jepa_overview.md](00_jepa_overview.md)** — What is JEPA and why it matters
- Core concepts: predict embeddings, not pixels
- JEPA vs generative models vs contrastive learning
- Joint latent spaces (Goku insight)
- Why JEPA for biology
- When to use JEPA vs generative models

### 2. Foundations
**[01_jepa_foundations.md](01_jepa_foundations.md)** — Architecture and components
- Encoder architecture
- Predictor design
- VICReg regularization (variance, invariance, covariance)
- Masking strategies
- Complete PyTorch implementation

### 3. Training
**[02_jepa_training.md](02_jepa_training.md)** — Training strategies and best practices
- Training loop
- Loss computation
- Hyperparameters
- Optimization strategies
- Debugging and monitoring
- Advanced techniques

### 4. Applications
**[03_jepa_applications.md](03_jepa_applications.md)** — From vision to biology
- I-JEPA (image masking)
- V-JEPA (video prediction)
- Bio-JEPA (perturbation prediction)
- Multi-omics integration
- Trajectory inference

### 5. Perturb-seq Application
**[04_jepa_perturbseq.md](04_jepa_perturbseq.md)** — Detailed Perturb-seq implementation
- Dataset preparation
- Perturbation conditioning
- Model architecture
- Training pipeline
- Evaluation metrics
- Comparison with scGen/CPA

---

## Supplementary Documents

### Open Research
**[open_research_joint_latent.md](open_research_joint_latent.md)** — Joint latent spaces
- Goku model insights
- Static + dynamic data in one latent space
- Patch n' Pack for variable-length sequences
- Biology applications

---

## Quick Navigation

### For Different Audiences

**New to JEPA?**
1. Start with [Overview](00_jepa_overview.md)
2. Read [Foundations](01_jepa_foundations.md) for architecture
3. Try toy examples from [Training](02_jepa_training.md)

**Coming from Generative Models?**
1. Read [Overview](00_jepa_overview.md) comparison section
2. Understand why prediction ≠ generation
3. Learn when to combine JEPA + diffusion

**Interested in Biology Applications?**
1. Read [Overview](00_jepa_overview.md) biology section
2. Jump to [Applications](03_jepa_applications.md)
3. Deep dive into [Perturb-seq](04_jepa_perturbseq.md)

**Ready to Implement?**
1. Review [Foundations](01_jepa_foundations.md) architecture
2. Follow [Training](02_jepa_training.md) pipeline
3. Adapt [Perturb-seq](04_jepa_perturbseq.md) code

---

## Key Concepts

### What Makes JEPA Different

**Traditional Generative Models** (VAE, Diffusion):
```
Input → Encoder → Latent → Decoder → Reconstruction
Loss: ||x - x̂||² (pixel-level)
```

**JEPA**:
```
Context → Encoder → z_context
                     ↓
                 Predictor → ẑ_target
                     ↑
Target → Encoder → z_target
Loss: ||z_target - ẑ_target||² (embedding-level)
```

**Key advantages**:
- No decoder (10-100× faster)
- Semantic prediction (robust to noise)
- No contrastive negatives (simpler than SimCLR)
- Compositional reasoning (combine perturbations)

### Core Components

**1. Encoder**: Maps inputs to embeddings
- Shared across all inputs
- Vision Transformer (ViT) for images
- MLP/Transformer for gene expression

**2. Predictor**: Predicts target embedding from context
- Transformer-based
- Conditioned on context (time, perturbation, etc.)
- Learns relationships in embedding space

**3. VICReg Loss**: Prevents collapse
- **Variance**: Keep embeddings spread out
- **Invariance**: Predictions match targets
- **Covariance**: Decorrelate dimensions

### Joint Latent Spaces

**Insight from Goku (ByteDance, 2024)**:
> If two data types differ only by dimensionality or observation density, they want the same latent space.

**For biology**:
- Bulk RNA-seq (static) + Time-series (dynamic) → Same latent space
- Static data teaches spatial priors (cell types, pathways)
- Dynamic data teaches temporal dynamics
- Both inform the same representation

---

## JEPA Variants

### I-JEPA (Image)
**Task**: Predict masked image regions in embedding space

**Key innovation**: Masking in embedding space, not pixel space

**Papers**: Assran et al. (2023)

### V-JEPA (Video)
**Task**: Predict future video frames in embedding space

**Key innovation**: Temporal prediction without generation

**Papers**: Bardes et al. (2024), Meta AI (2025)

### Bio-JEPA (Proposed)
**Task**: Predict perturbed/future cell states in embedding space

**Key innovation**: Perturbation operators in latent space

**Applications**:
- Perturb-seq prediction
- Trajectory inference
- Multi-omics translation
- Drug response prediction

---

## Biology Applications

### 1. Perturbation Prediction (Perturb-seq)

**Problem**: Predict cellular response to genetic/chemical perturbations

**JEPA approach**:
```python
z_baseline = encoder(x_baseline)
z_pert = perturbation_encoder(perturbation_info)
z_pred = predictor(z_baseline, z_pert)
loss = ||z_pred - encoder(x_perturbed)||²
```

**Advantages**:
- No need to reconstruct all 20K genes
- Learn perturbation operators
- Compositional (combine perturbations)
- Efficient (no decoder)

**Datasets**: Norman et al. (2019), Replogle et al. (2022)

### 2. Trajectory Inference

**Problem**: Predict developmental or disease trajectories

**JEPA approach**:
```python
z_t = encoder(x_t)
z_t1_pred = predictor(z_t, time_embedding)
loss = ||z_t1_pred - encoder(x_t1)||²
```

**Applications**:
- Developmental biology
- Disease progression
- Drug response over time

### 3. Multi-omics Integration

**Problem**: Predict one modality from another

**JEPA approach**:
```python
z_rna = encoder_rna(x_rna)
z_protein_pred = predictor(z_rna)
loss = ||z_protein_pred - encoder_protein(x_protein)||²
```

**Applications**:
- RNA → Protein prediction
- ATAC → RNA prediction
- Cross-species translation

### 4. Drug Response Prediction

**Problem**: Predict cellular response to drugs

**JEPA approach**:
```python
z_baseline = encoder(x_baseline)
z_drug = drug_encoder(drug_features)
z_response = predictor(z_baseline, z_drug)
loss = ||z_response - encoder(x_treated)||²
```

**Applications**:
- Drug screening
- Combination therapy
- Patient stratification

---

## Comparison with Other Methods

### JEPA vs VAE

| Aspect | VAE | JEPA |
|--------|-----|------|
| **Objective** | Reconstruct input | Predict target embedding |
| **Loss** | Pixel-level + KL | Embedding-level + VICReg |
| **Decoder** | Required | Not needed |
| **Speed** | Slow | Fast (10-100×) |
| **Generation** | Yes | No (need wrapper) |
| **Robustness** | Moderate | High |

### JEPA vs Diffusion

| Aspect | Diffusion | JEPA |
|--------|-----------|------|
| **Objective** | Denoise/predict velocity | Predict embedding |
| **Loss** | Pixel-level | Embedding-level |
| **Sampling** | ODE/SDE (slow) | Direct (fast) |
| **Generation** | Yes | No (need wrapper) |
| **Uncertainty** | Via sampling | Need wrapper |
| **Efficiency** | Moderate | High |

### JEPA vs Contrastive (SimCLR)

| Aspect | SimCLR | JEPA |
|--------|--------|------|
| **Objective** | Maximize agreement | Predict embedding |
| **Negatives** | Required | Not needed |
| **Loss** | Contrastive | MSE + VICReg |
| **Complexity** | High (negative sampling) | Low |
| **Prediction** | No | Yes |

### JEPA vs scGen/CPA (Perturbation Models)

| Aspect | scGen/CPA | JEPA |
|--------|-----------|------|
| **Architecture** | VAE + arithmetic | Encoder + Predictor |
| **Perturbation** | Latent arithmetic | Learned operators |
| **Reconstruction** | Required | Not needed |
| **Compositional** | Limited | Natural |
| **Efficiency** | Moderate | High |

---

## When to Use JEPA

### ✅ Use JEPA When:

**Prediction is the goal** (not generation)
- Perturbation prediction
- Trajectory inference
- Multi-omics translation

**Efficiency matters**
- Large-scale datasets
- Limited compute
- Need fast training

**Robustness is critical**
- Noisy data
- Batch effects
- Technical variation

**Compositional reasoning needed**
- Combine perturbations
- Transfer across contexts
- Causal modeling

### ❌ Use Generative Models When:

**Need actual samples**
- Data augmentation
- Synthetic data generation
- Uncertainty quantification

**Reconstruction quality matters**
- Image generation
- High-fidelity synthesis

**Distribution modeling is the goal**
- Density estimation
- Anomaly detection

### 🔄 Best: Hybrid JEPA + Generative

**Combine both**:
1. JEPA learns dynamics efficiently
2. Generative model handles sampling
3. Get prediction + generation + uncertainty

**Example**: JEPA + Diffusion
```python
# JEPA predicts perturbed embedding
z_pred = jepa_predictor(z_baseline, perturbation)

# Diffusion generates samples from embedding
x_samples = diffusion_decoder(z_pred, num_samples=100)

# Get both prediction and uncertainty
```

---

## Implementation Roadmap

### Phase 1: Basic JEPA
- [ ] Encoder architecture (ViT or MLP)
- [ ] Predictor network (Transformer)
- [ ] VICReg loss implementation
- [ ] Training loop
- [ ] Toy examples (MNIST, synthetic)

### Phase 2: Bio-JEPA
- [ ] Gene expression encoder
- [ ] Perturbation conditioning
- [ ] Perturb-seq dataset loader
- [ ] Training on Norman et al. data
- [ ] Evaluation metrics

### Phase 3: Joint Latent Spaces
- [ ] Joint encoder for bulk + single-cell
- [ ] Static + dynamic data training
- [ ] Multi-omics integration
- [ ] Cross-dataset transfer

### Phase 4: JEPA + Generative
- [ ] Diffusion decoder
- [ ] Uncertainty quantification
- [ ] Sample generation
- [ ] Full predictive-generative system

---

## Learning Path

### Beginner Path
1. **Understand the concept** — [Overview](00_jepa_overview.md)
2. **Learn the architecture** — [Foundations](01_jepa_foundations.md)
3. **Train on toy data** — [Training](02_jepa_training.md)
4. **Explore applications** — [Applications](03_jepa_applications.md)

### Intermediate Path
1. **Review architecture** — [Foundations](01_jepa_foundations.md)
2. **Implement training** — [Training](02_jepa_training.md)
3. **Apply to Perturb-seq** — [Perturb-seq](04_jepa_perturbseq.md)
4. **Compare with baselines** — Evaluate against scGen/CPA

### Advanced Path
1. **Joint latent spaces** — [Open Research](open_research_joint_latent.md)
2. **Hybrid JEPA + Diffusion** — Combine prediction and generation
3. **Multi-omics integration** — Cross-modality prediction
4. **Novel applications** — Extend to new biology problems

---

## Related Documentation

### Within This Project

**Generative Models**:
- [DDPM](../DDPM/) — Denoising diffusion
- [SDE](../SDE/) — Stochastic differential equations
- [Flow Matching](../flow_matching/) — Rectified flow
- [DiT](../DiT/) — Diffusion transformers
- [VAE](../VAE/) — Variational autoencoders

**Architecture Choices**:
- [Gene Expression Architectures](../DDPM/02a_diffusion_arch_gene_expression.md) — Tokenization for biology

**Incubation**:
- [Joint Latent Spaces](../incubation/joint_latent_space_and_JEPA.md) — Goku insights

### External Resources

**JEPA Papers**:
- LeCun (2022): "A Path Towards Autonomous Machine Intelligence"
- Assran et al. (2023): "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA)
- Bardes et al. (2024): "V-JEPA: Latent Video Prediction"
- Meta AI (2025): "V-JEPA 2: Understanding, Prediction, and Planning"

**Joint Latent Spaces**:
- ByteDance & HKU (2024): "Goku: Native Joint Image-Video Generation"

**VICReg**:
- Bardes et al. (2022): "VICReg: Variance-Invariance-Covariance Regularization"

**Biology Applications**:
- Norman et al. (2019): "Exploring genetic interaction manifolds" (Perturb-seq)
- Lotfollahi et al. (2019): "scGen predicts single-cell perturbation responses"
- Roohani et al. (2023): "Predicting transcriptional outcomes of novel multigene perturbations" (GEARS)

---

## Key Takeaways

### Conceptual
1. **Predict embeddings, not pixels** — More efficient, more robust
2. **No reconstruction needed** — Focus on semantic content
3. **No contrastive negatives** — Simpler than SimCLR/MoCo
4. **World models without generation** — Learn dynamics efficiently
5. **Joint latent spaces** — Static and dynamic data train each other

### Practical
1. **JEPA is not generative** — Predicts embeddings, not samples
2. **VICReg prevents collapse** — Variance + covariance regularization
3. **Powerful predictor needed** — Transformer-based works well
4. **Combine with generative** — For sampling and uncertainty
5. **Perfect for biology** — Perturbations, trajectories, multi-omics

### For Computational Biology
1. **Perturb-seq is ideal** — Predict perturbed states efficiently
2. **Efficiency matters** — 20K genes, millions of cells
3. **Robustness critical** — Technical noise, batch effects
4. **Compositional reasoning** — Combine perturbations naturally
5. **Hybrid approach best** — JEPA + diffusion for full system

---

## Getting Started

**Quick start**:
```bash
# Read overview
cat docs/JEPA/00_jepa_overview.md

# Understand architecture
cat docs/JEPA/01_jepa_foundations.md

# See training examples
cat docs/JEPA/02_jepa_training.md
```

**For biology applications**:
```bash
# Jump to applications
cat docs/JEPA/03_jepa_applications.md

# Deep dive into Perturb-seq
cat docs/JEPA/04_jepa_perturbseq.md
```

**For implementation**:
```bash
# Check source code (when available)
ls src/genailab/jepa/

# Run notebooks (when available)
ls notebooks/jepa/
```

---

## Status

**Documentation**: 🚧 In Progress
- [x] Overview
- [ ] Foundations
- [ ] Training
- [ ] Applications
- [ ] Perturb-seq
- [ ] Open Research

**Implementation**: 🔲 Planned
- [ ] Core JEPA modules
- [ ] Training infrastructure
- [ ] Perturb-seq application
- [ ] Evaluation metrics

**Notebooks**: 🔲 Planned
- [ ] Toy examples
- [ ] Gene expression JEPA
- [ ] Perturb-seq prediction
- [ ] Comparison with baselines
