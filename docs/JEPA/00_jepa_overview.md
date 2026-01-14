# JEPA Overview: Joint Embedding Predictive Architecture

**JEPA (Joint Embedding Predictive Architecture)** is a self-supervised learning paradigm that learns by **predicting in embedding space** rather than reconstructing in pixel/data space.

**Key insight**: Instead of generating outputs (like VAE/diffusion), JEPA predicts the latent representation of targets, enabling efficient learning of world models without expensive reconstruction.

---

## What is JEPA?

### The Core Idea

**Traditional generative models** (VAE, diffusion):
```
Input → Encoder → Latent → Decoder → Reconstruction
                                    ↓
                              Loss: ||x - x̂||²
```

**JEPA**:
```
Input x → Encoder → z_x
                     ↓
                 Predictor → ẑ_y
                     ↑
Target y → Encoder → z_y
                     ↓
              Loss: ||z_y - ẑ_y||²
```

**Key difference**: Predict embeddings, not pixels/counts.

### Why This Matters

**1. Computational efficiency**
- No expensive decoder
- Prediction in low-dimensional latent space
- Faster training, less memory

**2. Better representations**
- Focus on semantic content, not pixel details
- Invariant to nuisance factors
- More robust to noise

**3. World modeling**
- Learn dynamics without generation
- Predict future states efficiently
- Enable planning and reasoning

**4. No contrastive negatives**
- Unlike SimCLR, MoCo, CLIP
- No need to sample negative pairs
- Simpler training objective

---

## JEPA vs Other Approaches

### Comparison Table

| Method | What's Predicted | Loss Type | Negatives? | Decoder? |
|--------|------------------|-----------|------------|----------|
| **VAE** | Reconstruction | Pixel-level | No | Yes (expensive) |
| **Diffusion** | Noise/velocity | Pixel-level | No | Yes (expensive) |
| **Contrastive (SimCLR)** | Similarity | Embedding | Yes (required) | No |
| **JEPA** | Embedding | Embedding | No | No |

### Visual Comparison

**Generative (VAE/Diffusion)**:
```
Learn: p(x|z) — "How to generate x from z"
Goal: Reconstruct/generate realistic samples
Cost: High (decoder, pixel-level loss)
```

**Contrastive (SimCLR/MoCo)**:
```
Learn: Similarity in embedding space
Goal: Pull positives together, push negatives apart
Cost: Moderate (need negative sampling)
```

**JEPA**:
```
Learn: Predict z_y from z_x
Goal: Model relationships in embedding space
Cost: Low (no decoder, no negatives)
```

---

## Key Components

### 1. Encoder

Maps inputs to embeddings:
$$
z = f_\theta(x)
$$

**Shared across all inputs** (images, videos, gene expression, etc.)

### 2. Predictor

Predicts target embedding from context:
$$
\hat{z}_y = g_\phi(z_x, \text{context})
$$

**Context** can be:
- Time (predict future from past)
- Perturbation (predict perturbed state from baseline)
- Masked regions (predict masked from visible)

### 3. VICReg Regularization

Prevents collapse via three terms:

**Variance**: Keep embeddings spread out
$$
\mathcal{L}_{\text{var}} = \sum_d \max(0, \gamma - \sqrt{\text{Var}(z_d) + \epsilon})
$$

**Invariance**: Predictions match targets
$$
\mathcal{L}_{\text{inv}} = \| z_y - \hat{z}_y \|^2
$$

**Covariance**: Decorrelate dimensions
$$
\mathcal{L}_{\text{cov}} = \sum_{i \neq j} \text{Cov}(z_i, z_j)^2
$$

**Total loss**:
$$
\mathcal{L} = \lambda_{\text{inv}} \mathcal{L}_{\text{inv}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}} + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}}
$$

---

## Why JEPA for Biology?

### Problem with Generative Models in Biology

**Reconstruction is often not the goal**:
- We don't need to generate realistic gene expression
- We care about **predictions** (perturbations, trajectories)
- Pixel-level accuracy is overkill

**Example**: Perturb-seq
- Given: Baseline cell state + perturbation
- Want: Predicted perturbed state
- Don't need: Perfect reconstruction of all 20K genes

### JEPA Advantages for Biology

**1. Efficient perturbation modeling**
```
Baseline expression → Encoder → z_baseline
                                    ↓
Perturbation info → Predictor → ẑ_perturbed
                                    ↑
Actual perturbed → Encoder → z_perturbed
                                    ↓
                            Loss: ||z - ẑ||²
```

**2. Natural for time-series**
```
x_t → Encoder → z_t
                 ↓
             Predictor → ẑ_{t+1}
                 ↑
x_{t+1} → Encoder → z_{t+1}
```

**3. Handles heterogeneity**
- Cell-level predictions (not population average)
- Uncertainty in embedding space
- Robust to technical noise

**4. Compositional generalization**
- Learn perturbation operators
- Combine multiple perturbations
- Transfer across cell types

---

## Joint Latent Spaces: The Goku Insight

### The Problem with Separate Models

**Traditional approach**:
- Bulk RNA-seq model (static)
- Time-series model (dynamic)
- Perturb-seq model (perturbations)
- **Each has its own latent space**

**Issues**:
- No knowledge transfer
- Can't combine modalities
- Redundant learning

### Joint Latent Space Solution

**Key insight from Goku (ByteDance, 2024)**:
> If two data types differ only by dimensionality or observation density, they probably want the same latent space.

**For biology**:
- **Bulk RNA-seq** = "static image" (baseline state)
- **Time-series** = "video" (temporal dynamics)
- **Perturb-seq** = "video with interventions"
- **Single-cell snapshots** = "variable-length clips"

**All map to the same latent manifold**:
```
Bulk RNA-seq ──┐
               ├──> Shared Encoder ──> Joint Latent Space
Time-series ───┤
               │
Perturb-seq ───┘
```

**Benefits**:
- Static data teaches spatial priors (cell types, pathways)
- Dynamic data teaches temporal dynamics
- Perturbation data teaches causal relationships
- **All inform the same representation**

---

## JEPA Variants

### I-JEPA (Image JEPA)

**Task**: Predict masked image regions in embedding space

```
Visible patches → Encoder → z_visible
                              ↓
                          Predictor → ẑ_masked
                              ↑
Masked patches → Encoder → z_masked
```

**Key innovation**: Masking in embedding space, not pixel space

### V-JEPA (Video JEPA)

**Task**: Predict future video frames in embedding space

```
Past frames → Encoder → z_past
                         ↓
                     Predictor → ẑ_future
                         ↑
Future frames → Encoder → z_future
```

**V-JEPA 2 (Meta, 2025)**: Adds planning capabilities

### A-JEPA (Audio JEPA)

**Task**: Predict masked audio segments in embedding space

### Bio-JEPA (Proposed)

**Task**: Predict perturbed/future cell states in embedding space

```
Baseline + perturbation → Predictor → ẑ_perturbed
                                        ↑
Actual perturbed state → Encoder → z_perturbed
```

---

## JEPA + Generative Models

### Hybrid Architecture

**JEPA alone** doesn't generate samples — it predicts embeddings.

**To generate**, combine with generative model:

```
JEPA: x → z → Predictor → ẑ
                           ↓
Generative: ẑ → Decoder/Diffusion → x̂
```

**Example**: JEPA + Diffusion
1. JEPA predicts perturbed embedding
2. Diffusion generates samples from that embedding
3. Get both prediction AND uncertainty quantification

**Benefits**:
- JEPA learns dynamics efficiently
- Generative model handles distribution
- Best of both worlds

---

## Applications in Computational Biology

### 1. Perturbation Prediction (Perturb-seq)

**Task**: Predict cellular response to genetic/chemical perturbations

**JEPA approach**:
```python
# Baseline cell
z_baseline = encoder(x_baseline)

# Perturbation embedding
z_pert = perturbation_encoder(perturbation_info)

# Predict perturbed state
z_pred = predictor(z_baseline, z_pert)

# Compare to actual
z_actual = encoder(x_perturbed)
loss = ||z_pred - z_actual||²
```

**Advantages**:
- No need to reconstruct all 20K genes
- Learn perturbation operators
- Compositional (combine perturbations)

### 2. Trajectory Inference

**Task**: Predict developmental or disease trajectories

**JEPA approach**:
```python
# Current state
z_t = encoder(x_t)

# Time embedding
t_emb = time_encoder(t)

# Predict future
z_t1 = predictor(z_t, t_emb)

# Compare to actual
z_actual = encoder(x_t1)
loss = ||z_t1 - z_actual||²
```

### 3. Multi-omics Integration

**Task**: Predict one modality from another

**JEPA approach**:
```python
# RNA-seq
z_rna = encoder_rna(x_rna)

# Predict protein
z_protein_pred = predictor(z_rna)

# Compare to actual
z_protein = encoder_protein(x_protein)
loss = ||z_protein_pred - z_protein||²
```

### 4. Drug Response Prediction

**Task**: Predict cellular response to drugs

**JEPA approach**:
```python
# Baseline + drug
z_baseline = encoder(x_baseline)
z_drug = drug_encoder(drug_features)

# Predict response
z_response = predictor(z_baseline, z_drug)

# Compare to actual
z_actual = encoder(x_treated)
loss = ||z_response - z_actual||²
```

---

## Key Advantages for Biology

### 1. Efficiency

**Generative models**:
- Train decoder on 20K genes
- Pixel-level reconstruction loss
- Slow, memory-intensive

**JEPA**:
- No decoder
- Embedding-level loss (e.g., 256-dim)
- Fast, memory-efficient

**Speedup**: 10-100× faster training

### 2. Robustness

**Generative models**:
- Sensitive to technical noise
- Must model all variation
- Overfits to batch effects

**JEPA**:
- Focuses on semantic content
- Invariant to nuisance factors
- More robust representations

### 3. Interpretability

**Generative models**:
- Black box decoder
- Hard to interpret latents

**JEPA**:
- Direct prediction in embedding space
- Can probe what's being predicted
- Easier to analyze learned representations

### 4. Compositional Generalization

**Generative models**:
- Learn p(x|condition)
- Hard to combine conditions

**JEPA**:
- Learn perturbation operators
- Naturally compositional
- Combine multiple perturbations

---

## Limitations and Challenges

### 1. No Direct Generation

**JEPA predicts embeddings, not samples**.

**Solution**: Combine with generative model (VAE, diffusion) for sampling.

### 2. Embedding Collapse

**Without regularization, embeddings can collapse to constant**.

**Solution**: VICReg regularization (variance + covariance terms).

### 3. Predictor Capacity

**Predictor must be powerful enough to model relationships**.

**Solution**: Use transformer-based predictors with sufficient capacity.

### 4. Evaluation

**How do you evaluate embedding predictions?**

**Solutions**:
- Downstream task performance
- Embedding similarity metrics
- Probe classifiers
- Generate samples and evaluate

---

## When to Use JEPA

### Use JEPA When:

✅ **Prediction is the goal** (not generation)
- Perturbation prediction
- Trajectory inference
- Multi-omics translation

✅ **Efficiency matters**
- Large-scale datasets
- Limited compute
- Need fast training

✅ **Robustness is critical**
- Noisy data
- Batch effects
- Technical variation

✅ **Compositional reasoning needed**
- Combine perturbations
- Transfer across contexts
- Causal modeling

### Use Generative Models When:

❌ **Need actual samples**
- Data augmentation
- Synthetic data generation
- Uncertainty quantification

❌ **Reconstruction quality matters**
- Image generation
- High-fidelity synthesis

❌ **Distribution modeling is the goal**
- Density estimation
- Anomaly detection

### Best: Hybrid JEPA + Generative

**Combine both**:
1. JEPA learns dynamics efficiently
2. Generative model handles sampling
3. Get prediction + generation + uncertainty

---

## The Path Forward

### Stage 1: Basic JEPA (Current)
- Understand architecture
- Implement encoder + predictor
- VICReg regularization
- Toy examples

### Stage 2: Bio-JEPA
- Apply to Perturb-seq
- Perturbation conditioning
- Evaluate on held-out perturbations

### Stage 3: Joint Latent Spaces
- Combine bulk + single-cell
- Static + dynamic data
- Multi-omics integration

### Stage 4: JEPA + Generative
- Add diffusion decoder
- Uncertainty quantification
- Full predictive-generative system

---

## Key Takeaways

### Conceptual

1. **Predict embeddings, not pixels** — more efficient, more robust
2. **No reconstruction needed** — focus on semantic content
3. **No contrastive negatives** — simpler than SimCLR/MoCo
4. **World models without generation** — learn dynamics efficiently
5. **Joint latent spaces** — static and dynamic data train each other

### Practical

1. **JEPA is not generative** — predicts embeddings, not samples
2. **VICReg prevents collapse** — variance + covariance regularization
3. **Powerful predictor needed** — transformer-based works well
4. **Combine with generative** — for sampling and uncertainty
5. **Perfect for biology** — perturbations, trajectories, multi-omics

### For Computational Biology

1. **Perturb-seq is ideal use case** — predict perturbed states
2. **Efficiency matters** — 20K genes, millions of cells
3. **Robustness critical** — technical noise, batch effects
4. **Compositional reasoning** — combine perturbations
5. **Hybrid approach best** — JEPA + diffusion for full system

---

## Related Documents

- [01_jepa_foundations.md](01_jepa_foundations.md) — Architecture details
- [02_jepa_training.md](02_jepa_training.md) — Training strategies
- [03_jepa_applications.md](03_jepa_applications.md) — Vision to biology mapping
- [04_jepa_perturbseq.md](04_jepa_perturbseq.md) — Perturb-seq application
- [Joint Latent Spaces](../incubation/joint_latent_space_and_JEPA.md) — Goku insights

---

## References

**JEPA papers**:
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA, 2023)
- Bardes et al., "V-JEPA: Latent Video Prediction for Visual Representation Learning" (2024)
- Meta AI, "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning" (2025)

**Joint latent spaces**:
- ByteDance & HKU, "Goku: Native Joint Image-Video Generation" (2024)

**VICReg**:
- Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning" (2022)

**Biology applications** (to be developed):
- Perturb-seq prediction
- Trajectory inference
- Multi-omics integration
