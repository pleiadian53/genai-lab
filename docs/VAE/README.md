# Variational Autoencoders (VAE)

**Comprehensive guide to VAEs for gene expression and count data modeling.**

---

## Overview

Variational Autoencoders (VAEs) are powerful generative models that learn compressed representations of high-dimensional data. For computational biology, VAEs are particularly useful for:

- **Dimensionality reduction**: Learn low-dimensional latent spaces for gene expression
- **Denoising**: Remove technical noise from single-cell data
- **Generation**: Create synthetic samples for data augmentation
- **Perturbation prediction**: Model drug responses and genetic perturbations

This series covers VAE theory, implementation, and specialized variants for biological count data.

---

## Document Series

### Core Theory

| Document | Topic | Key Concepts |
|----------|-------|--------------|
| [VAE-01: Overview](VAE-01-overview.md) | Introduction to VAEs | Encoder-decoder, latent space, variational inference |
| [VAE-02: ELBO](VAE-02-elbo.md) | Evidence Lower Bound | Reconstruction + KL divergence, variational objective |
| [VAE-03: Inference](VAE-03-inference.md) | Inference & generation | Posterior q(z\|x), prior p(z), sampling |

### Gradient Estimation

| Document | Topic | Key Concepts |
|----------|-------|--------------|
| [VAE-04: Reparameterization](VAE-04-reparameterization.md) | Reparameterization trick | Backprop through stochastic nodes |
| [VAE-05: Pathwise Derivative](VAE-05-pathwise-derivative.md) | Pathwise gradient estimator | Score function vs. pathwise |
| [VAE-05a: Pathwise Details](VAE-05a-pathwise-gradient-estimator.md) | Implementation details | Practical considerations |

### Training & Optimization

| Document | Topic | Key Concepts |
|----------|-------|--------------|
| [VAE-06: Optimization](VAE-06-optimization-new.md) | Training strategies | KL annealing, batch normalization, regularization |
| [VAE Model Training](VAE-model-training.md) | Implementation guide | PyTorch training loops, hyperparameters |

### Count Data & Biology

| Document | Topic | Key Concepts |
|----------|-------|--------------|
| [VAE-07: NB & ZINB](VAE-07-NB-ZINB.md) | Count data decoders | Negative Binomial, Zero-Inflated NB |
| [VAE-08: NB Likelihood](VAE-08-NB-likelihood.md) | NB loss derivation | Gamma-Poisson mixture, dispersion |

### Applications

| Document | Topic | Key Concepts |
|----------|-------|--------------|
| [VAE for Prediction](VAE-for-prediction.md) | Predictive modeling | Conditioning, perturbation response |
| [VAE-09: Roadmap](VAE-09-roadmap-discussion.md) | Extensions & future work | Hierarchical VAE, disentanglement, causal |

---

## Quick Start Guide

### 1. **Start with the Basics**
Read in order:
1. [VAE-01: Overview](VAE-01-overview.md) - Understand the overall framework
2. [VAE-02: ELBO](VAE-02-elbo.md) - Learn the training objective
3. [VAE-03: Inference](VAE-03-inference.md) - Understand latent space and sampling

### 2. **Understand Gradients**
Essential for implementation:
1. [VAE-04: Reparameterization](VAE-04-reparameterization.md) - The key trick for backprop
2. [VAE-05: Pathwise Derivative](VAE-05-pathwise-derivative.md) - Why it works

### 3. **Train Your First VAE**
1. [VAE-06: Optimization](VAE-06-optimization-new.md) - Training strategies
2. [VAE Model Training](VAE-model-training.md) - Hands-on implementation

### 4. **Handle Count Data**
For scRNA-seq and bulk RNA-seq:
1. [VAE-07: NB & ZINB](VAE-07-NB-ZINB.md) - Specialized decoders
2. [VAE-08: NB Likelihood](VAE-08-NB-likelihood.md) - Mathematical details

### 5. **Build Applications**
1. [VAE for Prediction](VAE-for-prediction.md) - Perturbation modeling
2. [VAE-09: Roadmap](VAE-09-roadmap-discussion.md) - Advanced topics

---

## VAE Variants Implemented

### Conditional VAE (CVAE)
**Use case:** Conditional generation (e.g., cell type → expression)

```python
from genailab.model.vae import CVAE

model = CVAE(
    input_dim=2000,      # genes
    latent_dim=10,       # compressed representation
    condition_dim=5,     # cell types
    hidden_dims=[512, 256]
)
```

### CVAE with Negative Binomial (CVAE_NB)
**Use case:** Single-cell RNA-seq (count data with overdispersion)

```python
from genailab.model.vae import CVAE_NB

model = CVAE_NB(
    input_dim=2000,
    latent_dim=10,
    condition_dim=5
)
# Predicts mean μ and dispersion r for NB distribution
```

### CVAE with Zero-Inflated NB (CVAE_ZINB)
**Use case:** scRNA-seq with dropout (many zeros)

```python
from genailab.model.vae import CVAE_ZINB

model = CVAE_ZINB(
    input_dim=2000,
    latent_dim=10,
    condition_dim=5
)
# Predicts μ, r, and dropout probability π
```

---

## Key Concepts

### Encoder (Recognition Network)
**Purpose:** Map data x to latent distribution q(z|x)

```
x (gene expression) → Neural Net → μ_z, σ_z → z ~ N(μ_z, σ_z²)
```

### Decoder (Generative Network)
**Purpose:** Map latent z back to data distribution p(x|z)

```
z (latent code) → Neural Net → reconstruction x̂
```

**Decoder types:**
- **Gaussian**: MSE loss, for continuous data
- **Negative Binomial**: For count data with variance > mean
- **Zero-Inflated NB**: For sparse count data (scRNA-seq)

### ELBO (Evidence Lower Bound)
**Training objective:**

$$
\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{KL[q(z|x) || p(z)]}_{\text{Regularization}}
$$

- **Reconstruction**: How well can we regenerate x from z?
- **KL divergence**: How close is q(z|x) to prior p(z)?

---

## Applications in Computational Biology

### 1. **Denoising scRNA-seq**
**Problem:** Technical noise, dropout, batch effects  
**Solution:** VAE learns clean latent representation  
**Model:** CVAE_ZINB with batch/cell type conditioning

### 2. **Drug Response Prediction**
**Problem:** Predict perturbed expression from baseline  
**Solution:** Conditional VAE with drug embeddings  
**Model:** CVAE conditioned on [baseline expression, drug ID, dose]

### 3. **Data Augmentation**
**Problem:** Limited training samples  
**Solution:** Generate synthetic samples from learned distribution  
**Model:** Sample from p(z), decode to get new x

### 4. **Batch Correction**
**Problem:** Technical variation across experiments  
**Solution:** Learn batch-invariant latent space  
**Model:** CVAE with adversarial batch discriminator

### 5. **Cell Type Discovery**
**Problem:** Identify novel cell types  
**Solution:** Cluster in learned latent space  
**Model:** VAE → t-SNE/UMAP on z → clustering

---

## Comparison: VAE vs. Other Generative Models

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **VAE** | Fast, stable, explicit latent space | Can be blurry, mode averaging | Representation learning, denoising |
| **GAN** | Sharp samples, high quality | Training instability, mode collapse | Image generation |
| **Diffusion** | High quality, stable training | Slow sampling | State-of-the-art generation |
| **Flow** | Exact likelihood, invertible | Complex architecture | Density estimation |

**For gene expression:** VAE is often preferred for its:
- Interpretable latent space
- Fast inference (single forward pass)
- Stable training
- Uncertainty quantification

---

## Related Topics

### Within This Project

- **[DDPM](../DDPM/)** - Diffusion models (slower but higher quality)
- **[Flow Matching](../flow_matching/)** - Continuous normalizing flows
- **[Beta-VAE](../beta-VAE/)** - Disentangled representations
- **[Foundation Models](../foundation_models/)** - Pre-trained encoders

### External Resources

- **scVI** - Industry-standard VAE for scRNA-seq
- **CPA** (Compositional Perturbation Autoencoder) - Perturbation prediction
- **Geneformer** - Foundation model (can replace VAE encoder)

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| **CVAE (Gaussian)** | ✅ Complete | `src/genailab/model/vae.py` |
| **CVAE_NB** | ✅ Complete | `src/genailab/model/vae.py` |
| **CVAE_ZINB** | ✅ Complete | `src/genailab/model/vae.py` |
| **Training scripts** | ✅ Complete | `scripts/` |
| **Evaluation metrics** | ✅ Complete | `src/genailab/eval/` |
| **Interactive notebooks** | 📋 Planned | `notebooks/vae/` |

---

## Frequently Asked Questions

### When should I use VAE vs. Diffusion?

**Use VAE when:**
- Fast inference is critical
- You need interpretable latent representations
- Working with small-to-medium datasets
- Uncertainty quantification is important

**Use Diffusion when:**
- Generation quality is top priority
- You have large datasets and compute
- Slow sampling (100+ steps) is acceptable

### How to choose latent dimension?

**Guidelines:**
- **Gene expression**: 10-50 dims (10-20 typical for scRNA-seq)
- **Rule of thumb**: Start with 10-20, increase if reconstruction is poor
- **Validate**: Plot reconstruction error vs. latent dim

### What decoder should I use?

| Data Type | Decoder | Reason |
|-----------|---------|--------|
| Normalized (log-transformed) | Gaussian (MSE) | Simple, fast |
| Raw counts (bulk RNA-seq) | Negative Binomial | Handles overdispersion |
| scRNA-seq (sparse) | ZINB | Handles zeros and overdispersion |

---

**Questions or suggestions?** Open an issue on [GitHub](https://github.com/pleiadian53/genai-lab/issues)
