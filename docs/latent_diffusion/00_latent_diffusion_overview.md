# Latent Diffusion Models: Overview

**Latent Diffusion Models (LDMs)** combine the efficiency of latent variable models with the power of diffusion models, enabling high-quality generation at a fraction of the computational cost.

**Key insight**: Diffuse in a compressed latent space instead of high-dimensional pixel/gene space.

---

## The Problem with Pixel-Space Diffusion

### Computational Cost

**Standard diffusion models** operate directly on data:
- **Images**: Diffuse in $\mathbb{R}^{H \times W \times C}$ (e.g., 256×256×3 = 196,608 dims)
- **Gene expression**: Diffuse in $\mathbb{R}^{20000}$ (20K genes)
- **Multi-omics**: Even higher dimensional

**Consequences**:
1. **Slow training** — Many denoising steps on high-dim data
2. **Slow sampling** — 50-1000 steps in high-dim space
3. **Memory intensive** — Store gradients for all dimensions
4. **Inefficient** — Most dimensions are redundant

**Example**: DDPM on 256×256 images
- Training: ~1 week on 8 GPUs
- Sampling: ~10 seconds per image (50 steps)
- Memory: ~16GB per batch

---

## The Solution: Latent Diffusion

### Core Idea

**Two-stage approach**:

**Stage 1: Learn compressed latent space** (VAE/VQ-VAE)
```
x ∈ ℝ^20000 → Encoder → z ∈ ℝ^256 → Decoder → x̂ ∈ ℝ^20000
```

**Stage 2: Diffusion in latent space**
```
z₀ → ... → zₜ → ... → z_T
(Diffusion operates on ℝ^256 instead of ℝ^20000)
```

**Benefits**:
- **78× fewer dimensions** (20K → 256)
- **10-100× faster training**
- **5-10× faster sampling**
- **Better sample quality** (focuses on semantic content)

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Latent Diffusion Model              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐         ┌──────────┐                │
│  │   Data   │────────>│ Encoder  │                │
│  │ x ∈ ℝ^D  │         │ (VAE/VQ) │                │
│  └──────────┘         └────┬─────┘                │
│                            │                        │
│                            v                        │
│                      ┌──────────┐                  │
│                      │ Latent z │                  │
│                      │  ∈ ℝ^d   │                  │
│                      └────┬─────┘                  │
│                            │                        │
│                            v                        │
│                   ┌────────────────┐               │
│                   │   Diffusion    │               │
│                   │  z₀ → zₜ → z_T │               │
│                   └────────┬───────┘               │
│                            │                        │
│                            v                        │
│                      ┌──────────┐                  │
│                      │ Denoised │                  │
│                      │    z₀    │                  │
│                      └────┬─────┘                  │
│                            │                        │
│                            v                        │
│  ┌──────────┐         ┌──────────┐                │
│  │ Generated│<────────│ Decoder  │                │
│  │    x̂     │         │ (VAE/VQ) │                │
│  └──────────┘         └──────────┘                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Why Latent Diffusion for Biology?

### 1. Dimensionality Reduction

**Gene expression is high-dimensional but low-rank**:
- 20K genes measured
- ~100-500 effective dimensions (pathways, modules)
- Most variation captured by top PCs

**Latent diffusion exploits this**:
- Compress to semantic latent space
- Diffuse in compressed space
- Decode to full gene space

### 2. Computational Efficiency

**For single-cell data**:
- Millions of cells × 20K genes = intractable
- Latent space: Millions of cells × 256 dims = manageable

**Speedup example**:
- Pixel-space: 1000 steps × 20K dims = 20M operations
- Latent-space: 1000 steps × 256 dims = 256K operations
- **78× faster**

### 3. Better Generalization

**Latent space focuses on biology**:
- Remove technical noise (batch effects, dropout)
- Capture biological variation (cell types, states)
- Generalize better to new conditions

### 4. Multi-Modal Integration

**Natural for multi-omics**:
- Shared latent space for RNA + Protein + ATAC
- Diffusion operates on joint representation
- Generate any modality from latent

---

## Latent Diffusion vs Alternatives

### vs Pixel-Space Diffusion

| Aspect | Pixel-Space | Latent Diffusion |
|--------|-------------|------------------|
| **Training speed** | Slow | 10-100× faster |
| **Sampling speed** | Slow | 5-10× faster |
| **Memory** | High | Low |
| **Quality** | Good | Better (semantic focus) |
| **Interpretability** | Low | Higher (latent structure) |

### vs VAE Alone

| Aspect | VAE | Latent Diffusion |
|--------|-----|------------------|
| **Sample quality** | Blurry | Sharp |
| **Mode coverage** | Poor | Excellent |
| **Training** | Fast | Moderate |
| **Sampling** | Fast | Moderate |
| **Likelihood** | Tractable | Intractable |

### vs GAN

| Aspect | GAN | Latent Diffusion |
|--------|-----|------------------|
| **Training stability** | Unstable | Stable |
| **Mode coverage** | Poor | Excellent |
| **Sample quality** | Excellent | Excellent |
| **Likelihood** | No | Yes (approximate) |
| **Controllability** | Moderate | High |

---

## Key Components

### 1. Autoencoder (VAE or VQ-VAE)

**Purpose**: Compress data to latent space

**Options**:
- **VAE**: Continuous latent, probabilistic
- **VQ-VAE**: Discrete latent, deterministic
- **VQ-GAN**: Discrete + adversarial (best quality)

**For biology**: VAE is simpler and works well

### 2. Latent Diffusion Model

**Purpose**: Generate latent codes

**Options**:
- **DDPM**: Original diffusion
- **DDIM**: Faster sampling
- **Rectified Flow**: Straight paths
- **DiT**: Transformer-based

**For biology**: Rectified Flow + DiT (best efficiency)

### 3. Conditioning Mechanism

**Purpose**: Control generation

**Options**:
- **Class labels**: Cell type, perturbation
- **Continuous**: Time, dose, expression levels
- **Cross-attention**: Text, gene sets, pathways
- **Concatenation**: Simple but effective

**For biology**: Cross-attention for gene sets, concatenation for perturbations

---

## Applications in Computational Biology

### 1. Single-Cell Generation

**Task**: Generate realistic single-cell profiles

**Approach**:
```
Train VAE: scRNA-seq → latent
Train diffusion: latent → latent
Sample: noise → latent → scRNA-seq
```

**Benefits**:
- Data augmentation
- Rare cell type generation
- Batch effect removal

### 2. Perturbation Prediction

**Task**: Predict cellular response to perturbations

**Approach**:
```
Condition on: baseline + perturbation
Generate: perturbed state
```

**Benefits**:
- Virtual screening
- Combination prediction
- Mechanism discovery

### 3. Multi-Omics Translation

**Task**: Predict one modality from another

**Approach**:
```
Train joint VAE: RNA + Protein → shared latent
Condition diffusion on: RNA latent
Generate: Protein latent → Protein
```

**Benefits**:
- Fill missing modalities
- Cross-modality validation
- Integrated analysis

### 4. Trajectory Modeling

**Task**: Model developmental/disease trajectories

**Approach**:
```
Condition on: time, cell state
Generate: future states
```

**Benefits**:
- Predict differentiation
- Model disease progression
- Identify branch points

### 5. Spatial Transcriptomics

**Task**: Generate spatial gene expression

**Approach**:
```
Condition on: spatial coordinates
Generate: expression at location
```

**Benefits**:
- Super-resolution
- Missing region imputation
- 3D reconstruction

---

## Training Strategy

### Two-Stage Training

**Stage 1: Train Autoencoder**
```python
# Train VAE on gene expression
vae = VAE(input_dim=20000, latent_dim=256)
train_vae(vae, gene_expression_data)

# Freeze encoder/decoder
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
```

**Stage 2: Train Diffusion in Latent Space**
```python
# Encode data to latent
z = vae.encode(gene_expression_data)

# Train diffusion on latent codes
diffusion = LatentDiffusion(latent_dim=256)
train_diffusion(diffusion, z)
```

### Joint Fine-Tuning (Optional)

**After separate training, fine-tune together**:
```python
# Unfreeze all
for param in vae.parameters():
    param.requires_grad = True
for param in diffusion.parameters():
    param.requires_grad = True

# Fine-tune end-to-end
train_joint(vae, diffusion, gene_expression_data)
```

---

## Sampling Process

### Generation Pipeline

**1. Sample latent from diffusion**:
```python
# Start from noise
z_T = torch.randn(batch_size, latent_dim)

# Denoise
z_0 = diffusion.sample(z_T, num_steps=50)
```

**2. Decode to data space**:
```python
# Decode latent to gene expression
x_gen = vae.decode(z_0)
```

**3. Post-processing** (optional):
```python
# Ensure non-negative (for counts)
x_gen = torch.clamp(x_gen, min=0)

# Normalize
x_gen = normalize(x_gen)
```

### Conditional Generation

**With perturbation conditioning**:
```python
# Encode baseline
z_baseline = vae.encode(x_baseline)

# Add perturbation embedding
z_cond = torch.cat([z_baseline, pert_emb], dim=-1)

# Sample with conditioning
z_T = torch.randn(batch_size, latent_dim)
z_0 = diffusion.sample(z_T, condition=z_cond, num_steps=50)

# Decode
x_perturbed = vae.decode(z_0)
```

---

## Advantages for Biology

### 1. Efficiency

**Computational**:
- 10-100× faster training than pixel-space
- 5-10× faster sampling
- Scalable to millions of cells

**Memory**:
- Lower memory footprint
- Larger batch sizes possible
- Distributed training easier

### 2. Quality

**Better samples**:
- Sharper than VAE
- More diverse than GAN
- Biologically realistic

**Robustness**:
- Handles technical noise
- Generalizes across batches
- Stable training

### 3. Interpretability

**Latent structure**:
- Dimensions correspond to biology
- Can analyze latent space
- Identify key factors

**Controllability**:
- Fine-grained conditioning
- Interpolation in latent space
- Compositional generation

### 4. Flexibility

**Multi-modal**:
- Shared latent for multi-omics
- Cross-modality generation
- Integrated analysis

**Multi-task**:
- Single model for multiple tasks
- Transfer learning
- Few-shot adaptation

---

## Comparison: Stable Diffusion vs Bio Latent Diffusion

### Stable Diffusion (Images)

**Architecture**:
- VQ-GAN encoder/decoder (8× compression)
- U-Net diffusion model
- CLIP text conditioning

**Training**:
- LAION-5B dataset (5 billion images)
- 256×256 or 512×512 resolution
- Text-to-image generation

### Bio Latent Diffusion (Gene Expression)

**Architecture**:
- VAE encoder/decoder (78× compression)
- DiT or U-Net diffusion model
- Perturbation/cell-type conditioning

**Training**:
- Single-cell datasets (millions of cells)
- 20K genes → 256 latent dims
- Perturbation/trajectory prediction

**Key differences**:
1. **Compression ratio**: Higher for biology (78× vs 8×)
2. **Conditioning**: Biological metadata vs text
3. **Data structure**: Tabular vs spatial
4. **Objectives**: Prediction + generation vs generation only

---

## When to Use Latent Diffusion

### ✅ Use Latent Diffusion When:

**High-dimensional data**:
- Gene expression (20K genes)
- Multi-omics (RNA + Protein + ATAC)
- Spatial transcriptomics

**Need efficiency**:
- Large datasets (millions of cells)
- Limited compute
- Fast sampling required

**Want quality + diversity**:
- Better than VAE (sharper)
- Better than GAN (mode coverage)
- Stable training

**Multi-task learning**:
- Generation + prediction
- Multiple conditions
- Transfer across datasets

### ❌ Don't Use Latent Diffusion When:

**Low-dimensional data**:
- Already <1000 dims
- Pixel-space diffusion is fine

**Need exact likelihood**:
- VAE or normalizing flow better
- Latent diffusion likelihood is approximate

**Real-time inference**:
- Sampling still slower than VAE/GAN
- Consider distillation or few-step methods

**Simple tasks**:
- Linear models sufficient
- Overkill for simple prediction

---

## Key Takeaways

### Conceptual

1. **Two-stage approach** — VAE compression + latent diffusion
2. **Efficiency** — 10-100× faster than pixel-space
3. **Quality** — Better than VAE, stable than GAN
4. **Flexibility** — Multi-modal, multi-task, controllable

### Practical

1. **Train VAE first** — Get good latent space
2. **Freeze VAE** — Train diffusion on latent codes
3. **Optional fine-tuning** — Joint optimization
4. **Condition carefully** — Use appropriate conditioning mechanism

### For Biology

1. **Perfect for scRNA-seq** — High-dim, low-rank structure
2. **Enables multi-omics** — Shared latent space
3. **Scalable** — Millions of cells
4. **Interpretable** — Latent dimensions have meaning

---

## Related Documents

- [01_latent_diffusion_foundations.md](01_latent_diffusion_foundations.md) — Architecture details
- [02_latent_diffusion_training.md](02_latent_diffusion_training.md) — Training strategies
- [03_latent_diffusion_applications.md](03_latent_diffusion_applications.md) — Biology applications
- [04_latent_diffusion_combio.md](04_latent_diffusion_combio.md) — Complete implementation

---

## References

**Latent Diffusion**:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
- Vahdat et al. (2021): "Score-based Generative Modeling in Latent Space"

**Autoencoders**:
- Kingma & Welling (2014): "Auto-Encoding Variational Bayes" (VAE)
- van den Oord et al. (2017): "Neural Discrete Representation Learning" (VQ-VAE)
- Esser et al. (2021): "Taming Transformers for High-Resolution Image Synthesis" (VQ-GAN)

**Biology Applications**:
- Lopez et al. (2018): "Deep generative modeling for single-cell transcriptomics" (scVI)
- Lotfollahi et al. (2023): "Predicting cellular responses to novel drug combinations"
- Bunne et al. (2023): "Learning Single-Cell Perturbation Responses using Neural Optimal Transport"
