# Latent Diffusion Models Documentation

**Latent Diffusion Models (LDMs)** — Efficient high-quality generation by diffusing in compressed latent space instead of pixel/gene space.

This documentation series covers latent diffusion from theory through implementation, with a focus on computational biology applications including single-cell generation, perturbation prediction, and multi-omics integration.

---

## Core Documentation Series

### 1. Overview
**[00_latent_diffusion_overview.md](00_latent_diffusion_overview.md)** — What is latent diffusion and why it matters
- The problem with pixel-space diffusion
- Two-stage approach: VAE + diffusion
- Why latent diffusion for biology
- Comparison with alternatives
- Applications overview

### 2. Foundations
**[01_latent_diffusion_foundations.md](01_latent_diffusion_foundations.md)** — Architecture and components
- VAE/VQ-VAE autoencoders
- Latent diffusion models
- Conditioning mechanisms
- Complete PyTorch implementations

### 3. Training
**[02_latent_diffusion_training.md](02_latent_diffusion_training.md)** — Training strategies
- Two-stage training (VAE → Diffusion)
- Joint fine-tuning
- Hyperparameters
- Optimization strategies
- Monitoring and debugging

### 4. Applications
**[03_latent_diffusion_applications.md](03_latent_diffusion_applications.md)** — Biology applications
- Single-cell generation
- Perturbation prediction
- Multi-omics translation
- Trajectory modeling
- Spatial transcriptomics

### 5. Computational Biology Implementation
**[04_latent_diffusion_combio.md](04_latent_diffusion_combio.md)** — Complete implementation
- scRNA-seq latent diffusion
- Perturb-seq with latent diffusion
- Multi-omics latent diffusion
- End-to-end training and evaluation

---

## Quick Navigation

### For Different Audiences

**New to Latent Diffusion?**
1. Start with [Overview](00_latent_diffusion_overview.md)
2. Understand the two-stage approach
3. See why it's efficient (10-100× speedup)
4. Review biology applications

**Coming from Diffusion Models?**
1. Read [Overview](00_latent_diffusion_overview.md) comparison section
2. Understand VAE compression stage
3. Learn when latent diffusion is better
4. See efficiency gains

**Coming from VAE?**
1. Read why VAE alone isn't enough
2. Understand how diffusion improves quality
3. Learn the two-stage training
4. See applications

**Ready to Implement?**
1. Review [Foundations](01_latent_diffusion_foundations.md) architecture
2. Follow [Training](02_latent_diffusion_training.md) pipeline
3. Adapt [Combio Implementation](04_latent_diffusion_combio.md) code
4. Evaluate on your data

---

## Key Concepts

### The Two-Stage Approach

**Stage 1: Autoencoder** (VAE/VQ-VAE)
```
x ∈ ℝ^20000 → Encoder → z ∈ ℝ^256 → Decoder → x̂ ∈ ℝ^20000
```
- Compress high-dimensional data to latent space
- Learn semantic representation
- Freeze after training

**Stage 2: Diffusion in Latent Space**
```
z₀ → ... → zₜ → ... → z_T
(Operate on ℝ^256 instead of ℝ^20000)
```
- Diffuse in compressed space
- 78× fewer dimensions
- 10-100× faster

### Why This Works

**Biological data is low-rank**:
- 20K genes measured
- ~100-500 effective dimensions
- Most variation in top PCs

**Latent space captures biology**:
- Cell types
- Pathways
- States
- Transitions

**Diffusion adds quality**:
- Sharper than VAE
- Better mode coverage
- Stable training

---

## Comparison Tables

### Latent Diffusion vs Pixel-Space Diffusion

| Aspect | Pixel-Space | Latent Diffusion |
|--------|-------------|------------------|
| **Dimensions** | 20,000 | 256 |
| **Training time** | 1 week | 1 day |
| **Sampling time** | 10s | 1-2s |
| **Memory** | 16GB | 2GB |
| **Quality** | Good | Better |
| **Interpretability** | Low | Higher |

### Latent Diffusion vs VAE

| Aspect | VAE | Latent Diffusion |
|--------|-----|------------------|
| **Sample quality** | Blurry | Sharp |
| **Mode coverage** | Poor | Excellent |
| **Training** | Fast | Moderate |
| **Sampling** | Instant | Moderate (50 steps) |
| **Likelihood** | Exact | Approximate |

### Latent Diffusion vs GAN

| Aspect | GAN | Latent Diffusion |
|--------|-----|------------------|
| **Training stability** | Unstable | Stable |
| **Mode coverage** | Poor | Excellent |
| **Sample quality** | Excellent | Excellent |
| **Controllability** | Moderate | High |
| **Likelihood** | No | Yes (approximate) |

---

## Architecture Components

### 1. Autoencoder (Stage 1)

**Purpose**: Learn compressed latent representation

**Options**:
- **VAE** — Continuous latent, probabilistic
- **VQ-VAE** — Discrete latent, codebook
- **VQ-GAN** — Discrete + adversarial (best quality)

**For biology**: VAE works well, simple and effective

**Architecture**:
```python
class BiologicalVAE(nn.Module):
    def __init__(self, num_genes=20000, latent_dim=256):
        self.encoder = Encoder(num_genes, latent_dim)
        self.decoder = Decoder(latent_dim, num_genes)
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return z
    
    def decode(self, z):
        return self.decoder(z)
```

### 2. Latent Diffusion Model (Stage 2)

**Purpose**: Generate latent codes

**Options**:
- **DDPM** — Original diffusion
- **DDIM** — Deterministic, faster sampling
- **Rectified Flow** — Straight paths, optimal
- **DiT** — Transformer-based

**For biology**: Rectified Flow + DiT (best efficiency)

**Architecture**:
```python
class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim=256):
        self.model = DiT(input_dim=latent_dim)
        self.flow = RectifiedFlow()
    
    def forward(self, z, t):
        return self.model(z, t)
    
    def sample(self, num_samples, num_steps=50):
        z_T = torch.randn(num_samples, latent_dim)
        z_0 = self.flow.sample(z_T, num_steps)
        return z_0
```

### 3. Conditioning

**Purpose**: Control generation

**Mechanisms**:
- **Concatenation** — Simple, effective
- **Cross-attention** — Flexible, powerful
- **FiLM** — Affine transformation
- **Adaptive LayerNorm** — DiT-style

**For biology**:
- Cell type: Class embedding
- Perturbation: Gene embedding
- Time: Sinusoidal encoding
- Continuous: Direct concatenation

---

## Biology Applications

### 1. Single-Cell Generation

**Task**: Generate realistic single-cell profiles

**Workflow**:
1. Train VAE on scRNA-seq
2. Train diffusion on latent codes
3. Sample: noise → latent → scRNA-seq

**Benefits**:
- Data augmentation
- Rare cell type generation
- Batch effect removal

**Use cases**:
- Expand training data
- Generate synthetic controls
- Simulate experiments

### 2. Perturbation Prediction

**Task**: Predict cellular response to perturbations

**Workflow**:
1. Encode baseline to latent
2. Condition diffusion on perturbation
3. Generate perturbed latent
4. Decode to gene expression

**Benefits**:
- Virtual screening
- Combination prediction
- Mechanism discovery

**Use cases**:
- Drug discovery
- CRISPR screening
- Genetic interaction mapping

### 3. Multi-Omics Translation

**Task**: Predict one modality from another

**Workflow**:
1. Train joint VAE (RNA + Protein → shared latent)
2. Condition diffusion on RNA latent
3. Generate Protein latent
4. Decode to Protein expression

**Benefits**:
- Fill missing modalities
- Cross-modality validation
- Integrated analysis

**Use cases**:
- CITE-seq imputation
- Predict protein from RNA
- Multi-omics integration

### 4. Trajectory Modeling

**Task**: Model developmental/disease trajectories

**Workflow**:
1. Encode cells to latent
2. Condition diffusion on time
3. Generate future states
4. Decode to expression

**Benefits**:
- Predict differentiation
- Model disease progression
- Identify branch points

**Use cases**:
- Developmental biology
- Disease modeling
- Drug response over time

### 5. Spatial Transcriptomics

**Task**: Generate spatial gene expression

**Workflow**:
1. Encode spatial data to latent
2. Condition diffusion on coordinates
3. Generate expression at location
4. Decode to genes

**Benefits**:
- Super-resolution
- Missing region imputation
- 3D reconstruction

**Use cases**:
- Enhance spatial resolution
- Fill tissue gaps
- Predict 3D structure

---

## Training Pipeline

### Two-Stage Training

**Stage 1: Train Autoencoder**
```python
# Train VAE
vae = BiologicalVAE(num_genes=20000, latent_dim=256)
train_vae(vae, scrnaseq_data, num_epochs=100)

# Freeze
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
```

**Stage 2: Train Diffusion**
```python
# Encode data to latent
with torch.no_grad():
    latents = vae.encode(scrnaseq_data)

# Train diffusion
diffusion = LatentDiffusion(latent_dim=256)
train_diffusion(diffusion, latents, num_epochs=100)
```

**Optional: Joint Fine-Tuning**
```python
# Unfreeze all
for param in vae.parameters():
    param.requires_grad = True

# Fine-tune together
train_joint(vae, diffusion, scrnaseq_data, num_epochs=20)
```

### Sampling Pipeline

**Generate new samples**:
```python
# Sample latent from diffusion
z_0 = diffusion.sample(num_samples=100, num_steps=50)

# Decode to gene expression
x_gen = vae.decode(z_0)
```

**Conditional generation**:
```python
# Condition on cell type
cell_type_emb = encode_cell_type("T cell")
z_0 = diffusion.sample(num_samples=100, condition=cell_type_emb)
x_gen = vae.decode(z_0)
```

---

## Efficiency Gains

### Computational Savings

**For 20K genes → 256 latent dims**:

| Operation | Pixel-Space | Latent Space | Speedup |
|-----------|-------------|--------------|---------|
| **Forward pass** | 20K dims | 256 dims | 78× |
| **Training epoch** | 1 hour | 5 min | 12× |
| **Full training** | 1 week | 1 day | 7× |
| **Sampling** | 10s | 1-2s | 5-10× |
| **Memory** | 16GB | 2GB | 8× |

### Quality Improvements

**Better than VAE**:
- Sharper samples (no blurriness)
- Better mode coverage (all cell types)
- More realistic (passes biological QC)

**Comparable to pixel-space diffusion**:
- Same sample quality
- Better efficiency
- More interpretable

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
- Fast iteration required

**Want quality + diversity**:
- Better than VAE
- Stable than GAN
- Good mode coverage

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
- Sampling still slower than VAE
- Consider distillation

**Simple tasks**:
- Linear models sufficient
- Overkill for simple prediction

---

## Implementation Roadmap

### Phase 1: VAE Training
- [ ] Data preprocessing
- [ ] VAE architecture
- [ ] Training loop
- [ ] Reconstruction quality check
- [ ] Latent space visualization

### Phase 2: Latent Diffusion
- [ ] Encode data to latent
- [ ] Diffusion model architecture
- [ ] Training on latent codes
- [ ] Sampling quality check
- [ ] Conditional generation

### Phase 3: Applications
- [ ] Single-cell generation
- [ ] Perturbation prediction
- [ ] Multi-omics translation
- [ ] Trajectory modeling
- [ ] Evaluation metrics

### Phase 4: Optimization
- [ ] Joint fine-tuning
- [ ] Faster sampling (DDIM, few-step)
- [ ] Classifier-free guidance
- [ ] Distributed training
- [ ] Production deployment

---

## Learning Path

### Beginner Path
1. **Understand the concept** — [Overview](00_latent_diffusion_overview.md)
2. **Learn VAE basics** — Autoencoder stage
3. **Learn diffusion basics** — Latent diffusion stage
4. **See applications** — [Applications](03_latent_diffusion_applications.md)

### Intermediate Path
1. **Review architecture** — [Foundations](01_latent_diffusion_foundations.md)
2. **Implement VAE** — Train on scRNA-seq
3. **Implement diffusion** — Train on latent codes
4. **Apply to biology** — [Combio Implementation](04_latent_diffusion_combio.md)

### Advanced Path
1. **Joint fine-tuning** — End-to-end optimization
2. **Multi-modal** — Multi-omics integration
3. **Novel conditioning** — Custom conditioning mechanisms
4. **Production** — Scale to millions of cells

---

## Related Documentation

### Within This Project

**Diffusion Models**:
- [DDPM](../DDPM/) — Denoising diffusion
- [SDE](../SDE/) — Stochastic differential equations
- [Flow Matching](../flow_matching/) — Rectified flow
- [DiT](../DiT/) — Diffusion transformers

**Representation Learning**:
- [VAE](../VAE/) — Variational autoencoders
- [JEPA](../JEPA/) — Joint embedding predictive architecture

**Architecture Choices**:
- [Gene Expression Architectures](../DDPM/02a_diffusion_arch_gene_expression.md)

### External Resources

**Latent Diffusion Papers**:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
- Vahdat et al. (2021): "Score-based Generative Modeling in Latent Space"

**Autoencoder Papers**:
- Kingma & Welling (2014): "Auto-Encoding Variational Bayes" (VAE)
- van den Oord et al. (2017): "Neural Discrete Representation Learning" (VQ-VAE)
- Esser et al. (2021): "Taming Transformers for High-Resolution Image Synthesis" (VQ-GAN)

**Biology Applications**:
- Lopez et al. (2018): "Deep generative modeling for single-cell transcriptomics" (scVI)
- Lotfollahi et al. (2023): "Predicting cellular responses to novel drug combinations"
- Bunne et al. (2023): "Learning Single-Cell Perturbation Responses using Neural Optimal Transport"

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
4. **Condition carefully** — Use appropriate mechanism

### For Biology
1. **Perfect for scRNA-seq** — High-dim, low-rank
2. **Enables multi-omics** — Shared latent space
3. **Scalable** — Millions of cells
4. **Interpretable** — Latent dimensions have meaning

---

## Getting Started

**Quick start**:
```bash
# Read overview
cat docs/latent_diffusion/00_latent_diffusion_overview.md

# Understand architecture
cat docs/latent_diffusion/01_latent_diffusion_foundations.md

# See training examples
cat docs/latent_diffusion/02_latent_diffusion_training.md
```

**For biology applications**:
```bash
# Jump to applications
cat docs/latent_diffusion/03_latent_diffusion_applications.md

# Deep dive into implementation
cat docs/latent_diffusion/04_latent_diffusion_combio.md
```

**For implementation**:
```bash
# Check source code (when available)
ls src/genailab/latent_diffusion/

# Run notebooks (when available)
ls notebooks/latent_diffusion/
```

---

## Status

**Documentation**: 🚧 In Progress
- [x] Overview
- [ ] Foundations
- [ ] Training
- [ ] Applications
- [ ] Combio Implementation

**Implementation**: 🔲 Planned
- [ ] VAE modules
- [ ] Latent diffusion modules
- [ ] Training infrastructure
- [ ] Biology applications

**Notebooks**: 🔲 Planned
- [ ] VAE training
- [ ] Latent diffusion training
- [ ] Single-cell generation
- [ ] Perturbation prediction
- [ ] Multi-omics translation
