# Perturbation Prediction: A Complete Guide

**Status**: 🎯 Active Development

**Goal**: Predict single-cell responses to genetic and chemical perturbations using generative AI (VAE, JEPA, diffusion)

**Target Dataset**: Norman et al. 2019 Perturb-seq (K562 cells, CRISPR knockouts)

---

## Executive Summary

This application demonstrates the genai-lab approach to perturbation modeling:

1. **Start with strong baselines**: VAE with count-aware decoders (NB/ZINB)
2. **Add self-supervised prediction**: JEPA for latent space dynamics
3. **Enable uncertainty quantification**: Diffusion in latent space

**Why This Application**:
- Central problem in computational biology (drug discovery, target identification)
- Clear benchmarks (scGen, CPA, GEARS, scPPDM)
- Natural integration of VAE → JEPA → Diffusion architectures
- Demonstrates practical value of generative models over deterministic predictors

---

## Background: Why Generative AI for Perturbations?

### The Perturbation Modeling Problem

**Goal**: Given a baseline cell state and a perturbation (gene knockout, drug treatment), predict the resulting cellular response.

**Challenges**:
- **Biological heterogeneity**: Same perturbation produces diverse responses across cells
- **Compositional generalization**: Predict unseen perturbation combinations
- **Out-of-distribution robustness**: Transfer to new cell types, doses, timepoints
- **Uncertainty quantification**: Not just "what happens" but "what could happen"

### Current Approaches and Limitations

| Method | Type | Strengths | Limitations |
|--------|------|-----------|-------------|
| **scGen** (2019) | VAE | First conditional perturbation model | Point estimates only, limited compositionality |
| **CPA** (2021) | VAE | Compositional perturbation algebra | Still deterministic predictions |
| **GEARS** (2023) | GNN + VAE | Gene regulatory network structure | Requires known interactions |
| **scPPDM** (2023) | Diffusion | Uncertainty quantification | Full diffusion on high-dim data |

### Generative AI Value Proposition

**What we add**:
1. **Uncertainty quantification**: Sample multiple plausible responses
2. **Cell-level heterogeneity**: Model population distributions, not just means
3. **Compositional generalization**: Latent arithmetic for unseen combinations
4. **Counterfactual reasoning**: "What if we perturbed X instead of Y?"

**How we differ from GEM-1** (Synthesize Bio):
- GEM-1 learns $\mathbb{E}[x \mid \text{condition}]$ (conditional mean)
- We learn $p(x \mid \text{condition})$ (full distribution)
- GEM-1 is excellent for interpolation; we target extrapolation + uncertainty

---

## Architecture Overview

### Three-Stage Approach

```
Stage 1: VAE Baseline (CVAE_NB)
    ├── Input: x (gene expression), p (perturbation), c (covariates)
    ├── Encoder: q(z | x, p, c)
    ├── Decoder: p(x | z, p, c) with Negative Binomial likelihood
    └── Loss: ELBO with KL regularization

Stage 2: JEPA Predictor
    ├── Context encoder: E_ctx(x_control, c) → h
    ├── Target encoder: E_tgt(x_perturbed) → z (EMA updated)
    ├── Perturbation encoder: e(p)
    ├── Predictor: F(h, e(p)) → ẑ
    └── Loss: ||ẑ - sg(z)||² + VICReg regularization

Stage 3: Diffusion Wrapper
    ├── Latent diffusion in JEPA latent space
    ├── Conditional on perturbation + baseline context
    ├── Sampling: Generate diverse cellular responses
    └── Uncertainty: Variance across samples
```

### Why This Progression?

1. **VAE first**: Establishes strong count-aware decoder, validates on held-out cells
2. **JEPA second**: Learns robust latent space via self-supervised prediction
3. **Diffusion last**: Adds stochasticity in compact latent space, not raw expression

**Key insight**: Don't diffuse in gene expression space (20K+ dimensions, sparse, zero-inflated). Diffuse in JEPA latent space (~64-256 dimensions, continuous, well-structured).

---

## Dataset Strategy

### Overview: Staged Expansion Approach

This application follows a **staged dataset strategy**: start with a canonical benchmark for validation, then scale to larger datasets, then expand to cutting-edge multi-dataset collections.

**Philosophy**: Flagship ≠ Production. For the flagship application, we prioritize **benchmark comparability** and **iteration speed** over scale. Once validated, we expand to larger and more recent datasets.

### Phase 1: Norman et al. 2019 (Current Focus) 🎯

**Timeline**: Weeks 1-6 (Flagship application)

**Dataset Details**:
- **Size**: ~100,000 cells (manageable for rapid iteration)
- **Perturbations**: ~300 perturbations including combinatorial (double knockouts)
- **Cell Type**: K562 (myelogenous leukemia cell line)
- **Modality**: CRISPRa (activation) with Perturb-seq readout
- **Source**: [scPerturb collection](https://scperturb.org), original paper: Norman et al. 2019 *Science*

**Why This Dataset First**:

1. **Benchmark comparability**:
   - scGen, CPA, GEARS, and scPPDM all report results on Norman 2019
   - Direct comparison establishes credibility of our 3-stage approach (VAE → JEPA → Diffusion)
   - Industry standard for perturbation prediction benchmarks

2. **Combinatorial perturbations**:
   - Includes double knockouts (not just single perturbations)
   - Tests compositional generalization—a key JEPA value proposition
   - Scientifically interesting: Do perturbation effects compose linearly?

3. **Manageable size for iteration**:
   - 100K cells vs. Replogle's 2.5M → 25× faster training cycles
   - Enables rapid experimentation on local M1 Mac
   - 6-week timeline demands iteration speed over scale

4. **Well-characterized**:
   - Extensively studied in perturbation modeling literature
   - Known biological ground truth (genetic interactions)
   - Existing preprocessing pipelines documented in `docs/datasets/perturbation/scperturb.md`

**Goals for This Phase**:
- ✅ Match or exceed scGen baseline performance
- ✅ Demonstrate JEPA improves on VAE for held-out perturbations
- ✅ Show uncertainty calibration with diffusion wrapper
- ✅ Validate compositional generalization (double KOs from singles)

### Phase 2: Replogle et al. 2022 (Scale Demonstration)

**Timeline**: After flagship success (Weeks 7-10+)

**Dataset Details**:
- **Size**: ~2.5 million cells
- **Perturbations**: >5,000 genes (genome-scale)
- **Cell Type**: K562 + RPE1 (two cell lines)
- **Modality**: CRISPRi (interference)
- **Source**: [scPerturb collection](https://scperturb.org), original paper: Replogle et al. 2022 *Cell*

**Why This Dataset Second**:

1. **Scale validation**:
   - Largest single-cell perturbation dataset available
   - Tests whether JEPA + latent diffusion scales to production use
   - Addresses "but does it work on real data?" concerns

2. **Comprehensive perturbation coverage**:
   - >5,000 perturbations = broad biological coverage
   - Enables meta-learning across perturbation types
   - Tests zero-shot prediction (predict gene X from learning genes A-W)

3. **Transfer learning experiments**:
   - Pretrain on Replogle 2022, fine-tune on Norman 2019 → test generalization
   - Train on K562, test on RPE1 → test cell-type transfer
   - Cross-dataset validation

4. **Production readiness**:
   - Demonstrates system can handle production-scale data
   - Establishes computational requirements for real applications

**Goals for This Phase**:
- ✅ Demonstrate scalability (2.5M cells, 5K+ perturbations)
- ✅ Zero-shot perturbation prediction
- ✅ Transfer learning across cell types
- ✅ Production deployment considerations

### Phase 3: Newer and Multi-Dataset Collections (Future Research)

**Timeline**: After Phase 2 (3+ months)

#### Option A: Compressed Perturb-seq (2024) 📊

- **Innovation**: Statistical compression technique for increased power at lower cost
- **Status**: Recent publication (2024), cutting-edge methodology
- **Why interesting**: Could write comparative analysis ("JEPA on compressed vs. full Perturb-seq")
- **Challenge**: Newer, less benchmarked → harder to compare with published methods

#### Option B: PerturBase (Multi-Dataset Collection) 🗄️

- **Coverage**: 122 scPerturb datasets harmonized
- **Perturbation types**: Genetic (CRISPR), chemical (drugs), overexpression
- **Modalities**: Transcriptome (RNA-seq), ATAC-seq, protein (CITE-seq)
- **Why interesting**: 
  - Meta-learning across datasets
  - Multi-modal integration (ATAC + RNA, protein + RNA)
  - Cross-study generalization
- **Challenge**: Requires sophisticated data harmonization and multi-task learning infrastructure

#### Option C: Specific High-Value Datasets 🎯

| Dataset | Focus | Why Interesting |
|---------|-------|----------------|
| **Gasperini 2019** | Enhancer perturbations | Non-coding regulatory elements (different biology) |
| **Frangieh 2021** | Cancer immunotherapy | Clinical relevance, tumor microenvironment |
| **Papalexi 2021** (ECCITE-seq) | Protein + RNA | Multi-modal (surface protein markers + transcriptome) |
| **Adamson 2016** | UPR pathway | Focused pathway, interpretable biology |

### Dataset Access & Preprocessing

**Recommended source**: [scPerturb collection](https://scperturb.org)
- Harmonized format (AnnData h5ad)
- Consistent metadata schema
- Pre-processed and quality-controlled

**Alternative**: Individual lab repositories or GEO
- More work (format inconsistencies)
- Potentially newer/unpublished datasets

**Preprocessing pipeline** (already documented):
- See `docs/datasets/perturbation/scperturb.md` for detailed preprocessing
- Quality control, normalization, highly variable genes, train/val/test splits

### Key Design Decisions

#### Why Start Small → Scale Up?

**Iteration Speed Over Initial Scale**:
- Complex architecture (VAE → JEPA → Diffusion)
- Many hyperparameters to tune
- Need rapid experimentation
- Smaller dataset = faster training = more experiments in 6 weeks

**Benchmark Comparability**:
- Publishing results? Need comparison with scGen, CPA, GEARS
- Norman 2019 is the common benchmark
- Establish credibility before claiming novel scaling

**Scientific vs. Engineering Focus**:
- Phase 1: Does the method work? (Science)
- Phase 2: Does it scale? (Engineering)
- Phase 3: Does it generalize? (Robustness)

#### When to Use Each Dataset?

| Question | Dataset |
|----------|---------|
| "Does JEPA improve on VAE?" | Norman 2019 |
| "Can we match scGen/CPA?" | Norman 2019 |
| "Does it scale to genome-wide?" | Replogle 2022 |
| "Does it transfer across cell types?" | Replogle 2022 (K562 → RPE1) |
| "Does it work on compressed data?" | Compressed Perturb-seq 2024 |
| "Can we meta-learn across studies?" | PerturBase |
| "Can we handle multi-modal data?" | ECCITE-seq datasets |

### Dataset Expansion Roadmap

```
Current (Week 1-6):
└── Norman 2019
    ├── Validate VAE baseline vs. scGen
    ├── Implement JEPA for perturbation prediction
    ├── Add diffusion for uncertainty
    └── Benchmark compositional generalization

Next (Week 7-10):
└── Replogle 2022
    ├── Scale to 2.5M cells
    ├── Test zero-shot perturbation prediction
    ├── Transfer learning experiments
    └── Production deployment considerations

Future (3+ months):
├── Compressed Perturb-seq 2024 (comparative analysis)
├── PerturBase multi-dataset (meta-learning)
├── ECCITE-seq datasets (multi-modal)
└── Cross-study generalization benchmarks
```

### Notes for Future Sessions

**Dataset options documented** (for later reference):

1. **Replogle 2022**: Next after Norman 2019, scale demonstration
2. **Compressed Perturb-seq 2024**: Newer methodology, comparative analysis opportunity
3. **PerturBase**: 122 datasets, meta-learning potential
4. **Gasperini 2019**: Enhancer perturbations (regulatory elements)
5. **Frangieh 2021**: Cancer immunotherapy (clinical relevance)
6. **Papalexi 2021** (ECCITE-seq): Multi-modal (protein + RNA)
7. **Adamson 2016**: UPR pathway (focused, interpretable)

**Decision**: Start with Norman 2019 for **benchmark comparability** and **iteration speed**, expand to larger/newer datasets after validation.

---

## Implementation Roadmap

### Phase 1: Data + VAE Baseline (Week 1-2)

#### 1.1 Dataset Preparation

**Download Norman et al. 2019 dataset**:
```python
import scanpy as sc

# Download from CellxGene or original source
adata = sc.read_h5ad("norman2019_perturb_seq.h5ad")

# Key fields:
# - adata.X: Gene expression (cells × genes)
# - adata.obs['perturbation']: Perturbation label(s)
# - adata.obs['cell_type']: Cell type (K562)
# - adata.obs['guide_identity']: sgRNA identity
```

**Dataset statistics**:
- ~250,000 cells
- ~5,000 highly variable genes
- ~100 single-gene CRISPR knockouts
- Control cells available for counterfactual comparisons

**Preprocessing**:
1. Quality control (filter low-count cells/genes)
2. Normalization (library size normalization)
3. Log transformation for visualization (keep counts for NB decoder)
4. Train/val/test split (stratified by perturbation)

**Data loader**:
```python
class PerturbSeqDataset(torch.utils.data.Dataset):
    def __init__(self, adata, split='train'):
        self.X = adata.X  # Keep as counts
        self.perturbations = adata.obs['perturbation']
        self.covariates = adata.obs[['batch', 'cell_cycle', 'libsize']]
    
    def __getitem__(self, idx):
        return {
            'expression': self.X[idx],  # counts
            'perturbation': self.perturbations[idx],
            'covariates': self.covariates.iloc[idx]
        }
```

#### 1.2 CVAE_NB Baseline

**Model**: `genailab.model.vae.CVAE_NB` (already implemented)

**Configuration**:
```python
from genailab.model.vae import CVAE_NB
from genailab.model.encoders import ConditionEncoder

model = CVAE_NB(
    n_genes=5000,
    latent_dim=64,
    condition_encoder=ConditionEncoder(
        perturbation_dim=256,  # embedding dimension
        covariate_dim=32,
        combined_dim=256
    ),
    hidden_dims=[512, 256],
    use_batch_norm=True
)
```

**Training**:
```python
from genailab.objectives.losses import elbo_loss_nb

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for batch in train_loader:
        # Forward pass
        z_mean, z_logvar, x_recon_nb_params = model(
            batch['expression'],
            perturbation=batch['perturbation'],
            covariates=batch['covariates']
        )
        
        # ELBO with NB likelihood
        loss = elbo_loss_nb(
            x=batch['expression'],
            x_recon_params=x_recon_nb_params,
            z_mean=z_mean,
            z_logvar=z_logvar,
            kl_weight=kl_scheduler(epoch)  # Annealing
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Evaluation Metrics**:

1. **Reconstruction quality**:
   - MSE on held-out cells
   - NB log-likelihood

2. **Perturbation prediction**:
   - Train: Control cells → Predict perturbed
   - Compare predicted vs. observed perturbed cells
   - Metrics: MSE, Pearson correlation per gene

3. **DEG recovery**:
   - Identify top differentially expressed genes (DEGs) in real data
   - Check if predicted perturbations recover same DEGs
   - Metrics: Precision@K, Recall@K, AUROC

4. **Biological validation**:
   - Pathway enrichment of predicted DEGs
   - Known gene-gene interactions

**Expected baseline**: Should match or slightly improve on scGen performance

#### 1.3 Infrastructure Setup

**Experiment tracking**:
```python
import wandb

wandb.init(project='genai-lab-perturbseq', name='cvae_nb_baseline')
wandb.config.update({
    'model': 'CVAE_NB',
    'latent_dim': 64,
    'n_genes': 5000,
    'learning_rate': 1e-3
})
```

**Checkpointing**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'metrics': metrics
}
torch.save(checkpoint, f'checkpoints/cvae_nb_epoch{epoch}.pt')
```

**Documentation**:
- Create `examples/perturbation/README.md`
- Document preprocessing decisions
- Record hyperparameter choices

---

### Phase 2: JEPA Implementation (Week 3-4)

#### 2.1 Architecture Details

See [docs/JEPA/04_jepa_perturbseq.md](../JEPA/04_jepa_perturbseq.md) for complete JEPA architecture for Perturb-seq.

**Core components**:

```python
class PerturbSeqJEPA(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 256,
        perturbation_vocab_size: int = 100,
        perturbation_embed_dim: int = 128,
    ):
        # Context encoder (trainable)
        self.context_encoder = GeneExpressionEncoder(
            n_genes, latent_dim
        )
        
        # Target encoder (EMA updated)
        self.target_encoder = GeneExpressionEncoder(
            n_genes, latent_dim
        )
        
        # Perturbation encoder
        self.perturbation_encoder = nn.Embedding(
            perturbation_vocab_size, perturbation_embed_dim
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + perturbation_embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # EMA momentum
        self.ema_momentum = 0.996
    
    def forward(self, x_context, x_target, perturbation):
        # Context embedding
        h = self.context_encoder(x_context)
        
        # Target embedding (no gradients)
        with torch.no_grad():
            z = self.target_encoder(x_target)
        
        # Perturbation embedding
        p_emb = self.perturbation_encoder(perturbation)
        
        # Predict target from context + perturbation
        z_pred = self.predictor(torch.cat([h, p_emb], dim=-1))
        
        return z_pred, z
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder"""
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_tgt.data.mul_(self.ema_momentum)
            param_tgt.data.add_(
                (1 - self.ema_momentum) * param_ctx.data
            )
```

#### 2.2 Loss Functions

**Base JEPA loss**:
```python
def jepa_loss(z_pred, z_target):
    """MSE in latent space with stop-gradient on target"""
    return F.mse_loss(z_pred, z_target.detach())
```

**VICReg regularization** (prevent collapse):
```python
def vicreg_loss(z, variance_weight=1.0, covariance_weight=0.04):
    """
    Variance: Encourage non-zero variance across batch
    Covariance: Discourage correlated dimensions
    """
    batch_size, latent_dim = z.shape
    
    # Variance loss (hinge)
    std_z = torch.sqrt(z.var(dim=0) + 1e-4)
    variance_loss = torch.mean(F.relu(1 - std_z))
    
    # Covariance loss
    z_centered = z - z.mean(dim=0)
    cov_z = (z_centered.T @ z_centered) / (batch_size - 1)
    off_diagonal = cov_z - torch.diag(torch.diagonal(cov_z))
    covariance_loss = off_diagonal.pow(2).sum() / latent_dim
    
    return (
        variance_weight * variance_loss +
        covariance_weight * covariance_loss
    )
```

**Combined loss**:
```python
def total_loss(z_pred, z_target, vicreg_weight=0.1):
    loss_jepa = jepa_loss(z_pred, z_target)
    loss_vicreg = vicreg_loss(z_pred)
    return loss_jepa + vicreg_weight * loss_vicreg
```

#### 2.3 Training Strategy

**Two-phase training**:

**Phase 2a: Self-supervised pretraining** (optional but recommended):
- Ignore perturbation labels
- Use augmentations (gene masking, noise) as "perturbations"
- Learn robust cell state manifold

```python
# Augmentation for self-supervision
def augment_expression(x):
    # Random gene dropout
    mask = torch.bernoulli(torch.full_like(x, 0.8))
    x_aug = x * mask
    
    # Poisson noise
    x_aug = torch.poisson(x_aug + 1e-6)
    
    return x_aug

# Training loop
for batch in train_loader:
    x = batch['expression']
    x_aug1 = augment_expression(x)
    x_aug2 = augment_expression(x)
    
    z_pred, z_target = model(x_aug1, x_aug2, perturbation=None)
    loss = total_loss(z_pred, z_target)
```

**Phase 2b: Perturbation-conditioned training**:
- Add perturbation tokens
- Train to predict perturbed state from baseline + perturbation

```python
for batch in train_loader:
    # Get control and perturbed pairs
    x_control = batch['control_expression']
    x_perturbed = batch['perturbed_expression']
    perturbation = batch['perturbation']
    
    z_pred, z_target = model(
        x_control, x_perturbed, perturbation
    )
    
    loss = total_loss(z_pred, z_target)
    
    # Update context encoder + predictor
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # EMA update of target encoder
    model.update_target_encoder()
```

#### 2.4 Evaluation

**Latent space quality**:
1. Visualize with UMAP/t-SNE (color by perturbation)
2. Perturbation classification accuracy from latent embeddings
3. kNN purity (same perturbation neighbors)

**Prediction accuracy**:
1. Held-out perturbations (zero-shot prediction)
2. Compositional generalization (double knockouts from singles)
3. Transfer to unseen cell types (if available)

**Comparison with VAE baseline**:
- Does JEPA latent space better separate perturbations?
- Does prediction improve for held-out perturbations?

---

### Phase 3: Diffusion Wrapper (Week 5-6)

#### 3.1 Latent Diffusion Architecture

**Goal**: Add stochasticity to JEPA predictions for uncertainty quantification

**Approach**: Diffusion in JEPA latent space, not gene expression space

```python
from genailab.diffusion.sde import VPSDE
from genailab.diffusion.architectures import MLP

class LatentDiffusion(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        condition_dim: int = 256,
        hidden_dim: int = 512
    ):
        self.sde = VPSDE(schedule='cosine')
        
        # Score network in latent space
        self.score_network = MLP(
            input_dim=latent_dim + condition_dim + 1,  # +1 for time
            output_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            time_embedding_dim=128
        )
    
    def forward(self, z_t, t, condition):
        """Predict score at time t"""
        # Time embedding
        t_emb = self.get_time_embedding(t)
        
        # Concatenate inputs
        x = torch.cat([z_t, condition, t_emb], dim=-1)
        
        # Predict score
        score = self.score_network(x)
        return score
```

**Conditioning**: Use JEPA context embedding + perturbation embedding as condition

```python
# Get JEPA embeddings (frozen)
with torch.no_grad():
    h_context = jepa.context_encoder(x_control)
    p_emb = jepa.perturbation_encoder(perturbation)
    condition = torch.cat([h_context, p_emb], dim=-1)

# Train diffusion conditioned on these
z_t = diffusion.sde.forward_diffusion(z_target, t)
score_pred = diffusion(z_t, t, condition)
loss = score_matching_loss(score_pred, z_t, t)
```

#### 3.2 Sampling for Diversity

**Generate multiple plausible responses**:

```python
def sample_perturbation_responses(
    jepa_model,
    diffusion_model,
    x_control,
    perturbation,
    n_samples=100
):
    # Get condition
    with torch.no_grad():
        h_context = jepa_model.context_encoder(x_control)
        p_emb = jepa_model.perturbation_encoder(perturbation)
        condition = torch.cat([h_context, p_emb], dim=-1)
    
    # Sample from diffusion
    z_samples = diffusion_model.sample(
        n_samples=n_samples,
        condition=condition
    )
    
    # Optional: Decode to gene expression
    # x_samples = decoder(z_samples)
    
    return z_samples
```

**Uncertainty quantification**:
- Variance across samples → prediction uncertainty
- Entropy of predicted distribution → biological heterogeneity

#### 3.3 Comprehensive Evaluation

**Compare three approaches**:

| Model | Uncertainty | Held-out Perturbations | Compositionality |
|-------|-------------|------------------------|------------------|
| CVAE_NB | Posterior variance | ? | Limited |
| JEPA | None (deterministic) | Better | Good |
| JEPA + Diffusion | Explicit sampling | Best | Best |

**Benchmark against published methods**:

1. **scGen** (2019) - VAE baseline
2. **CPA** (2021) - Compositional perturbation autoencoder
3. **scPPDM** (2023) - Diffusion on full expression space

**Evaluation metrics**:

```python
metrics = {
    'reconstruction': {
        'mse': mean_squared_error(x_true, x_pred),
        'pearson': pearson_correlation(x_true, x_pred)
    },
    'deg_recovery': {
        'precision@100': precision_at_k(deg_true, deg_pred, k=100),
        'recall@100': recall_at_k(deg_true, deg_pred, k=100),
        'auroc': auroc(deg_true, deg_pred)
    },
    'pathway_consistency': {
        'enrichment_overlap': pathway_overlap(deg_pred, known_pathways),
        'known_interactions': interaction_recovery(deg_pred, interaction_db)
    },
    'compositional': {
        'double_ko_accuracy': evaluate_double_knockouts(model, test_pairs)
    },
    'uncertainty': {
        'calibration': calibration_curve(predicted_variance, true_variance),
        'diversity': sample_diversity(generated_samples)
    }
}
```

---

## Expected Outcomes

### Quantitative Results

**Baseline expectations**:
- CVAE_NB should match scGen (~0.8 Pearson correlation on held-out cells)
- JEPA should improve held-out perturbation prediction (+5-10% accuracy)
- Diffusion wrapper should provide calibrated uncertainty (correlation with true variance > 0.6)

**Success criteria**:
- ✅ Outperform scGen on held-out perturbations
- ✅ Demonstrate compositional generalization (double KOs from singles)
- ✅ Calibrated uncertainty (predicted variance correlates with true variance)
- ✅ Biologically validated (DEG recovery, pathway consistency)

### Qualitative Insights

**Scientific contributions**:
1. Demonstrate value of JEPA for biological prediction (vs. pure reconstruction)
2. Show when uncertainty quantification matters (rare perturbations, extreme responses)
3. Establish best practices for count data in generative models (NB decoders, latent diffusion)

**Technical contributions**:
1. End-to-end pipeline for Perturb-seq modeling
2. Reusable components (JEPA encoder, latent diffusion, evaluation metrics)
3. Benchmarking framework for perturbation prediction methods

---

## Beyond the Flagship: Extensions

Once the core application is validated, natural extensions include:

### Multi-Dataset Validation
- Test on other Perturb-seq datasets (different cell types, perturbation modalities)
- Transfer learning: Train on K562, test on primary cells
- Meta-learning across perturbations

### Compositional Generalization
- Double and triple knockouts from single knockouts
- Drug combination predictions
- Dose-response curves

### Temporal Dynamics
- Time-course Perturb-seq (predict response trajectories)
- Early vs. late effects of perturbations

### Multimodal Integration
- ATAC-seq + RNA-seq (chromatin + expression)
- Protein measurements (CITE-seq)
- Morphological features

### Causal Validation
- Integration with `causal-bio-lab` (sibling project)
- Counterfactual validation using structural causal models
- Causal discovery from perturbation data

---

## References

### Key Papers

**Perturbation Modeling:**
- Lotfollahi et al. (2019). [scGen predicts single-cell perturbation responses](https://www.nature.com/articles/s41592-019-0494-8)
- Lotfollahi et al. (2021). [Compositional perturbation autoencoder (CPA)](https://www.embopress.org/doi/full/10.15252/msb.202211517)
- Roohani et al. (2023). [GEARS: Predicting combinatorial perturbations](https://www.nature.com/articles/s41587-023-01905-6)
- scPPDM (2023). Single-cell Perturbation Prediction via Diffusion Models

**JEPA & Self-Supervised Learning:**
- Assran et al. (2023). [I-JEPA: Self-supervised learning from images](https://arxiv.org/abs/2301.08243)
- Meta AI (2025). [V-JEPA 2: Video models for understanding and prediction](https://arxiv.org/abs/2506.09985)

**Architectural Components:**
- See [docs/JEPA/04_jepa_perturbseq.md](../JEPA/04_jepa_perturbseq.md) - Complete JEPA for Perturb-seq
- See [docs/latent_diffusion/04_latent_diffusion_combio.md](../latent_diffusion/04_latent_diffusion_combio.md) - Latent diffusion for biology

### Datasets

**Primary Dataset:**
- Norman et al. (2019). [Exploring genetic interaction manifolds constructed from rich single-cell phenotypes](https://www.science.org/doi/10.1126/science.aax4438)
  - K562 cells, CRISPR perturbations, ~250K cells

**Additional Benchmarks:**
- Replogle et al. (2022). [Mapping genetic effects with Perturb-seq](https://www.cell.com/cell/fulltext/S0092-8674(22)00597-6)
- Dixit et al. (2016). [Perturb-Seq: Dissecting molecular circuits](https://www.cell.com/cell/fulltext/S0092-8674(16)31610-5)

---

## Implementation Checklist

### Phase 1: Data + VAE Baseline
- [ ] Download Norman et al. 2019 dataset
- [ ] Implement data loaders with QC
- [ ] Train CVAE_NB with perturbation conditioning
- [ ] Establish evaluation metrics (DEG recovery, prediction accuracy)
- [ ] Document preprocessing and hyperparameter choices

### Phase 2: JEPA Implementation
- [ ] Implement JEPA architecture (context/target encoders, predictor)
- [ ] VICReg regularization for collapse prevention
- [ ] Two-phase training (self-supervised → perturbation-conditioned)
- [ ] Evaluate latent space quality and prediction accuracy
- [ ] Compare with CVAE baseline

### Phase 3: Diffusion Wrapper
- [ ] Implement latent diffusion model
- [ ] Train diffusion conditioned on JEPA embeddings
- [ ] Sampling for diverse cellular responses
- [ ] Comprehensive benchmarking (scGen, CPA, scPPDM)
- [ ] Biological validation (pathways, known interactions)

### Documentation & Release
- [ ] Complete example notebook: `examples/perturbation/01_perturbseq_jepa_diffusion.ipynb`
- [ ] Training scripts with configs
- [ ] Benchmark results table
- [ ] Update project README with results
- [ ] Blog post / technical report

---

## Contact & Collaboration

This is an active research application. For questions, suggestions, or collaboration:

- Open an issue in the GitHub repository
- Refer to [docs/ROADMAP.md](../ROADMAP.md) for current progress
- See related work in `causal-bio-lab` (sibling project) for causal validation methods

---

**Last Updated**: 2026-01-31

**Status**: Active development - Week 1-2 (Data + VAE Baseline)
