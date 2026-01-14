# Leveraging Foundation Models for Computational Biology

A practical guide to adapting pretrained foundation models (DiT-like backbones, Geneformer, scGPT) for clinically useful computational biology tasks.

**Target applications**: Gene expression synthesis and perturbation response prediction

**Key insight**: Don't fine-tune everything—fine-tune the steering wheel.

---

## Overview

This guide covers:

1. **Clinical targets** — What "success" looks like in practice
2. **Latent-space strategy** — Why DiT works better on learned tokens
3. **Tokenization options** — Four approaches for gene expression
4. **Design patterns** — Six reusable strategies for model adaptation
5. **Task-specific recipes** — How to apply patterns to your use cases
6. **Implementation plan** — Modular architecture for experimentation
7. **Session roadmap** — Step-by-step learning path

---

## 1. Clinical Targets: Defining Success

Clinical utility means producing **actionable distributions** under interventions, not just generating pretty samples.

### Gene Expression Synthesis

**Goal**: Generate realistic gene expression profiles conditioned on biological context.

**Requirements**:
- **Realistic marginals** — Proper count distributions (NB/ZINB), zero-inflation, library size effects
- **Realistic conditionals** — Accurate tissue, disease subtype, and covariate dependencies
- **Calibrated uncertainty** — Reliable confidence estimates for downstream decisions

**Use cases**:
- Data augmentation for rare cell types
- Synthetic controls for experiments
- Batch effect correction
- Counterfactual cell state generation

### Perturbation Response Prediction

**Goal**: Predict cellular response to genetic or chemical perturbations.

**Requirements**:
- **Δ-expression prediction** — Counterfactual shift under perturbation, not just reconstruction
- **Generalization** — Accurate predictions for unseen perturbations or combinations
- **OOD-aware uncertainty** — Higher uncertainty for novel cell states or perturbations

**Use cases**:
- Virtual screening (predict without experiments)
- Combination therapy prediction
- Mechanism discovery
- Drug response modeling

**Note**: Models like scPPDM fall into this category—predicting response distributions conditioned on perturbation + context, not merely generating cells.

---

## 2. Latent-Space Strategy: Why Not Raw Counts?

### The Problem with Raw Gene Expression

DiT backbones work best with **sequences of continuous tokens** at stable scale. Raw gene expression has problematic properties:

| Property | Issue | Impact |
|----------|-------|--------|
| **Count noise** | NB/ZINB distribution | Non-Gaussian, heavy-tailed |
| **Library size** | Technical variation | Scaling artifacts |
| **Sparsity** | Many zeros | Gradient issues |
| **Dimensionality** | ~20K genes | Computational cost |

### The Solution: Latent Diffusion

**Standard pattern**:

```
Raw Counts → Encoder → Latent Tokens → DiT/Transformer → Latent → Decoder → Parameters
  (20K)                   (64×256)                        (64×256)              (20K)
```

**Architecture components**:

```python
# Encoder: Compress to latent space
encoder: gene_expression (20K) → latent_tokens (64, 256)

# Diffusion: DiT operates on latent tokens
diffusion: latent_tokens → denoised_latent_tokens

# Decoder: Reconstruct with biological distributions
decoder: latent_tokens → NB/ZINB parameters (mean, dispersion, dropout)
```

### Why This Works

**1. Learned semantic tokenization**
- Encoder discovers meaningful biological patterns
- Avoids arbitrary patch-based tokenization
- Tokens represent gene modules, pathways, or cell states

**2. Stable diffusion dynamics**
- Continuous latent space (no count artifacts)
- Normalized scale (better gradient flow)
- Lower dimensionality (faster training)

**3. Biological realism**
- NB/ZINB decoder handles count distributions
- Library size normalization in decoder
- Zero-inflation modeling

**4. Modular design**
- Encoder/decoder can be pretrained separately
- DiT backbone can be frozen or adapted
- Easy to swap components

---

## 3. Tokenization Options for Gene Expression

**Key question**: How do we factor gene expression so attention has something meaningful to attend over?

### Option 1: Latent Tokens (Recommended Default)

**Concept**: Learn a compressed, structured representation.

```python
# Architecture
encoder: R^20000 → R^(m×d)  # m=64 tokens, d=256 dimensions
DiT: R^(m×d) → R^(m×d)      # Attention over tokens
decoder: R^(m×d) → R^20000  # Back to gene space
```

**Advantages**:
- **Data-adaptive** — Model learns optimal tokenization
- **Compute-friendly** — 64 tokens << 20K genes (attention is O(m²))
- **LoRA-compatible** — Small adapters can steer large backbone
- **Flexible conditioning** — Easy to inject perturbation/cell type info

**Implementation**:

```python
class LatentTokenEncoder(nn.Module):
    def __init__(self, num_genes=20000, num_tokens=64, token_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, num_tokens * token_dim)
        )
        self.num_tokens = num_tokens
        self.token_dim = token_dim
    
    def forward(self, x):
        # x: (batch, num_genes)
        z = self.encoder(x)  # (batch, num_tokens * token_dim)
        z = z.view(-1, self.num_tokens, self.token_dim)  # (batch, num_tokens, token_dim)
        return z
```

**When to use**: Default choice for most applications.

### Option 2: Pathway/Module Tokens (Biologically Anchored)

**Concept**: Use biological knowledge to define tokens as gene pathways or modules.

```python
# Each token represents a biological pathway
token_1 → "Cell cycle genes"
token_2 → "Immune response genes"
token_3 → "Metabolic genes"
...
token_50 → "Apoptosis genes"
```

**Advantages**:
- **Interpretability** — Pathway-level explanations are clinically legible
- **Lower dimension** — ~50-500 pathways vs 20K genes
- **Transfer learning** — Pathways consistent across datasets
- **Inductive bias** — Encodes known biology, faster convergence

**Pathway databases**:

| Database | # Pathways | Coverage | Best For |
|----------|------------|----------|----------|
| MSigDB Hallmark | 50 | Broad processes | High-level analysis |
| MSigDB C2 (KEGG) | 186 | Metabolic/signaling | Mechanistic studies |
| Reactome | 2,500+ | Detailed processes | Fine-grained analysis |
| GO Biological Process | 10,000+ | Comprehensive | Full coverage |

**When to use**: Clinical applications requiring interpretability.

### Option 3: Graph-Structured Tokens (GRN-Aware)

**Concept**: Use gene regulatory networks to structure attention.

```python
# Sparse attention based on GRN edges
# Only genes with regulatory relationships attend to each other
attention_mask = GRN_adjacency_matrix  # Sparse (num_genes, num_genes)
```

**Advantages**:
- **Mechanistic** — Respects known regulatory relationships
- **Perturbation-aware** — Better inductive bias for interventions
- **Efficient** — Sparse attention O(num_edges) vs O(n²)
- **Interpretable** — Can trace predictions through regulatory paths

**When to use**: Perturbation modeling, causal inference.

### Option 4: Rank-Based Sequences (Geneformer Style)

**Concept**: Order genes by expression level and treat as sequence.

```python
# Rank genes by expression
sorted_genes = argsort(expression, descending=True)
# Treat as sequence for transformer
sequence = [gene_1, gene_2, ..., gene_k]  # Top-k genes
```

**Advantages**:
- **Empirically validated** — Works in Geneformer
- **Sequence-native** — Natural for transformers

**Disadvantages**:
- **Ordering artifacts** — Ties get arbitrary order
- **Scaling issues** — 20K sequence length is expensive
- **Truncation loss** — Top-k loses information
- **Not biologically motivated** — Ranking is artificial

**When to use**: Large-scale pretraining with massive data.

### Comparison Table

| Approach | Tokens | Compute | Interpretability | Biology | Transfer | Best For |
|----------|--------|---------|------------------|---------|----------|----------|
| **Latent tokens** | 32-128 | Low | Moderate | Learned | Good | Default choice |
| **Pathway tokens** | 50-500 | Low | High | Strong | Excellent | Clinical applications |
| **GRN-aware** | 20K (sparse) | Moderate | High | Strong | Moderate | Perturbation modeling |
| **Rank-based** | 2K-20K | High | Low | Weak | Poor | Large-scale pretraining |

---

## 4. Foundation Model Design Patterns

Six reusable strategies for adapting pretrained models without full fine-tuning.

### Pattern A: Frozen Backbone + Linear Probe

**Strategy**: Freeze pretrained encoder/backbone, train only a small task-specific head.

```python
# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False

# Train only head
head = nn.Linear(hidden_dim, num_classes)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
```

**When to use**:
- **Sanity check** — Test if representations are already good
- **Low data** — Few samples available
- **Fast iteration** — Quick baseline

**Pros**: Fast, stable, no catastrophic forgetting
**Cons**: Limited expressiveness

### Pattern B: Adapters

**Strategy**: Insert small bottleneck modules into frozen model.

```python
class Adapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))

# Insert after each transformer block
for block in transformer.blocks:
    block.adapter = Adapter(hidden_dim)
```

**When to use**:
- **Stable training** — Less prone to overfitting
- **Multiple tasks** — Swap adapters per task
- **Limited compute** — Fewer parameters than full fine-tuning

**Pros**: Cheap, stable, modular
**Cons**: Slightly less expressive than full fine-tuning

### Pattern C: LoRA (Low-Rank Adaptation)

**Strategy**: Add low-rank matrices to attention projections.

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)  # Frozen
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = 1.0 / rank
    
    def forward(self, x):
        # Original + low-rank update
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

# Apply to Q, K, V projections
attention.query = LoRALinear(hidden_dim, hidden_dim, rank=8)
attention.key = LoRALinear(hidden_dim, hidden_dim, rank=8)
attention.value = LoRALinear(hidden_dim, hidden_dim, rank=8)
```

**When to use**:
- **Best efficiency** — Highest utility per parameter
- **Multiple personas** — Easy to swap per-dataset/task
- **Memory constrained** — Minimal memory overhead

**Pros**: Extremely parameter-efficient, composable
**Cons**: Requires careful rank selection

**Typical ranks**: 4-16 for most applications

### Pattern D: Partial Unfreezing

**Strategy**: Freeze most layers, unfreeze last few blocks + layer norms.

```python
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last K blocks
K = 3
for block in model.blocks[-K:]:
    for param in block.parameters():
        param.requires_grad = True

# Always unfreeze layer norms
for module in model.modules():
    if isinstance(module, nn.LayerNorm):
        for param in module.parameters():
            param.requires_grad = True
```

**When to use**:
- **More expressiveness** — Beyond LoRA/adapters
- **Moderate data** — Enough to avoid overfitting
- **Task-specific features** — Need to adapt high-level representations

**Pros**: More expressive than LoRA
**Cons**: More parameters, risk of overfitting

### Pattern E: Conditional Control Modules

**Strategy**: Keep backbone fixed, steer via conditional pathways.

**FiLM (Feature-wise Linear Modulation)**:

```python
class FiLM(nn.Module):
    def __init__(self, condition_dim, hidden_dim):
        super().__init__()
        self.scale_net = nn.Linear(condition_dim, hidden_dim)
        self.shift_net = nn.Linear(condition_dim, hidden_dim)
    
    def forward(self, x, condition):
        scale = self.scale_net(condition)
        shift = self.shift_net(condition)
        return x * scale + shift

# Apply after each block
for block in transformer.blocks:
    block.film = FiLM(condition_dim, hidden_dim)
```

**Cross-attention conditioning**:

```python
# Condition tokens attend into backbone tokens
condition_tokens = condition_encoder(perturbation)  # (batch, num_cond, dim)
backbone_tokens = backbone(gene_tokens)  # (batch, num_tokens, dim)

# Cross-attention: condition → backbone
output = cross_attention(
    query=backbone_tokens,
    key=condition_tokens,
    value=condition_tokens
)
```

**When to use**:
- **Perturbation modeling** — Perturbation as control signal
- **Multi-modal conditioning** — Cell type + tissue + batch
- **Classifier-free guidance** — Sample-time steering

**Pros**: Flexible, interpretable, no backbone modification
**Cons**: Requires careful conditioning design

### Pattern F: Distillation

**Strategy**: Use large model as teacher, train small model for deployment.

```python
# Teacher (large, frozen)
teacher_output = teacher_model(x)

# Student (small, trainable)
student_output = student_model(x)

# Distillation loss
loss = KL_divergence(student_output, teacher_output)
```

**When to use**:
- **Deployment constraints** — Clinical pipelines need fast inference
- **Edge devices** — Limited compute
- **Cost reduction** — Cheaper API calls

**Pros**: Fast inference, smaller models
**Cons**: Requires teacher model, some performance loss

---

## 5. Task-Specific Recipes

### Gene Expression Synthesis

**Best approach**: Latent Diffusion + NB/ZINB decoder

```python
# Architecture
encoder: gene_expression → latent_tokens
diffusion: latent_tokens (+ condition) → denoised_latent
decoder: latent_tokens → NB/ZINB parameters

# Foundation leverage
backbone = pretrained_DiT()  # Frozen or LoRA
conditioning = FiLM(condition_dim, hidden_dim)  # Trainable

# Minimal-data trick
# Fine-tune only conditioning modules for new cohorts/diseases
```

**Training strategy**:
1. Pretrain VAE (encoder + decoder) on large dataset
2. Freeze VAE, train diffusion on latent codes
3. Add LoRA/adapters for new conditions

### Perturbation Response

**Two formulations**:

**Option 1: Direct prediction**

```python
# Input: baseline state + perturbation
# Output: post-perturbation distribution

model(baseline_expression, perturbation_embedding) → perturbed_expression
```

**Option 2: Delta prediction (recommended)**

```python
# Predict change in latent space
delta_z = model(z_baseline, perturbation_embedding)
z_perturbed = z_baseline + delta_z
x_perturbed = decoder(z_perturbed)
```

**Why delta is better**:
- More stable training
- Better generalization
- Easier to interpret
- Handles small perturbations better

**Foundation leverage**:

```python
# Freeze large backbone
backbone.requires_grad = False

# Train only:
perturbation_encoder = nn.Sequential(...)  # Trainable
lora_modules = [...]  # Trainable
delta_head = nn.Linear(...)  # Trainable
```

**This is the "maximum utility with minimal data" sweet spot.**

---

## 6. Modular Implementation Architecture

Package design patterns as composable components for easy experimentation.

### Proposed Structure

```
genailab/foundation/
├── backbones/
│   ├── dit.py                    # DiT-like transformer wrapper
│   ├── gene_transformer.py       # Geneformer/scGPT wrappers
│   └── unet.py                   # U-Net backbone
├── tuning/
│   ├── lora.py                   # LoRA implementation
│   ├── adapters.py               # Adapter modules
│   └── freeze.py                 # Freezing policies
├── conditioning/
│   ├── film.py                   # FiLM conditioning
│   ├── cross_attention.py        # Cross-attention modules
│   └── cfg.py                    # Classifier-free guidance
├── objectives/
│   ├── distillation.py           # Knowledge distillation
│   └── uncertainty.py            # Calibration metrics
└── recipes/
    ├── latent_diffusion_nb.py    # End-to-end: encoder→diffuse→decoder
    └── perturb_delta_latent.py   # Perturbation delta prediction
```

### Example Usage

```python
from genailab.foundation.backbones import DiT
from genailab.foundation.tuning import LoRA
from genailab.foundation.conditioning import FiLM

# Load pretrained backbone
backbone = DiT.from_pretrained("dit-base")

# Add LoRA
backbone = LoRA.wrap(backbone, rank=8, target_modules=["attention"])

# Add conditioning
conditioning = FiLM(condition_dim=128, hidden_dim=512)

# Compose
model = LatentDiffusionModel(
    backbone=backbone,
    conditioning=conditioning,
    encoder=encoder,
    decoder=decoder
)
```

**Benefits**:
- **Modular** — Swap components easily
- **Reusable** — Share across projects
- **Testable** — Unit test each component
- **Ablatable** — Compare strategies systematically

---

## 7. Learning Roadmap: Session-by-Session

Break implementation into manageable sessions that compound.

### Session 1: Architecture & Design Patterns ✓

**Goal**: Understand foundation model adaptation strategies.

**Topics**:
- Clinical targets and success criteria
- Latent-space strategy
- Tokenization options
- Six design patterns

**Deliverable**: This document

### Session 2: Reference Stack Implementation

**Goal**: Build one complete end-to-end system.

**Tasks**:
- Implement latent diffusion for expression
- Add NB/ZINB decoder
- Create conditioning API (perturbation, tissue, batch, covariates)

**Deliverable**: Working latent diffusion model

### Session 3: Tuning Modules

**Goal**: Package adaptation strategies as reusable modules.

**Tasks**:
- Implement LoRA, adapters, freeze policies
- Create "one-line switch" configs
- Benchmark strategies on same task

**Deliverable**: `genailab.foundation.tuning` package

### Session 4: Perturbation Response Recipe

**Goal**: Build delta-in-latent perturbation model.

**Tasks**:
- Implement delta predictor
- Add perturbation encoder
- Evaluate: directional accuracy, pathway consistency, calibration

**Deliverable**: Perturbation prediction pipeline

### Session 5: Clinical Constraints

**Goal**: Handle real-world deployment challenges.

**Tasks**:
- Batch effect / domain shift handling
- Uncertainty calibration + OOD detection
- Counterfactual validity checks

**Deliverable**: Production-ready system

---

## Key Takeaways

### Core Principles

1. **Don't fine-tune the world—fine-tune the steering wheel**
   - Keep large backbones frozen
   - Train small conditioning/adaptation modules
   - Maximize utility per parameter

2. **Latent space is your friend**
   - Learn semantic tokenization
   - Avoid raw count artifacts
   - Enable stable diffusion dynamics

3. **Modularity enables experimentation**
   - Package patterns as composable components
   - Easy to swap and compare strategies
   - Reusable across projects

4. **Clinical utility requires more than generation**
   - Calibrated uncertainty
   - Interpretable predictions
   - Actionable distributions

### Decision Framework

**Choose latent tokens** when:
- You want data-adaptive tokenization
- Compute efficiency matters
- You need flexibility

**Choose pathway tokens** when:
- Interpretability is critical
- Clinical legibility matters
- You have domain knowledge

**Choose GRN-aware** when:
- Modeling perturbations
- Mechanistic understanding needed
- You have regulatory network data

**Choose LoRA** when:
- Parameter efficiency is key
- You need multiple task-specific models
- Memory is constrained

**Choose adapters** when:
- You want stable training
- You need task modularity
- You're in low-data regime

**Choose partial unfreezing** when:
- You need more expressiveness
- You have sufficient data
- LoRA/adapters aren't enough

---

## Related Documents

- [data_shape.md](data_shape.md) — Understanding transformer tensor shapes
- [../latent_diffusion/](../latent_diffusion/) — Latent diffusion implementation
- [../DiT/](../DiT/) — DiT architecture details
- [../DDPM/02a_diffusion_arch_gene_expression.md](../DDPM/02a_diffusion_arch_gene_expression.md) — Tokenization options

---

## References

**Foundation models**:
- Theodoris et al. (2023): "Transfer learning enables predictions in network biology" (Geneformer)
- Cui et al. (2024): "scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics"

**Adaptation strategies**:
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Houlsby et al. (2019): "Parameter-Efficient Transfer Learning for NLP" (Adapters)
- Perez et al. (2018): "FiLM: Visual Reasoning with a General Conditioning Layer"

**Latent diffusion**:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers" (DiT)

**Computational biology applications**:
- Lotfollahi et al. (2023): "Predicting cellular responses to novel drug combinations" (CPA)
- Lopez et al. (2018): "Deep generative modeling for single-cell transcriptomics" (scVI)
