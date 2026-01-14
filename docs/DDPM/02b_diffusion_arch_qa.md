# Architecture Q&A: Gene Expression Diffusion Models

This document addresses common questions about architectural choices for gene expression data in diffusion models, particularly around the latent token approach.

**Related**: [02a_diffusion_arch_gene_expression.md](02a_diffusion_arch_gene_expression.md) — Main architecture document

---

## Question 1: Handling Thousands of Samples

**Question**: The latent token architecture shows encoding gene expression to tokens. But how does this account for gene expression data where each gene is represented by thousands of samples with expression levels?

### Answer: Sample-by-Sample Processing

The architecture processes gene expression **sample-by-sample**, not all samples at once. Let me clarify the dimensions:

#### What the Architecture Actually Does

```python
# ONE sample (single cell or bulk RNA-seq measurement)
x = gene_expression  # Shape: (num_genes,) = (20000,)
                     # This is ONE measurement: [gene1_count, gene2_count, ..., gene20000_count]

# Encode to tokens
z = encoder(x)  # Shape: (num_tokens, token_dim) = (64, 256)
                # Creates 64 "semantic tokens" from the 20K gene values

# In practice, we batch multiple samples
x_batch = gene_expressions  # Shape: (batch_size, num_genes) = (32, 20000)
z_batch = encoder(x_batch)  # Shape: (32, 64, 256)
                            # 32 samples, each encoded to 64 tokens
```

#### Data Flow Clarification

```
DATASET STRUCTURE:
──────────────────
You have: N samples × 20,000 genes
- Sample 1: [gene1=5, gene2=120, ..., gene20000=3]
- Sample 2: [gene1=8, gene2=95, ..., gene20000=7]
- ...
- Sample N: [gene1=12, gene2=150, ..., gene20000=2]

TRAINING:
─────────
Each training iteration:
1. Sample a batch (e.g., 32 samples)
2. Each sample processed independently through encoder
   Sample 1 → Encoder → 64 tokens of dim 256
   Sample 2 → Encoder → 64 tokens of dim 256
   ...
   Sample 32 → Encoder → 64 tokens of dim 256

3. Batch shape: (32, 64, 256)
   - 32 samples (batch dimension)
   - 64 tokens per sample (sequence dimension)
   - 256 features per token (feature dimension)

4. Transformer processes each sample's token sequence
5. Decoder: tokens → gene expression prediction
```

#### The Key Insight

The encoder is **NOT** trying to encode all samples into one representation. Instead:

- **Input**: One gene expression profile (20K genes for one cell/sample)
- **Output**: A compressed representation as 64 tokens (each 256-dim)
- **Batching**: Process multiple samples in parallel (standard minibatch training)

Think of it like processing images:

```python
# Images
images = (batch=32, channels=3, height=224, width=224)
# Each image processed independently

# Gene expression
gene_expr = (batch=32, num_genes=20000)
# Each sample processed independently → (batch=32, num_tokens=64, token_dim=256)
```

#### Complete Training Example

```python
# ═══════════════════════════════════════════════════════════
# GENE EXPRESSION DATA STRUCTURE
# ═══════════════════════════════════════════════════════════

# Dataset: Collection of samples
dataset = {
    'sample_1': [gene1=5, gene2=120, ..., gene20000=3],    # Cell 1
    'sample_2': [gene1=8, gene2=95, ..., gene20000=7],     # Cell 2
    'sample_3': [gene1=12, gene2=150, ..., gene20000=2],   # Cell 3
    ...
    'sample_N': [...]
}

# ═══════════════════════════════════════════════════════════
# DURING TRAINING (Minibatch)
# ═══════════════════════════════════════════════════════════

# Step 1: Sample a batch
batch = 32 samples
x = (32, 20000)  # 32 samples, each with 20K gene counts

# Step 2: Encode each sample to tokens
# Each sample processed independently!
z = encoder(x)  # (32, 64, 256)
                # ↑   ↑   ↑
                # |   |   └─ Features per token
                # |   └───── Tokens per sample
                # └───────── Batch size

# Step 3: Add positional encoding (per sample)
z = z + pos_embed  # pos_embed: (1, 64, 256), broadcasts across batch

# Step 4: Transformer processes each sample's token sequence
# Attention operates WITHIN each sample's 64 tokens
# (Can also attend across samples if desired, but typically within)
t = timesteps           # (32,) - Diffusion timestep for each sample in batch
                        #        e.g., [500, 732, 123, ..., 891]
                        #        Controls noise level (high t = more noise)
                        
condition = conditions  # (32, cond_dim) - Optional conditioning per sample
                        # Examples:
                        #   - Cell type: [CD4+, B_cell, NK, ..., Monocyte]
                        #   - Perturbation: [CRISPR_gene1, drug_A, ..., control]
                        #   - Disease state: [healthy, disease_A, ..., healthy]
                        # Can be None for unconditional generation

z_out = transformer(z, t, condition)  # (32, 64, 256)
# Transformer uses:
#   - z: Token sequences to process
#   - t: Time conditioning (via AdaLN - modulates features based on noise level)
#   - condition: Biological conditioning (affects what to generate)

# Step 5: Decode back to gene space
# Each sample's tokens → that sample's gene expression
x_pred = decoder(z_out)  # (32, 20000)
```

#### Understanding the Transformer Inputs

Let's break down what `transformer(z, t, condition)` means:

##### 1. Token Sequences (z)

```python
z = (32, 64, 256)
# Main input: The latent token sequences to process
# - 32 samples in batch
# - Each sample has 64 tokens
# - Each token has 256 features

# Think of each sample's 64 tokens as a "sentence"
# where each token represents a semantic gene module
```

##### 2. Timestep (t)

```python
t = (32,)  # One timestep per sample in batch
# Example values: [500, 732, 123, 891, ..., 445]

# What does t mean?
# - Diffusion timestep: ranges from 0 (clean) to T (pure noise)
# - t=0: Nearly clean gene expression
# - t=500: Medium noise level
# - t=1000: Almost pure noise

# How is t used?
# - Embedded via sinusoidal encoding: t → time_embed (256-dim)
# - Used for Adaptive LayerNorm (AdaLN):
#     γ, β = MLP(time_embed)
#     h_modulated = γ * LayerNorm(h) + β
# - Tells model "how much noise is present"
# - Model adjusts its behavior based on noise level

# Why different t per sample?
# - During training: randomly sample t for each sample
# - Makes model learn to denoise at ALL noise levels
# - More efficient training (diverse timesteps per batch)
```

##### 3. Condition (condition)

```python
condition = (32, cond_dim) or None
# Optional conditioning information per sample

# Examples of what condition could be:

# A) Cell type conditioning
condition = ['CD4+ T cell', 'B cell', 'NK cell', ...]
# After embedding: (32, 256)
# Purpose: Generate expression for specific cell types

# B) Perturbation conditioning
condition = {
    'gene_knockout': ['FOXO1', 'TP53', None, ...],
    'drug': ['drug_A', None, 'drug_B', ...],
    'dose': [1.0, 0.0, 0.5, ...]
}
# After embedding: (32, 512)  # Combined embeddings
# Purpose: Generate response to perturbations

# C) Disease state conditioning
condition = ['healthy', 'diabetes', 'healthy', 'cancer', ...]
# After embedding: (32, 256)
# Purpose: Generate disease-specific expression

# D) Multi-modal conditioning
condition = {
    'cell_type': ['T_cell', ...],
    'tissue': ['liver', ...],
    'age': [45, ...],
    'sex': ['M', ...]
}
# After embedding: (32, 1024)  # All concatenated
# Purpose: Complex, multi-factor conditioning

# E) No conditioning (unconditional generation)
condition = None
# Purpose: Generate generic, diverse gene expression
```

---

#### How to Represent Conditions of Different Complexity

**Key Question**: How do we go from raw conditioning information (strings, dictionaries, numbers) to the tensor that the transformer uses?

##### Strategy 1: Simple Categorical Conditions

**Example**: Cell type conditioning

```python
# Raw input
cell_types = ['CD4+ T cell', 'B cell', 'NK cell', 'Monocyte', ...]  # (32,)

# Step 1: Convert to indices
cell_type_to_idx = {
    'CD4+ T cell': 0,
    'B cell': 1,
    'NK cell': 2,
    'Monocyte': 3,
    ...
}
indices = [cell_type_to_idx[ct] for ct in cell_types]  # [0, 1, 2, 3, ...]

# Step 2: Embed via embedding layer
class CellTypeEmbedder(nn.Module):
    def __init__(self, num_cell_types=50, embed_dim=256):
        super().__init__()
        # Learned embedding for each cell type
        self.embedding = nn.Embedding(num_cell_types, embed_dim)
    
    def forward(self, cell_type_indices):
        # Input: (batch,) - integer indices
        # Output: (batch, embed_dim)
        return self.embedding(cell_type_indices)

# Result
embedder = CellTypeEmbedder(num_cell_types=50, embed_dim=256)
condition_embed = embedder(torch.tensor(indices))  # (32, 256)
```

**Why this works**: 
- Cell types are discrete categories
- Each gets a learnable vector representation
- Similar to word embeddings in NLP

##### Strategy 2: Complex Multi-Component Conditions

**Example**: Perturbation conditioning (gene knockout + drug + dose)

```python
# Raw input (different types of information)
perturbations = {
    'gene_knockout': ['FOXO1', 'TP53', None, 'MYC', ...],    # Categorical or None
    'drug': ['drug_A', None, 'drug_B', 'drug_A', ...],       # Categorical or None
    'dose': [1.0, 0.0, 0.5, 1.0, ...]                         # Continuous
}

# ════════════════════════════════════════════════════════════
# Approach A: Separate Embeddings + Concatenation
# ════════════════════════════════════════════════════════════

class PerturbationEmbedder(nn.Module):
    def __init__(
        self,
        num_genes=1000,       # Number of possible knockout targets
        num_drugs=100,        # Number of drugs
        embed_dim=128         # Embedding dimension per component
    ):
        super().__init__()
        
        # Separate embeddings for each component
        self.gene_embed = nn.Embedding(num_genes + 1, embed_dim)  # +1 for "none"
        self.drug_embed = nn.Embedding(num_drugs + 1, embed_dim)  # +1 for "none"
        
        # MLP for continuous dose
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Final projection (concatenated → combined)
        self.combiner = nn.Sequential(
            nn.Linear(3 * embed_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        # Special indices for "none"
        self.gene_none_idx = num_genes
        self.drug_none_idx = num_drugs
    
    def forward(self, gene_indices, drug_indices, doses):
        """
        Args:
            gene_indices: (batch,) - indices of knockout genes (or none_idx)
            drug_indices: (batch,) - indices of drugs (or none_idx)
            doses: (batch,) - continuous dose values
        
        Returns:
            (batch, 512) - combined perturbation embedding
        """
        # Embed each component separately
        gene_emb = self.gene_embed(gene_indices)      # (batch, 128)
        drug_emb = self.drug_embed(drug_indices)      # (batch, 128)
        dose_emb = self.dose_encoder(doses[:, None])  # (batch, 128)
        
        # Concatenate
        combined = torch.cat([gene_emb, drug_emb, dose_emb], dim=-1)  # (batch, 384)
        
        # Project to final dimension
        condition_embed = self.combiner(combined)  # (batch, 512)
        
        return condition_embed

# Usage
gene_to_idx = {'FOXO1': 0, 'TP53': 1, 'MYC': 2, None: embedder.gene_none_idx}
drug_to_idx = {'drug_A': 0, 'drug_B': 1, None: embedder.drug_none_idx}

gene_indices = torch.tensor([gene_to_idx[g] for g in perturbations['gene_knockout']])
drug_indices = torch.tensor([drug_to_idx[d] for d in perturbations['drug']])
doses = torch.tensor(perturbations['dose'])

embedder = PerturbationEmbedder()
condition_embed = embedder(gene_indices, drug_indices, doses)  # (32, 512)
```

**Why this works**:
- Each component embedded separately (preserves semantics)
- Concatenation combines all information
- MLP learns interactions between components
- Handles missing values naturally (None → special index)

##### Strategy 3: Variable-Length Complex Conditions

**Example**: Multiple perturbations per sample (variable number)

```python
# Raw input: Some samples have multiple perturbations
perturbations = [
    ['FOXO1_knockout', 'drug_A_0.5'],              # Sample 1: 2 perturbations
    ['TP53_knockout'],                             # Sample 2: 1 perturbation
    ['drug_B_1.0', 'drug_C_0.3', 'MYC_knockout'],  # Sample 3: 3 perturbations
    [],                                             # Sample 4: No perturbation
    ...
]

# ════════════════════════════════════════════════════════════
# Approach B: Set Embedding (Order-Invariant)
# ════════════════════════════════════════════════════════════

class SetPerturbationEmbedder(nn.Module):
    def __init__(self, num_perturbations=500, embed_dim=256, max_perts=10):
        super().__init__()
        
        # Embedding for each perturbation type
        self.pert_embed = nn.Embedding(num_perturbations + 1, embed_dim)  # +1 for padding
        
        # Transformer encoder (order-invariant aggregation)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4),
            num_layers=2
        )
        
        # Pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)  # or attention pooling
        
        self.pad_idx = num_perturbations
        self.max_perts = max_perts
    
    def forward(self, perturbation_indices, mask):
        """
        Args:
            perturbation_indices: (batch, max_perts) - padded perturbation indices
            mask: (batch, max_perts) - True where valid, False for padding
        
        Returns:
            (batch, embed_dim) - aggregated perturbation embedding
        """
        # Embed each perturbation
        pert_embs = self.pert_embed(perturbation_indices)  # (batch, max_perts, embed_dim)
        
        # Transformer (handles variable length via masking)
        pert_embs = pert_embs.transpose(0, 1)  # (max_perts, batch, embed_dim)
        encoded = self.transformer(pert_embs, src_key_padding_mask=~mask)
        encoded = encoded.transpose(0, 1)  # (batch, max_perts, embed_dim)
        
        # Pool to fixed size (mean over valid perturbations)
        # Apply mask before pooling
        encoded = encoded * mask.unsqueeze(-1)
        condition_embed = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
        return condition_embed  # (batch, embed_dim)

# Usage: Convert to padded format
pert_to_idx = {'FOXO1_knockout': 0, 'drug_A_0.5': 1, ...}

def collate_perturbations(pert_lists, max_perts=10, pad_idx=500):
    """Convert variable-length lists to padded tensors"""
    batch_size = len(pert_lists)
    
    # Create padded tensor
    indices = torch.full((batch_size, max_perts), pad_idx, dtype=torch.long)
    mask = torch.zeros(batch_size, max_perts, dtype=torch.bool)
    
    for i, perts in enumerate(pert_lists):
        num_perts = min(len(perts), max_perts)
        indices[i, :num_perts] = torch.tensor([pert_to_idx[p] for p in perts[:num_perts]])
        mask[i, :num_perts] = True
    
    return indices, mask

indices, mask = collate_perturbations(perturbations)
embedder = SetPerturbationEmbedder()
condition_embed = embedder(indices, mask)  # (32, 256)
```

**Why this works**:
- Handles variable number of perturbations
- Order-invariant (set semantics, not sequence)
- Masking handles different lengths
- Aggregation (mean/max/attention) gives fixed-size output

##### Strategy 4: Hierarchical Conditions

**Example**: Multi-level biological context (organism → tissue → cell type → state)

```python
# Raw input: Hierarchical structure
conditions = {
    'organism': ['human', 'human', 'mouse', ...],
    'tissue': ['liver', 'brain', 'liver', ...],
    'cell_type': ['hepatocyte', 'neuron', 'hepatocyte', ...],
    'state': ['healthy', 'healthy', 'diseased', ...]
}

# ════════════════════════════════════════════════════════════
# Approach C: Hierarchical Embeddings
# ════════════════════════════════════════════════════════════

class HierarchicalConditionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Separate embeddings for each level
        self.organism_embed = nn.Embedding(10, 64)   # Few organisms
        self.tissue_embed = nn.Embedding(50, 128)    # More tissues
        self.cell_type_embed = nn.Embedding(200, 256)  # Many cell types
        self.state_embed = nn.Embedding(20, 128)     # Few states
        
        # Hierarchical combination (bottom-up)
        self.combine_tissue_cell = nn.Sequential(
            nn.Linear(128 + 256, 256),
            nn.SiLU()
        )
        
        self.combine_org_tissue_cell = nn.Sequential(
            nn.Linear(64 + 256, 256),
            nn.SiLU()
        )
        
        self.combine_all = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
    
    def forward(self, organism_idx, tissue_idx, cell_type_idx, state_idx):
        """Hierarchical composition: organism > tissue > cell_type, + state"""
        
        # Embed each level
        org_emb = self.organism_embed(organism_idx)      # (batch, 64)
        tis_emb = self.tissue_embed(tissue_idx)          # (batch, 128)
        cel_emb = self.cell_type_embed(cell_type_idx)    # (batch, 256)
        sta_emb = self.state_embed(state_idx)            # (batch, 128)
        
        # Hierarchical composition
        # Level 1: Tissue + Cell type (cell exists in tissue)
        tissue_cell = self.combine_tissue_cell(
            torch.cat([tis_emb, cel_emb], dim=-1)
        )  # (batch, 256)
        
        # Level 2: Organism + (Tissue + Cell)
        org_tissue_cell = self.combine_org_tissue_cell(
            torch.cat([org_emb, tissue_cell], dim=-1)
        )  # (batch, 256)
        
        # Level 3: Add state (orthogonal to hierarchy)
        final = self.combine_all(
            torch.cat([org_tissue_cell, sta_emb], dim=-1)
        )  # (batch, 512)
        
        return final

# Usage
organism_to_idx = {'human': 0, 'mouse': 1, 'rat': 2}
tissue_to_idx = {'liver': 0, 'brain': 1, ...}
# ... similar for cell_type and state

embedder = HierarchicalConditionEmbedder()
condition_embed = embedder(
    torch.tensor([organism_to_idx[o] for o in conditions['organism']]),
    torch.tensor([tissue_to_idx[t] for t in conditions['tissue']]),
    torch.tensor([cell_type_to_idx[c] for c in conditions['cell_type']]),
    torch.tensor([state_to_idx[s] for s in conditions['state']])
)  # (32, 512)
```

**Why this works**:
- Respects biological hierarchy
- Bottom-up composition (cell → tissue → organism)
- Different embedding sizes reflect complexity
- Can incorporate prior knowledge about relationships

---

#### Comparison of Strategies

| Complexity | Example | Strategy | Output Shape | Best For |
|------------|---------|----------|--------------|----------|
| **Simple Categorical** | Cell type | Embedding layer | (batch, 256) | Single discrete condition |
| **Multi-Component** | Gene KO + Drug + Dose | Separate embeds + concat | (batch, 512) | Fixed set of heterogeneous conditions |
| **Variable-Length** | Multiple perturbations | Set embedding + pooling | (batch, 256) | Variable number of conditions |
| **Hierarchical** | Organism → Tissue → Cell | Hierarchical composition | (batch, 512) | Nested/structured conditions |
| **Very Complex** | Text descriptions | Pre-trained encoder (CLIP, T5) | (batch, 768) | Natural language |

---

#### Practical Implementation Template

```python
class UniversalConditionEmbedder(nn.Module):
    """Handles conditions of varying complexity"""
    
    def __init__(self):
        super().__init__()
        
        # Simple categorical
        self.cell_type_embed = nn.Embedding(50, 256)
        
        # Multi-component
        self.perturbation_embed = PerturbationEmbedder()
        
        # Continuous
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Combiner (if using multiple condition types)
        self.combiner = nn.Linear(256 + 512 + 128, 512)  # Adjust based on what you use
    
    def forward(self, batch):
        """
        Flexible forward pass based on what conditions are present
        
        Args:
            batch: Dictionary containing various condition types
        
        Returns:
            Combined condition embedding
        """
        embeds = []
        
        # Handle cell type if present
        if 'cell_type' in batch:
            cell_emb = self.cell_type_embed(batch['cell_type'])
            embeds.append(cell_emb)
        
        # Handle perturbation if present
        if 'perturbation' in batch:
            pert_emb = self.perturbation_embed(
                batch['perturbation']['gene'],
                batch['perturbation']['drug'],
                batch['perturbation']['dose']
            )
            embeds.append(pert_emb)
        
        # Handle continuous dose if present
        if 'dose' in batch:
            dose_emb = self.dose_encoder(batch['dose'][:, None])
            embeds.append(dose_emb)
        
        # Combine all present conditions
        if len(embeds) == 0:
            return None  # Unconditional
        elif len(embeds) == 1:
            return embeds[0]
        else:
            combined = torch.cat(embeds, dim=-1)
            return self.combiner(combined)
```

---

#### Summary: From Raw Data to Condition Tensor

The general pipeline:

```
Raw Condition Data
    ↓
[Convert to appropriate format]
    ↓ (strings → indices, numbers → tensors, etc.)
Processable Format
    ↓
[Embed each component]
    ↓ (embeddings, MLPs, encoders)
Component Embeddings
    ↓
[Combine components]
    ↓ (concatenation, addition, hierarchical composition)
Final Condition Tensor
    ↓ (batch, condition_dim)
[Input to Transformer]
```

**Key principles**:
1. **Each component** gets its own embedding strategy
2. **Categorical** → Embedding layers
3. **Continuous** → MLPs
4. **Complex** → Combination of above
5. **Variable-length** → Masking + pooling
6. **Hierarchical** → Compositional embeddings
7. **Combine** via concatenation, addition, or learned fusion

The condition tensor shape `(batch, condition_dim)` is then used by the transformer via AdaLN, cross-attention, or other conditioning mechanisms.

---

```python
# How is condition used?
# Option 1: AdaLN (like time)
#   γ_cond, β_cond = MLP(condition_embed)
#   h = γ_time * γ_cond * LayerNorm(h) + β_time + β_cond
#
# Option 2: Cross-attention
#   h_out = CrossAttention(query=h, key=condition, value=condition)
#
# Option 3: Concatenation
#   combined = concat([time_embed, condition_embed])
#   γ, β = MLP(combined)
```

##### Complete Example with All Inputs

```python
# Training step
for batch in dataloader:
    # Load data
    x_0 = batch['expression']      # (32, 20000) - Clean gene expression
    cell_types = batch['cell_type'] # (32,) - ['T_cell', 'B_cell', ...]
    
    # Sample random timesteps (different for each sample)
    t = torch.randint(0, 1000, (32,))  # e.g., [234, 789, 12, 901, ...]
    
    # Add noise according to timestep
    noise = torch.randn_like(x_0)
    x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * noise
    # x_t: (32, 20000) - Noisy gene expression
    
    # Embed conditions
    condition = cell_type_embedder(cell_types)  # (32, 256)
    
    # Forward pass
    z = encoder(x_t)                          # (32, 64, 256) - Encode to tokens
    z_out = transformer(z, t, condition)      # (32, 64, 256) - Process with context
    noise_pred = decoder(z_out)               # (32, 20000) - Predict noise
    
    # Loss
    loss = F.mse_loss(noise_pred, noise)
```

---

## Question 2: Is This Like a "Black and White Image"?

**Question**: Is the token representation `(num_tokens, token_dim)` almost like a "black and white image" with only 1 channel of size `(num_tokens, token_dim)`?

**Reference**: The positional embedding line in the architecture:
```python
self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim))
```

### Answer: No, It's a Sequence, Not an Image

Great intuition, but not quite! Let me explain the dimensionality:

#### Image vs Token Representation

```python
# GRAYSCALE IMAGE (what you're thinking of)
image = (batch, 1, height, width)
      = (32, 1, 224, 224)
# - 1 channel (grayscale)
# - Spatial dimensions: height × width
# - Still has 2D spatial structure

# LATENT TOKENS (what we have)
tokens = (batch, num_tokens, token_dim)
       = (32, 64, 256)
# - 64 tokens (like 64 "patches")
# - 256 features per token
# - NO spatial structure! Just a SEQUENCE of tokens
```

#### Better Analogy: Sequence, Not Image

The token representation is more like:

```
TEXT SEQUENCE (NLP):
────────────────────
sentence = ["The", "cat", "sat", "on", "mat"]
embeddings = (batch, seq_len, embed_dim)
           = (32, 5, 768)
# - 5 words in sequence
# - 768-dimensional embedding per word

GENE EXPRESSION TOKENS:
───────────────────────
gene_profile → [token1, token2, ..., token64]
tokens = (batch, num_tokens, token_dim)
       = (32, 64, 256)
# - 64 tokens in sequence
# - 256-dimensional features per token
# - Each token represents some "semantic cluster" of genes
```

#### Visual Comparison

```
IMAGE REPRESENTATION:
═══════════════════════
     ┌─────────────┐
     │ ░░░░░░░░░░░ │  Channel 1 (R)
     │ ░░░▓▓░░░░░░ │
     │ ░░▓▓▓▓░░░░░ │
     └─────────────┘
     ┌─────────────┐
     │ ░░░░░░░░░░░ │  Channel 2 (G)
     │ ░░░▓▓░░░░░░ │
     └─────────────┘
     ┌─────────────┐
     │ ░░░░░░░░░░░ │  Channel 3 (B)
     └─────────────┘
Shape: (height, width, channels)
Structure: 2D spatial grid


TOKEN REPRESENTATION:
═══════════════════════
Token 1: [0.5, -0.2, 0.8, ..., 0.3]  ← 256 features
Token 2: [0.1, 0.7, -0.4, ..., 0.9]
Token 3: [-0.3, 0.2, 0.6, ..., -0.1]
...
Token 64: [0.4, -0.5, 0.1, ..., 0.7]

Shape: (num_tokens, token_dim)
Structure: 1D sequence (like words in a sentence)
```

#### What Each Token Might Represent

Unlike pixels in an image, each token captures **semantic information**:

```python
# Hypothetical learned representation
Token 1 → "Cell cycle genes" (high for proliferating cells)
Token 2 → "Immune response genes" (high in activated immune cells)  
Token 3 → "Metabolic genes" (high in metabolically active cells)
Token 4 → "Housekeeping genes" (stable across conditions)
...
Token 64 → "Rare pathway genes"

# Each token is 256-dimensional, encoding complex patterns
Token 1 = [0.5, -0.2, 0.8, ..., 0.3]
          ↑    ↑    ↑        ↑
          Features capturing different aspects of "cell cycle-ness"
```

---

## Dimensionality Breakdown

### Comparison Table

| Dimension    | Name         | Size | Meaning                                |
|--------------|--------------|------|----------------------------------------|
| **Batch**    | `batch_size` | 32   | Number of samples processed together   |
| **Tokens**   | `num_tokens` | 64   | Number of semantic "chunks" per sample |
| **Features** | `token_dim`  | 256  | Features describing each token         |

### Cross-Domain Comparison

| Data Type  | Dimension 1 | Dimension 2   | Dimension 3   | Dimension 4 |
|------------|-------------|---------------|---------------|-------------|
| **Image**  | batch=32    | channels=3    | height=224    | width=224   |
| **Tokens** | batch=32    | num_tokens=64 | token_dim=256 | -           |
| **Text**   | batch=32    | seq_len=50    | embed_dim=768 | -           |

**Key difference**: 
- Images have **2D spatial structure** (height × width)
- Tokens have **1D sequence structure** (just num_tokens)
- The positional encoding provides ordering info (like in NLP)

---

## Why Not Actually Like an Image?

If we tried to make it image-like:

```python
# ❌ WRONG: Treating as image
tokens_as_image = (batch, channels=1, height=8, width=8)
                = (32, 1, 8, 8)  # 64 "pixels" arranged spatially

# ✅ CORRECT: Treating as sequence
tokens_as_sequence = (batch, seq_len=64, features=256)
                   = (32, 64, 256)  # 64 tokens with 256 features each
```

### Why Sequence is Better

**1. No inherent spatial structure in gene expression**
- Gene order in genome ≠ meaningful for expression patterns
- Unlike pixels, where neighbors have spatial meaning

**2. Transformers work on sequences**
- Self-attention doesn't assume spatial locality
- Can capture any gene-gene interactions

**3. More like NLP than vision**
- Genes are like "words" in a biological "sentence"
- Tokens are semantic clusters, not spatial patches

### Example: What Spatial Structure Would Mean

```python
# If we treated as 2D image (8×8 grid of tokens)
# ❌ This would imply:
# - Token at position (0,0) is "near" token at (0,1)
# - Token at position (0,0) is "far" from token at (7,7)
# - Spatial locality matters

# But in gene expression:
# - Token 1 (cell cycle) might interact strongly with Token 50 (DNA repair)
# - No reason to assume nearby tokens are more related
# - Attention should be free to connect any tokens
```

---

## Practical Implications

### For Model Design

**1. Use sequence models, not convolutional models**
```python
# ✅ Good: Transformer (no spatial bias)
transformer = nn.TransformerEncoder(...)

# ❌ Bad: CNN (assumes spatial locality)
cnn = nn.Conv2d(...)  # Wrong inductive bias for gene expression
```

**2. Positional encoding is flexible**
```python
# Can use learned or sinusoidal
self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim))

# Or sinusoidal (like BERT)
self.pos_embed = SinusoidalPositionEmbeddings(token_dim)
```

**3. Attention is unrestricted**
```python
# Each token can attend to all other tokens
# No spatial locality assumption
attn_scores = Q @ K.T  # (num_tokens, num_tokens)
# All-to-all attention
```

### For Interpretation

**1. Tokens are semantic, not spatial**
- Analyze what biological patterns each token captures
- Use gene loadings to interpret tokens
- Compare to known pathways/modules

**2. Token order doesn't matter (much)**
- Positional encoding adds ordering info
- But tokens aren't inherently ordered like pixels
- Could potentially shuffle and retrain

**3. Visualization differs from images**
```python
# For images: Show as 2D grid
plt.imshow(image)

# For tokens: Show as heatmap or t-SNE
plt.imshow(tokens.T)  # (token_dim, num_tokens)
# Or project to 2D
tsne = TSNE(n_components=2)
tokens_2d = tsne.fit_transform(tokens)
plt.scatter(tokens_2d[:, 0], tokens_2d[:, 1])
```

---

## Summary

### Question 1: Handling Thousands of Samples

The architecture processes **one sample at a time** (with batching for efficiency). The "thousands of samples" are your dataset, processed in minibatches during training, just like images in computer vision.

**Key points**:
- Each sample: 20K genes → 64 tokens (256-dim each)
- Batching: Process 32 samples in parallel
- Training: Iterate through dataset in minibatches
- Same as standard deep learning practice

### Question 2: Is This Like a Black-and-White Image?

The token representation is **NOT like a black-and-white image**. It's more like:

**Text embeddings**: A sequence of semantic tokens
- Each token is a 256-dimensional feature vector
- No 2D spatial structure, just 1D sequence
- Position info added via positional encoding

**Key intuition**: Think of it as compressing a 20,000-dimensional gene expression vector into a **sequence of 64 semantic tokens** (each 256-dim), where each token represents some learned biological pattern/module.

### Correct Mental Model

```
Gene Expression → Encoder → Sequence of Semantic Tokens → Transformer → Decoder → Prediction

NOT:
Gene Expression → Encoder → 2D Image → CNN → Decoder → Prediction
```

**Think**: NLP (BERT, GPT) not Computer Vision (ResNet, ViT)

---

## Related Questions

### Q: Why 64 tokens specifically?

**A**: Hyperparameter choice balancing:
- **Fewer tokens** (e.g., 32): Faster, but less capacity
- **More tokens** (e.g., 128): More capacity, but slower
- **64 tokens**: Sweet spot for most applications

Experiment to find optimal for your data.

### Q: Can tokens attend across samples in a batch?

**A**: Typically no, but possible:

```python
# Standard: Within-sample attention
# Each sample's 64 tokens attend to each other
attn_mask = None  # Full attention within sample

# Advanced: Cross-sample attention
# Tokens can attend across samples (rare for gene expression)
# Useful for: batch effects, sample relationships
```

### Q: How do I interpret learned tokens?

**A**: Several approaches:

```python
# 1. Gene loadings
# Which genes contribute most to each token?
encoder_weights = model.encoder[0].weight  # (2048, 20000)
# Analyze top genes per token

# 2. Activation patterns
# When is each token active?
z = model.encode(x_batch)  # (batch, 64, 256)
token_activations = z.mean(dim=-1)  # (batch, 64)
# Correlate with cell types, conditions

# 3. Pathway enrichment
# Do tokens align with known pathways?
# Use GSEA on gene loadings per token
```

---

## Related Documents

- [02a_diffusion_arch_gene_expression.md](02a_diffusion_arch_gene_expression.md) — Architecture options
- [02_ddpm_training.md](02_ddpm_training.md) — Training strategies
- [../DiT/01_dit_foundations.md](../DiT/01_dit_foundations.md) — Transformer details
- [../DiT/open_research_tokenization.md](../DiT/open_research_tokenization.md) — Tokenization deep dive

---

## References

**Sequence models for biology**:
- Theodoris et al. (2023): "Transfer learning enables predictions in network biology" (Geneformer)
- Cui et al. (2024): "scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics"

**Transformers and attention**:
- Vaswani et al. (2017): "Attention Is All You Need"
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"

**Latent representations**:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
- Kingma & Welling (2014): "Auto-Encoding Variational Bayes"
