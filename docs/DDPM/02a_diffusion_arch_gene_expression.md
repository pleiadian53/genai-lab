# Architecture Choices for Gene Expression Data

This document explores different architectural approaches for diffusion models on gene expression data, moving beyond simple "tabular MLP" treatment to more sophisticated tokenization and modeling strategies.

**Key question**: How should we represent and model gene expression data in diffusion models?

---

## The Problem with "Tabular Data" Treatment

### Why Gene Expression Isn't Just Tabular

**Standard tabular approach** (from `02_ddpm_training.md`):
```python
# Treat as flat vector
x = gene_expression  # (batch, 20000)
x_noisy = add_noise(x, t)
noise_pred = mlp(x_noisy, t)  # Simple MLP
```

**Problems**:
1. **Ignores structure**: Genes aren't independent features
2. **Ignores biology**: Gene regulatory networks, pathways, modules
3. **High dimensionality**: 20K genes → huge parameter count
4. **No inductive bias**: Model must learn everything from scratch
5. **Poor generalization**: Doesn't transfer across datasets/conditions

**Reality**: Gene expression has rich structure that should inform architecture.

---

## Rethinking Tokenization for Gene Expression

**Core insight**: "Tokenization" = "How we factor the object so attention has something meaningful to attend over"

**Not just a preprocessing step** — tokenization IS an architectural choice that determines:
- What the model can learn
- How efficiently it learns
- How well it generalizes
- How interpretable it is

---

## Option 1: Latent Tokens (Recommended Default)

### Concept

**Encoder-decoder architecture** with learned latent representation:

```
Gene expression (20K) → Encoder → Latent tokens (m×d) → DiT → Decoder → Output
```

**Key idea**: Learn a compressed, structured representation where each token captures meaningful biological variation.

### Architecture

```python
class LatentTokenDiffusion(nn.Module):
    def __init__(
        self,
        num_genes=20000,
        num_tokens=64,      # m tokens
        token_dim=256,      # d dimensions per token
        num_layers=12,
        num_heads=8
    ):
        super().__init__()
        
        # Encoder: gene expression → latent tokens
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, num_tokens * token_dim)
        )
        
        # Reshape to tokens
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        # Positional encoding for tokens
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        
        # Transformer (DiT)
        self.transformer = DiT(
            embed_dim=token_dim,
            depth=num_layers,
            num_heads=num_heads
        )
        
        # Decoder: latent tokens → gene expression parameters
        self.decoder = nn.Sequential(
            nn.Linear(num_tokens * token_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, num_genes * 2)  # Mean and variance for NB/ZINB
        )
    
    def encode(self, x):
        """Encode gene expression to latent tokens."""
        # x: (batch, num_genes)
        z = self.encoder(x)  # (batch, num_tokens * token_dim)
        z = z.view(-1, self.num_tokens, self.token_dim)  # (batch, num_tokens, token_dim)
        z = z + self.pos_embed  # Add positional encoding
        return z
    
    def decode(self, z):
        """Decode latent tokens to gene expression parameters."""
        # z: (batch, num_tokens, token_dim)
        z_flat = z.view(-1, self.num_tokens * self.token_dim)
        params = self.decoder(z_flat)  # (batch, num_genes * 2)
        mean, logvar = params.chunk(2, dim=-1)
        return mean, logvar
    
    def forward(self, x, t, condition=None):
        """
        Forward pass for diffusion.
        
        Args:
            x: Gene expression (batch, num_genes)
            t: Timesteps (batch,)
            condition: Optional conditioning (perturbations, cell types, etc.)
        
        Returns:
            noise_pred: Predicted noise (batch, num_genes)
        """
        # Encode to latent tokens
        z = self.encode(x)  # (batch, num_tokens, token_dim)
        
        # Run transformer on tokens
        z_out = self.transformer(z, t, condition)  # (batch, num_tokens, token_dim)
        
        # Decode to gene space
        mean, logvar = self.decode(z_out)
        
        return mean  # Or return distribution parameters
```

### Why It's Good

**1. Learned, data-adaptive tokenization**
- Model learns what biological variation to capture
- Tokens emerge from data, not imposed a priori
- Can discover novel gene modules/patterns

**2. Compute-friendly**
- 64 tokens << 20K genes
- Attention is O(m²) where m=64, not O(20000²)
- Enables scaling to large models

**3. Plays nicely with LoRA/adapters**
```python
# Fine-tune on new dataset with small adapter
class LoRAAdapter(nn.Module):
    def __init__(self, token_dim=256, rank=8):
        super().__init__()
        self.down = nn.Linear(token_dim, rank)
        self.up = nn.Linear(rank, token_dim)
    
    def forward(self, z):
        return z + self.up(self.down(z))

# Add adapter to frozen backbone
model.transformer.requires_grad_(False)
adapter = LoRAAdapter()
```

**4. Flexible conditioning**
- Easy to inject perturbation info at token level
- Can condition on cell type, time, etc.

### Training Strategy

```python
# Training loop
for x_0, condition in dataloader:
    # Sample noise and timestep
    t = torch.rand(batch_size)
    noise = torch.randn_like(x_0)
    
    # Add noise (in gene space or latent space)
    x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
    
    # Predict noise
    noise_pred = model(x_t, t, condition)
    
    # Loss
    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
```

**Alternative**: Diffuse in latent space
```python
# Encode to latent
z_0 = model.encode(x_0)

# Add noise in latent space
z_t = sqrt(alpha_t) * z_0 + sqrt(1 - alpha_t) * noise

# Predict in latent space
noise_pred = model.transformer(z_t, t, condition)

# Decode
x_pred = model.decode(noise_pred)
```

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| `num_tokens` | 32-128 | Balance between capacity and compute |
| `token_dim` | 256-512 | Higher for complex datasets |
| `num_layers` | 8-16 | Deeper for better quality |
| `num_heads` | 8-16 | Standard transformer setting |

---

## Option 2: Pathway/Module Tokens (Biologically Anchored)

### Concept

**Use biological knowledge** to define tokens as gene pathways or modules.

```
Genes → Group by pathway → Pathway embeddings → DiT → Output
```

**Key idea**: Each token represents a biological process (glycolysis, cell cycle, immune response, etc.)

### Architecture

```python
class PathwayTokenDiffusion(nn.Module):
    def __init__(
        self,
        num_genes=20000,
        pathway_db='msigdb',  # MSigDB, Reactome, GO, etc.
        token_dim=256,
        num_layers=12
    ):
        super().__init__()
        
        # Load pathway definitions
        self.pathways = load_pathways(pathway_db)  # Dict: pathway_name → gene_indices
        self.num_pathways = len(self.pathways)
        
        # Gene-to-pathway mapping
        self.gene_to_pathway = self._build_gene_pathway_matrix()  # (num_genes, num_pathways)
        
        # Pathway embedding
        self.pathway_embed = nn.ModuleDict({
            name: nn.Linear(len(genes), token_dim)
            for name, genes in self.pathways.items()
        })
        
        # Transformer on pathway tokens
        self.transformer = DiT(
            embed_dim=token_dim,
            depth=num_layers,
            num_heads=8
        )
        
        # Decoder: pathway tokens → gene expression
        self.gene_decoder = nn.ModuleDict({
            name: nn.Linear(token_dim, len(genes))
            for name, genes in self.pathways.items()
        })
    
    def encode_pathways(self, x):
        """
        Encode gene expression to pathway tokens.
        
        Args:
            x: Gene expression (batch, num_genes)
        
        Returns:
            pathway_tokens: (batch, num_pathways, token_dim)
        """
        tokens = []
        
        for pathway_name, gene_indices in self.pathways.items():
            # Extract genes for this pathway
            pathway_expr = x[:, gene_indices]  # (batch, num_genes_in_pathway)
            
            # Embed to token
            token = self.pathway_embed[pathway_name](pathway_expr)  # (batch, token_dim)
            tokens.append(token)
        
        # Stack tokens
        pathway_tokens = torch.stack(tokens, dim=1)  # (batch, num_pathways, token_dim)
        
        return pathway_tokens
    
    def decode_pathways(self, pathway_tokens):
        """
        Decode pathway tokens to gene expression.
        
        Args:
            pathway_tokens: (batch, num_pathways, token_dim)
        
        Returns:
            x_recon: Gene expression (batch, num_genes)
        """
        gene_predictions = torch.zeros(pathway_tokens.shape[0], self.num_genes, device=pathway_tokens.device)
        gene_counts = torch.zeros(self.num_genes, device=pathway_tokens.device)
        
        for i, (pathway_name, gene_indices) in enumerate(self.pathways.items()):
            # Decode token to genes
            token = pathway_tokens[:, i, :]  # (batch, token_dim)
            genes_pred = self.gene_decoder[pathway_name](token)  # (batch, num_genes_in_pathway)
            
            # Accumulate predictions
            gene_predictions[:, gene_indices] += genes_pred
            gene_counts[gene_indices] += 1
        
        # Average overlapping predictions
        x_recon = gene_predictions / gene_counts.clamp(min=1)
        
        return x_recon
    
    def forward(self, x, t, condition=None):
        # Encode to pathway tokens
        tokens = self.encode_pathways(x)
        
        # Transform with DiT
        tokens_out = self.transformer(tokens, t, condition)
        
        # Decode to genes
        x_out = self.decode_pathways(tokens_out)
        
        return x_out
```

### Why It's Good

**1. Interpretability**
- Each token has biological meaning
- Can explain predictions at pathway level
- Clinically legible ("upregulated immune pathways")

**2. Lower dimension**
- ~500 pathways vs 20K genes
- More tractable for analysis

**3. Transfer learning**
- Pathways are consistent across datasets
- Model trained on one dataset can transfer to another
- Easier to align across species (conserved pathways)

**4. Inductive bias**
- Encodes known biology
- Faster convergence
- Better generalization with limited data

### Pathway Databases

| Database | # Pathways | Coverage | Use Case |
|----------|------------|----------|----------|
| **MSigDB Hallmark** | 50 | Broad processes | High-level analysis |
| **MSigDB C2 (KEGG)** | 186 | Metabolic/signaling | Mechanistic studies |
| **Reactome** | 2,500+ | Detailed processes | Fine-grained analysis |
| **GO Biological Process** | 10,000+ | Very detailed | Comprehensive coverage |
| **Data-driven modules** | Variable | Dataset-specific | Custom applications |

**Recommendation**: Start with MSigDB Hallmark (50 pathways), scale to Reactome if needed.

### Data-Driven Module Discovery

**Alternative**: Learn modules from data instead of using databases.

```python
from sklearn.decomposition import NMF

# Learn gene modules via NMF
nmf = NMF(n_components=100, random_state=42)
W = nmf.fit_transform(gene_expression_matrix.T)  # (num_genes, 100)
H = nmf.components_  # (100, num_samples)

# W defines gene-to-module mapping
# Use W to create pathway_embed layers
```

---

## Option 3: Graph-Structured Tokens (GRN-Aware Attention)

### Concept

**Use gene regulatory networks** to structure attention.

```
Genes as tokens + GRN structure → Sparse attention → Output
```

**Key idea**: Attention is constrained by known regulatory relationships, avoiding full O(n²) computation.

### Architecture

```python
class GRNAwareDiffusion(nn.Module):
    def __init__(
        self,
        num_genes=20000,
        grn_adjacency=None,  # Sparse adjacency matrix
        token_dim=256,
        num_layers=8
    ):
        super().__init__()
        
        # Gene embeddings
        self.gene_embed = nn.Linear(1, token_dim)  # Each gene → token
        
        # GRN structure
        self.grn_adjacency = grn_adjacency  # (num_genes, num_genes) sparse
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(token_dim, num_heads=8, adjacency=grn_adjacency)
            for _ in range(num_layers)
        ])
        
        # Time embedding
        self.time_embed = TimestepEmbedding(token_dim)
        
        # Output projection
        self.output_proj = nn.Linear(token_dim, 1)
    
    def forward(self, x, t, condition=None):
        """
        Args:
            x: Gene expression (batch, num_genes)
            t: Timesteps (batch,)
            condition: Optional conditioning
        
        Returns:
            x_out: Predicted expression (batch, num_genes)
        """
        batch_size = x.shape[0]
        
        # Embed genes to tokens
        x_tokens = self.gene_embed(x.unsqueeze(-1))  # (batch, num_genes, token_dim)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, token_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, token_dim)
        
        # Add time to all tokens
        x_tokens = x_tokens + t_emb
        
        # Graph attention layers (structured by GRN)
        for layer in self.gat_layers:
            x_tokens = layer(x_tokens)  # (batch, num_genes, token_dim)
        
        # Project back to gene space
        x_out = self.output_proj(x_tokens).squeeze(-1)  # (batch, num_genes)
        
        return x_out


class GraphAttentionLayer(nn.Module):
    def __init__(self, token_dim, num_heads=8, adjacency=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.adjacency = adjacency  # Sparse mask
        
        self.qkv = nn.Linear(token_dim, token_dim * 3)
        self.proj = nn.Linear(token_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)
    
    def forward(self, x):
        """
        Sparse attention based on GRN structure.
        
        Args:
            x: (batch, num_genes, token_dim)
        
        Returns:
            out: (batch, num_genes, token_dim)
        """
        batch_size, num_genes, token_dim = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (batch, num_genes, token_dim * 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, num_genes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_genes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_genes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply GRN mask (only attend to neighbors)
        if self.adjacency is not None:
            mask = self.adjacency.unsqueeze(0).unsqueeze(0)  # (1, 1, num_genes, num_genes)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, num_heads, num_genes, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, num_genes, token_dim)
        
        # Output projection
        out = self.proj(out)
        
        # Residual + norm
        out = self.norm(x + out)
        
        return out
```

### Why It's Good

**1. Mechanistic flavor**
- Respects known regulatory relationships
- More biologically plausible
- Better for causal reasoning

**2. Better inductive bias for perturbations**
- Perturbations propagate through GRN
- Model learns regulatory logic
- More accurate predictions for unseen perturbations

**3. Computational efficiency**
- Sparse attention: O(num_edges) instead of O(n²)
- Typical GRN: ~100K edges for 20K genes
- Much faster than full attention

**4. Interpretability**
- Can trace predictions through regulatory paths
- Identify key regulators
- Explain perturbation effects mechanistically

### GRN Sources

| Source | Coverage | Quality | Use Case |
|--------|----------|---------|----------|
| **STRING** | Broad | Moderate | General purpose |
| **RegNetwork** | Human TFs | High | Transcriptional regulation |
| **SCENIC** | Data-driven | Variable | Dataset-specific |
| **ChIP-seq databases** | Validated | High | High-confidence edges |
| **Inferred from data** | Custom | Variable | Novel systems |

**Recommendation**: Start with STRING or RegNetwork, refine with data-driven inference.

---

## Option 4: Rank-Based Sequences (Geneformer Style)

### Concept

**Treat genes as sequence** by ranking by expression level.

```
Gene expression → Rank genes → Sequence of gene tokens → Transformer → Output
```

**Key idea**: Order matters for transformers, so impose ordering via expression level.

### Architecture

```python
class RankBasedDiffusion(nn.Module):
    def __init__(
        self,
        num_genes=20000,
        embed_dim=256,
        num_layers=12,
        num_heads=8,
        max_seq_len=2000  # Truncate to top-k genes
    ):
        super().__init__()
        
        # Gene embeddings (learned for each gene)
        self.gene_embed = nn.Embedding(num_genes, embed_dim)
        
        # Value embeddings (expression level)
        self.value_embed = nn.Linear(1, embed_dim)
        
        # Positional encoding (rank position)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4),
            num_layers=num_layers
        )
        
        # Output: predict expression for each gene
        self.output_proj = nn.Linear(embed_dim, 1)
        
        self.max_seq_len = max_seq_len
    
    def forward(self, x, t, condition=None):
        """
        Args:
            x: Gene expression (batch, num_genes)
            t: Timesteps (batch,)
        
        Returns:
            x_out: Predicted expression (batch, num_genes)
        """
        batch_size, num_genes = x.shape
        
        # Rank genes by expression
        sorted_indices = torch.argsort(x, dim=1, descending=True)  # (batch, num_genes)
        sorted_values = torch.gather(x, 1, sorted_indices)  # (batch, num_genes)
        
        # Truncate to top-k
        sorted_indices = sorted_indices[:, :self.max_seq_len]
        sorted_values = sorted_values[:, :self.max_seq_len]
        
        # Gene embeddings
        gene_emb = self.gene_embed(sorted_indices)  # (batch, max_seq_len, embed_dim)
        
        # Value embeddings
        value_emb = self.value_embed(sorted_values.unsqueeze(-1))  # (batch, max_seq_len, embed_dim)
        
        # Combine
        tokens = gene_emb + value_emb + self.pos_embed  # (batch, max_seq_len, embed_dim)
        
        # Transformer
        tokens = tokens.transpose(0, 1)  # (max_seq_len, batch, embed_dim)
        tokens_out = self.transformer(tokens)  # (max_seq_len, batch, embed_dim)
        tokens_out = tokens_out.transpose(0, 1)  # (batch, max_seq_len, embed_dim)
        
        # Predict expression
        expr_pred = self.output_proj(tokens_out).squeeze(-1)  # (batch, max_seq_len)
        
        # Unsort to original gene order
        x_out = torch.zeros(batch_size, num_genes, device=x.device)
        x_out.scatter_(1, sorted_indices, expr_pred)
        
        return x_out
```

### Why It's Good (and Not So Good)

**Pros**:
- Works empirically (Geneformer shows this)
- Natural for transformers (sequence processing)
- Can capture expression-dependent relationships

**Cons**:
- **Ordering artifacts**: Genes with same expression get arbitrary order
- **Scaling issues**: 20K sequence length is expensive
- **Truncation**: Top-k genes loses information
- **Not biologically motivated**: Ranking is artificial

**When to use**: 
- When you have lots of data (Geneformer trained on millions of cells)
- When interpretability is less important
- When other methods fail

---

## Comparison and Recommendations

### Summary Table

| Approach | Tokens | Compute | Interpretability | Biology | Transfer | Best For |
|----------|--------|---------|------------------|---------|----------|----------|
| **Latent tokens** | 32-128 | Low | Moderate | Learned | Good | Default choice |
| **Pathway tokens** | 50-500 | Low | High | Strong | Excellent | Clinical applications |
| **GRN-aware** | 20K (sparse) | Moderate | High | Strong | Moderate | Perturbation modeling |
| **Rank-based** | 2K-20K | High | Low | Weak | Poor | Large-scale pretraining |

### Decision Tree

```
Do you need interpretability?
├─ Yes → Pathway tokens or GRN-aware
│   ├─ Clinical application → Pathway tokens
│   └─ Perturbation prediction → GRN-aware
└─ No → Latent tokens or Rank-based
    ├─ Limited compute → Latent tokens
    └─ Massive data → Rank-based
```

### Recommended Starting Point

**For most applications**:
1. Start with **latent tokens** (Option 1)
2. Use 64 tokens, 256 dimensions
3. Train with rectified flow (simple objective)
4. Evaluate on your task

**If interpretability matters**:
1. Use **pathway tokens** (Option 2)
2. Start with MSigDB Hallmark (50 pathways)
3. Scale to Reactome if needed
4. Validate pathway-level predictions

**If modeling perturbations**:
1. Use **GRN-aware** (Option 3)
2. Start with STRING or RegNetwork
3. Refine with data-driven edges
4. Validate on held-out perturbations

---

## Hybrid Approaches

### Combining Multiple Strategies

**Latent + Pathway**:
```python
# Encoder produces pathway-structured latents
encoder_pathway = PathwayEncoder(pathways)
z_pathway = encoder_pathway(x)  # (batch, num_pathways, token_dim)

# Transformer on pathway tokens
z_out = transformer(z_pathway, t)

# Decoder with pathway structure
x_out = decoder_pathway(z_out)
```

**GRN + Latent**:
```python
# Encode to latent
z = encoder(x)  # (batch, num_tokens, token_dim)

# Graph attention on latent tokens (with learned adjacency)
z_out = graph_transformer(z, learned_adjacency)

# Decode
x_out = decoder(z_out)
```

---

## Implementation Recommendations

### Training Tips

**1. Start simple**
```python
# Begin with latent tokens, small model
model = LatentTokenDiffusion(
    num_genes=20000,
    num_tokens=64,
    token_dim=256,
    num_layers=8
)
```

**2. Validate tokenization quality**
```python
# Check reconstruction
z = model.encode(x)
x_recon = model.decode(z)
recon_error = F.mse_loss(x_recon, x)
print(f"Reconstruction error: {recon_error:.4f}")
```

**3. Visualize learned tokens**
```python
# For latent tokens: PCA on token activations
z = model.encode(x_batch)  # (batch, num_tokens, token_dim)
z_flat = z.mean(dim=0)  # (num_tokens, token_dim)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_flat.detach().cpu())

plt.scatter(z_pca[:, 0], z_pca[:, 1])
plt.title("Learned token structure")
```

**4. Compare to baselines**
```python
# Baseline: Simple MLP
baseline = SimpleMLP(num_genes=20000, hidden_dim=256)

# Your model
model = LatentTokenDiffusion(...)

# Compare on held-out data
baseline_loss = evaluate(baseline, test_data)
model_loss = evaluate(model, test_data)
print(f"Improvement: {(baseline_loss - model_loss) / baseline_loss * 100:.1f}%")
```

---

## Key Takeaways

### Conceptual

1. **Gene expression isn't tabular** — it has structure that should inform architecture
2. **Tokenization = architectural choice** — determines what model can learn
3. **Multiple valid approaches** — choose based on your goals
4. **Biology can help** — incorporating known structure improves models

### Practical

1. **Default: Latent tokens** — learned, flexible, compute-efficient
2. **Interpretability: Pathway tokens** — biologically meaningful, clinically legible
3. **Perturbations: GRN-aware** — mechanistic, better generalization
4. **Large-scale: Rank-based** — works but heavy and less principled

### Research Directions

1. **Optimal number of tokens** — how many latent tokens are needed?
2. **Token interpretability** — can we make latent tokens biologically meaningful?
3. **Hybrid approaches** — combining multiple tokenization strategies
4. **Transfer learning** — do tokens transfer across datasets/species?

---

## Related Documents

- [02_ddpm_training.md](02_ddpm_training.md) — General DDPM training (includes simple MLP baseline)
- [DiT Foundations](../DiT/01_dit_foundations.md) — Transformer architecture details
- [Tokenization Research](../DiT/open_research_tokenization.md) — Deep dive on tokenization challenges

---

## References

**Latent diffusion**:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"

**Gene expression models**:
- Theodoris et al. (2023): "Transfer learning enables predictions in network biology" (Geneformer)
- Cui et al. (2024): "scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics"
- Lotfollahi et al. (2023): "Predicting cellular responses to novel drug combinations"

**Pathway analysis**:
- Subramanian et al. (2005): "Gene set enrichment analysis" (GSEA)
- Liberzon et al. (2015): "The Molecular Signatures Database (MSigDB)"

**Gene regulatory networks**:
- Szklarczyk et al. (2021): "The STRING database" 
- Aibar et al. (2017): "SCENIC: single-cell regulatory network inference"
