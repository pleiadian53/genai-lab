# Advanced Score Network Architectures for Realistic Data

## Overview

This document explores **advanced neural network architectures** for score-based diffusion models when working with realistic, complex data such as:
- **Medical imaging** (CT, MRI, pathology slides)
- **Gene expression data** (bulk RNA-seq, microarray)
- **Single-cell RNA-seq (scRNA-seq)** data
- **High-resolution natural images**

The simple MLP used in the tutorial notebook works well for 2D toy data, but realistic datasets require more sophisticated architectures.

---

## Referenced From

- **Notebook**: [`notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb`](../../../notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb)
- **Related**: [Time Embedding and FiLM](./time_embedding_and_film.md) — Component details for time conditioning

---

## Why Simple MLPs Are Not Enough

### The Tutorial Architecture

The notebook uses a simple MLP:

```python
class SimpleScoreNetwork(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, time_dim=32):
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
```

**This works for 2D toy data because**:

- Only 2 dimensions to model
- Simple distributions (e.g., mixture of Gaussians)
- No spatial structure to capture

### Limitations for Realistic Data

| Data Type | Challenge | Why MLP Fails |
|-----------|-----------|---------------|
| **Images** | Spatial correlations, local patterns | MLPs treat pixels independently |
| **Gene expression** | High dimensionality (~20,000 genes) | Too many parameters, no structure |
| **scRNA-seq** | Sparse, high-dimensional, cell-type structure | Can't leverage biological priors |
| **Medical imaging** | Multi-scale features, fine details | No hierarchical feature extraction |

---

## Architecture 1: U-Net for Images

### Why U-Net?

U-Net is the **dominant architecture** for image diffusion models (DDPM, Stable Diffusion) because:

1. **Multi-scale processing**: Encoder-decoder structure captures features at multiple resolutions
2. **Skip connections**: Preserve fine details while processing global structure
3. **Fully convolutional**: Handles arbitrary image sizes
4. **Proven effectiveness**: State-of-the-art in image generation

### Architecture Overview

```
Input Image (H×W×C) + Time Embedding
        ↓
    ┌───────────────┐
    │   Encoder     │  (Downsample: capture global structure)
    │   Block 1     │ ────────────────┐
    └───────────────┘                 │ Skip
        ↓                             │
    ┌───────────────┐                 │
    │   Encoder     │                 │
    │   Block 2     │ ────────────┐   │
    └───────────────┘             │   │
        ↓                         │   │
    ┌───────────────┐             │   │
    │   Bottleneck  │             │   │
    └───────────────┘             │   │
        ↓                         │   │
    ┌───────────────┐             │   │
    │   Decoder     │ ←───────────┘   │
    │   Block 1     │                 │
    └───────────────┘                 │
        ↓                             │
    ┌───────────────┐                 │
    │   Decoder     │ ←───────────────┘
    │   Block 2     │
    └───────────────┘
        ↓
    Output Score (H×W×C)
```

### Key Components

#### 1. Residual Blocks with Time Conditioning

```python
class ResBlock(nn.Module):
    """Residual block with time conditioning via FiLM."""
    
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time conditioning (FiLM)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * out_channels)  # scale and shift
        )
        
        # Skip connection if dimensions change
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Apply time conditioning
        scale, shift = self.time_mlp(time_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]  # Broadcast to spatial dims
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + self.skip(x)
```

#### 2. Attention Layers

For capturing long-range dependencies (especially important for large images):

```python
class SelfAttention(nn.Module):
    """Self-attention for spatial features."""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Reshape to sequence: (batch, h*w, channels)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Apply attention
        x_norm = self.norm(x).flatten(2).transpose(1, 2)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        
        # Reshape back
        out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        
        return x + out
```

### Complete U-Net Skeleton

```python
class UNet(nn.Module):
    """U-Net for score estimation in diffusion models."""
    
    def __init__(self, in_channels=3, base_channels=64, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Encoder
        self.enc1 = ResBlock(in_channels, base_channels, time_dim)
        self.enc2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 4, time_dim),
            SelfAttention(base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4, time_dim)
        )
        
        # Decoder (with skip connections)
        self.dec3 = ResBlock(base_channels * 8, base_channels * 2, time_dim)  # *8 from concat
        self.dec2 = ResBlock(base_channels * 4, base_channels, time_dim)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_dim)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h3 = self.enc3(self.down2(h2), t_emb)
        
        # Bottleneck
        h = self.bottleneck(h3)
        
        # Decoder with skip connections
        h = self.up2(h)
        h = self.dec3(torch.cat([h, h2], dim=1), t_emb)
        h = self.up1(h)
        h = self.dec2(torch.cat([h, h1], dim=1), t_emb)
        h = self.dec1(h, t_emb)
        
        return self.out(h)
```

### Medical Imaging Considerations

For medical imaging (CT, MRI, pathology):

1. **3D U-Net**: Replace 2D convolutions with 3D for volumetric data
2. **Higher resolution**: More downsampling stages for high-res pathology
3. **Domain-specific augmentation**: Intensity normalization, organ-specific priors
4. **Conditional generation**: Condition on patient metadata, modality, etc.

```python
# 3D convolution for volumetric medical imaging
self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
```

---

## Architecture 2: Vision Transformer (DiT)

### Why Vision Transformers?

The **Diffusion Transformer (DiT)** architecture shows that pure Transformers can match or exceed U-Net:

1. **Scalability**: Performance scales predictably with model size
2. **Simplicity**: Fewer inductive biases than CNNs
3. **Long-range attention**: Global receptive field from the start
4. **Transfer learning**: Leverage pretrained Vision Transformers

### Architecture Overview

```
Input Image → Patchify → Linear Projection
                             ↓
                      Positional Encoding
                             ↓
              ┌──────────────────────────────┐
              │     Transformer Block 1      │ ← Time/Class Conditioning
              │  (Self-Attention + FFN)      │   (via AdaLN)
              └──────────────────────────────┘
                             ↓
              ┌──────────────────────────────┐
              │     Transformer Block 2      │ ← Time/Class Conditioning
              └──────────────────────────────┘
                             ↓
                            ...
                             ↓
              ┌──────────────────────────────┐
              │     Transformer Block N      │ ← Time/Class Conditioning
              └──────────────────────────────┘
                             ↓
                      Linear Projection
                             ↓
                      Unpatchify → Score
```

### Key Innovation: Adaptive Layer Normalization (AdaLN)

DiT conditions on time using **Adaptive Layer Normalization**:

```python
class AdaLN(nn.Module):
    """Adaptive Layer Normalization for DiT."""
    
    def __init__(self, hidden_dim, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * hidden_dim)  # scale, shift, gate for 2 norms
        )
    
    def forward(self, x, condition):
        # Get modulation parameters
        params = self.adaLN_modulation(condition)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
```

### DiT Block

```python
class DiTBlock(nn.Module):
    """Diffusion Transformer block with AdaLN conditioning."""
    
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, condition_dim=256):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim)
        )
        
        # AdaLN modulation
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * hidden_dim)
        )
    
    def forward(self, x, condition):
        # Get modulation parameters
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(condition).chunk(6, dim=-1)
        
        # Attention with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate1.unsqueeze(1) * h
        
        # MLP with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h
        
        return x
```

### When to Use DiT vs U-Net

| Aspect | U-Net | DiT |
|--------|-------|-----|
| **Inductive bias** | Strong (locality, hierarchy) | Weak (learns structure) |
| **Data efficiency** | Better with less data | Needs more data |
| **Scaling** | Saturates earlier | Scales well |
| **Compute** | More efficient for small models | More efficient at scale |
| **Best for** | Small-medium datasets | Large-scale training |

---

## Architecture 3: Networks for Tabular/Biological Data

### Gene Expression Data

Gene expression data (bulk RNA-seq, microarray) is:
- **High-dimensional**: ~20,000 genes
- **Tabular**: No spatial structure
- **Structured**: Gene pathways, co-expression patterns

#### Architecture Options

**Option 1: Deep MLP with Residual Connections**

```python
class GeneExpressionScoreNet(nn.Module):
    """Score network for gene expression data."""
    
    def __init__(self, n_genes=20000, hidden_dim=2048, n_layers=6, time_dim=128):
        super().__init__()
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Initial projection
        self.input_proj = nn.Linear(n_genes, hidden_dim)
        
        # Deep residual blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, time_dim) 
            for _ in range(n_layers)
        ])
        
        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_genes)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        
        return self.output_proj(h)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.time_proj = nn.Linear(time_dim, hidden_dim * 2)  # FiLM
    
    def forward(self, x, t_emb):
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        
        h = self.norm(x)
        h = h * (1 + scale) + shift
        h = self.mlp(h)
        
        return x + h
```

**Option 2: Graph Neural Network (for pathway structure)**

If gene-gene interaction networks are available:

```python
class PathwayAwareScoreNet(nn.Module):
    """Score network using gene pathway structure."""
    
    def __init__(self, n_genes, hidden_dim, adjacency_matrix, time_dim=128):
        super().__init__()
        
        # Register adjacency as buffer (not parameter)
        self.register_buffer('adj', adjacency_matrix)
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.node_embed = nn.Linear(1, hidden_dim)
        
        # Graph convolution layers
        self.gcn_layers = nn.ModuleList([
            GraphConvWithTime(hidden_dim, time_dim)
            for _ in range(4)
        ])
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, t):
        # x: (batch, n_genes)
        t_emb = self.time_embed(t)
        
        # Treat each gene as a node
        h = self.node_embed(x.unsqueeze(-1))  # (batch, n_genes, hidden)
        
        for gcn in self.gcn_layers:
            h = gcn(h, self.adj, t_emb)
        
        return self.output(h).squeeze(-1)  # (batch, n_genes)
```

### scRNA-seq Data

Single-cell RNA-seq data has additional challenges:
- **Sparse**: Many zero counts (dropout)
- **Cell-type structure**: Cells cluster by type
- **Batch effects**: Technical variation across experiments

#### Architecture Considerations

```python
class scRNAScoreNet(nn.Module):
    """Score network for scRNA-seq with sparsity handling."""
    
    def __init__(self, n_genes, hidden_dim=512, time_dim=64):
        super().__init__()
        
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Sparse-aware input processing
        self.input_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Attention over genes (learn which genes are important)
        self.gene_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Score prediction
        self.decoder = nn.Sequential(
            ResidualMLPBlock(hidden_dim, time_dim),
            ResidualMLPBlock(hidden_dim, time_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_genes)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        # Encode
        h = self.input_encoder(x)
        
        # Self-attention to learn gene relationships
        h_attended, _ = self.gene_attention(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        h = h + h_attended.squeeze(1)
        
        # Decode with time conditioning
        return self.decoder(h, t_emb)
```

---

## Choosing the Right Architecture

### Decision Framework

```
What is your data type?
    │
    ├── Images (2D)
    │       │
    │       ├── Small dataset (<50K) → U-Net
    │       ├── Large dataset (>1M) → DiT
    │       └── Medical imaging → 2D/3D U-Net with domain priors
    │
    ├── Volumes (3D)
    │       └── 3D U-Net
    │
    ├── Tabular (gene expression, features)
    │       │
    │       ├── No structure → Deep MLP with residual connections
    │       └── Known graph structure → GNN
    │
    └── Sequences
            └── Transformer-based
```

### Practical Recommendations

| Data Type | Recommended Architecture | Key Components |
|-----------|-------------------------|----------------|
| **Natural images** | U-Net or DiT | Time embedding, FiLM/AdaLN, attention |
| **Medical imaging** | 3D U-Net | Domain-specific normalization, high-res handling |
| **Gene expression** | Deep residual MLP | FiLM, layer normalization |
| **scRNA-seq** | MLP + attention | Sparsity handling, gene attention |
| **Pathology slides** | Hierarchical U-Net | Multi-scale, memory-efficient |

---

## Implementation Tips

### 1. Start Simple, Then Scale

```python
# Start with small model for debugging
model_small = UNet(base_channels=32)

# Scale up once working
model_large = UNet(base_channels=256)
```

### 2. Memory Management for Large Data

```python
# Gradient checkpointing for U-Net
def forward_with_checkpointing(self, x, t):
    t_emb = self.time_mlp(t)
    
    # Use checkpointing for encoder blocks
    h1 = torch.utils.checkpoint.checkpoint(self.enc1, x, t_emb)
    h2 = torch.utils.checkpoint.checkpoint(self.enc2, self.down1(h1), t_emb)
    # ... etc
```

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    score = model(x_t, t)
    loss = score_matching_loss(score, noise, std)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Summary

| Architecture | Best For | Key Features |
|--------------|----------|--------------|
| **Simple MLP** | 2D toy data, quick experiments | Concatenation or FiLM |
| **U-Net** | Images, medical imaging | Skip connections, multi-scale, FiLM |
| **DiT** | Large-scale image generation | Transformer, AdaLN, scalable |
| **Deep MLP** | Tabular/gene expression | Residual connections, FiLM |
| **GNN** | Data with graph structure | Pathway-aware, message passing |

The key insight: **match the architecture's inductive biases to your data's structure**.

---

## Further Reading

- **DDPM U-Net**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **DiT**: Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
- **Medical Diffusion**: Kazerouni et al., "Diffusion Models in Medical Imaging" (2023)
- **scRNA Diffusion**: Various recent works on single-cell generation

---

## Related Documents

- [Time Embedding and FiLM](./time_embedding_and_film.md) — Detailed explanation of time conditioning components
- [Score Network Architecture (private)](../../../dev/notebooks/diffusion/02_sde_formulation/score_network_architecture.md) — Implementation notes
- [Training Loss and Denoising](../../../dev/notebooks/diffusion/02_sde_formulation/training_loss_and_denoising.md) — How training works
