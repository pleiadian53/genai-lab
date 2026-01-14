# DiT Foundations: Architecture and Components

This document provides detailed coverage of the Diffusion Transformer (DiT) architecture, explaining each component and design choice.

**Prerequisites**: Understanding of [rectified flow](../flow_matching/01_flow_matching_foundations.md) and basic [Transformer architecture](https://arxiv.org/abs/1706.03762).

---

## Overview

DiT replaces convolutional U-Nets with Transformers for diffusion/flow-based generative modeling. The key architectural shift:

```
U-Net: Spatial grids + Local convolutions + Hierarchical downsampling
DiT:   Token sequences + Global attention + Flat architecture
```

**Core components**:
1. Tokenization (input → tokens)
2. Positional encoding (preserve structure)
3. Time conditioning (AdaLN)
4. Transformer blocks (attention + MLP)
5. Output projection (tokens → predictions)

---

## 1. Input Tokenization

### 1.1 Patchification (Images)

**Goal**: Convert image to sequence of tokens

**Process**:
```python
# Input: image of shape (B, C, H, W)
# Example: (batch_size, 3, 256, 256)

# Step 1: Split into patches
patch_size = 16
num_patches = (H // patch_size) * (W // patch_size)  # 256 patches for 256×256

# Step 2: Reshape
# (B, C, H, W) → (B, num_patches, patch_size², C)
patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
patches = patches.contiguous().view(B, C, num_patches, -1)
patches = patches.permute(0, 2, 3, 1)  # (B, num_patches, patch_dim, C)

# Step 3: Flatten each patch
# (B, num_patches, patch_size * patch_size * C)
patches = patches.reshape(B, num_patches, -1)
```

**Result**: Image → sequence of 256 tokens (for 16×16 patches on 256×256 image)

### 1.2 Patch Embedding

**Linear projection** to model dimension:

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        
        # Linear projection
        patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(patch_dim, embed_dim)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Patchify
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, self.patch_size * self.patch_size * C)
        
        # Project to embedding dimension
        x = self.proj(x)  # (B, num_patches, embed_dim)
        return x
```

**Dimensions**:
- Input: `(B, 3, 256, 256)`
- After patchify: `(B, 256, 768)` where 768 = 16×16×3
- After projection: `(B, 256, embed_dim)`

### 1.3 Tokenization for Other Modalities

**Gene expression**:
```python
# Option 1: Direct embedding (no explicit tokens)
z = nn.Linear(num_genes, embed_dim)(gene_expression)

# Option 2: Gene-level tokens
gene_tokens = [embed(gene_i) for gene_i in genes]

# Option 3: Module-level tokens
module_tokens = [embed(module_j) for module_j in pathways]
```

**See**: [open_research_tokenization.md](open_research_tokenization.md) for detailed discussion.

---

## 2. Positional Encoding

### 2.1 Why Positional Encoding?

Transformers are **permutation-invariant** — they treat input as a set, not a sequence.

For images, **spatial structure matters**:
- Top-left patch vs bottom-right patch
- Neighboring patches are related
- Absolute and relative positions

**Solution**: Add positional information to patch embeddings.

### 2.2 Types of Positional Encoding

**Learned Positional Embeddings** (most common for DiT):

```python
class DiT(nn.Module):
    def __init__(self, num_patches=256, embed_dim=768):
        super().__init__()
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        x = x + self.pos_embed  # Broadcast across batch
        return x
```

**Sinusoidal Positional Encoding** (from original Transformer):

```python
def sinusoidal_position_embedding(num_patches, embed_dim):
    """Generate sinusoidal position embeddings."""
    position = torch.arange(num_patches).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * 
                         -(math.log(10000.0) / embed_dim))
    
    pos_embed = torch.zeros(num_patches, embed_dim)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    
    return pos_embed
```

**2D Positional Encoding** (for images):

```python
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 2D sinusoidal position embeddings.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Image size in patches (e.g., 16 for 256×256 with 16×16 patches)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed
```

**Trade-offs**:

| Type | Pros | Cons |
|------|------|------|
| **Learned** | Flexible, adapts to data | Doesn't generalize to different sizes |
| **Sinusoidal** | Generalizes to longer sequences | Fixed pattern |
| **2D** | Respects image structure | More complex |

**DiT standard**: Learned positional embeddings.

---

## 3. Time Conditioning via AdaLN

### 3.1 The Time Conditioning Problem

Diffusion models are **time-dependent**: behavior must change based on noise level.

**At t=0** (clean data): Refine details
**At t=1** (pure noise): Generate coarse structure

**Challenge**: How to inject time information into every layer?

### 3.2 Adaptive Layer Normalization (AdaLN)

**Standard LayerNorm**:
$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

where $\gamma$, $\beta$ are **learned parameters** (same for all timesteps).

**Adaptive LayerNorm**:
$$
\text{AdaLN}(x, t) = \gamma(t) \cdot \frac{x - \mu}{\sigma} + \beta(t)
$$

where $\gamma(t)$, $\beta(t)$ are **functions of time**.

**Key insight**: Time controls the normalization parameters at every layer.

### 3.3 Implementation

```python
class AdaLN(nn.Module):
    def __init__(self, embed_dim, time_embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        
        # MLP to produce scale and shift from time embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * embed_dim)
        )
    
    def forward(self, x, t_emb):
        # x: (B, N, embed_dim) - token features
        # t_emb: (B, time_embed_dim) - time embedding
        
        # Normalize
        x_norm = self.norm(x)
        
        # Get scale and shift from time
        scale, shift = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # (B, 1, embed_dim)
        shift = shift.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Modulate
        x_modulated = scale * x_norm + shift
        
        return x_modulated
```

**Flow**:
```
Time t → TimeEmbed(t) → MLP → (γ(t), β(t)) → Modulate features
```

### 3.4 Time Embedding

**Sinusoidal time embedding** (similar to positional encoding):

```python
class TimestepEmbedding(nn.Module):
    def __init__(self, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        # MLP to process sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
    
    def forward(self, t):
        # t: (B,) - timesteps
        
        # Sinusoidal encoding
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Process through MLP
        emb = self.mlp(emb)
        
        return emb
```

**Why sinusoidal?**
- Smooth, continuous representation
- Similar timesteps → similar embeddings
- Well-studied in Transformers

**See**: [time_embeddings_explained.md](../diffusion/DiT/time_embeddings_explained.md) for detailed explanation.

---

## 4. Transformer Blocks

### 4.1 DiT Block Architecture

**Standard Transformer block**:
```
x → LayerNorm → Self-Attention → Add → LayerNorm → MLP → Add → output
```

**DiT block with AdaLN**:
```
x → AdaLN(·, t) → Self-Attention → Add → AdaLN(·, t) → MLP → Add → output
```

**Key difference**: Time-dependent normalization at every step.

### 4.2 Implementation

```python
class DiTBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        # Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
        # AdaLN for attention
        self.adaLN_attn = AdaLN(embed_dim, time_embed_dim=256)
        
        # AdaLN for MLP
        self.adaLN_mlp = AdaLN(embed_dim, time_embed_dim=256)
    
    def forward(self, x, t_emb):
        # x: (B, N, embed_dim)
        # t_emb: (B, time_embed_dim)
        
        # Attention block
        x_norm = self.adaLN_attn(x, t_emb)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP block
        x_norm = self.adaLN_mlp(x, t_emb)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x
```

### 4.3 Self-Attention Mechanism

**Multi-head self-attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q = XW_Q$ (queries)
- $K = XW_K$ (keys)
- $V = XW_V$ (values)

**Multi-head**:
```python
# Split into multiple heads
Q = Q.view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
K = K.view(B, N, num_heads, head_dim).transpose(1, 2)
V = V.view(B, N, num_heads, head_dim).transpose(1, 2)

# Attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
attn = F.softmax(scores, dim=-1)

# Apply attention to values
out = torch.matmul(attn, V)  # (B, num_heads, N, head_dim)

# Concatenate heads
out = out.transpose(1, 2).contiguous().view(B, N, embed_dim)
```

**Complexity**: $O(N^2 \cdot d)$ where $N$ is sequence length, $d$ is dimension.

**For images**: $N = (H/p)^2$ where $p$ is patch size.

### 4.4 MLP (Feed-Forward Network)

**Standard two-layer MLP**:

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
```

**Typical ratio**: `hidden_features = 4 × in_features`

**Purpose**: Add non-linearity and capacity after attention.

---

## 5. Conditioning Mechanisms

### 5.1 Time Conditioning (Covered Above)

Via AdaLN: $\gamma(t)$, $\beta(t)$ modulate features.

### 5.2 Class Conditioning

**For class-conditional generation** (e.g., ImageNet):

```python
class DiTWithClassConditioning(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=768):
        super().__init__()
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        # Combine with time embedding
        self.time_class_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x, t, class_labels):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Class embedding
        c_emb = self.class_embed(class_labels)
        
        # Combine
        tc_emb = torch.cat([t_emb, c_emb], dim=-1)
        tc_emb = self.time_class_mlp(tc_emb)
        
        # Use tc_emb for AdaLN
        for block in self.blocks:
            x = block(x, tc_emb)
        
        return x
```

### 5.3 Cross-Attention Conditioning

**For text-to-image** (like Stable Diffusion):

```python
class DiTBlockWithCrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Cross-attention (attend to text)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # MLP
        self.mlp = MLP(embed_dim, embed_dim * 4, embed_dim)
        
        # AdaLN
        self.adaLN_self = AdaLN(embed_dim, time_embed_dim=256)
        self.adaLN_cross = AdaLN(embed_dim, time_embed_dim=256)
        self.adaLN_mlp = AdaLN(embed_dim, time_embed_dim=256)
    
    def forward(self, x, t_emb, context):
        # x: image tokens
        # context: text tokens
        
        # Self-attention
        x_norm = self.adaLN_self(x, t_emb)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        
        # Cross-attention
        x_norm = self.adaLN_cross(x, t_emb)
        x = x + self.cross_attn(x_norm, context, context)[0]
        
        # MLP
        x_norm = self.adaLN_mlp(x, t_emb)
        x = x + self.mlp(x_norm)
        
        return x
```

**Use cases**:
- Text-to-image: Text tokens as context
- Perturbation modeling: Perturbation embeddings as context
- Multi-modal: Any conditioning modality

---

## 6. Output Projection

### 6.1 From Tokens to Predictions

**Goal**: Map token representations back to target space.

**For rectified flow** (velocity prediction):

```python
class DiTOutput(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, out_channels=3):
        super().__init__()
        
        # Linear projection
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
        self.patch_size = patch_size
        self.out_channels = out_channels
    
    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        
        # Project to patch space
        x = self.proj(x)  # (B, num_patches, patch_size² * out_channels)
        
        # Reshape to image
        B, N, _ = x.shape
        H = W = int(math.sqrt(N))
        
        x = x.reshape(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, H, p, W, p)
        x = x.reshape(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        
        return x
```

**Dimensions**:
- Input: `(B, 256, 768)` (256 tokens, 768-dim)
- After projection: `(B, 256, 768)` where 768 = 16×16×3
- After reshape: `(B, 3, 256, 256)` (image)

### 6.2 Final Layer Normalization

**Optional**: Apply final AdaLN before output projection:

```python
def forward(self, x, t_emb):
    # Process through transformer blocks
    for block in self.blocks:
        x = block(x, t_emb)
    
    # Final AdaLN
    x = self.final_adaLN(x, t_emb)
    
    # Output projection
    x = self.output_proj(x)
    
    return x
```

---

## 7. Complete DiT Architecture

### 7.1 Full Model

```python
class DiT(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=None
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Time embedding
        self.time_embed = TimestepEmbedding(embed_dim)
        
        # Class embedding (optional)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, embed_dim)
        else:
            self.class_embed = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_adaLN = AdaLN(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = DiTOutput(embed_dim, patch_size, in_channels)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_attn.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_attn.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, t, y=None):
        # x: (B, C, H, W) - noisy images
        # t: (B,) - timesteps
        # y: (B,) - class labels (optional)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Class embedding (if provided)
        if y is not None and self.class_embed is not None:
            c_emb = self.class_embed(y)
            t_emb = t_emb + c_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)
        
        # Final layer
        x = self.final_adaLN(x, t_emb)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
```

### 7.2 Model Sizes

**DiT variants** (following ViT naming):

| Model | Depth | Hidden Dim | Heads | Params |
|-------|-------|------------|-------|--------|
| DiT-S/2 | 12 | 384 | 6 | 33M |
| DiT-B/2 | 12 | 768 | 12 | 130M |
| DiT-L/2 | 24 | 1024 | 16 | 458M |
| DiT-XL/2 | 28 | 1152 | 16 | 675M |

**Notation**: DiT-{Size}/{Patch size}
- DiT-XL/2: Extra-large model, 2×2 patches
- DiT-XL/8: Extra-large model, 8×8 patches

**Trade-off**: Smaller patches = more tokens = more compute, but better quality.

---

## 8. Design Choices and Ablations

### 8.1 Patch Size

**Impact on performance**:

| Patch Size | Tokens (256×256) | FID Score | Speed |
|------------|------------------|-----------|-------|
| 2×2 | 16,384 | Best | Slowest |
| 4×4 | 4,096 | Good | Moderate |
| 8×8 | 1,024 | Moderate | Fast |
| 16×16 | 256 | Worse | Fastest |

**Recommendation**: 
- High quality: 2×2 or 4×4
- Balanced: 8×8
- Fast: 16×16

### 8.2 Model Depth vs Width

**Depth** (number of layers):
- More depth = better long-range dependencies
- DiT-XL uses 28 layers

**Width** (embedding dimension):
- More width = more capacity per layer
- DiT-XL uses 1152 dimensions

**Empirical finding**: Depth matters more than width for DiT.

### 8.3 AdaLN vs Other Conditioning

**Alternatives tested**:
1. **Concatenation**: Concat time to input (worse)
2. **FiLM**: Similar to AdaLN (comparable)
3. **Cross-attention**: More expensive (slight improvement)

**Winner**: AdaLN (best trade-off of performance and efficiency)

### 8.4 Positional Encoding

**Learned vs Sinusoidal**:
- Learned: Slightly better for fixed resolution
- Sinusoidal: Better for variable resolution

**2D vs 1D**:
- 2D: Respects image structure (slightly better)
- 1D: Simpler (minimal difference)

**DiT default**: Learned 1D (simplicity wins)

---

## 9. Comparison with U-Net

### 9.1 Architectural Differences

| Aspect | U-Net | DiT |
|--------|-------|-----|
| **Structure** | Hierarchical (encoder-decoder) | Flat (uniform depth) |
| **Operations** | Convolutions | Self-attention |
| **Receptive field** | Local → Global (via depth) | Global from start |
| **Skip connections** | Between encoder-decoder | Residual within blocks |
| **Time conditioning** | Concatenation or FiLM | AdaLN |
| **Inductive bias** | Spatial locality | None (learned) |

### 9.2 When to Use Which

**U-Net advantages**:
- Faster (convolutions are efficient)
- Lower memory
- Strong spatial inductive bias
- Proven for images

**DiT advantages**:
- Better scaling (to larger models)
- Flexible conditioning
- Modality-agnostic
- State-of-the-art results

**Recommendation**:
- Images, fixed size, limited compute: U-Net
- Large-scale, multi-modal, flexible: DiT

---

## 10. Implementation Tips

### 10.1 Memory Optimization

**Gradient checkpointing**:
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x, t_emb):
    for block in self.blocks:
        x = checkpoint(block, x, t_emb, use_reentrant=False)
    return x
```

**Flash Attention**:
```python
from flash_attn import flash_attn_func

# Replace standard attention with flash attention
attn_out = flash_attn_func(q, k, v, causal=False)
```

**Mixed precision**:
```python
from torch.cuda.amp import autocast

with autocast():
    output = model(x, t)
```

### 10.2 Training Stability

**Layer scale**:
```python
class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, init_values=1e-4):
        super().__init__()
        self.gamma_1 = nn.Parameter(init_values * torch.ones(embed_dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(embed_dim))
    
    def forward(self, x, t_emb):
        x = x + self.gamma_1 * self.attn(self.adaLN_attn(x, t_emb))
        x = x + self.gamma_2 * self.mlp(self.adaLN_mlp(x, t_emb))
        return x
```

**Gradient clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 10.3 Efficient Inference

**Compile model** (PyTorch 2.0+):
```python
model = torch.compile(model)
```

**Batch inference**:
```python
# Generate multiple samples in parallel
batch_size = 16
x = torch.randn(batch_size, 3, 256, 256)
samples = sampler.sample(x, num_steps=50)
```

---

## Key Takeaways

### Architectural

1. **Tokenization**: Images → patches → embeddings
2. **Positional encoding**: Preserve spatial structure
3. **AdaLN**: Time-dependent normalization
4. **Self-attention**: Global dependencies
5. **Output projection**: Tokens → predictions

### Design Choices

1. **Patch size**: Trade-off between quality and speed
2. **Model size**: Depth matters more than width
3. **Conditioning**: AdaLN is efficient and effective
4. **Positional encoding**: Learned works well

### Practical

1. **Memory**: Use gradient checkpointing and flash attention
2. **Stability**: Layer scale and gradient clipping
3. **Speed**: Compile model, batch inference
4. **Quality**: Smaller patches, larger models

---

## Related Documents

- [00_dit_overview.md](00_dit_overview.md) — High-level introduction
- [02_dit_training.md](02_dit_training.md) — Training pipeline
- [03_dit_sampling.md](03_dit_sampling.md) — Sampling strategies
- [time_embeddings_explained.md](../diffusion/DiT/time_embeddings_explained.md) — Deep dive on time conditioning

---

## References

- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers"
- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words" (ViT)
- Vaswani et al. (2017): "Attention is All You Need"
- Perez et al. (2018): "FiLM: Visual Reasoning with a General Conditioning Layer"
