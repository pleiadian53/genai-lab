# JEPA Foundations: Architecture and Components

This document covers the detailed architecture of Joint Embedding Predictive Architecture (JEPA), including encoder design, predictor networks, VICReg regularization, and complete PyTorch implementations.

**Prerequisites**: Understanding of [JEPA overview](00_jepa_overview.md), transformers, and self-supervised learning.

---

## Architecture Overview

### High-Level Structure

JEPA consists of three main components:

```
Context x → Encoder f_θ → z_x
                            ↓
                        Predictor g_φ → ẑ_y
                            ↑
Target y → Encoder f_θ → z_y
                            ↓
                    Loss: ||z_y - ẑ_y||² + VICReg
```

**Key principles**:
1. **Shared encoder** — Same encoder for context and target
2. **Predictor** — Learns to predict target embedding from context
3. **No decoder** — Prediction in embedding space only
4. **VICReg regularization** — Prevents collapse

---

## 1. Encoder Architecture

### 1.1 Vision Transformer (ViT) Encoder

**For images/spatial data**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for JEPA.
    
    Args:
        img_size: Input image size (H, W)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
    """
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Images (B, C, H, W)
            mask: Optional mask for patches (B, num_patches)
        
        Returns:
            z: Embeddings (B, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        # Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
```

### 1.2 MLP Encoder for Gene Expression

**For tabular/gene expression data**:

```python
class GeneExpressionEncoder(nn.Module):
    """
    MLP encoder for gene expression data.
    
    Args:
        num_genes: Number of genes
        embed_dim: Embedding dimension
        hidden_dims: List of hidden layer dimensions
        num_tokens: Number of output tokens
    """
    def __init__(
        self,
        num_genes=20000,
        embed_dim=256,
        hidden_dims=[2048, 1024],
        num_tokens=64,
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # MLP layers
        layers = []
        in_dim = num_genes
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        # Final projection to tokens
        layers.append(nn.Linear(in_dim, num_tokens * embed_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Positional embedding for tokens
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )
        
        # Optional: Transformer layers on tokens
        self.token_transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=8)
            for _ in range(4)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: Gene expression (B, num_genes)
        
        Returns:
            z: Token embeddings (B, num_tokens, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Encode to flat representation
        z = self.encoder(x)  # (B, num_tokens * embed_dim)
        
        # Reshape to tokens
        z = z.view(batch_size, self.num_tokens, self.embed_dim)
        
        # Add positional embedding
        z = z + self.pos_embed
        
        # Transformer on tokens
        for block in self.token_transformer:
            z = block(z)
        
        # Final norm
        z = self.norm(z)
        
        return z
```

### 1.3 Encoder Design Choices

**Key decisions**:

| Choice | Options | Recommendation |
|--------|---------|----------------|
| **Architecture** | ViT, CNN, MLP | ViT for images, MLP for gene expression |
| **Depth** | 6-24 layers | 12 for images, 4-8 for gene expression |
| **Embed dim** | 256-1024 | 768 for images, 256-512 for gene expression |
| **Normalization** | LayerNorm, BatchNorm | LayerNorm (more stable) |
| **Activation** | GELU, ReLU | GELU (smoother gradients) |

---

## 2. Predictor Architecture

### 2.1 Transformer Predictor

**Standard predictor for JEPA**:

```python
class JEPAPredictor(nn.Module):
    """
    Predictor network for JEPA.
    
    Predicts target embeddings from context embeddings.
    
    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
    """
    def __init__(
        self,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, z_context, target_positions=None):
        """
        Args:
            z_context: Context embeddings (B, num_context, embed_dim)
            target_positions: Optional target position embeddings
        
        Returns:
            z_pred: Predicted target embeddings (B, num_targets, embed_dim)
        """
        # Process context
        z = z_context
        
        for block in self.blocks:
            z = block(z)
        
        z = self.norm(z)
        
        return z
```

### 2.2 Conditional Predictor (for Perturbations)

**Predictor with perturbation conditioning**:

```python
class ConditionalPredictor(nn.Module):
    """
    Predictor with conditioning (e.g., perturbation, time).
    
    Args:
        embed_dim: Embedding dimension
        condition_dim: Dimension of condition embedding
        depth: Number of transformer layers
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        embed_dim=256,
        condition_dim=128,
        depth=6,
        num_heads=8,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Condition embedding
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        # Cross-attention: context attends to condition
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # Self-attention on context
        self.self_attn_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, z_context, condition):
        """
        Args:
            z_context: Context embeddings (B, num_tokens, embed_dim)
            condition: Condition vector (B, condition_dim)
        
        Returns:
            z_pred: Predicted embeddings (B, num_tokens, embed_dim)
        """
        # Project condition
        cond_emb = self.condition_proj(condition)  # (B, embed_dim)
        cond_emb = cond_emb.unsqueeze(1)  # (B, 1, embed_dim)
        
        z = z_context
        
        # Alternate cross-attention and self-attention
        for cross_block, self_block in zip(self.cross_attn_blocks, self.self_attn_blocks):
            # Cross-attention: context attends to condition
            z = cross_block(z, cond_emb)
            
            # Self-attention on context
            z = self_block(z)
        
        z = self.norm(z)
        
        return z


class CrossAttentionBlock(nn.Module):
    """Cross-attention block."""
    def __init__(self, dim, num_heads):
        super().__init__()
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, q, kv):
        """
        Args:
            q: Query (B, num_q, dim)
            kv: Key/Value (B, num_kv, dim)
        
        Returns:
            out: (B, num_q, dim)
        """
        # Cross-attention
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out = self.attn(q_norm, kv_norm, kv_norm)[0]
        q = q + attn_out
        
        # MLP
        q = q + self.mlp(self.norm_mlp(q))
        
        return q
```

### 2.3 Predictor Design Choices

| Choice | Options | Recommendation |
|--------|---------|----------------|
| **Depth** | 4-12 layers | 6 for images, 4-6 for gene expression |
| **Capacity** | Same as encoder, smaller | Smaller (0.5× encoder depth) |
| **Conditioning** | None, cross-attention, FiLM | Cross-attention for perturbations |
| **Output** | Direct, with projection | Direct (same dim as encoder) |

---

## 3. VICReg Regularization

### 3.1 The Collapse Problem

**Without regularization**, embeddings collapse:
- All embeddings → same vector
- Predictor learns trivial solution
- No useful representation

**VICReg prevents collapse** via three terms:
1. **Variance**: Keep embeddings spread out
2. **Invariance**: Predictions match targets
3. **Covariance**: Decorrelate dimensions

### 3.2 VICReg Loss Implementation

```python
class VICRegLoss(nn.Module):
    """
    VICReg loss: Variance-Invariance-Covariance Regularization.
    
    Args:
        lambda_inv: Weight for invariance term
        lambda_var: Weight for variance term
        lambda_cov: Weight for covariance term
        gamma: Target variance
        epsilon: Small constant for numerical stability
    """
    def __init__(
        self,
        lambda_inv=25.0,
        lambda_var=25.0,
        lambda_cov=1.0,
        gamma=1.0,
        epsilon=1e-4,
    ):
        super().__init__()
        
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, z_pred, z_target):
        """
        Args:
            z_pred: Predicted embeddings (B, num_tokens, embed_dim)
            z_target: Target embeddings (B, num_tokens, embed_dim)
        
        Returns:
            loss: Total VICReg loss
            loss_dict: Dictionary with individual losses
        """
        batch_size, num_tokens, embed_dim = z_pred.shape
        
        # Flatten tokens
        z_pred = z_pred.reshape(-1, embed_dim)  # (B*num_tokens, embed_dim)
        z_target = z_target.reshape(-1, embed_dim)
        
        # 1. Invariance loss: MSE between predictions and targets
        loss_inv = F.mse_loss(z_pred, z_target)
        
        # 2. Variance loss: Ensure embeddings have sufficient variance
        loss_var = self.variance_loss(z_pred) + self.variance_loss(z_target)
        
        # 3. Covariance loss: Decorrelate dimensions
        loss_cov = self.covariance_loss(z_pred) + self.covariance_loss(z_target)
        
        # Total loss
        loss = (
            self.lambda_inv * loss_inv +
            self.lambda_var * loss_var +
            self.lambda_cov * loss_cov
        )
        
        loss_dict = {
            'loss': loss.item(),
            'inv': loss_inv.item(),
            'var': loss_var.item(),
            'cov': loss_cov.item(),
        }
        
        return loss, loss_dict
    
    def variance_loss(self, z):
        """
        Variance loss: Penalize if variance is below gamma.
        
        Args:
            z: Embeddings (N, D)
        
        Returns:
            loss: Variance loss
        """
        # Compute variance along batch dimension
        std = torch.sqrt(z.var(dim=0) + self.epsilon)  # (D,)
        
        # Penalize if std < gamma
        loss = torch.mean(F.relu(self.gamma - std))
        
        return loss
    
    def covariance_loss(self, z):
        """
        Covariance loss: Decorrelate dimensions.
        
        Args:
            z: Embeddings (N, D)
        
        Returns:
            loss: Covariance loss
        """
        N, D = z.shape
        
        # Center embeddings
        z = z - z.mean(dim=0)
        
        # Covariance matrix
        cov = (z.T @ z) / (N - 1)  # (D, D)
        
        # Off-diagonal elements should be zero
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        loss = off_diag.pow(2).sum() / D
        
        return loss
```

### 3.3 VICReg Hyperparameters

**Typical values**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda_inv` | 25.0 | Invariance weight |
| `lambda_var` | 25.0 | Variance weight |
| `lambda_cov` | 1.0 | Covariance weight |
| `gamma` | 1.0 | Target std |
| `epsilon` | 1e-4 | Numerical stability |

**Tuning guidelines**:

- Increase `lambda_var` if embeddings collapse
- Increase `lambda_cov` if dimensions are correlated
- Decrease `lambda_inv` if predictions are too rigid

---

## 4. Masking Strategies

### 4.1 Random Block Masking (I-JEPA)

**For images**: Mask random blocks

```python
def random_block_mask(
    num_patches,
    num_blocks=4,
    block_aspect_ratio=(0.75, 1.5),
    block_scale=(0.15, 0.2),
):
    """
    Generate random block mask for I-JEPA.
    
    Args:
        num_patches: Total number of patches
        num_blocks: Number of blocks to mask
        block_aspect_ratio: Range of aspect ratios
        block_scale: Range of block scales (fraction of image)
    
    Returns:
        mask: Binary mask (num_patches,)
        target_indices: Indices of masked patches
    """
    H = W = int(num_patches ** 0.5)
    mask = torch.ones(H, W)
    target_indices = []
    
    for _ in range(num_blocks):
        # Sample block size
        scale = torch.rand(1) * (block_scale[1] - block_scale[0]) + block_scale[0]
        aspect = torch.rand(1) * (block_aspect_ratio[1] - block_aspect_ratio[0]) + block_aspect_ratio[0]
        
        h = int((scale * H * W / aspect) ** 0.5)
        w = int(h * aspect)
        
        # Sample block position
        top = torch.randint(0, H - h + 1, (1,)).item()
        left = torch.randint(0, W - w + 1, (1,)).item()
        
        # Mask block
        mask[top:top+h, left:left+w] = 0
        
        # Record target indices
        for i in range(top, top+h):
            for j in range(left, left+w):
                target_indices.append(i * W + j)
    
    mask = mask.flatten()
    target_indices = torch.tensor(target_indices)
    
    return mask, target_indices
```

### 4.2 Temporal Masking (V-JEPA)

**For videos/time-series**: Predict future from past

```python
def temporal_mask(num_frames, context_frames=4, target_frames=4):
    """
    Generate temporal mask for V-JEPA.
    
    Args:
        num_frames: Total number of frames
        context_frames: Number of context frames
        target_frames: Number of target frames
    
    Returns:
        context_indices: Indices of context frames
        target_indices: Indices of target frames
    """
    # Context: first context_frames
    context_indices = torch.arange(context_frames)
    
    # Target: next target_frames
    target_indices = torch.arange(context_frames, context_frames + target_frames)
    
    return context_indices, target_indices
```

### 4.3 Perturbation Masking (Bio-JEPA)

**For perturbations**: Baseline is context, perturbed is target

```python
def perturbation_mask(baseline, perturbed):
    """
    No masking needed - baseline is context, perturbed is target.
    
    Args:
        baseline: Baseline expression (B, num_genes)
        perturbed: Perturbed expression (B, num_genes)
    
    Returns:
        context: Baseline (context)
        target: Perturbed (target)
    """
    return baseline, perturbed
```

---

## 5. Complete JEPA Model

### 5.1 Full JEPA Implementation

```python
class JEPA(nn.Module):
    """
    Complete JEPA model.
    
    Args:
        encoder: Encoder network
        predictor: Predictor network
        embed_dim: Embedding dimension
    """
    def __init__(self, encoder, predictor, embed_dim=768):
        super().__init__()
        
        self.encoder = encoder
        self.predictor = predictor
        self.embed_dim = embed_dim
        
        # VICReg loss
        self.vicreg = VICRegLoss(
            lambda_inv=25.0,
            lambda_var=25.0,
            lambda_cov=1.0
        )
    
    def forward(self, x_context, x_target, context_mask=None, target_mask=None):
        """
        Args:
            x_context: Context input (B, ...)
            x_target: Target input (B, ...)
            context_mask: Optional context mask
            target_mask: Optional target mask
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual losses
        """
        # Encode context and target
        z_context = self.encoder(x_context, mask=context_mask)
        z_target = self.encoder(x_target, mask=target_mask)
        
        # Predict target from context
        z_pred = self.predictor(z_context)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_pred, z_target)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def encode(self, x):
        """Encode input to embeddings."""
        return self.encoder(x)
    
    @torch.no_grad()
    def predict(self, z_context):
        """Predict target embedding from context."""
        return self.predictor(z_context)
```

### 5.2 JEPA for Gene Expression

```python
class BioJEPA(nn.Module):
    """
    JEPA for gene expression / perturbation prediction.
    
    Args:
        num_genes: Number of genes
        embed_dim: Embedding dimension
        num_tokens: Number of tokens
        encoder_depth: Encoder depth
        predictor_depth: Predictor depth
        condition_dim: Perturbation condition dimension
    """
    def __init__(
        self,
        num_genes=20000,
        embed_dim=256,
        num_tokens=64,
        encoder_depth=6,
        predictor_depth=4,
        condition_dim=128,
    ):
        super().__init__()
        
        # Encoder
        self.encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
        )
        
        # Conditional predictor
        self.predictor = ConditionalPredictor(
            embed_dim=embed_dim,
            condition_dim=condition_dim,
            depth=predictor_depth,
        )
        
        # VICReg loss
        self.vicreg = VICRegLoss()
    
    def forward(self, x_baseline, x_perturbed, perturbation_emb):
        """
        Args:
            x_baseline: Baseline expression (B, num_genes)
            x_perturbed: Perturbed expression (B, num_genes)
            perturbation_emb: Perturbation embedding (B, condition_dim)
        
        Returns:
            loss: Total loss
            loss_dict: Loss components
        """
        # Encode baseline and perturbed
        z_baseline = self.encoder(x_baseline)
        z_perturbed = self.encoder(x_perturbed)
        
        # Predict perturbed from baseline + perturbation
        z_pred = self.predictor(z_baseline, perturbation_emb)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_pred, z_perturbed)
        
        return loss, loss_dict
    
    @torch.no_grad()
    def predict_perturbation(self, x_baseline, perturbation_emb):
        """
        Predict perturbed state from baseline + perturbation.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            perturbation_emb: Perturbation embedding (B, condition_dim)
        
        Returns:
            z_pred: Predicted perturbed embedding (B, num_tokens, embed_dim)
        """
        z_baseline = self.encoder(x_baseline)
        z_pred = self.predictor(z_baseline, perturbation_emb)
        return z_pred
```

---

## 6. Design Principles

### 6.1 Encoder-Predictor Asymmetry

**Key insight**: Predictor should be smaller than encoder

**Rationale**:

- Encoder learns rich representations
- Predictor learns relationships
- Asymmetry prevents shortcut solutions

**Typical ratios**:

- Predictor depth = 0.5× encoder depth
- Predictor width = 1.0× encoder width

### 6.2 Shared vs Separate Encoders

**Shared encoder** (standard):
```python
z_context = encoder(x_context)
z_target = encoder(x_target)
```

**Pros**: Parameter efficient, consistent representations
**Cons**: Must handle both context and target

**Separate encoders** (rare):
```python
z_context = encoder_context(x_context)
z_target = encoder_target(x_target)
```

**Pros**: Specialized for each input
**Cons**: More parameters, less parameter sharing

**Recommendation**: Use shared encoder (standard practice)

### 6.3 Stop-Gradient on Target

**Important**: Stop gradient through target encoder

```python
# Correct
z_target = encoder(x_target).detach()

# Or use torch.no_grad()
with torch.no_grad():
    z_target = encoder(x_target)
```

**Why**: Prevents collapse via shortcut through target path

---

## Key Takeaways

### Architecture

1. **Encoder** — ViT for images, MLP for gene expression
2. **Predictor** — Transformer, smaller than encoder
3. **VICReg** — Prevents collapse via variance + covariance
4. **Masking** — Block for images, temporal for videos, perturbation for biology

### Design Choices

1. **Shared encoder** — Same for context and target
2. **Asymmetric** — Predictor smaller than encoder
3. **Stop-gradient** — On target encoder
4. **No decoder** — Prediction in embedding space only

### Implementation

1. **Embed dim** — 768 for images, 256-512 for gene expression
2. **Encoder depth** — 12 for images, 4-8 for gene expression
3. **Predictor depth** — 0.5× encoder depth
4. **VICReg weights** — λ_inv=25, λ_var=25, λ_cov=1

---

## Related Documents

- [00_jepa_overview.md](00_jepa_overview.md) — High-level concepts
- [02_jepa_training.md](02_jepa_training.md) — Training strategies
- [03_jepa_applications.md](03_jepa_applications.md) — Applications
- [04_jepa_perturbseq.md](04_jepa_perturbseq.md) — Perturb-seq implementation

---

## References

- Assran et al. (2023): "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA)
- Bardes et al. (2022): "VICReg: Variance-Invariance-Covariance Regularization"
- Bardes et al. (2024): "V-JEPA: Latent Video Prediction"
- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
