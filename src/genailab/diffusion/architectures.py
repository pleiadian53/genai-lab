"""Advanced score network architectures for diffusion models.

This module provides state-of-the-art architectures for different data modalities:
- TabularScoreNetwork: Advanced MLP with attention for gene expression/tabular data
- UNet2D: U-Net for 2D images (medical imaging, natural images)
- UNet3D: 3D U-Net for volumetric medical data (CT, MRI)
- TransformerScoreNetwork: Transformer-based architecture for modern approaches

References:
- U-Net: Ronneberger et al. (2015)
- Attention: Vaswani et al. (2017)
- DiT: Peebles & Xie (2023)
- Stable Diffusion: Rombach et al. (2022)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# Time Embedding Utilities
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (like Transformer positional encoding)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values, shape (batch_size,)
        Returns:
            Time embeddings, shape (batch_size, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================================
# Tabular/Gene Expression Architecture
# ============================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention for feature interactions."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch_size, dim)
        Returns:
            Output features, shape (batch_size, dim)
        """
        B, D = x.shape
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (B, 1, D)
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.proj(out).squeeze(1)  # (B, D)
        
        return out


class TabularScoreNetwork(nn.Module):
    """Advanced MLP with attention for tabular/gene expression data.
    
    Features:
    - Deep residual architecture with skip connections
    - Self-attention for feature interactions
    - Layer normalization for stability
    - Adaptive time conditioning via FiLM (Feature-wise Linear Modulation)
    - Suitable for high-dimensional gene expression data (1000s of genes)
    
    Args:
        data_dim: Input dimension (e.g., number of genes)
        hidden_dim: Hidden layer dimension
        time_dim: Time embedding dimension
        num_layers: Number of residual blocks
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_attention: Whether to use self-attention layers
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim * 2),  # For FiLM (scale & shift)
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        
        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # Residual blocks with optional attention
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ])
            self.blocks.append(block)
            
            # Add attention every few layers
            if use_attention and (i + 1) % 2 == 0:
                self.blocks.append(nn.ModuleList([
                    nn.LayerNorm(hidden_dim),
                    SelfAttention(hidden_dim, num_heads),
                ]))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict score: s(x, t) ≈ ∇_x log p_t(x).
        
        Args:
            x: Data, shape (batch_size, data_dim)
            t: Time, shape (batch_size,)
        
        Returns:
            Predicted score, shape (batch_size, data_dim)
        """
        # Time embedding with FiLM conditioning
        t_emb = self.time_embed(t)  # (B, hidden_dim * 2)
        scale, shift = t_emb.chunk(2, dim=-1)  # Each (B, hidden_dim)
        
        # Input projection
        h = self.input_proj(x)
        
        # Apply residual blocks with time conditioning
        for block in self.blocks:
            if len(block) == 2:  # Attention block
                norm, attn = block
                h = h + attn(norm(h))
            else:  # Residual block with FiLM
                norm, fc1, act, drop1, fc2, drop2 = block
                h_norm = norm(h)
                # FiLM: scale and shift
                h_norm = h_norm * (1 + scale) + shift
                # Residual connection
                h = h + drop2(fc2(drop1(act(fc1(h_norm)))))
        
        # Output projection
        return self.output_proj(h)


# ============================================================================
# 2D U-Net for Images
# ============================================================================

class ConvBlock2D(nn.Module):
    """2D Convolutional block with GroupNorm and SiLU."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2) if time_emb_dim else nn.Identity()
        ) if time_emb_dim else None
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input, shape (B, C, H, W)
            t_emb: Time embedding, shape (B, time_emb_dim)
        Returns:
            Output, shape (B, out_channels, H, W)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding via FiLM
        if self.time_mlp is not None and t_emb is not None:
            t_out = self.time_mlp(t_emb)
            scale, shift = t_out.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.residual(x)


class UNet2D(nn.Module):
    """2D U-Net for image diffusion models.
    
    Standard architecture for image generation (DDPM, Stable Diffusion).
    Features:
    - Encoder-decoder structure with skip connections
    - Multi-scale feature extraction
    - Time conditioning at each resolution
    - GroupNorm for stability with small batches
    
    Args:
        in_channels: Input channels (1 for grayscale, 3 for RGB)
        out_channels: Output channels (usually same as in_channels)
        base_channels: Base number of channels (doubled at each downsampling)
        channel_multipliers: Channel multipliers for each resolution
        num_res_blocks: Number of residual blocks per resolution
        time_emb_dim: Time embedding dimension
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_resolutions = len(channel_multipliers)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Build channel sequence for encoder
        channels = [base_channels]
        ch = base_channels
        for mult in channel_multipliers:
            ch = base_channels * mult
            channels.append(ch)
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        in_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            # Residual blocks at this resolution
            for _ in range(num_res_blocks):
                self.encoder.append(ConvBlock2D(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
            
            # Downsample (except at the last resolution)
            if i < len(channel_multipliers) - 1:
                self.downsamplers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.downsamplers.append(None)
        
        # Bottleneck
        self.bottleneck1 = ConvBlock2D(in_ch, in_ch, time_emb_dim)
        self.bottleneck2 = ConvBlock2D(in_ch, in_ch, time_emb_dim)
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        reversed_mults = list(reversed(channel_multipliers))
        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            
            # Upsample first (except at the first decoder level)
            if i > 0:
                self.upsamplers.append(nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1))
            else:
                self.upsamplers.append(None)
            
            # Residual blocks with skip connections
            for j in range(num_res_blocks):
                # First block at each resolution receives skip connection
                skip_ch = out_ch  # Skip connection has out_ch channels
                block_in_ch = in_ch + skip_ch
                self.decoder.append(ConvBlock2D(block_in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise or score.
        
        Args:
            x: Input image, shape (B, C, H, W)
            t: Time, shape (B,)
        
        Returns:
            Predicted noise/score, shape (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder - collect skip connections
        skips = [h]
        encoder_idx = 0
        num_res_blocks = len(self.encoder) // self.num_resolutions
        
        for i in range(self.num_resolutions):
            # Apply residual blocks
            for _ in range(num_res_blocks):
                h = self.encoder[encoder_idx](h, t_emb)
                skips.append(h)
                encoder_idx += 1
            
            # Downsample
            if self.downsamplers[i] is not None:
                h = self.downsamplers[i](h)
        
        # Bottleneck
        h = self.bottleneck1(h, t_emb)
        h = self.bottleneck2(h, t_emb)
        
        # Decoder - use skip connections in reverse
        decoder_idx = 0
        for i in range(self.num_resolutions):
            # Upsample
            if self.upsamplers[i] is not None:
                h = self.upsamplers[i](h)
            
            # Apply residual blocks with skip connections
            for _ in range(num_res_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder[decoder_idx](h, t_emb)
                decoder_idx += 1
        
        # Output
        return self.output_conv(h)


# ============================================================================
# 3D U-Net for Volumetric Medical Data
# ============================================================================

class ConvBlock3D(nn.Module):
    """3D Convolutional block for volumetric data."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2) if time_emb_dim else nn.Identity()
        ) if time_emb_dim else None
        
        # Residual connection
        self.residual = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input, shape (B, C, D, H, W)
            t_emb: Time embedding, shape (B, time_emb_dim)
        Returns:
            Output, shape (B, out_channels, D, H, W)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time embedding via FiLM
        if self.time_mlp is not None and t_emb is not None:
            t_out = self.time_mlp(t_emb)
            scale, shift = t_out.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.residual(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric medical data (CT, MRI).
    
    Extends 2D U-Net to 3D for volumetric medical imaging.
    Features:
    - 3D convolutions for spatial-temporal/volumetric data
    - Same U-Net structure as 2D version
    - Suitable for CT scans, MRI volumes, video data
    
    Args:
        in_channels: Input channels (1 for single modality, more for multi-modal)
        out_channels: Output channels
        base_channels: Base number of channels
        channel_multipliers: Channel multipliers for each resolution
        num_res_blocks: Number of residual blocks per resolution
        time_emb_dim: Time embedding dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,  # Smaller for memory constraints
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim // 4),
            nn.Linear(time_emb_dim // 4, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input convolution
        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ConvBlock3D(ch, out_ch, time_emb_dim))
                ch = out_ch
            self.encoder_blocks.append(blocks)
            
            if i < len(channel_multipliers) - 1:
                self.downsample_blocks.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsample_blocks.append(nn.Identity())
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(ch, ch, time_emb_dim)
        
        # Decoder (upsampling)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                in_ch = ch + out_ch if j == 0 else ch
                blocks.append(ConvBlock3D(in_ch, out_ch, time_emb_dim))
                ch = out_ch
            self.decoder_blocks.append(blocks)
            
            if i < len(channel_multipliers) - 1:
                self.upsample_blocks.append(nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1))
            else:
                self.upsample_blocks.append(nn.Identity())
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise or score.
        
        Args:
            x: Input volume, shape (B, C, D, H, W)
            t: Time, shape (B,)
        
        Returns:
            Predicted noise/score, shape (B, C, D, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsample_blocks):
            for block in blocks:
                h = block(h, t_emb)
            # Save skip connection before downsampling
            skip_connections.append(h)
            h = downsample(h)
        
        # Bottleneck
        h = self.bottleneck(h, t_emb)
        
        # Decoder with skip connections
        for blocks, upsample in zip(self.decoder_blocks, self.upsample_blocks):
            h = upsample(h)
            for i, block in enumerate(blocks):
                if i == 0:
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
        
        # Output
        return self.output_conv(h)


# ============================================================================
# Architecture Factory
# ============================================================================

def get_score_network(
    architecture: str,
    **kwargs
) -> nn.Module:
    """Factory function to create score networks.
    
    Args:
        architecture: One of 'simple', 'tabular', 'unet2d', 'unet3d'
        **kwargs: Architecture-specific parameters
    
    Returns:
        Score network module
    
    Examples:
        >>> # For gene expression data
        >>> model = get_score_network('tabular', data_dim=5000, hidden_dim=512)
        
        >>> # For 2D medical images
        >>> model = get_score_network('unet2d', in_channels=1, base_channels=64)
        
        >>> # For 3D CT scans
        >>> model = get_score_network('unet3d', in_channels=1, base_channels=32)
    """
    architectures = {
        'tabular': TabularScoreNetwork,
        'unet2d': UNet2D,
        'unet3d': UNet3D,
    }
    
    if architecture not in architectures:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from {list(architectures.keys())}"
        )
    
    return architectures[architecture](**kwargs)
