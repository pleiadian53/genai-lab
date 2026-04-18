"""Velocity field networks for flow matching.

The velocity network v_theta(x, t) is the neural network trained to approximate
the marginal velocity field. It takes a noisy sample x_t and time t as input
and predicts the velocity that should be applied to move toward the data distribution.

Architecture choice:
  - VelocityUNet2D: for 2D images (MNIST, PathMNIST, histopathology). Wraps the
    existing UNet2D from diffusion.architectures with flow-matching defaults.
  - VelocityMLP: lightweight MLP for 1D/tabular/toy 2D data.

Both expose the same interface: forward(x, t) -> predicted velocity.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the existing UNet2D and time embedding from the diffusion module.
from genailab.diffusion.architectures import UNet2D, SinusoidalTimeEmbedding


class VelocityUNet2D(nn.Module):
    """U-Net velocity field for 2D image flow matching.

    A thin wrapper around the project's existing UNet2D that:
    - Sets flow-matching-appropriate defaults (smaller base_channels for MPS)
    - Documents the interface clearly for flow matching use
    - Accepts t as shape (B,) — the standard convention throughout this package

    For RunPod (GPU) training, increase base_channels to 128 or 256.
    For MPS (M1 MacBook) training, keep base_channels ≤ 64.

    Args:
        in_channels:        Number of image channels (1 for grayscale, 3 for RGB).
        base_channels:      Base channel width. Doubles at each downsampling level.
                            32 → MPS-safe; 64 → good quality; 128+ → RunPod.
        channel_multipliers: Channel width multiplier at each U-Net resolution.
        num_res_blocks:     Residual blocks per resolution level.
        time_emb_dim:       Dimension of the sinusoidal time embedding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.unet = UNet2D(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the velocity field v_theta(x, t).

        Args:
            x: Interpolated sample, shape (B, C, H, W).
            t: Time values, shape (B,) in [0, 1].

        Returns:
            Predicted velocity, same shape as x.
        """
        return self.unet(x, t)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class VelocityMLP(nn.Module):
    """Lightweight MLP velocity field for low-dimensional data.

    Suitable for:
    - 1D/2D toy distributions (Gaussian, Swiss roll, checkerboard)
    - Flat/tabular feature vectors
    - Quick POC experiments before scaling to image models

    Time conditioning is applied via FiLM (Feature-wise Linear Modulation):
    the time embedding produces per-channel scale and shift parameters applied
    at every hidden layer.

    Args:
        data_dim:   Dimensionality of x (e.g., 2 for 2D toy, 784 for flat MNIST).
        hidden_dim: Hidden layer width.
        n_layers:   Number of hidden layers (depth of the MLP).
        time_dim:   Sinusoidal time embedding dimension.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        time_dim: int = 64,
    ):
        super().__init__()
        self.data_dim = data_dim

        # Sinusoidal time embedding → FiLM scale + shift
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # Hidden layers with LayerNorm + FiLM + residual
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity v_theta(x, t).

        Args:
            x: Input data, shape (B, data_dim).
            t: Time values, shape (B,) in [0, 1].

        Returns:
            Predicted velocity, shape (B, data_dim).
        """
        # FiLM conditioning from time
        t_emb = self.time_embed(t)           # (B, hidden_dim * 2)
        scale, shift = t_emb.chunk(2, dim=-1)  # each (B, hidden_dim)

        h = self.input_proj(x)
        for layer in self.layers:
            h = h + layer(h * (1.0 + scale) + shift)  # FiLM + residual

        return self.output_proj(h)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def build_velocity_network(
    architecture: str,
    **kwargs,
) -> nn.Module:
    """Factory: construct a velocity network by name.

    Args:
        architecture: ``"unet2d"`` or ``"mlp"``.
        **kwargs:     Forwarded to the chosen class constructor.

    Returns:
        Instantiated velocity network.

    Examples::

        # Small U-Net for 28x28 grayscale MNIST on MPS
        model = build_velocity_network("unet2d", in_channels=1, base_channels=32)

        # Larger U-Net for 64x64 RGB PathMNIST on RunPod
        model = build_velocity_network("unet2d", in_channels=3, base_channels=128)

        # MLP for 2D toy distributions
        model = build_velocity_network("mlp", data_dim=2, hidden_dim=128, n_layers=3)
    """
    registry = {
        "unet2d": VelocityUNet2D,
        "mlp": VelocityMLP,
    }
    if architecture not in registry:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choose from {list(registry.keys())}."
        )
    return registry[architecture](**kwargs)
