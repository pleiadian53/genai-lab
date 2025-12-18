"""Diffusion models for gene expression (placeholder for future implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentDiffusion(nn.Module):
    """Latent diffusion model for expression generation.

    This is a placeholder for future implementation.
    The idea is to:
    1. Train a VAE to get a good latent space
    2. Train a diffusion model in that latent space
    3. Generate by: sample z from diffusion -> decode to expression

    Args:
        latent_dim: Dimension of the latent space
        cond_dim: Condition dimension
        n_steps: Number of diffusion steps
        hidden_dim: Hidden dimension for the denoising network
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        n_steps: int = 1000,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.n_steps = n_steps

        # TODO: Implement diffusion components
        # 1. Noise schedule (linear, cosine, etc.)
        # 2. Denoising network (U-Net style or MLP for latent)
        # 3. Training: predict noise or x0
        # 4. Sampling: DDPM, DDIM, etc.

        raise NotImplementedError(
            "Latent diffusion not yet implemented. "
            "Consider starting with a simple DDPM in latent space."
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy latent, condition, and timestep."""
        raise NotImplementedError

    def sample(self, n_samples: int, cond: torch.Tensor) -> torch.Tensor:
        """Sample from the diffusion model."""
        raise NotImplementedError

    def training_loss(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion training loss."""
        raise NotImplementedError


class ExpressionDiffusion(nn.Module):
    """Direct diffusion model on expression space.

    This is a placeholder. Diffusion directly on expression is tricky because:
    - Expression has biological constraints (non-negative, gene correlations)
    - High dimensionality (thousands of genes)

    Consider latent diffusion instead.
    """

    def __init__(self, n_genes: int, cond_dim: int, n_steps: int = 1000):
        super().__init__()
        raise NotImplementedError(
            "Direct expression diffusion not recommended. "
            "Use LatentDiffusion instead."
        )
