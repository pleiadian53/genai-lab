"""Normalizing flow models for gene expression (placeholder for future implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalFlow(nn.Module):
    """Conditional normalizing flow for expression generation.

    This is a placeholder for future implementation.
    Consider using nflows or normflows libraries.

    Args:
        n_genes: Number of genes
        cond_dim: Condition dimension
        n_transforms: Number of flow transforms
        hidden_dim: Hidden dimension for transforms
    """

    def __init__(
        self,
        n_genes: int,
        cond_dim: int,
        n_transforms: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.cond_dim = cond_dim
        self.n_transforms = n_transforms

        # TODO: Implement flow transforms
        # Options:
        # 1. Affine coupling layers (RealNVP)
        # 2. Autoregressive flows (MAF)
        # 3. Continuous normalizing flows (FFJORD)

        raise NotImplementedError(
            "Normalizing flows not yet implemented. "
            "Consider using nflows: pip install nflows"
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform data to latent space and compute log determinant."""
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Transform latent to data space (generation)."""
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute log probability of data."""
        raise NotImplementedError

    def sample(self, n_samples: int, cond: torch.Tensor) -> torch.Tensor:
        """Sample from the model."""
        raise NotImplementedError
