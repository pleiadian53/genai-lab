"""Decoder architectures for gene expression generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDecoder(nn.Module):
    """Gaussian decoder for log-normalized expression data.

    Outputs mean (and optionally log-variance) for each gene.

    Args:
        input_dim: Input dimension (latent dim + condition dim)
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of genes)
        learn_var: Whether to learn per-gene variance
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        learn_var: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.learn_var = learn_var

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, output_dim)

        if learn_var:
            self.logvar_head = nn.Linear(in_dim, output_dim)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: Latent + condition vector of shape (B, input_dim)

        Returns:
            Dict with 'mean' and optionally 'logvar'
        """
        h = self.net(z)
        out = {"mean": self.mean_head(h)}

        if self.learn_var:
            out["logvar"] = self.logvar_head(h).clamp(-10, 10)

        return out


class NegativeBinomialDecoder(nn.Module):
    """Negative Binomial decoder for count data (scRNA-seq).

    Models counts as NB(mu, theta) where:
    - mu: mean expression
    - theta: dispersion (inverse overdispersion)

    Args:
        input_dim: Input dimension (latent dim + condition dim)
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of genes)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.net = nn.Sequential(*layers)

        # Rate parameter (will be scaled by library size)
        self.rho_head = nn.Linear(in_dim, output_dim)

        # Dispersion parameter (gene-specific, learned)
        self.log_theta = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            z: Latent + condition vector of shape (B, input_dim)
            library_size: Library size per sample of shape (B,) or (B, 1)

        Returns:
            Dict with 'mu' (mean), 'theta' (dispersion), 'rho' (rate before scaling)
        """
        h = self.net(z)

        # Rate (softmax ensures non-negative and sums to 1 across genes)
        rho = F.softmax(self.rho_head(h), dim=-1)

        # Scale by library size
        if library_size is not None:
            if library_size.dim() == 1:
                library_size = library_size.unsqueeze(-1)
            mu = rho * library_size
        else:
            mu = rho

        # Dispersion
        theta = torch.exp(self.log_theta).unsqueeze(0).expand(z.shape[0], -1)

        return {"mu": mu, "theta": theta, "rho": rho}


class ZINBDecoder(NegativeBinomialDecoder):
    """Zero-Inflated Negative Binomial decoder for sparse scRNA-seq data.

    Extends NB with a dropout probability for excess zeros.

    Args:
        input_dim: Input dimension (latent dim + condition dim)
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of genes)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__(input_dim, hidden_dims, output_dim, dropout)

        # Dropout logits (probability of excess zero)
        self.pi_head = nn.Linear(
            sum(hidden_dims[-1:]) if hidden_dims else input_dim,
            output_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'mu', 'theta', 'rho', and 'pi' (dropout probability)
        """
        out = super().forward(z, library_size)

        # Get hidden representation for pi
        h = self.net(z)
        out["pi"] = torch.sigmoid(self.pi_head(h))

        return out
