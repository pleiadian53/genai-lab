"""Conditional Variational Autoencoder for gene expression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from genailab.model.conditioning import ConditionEncoder


class CVAE(nn.Module):
    """Conditional Variational Autoencoder for gene expression.

    Models p(x|z,c) and q(z|x,c) where:
    - x: gene expression vector
    - z: latent representation
    - c: condition (tissue, disease, batch, etc.)

    Args:
        n_genes: Number of genes
        z_dim: Latent dimension
        cond_encoder: Condition encoder module
        hidden: Hidden layer dimension
        n_layers: Number of hidden layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_genes: int,
        z_dim: int,
        cond_encoder: ConditionEncoder,
        hidden: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.z_dim = z_dim
        self.cond_encoder = cond_encoder

        cond_dim = cond_encoder.spec.out_dim

        # Encoder q(z|x,c)
        enc_layers = []
        in_dim = n_genes + cond_dim
        for _ in range(n_layers):
            enc_layers.extend([
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden

        self.enc = nn.Sequential(*enc_layers)
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        # Decoder p(x|z,c)
        dec_layers = []
        in_dim = z_dim + cond_dim
        for _ in range(n_layers):
            dec_layers.extend([
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden

        dec_layers.append(nn.Linear(hidden, n_genes))
        self.dec = nn.Sequential(*dec_layers)

    def encode(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode expression to latent distribution parameters.

        Args:
            x: Expression vector of shape (B, n_genes)
            cond: Dict of condition tensors

        Returns:
            Tuple of (mu, logvar) each of shape (B, z_dim)
        """
        c = self.cond_encoder(cond)
        h = self.enc(torch.cat([x, c], dim=-1))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(-12, 12)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Decode latent to expression.

        Args:
            z: Latent vector of shape (B, z_dim)
            cond: Dict of condition tensors

        Returns:
            Reconstructed expression of shape (B, n_genes)
        """
        c = self.cond_encoder(cond)
        return self.dec(torch.cat([z, c], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode, sample, decode.

        Args:
            x: Expression vector of shape (B, n_genes)
            cond: Dict of condition tensors

        Returns:
            Dict with 'x_hat', 'mu', 'logvar', 'z'
        """
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, cond)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}

    def sample(
        self,
        n_samples: int,
        cond: dict[str, torch.Tensor],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Sample from the prior p(z) and decode.

        Args:
            n_samples: Number of samples to generate
            cond: Dict of condition tensors (each of shape (n_samples,))
            device: Device to generate on

        Returns:
            Generated expression of shape (n_samples, n_genes)
        """
        z = torch.randn(n_samples, self.z_dim, device=device)
        return self.decode(z, cond)


def elbo_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute ELBO loss for VAE.

    Loss = reconstruction + beta * KL divergence

    Args:
        x: Original expression
        x_hat: Reconstructed expression
        mu: Latent mean
        logvar: Latent log-variance
        beta: KL weight (beta-VAE)

    Returns:
        Tuple of (total_loss, dict with 'recon' and 'kl')
    """
    # Reconstruction loss (Gaussian with fixed variance -> MSE)
    recon = F.mse_loss(x_hat, x, reduction="mean")

    # KL divergence: KL(q(z|x,c) || N(0,I))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon + beta * kl

    return total, {"recon": recon.detach(), "kl": kl.detach()}
