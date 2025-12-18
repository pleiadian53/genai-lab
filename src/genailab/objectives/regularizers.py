"""Regularizers for generative models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """KL divergence from N(mu, sigma) to N(0, I).

    KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean of approximate posterior
        logvar: Log-variance of approximate posterior
        reduction: 'mean', 'sum', or 'none'

    Returns:
        KL divergence
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


class BatchDiscriminator(nn.Module):
    """Discriminator for adversarial batch effect removal.

    Tries to predict batch from latent representation.
    The encoder is trained to fool this discriminator.

    Args:
        z_dim: Latent dimension
        n_batches: Number of batch categories
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, z_dim: int, n_batches: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_batches),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict batch logits from latent."""
        return self.net(z)


def batch_adversarial_loss(
    z: torch.Tensor,
    batch_labels: torch.Tensor,
    discriminator: BatchDiscriminator,
    train_discriminator: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adversarial loss for batch effect removal.

    Args:
        z: Latent representation
        batch_labels: True batch labels
        discriminator: Batch discriminator network
        train_discriminator: If True, return discriminator loss; else encoder loss

    Returns:
        Tuple of (discriminator_loss, encoder_loss)
    """
    logits = discriminator(z)

    # Discriminator wants to correctly classify batch
    disc_loss = F.cross_entropy(logits, batch_labels)

    # Encoder wants to maximize entropy (confuse discriminator)
    # This is equivalent to minimizing negative entropy
    probs = F.softmax(logits, dim=-1)
    enc_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))

    return disc_loss, enc_loss


def gene_set_prior(
    z: torch.Tensor,
    gene_sets: dict[str, list[int]],
    decoder: nn.Module,
    cond: dict[str, torch.Tensor],
    strength: float = 0.1,
) -> torch.Tensor:
    """Regularizer encouraging gene set coherence.

    Penalizes decoded expression where genes in the same pathway
    have inconsistent activation patterns.

    Args:
        z: Latent representation
        gene_sets: Dict mapping pathway name to list of gene indices
        decoder: Decoder module
        cond: Condition dict
        strength: Regularization strength

    Returns:
        Gene set coherence penalty
    """
    x_hat = decoder(z, cond)

    penalty = torch.tensor(0.0, device=z.device)
    for pathway_name, gene_idx in gene_sets.items():
        if len(gene_idx) < 2:
            continue

        # Get expression of genes in this pathway
        pathway_expr = x_hat[:, gene_idx]  # (B, n_pathway_genes)

        # Penalize variance within pathway (encourage coherence)
        # This is a simple heuristic; more sophisticated approaches exist
        pathway_var = pathway_expr.var(dim=1).mean()
        penalty = penalty + pathway_var

    return strength * penalty / max(len(gene_sets), 1)


def covariance_penalty(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    n_genes_sample: int = 100,
    strength: float = 0.01,
) -> torch.Tensor:
    """Penalize difference in gene-gene covariance structure.

    Encourages the model to preserve correlation patterns.

    Args:
        x: Original expression (B, G)
        x_hat: Reconstructed expression (B, G)
        n_genes_sample: Number of genes to sample for efficiency
        strength: Regularization strength

    Returns:
        Covariance penalty
    """
    B, G = x.shape

    # Sample genes for efficiency
    if G > n_genes_sample:
        idx = torch.randperm(G)[:n_genes_sample]
        x = x[:, idx]
        x_hat = x_hat[:, idx]

    # Compute covariance matrices
    x_centered = x - x.mean(dim=0, keepdim=True)
    x_hat_centered = x_hat - x_hat.mean(dim=0, keepdim=True)

    cov_x = (x_centered.T @ x_centered) / (B - 1)
    cov_x_hat = (x_hat_centered.T @ x_hat_centered) / (B - 1)

    # Frobenius norm of difference
    penalty = torch.norm(cov_x - cov_x_hat, p="fro")

    return strength * penalty
