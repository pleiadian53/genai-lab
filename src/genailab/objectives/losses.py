"""Loss functions for generative gene expression models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE reconstruction loss for Gaussian decoder.

    Args:
        x: Original expression
        x_hat: Reconstructed expression
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Reconstruction loss
    """
    return F.mse_loss(x_hat, x, reduction=reduction)


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
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, {"recon": recon.detach(), "kl": kl.detach()}


def nb_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative Binomial negative log-likelihood.

    NB(x; mu, theta) where theta is the dispersion parameter.

    Args:
        x: Observed counts
        mu: Mean parameter
        theta: Dispersion parameter (inverse overdispersion)
        eps: Small constant for numerical stability

    Returns:
        Negative log-likelihood (mean over batch and genes)
    """
    # NB log-likelihood
    # log p(x|mu,theta) = log Gamma(x+theta) - log Gamma(theta) - log Gamma(x+1)
    #                   + theta*log(theta/(theta+mu)) + x*log(mu/(theta+mu))

    log_theta_mu = torch.log(theta + mu + eps)
    log_theta = torch.log(theta + eps)
    log_mu = torch.log(mu + eps)

    ll = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * (log_theta - log_theta_mu)
        + x * (log_mu - log_theta_mu)
    )

    return -ll.mean()


def zinb_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Zero-Inflated Negative Binomial negative log-likelihood.

    ZINB(x; mu, theta, pi) where pi is the dropout probability.

    Args:
        x: Observed counts
        mu: Mean parameter
        theta: Dispersion parameter
        pi: Dropout probability (probability of excess zero)
        eps: Small constant for numerical stability

    Returns:
        Negative log-likelihood (mean over batch and genes)
    """
    # For zeros: log(pi + (1-pi) * NB(0|mu,theta))
    # For non-zeros: log((1-pi) * NB(x|mu,theta))

    log_theta_mu = torch.log(theta + mu + eps)
    log_theta = torch.log(theta + eps)
    log_mu = torch.log(mu + eps)

    # NB log-likelihood
    nb_ll = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * (log_theta - log_theta_mu)
        + x * (log_mu - log_theta_mu)
    )

    # NB(0|mu,theta) = (theta/(theta+mu))^theta
    nb_zero = theta * (log_theta - log_theta_mu)

    # Zero case: log(pi + (1-pi)*exp(nb_zero))
    zero_case = torch.log(pi + (1 - pi) * torch.exp(nb_zero) + eps)

    # Non-zero case: log(1-pi) + nb_ll
    nonzero_case = torch.log(1 - pi + eps) + nb_ll

    # Select based on whether x is zero
    is_zero = (x < 0.5).float()
    ll = is_zero * zero_case + (1 - is_zero) * nonzero_case

    return -ll.mean()


def kl_annealing_schedule(
    step: int,
    warmup_steps: int = 1000,
    max_beta: float = 1.0,
    schedule: str = "linear",
) -> float:
    """KL annealing schedule for beta-VAE training.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_beta: Maximum beta value
        schedule: 'linear', 'cosine', or 'cyclical'

    Returns:
        Current beta value
    """
    if schedule == "linear":
        return min(max_beta, max_beta * step / warmup_steps)

    elif schedule == "cosine":
        import math
        if step >= warmup_steps:
            return max_beta
        return max_beta * (1 - math.cos(math.pi * step / warmup_steps)) / 2

    elif schedule == "cyclical":
        # Cyclical annealing (Fu et al., 2019)
        cycle_length = warmup_steps * 2
        position = step % cycle_length
        if position < warmup_steps:
            return max_beta * position / warmup_steps
        return max_beta

    else:
        raise ValueError(f"Unknown schedule: {schedule}")
