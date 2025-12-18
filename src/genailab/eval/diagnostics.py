"""Diagnostic tools for generative models."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def batch_leakage_score(
    z: np.ndarray,
    batch_labels: np.ndarray,
    cv: int = 5,
) -> dict[str, float]:
    """Measure batch information leakage in latent space.

    A good model should have low batch leakage (batch not predictable from z).

    Args:
        z: Latent representations (samples x z_dim)
        batch_labels: Batch labels for each sample
        cv: Number of cross-validation folds

    Returns:
        Dict with 'accuracy' (lower is better), 'baseline' (random chance)
    """
    # Fit classifier to predict batch from z
    clf = LogisticRegression(max_iter=1000, random_state=42)

    try:
        scores = cross_val_score(clf, z, batch_labels, cv=cv, scoring="accuracy")
        accuracy = scores.mean()
    except Exception:
        accuracy = 1.0 / len(np.unique(batch_labels))

    # Baseline: random chance
    n_batches = len(np.unique(batch_labels))
    baseline = 1.0 / n_batches

    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "leakage_ratio": accuracy / baseline if baseline > 0 else float("inf"),
        "n_batches": n_batches,
    }


def posterior_collapse_check(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    threshold: float = 0.1,
) -> dict[str, float]:
    """Check for posterior collapse in VAE.

    Posterior collapse occurs when q(z|x) ≈ p(z) for all x,
    meaning the latent code carries no information about x.

    Args:
        model: VAE model with encode method
        dataloader: Data loader
        device: Device to run on
        threshold: KL threshold below which a dimension is considered collapsed

    Returns:
        Dict with collapse metrics
    """
    model.eval()
    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            mu, logvar = model.encode(x, cond)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    # Per-dimension KL from N(mu, sigma) to N(0, 1)
    kl_per_dim = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())
    kl_per_dim = kl_per_dim.mean(dim=0)  # Average over samples

    # Count collapsed dimensions
    n_collapsed = (kl_per_dim < threshold).sum().item()
    z_dim = kl_per_dim.shape[0]

    # Active units (AU) metric
    # A dimension is active if its variance across samples is > threshold
    mu_var = all_mu.var(dim=0)
    n_active = (mu_var > threshold).sum().item()

    return {
        "n_collapsed": n_collapsed,
        "n_active": n_active,
        "z_dim": z_dim,
        "collapse_ratio": n_collapsed / z_dim,
        "mean_kl_per_dim": kl_per_dim.mean().item(),
        "kl_per_dim": kl_per_dim.numpy(),
    }


def reconstruction_quality(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate reconstruction quality.

    Args:
        model: VAE model
        dataloader: Data loader
        device: Device to run on

    Returns:
        Dict with reconstruction metrics
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_corr = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            out = model(x, cond)
            x_hat = out["x_hat"]

            # MSE
            mse = ((x - x_hat) ** 2).mean(dim=1)
            total_mse += mse.sum().item()

            # MAE
            mae = (x - x_hat).abs().mean(dim=1)
            total_mae += mae.sum().item()

            # Per-sample correlation
            for i in range(x.shape[0]):
                corr = torch.corrcoef(torch.stack([x[i], x_hat[i]]))[0, 1]
                if not torch.isnan(corr):
                    total_corr += corr.item()

            n_samples += x.shape[0]

    return {
        "mse": total_mse / n_samples,
        "mae": total_mae / n_samples,
        "mean_correlation": total_corr / n_samples,
        "n_samples": n_samples,
    }
