"""Counterfactual generation utilities."""

from __future__ import annotations

import torch
import numpy as np


@torch.no_grad()
def counterfactual_decode(
    model: torch.nn.Module,
    x: torch.Tensor,
    cond: dict[str, torch.Tensor],
    cond_cf: dict[str, torch.Tensor],
    use_mean: bool = True,
) -> torch.Tensor:
    """Generate counterfactual expression.

    Infer z from (x, cond), then decode under cond_cf.
    This is the basic "same sample, changed condition" operation.

    Args:
        model: VAE model with encode and decode methods
        x: Original expression (B, n_genes)
        cond: Original conditions
        cond_cf: Counterfactual conditions
        use_mean: If True, use mean of posterior; else sample

    Returns:
        Counterfactual expression (B, n_genes)
    """
    model.eval()
    mu, logvar = model.encode(x, cond)

    if use_mean:
        z = mu
    else:
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

    x_cf = model.decode(z, cond_cf)
    return x_cf


@torch.no_grad()
def batch_counterfactual(
    model: torch.nn.Module,
    x: torch.Tensor,
    cond: dict[str, torch.Tensor],
    target_condition: str,
    target_value: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate counterfactuals by changing a single condition.

    Args:
        model: VAE model
        x: Original expression (B, n_genes)
        cond: Original conditions dict
        target_condition: Name of condition to change (e.g., 'disease')
        target_value: New value for the condition

    Returns:
        Counterfactual expression (B, n_genes)
    """
    model.eval()
    x = x.to(device)
    cond = {k: v.to(device) for k, v in cond.items()}

    # Create counterfactual condition
    cond_cf = cond.copy()
    cond_cf[target_condition] = torch.full_like(cond[target_condition], target_value)

    return counterfactual_decode(model, x, cond, cond_cf)


def compute_counterfactual_effect(
    x: np.ndarray,
    x_cf: np.ndarray,
    gene_names: list[str] | None = None,
    top_k: int = 20,
) -> dict:
    """Analyze the effect of a counterfactual intervention.

    Args:
        x: Original expression (B, n_genes)
        x_cf: Counterfactual expression (B, n_genes)
        gene_names: Optional gene names
        top_k: Number of top affected genes to return

    Returns:
        Dict with effect statistics
    """
    # Per-gene mean change
    mean_change = (x_cf - x).mean(axis=0)
    abs_mean_change = np.abs(mean_change)

    # Top affected genes
    top_idx = np.argsort(abs_mean_change)[-top_k:][::-1]

    if gene_names is not None:
        top_genes = [(gene_names[i], mean_change[i]) for i in top_idx]
    else:
        top_genes = [(f"gene_{i}", mean_change[i]) for i in top_idx]

    # Overall statistics
    return {
        "mean_absolute_change": abs_mean_change.mean(),
        "max_change": abs_mean_change.max(),
        "n_upregulated": (mean_change > 0.5).sum(),
        "n_downregulated": (mean_change < -0.5).sum(),
        "top_affected_genes": top_genes,
        "mean_change_per_gene": mean_change,
    }


def intervention_consistency(
    model: torch.nn.Module,
    x: torch.Tensor,
    cond: dict[str, torch.Tensor],
    target_condition: str,
    n_samples: int = 10,
    device: str = "cpu",
) -> dict[str, float]:
    """Check consistency of counterfactual predictions.

    Generates multiple counterfactuals with sampling and measures variance.

    Args:
        model: VAE model
        x: Original expression (B, n_genes)
        cond: Original conditions
        target_condition: Condition to intervene on
        n_samples: Number of samples to generate
        device: Device

    Returns:
        Dict with consistency metrics
    """
    model.eval()
    x = x.to(device)
    cond = {k: v.to(device) for k, v in cond.items()}

    # Get unique values for target condition
    unique_vals = torch.unique(cond[target_condition])

    all_cfs = []
    for val in unique_vals:
        cond_cf = cond.copy()
        cond_cf[target_condition] = torch.full_like(cond[target_condition], val.item())

        # Generate multiple samples
        samples = []
        for _ in range(n_samples):
            x_cf = counterfactual_decode(model, x, cond, cond_cf, use_mean=False)
            samples.append(x_cf.cpu().numpy())

        samples = np.stack(samples, axis=0)  # (n_samples, B, n_genes)
        all_cfs.append(samples)

    # Compute variance across samples
    all_cfs = np.stack(all_cfs, axis=0)  # (n_conditions, n_samples, B, n_genes)
    sample_variance = all_cfs.var(axis=1).mean()  # Variance across samples

    # Compute effect size (variance across conditions)
    condition_variance = all_cfs.mean(axis=1).var(axis=0).mean()

    return {
        "sample_variance": float(sample_variance),
        "condition_variance": float(condition_variance),
        "signal_to_noise": float(condition_variance / (sample_variance + 1e-8)),
    }
