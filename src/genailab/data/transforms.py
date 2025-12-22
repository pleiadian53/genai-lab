"""Data transforms for gene expression preprocessing."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def log1p_transform(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Apply log1p transform: log(1 + x)."""
    if isinstance(x, torch.Tensor):
        return torch.log1p(x)
    return np.log1p(x)


def standardize(
    x: np.ndarray,
    axis: int | None = 0,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    eps: float = 1e-6,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize array to zero mean, unit variance along specified axis.

    Args:
        x: Input array (e.g., samples x genes)
        axis: Axis along which to compute mean/std. Default 0 (per-gene for samples x genes).
              Use None to standardize over all elements.
        mean: Pre-computed mean (for applying to new data). If None, computed from x.
        std: Pre-computed std (for applying to new data). If None, computed from x.
        eps: Small constant for numerical stability
        return_stats: If True, return (standardized_x, mean, std). If False, return only standardized_x.

    Returns:
        If return_stats=False: standardized array
        If return_stats=True: tuple of (standardized_x, mean, std)

    Examples:
        # Simple standardization (per-gene, axis=0)
        x_std = standardize(x)

        # Fit on train, apply to test
        x_train_std, mu, sigma = standardize(x_train, return_stats=True)
        x_test_std = standardize(x_test, mean=mu, std=sigma)

        # Per-sample standardization (axis=1)
        x_std = standardize(x, axis=1)
    """
    if mean is None:
        mean = x.mean(axis=axis, keepdims=True)
    if std is None:
        std = x.std(axis=axis, keepdims=True)

    x_std = (x - mean) / (std + eps)

    if return_stats:
        return x_std, mean.squeeze(), std.squeeze()
    return x_std


def hvg_filter(
    x: np.ndarray,
    n_top_genes: int = 2000,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select highly variable genes (HVG) based on mean-variance relationship.

    This is a simplified version; for production, use scanpy.pp.highly_variable_genes.

    Args:
        x: Expression matrix (samples x genes), assumed log-normalized
        n_top_genes: Number of top variable genes to select
        min_mean: Minimum mean expression threshold
        max_mean: Maximum mean expression threshold
        min_disp: Minimum dispersion threshold

    Returns:
        Tuple of (filtered_x, gene_indices)
    """
    mean = x.mean(axis=0)
    var = x.var(axis=0)

    # Dispersion = var / mean (for log data, this is approximate)
    disp = var / (mean + 1e-6)

    # Apply thresholds
    mask = (mean >= min_mean) & (mean <= max_mean) & (disp >= min_disp)

    # If too few genes pass, just take top by variance
    if mask.sum() < n_top_genes:
        top_idx = np.argsort(var)[-n_top_genes:]
    else:
        # Among passing genes, take top by dispersion
        passing_idx = np.where(mask)[0]
        passing_disp = disp[passing_idx]
        top_in_passing = np.argsort(passing_disp)[-n_top_genes:]
        top_idx = passing_idx[top_in_passing]

    return x[:, top_idx], top_idx


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor:
    """Convert numpy array to torch tensor."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)
