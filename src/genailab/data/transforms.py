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
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize expression matrix (samples x genes) to zero mean, unit variance per gene.

    Args:
        x: Expression matrix (samples x genes)
        mean: Pre-computed mean per gene (for applying to new data)
        std: Pre-computed std per gene (for applying to new data)
        eps: Small constant for numerical stability

    Returns:
        Tuple of (standardized_x, mean, std)
    """
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)

    x_std = (x - mean) / (std + eps)
    return x_std, mean, std


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
