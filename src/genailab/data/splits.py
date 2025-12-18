"""Dataset splitting utilities with awareness of biological structure."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset, Subset


def donor_aware_split(
    n_samples: int,
    donor_ids: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data ensuring samples from the same donor stay together.

    This prevents data leakage when the same donor has multiple samples
    (e.g., multiple tissues, timepoints, or technical replicates).

    Args:
        n_samples: Total number of samples
        donor_ids: Donor ID for each sample
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(seed)

    unique_donors = np.unique(donor_ids)
    n_donors = len(unique_donors)

    # Shuffle donors
    rng.shuffle(unique_donors)

    # Split donors
    n_test_donors = max(1, int(n_donors * test_frac))
    n_val_donors = max(1, int(n_donors * val_frac))

    test_donors = set(unique_donors[:n_test_donors])
    val_donors = set(unique_donors[n_test_donors : n_test_donors + n_val_donors])
    train_donors = set(unique_donors[n_test_donors + n_val_donors :])

    # Map back to sample indices
    train_idx = np.where([d in train_donors for d in donor_ids])[0]
    val_idx = np.where([d in val_donors for d in donor_ids])[0]
    test_idx = np.where([d in test_donors for d in donor_ids])[0]

    return train_idx, val_idx, test_idx


def tissue_aware_split(
    n_samples: int,
    tissue_ids: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data with stratification by tissue type.

    Ensures each split has similar tissue proportions.

    Args:
        n_samples: Total number of samples
        tissue_ids: Tissue ID for each sample
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed
        stratify: Whether to stratify by tissue

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(seed)

    if not stratify:
        # Simple random split
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        n_test = int(n_samples * test_frac)
        n_val = int(n_samples * val_frac)
        return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]

    # Stratified split
    train_idx, val_idx, test_idx = [], [], []

    for tissue in np.unique(tissue_ids):
        tissue_mask = tissue_ids == tissue
        tissue_indices = np.where(tissue_mask)[0]
        rng.shuffle(tissue_indices)

        n_tissue = len(tissue_indices)
        n_test = max(1, int(n_tissue * test_frac))
        n_val = max(1, int(n_tissue * val_frac))

        test_idx.extend(tissue_indices[:n_test])
        val_idx.extend(tissue_indices[n_test : n_test + n_val])
        train_idx.extend(tissue_indices[n_test + n_val :])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def create_subset(ds: Dataset, indices: np.ndarray) -> Subset:
    """Create a Subset from indices."""
    return Subset(ds, indices.tolist())


def kfold_by_group(
    n_samples: int,
    group_ids: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    K-fold cross-validation with group awareness.

    Samples from the same group (e.g., donor, study) are kept together.

    Args:
        n_samples: Total number of samples
        group_ids: Group ID for each sample
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    rng = np.random.default_rng(seed)

    unique_groups = np.unique(group_ids)
    rng.shuffle(unique_groups)

    # Assign groups to folds
    fold_assignment = np.array_split(unique_groups, n_folds)

    folds = []
    for fold_idx in range(n_folds):
        val_groups = set(fold_assignment[fold_idx])
        val_idx = np.where([g in val_groups for g in group_ids])[0]
        train_idx = np.where([g not in val_groups for g in group_ids])[0]
        folds.append((train_idx, val_idx))

    return folds
