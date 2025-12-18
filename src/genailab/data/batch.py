"""Batch effect handling and harmonization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BatchEncoder:
    """Encode batch/study labels as integers for conditioning."""

    def __init__(self):
        self._label_to_idx: dict[str, int] = {}
        self._idx_to_label: dict[int, str] = {}
        self._fitted = False

    def fit(self, labels: list[str] | np.ndarray | pd.Series) -> "BatchEncoder":
        """Fit encoder on batch labels."""
        unique_labels = sorted(set(labels))
        self._label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self._idx_to_label = {idx: label for label, idx in self._label_to_idx.items()}
        self._fitted = True
        return self

    def transform(self, labels: list[str] | np.ndarray | pd.Series) -> np.ndarray:
        """Transform batch labels to integer indices."""
        if not self._fitted:
            raise RuntimeError("BatchEncoder must be fitted before transform")
        return np.array([self._label_to_idx[label] for label in labels], dtype=np.int64)

    def fit_transform(self, labels: list[str] | np.ndarray | pd.Series) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(labels).transform(labels)

    def inverse_transform(self, indices: np.ndarray) -> list[str]:
        """Convert indices back to labels."""
        if not self._fitted:
            raise RuntimeError("BatchEncoder must be fitted before inverse_transform")
        return [self._idx_to_label[idx] for idx in indices]

    @property
    def n_categories(self) -> int:
        """Number of unique batch categories."""
        return len(self._label_to_idx)

    @property
    def categories(self) -> list[str]:
        """List of category labels in order."""
        return [self._idx_to_label[i] for i in range(len(self._idx_to_label))]


def harmonize_batches(
    expression: np.ndarray,
    batch_labels: np.ndarray,
    method: str = "combat",
) -> np.ndarray:
    """
    Harmonize expression data across batches.

    Args:
        expression: Expression matrix (samples x genes)
        batch_labels: Batch labels for each sample
        method: Harmonization method ('combat', 'limma', 'simple')

    Returns:
        Harmonized expression matrix
    """
    if method == "simple":
        # Simple mean-centering per batch
        harmonized = expression.copy()
        for batch in np.unique(batch_labels):
            mask = batch_labels == batch
            batch_mean = expression[mask].mean(axis=0)
            global_mean = expression.mean(axis=0)
            harmonized[mask] = expression[mask] - batch_mean + global_mean
        return harmonized

    elif method == "combat":
        # TODO: Implement ComBat or use pycombat
        raise NotImplementedError(
            "ComBat harmonization not yet implemented. "
            "Consider using pycombat: pip install pycombat"
        )

    elif method == "limma":
        # TODO: Implement limma removeBatchEffect equivalent
        raise NotImplementedError(
            "limma-style harmonization not yet implemented."
        )

    else:
        raise ValueError(f"Unknown harmonization method: {method}")


class ConditionMetadata:
    """Container for multiple condition variables (tissue, disease, batch, etc.)."""

    def __init__(self):
        self.encoders: dict[str, BatchEncoder] = {}
        self.encoded: dict[str, np.ndarray] = {}

    def add_condition(
        self,
        name: str,
        labels: list[str] | np.ndarray | pd.Series,
    ) -> "ConditionMetadata":
        """Add and encode a condition variable."""
        encoder = BatchEncoder()
        self.encoders[name] = encoder
        self.encoded[name] = encoder.fit_transform(labels)
        return self

    def get_n_categories(self) -> dict[str, int]:
        """Get number of categories for each condition."""
        return {name: enc.n_categories for name, enc in self.encoders.items()}

    def get_encoded(self) -> dict[str, np.ndarray]:
        """Get all encoded conditions."""
        return self.encoded.copy()

    def __getitem__(self, name: str) -> np.ndarray:
        return self.encoded[name]
