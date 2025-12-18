"""Dataset loaders for GEO, TCGA, and single-cell data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseExpressionDataset(Dataset, ABC):
    """Abstract base class for expression datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        pass

    @property
    @abstractmethod
    def n_genes(self) -> int:
        pass

    @property
    @abstractmethod
    def gene_names(self) -> list[str]:
        pass


class ToyBulkDataset(Dataset):
    """
    Synthetic bulk expression with tissue/disease/batch effects.
    Useful to validate that conditioning works and counterfactuals behave sanely.
    """

    def __init__(
        self,
        n: int = 10000,
        n_genes: int = 2000,
        n_tissues: int = 5,
        n_diseases: int = 3,
        n_batches: int = 8,
        seed: int = 7,
    ):
        rng = np.random.default_rng(seed)
        self.n = n
        self.n_genes = n_genes
        self.n_tissues = n_tissues
        self.n_diseases = n_diseases
        self.n_batches = n_batches

        tissue = rng.integers(0, n_tissues, size=n)
        disease = rng.integers(0, n_diseases, size=n)
        batch = rng.integers(0, n_batches, size=n)

        # Latent biology z_true + additive condition effects
        z_dim = 20
        z = rng.normal(size=(n, z_dim)).astype(np.float32)

        W = rng.normal(scale=0.2, size=(z_dim, n_genes)).astype(np.float32)
        base = z @ W

        tissue_eff = rng.normal(scale=0.4, size=(n_tissues, n_genes)).astype(np.float32)
        disease_eff = rng.normal(scale=0.3, size=(n_diseases, n_genes)).astype(np.float32)
        batch_eff = rng.normal(scale=0.2, size=(n_batches, n_genes)).astype(np.float32)

        x = base + tissue_eff[tissue] + disease_eff[disease] + batch_eff[batch]
        x += rng.normal(scale=0.3, size=x.shape).astype(np.float32)

        # "log1p-like" range: standardize per gene
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

        self.x = torch.tensor(x, dtype=torch.float32)
        self.cond = {
            "tissue": torch.tensor(tissue, dtype=torch.long),
            "disease": torch.tensor(disease, dtype=torch.long),
            "batch": torch.tensor(batch, dtype=torch.long),
        }
        self._gene_names = [f"gene_{i}" for i in range(n_genes)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"x": self.x[idx], "cond": {k: v[idx] for k, v in self.cond.items()}}

    @property
    def gene_names(self) -> list[str]:
        return self._gene_names


class GEOLoader:
    """Loader for Gene Expression Omnibus (GEO) datasets."""

    def __init__(self, cache_dir: str | Path = ".cache/geo"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, accession: str) -> pd.DataFrame:
        """
        Load a GEO dataset by accession number.

        Args:
            accession: GEO accession (e.g., 'GSE12345')

        Returns:
            Expression matrix as DataFrame (samples x genes)
        """
        # TODO: Implement GEO fetching via GEOparse or direct download
        raise NotImplementedError(
            f"GEO loading for {accession} not yet implemented. "
            "Consider using GEOparse: pip install GEOparse"
        )


class TCGALoader:
    """Loader for TCGA (The Cancer Genome Atlas) datasets."""

    def __init__(self, cache_dir: str | Path = ".cache/tcga"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, cancer_type: str, data_type: str = "rnaseq") -> pd.DataFrame:
        """
        Load TCGA data for a specific cancer type.

        Args:
            cancer_type: TCGA cancer type code (e.g., 'BRCA', 'LUAD')
            data_type: Type of data ('rnaseq', 'mirna', etc.)

        Returns:
            Expression matrix as DataFrame (samples x genes)
        """
        # TODO: Implement TCGA fetching via TCGAbiolinks or recount3
        raise NotImplementedError(
            f"TCGA loading for {cancer_type}/{data_type} not yet implemented. "
            "Consider using recount3 or downloading from GDC portal."
        )


def split_dataset(
    ds: Dataset,
    val_frac: float = 0.1,
    seed: int = 1,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train and validation subsets."""
    from torch.utils.data import Subset

    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    nval = int(len(ds) * val_frac)
    val_idx = idx[:nval]
    tr_idx = idx[nval:]
    return Subset(ds, tr_idx.tolist()), Subset(ds, val_idx.tolist())
