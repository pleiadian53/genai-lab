"""PyTorch Dataset wrapper for single-cell AnnData objects.

This module provides a PyTorch-compatible Dataset that wraps AnnData objects,
preserving raw counts and library size for NB/ZINB generative models.

Usage:
    from genailab.data.sc_dataset import SingleCellDataset
    
    # Load preprocessed data
    adata = sc.read_h5ad("pbmc3k_counts.h5ad")
    
    # Create dataset
    dataset = SingleCellDataset(
        adata,
        layer="counts",  # Use raw counts
        condition_keys=["cell_type"],  # Optional conditions
    )
    
    # Use with DataLoader
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


class SingleCellDataset(Dataset):
    """PyTorch Dataset for single-cell expression data from AnnData.
    
    Designed for use with NB/ZINB generative models that require:
    - Raw counts (not normalized)
    - Library size per cell
    - Optional condition labels
    
    Args:
        adata: AnnData object with expression data
        layer: Which layer to use for expression. Options:
            - None: use adata.X
            - "counts": use adata.layers["counts"]
            - Any other layer name
        condition_keys: List of obs column names to use as conditions
        gene_subset: Optional list of gene names or boolean mask to subset
        return_gene_ids: Whether to return gene indices (for gene-level models)
    """
    
    def __init__(
        self,
        adata,  # sc.AnnData, but we don't type-hint to avoid import
        layer: str | None = "counts",
        condition_keys: list[str] | None = None,
        gene_subset: list[str] | np.ndarray | None = None,
        return_gene_ids: bool = False,
    ):
        self.adata = adata
        self.layer = layer
        self.condition_keys = condition_keys or []
        self.return_gene_ids = return_gene_ids
        
        # Get expression matrix
        if layer is None:
            X = adata.X
        elif layer in adata.layers:
            X = adata.layers[layer]
        else:
            raise ValueError(
                f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}"
            )
        
        # Handle gene subset
        if gene_subset is not None:
            if isinstance(gene_subset, list):
                # Gene names
                gene_mask = adata.var_names.isin(gene_subset)
                X = X[:, gene_mask]
                self._gene_names = adata.var_names[gene_mask].tolist()
            else:
                # Boolean mask or indices
                X = X[:, gene_subset]
                self._gene_names = adata.var_names[gene_subset].tolist()
        else:
            self._gene_names = adata.var_names.tolist()
        
        # Convert to dense numpy array
        if sparse.issparse(X):
            self.X = np.asarray(X.todense(), dtype=np.float32)
        else:
            self.X = np.asarray(X, dtype=np.float32)
        
        # Library size (total counts per cell)
        if "library_size" in adata.obs:
            self.library_size = adata.obs["library_size"].values.astype(np.float32)
        else:
            # Compute from the expression matrix we're using
            self.library_size = self.X.sum(axis=1).astype(np.float32)
        
        # Build condition encodings
        self.conditions = {}
        self.condition_dims = {}
        for key in self.condition_keys:
            if key not in adata.obs:
                raise ValueError(f"Condition key '{key}' not found in adata.obs")
            
            values = adata.obs[key]
            if values.dtype.name == "category":
                # Categorical: encode as integers
                codes = values.cat.codes.values.astype(np.int64)
                n_categories = len(values.cat.categories)
            else:
                # Try to convert to categorical
                cat = values.astype("category")
                codes = cat.cat.codes.values.astype(np.int64)
                n_categories = len(cat.cat.categories)
            
            self.conditions[key] = codes
            self.condition_dims[key] = n_categories
        
        self._n_cells = self.X.shape[0]
        self._n_genes = self.X.shape[1]
    
    def __len__(self) -> int:
        return self._n_cells
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single cell's data.
        
        Returns:
            Dict with:
                - 'x': Expression vector (n_genes,)
                - 'library_size': Total counts for this cell
                - 'cond': Dict of condition values (if any)
                - 'gene_ids': Gene indices (if return_gene_ids=True)
        """
        item = {
            "x": torch.from_numpy(self.X[idx]),
            "library_size": torch.tensor(self.library_size[idx]),
        }
        
        if self.condition_keys:
            item["cond"] = {
                key: torch.tensor(self.conditions[key][idx])
                for key in self.condition_keys
            }
        
        if self.return_gene_ids:
            item["gene_ids"] = torch.arange(self._n_genes)
        
        return item
    
    @property
    def n_genes(self) -> int:
        """Number of genes in the dataset."""
        return self._n_genes
    
    @property
    def n_cells(self) -> int:
        """Number of cells in the dataset."""
        return self._n_cells
    
    @property
    def gene_names(self) -> list[str]:
        """List of gene names."""
        return self._gene_names
    
    @property
    def n_conditions(self) -> dict[str, int]:
        """Number of categories for each condition."""
        return self.condition_dims
    
    def get_library_size_stats(self) -> dict[str, float]:
        """Return library size statistics for normalization."""
        return {
            "mean": float(self.library_size.mean()),
            "std": float(self.library_size.std()),
            "median": float(np.median(self.library_size)),
            "min": float(self.library_size.min()),
            "max": float(self.library_size.max()),
        }
    
    def subset_cells(self, indices: np.ndarray | list[int]) -> "SingleCellDataset":
        """Create a new dataset with a subset of cells.
        
        Args:
            indices: Cell indices to keep
            
        Returns:
            New SingleCellDataset with subset of cells
        """
        return SingleCellDataset(
            self.adata[indices],
            layer=self.layer,
            condition_keys=self.condition_keys,
            return_gene_ids=self.return_gene_ids,
        )


def collate_sc_batch(batch: list[dict]) -> dict[str, Any]:
    """Custom collate function for SingleCellDataset.
    
    Handles the nested 'cond' dictionary properly.
    
    Args:
        batch: List of items from SingleCellDataset.__getitem__
        
    Returns:
        Batched tensors
    """
    result = {
        "x": torch.stack([item["x"] for item in batch]),
        "library_size": torch.stack([item["library_size"] for item in batch]),
    }
    
    if "cond" in batch[0]:
        result["cond"] = {
            key: torch.stack([item["cond"][key] for item in batch])
            for key in batch[0]["cond"]
        }
    
    if "gene_ids" in batch[0]:
        result["gene_ids"] = batch[0]["gene_ids"]  # Same for all items
    
    return result
