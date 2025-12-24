"""Highly Variable Gene (HVG) selection utilities.

HVG selection is a critical preprocessing step that:
- Reduces dimensionality while preserving biological signal
- Focuses the model on genes with meaningful variation
- Removes technical noise from lowly-expressed genes

IMPORTANT: HVG selection uses temporary normalization, but the
raw counts are preserved for generative modeling.

Usage:
    from genailab.analysis.hvg import select_hvg, plot_hvg_dispersion
    
    adata = select_hvg(adata, n_top_genes=2000)
    plot_hvg_dispersion(adata)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import scanpy as sc


def select_hvg(
    adata: "sc.AnnData",
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    subset: bool = False,
    batch_key: str | None = None,
    inplace: bool = True,
) -> "sc.AnnData":
    """Select highly variable genes while preserving raw counts.
    
    This function:
    1. Stores raw counts in adata.layers["counts"] (if not already)
    2. Creates a temporary normalized copy for HVG selection
    3. Identifies HVGs using the specified method
    4. Transfers HVG annotations back to original (raw count) data
    
    The raw counts in adata.X are NEVER modified.
    
    Args:
        adata: AnnData object with raw counts
        n_top_genes: Number of HVGs to select
        flavor: HVG selection method ("seurat", "seurat_v3", "cell_ranger")
        subset: If True, subset adata to HVGs only
        batch_key: If provided, select HVGs within each batch
        inplace: If True, modify adata in place
        
    Returns:
        AnnData with HVG annotations in adata.var["highly_variable"]
        
    Example:
        >>> adata = sc.read_h5ad("counts.h5ad")
        >>> adata = select_hvg(adata, n_top_genes=2000)
        >>> print(f"Selected {adata.var.highly_variable.sum()} HVGs")
    """
    import scanpy as sc
    from scipy import sparse
    
    if not inplace:
        adata = adata.copy()
    
    # Store raw counts if not already stored
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
        print("Stored raw counts in adata.layers['counts']")
    
    # Create temporary normalized copy for HVG selection
    # This is the ONLY place we normalize - and it's temporary
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    
    # Determine flavor based on matrix type if using seurat_v3
    if flavor == "seurat_v3" and not sparse.issparse(adata.X):
        print("Note: seurat_v3 works best with sparse matrices, using 'seurat' instead")
        flavor = "seurat"
    
    # Select HVGs
    sc.pp.highly_variable_genes(
        adata_norm,
        n_top_genes=n_top_genes,
        flavor=flavor,
        batch_key=batch_key,
    )
    
    # Transfer HVG annotations to original data
    hvg_cols = ["highly_variable", "means", "dispersions", "dispersions_norm"]
    for col in hvg_cols:
        if col in adata_norm.var.columns:
            adata.var[col] = adata_norm.var[col]
    
    n_hvg = adata.var["highly_variable"].sum()
    print(f"Selected {n_hvg} highly variable genes (flavor={flavor})")
    
    # Optionally subset
    if subset:
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"Subset to {adata.n_vars} HVGs")
    
    return adata


def plot_hvg_dispersion(
    adata: "sc.AnnData",
    log: bool = True,
    figsize: tuple = (10, 5),
) -> None:
    """Plot mean-dispersion relationship for HVG selection.
    
    This visualization shows:
    - How genes are distributed by mean expression and dispersion
    - Which genes were selected as highly variable
    - The relationship between mean and variance (overdispersion)
    
    Args:
        adata: AnnData with HVG annotations (run select_hvg first)
        log: If True, use log scale for axes
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    if "highly_variable" not in adata.var.columns:
        raise ValueError("Run select_hvg() first to compute HVG annotations")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Mean vs dispersion
    hvg = adata.var["highly_variable"]
    means = adata.var["means"]
    dispersions = adata.var["dispersions_norm"] if "dispersions_norm" in adata.var else adata.var["dispersions"]
    
    axes[0].scatter(means[~hvg], dispersions[~hvg], s=3, alpha=0.3, label="Non-HVG")
    axes[0].scatter(means[hvg], dispersions[hvg], s=3, alpha=0.5, c='red', label="HVG")
    axes[0].set_xlabel("Mean expression")
    axes[0].set_ylabel("Normalized dispersion")
    axes[0].set_title("Mean vs Dispersion")
    axes[0].legend()
    if log:
        axes[0].set_xscale('log')
    
    # Dispersion histogram
    axes[1].hist(dispersions[~hvg], bins=50, alpha=0.5, label="Non-HVG", density=True)
    axes[1].hist(dispersions[hvg], bins=50, alpha=0.5, label="HVG", density=True)
    axes[1].set_xlabel("Normalized dispersion")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Dispersion Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def get_hvg_subset(
    adata: "sc.AnnData",
    as_tensor: bool = False,
) -> "sc.AnnData | tuple":
    """Get HVG-subset data ready for model training.
    
    This is a convenience function that:
    1. Subsets to HVGs
    2. Extracts raw counts (not normalized)
    3. Optionally converts to PyTorch tensors
    
    Args:
        adata: AnnData with HVG annotations
        as_tensor: If True, return PyTorch tensors
        
    Returns:
        If as_tensor=False: AnnData subset to HVGs
        If as_tensor=True: (X_tensor, library_size_tensor, adata_subset)
    """
    import numpy as np
    
    if "highly_variable" not in adata.var.columns:
        raise ValueError("Run select_hvg() first")
    
    # Subset to HVGs
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    
    # Get raw counts (from layers if available, else from X)
    if "counts" in adata_hvg.layers:
        X = adata_hvg.layers["counts"]
    else:
        X = adata_hvg.X
    
    # Convert to dense if sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Get library size
    if "library_size" in adata_hvg.obs.columns:
        lib_size = adata_hvg.obs["library_size"].values
    else:
        lib_size = X.sum(axis=1)
    
    if not as_tensor:
        return adata_hvg
    
    # Convert to PyTorch tensors
    import torch
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    lib_size_tensor = torch.tensor(lib_size, dtype=torch.float32).unsqueeze(1)
    
    return X_tensor, lib_size_tensor, adata_hvg


def compute_gene_stats(adata: "sc.AnnData") -> "sc.AnnData":
    """Compute per-gene statistics useful for understanding overdispersion.
    
    Adds to adata.var:
    - mean: Mean expression
    - var: Variance
    - cv2: Coefficient of variation squared (var/mean^2)
    - fano: Fano factor (var/mean) - measures overdispersion
    - dropout_rate: Fraction of cells with zero expression
    
    Args:
        adata: AnnData object
        
    Returns:
        AnnData with gene statistics in var
    """
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    
    # Avoid division by zero
    mean_safe = np.where(mean > 0, mean, 1)
    
    adata.var["mean"] = mean
    adata.var["var"] = var
    adata.var["cv2"] = var / (mean_safe ** 2)
    adata.var["fano"] = var / mean_safe  # Fano factor
    adata.var["dropout_rate"] = (X == 0).mean(axis=0)
    
    # For Poisson, Fano = 1. For NB with overdispersion, Fano > 1
    fano_median = np.median(adata.var["fano"][mean > 0])
    print(f"Median Fano factor: {fano_median:.2f} (Poisson=1, overdispersed>1)")
    
    return adata


if __name__ == "__main__":
    # Demo usage
    import scanpy as sc
    from genailab.data.paths import get_data_paths
    
    paths = get_data_paths()
    adata_path = paths.scrna_processed("pbmc3k", "counts.h5ad")
    
    if adata_path.exists():
        print(f"Loading {adata_path}...")
        adata = sc.read_h5ad(adata_path)
        
        print("\n" + "=" * 60)
        print("Computing gene statistics...")
        adata = compute_gene_stats(adata)
        
        print("\n" + "=" * 60)
        print("Selecting HVGs...")
        adata = select_hvg(adata, n_top_genes=2000)
        
        print("\n" + "=" * 60)
        print("Getting HVG subset for training...")
        X, lib_size, adata_hvg = get_hvg_subset(adata, as_tensor=True)
        print(f"X shape: {X.shape}")
        print(f"Library size shape: {lib_size.shape}")
    else:
        print(f"File not found: {adata_path}")
