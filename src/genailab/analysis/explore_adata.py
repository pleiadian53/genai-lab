"""Utilities for exploring and understanding AnnData objects.

This module provides functions to inspect h5ad files and understand
the structure of scRNA-seq data before modeling.

Usage:
    from genailab.analysis.explore_adata import summarize_adata, inspect_sparsity
    
    adata = sc.read_h5ad("data/scrna/pbmc3k/processed/counts.h5ad")
    summarize_adata(adata)
    inspect_sparsity(adata)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import scanpy as sc
    import matplotlib.pyplot as plt


def summarize_adata(adata: "sc.AnnData", verbose: bool = True) -> dict:
    """Summarize key properties of an AnnData object.
    
    This is the first thing to run when loading a new h5ad file.
    
    Args:
        adata: AnnData object to summarize
        verbose: If True, print summary to console
        
    Returns:
        Dictionary with summary statistics
        
    Example:
        >>> adata = sc.read_h5ad("counts.h5ad")
        >>> summary = summarize_adata(adata)
        AnnData Summary
        ===============
        Shape: 2698 cells × 13714 genes
        ...
    """
    # Basic shape
    n_cells, n_genes = adata.shape
    
    # Sparsity
    if hasattr(adata.X, 'toarray'):
        # Sparse matrix
        total_elements = n_cells * n_genes
        nonzero = adata.X.nnz
        sparsity = 1 - (nonzero / total_elements)
        matrix_type = "sparse"
    else:
        # Dense matrix
        nonzero = np.count_nonzero(adata.X)
        total_elements = adata.X.size
        sparsity = 1 - (nonzero / total_elements)
        matrix_type = "dense"
    
    # Library size stats
    if "library_size" in adata.obs.columns:
        lib_size = adata.obs["library_size"]
    else:
        lib_size = np.array(adata.X.sum(axis=1)).ravel()
    
    # Count statistics
    X_data = adata.X.data if hasattr(adata.X, 'data') else adata.X.ravel()
    X_nonzero = X_data[X_data > 0] if hasattr(adata.X, 'data') else X_data[X_data > 0]
    
    summary = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "matrix_type": matrix_type,
        "sparsity": sparsity,
        "nonzero_entries": nonzero,
        "library_size_min": lib_size.min(),
        "library_size_max": lib_size.max(),
        "library_size_mean": lib_size.mean(),
        "library_size_median": np.median(lib_size),
        "count_min": X_nonzero.min() if len(X_nonzero) > 0 else 0,
        "count_max": X_nonzero.max() if len(X_nonzero) > 0 else 0,
        "count_mean": X_nonzero.mean() if len(X_nonzero) > 0 else 0,
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "layers": list(adata.layers.keys()) if adata.layers else [],
    }
    
    if verbose:
        print("AnnData Summary")
        print("=" * 50)
        print(f"Shape: {n_cells:,} cells × {n_genes:,} genes")
        print(f"Matrix type: {matrix_type}")
        print(f"Sparsity: {sparsity:.1%} zeros")
        print(f"Non-zero entries: {nonzero:,}")
        print()
        print("Library Size (total counts per cell)")
        print(f"  Min: {lib_size.min():,.0f}")
        print(f"  Max: {lib_size.max():,.0f}")
        print(f"  Mean: {lib_size.mean():,.1f}")
        print(f"  Median: {np.median(lib_size):,.1f}")
        print()
        print("Count Values (non-zero)")
        print(f"  Min: {summary['count_min']:.0f}")
        print(f"  Max: {summary['count_max']:.0f}")
        print(f"  Mean: {summary['count_mean']:.2f}")
        print()
        print(f"Cell metadata (obs): {summary['obs_columns']}")
        print(f"Gene metadata (var): {summary['var_columns']}")
        if summary['layers']:
            print(f"Layers: {summary['layers']}")
    
    return summary


def inspect_sparsity(
    adata: "sc.AnnData",
    plot: bool = True,
    save: str | None = None,
    figsize: tuple = (12, 4),
) -> dict:
    """Analyze sparsity patterns in the count matrix.
    
    Understanding sparsity is crucial for:
    - Choosing between NB vs ZINB likelihoods
    - Understanding dropout vs biological zeros
    - Validating preprocessing
    
    Args:
        adata: AnnData object
        plot: If True, create visualization
        save: If provided, save figure to this filename in results/figures/exploration/
              If None, display interactively
        figsize: Figure size for plots
        
    Returns:
        Dictionary with sparsity statistics
    """
    import matplotlib.pyplot as plt
    from genailab.data.paths import get_results_paths
    
    # Get count matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    # Per-cell statistics
    zeros_per_cell = (X == 0).sum(axis=1)
    zero_fraction_per_cell = zeros_per_cell / X.shape[1]
    
    # Per-gene statistics
    zeros_per_gene = (X == 0).sum(axis=0)
    zero_fraction_per_gene = zeros_per_gene / X.shape[0]
    
    # Cells expressing each gene
    cells_per_gene = (X > 0).sum(axis=0)
    
    stats = {
        "overall_sparsity": (X == 0).sum() / X.size,
        "zero_fraction_per_cell_mean": zero_fraction_per_cell.mean(),
        "zero_fraction_per_cell_std": zero_fraction_per_cell.std(),
        "zero_fraction_per_gene_mean": zero_fraction_per_gene.mean(),
        "zero_fraction_per_gene_std": zero_fraction_per_gene.std(),
        "genes_with_100pct_zeros": (zeros_per_gene == X.shape[0]).sum(),
        "cells_with_over_99pct_zeros": (zero_fraction_per_cell > 0.99).sum(),
    }
    
    print("Sparsity Analysis")
    print("=" * 50)
    print(f"Overall sparsity: {stats['overall_sparsity']:.1%}")
    print(f"Per-cell zero fraction: {stats['zero_fraction_per_cell_mean']:.1%} ± {stats['zero_fraction_per_cell_std']:.1%}")
    print(f"Per-gene zero fraction: {stats['zero_fraction_per_gene_mean']:.1%} ± {stats['zero_fraction_per_gene_std']:.1%}")
    print(f"Genes with 100% zeros: {stats['genes_with_100pct_zeros']}")
    print(f"Cells with >99% zeros: {stats['cells_with_over_99pct_zeros']}")
    
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Zero fraction per cell
        axes[0].hist(zero_fraction_per_cell, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel("Zero fraction")
        axes[0].set_ylabel("Number of cells")
        axes[0].set_title("Sparsity per Cell")
        axes[0].axvline(zero_fraction_per_cell.mean(), color='red', linestyle='--', 
                        label=f'Mean: {zero_fraction_per_cell.mean():.1%}')
        axes[0].legend()
        
        # Zero fraction per gene
        axes[1].hist(zero_fraction_per_gene, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Zero fraction")
        axes[1].set_ylabel("Number of genes")
        axes[1].set_title("Sparsity per Gene")
        axes[1].axvline(zero_fraction_per_gene.mean(), color='red', linestyle='--',
                        label=f'Mean: {zero_fraction_per_gene.mean():.1%}')
        axes[1].legend()
        
        # Cells expressing each gene (log scale)
        axes[2].hist(cells_per_gene, bins=50, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel("Number of cells expressing gene")
        axes[2].set_ylabel("Number of genes")
        axes[2].set_title("Gene Detection Frequency")
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        
        if save:
            results = get_results_paths()
            save_path = results.figure("exploration", save)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        else:
            plt.show()
    
    return stats


def plot_library_size(
    adata: "sc.AnnData",
    log_scale: bool = True,
    save: str | None = None,
    figsize: tuple = (10, 4),
) -> None:
    """Plot library size distribution.
    
    Library size (total counts per cell) is critical for NB/ZINB models.
    This visualization helps understand sequencing depth variation.
    
    Args:
        adata: AnnData object
        log_scale: If True, use log scale for x-axis
        save: If provided, save figure to this filename in results/figures/exploration/
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from genailab.data.paths import get_results_paths
    
    if "library_size" in adata.obs.columns:
        lib_size = adata.obs["library_size"].values
    else:
        lib_size = np.array(adata.X.sum(axis=1)).ravel()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(lib_size, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Library size (total counts)")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("Library Size Distribution")
    if log_scale:
        axes[0].set_xscale('log')
    
    # Add statistics
    stats_text = f"Min: {lib_size.min():,.0f}\nMax: {lib_size.max():,.0f}\nMean: {lib_size.mean():,.0f}\nMedian: {np.median(lib_size):,.0f}"
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Rank plot
    sorted_lib = np.sort(lib_size)[::-1]
    axes[1].plot(range(len(sorted_lib)), sorted_lib)
    axes[1].set_xlabel("Cell rank")
    axes[1].set_ylabel("Library size")
    axes[1].set_title("Library Size Rank Plot")
    if log_scale:
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save:
        results = get_results_paths()
        save_path = results.figure("exploration", save)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_gene_expression(
    adata: "sc.AnnData",
    genes: list[str],
    log1p: bool = True,
    save: str | None = None,
    figsize: tuple = None,
) -> None:
    """Plot expression distribution for specific genes.
    
    Useful for understanding:
    - Overdispersion patterns
    - Zero inflation
    - Gene-specific behavior
    
    Args:
        adata: AnnData object
        genes: List of gene names to plot
        log1p: If True, apply log1p transform for visualization
        save: If provided, save figure to this filename in results/figures/exploration/
        figsize: Figure size (auto-calculated if None)
    """
    import matplotlib.pyplot as plt
    from genailab.data.paths import get_results_paths
    
    n_genes = len(genes)
    if figsize is None:
        figsize = (4 * n_genes, 4)
    
    fig, axes = plt.subplots(1, n_genes, figsize=figsize)
    if n_genes == 1:
        axes = [axes]
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    for ax, gene in zip(axes, genes):
        if gene not in adata.var_names:
            ax.set_title(f"{gene}\n(not found)")
            continue
        
        gene_idx = adata.var_names.get_loc(gene)
        expr = X[:, gene_idx]
        
        if log1p:
            expr_plot = np.log1p(expr)
            xlabel = "log1p(counts)"
        else:
            expr_plot = expr
            xlabel = "counts"
        
        # Separate zeros and non-zeros
        n_zeros = (expr == 0).sum()
        zero_frac = n_zeros / len(expr)
        
        ax.hist(expr_plot, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of cells")
        ax.set_title(f"{gene}\n{zero_frac:.1%} zeros, mean={expr.mean():.2f}")
    
    plt.tight_layout()
    
    if save:
        results = get_results_paths()
        save_path = results.figure("exploration", save)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def compare_cells(
    adata: "sc.AnnData",
    cell_indices: list[int],
    top_n: int = 20,
) -> pd.DataFrame:
    """Compare expression profiles of specific cells.
    
    Useful for understanding cell-to-cell variation and
    validating that the data looks reasonable.
    
    Args:
        adata: AnnData object
        cell_indices: List of cell indices to compare
        top_n: Number of top expressed genes to show
        
    Returns:
        DataFrame with expression values for top genes
    """
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    results = []
    for idx in cell_indices:
        expr = X[idx, :]
        top_genes_idx = np.argsort(expr)[::-1][:top_n]
        
        for rank, gene_idx in enumerate(top_genes_idx):
            results.append({
                "cell_idx": idx,
                "cell_barcode": adata.obs_names[idx],
                "rank": rank + 1,
                "gene": adata.var_names[gene_idx],
                "count": expr[gene_idx],
            })
    
    df = pd.DataFrame(results)
    
    # Pivot for easier comparison
    pivot = df.pivot_table(
        index=["rank", "gene"], 
        columns="cell_idx", 
        values="count",
        aggfunc="first"
    ).reset_index()
    
    return pivot


def describe_obs(adata: "sc.AnnData") -> pd.DataFrame:
    """Describe cell metadata (obs) columns.
    
    Args:
        adata: AnnData object
        
    Returns:
        DataFrame with column descriptions
    """
    descriptions = []
    for col in adata.obs.columns:
        dtype = adata.obs[col].dtype
        n_unique = adata.obs[col].nunique()
        n_missing = adata.obs[col].isna().sum()
        
        if pd.api.types.is_numeric_dtype(dtype):
            desc = {
                "column": col,
                "dtype": str(dtype),
                "n_unique": n_unique,
                "n_missing": n_missing,
                "min": adata.obs[col].min(),
                "max": adata.obs[col].max(),
                "mean": adata.obs[col].mean(),
            }
        else:
            desc = {
                "column": col,
                "dtype": str(dtype),
                "n_unique": n_unique,
                "n_missing": n_missing,
                "min": None,
                "max": None,
                "mean": None,
                "top_values": adata.obs[col].value_counts().head(3).to_dict(),
            }
        descriptions.append(desc)
    
    return pd.DataFrame(descriptions)


if __name__ == "__main__":
    # Demo usage
    import scanpy as sc
    from genailab.data.paths import get_data_paths
    
    paths = get_data_paths()
    adata_path = paths.scrna_processed("pbmc3k", "counts.h5ad")
    print(f"Analyzing {adata_path}...")
    
    if adata_path.exists():
        print(f"Loading {adata_path}...")
        adata = sc.read_h5ad(adata_path)
        print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
        
        print("\n" + "=" * 60)
        summarize_adata(adata)
        
        print("\n" + "=" * 60)
        inspect_sparsity(adata, plot=False)
    else:
        print(f"File not found: {adata_path}")
        print("Run: python -m genailab.data.sc_preprocess --dataset pbmc3k")
