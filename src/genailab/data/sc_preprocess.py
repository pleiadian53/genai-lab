"""scRNA-seq preprocessing for NB/ZINB generative models.

This script preprocesses scRNA-seq data while preserving raw counts
for use with Negative Binomial or Zero-Inflated NB likelihoods.

Key principle: Do NOT normalize or log-transform the counts.

Usage:
    # Download PBMC3k and preprocess to standard location
    python -m genailab.data.sc_preprocess --dataset pbmc3k
    
    # From local 10x MTX folder
    python -m genailab.data.sc_preprocess -i /path/to/filtered_feature_bc_matrix -o custom.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse

from genailab.data.paths import get_data_paths, pbmc3k_paths


def load_pbmc3k() -> sc.AnnData:
    """Download and return the PBMC3k dataset from 10x Genomics.
    
    This is a classic starter dataset with ~2,700 PBMCs.
    """
    # Scanpy provides a built-in download
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    return adata


def load_10x_mtx(path: str | Path) -> sc.AnnData:
    """Load 10x Genomics MTX format (filtered_feature_bc_matrix folder).
    
    Args:
        path: Path to the folder containing matrix.mtx, barcodes.tsv, genes.tsv
    """
    adata = sc.read_10x_mtx(
        path,
        var_names="gene_symbols",
        cache=True,
    )
    adata.var_names_make_unique()
    return adata


def compute_qc_metrics(
    adata: sc.AnnData,
    mito_prefix: str = "MT-",
) -> sc.AnnData:
    """Compute QC metrics including library size and mitochondrial content.
    
    Args:
        adata: AnnData object with raw counts
        mito_prefix: Prefix for mitochondrial genes ("MT-" for human, "mt-" for mouse)
    """
    # Library size (total counts per cell) - critical for NB models
    adata.obs["library_size"] = np.array(adata.X.sum(axis=1)).ravel()
    adata.obs["n_genes"] = np.array((adata.X > 0).sum(axis=1)).ravel()
    
    # Mitochondrial content
    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    
    return adata


def filter_cells_and_genes(
    adata: sc.AnnData,
    min_genes: int = 200,
    max_genes: int | None = None,
    min_cells: int = 3,
    max_mito_pct: float = 20.0,
) -> sc.AnnData:
    """Filter low-quality cells and lowly-expressed genes.
    
    Args:
        adata: AnnData object with QC metrics computed
        min_genes: Minimum genes per cell
        max_genes: Maximum genes per cell (None = no upper limit)
        min_cells: Minimum cells per gene
        max_mito_pct: Maximum mitochondrial percentage
    """
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        adata = adata[adata.obs.n_genes < max_genes].copy()
    adata = adata[adata.obs.pct_counts_mt < max_mito_pct].copy()
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"Filtered: {n_cells_before} -> {adata.n_obs} cells, "
          f"{n_genes_before} -> {adata.n_vars} genes")
    
    return adata


def select_highly_variable_genes(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
    subset: bool = False,
) -> sc.AnnData:
    """Identify highly variable genes (HVGs) for dimensionality reduction.
    
    Note: This uses a temporary log-normalization for HVG selection only.
    The raw counts are preserved for NB/ZINB modeling.
    
    Args:
        adata: AnnData object with raw counts
        n_top_genes: Number of HVGs to select
        subset: If True, subset to HVGs only; if False, just mark them
    """
    # Store raw counts
    adata.layers["counts"] = adata.X.copy()
    
    # Temporary normalization for HVG selection
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    
    # Find HVGs
    sc.pp.highly_variable_genes(
        adata_norm,
        n_top_genes=n_top_genes,
        flavor="seurat_v3" if sparse.issparse(adata.X) else "seurat",
    )
    
    # Transfer HVG annotations to original
    adata.var["highly_variable"] = adata_norm.var["highly_variable"]
    
    if subset:
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"Subset to {adata.n_vars} highly variable genes")
    
    return adata


def preprocess_for_ml(
    adata: sc.AnnData,
    output_path: str | Path = "counts_qc.h5ad",
) -> sc.AnnData:
    """Final preparation for ML: ensure counts are in correct format.
    
    Args:
        adata: Preprocessed AnnData
        output_path: Path to save the h5ad file
    """
    # Ensure counts are stored (not normalized)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    
    # Ensure library size is computed
    if "library_size" not in adata.obs:
        adata.obs["library_size"] = np.array(adata.X.sum(axis=1)).ravel()
    
    # Save
    adata.write_h5ad(output_path)
    print(f"Saved to {output_path}")
    print(f"  Cells: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Library size range: [{adata.obs.library_size.min():.0f}, "
          f"{adata.obs.library_size.max():.0f}]")
    
    return adata


def main():
    """Command-line interface for scRNA-seq preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess scRNA-seq data for NB/ZINB generative models"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        choices=["pbmc3k", "pbmc68k"],
        help="Dataset to download and preprocess (uses standard paths)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to 10x MTX folder (alternative to --dataset)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output h5ad file path (default: standard location based on dataset)",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=200,
        help="Minimum genes per cell",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=3,
        help="Minimum cells per gene",
    )
    parser.add_argument(
        "--max-mito",
        type=float,
        default=20.0,
        help="Maximum mitochondrial percentage",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=None,
        help="Number of highly variable genes to select (optional)",
    )
    parser.add_argument(
        "--mito-prefix",
        type=str,
        default="MT-",
        help="Mitochondrial gene prefix (MT- for human, mt- for mouse)",
    )
    parser.add_argument(
        "--setup-dirs",
        action="store_true",
        help="Create data directory structure and exit",
    )
    
    args = parser.parse_args()
    
    # Setup directories mode
    if args.setup_dirs:
        from genailab.data.paths import setup_data_directories
        setup_data_directories()
        return
    
    # Determine output path
    if args.output is not None:
        output_path = Path(args.output)
    elif args.dataset is not None:
        paths = get_data_paths()
        paths.ensure_dirs(args.dataset, "scrna")
        output_path = paths.scrna_processed(args.dataset, "counts.h5ad")
    else:
        output_path = Path("counts_qc.h5ad")
    
    # Load data
    if args.input is not None:
        print(f"Loading from {args.input}...")
        adata = load_10x_mtx(args.input)
    elif args.dataset == "pbmc3k" or args.dataset is None:
        print("Downloading PBMC3k dataset...")
        adata = load_pbmc3k()
    else:
        raise ValueError(f"Dataset {args.dataset} not yet supported for auto-download")
    
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # QC metrics
    adata = compute_qc_metrics(adata, mito_prefix=args.mito_prefix)
    
    # Filter
    adata = filter_cells_and_genes(
        adata,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        max_mito_pct=args.max_mito,
    )
    
    # HVG selection (optional)
    if args.n_hvg is not None:
        adata = select_highly_variable_genes(adata, n_top_genes=args.n_hvg)
    
    # Save
    preprocess_for_ml(adata, output_path=output_path)


if __name__ == "__main__":
    main()
