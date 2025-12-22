"""Bulk RNA-seq preprocessing for NB/ZINB generative models.

This script provides Python-native options for preprocessing bulk RNA-seq data
without requiring R/Bioconductor. It supports:

1. Loading from CSV files (e.g., exported from R or downloaded from portals)
2. Loading from GEO using GEOparse
3. Converting to AnnData format (same as scRNA-seq)

Key principle: Do NOT normalize or log-transform the counts for NB/ZINB models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def load_from_csv(
    counts_path: str | Path,
    metadata_path: str | Path | None = None,
    genes_as_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load counts matrix and optional metadata from CSV files.
    
    Args:
        counts_path: Path to counts CSV (genes x samples or samples x genes)
        metadata_path: Optional path to sample metadata CSV
        genes_as_rows: If True, counts are genes x samples; if False, samples x genes
        
    Returns:
        Tuple of (counts DataFrame with genes as rows, metadata DataFrame or None)
    """
    counts = pd.read_csv(counts_path, index_col=0)
    
    if not genes_as_rows:
        counts = counts.T
    
    print(f"Loaded counts: {counts.shape[0]} genes x {counts.shape[1]} samples")
    
    metadata = None
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path, index_col=0)
        print(f"Loaded metadata: {metadata.shape[0]} samples x {metadata.shape[1]} columns")
    
    return counts, metadata


def load_from_geo(
    geo_id: str,
    output_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and load a GEO dataset using GEOparse.
    
    Args:
        geo_id: GEO accession (e.g., "GSE12345")
        output_dir: Directory to cache downloaded files
        
    Returns:
        Tuple of (counts DataFrame, metadata DataFrame)
        
    Note:
        Requires GEOparse: pip install GEOparse
        Not all GEO datasets have raw counts; many only have processed data.
    """
    try:
        import GEOparse
    except ImportError:
        raise ImportError("GEOparse not installed. Run: pip install GEOparse")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {geo_id}...")
    gse = GEOparse.get_GEO(geo=geo_id, destdir=str(output_dir))
    
    # Extract expression data
    # Note: GEO format varies; this handles the common case
    pivot_samples = gse.pivot_samples("VALUE")
    counts = pivot_samples.T  # samples x genes -> genes x samples after transpose back
    
    # Extract sample metadata
    metadata_list = []
    for gsm_name, gsm in gse.gsms.items():
        meta = {"sample_id": gsm_name}
        meta.update(gsm.metadata)
        # Flatten list values
        for k, v in meta.items():
            if isinstance(v, list) and len(v) == 1:
                meta[k] = v[0]
        metadata_list.append(meta)
    
    metadata = pd.DataFrame(metadata_list).set_index("sample_id")
    
    print(f"Loaded from GEO: {counts.shape[0]} genes x {counts.shape[1]} samples")
    
    return counts, metadata


def compute_library_size(counts: pd.DataFrame) -> pd.Series:
    """Compute library size (total counts per sample).
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        
    Returns:
        Series with library size per sample
    """
    library_size = counts.sum(axis=0)
    print(f"Library size range: [{library_size.min():.0f}, {library_size.max():.0f}]")
    return library_size


def filter_genes(
    counts: pd.DataFrame,
    min_samples: int = 10,
    min_counts: int = 1,
) -> pd.DataFrame:
    """Filter lowly-expressed genes.
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        min_samples: Minimum number of samples where gene must be expressed
        min_counts: Minimum count threshold for "expressed"
        
    Returns:
        Filtered counts DataFrame
    """
    n_genes_before = counts.shape[0]
    
    # Keep genes expressed in at least min_samples
    expressed = (counts >= min_counts).sum(axis=1)
    keep = expressed >= min_samples
    counts_filtered = counts.loc[keep]
    
    print(f"Filtered genes: {n_genes_before} -> {counts_filtered.shape[0]}")
    
    return counts_filtered


def to_anndata(
    counts: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    library_size: pd.Series | None = None,
) -> "ad.AnnData":
    """Convert counts and metadata to AnnData format.
    
    This allows using the same format as scRNA-seq data.
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        metadata: Optional sample metadata DataFrame
        library_size: Optional precomputed library sizes
        
    Returns:
        AnnData object with samples as obs and genes as var
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata not installed. Run: pip install anndata")
    
    # AnnData expects samples x genes
    X = counts.T.values
    
    # Create obs (sample metadata)
    obs = pd.DataFrame(index=counts.columns)
    if metadata is not None:
        # Align metadata with counts columns
        obs = metadata.loc[counts.columns].copy()
    
    # Add library size
    if library_size is None:
        library_size = counts.sum(axis=0)
    obs["library_size"] = library_size.loc[counts.columns].values
    
    # Create var (gene metadata)
    var = pd.DataFrame(index=counts.index)
    var["n_samples"] = (counts > 0).sum(axis=1).values
    
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Store raw counts in layers
    adata.layers["counts"] = X.copy()
    
    print(f"Created AnnData: {adata.n_obs} samples x {adata.n_vars} genes")
    
    return adata


def preprocess_bulk(
    counts: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    min_samples: int = 10,
    output_path: str | Path | None = None,
) -> "ad.AnnData":
    """Full preprocessing pipeline for bulk RNA-seq.
    
    Args:
        counts: DataFrame with genes as rows, samples as columns
        metadata: Optional sample metadata
        min_samples: Minimum samples per gene for filtering
        output_path: Optional path to save h5ad file
        
    Returns:
        Preprocessed AnnData object
    """
    # Compute library size BEFORE filtering
    library_size = compute_library_size(counts)
    
    # Filter genes
    counts_filtered = filter_genes(counts, min_samples=min_samples)
    
    # Convert to AnnData
    adata = to_anndata(counts_filtered, metadata, library_size)
    
    # Save if requested
    if output_path is not None:
        adata.write_h5ad(output_path)
        print(f"Saved to {output_path}")
    
    return adata


def main():
    """Command-line interface for bulk RNA-seq preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess bulk RNA-seq data for NB/ZINB generative models"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Data source")
    
    # CSV subcommand
    csv_parser = subparsers.add_parser("csv", help="Load from CSV files")
    csv_parser.add_argument(
        "--counts", "-c",
        type=str,
        required=True,
        help="Path to counts CSV (genes x samples)",
    )
    csv_parser.add_argument(
        "--metadata", "-m",
        type=str,
        default=None,
        help="Path to metadata CSV",
    )
    csv_parser.add_argument(
        "--samples-as-rows",
        action="store_true",
        help="If set, counts CSV has samples as rows (will be transposed)",
    )
    
    # GEO subcommand
    geo_parser = subparsers.add_parser("geo", help="Download from GEO")
    geo_parser.add_argument(
        "--geo-id", "-g",
        type=str,
        required=True,
        help="GEO accession (e.g., GSE12345)",
    )
    geo_parser.add_argument(
        "--cache-dir",
        type=str,
        default="./geo_cache",
        help="Directory to cache downloaded files",
    )
    
    # Common arguments
    for subparser in [csv_parser, geo_parser]:
        subparser.add_argument(
            "--output", "-o",
            type=str,
            default="bulk_counts.h5ad",
            help="Output h5ad file path",
        )
        subparser.add_argument(
            "--min-samples",
            type=int,
            default=10,
            help="Minimum samples per gene",
        )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Load data based on source
    if args.command == "csv":
        counts, metadata = load_from_csv(
            args.counts,
            args.metadata,
            genes_as_rows=not args.samples_as_rows,
        )
    elif args.command == "geo":
        counts, metadata = load_from_geo(args.geo_id, args.cache_dir)
    else:
        parser.print_help()
        return
    
    # Preprocess and save
    preprocess_bulk(
        counts,
        metadata,
        min_samples=args.min_samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
