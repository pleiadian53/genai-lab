"""Data loading, transforms, and batch handling for gene expression data."""

from genailab.data.loaders import GEOLoader, TCGALoader, ToyBulkDataset
from genailab.data.sc_dataset import SingleCellDataset, collate_sc_batch
from genailab.data.transforms import log1p_transform, standardize, hvg_filter
from genailab.data.batch import BatchEncoder, harmonize_batches
from genailab.data.splits import donor_aware_split, tissue_aware_split
from genailab.data.paths import (
    DataPaths,
    ResultsPaths,
    get_data_paths,
    get_results_paths,
    reset_data_paths,
    reset_results_paths,
    pbmc3k_paths,
    gtex_paths,
    setup_data_directories,
)

__all__ = [
    # Loaders
    "GEOLoader",
    "TCGALoader",
    "ToyBulkDataset",
    # Single-cell datasets
    "SingleCellDataset",
    "collate_sc_batch",
    # Transforms
    "log1p_transform",
    "standardize",
    "hvg_filter",
    # Batch handling
    "BatchEncoder",
    "harmonize_batches",
    # Splits
    "donor_aware_split",
    "tissue_aware_split",
    # Path management
    "DataPaths",
    "ResultsPaths",
    "get_data_paths",
    "get_results_paths",
    "reset_data_paths",
    "reset_results_paths",
    "pbmc3k_paths",
    "gtex_paths",
    "setup_data_directories",
]
