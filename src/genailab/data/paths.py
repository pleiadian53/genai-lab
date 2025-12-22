"""Data path management for genai-lab.

This module provides standardized paths for expression datasets, ensuring
consistency across the codebase. Similar to MetaSpliceAI's genomic_resources
but simplified for expression data.

Directory Structure:
    data/                           # Root data directory (not in git)
    ├── scrna/                      # scRNA-seq datasets
    │   ├── pbmc3k/                 # PBMC 3k dataset
    │   │   ├── raw/                # Raw downloaded files
    │   │   └── processed/          # Preprocessed h5ad files
    │   ├── pbmc68k/
    │   └── tabula_sapiens/
    ├── bulk/                       # Bulk RNA-seq datasets
    │   ├── gtex/
    │   ├── recount3/
    │   └── tcga/
    └── models/                     # Trained model checkpoints

Environment Variables:
    GENAILAB_DATA_ROOT: Override default data root directory
    GENAILAB_SCRNA_ROOT: Override scRNA-seq data directory
    GENAILAB_BULK_ROOT: Override bulk RNA-seq data directory

Usage:
    from genailab.data.paths import get_data_paths, DataPaths
    
    paths = get_data_paths()
    pbmc3k_processed = paths.scrna_processed("pbmc3k")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _find_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or .git."""
    current = Path(__file__).resolve()
    
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    
    # Fallback to current working directory
    return Path.cwd()


@dataclass
class DataPaths:
    """Centralized data path management.
    
    Attributes:
        root: Root data directory
        scrna_root: scRNA-seq data directory
        bulk_root: Bulk RNA-seq data directory
        models_root: Model checkpoints directory
    """
    root: Path
    scrna_root: Path = field(init=False)
    bulk_root: Path = field(init=False)
    models_root: Path = field(init=False)
    
    def __post_init__(self):
        self.root = Path(self.root)
        self.scrna_root = Path(os.getenv("GENAILAB_SCRNA_ROOT", self.root / "scrna"))
        self.bulk_root = Path(os.getenv("GENAILAB_BULK_ROOT", self.root / "bulk"))
        self.models_root = self.root / "models"
    
    # =========================================================================
    # scRNA-seq paths
    # =========================================================================
    
    def scrna_dataset_dir(self, dataset: str) -> Path:
        """Get directory for a scRNA-seq dataset.
        
        Args:
            dataset: Dataset name (e.g., "pbmc3k", "tabula_sapiens")
        """
        return self.scrna_root / dataset
    
    def scrna_raw(self, dataset: str) -> Path:
        """Get raw data directory for a scRNA-seq dataset."""
        return self.scrna_dataset_dir(dataset) / "raw"
    
    def scrna_processed(self, dataset: str, filename: str = "counts.h5ad") -> Path:
        """Get processed file path for a scRNA-seq dataset.
        
        Args:
            dataset: Dataset name
            filename: Processed file name (default: counts.h5ad)
        """
        return self.scrna_dataset_dir(dataset) / "processed" / filename
    
    # =========================================================================
    # Bulk RNA-seq paths
    # =========================================================================
    
    def bulk_dataset_dir(self, dataset: str) -> Path:
        """Get directory for a bulk RNA-seq dataset.
        
        Args:
            dataset: Dataset name (e.g., "gtex", "recount3", "tcga")
        """
        return self.bulk_root / dataset
    
    def bulk_raw(self, dataset: str) -> Path:
        """Get raw data directory for a bulk RNA-seq dataset."""
        return self.bulk_dataset_dir(dataset) / "raw"
    
    def bulk_processed(self, dataset: str, filename: str = "counts.h5ad") -> Path:
        """Get processed file path for a bulk RNA-seq dataset.
        
        Args:
            dataset: Dataset name
            filename: Processed file name (default: counts.h5ad)
        """
        return self.bulk_dataset_dir(dataset) / "processed" / filename
    
    # =========================================================================
    # Model paths
    # =========================================================================
    
    def model_dir(self, model_name: str) -> Path:
        """Get directory for a trained model."""
        return self.models_root / model_name
    
    def model_checkpoint(self, model_name: str, checkpoint: str = "best.pt") -> Path:
        """Get checkpoint path for a trained model."""
        return self.model_dir(model_name) / checkpoint
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def ensure_dirs(self, dataset: str, data_type: Literal["scrna", "bulk"] = "scrna"):
        """Create directory structure for a dataset.
        
        Args:
            dataset: Dataset name
            data_type: "scrna" or "bulk"
        """
        if data_type == "scrna":
            self.scrna_raw(dataset).mkdir(parents=True, exist_ok=True)
            self.scrna_processed(dataset).parent.mkdir(parents=True, exist_ok=True)
        else:
            self.bulk_raw(dataset).mkdir(parents=True, exist_ok=True)
            self.bulk_processed(dataset).parent.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self, data_type: Literal["scrna", "bulk"] = "scrna") -> list[str]:
        """List available datasets.
        
        Args:
            data_type: "scrna" or "bulk"
            
        Returns:
            List of dataset names that have processed files
        """
        root = self.scrna_root if data_type == "scrna" else self.bulk_root
        if not root.exists():
            return []
        
        datasets = []
        for d in root.iterdir():
            if d.is_dir() and (d / "processed").exists():
                datasets.append(d.name)
        return sorted(datasets)
    
    def __repr__(self) -> str:
        return (
            f"DataPaths(\n"
            f"  root={self.root},\n"
            f"  scrna_root={self.scrna_root},\n"
            f"  bulk_root={self.bulk_root},\n"
            f"  models_root={self.models_root}\n"
            f")"
        )


# Global singleton
_data_paths: DataPaths | None = None


def get_data_paths(data_root: str | Path | None = None) -> DataPaths:
    """Get the global DataPaths instance.
    
    Args:
        data_root: Override data root directory. If None, uses:
            1. GENAILAB_DATA_ROOT environment variable
            2. <project_root>/data/
            
    Returns:
        DataPaths instance
    """
    global _data_paths
    
    if _data_paths is None or data_root is not None:
        if data_root is None:
            data_root = os.getenv("GENAILAB_DATA_ROOT")
        
        if data_root is None:
            project_root = _find_project_root()
            data_root = project_root / "data"
        
        _data_paths = DataPaths(root=Path(data_root))
    
    return _data_paths


def reset_data_paths():
    """Reset the global DataPaths instance (useful for testing)."""
    global _data_paths
    _data_paths = None


# ============================================================================
# Convenience functions for common datasets
# ============================================================================

def pbmc3k_paths() -> dict[str, Path]:
    """Get paths for PBMC 3k dataset.
    
    Returns:
        Dict with 'raw', 'processed', and 'counts' paths
    """
    paths = get_data_paths()
    return {
        "raw": paths.scrna_raw("pbmc3k"),
        "processed": paths.scrna_processed("pbmc3k").parent,
        "counts": paths.scrna_processed("pbmc3k", "counts.h5ad"),
    }


def gtex_paths() -> dict[str, Path]:
    """Get paths for GTEx dataset.
    
    Returns:
        Dict with 'raw', 'processed', and 'counts' paths
    """
    paths = get_data_paths()
    return {
        "raw": paths.bulk_raw("gtex"),
        "processed": paths.bulk_processed("gtex").parent,
        "counts": paths.bulk_processed("gtex", "counts.h5ad"),
    }


# ============================================================================
# CLI for setup
# ============================================================================

def setup_data_directories():
    """Create the standard data directory structure."""
    paths = get_data_paths()
    
    # Create main directories
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.scrna_root.mkdir(parents=True, exist_ok=True)
    paths.bulk_root.mkdir(parents=True, exist_ok=True)
    paths.models_root.mkdir(parents=True, exist_ok=True)
    
    # Create dataset directories
    for dataset in ["pbmc3k", "pbmc68k", "tabula_sapiens"]:
        paths.ensure_dirs(dataset, "scrna")
    
    for dataset in ["gtex", "recount3", "tcga"]:
        paths.ensure_dirs(dataset, "bulk")
    
    # Create .gitignore in data directory
    gitignore = paths.root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# Ignore all data files\n*\n!.gitignore\n!README.md\n")
    
    # Create README
    readme = paths.root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Data Directory\n\n"
            "This directory contains expression datasets for training generative models.\n\n"
            "## Structure\n\n"
            "```\n"
            "data/\n"
            "├── scrna/           # scRNA-seq datasets\n"
            "│   ├── pbmc3k/\n"
            "│   ├── pbmc68k/\n"
            "│   └── tabula_sapiens/\n"
            "├── bulk/            # Bulk RNA-seq datasets\n"
            "│   ├── gtex/\n"
            "│   ├── recount3/\n"
            "│   └── tcga/\n"
            "└── models/          # Trained model checkpoints\n"
            "```\n\n"
            "## Usage\n\n"
            "```python\n"
            "from genailab.data.paths import get_data_paths\n\n"
            "paths = get_data_paths()\n"
            "pbmc3k = paths.scrna_processed('pbmc3k')\n"
            "```\n"
        )
    
    print(f"Created data directory structure at: {paths.root}")
    print(paths)


if __name__ == "__main__":
    setup_data_directories()
