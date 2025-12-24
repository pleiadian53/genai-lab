"""Data and results path management for genai-lab.

This module provides standardized paths for:
- Input data (expression datasets)
- Output results (figures, models, logs, experiments)

Directory Structure:
    data/                           # Input data (gitignored, large files)
    ├── scrna/                      # scRNA-seq datasets
    │   ├── pbmc3k/
    │   │   ├── raw/
    │   │   └── processed/
    │   ├── pbmc68k/
    │   └── tabula_sapiens/
    ├── bulk/                       # Bulk RNA-seq datasets
    │   ├── gtex/
    │   ├── recount3/
    │   └── tcga/
    └── models/                     # Trained model checkpoints (legacy)
    
    results/                        # Output artifacts (can be versioned)
    ├── figures/                    # Plots from analysis
    │   ├── exploration/            # Data exploration plots
    │   ├── training/               # Training curves, diagnostics
    │   └── evaluation/             # Model evaluation plots
    ├── models/                     # Trained model checkpoints
    ├── logs/                       # Training logs, metrics
    └── experiments/                # Experiment-specific outputs
        └── <experiment_name>/      # Each experiment gets a folder

Environment Variables:
    GENAILAB_DATA_ROOT: Override default data root directory
    GENAILAB_RESULTS_ROOT: Override default results root directory
    GENAILAB_SCRNA_ROOT: Override scRNA-seq data directory
    GENAILAB_BULK_ROOT: Override bulk RNA-seq data directory

Usage:
    from genailab.data.paths import get_data_paths, get_results_paths
    
    # Data paths
    data_paths = get_data_paths()
    pbmc3k = data_paths.scrna_processed("pbmc3k")
    
    # Results paths
    results = get_results_paths()
    fig_path = results.figure("exploration", "sparsity_analysis.png")
    model_path = results.model_checkpoint("vae_pbmc3k", "best.pt")
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


# =============================================================================
# Results Path Management
# =============================================================================

@dataclass
class ResultsPaths:
    """Centralized results/output path management.
    
    Organizes all outputs (figures, models, logs, experiments) in a
    structured way for reproducibility and documentation.
    
    Attributes:
        root: Root results directory
        figures_root: Figures directory
        models_root: Model checkpoints directory
        logs_root: Training logs directory
        experiments_root: Experiment-specific outputs
    """
    root: Path
    figures_root: Path = field(init=False)
    models_root: Path = field(init=False)
    logs_root: Path = field(init=False)
    experiments_root: Path = field(init=False)
    
    def __post_init__(self):
        self.root = Path(self.root)
        self.figures_root = self.root / "figures"
        self.models_root = self.root / "models"
        self.logs_root = self.root / "logs"
        self.experiments_root = self.root / "experiments"
    
    # =========================================================================
    # Figure paths
    # =========================================================================
    
    def figure(
        self,
        category: str,
        filename: str,
        create_dir: bool = True,
    ) -> Path:
        """Get path for a figure file.
        
        Args:
            category: Figure category (e.g., "exploration", "training", "evaluation")
            filename: Figure filename (e.g., "sparsity_analysis.png")
            create_dir: If True, create the directory if it doesn't exist
            
        Returns:
            Path to the figure file
            
        Example:
            >>> results = get_results_paths()
            >>> path = results.figure("exploration", "library_size.png")
            >>> plt.savefig(path)
        """
        fig_dir = self.figures_root / category
        if create_dir:
            fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir / filename
    
    def figure_dir(self, category: str, create: bool = True) -> Path:
        """Get directory for a figure category.
        
        Args:
            category: Figure category
            create: If True, create the directory
        """
        fig_dir = self.figures_root / category
        if create:
            fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir
    
    # =========================================================================
    # Model paths
    # =========================================================================
    
    def model_dir(self, model_name: str, create: bool = True) -> Path:
        """Get directory for a trained model.
        
        Args:
            model_name: Name of the model (e.g., "vae_pbmc3k_20241222")
            create: If True, create the directory
        """
        model_dir = self.models_root / model_name
        if create:
            model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def model_checkpoint(
        self,
        model_name: str,
        checkpoint: str = "best.pt",
        create_dir: bool = True,
    ) -> Path:
        """Get checkpoint path for a trained model.
        
        Args:
            model_name: Name of the model
            checkpoint: Checkpoint filename (default: best.pt)
            create_dir: If True, create the model directory
        """
        return self.model_dir(model_name, create=create_dir) / checkpoint
    
    # =========================================================================
    # Log paths
    # =========================================================================
    
    def log_dir(self, experiment_name: str, create: bool = True) -> Path:
        """Get log directory for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            create: If True, create the directory
        """
        log_dir = self.logs_root / experiment_name
        if create:
            log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def log_file(self, experiment_name: str, filename: str = "train.log") -> Path:
        """Get log file path."""
        return self.log_dir(experiment_name) / filename
    
    # =========================================================================
    # Experiment paths
    # =========================================================================
    
    def experiment_dir(self, experiment_name: str, create: bool = True) -> Path:
        """Get directory for an experiment.
        
        Each experiment can have its own figures, models, and logs.
        
        Args:
            experiment_name: Name of the experiment (e.g., "vae_pbmc3k_v1")
            create: If True, create the directory structure
        """
        exp_dir = self.experiments_root / experiment_name
        if create:
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "figures").mkdir(exist_ok=True)
            (exp_dir / "checkpoints").mkdir(exist_ok=True)
            (exp_dir / "logs").mkdir(exist_ok=True)
        return exp_dir
    
    def experiment_figure(self, experiment_name: str, filename: str) -> Path:
        """Get figure path within an experiment."""
        return self.experiment_dir(experiment_name) / "figures" / filename
    
    def experiment_checkpoint(self, experiment_name: str, filename: str = "best.pt") -> Path:
        """Get checkpoint path within an experiment."""
        return self.experiment_dir(experiment_name) / "checkpoints" / filename
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def setup(self) -> None:
        """Create the standard results directory structure."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.figures_root.mkdir(exist_ok=True)
        self.models_root.mkdir(exist_ok=True)
        self.logs_root.mkdir(exist_ok=True)
        self.experiments_root.mkdir(exist_ok=True)
        
        # Create standard figure categories
        for category in ["exploration", "training", "evaluation"]:
            (self.figures_root / category).mkdir(exist_ok=True)
        
        # Create .gitignore (keep structure but ignore large files)
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(
                "# Ignore large model files\n"
                "models/**/*.pt\n"
                "models/**/*.pth\n"
                "experiments/**/checkpoints/*.pt\n"
                "experiments/**/checkpoints/*.pth\n"
                "\n"
                "# Keep directory structure\n"
                "!.gitignore\n"
                "!**/\n"
            )
        
        print(f"Created results directory structure at: {self.root}")
    
    def __repr__(self) -> str:
        return (
            f"ResultsPaths(\n"
            f"  root={self.root},\n"
            f"  figures_root={self.figures_root},\n"
            f"  models_root={self.models_root},\n"
            f"  logs_root={self.logs_root},\n"
            f"  experiments_root={self.experiments_root}\n"
            f")"
        )


# Global singletons
_data_paths: DataPaths | None = None
_results_paths: ResultsPaths | None = None


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


def get_results_paths(results_root: str | Path | None = None) -> ResultsPaths:
    """Get the global ResultsPaths instance.
    
    Args:
        results_root: Override results root directory. If None, uses:
            1. GENAILAB_RESULTS_ROOT environment variable
            2. <project_root>/results/
            
    Returns:
        ResultsPaths instance
    """
    global _results_paths
    
    if _results_paths is None or results_root is not None:
        if results_root is None:
            results_root = os.getenv("GENAILAB_RESULTS_ROOT")
        
        if results_root is None:
            project_root = _find_project_root()
            results_root = project_root / "results"
        
        _results_paths = ResultsPaths(root=Path(results_root))
    
    return _results_paths


def reset_data_paths():
    """Reset the global DataPaths instance (useful for testing)."""
    global _data_paths
    _data_paths = None


def reset_results_paths():
    """Reset the global ResultsPaths instance (useful for testing)."""
    global _results_paths
    _results_paths = None


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
