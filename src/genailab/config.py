"""Configuration module for genai-lab.

This module provides centralized configuration for paths, datasets, models,
and other settings. It eliminates the need for sys.path manipulation in notebooks.

Usage in notebooks:
    from genailab.config import Config, get_data_dir, get_checkpoint_dir
    
    config = Config()
    data_path = config.data_dir / "my_dataset.h5"
    
    # Or use convenience functions
    data_dir = get_data_dir()
    checkpoint_dir = get_checkpoint_dir("diffusion/medical_imaging")
"""

from pathlib import Path
from typing import Optional, Dict, Any
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Global configuration for genai-lab.
    
    This class provides centralized access to project paths, settings,
    and configurations. It automatically detects the project root and
    sets up standard directories.
    
    Attributes:
        project_root: Root directory of the genai-lab project
        data_dir: Directory for datasets
        checkpoint_dir: Directory for model checkpoints
        results_dir: Directory for results and outputs
        cache_dir: Directory for cached data
        device: Default device for PyTorch (auto-detected)
    """
    
    # Project structure
    project_root: Path = field(default_factory=lambda: _find_project_root())
    src_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    notebooks_dir: Path = field(init=False)
    examples_dir: Path = field(init=False)
    
    # Device configuration
    device: Optional[str] = None
    
    # Dataset paths (can be customized)
    dataset_paths: Dict[str, Path] = field(default_factory=dict)
    
    # Model configurations
    model_configs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.results_dir = self.project_root / "results"
        self.cache_dir = self.project_root / ".cache"
        self.notebooks_dir = self.project_root / "notebooks"
        self.examples_dir = self.project_root / "examples"
        
        # Auto-detect device if not specified
        if self.device is None:
            self.device = _detect_device()
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create standard directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.checkpoint_dir,
            self.results_dir,
            self.cache_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, experiment_name: str, create: bool = True) -> Path:
        """Get checkpoint directory for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment (e.g., "diffusion/medical_imaging")
            create: Whether to create the directory if it doesn't exist
        
        Returns:
            Path to the checkpoint directory
        """
        checkpoint_path = self.checkpoint_dir / experiment_name
        if create:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        return checkpoint_path
    
    def get_results_path(self, experiment_name: str, create: bool = True) -> Path:
        """Get results directory for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
            create: Whether to create the directory if it doesn't exist
        
        Returns:
            Path to the results directory
        """
        results_path = self.results_dir / experiment_name
        if create:
            results_path.mkdir(parents=True, exist_ok=True)
        return results_path
    
    def register_dataset(self, name: str, path: Path):
        """Register a dataset path.
        
        Args:
            name: Dataset name (e.g., "chest_xray", "gene_expression")
            path: Path to the dataset
        """
        self.dataset_paths[name] = Path(path)
    
    def get_dataset_path(self, name: str) -> Optional[Path]:
        """Get registered dataset path.
        
        Args:
            name: Dataset name
        
        Returns:
            Path to the dataset, or None if not registered
        """
        return self.dataset_paths.get(name)
    
    def register_model_config(self, name: str, config: Dict[str, Any]):
        """Register a model configuration.
        
        Args:
            name: Model name (e.g., "unet2d_medical", "tabular_gene")
            config: Model configuration dictionary
        """
        self.model_configs[name] = config
    
    def get_model_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get registered model configuration.
        
        Args:
            name: Model name
        
        Returns:
            Model configuration, or None if not registered
        """
        return self.model_configs.get(name)


def _find_project_root() -> Path:
    """Find the project root directory.
    
    Searches upward from the current file until finding a directory
    containing pyproject.toml or setup.py.
    
    Returns:
        Path to the project root
    
    Raises:
        RuntimeError: If project root cannot be found
    """
    current = Path(__file__).resolve()
    
    # Search upward for project markers
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    
    # Fallback: assume we're in src/genailab/
    # Go up two levels: src/genailab -> src -> project_root
    return current.parent.parent.parent


def _detect_device() -> str:
    """Auto-detect the best available device.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set the global configuration instance.
    
    Args:
        config: Config instance to use globally
    """
    global _global_config
    _global_config = config


# Convenience functions for common paths

def get_project_root() -> Path:
    """Get the project root directory."""
    return get_config().project_root


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_config().data_dir


def get_checkpoint_dir(experiment_name: Optional[str] = None) -> Path:
    """Get the checkpoint directory.
    
    Args:
        experiment_name: Optional experiment name for subdirectory
    
    Returns:
        Path to checkpoint directory
    """
    config = get_config()
    if experiment_name:
        return config.get_checkpoint_path(experiment_name)
    return config.checkpoint_dir


def get_results_dir(experiment_name: Optional[str] = None) -> Path:
    """Get the results directory.
    
    Args:
        experiment_name: Optional experiment name for subdirectory
    
    Returns:
        Path to results directory
    """
    config = get_config()
    if experiment_name:
        return config.get_results_path(experiment_name)
    return config.results_dir


def get_device() -> str:
    """Get the default device."""
    return get_config().device


# Model configuration presets

DIFFUSION_CONFIGS = {
    "unet2d_small": {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 32,
        "channel_multipliers": (1, 2, 4),
        "num_res_blocks": 2,
    },
    "unet2d_medium": {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 64,
        "channel_multipliers": (1, 2, 4, 8),
        "num_res_blocks": 2,
    },
    "unet2d_large": {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 128,
        "channel_multipliers": (1, 2, 4, 8),
        "num_res_blocks": 3,
    },
    "tabular_gene_expression": {
        "hidden_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.1,
        "use_attention": True,
    },
}


def get_diffusion_config(name: str) -> Dict[str, Any]:
    """Get a preset diffusion model configuration.
    
    Args:
        name: Configuration name (e.g., "unet2d_medium")
    
    Returns:
        Configuration dictionary
    
    Raises:
        KeyError: If configuration name not found
    """
    if name not in DIFFUSION_CONFIGS:
        available = ", ".join(DIFFUSION_CONFIGS.keys())
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    return DIFFUSION_CONFIGS[name].copy()
