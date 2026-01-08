"""
genailab: Generative AI for Computational Biology

A research-focused package for foundation models, VAEs, diffusion models,
and counterfactual simulation across genomics and multi-omics.
"""

__version__ = "0.1.0"

from genailab.model.vae import CVAE
from genailab.model.conditioning import ConditionSpec, ConditionEncoder
from genailab.config import (
    Config,
    get_config,
    get_project_root,
    get_data_dir,
    get_checkpoint_dir,
    get_results_dir,
    get_device,
    get_diffusion_config,
)

__all__ = [
    "__version__",
    "CVAE",
    "ConditionSpec",
    "ConditionEncoder",
    "Config",
    "get_config",
    "get_project_root",
    "get_data_dir",
    "get_checkpoint_dir",
    "get_results_dir",
    "get_device",
    "get_diffusion_config",
]
