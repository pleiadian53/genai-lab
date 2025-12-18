"""
genailab: Generative AI for Computational Biology

A research-focused package for foundation models, VAEs, diffusion models,
and counterfactual simulation across genomics and multi-omics.
"""

__version__ = "0.1.0"

from genailab.model.vae import CVAE
from genailab.model.conditioning import ConditionSpec, ConditionEncoder

__all__ = [
    "__version__",
    "CVAE",
    "ConditionSpec",
    "ConditionEncoder",
]
