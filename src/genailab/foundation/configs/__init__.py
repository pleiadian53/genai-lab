"""
Model and resource configuration management.
"""

from genailab.foundation.configs.model_configs import (
    ModelConfig,
    get_model_config,
    SMALL_CONFIG,
    MEDIUM_CONFIG,
    LARGE_CONFIG,
)
from genailab.foundation.configs.resource_profiles import (
    ResourceProfile,
    get_resource_profile,
    M1_PROFILE,
    RUNPOD_PROFILE,
    CLOUD_PROFILE,
)

__all__ = [
    "ModelConfig",
    "get_model_config",
    "SMALL_CONFIG",
    "MEDIUM_CONFIG",
    "LARGE_CONFIG",
    "ResourceProfile",
    "get_resource_profile",
    "M1_PROFILE",
    "RUNPOD_PROFILE",
    "CLOUD_PROFILE",
]
