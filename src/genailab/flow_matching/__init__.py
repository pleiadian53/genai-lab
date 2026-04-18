"""Flow matching generative models.

Public API
----------
Interpolants::

    from genailab.flow_matching import LinearInterpolant, VPInterpolant

Velocity networks::

    from genailab.flow_matching import VelocityUNet2D, VelocityMLP
    from genailab.flow_matching import build_velocity_network

Training::

    from genailab.flow_matching import FlowMatchingTrainer, cfm_loss

Sampling::

    from genailab.flow_matching import EulerSampler, RK4Sampler, AdaptiveSampler
    from genailab.flow_matching import build_sampler

Data::

    from genailab.flow_matching import (
        get_mnist_dataloader,
        get_pathmnist_dataloader,
        get_medmnist_dataloader,
    )
"""

from .interpolants import (
    BaseInterpolant,
    LinearInterpolant,
    VPInterpolant,
    broadcast_time,
)
from .velocity_networks import (
    VelocityUNet2D,
    VelocityMLP,
    build_velocity_network,
)
from .training import (
    cfm_loss,
    FlowMatchingTrainer,
)
from .sampling import (
    EulerSampler,
    RK4Sampler,
    AdaptiveSampler,
    build_sampler,
)
from .datasets import (
    get_mnist_dataloader,
    get_pathmnist_dataloader,
    get_medmnist_dataloader,
    dataset_info,
)

__all__ = [
    # Interpolants
    "BaseInterpolant",
    "LinearInterpolant",
    "VPInterpolant",
    "broadcast_time",
    # Networks
    "VelocityUNet2D",
    "VelocityMLP",
    "build_velocity_network",
    # Training
    "cfm_loss",
    "FlowMatchingTrainer",
    # Sampling
    "EulerSampler",
    "RK4Sampler",
    "AdaptiveSampler",
    "build_sampler",
    # Data
    "get_mnist_dataloader",
    "get_pathmnist_dataloader",
    "get_medmnist_dataloader",
    "dataset_info",
]
