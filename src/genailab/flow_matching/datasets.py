"""Dataset loaders for flow matching experiments.

Provides standardised DataLoader factories for:
    - MNIST (grayscale, 1×28×28) — baseline warmup
    - PathMNIST (RGB histopathology, 3×28×28 or 3×64×64) — medical POC
    - Any MedMNIST dataset (BloodMNIST, ChestMNIST, DermaMNIST, …)

All loaders return images normalised to [-1, 1] (standard for generative models).

MedMNIST installation::

    pip install medmnist

MedMNIST reference:
    Yang et al. (2023) MedMNIST v2. Scientific Data.
    https://medmnist.com/
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Default data root — can be overridden per call
_DEFAULT_DATA_ROOT = Path.home() / ".cache" / "genailab" / "data"


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def get_mnist_dataloader(
    batch_size: int = 128,
    image_size: int = 28,
    split: str = "train",
    data_root: str | Path = _DEFAULT_DATA_ROOT,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Standard MNIST dataloader, images normalised to [-1, 1].

    Args:
        batch_size:  Samples per batch.
        image_size:  Resize target (28 = native MNIST size).
        split:       ``"train"`` or ``"test"``.
        data_root:   Root directory for dataset download/cache.
        num_workers: DataLoader worker processes.
        pin_memory:  Pin memory for faster GPU transfer.

    Returns:
        DataLoader yielding (image, label) tuples,
        image shape (B, 1, image_size, image_size), values in [-1, 1].
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),                           # [0, 1]
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # → [-1, 1]
    ])
    dataset = datasets.MNIST(
        root=str(data_root),
        train=(split == "train"),
        transform=transform,
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# MedMNIST (PathMNIST and friends)
# ---------------------------------------------------------------------------

def get_medmnist_dataloader(
    dataset_name: str = "pathmnist",
    batch_size: int = 128,
    image_size: int = 28,
    split: str = "train",
    data_root: str | Path = _DEFAULT_DATA_ROOT,
    num_workers: int = 2,
    pin_memory: bool = True,
    as_rgb: bool = True,
) -> DataLoader:
    """MedMNIST dataloader for any dataset in the MedMNIST v2 collection.

    Supported dataset names (case-insensitive):
        pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist,
        retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist,
        organcmnist, organsmnist

    Args:
        dataset_name: MedMNIST dataset identifier (e.g. ``"pathmnist"``).
        batch_size:   Samples per batch.
        image_size:   Resize target; MedMNIST native is 28 or 64.
        split:        ``"train"``, ``"val"``, or ``"test"``.
        data_root:    Root directory for dataset download/cache.
        num_workers:  DataLoader worker processes.
        pin_memory:   Pin memory for faster GPU transfer.
        as_rgb:       Convert single-channel datasets to 3-channel RGB.
                      Has no effect on already-RGB datasets (e.g. PathMNIST).

    Returns:
        DataLoader yielding (image, label) tuples,
        image shape (B, C, image_size, image_size), values in [-1, 1].

    Example::

        # PathMNIST: colorectal cancer histology, 9 classes, RGB patches
        loader = get_medmnist_dataloader("pathmnist", batch_size=64, image_size=64)

        # BloodMNIST: blood cell microscopy
        loader = get_medmnist_dataloader("bloodmnist", batch_size=128)
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as e:
        raise ImportError(
            "MedMNIST is required: pip install medmnist"
        ) from e

    name = dataset_name.lower()
    if name not in INFO:
        raise ValueError(
            f"Unknown MedMNIST dataset '{dataset_name}'. "
            f"Available: {sorted(INFO.keys())}"
        )

    info = INFO[name]
    n_channels = info["n_channels"]
    DataClass = getattr(medmnist, info["python_class"])

    # Build transforms
    tf_list: list = [transforms.Resize(image_size)]
    if as_rgb and n_channels == 1:
        tf_list.append(transforms.Grayscale(num_output_channels=3))
    tf_list += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5] * (3 if (as_rgb and n_channels == 1) else n_channels),
            std=[0.5]  * (3 if (as_rgb and n_channels == 1) else n_channels),
        ),
    ]
    transform = transforms.Compose(tf_list)

    dataset = DataClass(
        split=split,
        transform=transform,
        download=True,
        root=str(data_root),
        size=image_size if image_size in (28, 64) else 28,
    )

    logger.info(
        "MedMNIST '%s' — split=%s  n=%d  channels=%d  classes=%d",
        name, split, len(dataset), n_channels, len(info["label"]),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def get_pathmnist_dataloader(
    batch_size: int = 128,
    image_size: int = 28,
    split: str = "train",
    **kwargs,
) -> DataLoader:
    """Convenience wrapper for PathMNIST (colorectal cancer histology).

    PathMNIST: 107,180 RGB patches (28×28) from colorectal cancer tissue,
    9 tissue classes. Directly relevant for digital pathology applications.

    See :func:`get_medmnist_dataloader` for full parameter documentation.
    """
    return get_medmnist_dataloader(
        dataset_name="pathmnist",
        batch_size=batch_size,
        image_size=image_size,
        split=split,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Dataset info helper
# ---------------------------------------------------------------------------

def dataset_info(dataset_name: str) -> dict:
    """Return metadata for a MedMNIST dataset.

    Args:
        dataset_name: e.g. ``"pathmnist"``.

    Returns:
        Dict with keys: n_channels, n_classes, label, task, …
    """
    try:
        from medmnist import INFO
        return INFO[dataset_name.lower()]
    except ImportError:
        raise ImportError("pip install medmnist")
    except KeyError:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
