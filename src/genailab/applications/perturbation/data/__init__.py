"""Perturb-seq data loaders, QC, and train/val/holdout splits.

Datasets:
  - :mod:`norman` — Norman et al. 2019 (K562 CRISPRa, combinatorial)
"""

from genailab.applications.perturbation.data.norman import (
    DEFAULT_CACHE_DIR as NORMAN_DEFAULT_CACHE_DIR,
    DEFAULT_NORMAN_URL,
    QCReport,
    QCThresholds,
    download_norman,
    load_norman,
    qc_norman,
    split_norman,
)

__all__ = [
    "NORMAN_DEFAULT_CACHE_DIR",
    "DEFAULT_NORMAN_URL",
    "QCThresholds",
    "QCReport",
    "download_norman",
    "load_norman",
    "qc_norman",
    "split_norman",
]
