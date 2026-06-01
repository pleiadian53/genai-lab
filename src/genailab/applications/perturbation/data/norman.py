"""Norman et al. 2019 Perturb-seq loader, QC, and splits.

Public API:
    download_norman(cache_dir, url=..., force=False) -> Path
    load_norman(cache_dir, download_if_missing=True) -> AnnData
    qc_norman(adata, thresholds=...) -> (AnnData, QCReport)
    split_norman(adata, strategy="cell", val_frac=..., holdout_frac=...) -> dict[str, np.ndarray]

Background, QC rationale, and split semantics are documented in
``notebooks/perturbation/docs/norman_2019_dataset_tutorial.md``.

Data convention: caches under ``data/scrna/perturb_seq/norman_2019/``
following the ``data/<modality>/<sub-topic>/<dataset>/`` layout.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — adjust here when scPerturb deposits a new version
# ---------------------------------------------------------------------------

#: scPerturb's canonical Norman 2019 release (hosted on Zenodo).
#: If download fails with 404, scPerturb may have updated the deposit —
#: check https://scperturb.org and pass ``url=`` explicitly to override.
DEFAULT_NORMAN_URL = (
    "https://zenodo.org/records/13350497/files/NormanWeissman2019.h5ad"
)

#: Cache location follows ``data/<modality>/<sub-topic>/<dataset>/``.
DEFAULT_CACHE_DIR = Path("data/scrna/perturb_seq/norman_2019")
DEFAULT_FILENAME = "NormanWeissman2019.h5ad"

#: scPerturb releases use varying column names for the perturbation label.
#: We probe in order; the first hit wins.
_PERTURBATION_COL_CANDIDATES = (
    "perturbation",
    "perturbation_name",
    "condition",
    "guide_id",
)

#: Common spellings for the control label used by scPerturb releases.
_CONTROL_LABELS = ("control", "ctrl", "non-targeting", "NT")


# ---------------------------------------------------------------------------
# QC dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QCThresholds:
    """QC thresholds for Norman 2019.

    Rationale for each threshold is in
    ``notebooks/perturbation/docs/norman_2019_dataset_tutorial.md`` §5.
    """

    #: Minimum number of expressed genes per cell. Cells below this are empty
    #: or damaged droplets. Standard 10x scRNA-seq threshold.
    min_genes: int = 200

    #: Maximum percent mitochondrial counts per cell. K562 (cell line) has
    #: higher baseline MT than primary cells; 20% matches the original paper.
    pct_mt_max: float = 20.0

    #: Minimum cells per gene. Genes detected in fewer cells carry no signal.
    min_cells: int = 3

    #: Minimum cells per perturbation. Below this, perturbation effects
    #: cannot be reliably estimated. Controls are exempt from this filter.
    min_cells_per_perturbation: int = 30


@dataclass
class QCReport:
    """Summary of what QC did. Returned alongside the QC'd AnnData."""

    n_cells_initial: int
    n_cells_final: int
    n_genes_initial: int
    n_genes_final: int
    n_perturbations_initial: int
    n_perturbations_final: int
    dropped_cells_by_min_genes: int
    dropped_cells_by_pct_mt: int
    dropped_cells_by_low_count_perturbation: int
    dropped_genes_by_min_cells: int
    perturbation_col: str
    control_label: Optional[str]
    n_control_cells: int
    library_size_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path, indent: int = 2) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_with_progress(url: str, target: Path) -> None:
    """Download a URL to ``target`` with a TTY-friendly progress line."""
    tmp = target.with_suffix(target.suffix + ".partial")

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, 100.0 * downloaded / total_size)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        print(
            f"\r  [{bar}] {pct:5.1f}%  {mb_done:7.1f} / {mb_total:7.1f} MB",
            end="",
            flush=True,
        )

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Failed to download Norman 2019 from {url} (HTTP {exc.code}). "
            "If this is a 404, the scPerturb deposit version may have "
            "changed — check https://scperturb.org and pass url= explicitly."
        ) from exc
    print()  # newline after progress bar
    tmp.rename(target)


def download_norman(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    url: str = DEFAULT_NORMAN_URL,
    force: bool = False,
) -> Path:
    """Download Norman 2019 ``.h5ad`` from scPerturb.

    Args:
        cache_dir: Directory to cache the file in.
        url: Source URL (default: scPerturb Zenodo deposit).
        force: Re-download even if the cached file exists.

    Returns:
        Path to the cached file.
    """
    cache_dir = Path(cache_dir)
    target = cache_dir / DEFAULT_FILENAME
    if target.exists() and not force:
        size_mb = target.stat().st_size / 1e6
        logger.info("Norman 2019 already cached at %s (%.1f MB)", target, size_mb)
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Norman 2019 to %s ...", target)
    _download_with_progress(url, target)
    logger.info("Download complete: %s", target)
    return target


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_norman(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    download_if_missing: bool = True,
):
    """Load the Norman 2019 raw AnnData.

    Args:
        cache_dir: Where to look for / cache the dataset.
        download_if_missing: If True and the file is absent, fetch it from
            scPerturb. If False, raise FileNotFoundError instead.

    Returns:
        Raw AnnData with integer UMI counts in ``.X``.
    """
    import anndata as ad  # heavy; lazy import

    cache_dir = Path(cache_dir)
    target = cache_dir / DEFAULT_FILENAME
    if not target.exists():
        if download_if_missing:
            download_norman(cache_dir)
        else:
            raise FileNotFoundError(
                f"Norman 2019 not found at {target}. "
                "Pass download_if_missing=True or call download_norman() first."
            )
    logger.info("Loading Norman 2019 from %s", target)
    adata = ad.read_h5ad(target)
    logger.info(
        "Loaded: %d cells x %d genes (X dtype=%s, sparse=%s)",
        adata.n_obs,
        adata.n_vars,
        adata.X.dtype,
        hasattr(adata.X, "toarray"),
    )
    return adata


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------


def _find_perturbation_col(adata) -> str:
    """Probe for the perturbation column. Returns the first match."""
    for col in _PERTURBATION_COL_CANDIDATES:
        if col in adata.obs.columns:
            return col
    raise ValueError(
        f"No perturbation column found in adata.obs. "
        f"Tried: {_PERTURBATION_COL_CANDIDATES}. "
        f"Available: {list(adata.obs.columns)}"
    )


def _find_control_label(adata, perturbation_col: str) -> Optional[str]:
    """Probe for the control label among common spellings. None if absent."""
    values = set(adata.obs[perturbation_col].astype(str).unique())
    for label in _CONTROL_LABELS:
        if label in values:
            return label
    # Some releases use "ctrl+ctrl" or similar for paired controls
    for label in _CONTROL_LABELS:
        for v in values:
            if v.startswith(f"{label}+") or v.endswith(f"+{label}") or v == f"{label}+{label}":
                return v
    return None


def qc_norman(
    adata,
    thresholds: Optional[QCThresholds] = None,
    perturbation_col: Optional[str] = None,
):
    """Apply standard QC to Norman 2019, preserving raw counts.

    Returns:
        (adata_qcd, QCReport)

    Order of operations:
        1. Preserve raw counts in ``adata.layers["counts"]`` (if not already)
        2. Compute QC metrics (n_genes_by_counts, pct_counts_mt, ...)
        3. Filter cells: min_genes, pct_mt_max
        4. Filter genes: min_cells
        5. Filter perturbations: min_cells_per_perturbation (controls exempt)
        6. Compute library_size on the full filtered gene set (covariate
           for downstream NB/ZINB decoders)
    """
    import scanpy as sc  # heavy; lazy import

    thresholds = thresholds or QCThresholds()
    perturbation_col = perturbation_col or _find_perturbation_col(adata)
    logger.info("Using perturbation column: %s", perturbation_col)

    control_label = _find_control_label(adata, perturbation_col)
    if control_label:
        logger.info("Detected control label: %s", control_label)
    else:
        logger.warning("No control label detected; all cells will be subject to perturbation-count filter")

    # 0. Initial counts (for the report)
    n_cells_initial = adata.n_obs
    n_genes_initial = adata.n_vars
    n_perts_initial = adata.obs[perturbation_col].nunique()

    # 1. Preserve raw counts before any modification
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
        logger.info("Stashed raw counts in adata.layers['counts']")
    else:
        logger.info("adata.layers['counts'] already present; not overwriting")

    # 2. QC metrics
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    logger.info(
        "QC metrics computed. MT genes: %d. pct_counts_mt range: %.2f-%.2f",
        int(adata.var["mt"].sum()),
        float(adata.obs["pct_counts_mt"].min()),
        float(adata.obs["pct_counts_mt"].max()),
    )

    # 3. Cell-level filters
    n_before_min_genes = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=thresholds.min_genes)
    dropped_min_genes = n_before_min_genes - adata.n_obs

    n_before_mt = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < thresholds.pct_mt_max].copy()
    dropped_mt = n_before_mt - adata.n_obs

    logger.info(
        "Cell QC: dropped %d (min_genes<%d), %d (pct_mt>=%.1f) -> %d cells remain",
        dropped_min_genes,
        thresholds.min_genes,
        dropped_mt,
        thresholds.pct_mt_max,
        adata.n_obs,
    )

    # 4. Gene-level filter
    n_before_gene_filter = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=thresholds.min_cells)
    dropped_genes = n_before_gene_filter - adata.n_vars
    logger.info(
        "Gene QC: dropped %d (min_cells<%d) -> %d genes remain",
        dropped_genes,
        thresholds.min_cells,
        adata.n_vars,
    )

    # 5. Perturbation-level filter (exempt control)
    pert_counts = adata.obs[perturbation_col].value_counts()
    low_count_perts = pert_counts[pert_counts < thresholds.min_cells_per_perturbation].index
    if control_label in low_count_perts:
        low_count_perts = low_count_perts.drop(control_label)
    if len(low_count_perts) > 0:
        n_before_pert_filter = adata.n_obs
        mask = ~adata.obs[perturbation_col].isin(low_count_perts)
        adata = adata[mask].copy()
        dropped_low_pert = n_before_pert_filter - adata.n_obs
        logger.info(
            "Perturbation QC: dropped %d cells across %d perturbations "
            "(cells_per_pert < %d)",
            dropped_low_pert,
            len(low_count_perts),
            thresholds.min_cells_per_perturbation,
        )
    else:
        dropped_low_pert = 0
        logger.info("Perturbation QC: no perturbations below threshold")

    # 6. Library size (covariate for NB/ZINB decoders)
    X = adata.layers["counts"]
    library_size = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    adata.obs["library_size"] = library_size
    lib_stats = {
        "mean": float(library_size.mean()),
        "median": float(np.median(library_size)),
        "min": float(library_size.min()),
        "max": float(library_size.max()),
        "std": float(library_size.std()),
    }
    logger.info(
        "library_size: median=%.0f, mean=%.0f, min=%.0f, max=%.0f",
        lib_stats["median"],
        lib_stats["mean"],
        lib_stats["min"],
        lib_stats["max"],
    )

    n_control_cells = (
        int((adata.obs[perturbation_col] == control_label).sum())
        if control_label
        else 0
    )

    report = QCReport(
        n_cells_initial=n_cells_initial,
        n_cells_final=adata.n_obs,
        n_genes_initial=n_genes_initial,
        n_genes_final=adata.n_vars,
        n_perturbations_initial=n_perts_initial,
        n_perturbations_final=adata.obs[perturbation_col].nunique(),
        dropped_cells_by_min_genes=int(dropped_min_genes),
        dropped_cells_by_pct_mt=int(dropped_mt),
        dropped_cells_by_low_count_perturbation=int(dropped_low_pert),
        dropped_genes_by_min_cells=int(dropped_genes),
        perturbation_col=perturbation_col,
        control_label=control_label,
        n_control_cells=n_control_cells,
        library_size_stats=lib_stats,
    )
    return adata, report


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def split_norman(
    adata,
    strategy: Literal["cell"] = "cell",
    val_frac: float = 0.15,
    holdout_frac: float = 0.15,
    seed: int = 42,
    perturbation_col: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Build train/val/holdout splits.

    Args:
        adata: QC'd AnnData (output of :func:`qc_norman`).
        strategy: Splitting strategy. Only ``"cell"`` is implemented for P1.
            Perturbation-level and combination-level splits land in P2+.
        val_frac: Fraction of each perturbation's cells to assign to val.
        holdout_frac: Fraction of each perturbation's cells to assign to
            holdout (used for final evaluation).
        seed: RNG seed for reproducibility.
        perturbation_col: Override the perturbation column name.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"holdout"`` mapping to
        boolean masks over ``adata.obs`` of length ``adata.n_obs``.

        As a side effect, sets ``adata.obs[f"split_{strategy}"]`` with
        string labels ``"train"`` / ``"val"`` / ``"holdout"``.

    Note:
        ``strategy="cell"`` is **stratified by perturbation** — every
        perturbation contributes proportionally to all three splits. This
        is the trivial "can we reconstruct seen perturbations" check; it
        does NOT test generalization to unseen perturbations.
    """
    if strategy != "cell":
        raise NotImplementedError(
            f"Split strategy {strategy!r} will be added in P2+. "
            "P1 supports 'cell' (stratified random) only."
        )

    perturbation_col = perturbation_col or _find_perturbation_col(adata)
    rng = np.random.default_rng(seed)
    n = adata.n_obs

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    holdout_mask = np.zeros(n, dtype=bool)

    pert_labels = adata.obs[perturbation_col].values
    for pert in pd.unique(pert_labels):
        idx = np.where(pert_labels == pert)[0]
        rng.shuffle(idx)
        n_pert = len(idx)
        n_val = int(round(val_frac * n_pert))
        n_holdout = int(round(holdout_frac * n_pert))
        val_mask[idx[:n_val]] = True
        holdout_mask[idx[n_val : n_val + n_holdout]] = True
        train_mask[idx[n_val + n_holdout :]] = True

    # Sanity: masks partition the cells
    assert (train_mask | val_mask | holdout_mask).all(), "splits do not cover all cells"
    assert not (train_mask & val_mask).any(), "train and val overlap"
    assert not (train_mask & holdout_mask).any(), "train and holdout overlap"
    assert not (val_mask & holdout_mask).any(), "val and holdout overlap"

    split_col = f"split_{strategy}"
    adata.obs[split_col] = "train"
    adata.obs.loc[val_mask, split_col] = "val"
    adata.obs.loc[holdout_mask, split_col] = "holdout"

    logger.info(
        "Split (strategy=%s, seed=%d): train=%d val=%d holdout=%d",
        strategy,
        seed,
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(holdout_mask.sum()),
    )

    return {"train": train_mask, "val": val_mask, "holdout": holdout_mask}
