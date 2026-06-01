"""P1 — Download, QC, and split Norman et al. 2019 Perturb-seq.

The first milestone of the perturbation flagship. Demonstrates the full
data-prep pipeline end-to-end and produces a summary artifact that
subsequent milestones (P2 baseline, P3 JEPA, P4 diffusion) consume.

Usage
-----
# First run — downloads ~1-2 GB to data/scrna/perturb_seq/norman_2019/
python examples/perturbation/P1_download_and_qc.py

# Custom QC thresholds
python examples/perturbation/P1_download_and_qc.py \\
    --min-genes 300 --pct-mt-max 15 --min-cells-per-pert 50

# Write the split-annotated AnnData to disk for downstream milestones
python examples/perturbation/P1_download_and_qc.py --save-qcd

# Skip download (assume file already present)
python examples/perturbation/P1_download_and_qc.py --no-download

Output
------
- examples/perturbation/results/P1_summary.json  (QC report + split stats)
- (optional) data/scrna/perturb_seq/norman_2019/NormanWeissman2019_qcd.h5ad

Background
----------
- Dataset: see notebooks/perturbation/docs/norman_2019_dataset_tutorial.md
- Milestone roadmap: see examples/perturbation/README.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when running as a script outside the package.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from genailab.applications.perturbation.data import (  # noqa: E402
    QCThresholds,
    download_norman,
    load_norman,
    qc_norman,
    split_norman,
    NORMAN_DEFAULT_CACHE_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P1 — Download, QC, and split Norman 2019 Perturb-seq.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=NORMAN_DEFAULT_CACHE_DIR,
        help=f"Where to cache the .h5ad (default: {NORMAN_DEFAULT_CACHE_DIR})",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download if the file is missing — fail instead.",
    )
    p.add_argument(
        "--save-qcd",
        action="store_true",
        help="Save the QC'd + split-annotated AnnData under cache-dir.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("examples/perturbation/results"),
        help="Where to write P1_summary.json.",
    )

    # QC overrides
    p.add_argument("--min-genes", type=int, default=200,
                   help="Drop cells with <N expressed genes (default: 200)")
    p.add_argument("--pct-mt-max", type=float, default=20.0,
                   help="Drop cells with >X%% mitochondrial counts (default: 20.0)")
    p.add_argument("--min-cells", type=int, default=3,
                   help="Drop genes in <N cells (default: 3)")
    p.add_argument("--min-cells-per-pert", type=int, default=30,
                   help="Drop perturbations with <N cells, controls exempt (default: 30)")

    # Split
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction per perturbation for validation (default: 0.15)")
    p.add_argument("--holdout-frac", type=float, default=0.15,
                   help="Fraction per perturbation for holdout (default: 0.15)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for split reproducibility (default: 42)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("P1: Norman 2019 — download, QC, split")
    logger.info("  cache_dir       = %s", args.cache_dir)
    logger.info("  min_genes       = %d", args.min_genes)
    logger.info("  pct_mt_max      = %.1f", args.pct_mt_max)
    logger.info("  min_cells       = %d", args.min_cells)
    logger.info("  min_cells_per_pert = %d", args.min_cells_per_pert)
    logger.info("  val_frac / holdout_frac = %.2f / %.2f", args.val_frac, args.holdout_frac)
    logger.info("  seed            = %d", args.seed)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Download (if needed) + load
    # ------------------------------------------------------------------
    if not args.no_download:
        download_norman(cache_dir=args.cache_dir)

    t_load_start = time.time()
    adata = load_norman(cache_dir=args.cache_dir, download_if_missing=False)
    t_load = time.time() - t_load_start
    logger.info("Load took %.1fs", t_load)

    # ------------------------------------------------------------------
    # QC
    # ------------------------------------------------------------------
    thresholds = QCThresholds(
        min_genes=args.min_genes,
        pct_mt_max=args.pct_mt_max,
        min_cells=args.min_cells,
        min_cells_per_perturbation=args.min_cells_per_pert,
    )

    t_qc_start = time.time()
    adata, qc_report = qc_norman(adata, thresholds=thresholds)
    t_qc = time.time() - t_qc_start
    logger.info("QC took %.1fs", t_qc)

    # ------------------------------------------------------------------
    # Split (cell-level stratified)
    # ------------------------------------------------------------------
    t_split_start = time.time()
    splits = split_norman(
        adata,
        strategy="cell",
        val_frac=args.val_frac,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
    )
    t_split = time.time() - t_split_start
    logger.info("Split took %.1fs", t_split)

    split_stats = {
        "strategy": "cell",
        "seed": args.seed,
        "val_frac": args.val_frac,
        "holdout_frac": args.holdout_frac,
        "n_train": int(splits["train"].sum()),
        "n_val": int(splits["val"].sum()),
        "n_holdout": int(splits["holdout"].sum()),
    }

    # ------------------------------------------------------------------
    # Distribution per perturbation (top 10 + control)
    # ------------------------------------------------------------------
    pert_counts = (
        adata.obs[qc_report.perturbation_col]
        .value_counts()
        .head(11)
        .to_dict()
    )

    # ------------------------------------------------------------------
    # Persist summary
    # ------------------------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "P1_summary.json"
    summary = {
        "thresholds": {
            "min_genes": thresholds.min_genes,
            "pct_mt_max": thresholds.pct_mt_max,
            "min_cells": thresholds.min_cells,
            "min_cells_per_perturbation": thresholds.min_cells_per_perturbation,
        },
        "qc_report": qc_report.to_dict(),
        "split": split_stats,
        "top_perturbations_by_cell_count": pert_counts,
        "timings_seconds": {
            "load": round(t_load, 2),
            "qc": round(t_qc, 2),
            "split": round(t_split, 2),
            "total": round(time.time() - t_start, 2),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary: %s", summary_path)

    # ------------------------------------------------------------------
    # Optionally save QC'd + split-annotated AnnData for downstream Px
    # ------------------------------------------------------------------
    if args.save_qcd:
        out_path = args.cache_dir / "NormanWeissman2019_qcd.h5ad"
        logger.info("Saving QC'd AnnData -> %s", out_path)
        adata.write_h5ad(out_path)
        logger.info("Saved: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("P1 Summary")
    print("=" * 60)
    print(f"  Cells:   {qc_report.n_cells_initial:>8,} -> {qc_report.n_cells_final:>8,}")
    print(f"  Genes:   {qc_report.n_genes_initial:>8,} -> {qc_report.n_genes_final:>8,}")
    print(f"  Perts:   {qc_report.n_perturbations_initial:>8,} -> {qc_report.n_perturbations_final:>8,}")
    print(f"  Control label: {qc_report.control_label}  ({qc_report.n_control_cells:,} cells)")
    print()
    print(f"  Splits:  train={split_stats['n_train']:,}  "
          f"val={split_stats['n_val']:,}  holdout={split_stats['n_holdout']:,}")
    print()
    print(f"  Library size: median={qc_report.library_size_stats['median']:.0f}  "
          f"mean={qc_report.library_size_stats['mean']:.0f}")
    print()
    print(f"  Total time: {time.time() - t_start:.1f}s")
    print(f"  Summary written: {summary_path}")
    print()


if __name__ == "__main__":
    main()
