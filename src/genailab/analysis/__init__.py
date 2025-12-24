"""Analysis utilities for exploring and understanding gene expression data.

This module provides:
- AnnData exploration and inspection (explore_adata.py)
- Highly variable gene selection utilities (hvg.py)
- Visualization helpers (visualize.py)
- Reference model demos (scvi_demo.py, scgen_demo.py)
"""

from genailab.analysis.explore_adata import (
    summarize_adata,
    inspect_sparsity,
    plot_library_size,
    plot_gene_expression,
    compare_cells,
)

from genailab.analysis.hvg import (
    select_hvg,
    plot_hvg_dispersion,
)

__all__ = [
    # AnnData exploration
    "summarize_adata",
    "inspect_sparsity",
    "plot_library_size",
    "plot_gene_expression",
    "compare_cells",
    # HVG utilities
    "select_hvg",
    "plot_hvg_dispersion",
]
