"""Evaluation metrics for generative gene expression models."""

from __future__ import annotations

import numpy as np
from scipy import stats


def de_agreement(
    real_group1: np.ndarray,
    real_group2: np.ndarray,
    gen_group1: np.ndarray,
    gen_group2: np.ndarray,
    top_k: int = 100,
    method: str = "ttest",
) -> dict[str, float]:
    """Measure agreement in differentially expressed genes.

    Compares DE genes between real and generated contrasts.

    Args:
        real_group1: Real expression for group 1 (samples x genes)
        real_group2: Real expression for group 2 (samples x genes)
        gen_group1: Generated expression for group 1
        gen_group2: Generated expression for group 2
        top_k: Number of top DE genes to compare
        method: Statistical test ('ttest' or 'wilcoxon')

    Returns:
        Dict with 'overlap', 'jaccard', 'rank_correlation'
    """
    def get_de_genes(g1, g2, method):
        n_genes = g1.shape[1]
        pvals = np.zeros(n_genes)

        for i in range(n_genes):
            if method == "ttest":
                _, pvals[i] = stats.ttest_ind(g1[:, i], g2[:, i])
            else:
                _, pvals[i] = stats.mannwhitneyu(g1[:, i], g2[:, i], alternative="two-sided")

        return np.argsort(pvals)[:top_k]

    real_de = set(get_de_genes(real_group1, real_group2, method))
    gen_de = set(get_de_genes(gen_group1, gen_group2, method))

    overlap = len(real_de & gen_de)
    jaccard = overlap / len(real_de | gen_de) if len(real_de | gen_de) > 0 else 0

    # Rank correlation of p-values
    def get_pvals(g1, g2, method):
        n_genes = g1.shape[1]
        pvals = np.zeros(n_genes)
        for i in range(n_genes):
            if method == "ttest":
                _, pvals[i] = stats.ttest_ind(g1[:, i], g2[:, i])
            else:
                _, pvals[i] = stats.mannwhitneyu(g1[:, i], g2[:, i], alternative="two-sided")
        return pvals

    real_pvals = get_pvals(real_group1, real_group2, method)
    gen_pvals = get_pvals(gen_group1, gen_group2, method)

    # Handle NaN p-values
    valid = ~(np.isnan(real_pvals) | np.isnan(gen_pvals))
    if valid.sum() > 10:
        rank_corr, _ = stats.spearmanr(real_pvals[valid], gen_pvals[valid])
    else:
        rank_corr = 0.0

    return {
        "overlap": overlap,
        "jaccard": jaccard,
        "rank_correlation": rank_corr,
        "top_k": top_k,
    }


def pathway_concordance(
    real_expr: np.ndarray,
    gen_expr: np.ndarray,
    gene_sets: dict[str, list[int]],
    method: str = "mean",
) -> dict[str, float]:
    """Measure pathway activity concordance between real and generated data.

    Args:
        real_expr: Real expression (samples x genes)
        gen_expr: Generated expression (samples x genes)
        gene_sets: Dict mapping pathway name to gene indices
        method: Aggregation method ('mean', 'median', 'pca')

    Returns:
        Dict with pathway-level correlations
    """
    def pathway_score(expr, gene_idx, method):
        pathway_expr = expr[:, gene_idx]
        if method == "mean":
            return pathway_expr.mean(axis=1)
        elif method == "median":
            return np.median(pathway_expr, axis=1)
        elif method == "pca":
            from sklearn.decomposition import PCA
            if pathway_expr.shape[1] > 1:
                pca = PCA(n_components=1)
                return pca.fit_transform(pathway_expr).flatten()
            return pathway_expr.flatten()
        else:
            raise ValueError(f"Unknown method: {method}")

    correlations = {}
    for pathway_name, gene_idx in gene_sets.items():
        if len(gene_idx) == 0:
            continue

        # Filter valid indices
        valid_idx = [i for i in gene_idx if i < real_expr.shape[1] and i < gen_expr.shape[1]]
        if len(valid_idx) == 0:
            continue

        real_score = pathway_score(real_expr, valid_idx, method)
        gen_score = pathway_score(gen_expr, valid_idx, method)

        if len(real_score) > 2 and len(gen_score) > 2:
            corr, _ = stats.pearsonr(real_score, gen_score)
            correlations[pathway_name] = corr

    mean_corr = np.mean(list(correlations.values())) if correlations else 0.0

    return {
        "pathway_correlations": correlations,
        "mean_correlation": mean_corr,
        "n_pathways": len(correlations),
    }


def correlation_preservation(
    real_expr: np.ndarray,
    gen_expr: np.ndarray,
    n_genes_sample: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Measure preservation of gene-gene correlation structure.

    Args:
        real_expr: Real expression (samples x genes)
        gen_expr: Generated expression (samples x genes)
        n_genes_sample: Number of genes to sample for efficiency
        seed: Random seed

    Returns:
        Dict with correlation metrics
    """
    rng = np.random.default_rng(seed)

    n_genes = min(real_expr.shape[1], gen_expr.shape[1])
    if n_genes > n_genes_sample:
        idx = rng.choice(n_genes, n_genes_sample, replace=False)
        real_expr = real_expr[:, idx]
        gen_expr = gen_expr[:, idx]

    # Compute correlation matrices
    real_corr = np.corrcoef(real_expr.T)
    gen_corr = np.corrcoef(gen_expr.T)

    # Handle NaN
    valid = ~(np.isnan(real_corr) | np.isnan(gen_corr))

    if valid.sum() > 10:
        # Correlation of correlations (upper triangle)
        triu_idx = np.triu_indices(real_corr.shape[0], k=1)
        real_triu = real_corr[triu_idx]
        gen_triu = gen_corr[triu_idx]

        valid_triu = ~(np.isnan(real_triu) | np.isnan(gen_triu))
        if valid_triu.sum() > 10:
            corr_of_corr, _ = stats.pearsonr(real_triu[valid_triu], gen_triu[valid_triu])
        else:
            corr_of_corr = 0.0

        # Frobenius norm of difference
        frob_diff = np.linalg.norm(real_corr[valid] - gen_corr[valid])
    else:
        corr_of_corr = 0.0
        frob_diff = float("inf")

    return {
        "correlation_of_correlations": corr_of_corr,
        "frobenius_difference": frob_diff,
        "n_genes_compared": real_expr.shape[1],
    }
