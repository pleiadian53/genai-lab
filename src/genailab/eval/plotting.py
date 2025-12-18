"""Plotting utilities for model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_latent_space(
    z: np.ndarray,
    labels: np.ndarray | None = None,
    label_name: str = "label",
    method: str = "umap",
    ax: Any = None,
    **kwargs,
) -> Any:
    """Plot latent space with dimensionality reduction.

    Args:
        z: Latent representations (samples x z_dim)
        labels: Optional labels for coloring
        label_name: Name for the label legend
        method: Reduction method ('umap', 'tsne', 'pca')
        ax: Matplotlib axis (created if None)
        **kwargs: Additional arguments for the reduction method

    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Dimensionality reduction
    if z.shape[1] > 2:
        if method == "umap":
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, **kwargs)
                z_2d = reducer.fit_transform(z)
            except ImportError:
                method = "pca"

        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, **kwargs)
            z_2d = reducer.fit_transform(z)

        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            z_2d = reducer.fit_transform(z)
    else:
        z_2d = z

    # Plot
    if labels is not None:
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10)
        ax.legend(*scatter.legend_elements(), title=label_name)
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, s=10)

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title("Latent Space")

    return ax


def plot_reconstruction(
    x: np.ndarray,
    x_hat: np.ndarray,
    gene_names: list[str] | None = None,
    n_samples: int = 5,
    n_genes: int = 50,
    ax: Any = None,
) -> Any:
    """Plot original vs reconstructed expression.

    Args:
        x: Original expression (samples x genes)
        x_hat: Reconstructed expression
        gene_names: Optional gene names
        n_samples: Number of samples to plot
        n_genes: Number of genes to show
        ax: Matplotlib axis

    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
        if n_samples == 1:
            axes = [axes]
    else:
        axes = [ax]
        n_samples = 1

    for i, ax in enumerate(axes[:n_samples]):
        if i >= x.shape[0]:
            break

        # Select top variable genes
        gene_var = x[i].var() if x.ndim == 1 else np.var(x, axis=0)
        top_genes = np.argsort(gene_var)[-n_genes:]

        ax.scatter(x[i, top_genes], x_hat[i, top_genes], alpha=0.5, s=20)

        # Add diagonal line
        lims = [
            min(x[i, top_genes].min(), x_hat[i, top_genes].min()),
            max(x[i, top_genes].max(), x_hat[i, top_genes].max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5)

        corr = np.corrcoef(x[i, top_genes], x_hat[i, top_genes])[0, 1]
        ax.set_xlabel("Original")
        ax.set_ylabel("Reconstructed")
        ax.set_title(f"Sample {i} (r={corr:.3f})")

    plt.tight_layout()
    return axes


def plot_counterfactual_effect(
    effect: dict,
    ax: Any = None,
    top_k: int = 20,
) -> Any:
    """Plot top genes affected by counterfactual intervention.

    Args:
        effect: Output from compute_counterfactual_effect
        ax: Matplotlib axis
        top_k: Number of top genes to show

    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    top_genes = effect["top_affected_genes"][:top_k]
    names = [g[0] for g in top_genes]
    changes = [g[1] for g in top_genes]

    colors = ["red" if c > 0 else "blue" for c in changes]
    ax.barh(range(len(names)), changes, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean Expression Change")
    ax.set_title("Top Affected Genes (Counterfactual)")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)

    return ax


def plot_training_curves(
    history: dict[str, list[float]],
    ax: Any = None,
) -> Any:
    """Plot training loss curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', etc.
        ax: Matplotlib axis

    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for key, values in history.items():
        ax.plot(values, label=key)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
