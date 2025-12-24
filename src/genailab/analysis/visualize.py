"""Visualization utilities for gene expression data and VAE results.

This module provides plotting functions for:
- UMAP/t-SNE embeddings
- Latent space visualization
- Reconstruction quality
- Training diagnostics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import scanpy as sc
    import matplotlib.pyplot as plt
    import torch


def plot_umap(
    adata: "sc.AnnData",
    color: str | list[str] | None = None,
    use_raw: bool = False,
    figsize: tuple = (8, 6),
    title: str | None = None,
    save: str | None = None,
    category: str = "exploration",
) -> None:
    """Plot UMAP embedding colored by metadata or gene expression.
    
    Args:
        adata: AnnData with UMAP computed (run sc.tl.umap first)
        color: Column(s) in obs or gene name(s) to color by
        use_raw: If True, use raw counts for gene expression coloring
        figsize: Figure size
        title: Plot title
        save: If provided, save figure to this filename in results/figures/{category}/
        category: Figure category for saving (default: "exploration")
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    from genailab.data.paths import get_results_paths
    
    # Compute UMAP if not present
    if "X_umap" not in adata.obsm:
        print("Computing UMAP...")
        # Need to compute neighbors first
        if "neighbors" not in adata.uns:
            # Temporary normalization for UMAP
            adata_temp = adata.copy()
            sc.pp.normalize_total(adata_temp, target_sum=1e4)
            sc.pp.log1p(adata_temp)
            sc.pp.pca(adata_temp)
            sc.pp.neighbors(adata_temp)
            sc.tl.umap(adata_temp)
            adata.obsm["X_umap"] = adata_temp.obsm["X_umap"]
        else:
            sc.tl.umap(adata)
    
    # Use our ResultsPaths instead of scanpy's default
    if save:
        results = get_results_paths()
        save_path = results.figure(category, save)
        sc.pl.umap(adata, color=color, use_raw=use_raw, title=title, show=False)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to: {save_path}")
    else:
        sc.pl.umap(adata, color=color, use_raw=use_raw, title=title)


def plot_latent_space(
    z: "np.ndarray | torch.Tensor",
    labels: "np.ndarray | None" = None,
    method: str = "umap",
    figsize: tuple = (8, 6),
    title: str = "Latent Space",
    alpha: float = 0.5,
    s: float = 5,
    save: str | None = None,
    category: str = "training",
) -> None:
    """Visualize VAE latent space using dimensionality reduction.
    
    Args:
        z: Latent representations (n_samples, latent_dim)
        labels: Optional labels for coloring points
        method: "umap", "tsne", or "pca"
        figsize: Figure size
        title: Plot title
        alpha: Point transparency
        s: Point size
        save: If provided, save figure to this filename in results/figures/{category}/
        category: Figure category for saving (default: "training")
    """
    import matplotlib.pyplot as plt
    from genailab.data.paths import get_results_paths
    
    # Convert to numpy if tensor
    if hasattr(z, 'detach'):
        z = z.detach().cpu().numpy()
    
    # Reduce to 2D if needed
    if z.shape[1] > 2:
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                z_2d = reducer.fit_transform(z)
            except ImportError:
                print("UMAP not installed, falling back to PCA")
                method = "pca"
        
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z)-1))
            z_2d = reducer.fit_transform(z)
        
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            z_2d = reducer.fit_transform(z)
    else:
        z_2d = z
        method = "direct"
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], label=label, alpha=alpha, s=s)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=alpha, s=s)
    
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save:
        results = get_results_paths()
        save_path = results.figure(category, save)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_reconstruction(
    x_true: "np.ndarray | torch.Tensor",
    x_recon: "np.ndarray | torch.Tensor",
    gene_names: list[str] | None = None,
    n_genes: int = 20,
    n_cells: int = 5,
    figsize: tuple = (12, 8),
    save: str | None = None,
    category: str = "evaluation",
) -> None:
    """Compare true vs reconstructed gene expression.
    
    Args:
        x_true: True expression values (n_cells, n_genes)
        x_recon: Reconstructed values (n_cells, n_genes)
        gene_names: Optional gene names
        n_genes: Number of top genes to show per cell
        n_cells: Number of cells to compare
        save: If provided, save figure to this filename in results/figures/{category}/
        category: Figure category for saving (default: "evaluation")
    """
    import matplotlib.pyplot as plt
    from genailab.data.paths import get_results_paths
    
    # Convert to numpy
    if hasattr(x_true, 'detach'):
        x_true = x_true.detach().cpu().numpy()
    if hasattr(x_recon, 'detach'):
        x_recon = x_recon.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, n_cells, figsize=figsize)
    
    for i in range(min(n_cells, x_true.shape[0])):
        # Top genes by true expression
        top_idx = np.argsort(x_true[i])[::-1][:n_genes]
        
        # Bar plot comparison
        x_pos = np.arange(n_genes)
        width = 0.35
        
        axes[0, i].bar(x_pos - width/2, x_true[i, top_idx], width, label='True', alpha=0.7)
        axes[0, i].bar(x_pos + width/2, x_recon[i, top_idx], width, label='Recon', alpha=0.7)
        axes[0, i].set_title(f"Cell {i}")
        axes[0, i].set_xlabel("Gene rank")
        axes[0, i].set_ylabel("Expression")
        if i == 0:
            axes[0, i].legend()
        
        # Scatter plot
        axes[1, i].scatter(x_true[i], x_recon[i], alpha=0.3, s=3)
        max_val = max(x_true[i].max(), x_recon[i].max())
        axes[1, i].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        axes[1, i].set_xlabel("True")
        axes[1, i].set_ylabel("Reconstructed")
        
        # Correlation
        corr = np.corrcoef(x_true[i], x_recon[i])[0, 1]
        axes[1, i].set_title(f"r = {corr:.3f}")
    
    plt.tight_layout()
    
    if save:
        results = get_results_paths()
        save_path = results.figure(category, save)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_training_curves(
    losses: dict[str, list[float]],
    figsize: tuple = (12, 4),
    save: str | None = None,
    category: str = "training",
) -> None:
    """Plot training loss curves.
    
    Args:
        losses: Dictionary with loss names as keys and lists of values
                e.g., {"total": [...], "recon": [...], "kl": [...]}
        figsize: Figure size
        save: If provided, save figure to this filename in results/figures/{category}/
        category: Figure category for saving (default: "training")
    """
    import matplotlib.pyplot as plt
    from genailab.data.paths import get_results_paths
    
    n_plots = len(losses)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, losses.items()):
        ax.plot(values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(f"{name} Loss")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        results = get_results_paths()
        save_path = results.figure(category, save)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


def plot_kl_per_dimension(
    mu: "np.ndarray | torch.Tensor",
    logvar: "np.ndarray | torch.Tensor",
    figsize: tuple = (10, 4),
) -> None:
    """Plot KL divergence contribution per latent dimension.
    
    Useful for diagnosing:
    - Posterior collapse (dimensions with KL ≈ 0)
    - Active vs inactive latent dimensions
    
    Args:
        mu: Mean of approximate posterior (n_samples, latent_dim)
        logvar: Log variance of approximate posterior
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy
    if hasattr(mu, 'detach'):
        mu = mu.detach().cpu().numpy()
    if hasattr(logvar, 'detach'):
        logvar = logvar.detach().cpu().numpy()
    
    # KL per dimension: 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    kl_per_dim = 0.5 * (mu**2 + np.exp(logvar) - logvar - 1)
    kl_mean = kl_per_dim.mean(axis=0)
    kl_std = kl_per_dim.std(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot of mean KL per dimension
    dims = np.arange(len(kl_mean))
    axes[0].bar(dims, kl_mean, yerr=kl_std, alpha=0.7, capsize=2)
    axes[0].set_xlabel("Latent dimension")
    axes[0].set_ylabel("KL divergence")
    axes[0].set_title("KL per Latent Dimension")
    axes[0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Collapse threshold')
    axes[0].legend()
    
    # Sorted KL
    sorted_idx = np.argsort(kl_mean)[::-1]
    axes[1].bar(range(len(kl_mean)), kl_mean[sorted_idx], alpha=0.7)
    axes[1].set_xlabel("Dimension (sorted)")
    axes[1].set_ylabel("KL divergence")
    axes[1].set_title("KL Sorted by Magnitude")
    
    # Count active dimensions
    n_active = (kl_mean > 0.1).sum()
    print(f"Active dimensions (KL > 0.1): {n_active}/{len(kl_mean)}")
    
    plt.tight_layout()
    plt.show()
