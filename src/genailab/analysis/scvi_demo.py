"""Demo script for understanding scVI (Single-cell Variational Inference).

scVI is the gold standard for VAE-based scRNA-seq analysis. Understanding
how it works helps us build our own implementations correctly.

Key scVI concepts:
1. Raw counts as input (no normalization)
2. Library size as explicit covariate
3. Negative Binomial decoder (not Gaussian)
4. Batch correction via conditioning

This script demonstrates:
- Loading and preparing data for scVI
- Training a basic scVI model
- Extracting latent representations
- Understanding the model architecture

Requirements:
    pip install scvi-tools

References:
    - Lopez et al., "Deep generative modeling for single-cell transcriptomics" (2018)
    - https://scvi-tools.org/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def check_scvi_installed() -> bool:
    """Check if scvi-tools is installed."""
    try:
        import scvi
        return True
    except ImportError:
        return False


def prepare_data_for_scvi(adata_path: str | Path) -> "sc.AnnData":
    """Prepare AnnData for scVI training.
    
    scVI expects:
    - Raw counts in adata.X (integers, not normalized)
    - Optional batch information in adata.obs
    
    Args:
        adata_path: Path to h5ad file with raw counts
        
    Returns:
        AnnData ready for scVI
    """
    import scanpy as sc
    
    adata = sc.read_h5ad(adata_path)
    
    # Verify raw counts
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    if not np.allclose(X, X.astype(int)):
        warnings.warn("Data may not be raw counts (non-integer values detected)")
    
    # scVI works best with HVGs
    if "highly_variable" not in adata.var.columns:
        print("Selecting HVGs for scVI...")
        # scVI has its own HVG selection that works on raw counts
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=2000,
            flavor="seurat_v3",
            subset=False,
        )
    
    print(f"Data shape: {adata.shape}")
    print(f"HVGs: {adata.var.highly_variable.sum()}")
    
    return adata


def train_scvi_model(
    adata: "sc.AnnData",
    n_latent: int = 10,
    n_layers: int = 1,
    n_hidden: int = 128,
    max_epochs: int = 100,
    batch_key: str | None = None,
    use_hvg: bool = True,
) -> tuple:
    """Train a basic scVI model.
    
    This demonstrates the standard scVI workflow.
    
    Args:
        adata: AnnData with raw counts
        n_latent: Latent space dimension
        n_layers: Number of hidden layers
        n_hidden: Hidden layer size
        max_epochs: Training epochs
        batch_key: Column in obs for batch correction
        use_hvg: If True, subset to HVGs
        
    Returns:
        Tuple of (trained model, adata with latent representation)
    """
    import scvi
    
    # Subset to HVGs if requested
    if use_hvg and "highly_variable" in adata.var.columns:
        adata = adata[:, adata.var.highly_variable].copy()
        print(f"Subset to {adata.n_vars} HVGs")
    
    # Setup anndata for scVI
    # This registers the data and validates it
    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
        # layer="counts",  # Use if counts are in a layer
    )
    
    # Create model
    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood="nb",  # Negative Binomial - the key choice!
    )
    
    print("\nscVI Model Architecture:")
    print(f"  Input: {adata.n_vars} genes")
    print(f"  Hidden: {n_hidden} units × {n_layers} layers")
    print(f"  Latent: {n_latent} dimensions")
    print(f"  Likelihood: Negative Binomial")
    print(f"  Batch correction: {batch_key is not None}")
    
    # Train
    print(f"\nTraining for {max_epochs} epochs...")
    model.train(max_epochs=max_epochs, early_stopping=True)
    
    # Get latent representation
    adata.obsm["X_scVI"] = model.get_latent_representation()
    
    return model, adata


def analyze_scvi_model(model, adata: "sc.AnnData") -> dict:
    """Analyze a trained scVI model.
    
    This shows what we can extract from scVI:
    - Latent representations
    - Reconstructed expression
    - Denoised expression
    - Differential expression
    
    Args:
        model: Trained scVI model
        adata: AnnData used for training
        
    Returns:
        Dictionary with analysis results
    """
    import scanpy as sc
    
    results = {}
    
    # 1. Latent representation
    z = model.get_latent_representation()
    results["latent_mean"] = z.mean(axis=0)
    results["latent_std"] = z.std(axis=0)
    print(f"Latent shape: {z.shape}")
    print(f"Latent mean range: [{z.mean(axis=0).min():.2f}, {z.mean(axis=0).max():.2f}]")
    
    # 2. Normalized expression (denoised)
    # This is what scVI thinks the "true" expression is
    denoised = model.get_normalized_expression(return_numpy=True)
    results["denoised_mean"] = denoised.mean()
    print(f"Denoised expression mean: {denoised.mean():.2f}")
    
    # 3. ELBO (evidence lower bound)
    elbo = model.get_elbo()
    results["elbo"] = elbo
    print(f"ELBO: {elbo:.2f}")
    
    # 4. Compute UMAP on latent space
    adata.obsm["X_scVI"] = z
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)
    
    return results


def compare_with_pca(adata: "sc.AnnData") -> None:
    """Compare scVI latent space with PCA.
    
    This visualization shows how scVI captures structure
    differently from linear methods.
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    # Compute PCA on log-normalized data
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    sc.pp.pca(adata_norm)
    
    # Transfer PCA to original adata
    adata.obsm["X_pca"] = adata_norm.obsm["X_pca"]
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA
    sc.pl.embedding(adata, basis="pca", ax=axes[0], show=False, title="PCA")
    
    # scVI
    sc.pl.embedding(adata, basis="umap", ax=axes[1], show=False, title="scVI UMAP")
    
    plt.tight_layout()
    plt.show()


def demonstrate_counterfactual(model, adata: "sc.AnnData") -> None:
    """Demonstrate counterfactual generation with scVI.
    
    This is the key capability for generative biology:
    "What would this cell look like under different conditions?"
    
    Note: Requires batch_key to be set during training.
    """
    if "batch" not in adata.obs.columns:
        print("Counterfactual demo requires batch information")
        return
    
    # Get expression under different batch conditions
    batches = adata.obs["batch"].unique()
    
    print(f"Generating counterfactuals for {len(batches)} batches...")
    
    for batch in batches[:2]:  # Just show first 2
        # Get normalized expression as if all cells were in this batch
        expr = model.get_normalized_expression(
            transform_batch=batch,
            return_numpy=True,
        )
        print(f"  Batch '{batch}': mean expression = {expr.mean():.2f}")


def main():
    """Main demo function."""
    from genailab.data.paths import get_data_paths
    
    if not check_scvi_installed():
        print("scvi-tools not installed!")
        print("Install with: pip install scvi-tools")
        print("\nThis demo shows how scVI works conceptually.")
        print("Key takeaways:")
        print("  1. scVI uses raw counts (no normalization)")
        print("  2. Library size is modeled explicitly")
        print("  3. Negative Binomial likelihood handles overdispersion")
        print("  4. Latent space captures biological variation")
        print("  5. Batch effects can be corrected via conditioning")
        return
    
    paths = get_data_paths()
    adata_path = paths.scrna_processed("pbmc3k", "counts.h5ad")
    
    if not adata_path.exists():
        print(f"Data not found: {adata_path}")
        print("Run: python -m genailab.data.sc_preprocess --dataset pbmc3k")
        return
    
    print("=" * 60)
    print("scVI Demo: Understanding VAE for scRNA-seq")
    print("=" * 60)
    
    # Prepare data
    print("\n1. Preparing data...")
    adata = prepare_data_for_scvi(adata_path)
    
    # Train model
    print("\n2. Training scVI model...")
    model, adata = train_scvi_model(
        adata,
        n_latent=10,
        max_epochs=50,  # Short for demo
    )
    
    # Analyze
    print("\n3. Analyzing model...")
    results = analyze_scvi_model(model, adata)
    
    # Compare with PCA
    print("\n4. Comparing with PCA...")
    compare_with_pca(adata)
    
    print("\n" + "=" * 60)
    print("Key observations:")
    print("  - scVI learns a nonlinear latent space")
    print("  - NB likelihood handles count overdispersion")
    print("  - Library size variation is factored out")
    print("  - Latent space captures biological variation")
    print("=" * 60)


if __name__ == "__main__":
    main()
