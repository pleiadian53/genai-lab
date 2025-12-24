"""Demo script for understanding scGen (Single-cell Generative model).

scGen extends VAE ideas for perturbation prediction:
- Learn latent space from control cells
- Predict response to perturbation via latent arithmetic
- Generate counterfactual expression profiles

Key concept: "What would cell X look like after treatment Y?"

This is directly relevant to:
- Drug response prediction
- Perturbation biology (Perturb-seq)
- Counterfactual reasoning in causal-bio-lab

Requirements:
    pip install scgen

References:
    - Lotfollahi et al., "scGen predicts single-cell perturbation responses" (2019)
    - https://scgen.readthedocs.io/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def check_scgen_installed() -> bool:
    """Check if scgen is installed."""
    try:
        import scgen
        return True
    except ImportError:
        return False


def explain_scgen_concept() -> None:
    """Explain the core scGen concept without requiring installation."""
    print("""
scGen: Perturbation Response Prediction via Latent Arithmetic
=============================================================

Core Idea:
----------
scGen learns a VAE where perturbation effects are *linear* in latent space.

If you have:
  - Control cells: z_ctrl
  - Treated cells: z_treat
  
The "treatment effect" is approximately:
  δ = mean(z_treat) - mean(z_ctrl)

To predict how a NEW cell type responds to treatment:
  z_new_treated = z_new_ctrl + δ

This is "latent arithmetic" - similar to word2vec's king - man + woman = queen.

Why This Works:
---------------
1. VAE disentangles cell identity from perturbation response
2. Perturbation effects are often shared across cell types
3. Linear structure in latent space enables extrapolation

Limitations:
------------
1. Assumes perturbation effects are cell-type independent (often false)
2. Requires good latent space structure
3. Can't predict novel perturbations (only interpolate)

Connection to Our Roadmap:
--------------------------
- VAE: Learn good latent representations
- cVAE: Condition on cell type AND perturbation
- Counterfactual: Generate "what if" scenarios
- Causal: Ensure predictions respect causal structure

scGen is a stepping stone toward more sophisticated causal models.
""")


def demonstrate_latent_arithmetic(
    z_ctrl: np.ndarray,
    z_treat: np.ndarray,
    z_new: np.ndarray,
) -> np.ndarray:
    """Demonstrate the latent arithmetic concept.
    
    This is the core of scGen's prediction mechanism.
    
    Args:
        z_ctrl: Latent representations of control cells (n_ctrl, latent_dim)
        z_treat: Latent representations of treated cells (n_treat, latent_dim)
        z_new: Latent representations of new cells to predict (n_new, latent_dim)
        
    Returns:
        Predicted latent representations for new cells after treatment
    """
    # Compute treatment effect vector
    delta = z_treat.mean(axis=0) - z_ctrl.mean(axis=0)
    
    # Apply to new cells
    z_new_predicted = z_new + delta
    
    print(f"Treatment effect magnitude: {np.linalg.norm(delta):.3f}")
    print(f"Effect direction (top 3 dims): {delta[:3]}")
    
    return z_new_predicted


def simulate_scgen_workflow() -> None:
    """Simulate the scGen workflow with synthetic data.
    
    This demonstrates the concept without requiring real data or scgen installation.
    """
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # Simulate latent space
    latent_dim = 10
    n_cells = 100
    
    # Cell type A: control and treated
    z_A_ctrl = np.random.randn(n_cells, latent_dim) + np.array([1, 0, 0] + [0] * 7)
    z_A_treat = z_A_ctrl + np.array([0, 2, 0] + [0] * 7) + 0.1 * np.random.randn(n_cells, latent_dim)
    
    # Cell type B: only control available
    z_B_ctrl = np.random.randn(n_cells, latent_dim) + np.array([-1, 0, 0] + [0] * 7)
    
    # Predict B treated using latent arithmetic
    z_B_treat_pred = demonstrate_latent_arithmetic(z_A_ctrl, z_A_treat, z_B_ctrl)
    
    # Ground truth (for validation)
    z_B_treat_true = z_B_ctrl + np.array([0, 2, 0] + [0] * 7) + 0.1 * np.random.randn(n_cells, latent_dim)
    
    # Visualize (first 2 dimensions)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training data
    axes[0].scatter(z_A_ctrl[:, 0], z_A_ctrl[:, 1], alpha=0.5, label='A control', c='blue')
    axes[0].scatter(z_A_treat[:, 0], z_A_treat[:, 1], alpha=0.5, label='A treated', c='red')
    axes[0].scatter(z_B_ctrl[:, 0], z_B_ctrl[:, 1], alpha=0.5, label='B control', c='green')
    axes[0].set_xlabel('Latent dim 1')
    axes[0].set_ylabel('Latent dim 2')
    axes[0].set_title('Training Data')
    axes[0].legend()
    
    # Prediction
    axes[1].scatter(z_B_ctrl[:, 0], z_B_ctrl[:, 1], alpha=0.5, label='B control', c='green')
    axes[1].scatter(z_B_treat_pred[:, 0], z_B_treat_pred[:, 1], alpha=0.5, label='B treated (predicted)', c='orange')
    axes[1].scatter(z_B_treat_true[:, 0], z_B_treat_true[:, 1], alpha=0.3, label='B treated (true)', c='purple', marker='x')
    axes[1].set_xlabel('Latent dim 1')
    axes[1].set_ylabel('Latent dim 2')
    axes[1].set_title('Prediction via Latent Arithmetic')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compute prediction error
    error = np.linalg.norm(z_B_treat_pred - z_B_treat_true, axis=1).mean()
    print(f"\nMean prediction error: {error:.3f}")
    print("(Lower is better; 0 would be perfect)")


def train_scgen_model(
    adata: "sc.AnnData",
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    ctrl_key: str = "control",
    stim_key: str = "stimulated",
    n_latent: int = 100,
    max_epochs: int = 100,
) -> tuple:
    """Train scGen model on perturbation data.
    
    Args:
        adata: AnnData with perturbation labels
        condition_key: Column in obs for perturbation condition
        cell_type_key: Column in obs for cell type
        ctrl_key: Value indicating control condition
        stim_key: Value indicating stimulated/treated condition
        n_latent: Latent dimension
        max_epochs: Training epochs
        
    Returns:
        Tuple of (trained model, adata)
    """
    import scgen
    import scanpy as sc
    
    # Setup
    scgen.SCGEN.setup_anndata(adata, batch_key=condition_key)
    
    # Create model
    model = scgen.SCGEN(adata, n_latent=n_latent)
    
    # Train
    model.train(max_epochs=max_epochs, early_stopping=True)
    
    return model, adata


def predict_perturbation_response(
    model,
    adata: "sc.AnnData",
    cell_type_to_predict: str,
    condition_key: str = "condition",
    cell_type_key: str = "cell_type",
    ctrl_key: str = "control",
    stim_key: str = "stimulated",
) -> "sc.AnnData":
    """Predict perturbation response for a cell type.
    
    This is the key scGen capability: predict how cells respond
    to a perturbation they haven't seen.
    
    Args:
        model: Trained scGen model
        adata: Original AnnData
        cell_type_to_predict: Cell type to generate predictions for
        condition_key: Column for perturbation condition
        cell_type_key: Column for cell type
        ctrl_key: Control condition value
        stim_key: Stimulated condition value
        
    Returns:
        AnnData with predicted expression
    """
    # Get control cells of the target cell type
    ctrl_cells = adata[
        (adata.obs[cell_type_key] == cell_type_to_predict) &
        (adata.obs[condition_key] == ctrl_key)
    ]
    
    # Predict stimulated expression
    pred, delta = model.predict(
        ctrl_x=ctrl_cells,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
    )
    
    print(f"Predicted {pred.n_obs} cells")
    print(f"Treatment effect magnitude: {np.linalg.norm(delta):.3f}")
    
    return pred


def main():
    """Main demo function."""
    print("=" * 60)
    print("scGen Demo: Perturbation Response Prediction")
    print("=" * 60)
    
    # Always show the concept explanation
    explain_scgen_concept()
    
    if not check_scgen_installed():
        print("\nscgen not installed. Showing simulation instead...")
        print("Install with: pip install scgen")
        print("\n" + "=" * 60)
        print("Simulated Latent Arithmetic Demo")
        print("=" * 60)
        simulate_scgen_workflow()
    else:
        print("\nscgen is installed!")
        print("For a full demo, you need perturbation data (e.g., Kang et al. PBMC)")
        print("See: https://scgen.readthedocs.io/en/latest/tutorials/")
        
        # Still show simulation for concept
        print("\n" + "=" * 60)
        print("Simulated Latent Arithmetic Demo")
        print("=" * 60)
        simulate_scgen_workflow()
    
    print("\n" + "=" * 60)
    print("Key Takeaways for Our Roadmap:")
    print("=" * 60)
    print("""
1. VAE latent space should have linear structure for perturbations
2. Conditioning (cVAE) can separate cell type from treatment
3. Counterfactual = latent arithmetic + decoder
4. This is a stepping stone to causal perturbation models

Next steps in genai-lab:
- Implement basic VAE on PBMC
- Add conditioning (cVAE) on cell type
- Test latent arithmetic for "counterfactuals"
- Compare with scVI/scGen baselines
""")


if __name__ == "__main__":
    main()
