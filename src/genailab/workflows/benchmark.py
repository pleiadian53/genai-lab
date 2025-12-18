"""Benchmarking workflow for model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from genailab.eval.metrics import de_agreement, pathway_concordance, correlation_preservation
from genailab.eval.diagnostics import batch_leakage_score, posterior_collapse_check, reconstruction_quality


def run_benchmark(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
    gene_sets: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Run comprehensive benchmark on a trained model.

    Args:
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        gene_sets: Optional gene sets for pathway analysis

    Returns:
        Dict with all benchmark results
    """
    model.eval()
    model.to(device)

    results = {}

    # 1. Reconstruction quality
    print("Evaluating reconstruction quality...")
    results["reconstruction"] = reconstruction_quality(model, val_loader, device)

    # 2. Posterior collapse check
    print("Checking for posterior collapse...")
    results["posterior"] = posterior_collapse_check(model, val_loader, device)
    # Remove numpy array for JSON serialization
    if "kl_per_dim" in results["posterior"]:
        results["posterior"]["kl_per_dim"] = results["posterior"]["kl_per_dim"].tolist()

    # 3. Batch leakage
    print("Measuring batch leakage...")
    all_z = []
    all_batch = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            mu, _ = model.encode(x, cond)
            all_z.append(mu.cpu().numpy())
            all_batch.append(batch["cond"]["batch"].numpy())

    z = np.concatenate(all_z, axis=0)
    batch_labels = np.concatenate(all_batch, axis=0)
    results["batch_leakage"] = batch_leakage_score(z, batch_labels)

    # 4. Correlation preservation
    print("Measuring correlation preservation...")
    all_x = []
    all_x_hat = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            out = model(x, cond)
            all_x.append(x.cpu().numpy())
            all_x_hat.append(out["x_hat"].cpu().numpy())

    x_real = np.concatenate(all_x, axis=0)
    x_gen = np.concatenate(all_x_hat, axis=0)
    results["correlation"] = correlation_preservation(x_real, x_gen)

    # 5. Pathway concordance (if gene sets provided)
    if gene_sets is not None:
        print("Measuring pathway concordance...")
        results["pathway"] = pathway_concordance(x_real, x_gen, gene_sets)

    return results


def save_benchmark_results(results: dict, outdir: str = "runs/benchmark"):
    """Save benchmark results to JSON."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir) / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {outdir}/benchmark_results.json")


def print_benchmark_summary(results: dict):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if "reconstruction" in results:
        r = results["reconstruction"]
        print(f"\nReconstruction Quality:")
        print(f"  MSE: {r['mse']:.4f}")
        print(f"  MAE: {r['mae']:.4f}")
        print(f"  Mean Correlation: {r['mean_correlation']:.4f}")

    if "posterior" in results:
        p = results["posterior"]
        print(f"\nPosterior Analysis:")
        print(f"  Active dimensions: {p['n_active']} / {p['z_dim']}")
        print(f"  Collapsed dimensions: {p['n_collapsed']} / {p['z_dim']}")
        print(f"  Mean KL per dim: {p['mean_kl_per_dim']:.4f}")

    if "batch_leakage" in results:
        b = results["batch_leakage"]
        print(f"\nBatch Leakage:")
        print(f"  Classifier accuracy: {b['accuracy']:.4f}")
        print(f"  Random baseline: {b['baseline']:.4f}")
        print(f"  Leakage ratio: {b['leakage_ratio']:.2f}x")

    if "correlation" in results:
        c = results["correlation"]
        print(f"\nCorrelation Preservation:")
        print(f"  Correlation of correlations: {c['correlation_of_correlations']:.4f}")
        print(f"  Frobenius difference: {c['frobenius_difference']:.4f}")

    if "pathway" in results:
        pw = results["pathway"]
        print(f"\nPathway Concordance:")
        print(f"  Mean correlation: {pw['mean_correlation']:.4f}")
        print(f"  Pathways evaluated: {pw['n_pathways']}")

    print("=" * 60)


def main():
    """Main entry point for benchmarking."""
    from genailab.data.loaders import ToyBulkDataset, split_dataset
    from genailab.model.conditioning import ConditionSpec, ConditionEncoder
    from genailab.model.vae import CVAE

    # Load data
    print("Loading data...")
    ds = ToyBulkDataset(n=5000, n_genes=2000, n_tissues=6, n_diseases=3, n_batches=10)
    tr_ds, va_ds = split_dataset(ds, val_frac=0.1)

    train_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=0)

    # Create and load model
    print("Loading model...")
    spec = ConditionSpec(
        n_cats={"tissue": ds.n_tissues, "disease": ds.n_diseases, "batch": ds.n_batches},
        emb_dim=32,
        out_dim=128,
    )
    cond_enc = ConditionEncoder(spec)
    model = CVAE(n_genes=ds.n_genes, z_dim=64, cond_encoder=cond_enc, hidden=512)

    checkpoint_path = Path("runs/cvae_toy/best.pt")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using random weights")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run benchmark
    results = run_benchmark(model, train_loader, val_loader, device)

    # Save and print
    save_benchmark_results(results)
    print_benchmark_summary(results)


if __name__ == "__main__":
    main()
