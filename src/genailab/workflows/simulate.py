"""Counterfactual simulation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from genailab.data.loaders import ToyBulkDataset, split_dataset
from genailab.model.conditioning import ConditionSpec, ConditionEncoder
from genailab.model.vae import CVAE
from genailab.eval.counterfactual import counterfactual_decode, compute_counterfactual_effect


def simulate_counterfactual(
    model: CVAE,
    dataloader: DataLoader,
    target_condition: str,
    source_value: int,
    target_value: int,
    device: str = "cpu",
    n_samples: int | None = None,
) -> dict:
    """Simulate counterfactual expression changes.

    Args:
        model: Trained CVAE model
        dataloader: Data loader with samples to transform
        target_condition: Condition to change (e.g., 'disease')
        source_value: Original condition value to filter
        target_value: New condition value
        device: Device to run on
        n_samples: Max samples to process (None for all)

    Returns:
        Dict with original, counterfactual expressions and analysis
    """
    model.eval()
    model.to(device)

    all_x = []
    all_x_cf = []
    all_cond = []

    count = 0
    for batch in dataloader:
        # Filter to source condition
        mask = batch["cond"][target_condition] == source_value
        if mask.sum() == 0:
            continue

        x = batch["x"][mask].to(device)
        cond = {k: v[mask].to(device) for k, v in batch["cond"].items()}

        # Create counterfactual condition
        cond_cf = {k: v.clone() for k, v in cond.items()}
        cond_cf[target_condition] = torch.full_like(cond[target_condition], target_value)

        # Generate counterfactual
        with torch.no_grad():
            x_cf = counterfactual_decode(model, x, cond, cond_cf)

        all_x.append(x.cpu().numpy())
        all_x_cf.append(x_cf.cpu().numpy())
        all_cond.append({k: v.cpu().numpy() for k, v in cond.items()})

        count += x.shape[0]
        if n_samples is not None and count >= n_samples:
            break

    # Concatenate
    x_all = np.concatenate(all_x, axis=0)
    x_cf_all = np.concatenate(all_x_cf, axis=0)

    # Analyze effect
    effect = compute_counterfactual_effect(x_all, x_cf_all)

    return {
        "x_original": x_all,
        "x_counterfactual": x_cf_all,
        "effect": effect,
        "n_samples": x_all.shape[0],
        "source_condition": {target_condition: source_value},
        "target_condition": {target_condition: target_value},
    }


def main():
    """Main entry point for simulation."""
    # Load toy dataset
    print("Loading toy dataset...")
    ds = ToyBulkDataset(n=5000, n_genes=2000, n_tissues=6, n_diseases=3, n_batches=10)
    _, va_ds = split_dataset(ds, val_frac=0.1)
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

    # Try to load trained weights
    checkpoint_path = Path("runs/cvae_toy/best.pt")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using random weights")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate: disease 0 -> disease 1
    print("Simulating counterfactual: disease 0 -> disease 1...")
    result = simulate_counterfactual(
        model,
        val_loader,
        target_condition="disease",
        source_value=0,
        target_value=1,
        device=device,
        n_samples=100,
    )

    print(f"\nCounterfactual Analysis:")
    print(f"  Samples processed: {result['n_samples']}")
    print(f"  Mean absolute change: {result['effect']['mean_absolute_change']:.4f}")
    print(f"  Max change: {result['effect']['max_change']:.4f}")
    print(f"  Genes upregulated: {result['effect']['n_upregulated']}")
    print(f"  Genes downregulated: {result['effect']['n_downregulated']}")
    print(f"\nTop affected genes:")
    for gene, change in result['effect']['top_affected_genes'][:10]:
        direction = "↑" if change > 0 else "↓"
        print(f"    {gene}: {change:+.4f} {direction}")


if __name__ == "__main__":
    main()
