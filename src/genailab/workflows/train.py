"""Training workflow for conditional VAE."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genailab.data.loaders import ToyBulkDataset, split_dataset
from genailab.model.conditioning import ConditionSpec, ConditionEncoder
from genailab.model.vae import CVAE, elbo_loss


def train_cvae(
    model: CVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cuda",
    lr: float = 1e-3,
    epochs: int = 50,
    beta: float = 1.0,
    outdir: str = "runs/cvae",
    verbose: bool = True,
) -> dict:
    """Train a conditional VAE.

    Args:
        model: CVAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        lr: Learning rate
        epochs: Number of epochs
        beta: KL weight (beta-VAE)
        outdir: Output directory for checkpoints
        verbose: Print progress

    Returns:
        Training history dict
    """
    os.makedirs(outdir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [], "train_kl": [], "val_kl": []}
    best_val = float("inf")

    for ep in range(1, epochs + 1):
        # Training
        model.train()
        tr = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for batch in train_loader:
            x = batch["x"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            out = model(x, cond)
            loss, parts = elbo_loss(x, out["x_hat"], out["mu"], out["logvar"], beta=beta)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tr["loss"] += loss.item()
            tr["recon"] += parts["recon"].item()
            tr["kl"] += parts["kl"].item()

        # Validation
        model.eval()
        va = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                cond = {k: v.to(device) for k, v in batch["cond"].items()}
                out = model(x, cond)
                loss, parts = elbo_loss(x, out["x_hat"], out["mu"], out["logvar"], beta=beta)

                va["loss"] += loss.item()
                va["recon"] += parts["recon"].item()
                va["kl"] += parts["kl"].item()

        # Average
        ntr = len(train_loader)
        nva = len(val_loader)
        tr = {k: v / max(ntr, 1) for k, v in tr.items()}
        va = {k: v / max(nva, 1) for k, v in va.items()}

        # Record history
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_recon"].append(tr["recon"])
        history["val_recon"].append(va["recon"])
        history["train_kl"].append(tr["kl"])
        history["val_kl"].append(va["kl"])

        if verbose:
            print(f"epoch {ep:03d} | train loss={tr['loss']:.4f} recon={tr['recon']:.4f} kl={tr['kl']:.4f} | val loss={va['loss']:.4f}")

        # Save best model
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), os.path.join(outdir, "best.pt"))
            with open(os.path.join(outdir, "best_metrics.json"), "w") as f:
                json.dump({"epoch": ep, "train": tr, "val": va}, f, indent=2)

    # Save final model and history
    torch.save(model.state_dict(), os.path.join(outdir, "final.pt"))
    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


def main():
    """Main entry point for training."""
    # Create toy dataset
    print("Creating toy dataset...")
    ds = ToyBulkDataset(n=5000, n_genes=2000, n_tissues=6, n_diseases=3, n_batches=10)
    tr_ds, va_ds = split_dataset(ds, val_frac=0.1)

    train_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=0)

    # Create model
    print("Creating model...")
    spec = ConditionSpec(
        n_cats={"tissue": ds.n_tissues, "disease": ds.n_diseases, "batch": ds.n_batches},
        emb_dim=32,
        out_dim=128,
    )
    cond_enc = ConditionEncoder(spec)
    model = CVAE(n_genes=ds.n_genes, z_dim=64, cond_encoder=cond_enc, hidden=512)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Train
    print("Starting training...")
    history = train_cvae(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=30,
        beta=0.5,
        outdir="runs/cvae_toy",
    )

    print(f"Training complete. Best val loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
