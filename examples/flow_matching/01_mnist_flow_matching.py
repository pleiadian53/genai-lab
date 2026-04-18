"""MNIST Flow Matching — training and sampling baseline.

Trains a small U-Net velocity field on MNIST using conditional flow matching
(rectified flow / linear interpolant). Saves checkpoints and generates a
sample grid at the end of training.

Usage
-----
# MPS (M1 MacBook) — fast POC with small model:
python examples/flow_matching/01_mnist_flow_matching.py

# Override key hyperparameters:
python examples/flow_matching/01_mnist_flow_matching.py \\
    --epochs 100 --batch-size 128 --base-channels 64 --n-steps 50

# RunPod (GPU) — full-quality run:
python examples/flow_matching/01_mnist_flow_matching.py \\
    --epochs 300 --batch-size 256 --base-channels 128 --n-steps 100 \\
    --checkpoint-dir runs/mnist_fm/

Output
------
- Checkpoint files: <checkpoint-dir>/checkpoint_epoch<N>.pt
- Sample grid:      <checkpoint-dir>/samples_final.png
- Training log:     stdout (configure logging to file if needed)

Architecture
------------
Velocity network: small U-Net (reuses genailab.diffusion.architectures.UNet2D)
Interpolant:      LinearInterpolant (rectified flow, straight paths)
Sampler:          EulerSampler (50 steps default) or RK4Sampler (20 steps)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils

# Ensure src/ is on the path when running as a script outside the package
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Load .env if present — makes WANDB_API_KEY etc. available without explicit
# shell export. Silent no-op if python-dotenv is not installed or .env is
# absent.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from genailab.flow_matching import (
    FlowMatchingTrainer,
    LinearInterpolant,
    VelocityUNet2D,
    EulerSampler,
    RK4Sampler,
    get_mnist_dataloader,
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
    p = argparse.ArgumentParser(description="MNIST flow matching baseline")

    # Training
    p.add_argument("--epochs",        type=int,   default=50,
                   help="Number of training epochs (default: 50)")
    p.add_argument("--batch-size",    type=int,   default=128,
                   help="Training batch size (default: 128)")
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--image-size",    type=int,   default=28,
                   help="Image size; MNIST native is 28 (default: 28)")

    # Architecture
    p.add_argument("--base-channels", type=int,   default=32,
                   help="U-Net base channel width. 32=MPS-safe, 128=RunPod (default: 32)")
    p.add_argument("--time-emb-dim",  type=int,   default=128,
                   help="Sinusoidal time embedding dimension (default: 128)")

    # Sampling
    p.add_argument("--sampler",       choices=["euler", "rk4"], default="euler",
                   help="ODE sampler for final sample generation (default: euler)")
    p.add_argument("--n-steps",       type=int,   default=50,
                   help="ODE integration steps at inference (default: 50)")
    p.add_argument("--n-samples",     type=int,   default=64,
                   help="Number of samples to generate at the end (default: 64)")

    # I/O
    p.add_argument("--checkpoint-dir", type=str,  default="runs/mnist_fm",
                   help="Directory for checkpoints and output (default: runs/mnist_fm)")
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="Save checkpoint every N epochs (default: 10)")
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint to resume training from")

    # Experiment tracking
    p.add_argument("--wandb",         action="store_true",
                   help="Log metrics, config, and samples to Weights & Biases. "
                        "Requires WANDB_API_KEY (from .env or shell env).")
    p.add_argument("--wandb-run-name", type=str,  default=None,
                   help="W&B run name (defaults to auto-generated from config)")
    p.add_argument("--wandb-tags",    type=str,   default="",
                   help="Comma-separated W&B run tags (e.g. 'baseline,mnist')")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MNIST Flow Matching")
    logger.info("  epochs=%d  batch=%d  lr=%.2e", args.epochs, args.batch_size, args.lr)
    logger.info("  base_channels=%d  image_size=%d", args.base_channels, args.image_size)
    logger.info("  sampler=%s  n_steps=%d", args.sampler, args.n_steps)
    logger.info("  output → %s", out_dir)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Experiment tracking (optional)
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        import wandb
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "genai-lab"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=args.wandb_run_name,
            tags=tags or None,
            config=vars(args),
            dir=str(out_dir),
        )
        logger.info("W&B run: %s", wandb_run.url if wandb_run else "(failed)")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader = get_mnist_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        split="train",
    )
    val_loader = get_mnist_dataloader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        split="test",
    )
    logger.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = VelocityUNet2D(
        in_channels=1,
        base_channels=args.base_channels,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=args.time_emb_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Velocity U-Net: %.2fM parameters", n_params / 1e6)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = FlowMatchingTrainer(
        model=model,
        interpolant=LinearInterpolant(),
        lr=args.lr,
        checkpoint_dir=out_dir,
        checkpoint_every=args.checkpoint_every,
    )

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=trainer.device)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _log_epoch_to_wandb(metrics: dict) -> None:
        """Callback: forward per-epoch metrics to W&B if enabled."""
        if wandb_run is None:
            return
        import wandb
        payload = {"train/loss": metrics["train_loss"], "lr": metrics["lr"]}
        if metrics["val_loss"] is not None:
            payload["val/loss"] = metrics["val_loss"]
        wandb.log(payload, step=metrics["epoch"])

    history = trainer.fit(
        train_loader=train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        on_epoch_end=_log_epoch_to_wandb if wandb_run else None,
    )

    # Save final weights
    trainer.save(out_dir / "model_final.pt")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    logger.info("Generating %d samples with %s (%d steps)…",
                args.n_samples, args.sampler, args.n_steps)

    if args.sampler == "euler":
        sampler = EulerSampler(model, n_steps=args.n_steps)
    else:
        sampler = RK4Sampler(model, n_steps=args.n_steps)

    samples = sampler.sample(
        n=args.n_samples,
        image_shape=(1, args.image_size, args.image_size),
    )

    # Rescale from [-1, 1] → [0, 1] for saving
    samples = (samples.clamp(-1, 1) + 1.0) / 2.0

    grid_path = out_dir / "samples_final.png"
    vutils.save_image(samples, grid_path, nrow=8, padding=2)
    logger.info("Sample grid saved → %s", grid_path)

    # ------------------------------------------------------------------
    # Loss summary
    # ------------------------------------------------------------------
    final_train = history["train_loss"][-1]
    logger.info("Final train loss: %.4f", final_train)
    if history["val_loss"]:
        logger.info("Final val   loss: %.4f", history["val_loss"][-1])

    # ------------------------------------------------------------------
    # Finalize W&B: log sample grid + summary metrics, upload checkpoint
    # ------------------------------------------------------------------
    if wandb_run is not None:
        import wandb
        summary = {"final/train_loss": final_train}
        if history["val_loss"]:
            summary["final/val_loss"] = history["val_loss"][-1]
        wandb.log({"samples_final": wandb.Image(str(grid_path))})
        for key, val in summary.items():
            wandb.run.summary[key] = val
        # Register the final checkpoint as a W&B artifact for reproducibility
        artifact = wandb.Artifact(
            name=f"{wandb_run.name}-model",
            type="model",
            metadata=vars(args),
        )
        artifact.add_file(str(out_dir / "model_final.pt"))
        wandb_run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
