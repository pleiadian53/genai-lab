"""Training utilities for conditional flow matching (CFM).

The CFM training objective:

    L_CFM(theta) = E_{t, x0, x1} [ || v_theta(x_t, t) - u_t(x0, x1) ||^2 ]

where:
    t   ~ Uniform[0, 1]
    x0  ~ p_data
    x1  ~ N(0, I)
    x_t = psi_t(x0, x1)        (interpolated sample — network input)
    u_t = d/dt psi_t(x0, x1)   (conditional velocity — regression target)

This is a simple MSE regression. No score functions, no importance weights,
no noise schedules. One batch = one forward pass + one loss.backward().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .interpolants import BaseInterpolant, LinearInterpolant, broadcast_time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-step loss
# ---------------------------------------------------------------------------

def cfm_loss(
    model: nn.Module,
    x0: torch.Tensor,
    interpolant: BaseInterpolant,
    x1: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the CFM loss for one batch.

    Args:
        model:       Velocity network v_theta(x_t, t).
        x0:          Data batch, shape (B, *).
        interpolant: Defines psi_t and u_t.
        x1:          Noise batch (same shape as x0). Sampled from N(0,I) if None.

    Returns:
        Scalar loss (mean squared error over the batch).
    """
    B = x0.shape[0]
    device = x0.device

    # Sample noise if not provided
    if x1 is None:
        x1 = torch.randn_like(x0)

    # Sample time uniformly; reshape for broadcast against x0
    t_scalar = torch.rand(B, device=device)            # (B,)
    t = broadcast_time(t_scalar, x0)                   # (B, 1, 1, ...) matching x0.ndim

    # Build interpolated sample and velocity target
    x_t = interpolant.interpolate(x0, x1, t)           # network input
    u_t = interpolant.velocity(x0, x1, t)              # regression target

    # Forward pass: predict velocity at (x_t, t)
    v_pred = model(x_t, t_scalar)                      # t as (B,) for the network

    return ((v_pred - u_t) ** 2).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FlowMatchingTrainer:
    """Trains a velocity network using conditional flow matching.

    Handles:
    - Training loop with progress bars
    - Device detection (CUDA / MPS / CPU)
    - Periodic checkpoint saving
    - Optional validation loss logging

    Args:
        model:        Velocity network (VelocityUNet2D or VelocityMLP).
        interpolant:  Interpolant defining the training paths. Defaults to
                      LinearInterpolant (rectified flow).
        lr:           Learning rate for AdamW.
        device:       Training device. Auto-detected if None.
        checkpoint_dir: Directory for saving checkpoints. No saving if None.
        checkpoint_every: Save a checkpoint every N epochs.

    Example::

        from genailab.flow_matching import FlowMatchingTrainer, LinearInterpolant
        from genailab.flow_matching.velocity_networks import VelocityUNet2D

        model = VelocityUNet2D(in_channels=1, base_channels=32)
        trainer = FlowMatchingTrainer(model)
        history = trainer.fit(train_loader, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        interpolant: BaseInterpolant | None = None,
        lr: float = 1e-4,
        device: str | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 10,
    ):
        self.model = model
        self.interpolant = interpolant or LinearInterpolant()
        self.lr = lr
        self.checkpoint_every = checkpoint_every

        # Device: prefer CUDA, then MPS (M1), then CPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.01
        )

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: DataLoader | None = None,
        x1_loader: DataLoader | None = None,
        on_epoch_end: Callable[[dict], None] | None = None,
    ) -> dict[str, list[float]]:
        """Train for a given number of epochs.

        Args:
            train_loader:  DataLoader yielding (x0,) or (x0, label) batches.
            epochs:        Number of training epochs.
            val_loader:    Optional validation DataLoader for loss tracking.
            x1_loader:     Optional separate noise DataLoader (for OT coupling;
                           use None for independent N(0,I) noise, the standard).
            on_epoch_end:  Optional callback invoked after each epoch with a
                           dict ``{"epoch", "train_loss", "val_loss", "lr"}``.
                           Used by external experiment trackers (e.g., W&B) to
                           log per-epoch metrics without coupling the library
                           to a specific tracker.

        Returns:
            Training history dict with keys ``"train_loss"`` and ``"val_loss"``.
        """
        logger.info("Training on %s for %d epochs (lr=%.2e)", self.device, epochs, self.lr)

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, x1_loader)
            self.history["train_loss"].append(train_loss)
            self.scheduler.step()

            val_loss = None
            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                self.history["val_loss"].append(val_loss)

            self._log_epoch(epoch, epochs, train_loss, val_loss)

            if self.checkpoint_dir and epoch % self.checkpoint_every == 0:
                self._save_checkpoint(epoch)

            if on_epoch_end is not None:
                on_epoch_end({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

        return self.history

    def save(self, path: str | Path) -> None:
        """Save model weights to a file."""
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model weights from a file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        x1_loader: DataLoader | None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        x1_iter = iter(x1_loader) if x1_loader else None

        with tqdm(loader, leave=False, desc="  batch") as pbar:
            for batch in pbar:
                x0 = self._extract_images(batch).to(self.device)
                x1 = self._get_noise(x0, x1_iter)

                self.optimizer.zero_grad()
                loss = cfm_loss(self.model, x0, self.interpolant, x1)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            x0 = self._extract_images(batch).to(self.device)
            total_loss += cfm_loss(self.model, x0, self.interpolant).item()
        return total_loss / len(loader)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)
        logger.info("Checkpoint saved: %s", path)

    @staticmethod
    def _extract_images(batch: object) -> torch.Tensor:
        """Handle dataloaders that yield (image,) or (image, label)."""
        if isinstance(batch, (list, tuple)):
            return batch[0].float()
        return batch.float()

    @staticmethod
    def _get_noise(
        x0: torch.Tensor,
        x1_iter: object | None,
    ) -> torch.Tensor | None:
        """Get paired noise for OT coupling, or None for independent N(0,I)."""
        if x1_iter is None:
            return None
        try:
            x1_batch = next(x1_iter)
            x1 = x1_batch[0] if isinstance(x1_batch, (list, tuple)) else x1_batch
            return x1.float().to(x0.device)
        except StopIteration:
            return None

    @staticmethod
    def _log_epoch(epoch: int, total: int, train_loss: float, val_loss: float | None) -> None:
        msg = f"Epoch {epoch:4d}/{total}  train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.4f}"
        logger.info(msg)
