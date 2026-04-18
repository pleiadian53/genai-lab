"""Interpolant functions for conditional flow matching.

An interpolant defines how data points (x0) and noise points (x1) are
connected over time t ∈ [0, 1]. It must satisfy:
    psi_0(x0, x1) = x0   (path starts at data)
    psi_1(x0, x1) = x1   (path ends at noise)

The conditional velocity u_t = d/dt psi_t is the regression target during training.

References:
    Lipman et al. (2023) Flow Matching for Generative Modeling. ICLR.
    Liu et al. (2023) Flow Straight and Fast: Rectified Flow. ICLR.
    Albergo & Vanden-Eijnden (2023) Stochastic Interpolants. ICLR.
"""

from __future__ import annotations

import torch


class BaseInterpolant:
    """Abstract base for interpolants.

    Subclasses must implement :meth:`interpolate` and :meth:`velocity`.
    """

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the interpolated point x_t = psi_t(x0, x1).

        Args:
            x0: Data samples, shape (B, *).
            x1: Noise samples, same shape as x0.
            t:  Time values, shape (B, 1) broadcast-compatible with x0.

        Returns:
            Interpolated points x_t, same shape as x0.
        """
        raise NotImplementedError

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the conditional velocity u_t = d/dt psi_t(x0, x1).

        Args:
            x0: Data samples, shape (B, *).
            x1: Noise samples, same shape as x0.
            t:  Time values, shape (B, 1) broadcast-compatible with x0.
                Not used for the linear interpolant (constant velocity)
                but required for the general interface.

        Returns:
            Conditional velocity u_t, same shape as x0.
        """
        raise NotImplementedError

    def sample_pair(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x0, x1) with x1 sampled from N(0, I) if not provided."""
        if noise is None:
            noise = torch.randn_like(x0)
        return x0, noise


class LinearInterpolant(BaseInterpolant):
    """Linear (rectified-flow) interpolant: psi_t = (1-t)*x0 + t*x1.

    The path is a straight line from x0 to x1, giving a time-independent
    conditional velocity u_t = x1 - x0.

    This is the simplest and most commonly used interpolant. Straight paths
    are easy for ODE solvers to integrate (few NFEs needed).
    """

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = (1 - t) * x0 + t * x1."""
        return (1.0 - t) * x0 + t * x1

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u_t = x1 - x0 (constant, independent of t)."""
        return x1 - x0


class VPInterpolant(BaseInterpolant):
    """Variance-preserving interpolant: psi_t = sqrt(1-t)*x0 + sqrt(t)*x1.

    The coefficients maintain E[||x_t||^2] ≈ const when x0 and x1 have
    unit variance, mimicking the signal-to-noise schedule of DDPM-style
    diffusion models.

    Useful when comparing against diffusion baselines or when data scale
    must be preserved throughout the trajectory.
    """

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(1-t) * x0 + sqrt(t) * x1."""
        return (1.0 - t).sqrt() * x0 + t.sqrt() * x1

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u_t = d/dt [sqrt(1-t)*x0 + sqrt(t)*x1] = -x0/(2*sqrt(1-t)) + x1/(2*sqrt(t))."""
        eps = 1e-5  # numerical stability near t=0 and t=1
        return -x0 / (2.0 * (1.0 - t).clamp(min=eps).sqrt()) + x1 / (2.0 * t.clamp(min=eps).sqrt())


def broadcast_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape t to be broadcast-compatible with x.

    Converts t of shape (B,) or (B, 1) to (B, 1, 1, ...) matching x.ndim,
    so arithmetic like ``t * x`` works without manual reshaping in calling code.

    Args:
        t: Time tensor of shape (B,) or (B, 1).
        x: Reference tensor; only its shape after the batch dimension matters.

    Returns:
        t reshaped to (B, 1, 1, ...) with x.ndim total dimensions.
    """
    # Ensure t is (B, 1) first
    if t.ndim == 1:
        t = t.unsqueeze(1)
    # Expand to match x.ndim
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t
