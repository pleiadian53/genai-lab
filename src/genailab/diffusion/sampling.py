"""Sampling utilities for score-based diffusion models."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .sde import BaseSDE


@torch.no_grad()
def sample_reverse_sde(
    model: nn.Module,
    sde: BaseSDE,
    n_samples: int = 1000,
    num_steps: int = 500,
    data_dim: int = 2,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from reverse SDE using Euler-Maruyama discretization.
    
    The reverse SDE is:
        dx = [f(x,t) - g(t)² s_θ(x,t)] dt + g(t) dw̄
    
    Args:
        model: Trained score network
        sde: Forward SDE
        n_samples: Number of samples to generate
        num_steps: Number of discretization steps
        data_dim: Dimension of the data
        device: Device to run on (auto-detected if None)
    
    Returns:
        samples: Generated samples, shape (n_samples, data_dim)
        trajectory: Full trajectory, shape (num_snapshots, n_samples, data_dim)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    # Start from noise
    x = torch.randn(n_samples, data_dim, device=device)
    
    dt = -sde.T / num_steps
    trajectory = [x.cpu().numpy()]
    
    for i in tqdm(range(num_steps), desc="Sampling", leave=False):
        t = sde.T - i * (-dt)
        t_batch = torch.ones(n_samples, device=device) * t
        
        # Predict score
        score = model(x, t_batch)
        
        # Drift: f(x,t) - g(t)² * score
        drift = sde.drift(x.cpu().numpy(), t)
        drift = torch.FloatTensor(drift).to(device)
        g_t = sde.diffusion(t)
        drift = drift - (g_t ** 2) * score
        
        # Diffusion: g(t) * dw
        noise = torch.randn_like(x)
        diffusion = g_t * noise * np.sqrt(-dt)
        
        # Update
        x = x + drift * dt + diffusion
        
        if i % 50 == 0:
            trajectory.append(x.cpu().numpy())
    
    return x.cpu().numpy(), np.array(trajectory)


@torch.no_grad()
def sample_probability_flow_ode(
    model: nn.Module,
    sde: BaseSDE,
    n_samples: int = 1000,
    num_steps: int = 100,
    data_dim: int = 2,
    device: Optional[str] = None,
) -> np.ndarray:
    """Sample using probability flow ODE (deterministic).
    
    The probability flow ODE is:
        dx/dt = f(x,t) - 0.5 * g(t)² * s_θ(x,t)
    
    This generates samples deterministically (like DDIM).
    
    Args:
        model: Trained score network
        sde: Forward SDE
        n_samples: Number of samples to generate
        num_steps: Number of discretization steps
        data_dim: Dimension of the data
        device: Device to run on (auto-detected if None)
    
    Returns:
        samples: Generated samples, shape (n_samples, data_dim)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(n_samples, data_dim, device=device)
    dt = -sde.T / num_steps
    
    for i in tqdm(range(num_steps), desc="ODE Sampling", leave=False):
        t = sde.T - i * (-dt)
        t_batch = torch.ones(n_samples, device=device) * t
        
        score = model(x, t_batch)
        
        # ODE drift: f(x,t) - 0.5 * g(t)² * score
        drift = sde.drift(x.cpu().numpy(), t)
        drift = torch.FloatTensor(drift).to(device)
        g_t = sde.diffusion(t)
        drift = drift - 0.5 * (g_t ** 2) * score
        
        # Update (no noise!)
        x = x + drift * dt
    
    return x.cpu().numpy()
