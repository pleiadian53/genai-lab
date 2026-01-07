"""Training utilities for score-based diffusion models."""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .sde import BaseSDE


def train_score_network(
    model: nn.Module,
    data: np.ndarray,
    sde: BaseSDE,
    num_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: Optional[str] = None,
    eps: float = 1e-3,
) -> List[float]:
    """Train score network using denoising score matching.
    
    The network learns to predict the score: s_θ(x_t, t) ≈ -ε/σ(t)
    
    We use the standard DSM loss with proper weighting:
        L = E_t[ λ(t) * E_{x0,ε}[ ||s_θ(x_t, t) - ∇log p(x_t|x0)||² ] ]
    
    where ∇log p(x_t|x0) = -ε/σ(t) and λ(t) = σ(t)² for variance reduction.
    
    Args:
        model: Score network to train
        data: Training data, shape (n_samples, data_dim)
        sde: Forward SDE defining the diffusion process
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on (auto-detected if None)
        eps: Minimum time value to avoid σ(t) ≈ 0
        
    Returns:
        List of loss values per epoch
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    data_tensor = torch.FloatTensor(data).to(device)
    n_samples = len(data)
    
    losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Sample batch
        idx = np.random.choice(n_samples, batch_size, replace=True)
        x0 = data_tensor[idx]
        
        # Sample time uniformly, avoiding t near 0 where std ≈ 0
        t = torch.rand(batch_size, device=device) * (sde.T - eps) + eps
        
        # Compute marginal distribution parameters
        mean_np, std_np = sde.marginal_prob(x0.cpu().numpy(), t.cpu().numpy())
        mean = torch.FloatTensor(mean_np).to(device)
        std = torch.FloatTensor(std_np).to(device)
        
        # Sample noisy data: x_t = mean + std * noise
        noise = torch.randn_like(x0)
        xt = mean + std * noise
        
        # Target score: -noise / std (the true conditional score)
        target_score = -noise / std
        
        # Predict score
        pred_score = model(xt, t)
        
        # DSM loss with λ(t) = σ(t)² weighting (reduces to noise prediction MSE)
        # ||s_θ - (-ε/σ)||² * σ² = ||σ*s_θ + ε||²
        loss = torch.mean((pred_score - target_score) ** 2 * (std ** 2))
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return losses
