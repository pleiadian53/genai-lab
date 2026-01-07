"""Score network architectures for diffusion models."""

import numpy as np
import torch
import torch.nn as nn


class SimpleScoreNetwork(nn.Module):
    """Simple MLP score network for low-dimensional data.
    
    The network predicts the score function s(x, t) = ∇_x log p_t(x).
    For denoising score matching, this equals -ε/σ(t) where ε is the noise.
    
    Architecture uses residual connections and layer normalization for stability.
    """
    
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, time_dim: int = 64, num_layers: int = 4):
        """
        Args:
            data_dim: Dimension of the data
            hidden_dim: Hidden layer dimension
            time_dim: Dimension of time embedding
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.time_dim = time_dim
        self.data_dim = data_dim
        
        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Main network with residual connections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )
    
    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding.
        
        Args:
            t: Time values, shape (batch_size,)
            
        Returns:
            Time embeddings, shape (batch_size, time_dim)
        """
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict score: s(x, t) ≈ ∇_x log p_t(x).
        
        Args:
            x: Data, shape (batch_size, data_dim)
            t: Time, shape (batch_size,)
        
        Returns:
            Predicted score, shape (batch_size, data_dim)
        """
        # Project input and time
        h = self.input_proj(x)
        t_emb = self.time_embedding(t)
        t_emb = self.time_proj(t_emb)
        
        # Add time embedding
        h = h + t_emb
        
        # Apply residual layers
        for layer in self.layers:
            h = h + layer(h)
        
        # Output projection
        return self.output_proj(h)
