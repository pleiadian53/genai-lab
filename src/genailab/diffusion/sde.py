"""Stochastic Differential Equations for diffusion models."""

import numpy as np
from abc import ABC, abstractmethod


class BaseSDE(ABC):
    """Abstract base class for SDEs."""
    
    @abstractmethod
    def beta(self, t):
        """Noise schedule."""
        pass
    
    @abstractmethod
    def drift(self, x, t):
        """Drift coefficient f(x, t)."""
        pass
    
    @abstractmethod
    def diffusion(self, t):
        """Diffusion coefficient g(t)."""
        pass
    
    @abstractmethod
    def marginal_prob(self, x0, t):
        """Compute mean and std of p_t(x | x_0)."""
        pass
    
    def sample_from_marginal(self, x0, t):
        """Sample x_t ~ p_t(x | x_0)."""
        mean, std = self.marginal_prob(x0, t)
        noise = np.random.randn(*x0.shape)
        return mean + std * noise, noise


class VPSDE(BaseSDE):
    """Variance-Preserving SDE.
    
    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
    
    This is the continuous-time limit of DDPM.
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        """
        Args:
            beta_min: Minimum noise level
            beta_max: Maximum noise level  
            T: Total diffusion time
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def beta(self, t):
        """Linear noise schedule."""
        return self.beta_min + (self.beta_max - self.beta_min) * t / self.T
    
    def drift(self, x, t):
        """Drift coefficient: f(x,t) = -0.5 * beta(t) * x"""
        return -0.5 * self.beta(t) * x
    
    def diffusion(self, t):
        """Diffusion coefficient: g(t) = sqrt(beta(t))"""
        return np.sqrt(self.beta(t))
    
    def marginal_prob(self, x0, t):
        """Compute mean and std of p_t(x | x_0).
        
        For VP-SDE:
            mean = sqrt(alpha_bar_t) * x_0
            std = sqrt(1 - alpha_bar_t)
        
        where alpha_bar_t = exp(-0.5 * integral_0^t beta(s) ds)
        
        Args:
            x0: Original data, shape (batch_size, data_dim) or (data_dim,)
            t: Time, scalar or shape (batch_size,)
            
        Returns:
            mean: Same shape as x0
            std: Scalar if t is scalar, else (batch_size, 1) for broadcasting
        """
        log_alpha_bar = -0.25 * t**2 * (self.beta_max - self.beta_min) / self.T - 0.5 * t * self.beta_min
        alpha_bar = np.exp(log_alpha_bar)
        
        # Handle broadcasting for batch case
        # Only reshape when x0 is 2D AND t is an array (batch case)
        if x0.ndim > 1 and np.ndim(t) > 0:
            alpha_bar = alpha_bar.reshape(-1, 1)
        
        mean = np.sqrt(alpha_bar) * x0
        std = np.sqrt(1 - alpha_bar)
        
        return mean, std


class VESDE(BaseSDE):
    """Variance-Exploding SDE.
    
    dx = sqrt(d[sigma^2(t)]/dt) * dw
    
    The variance grows without bound as t increases.
    """
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, T: float = 1.0):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            T: Total diffusion time
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
    
    def sigma(self, t):
        """Exponential noise schedule."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.T)
    
    def beta(self, t):
        """Effective beta for VE-SDE."""
        return 2 * self.sigma(t) * np.log(self.sigma_max / self.sigma_min) / self.T
    
    def drift(self, x, t):
        """Drift coefficient: f(x,t) = 0 for VE-SDE"""
        return np.zeros_like(x)
    
    def diffusion(self, t):
        """Diffusion coefficient: g(t) = sigma(t) * sqrt(2 * log(sigma_max/sigma_min))"""
        return self.sigma(t) * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min) / self.T)
    
    def marginal_prob(self, x0, t):
        """Compute mean and std of p_t(x | x_0).
        
        For VE-SDE:
            mean = x_0
            std = sigma(t)
        """
        sigma_t = self.sigma(t)
        
        # Handle broadcasting for batch case
        if x0.ndim > 1 and np.ndim(t) > 0:
            sigma_t = sigma_t.reshape(-1, 1)
        
        mean = x0.copy() if isinstance(x0, np.ndarray) else x0
        std = sigma_t
        
        return mean, std
