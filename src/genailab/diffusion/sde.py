"""Stochastic Differential Equations for diffusion models."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union

from .schedules import NoiseSchedule, LinearSchedule, get_schedule


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
    
    Supports different noise schedules:
    - 'linear': Original DDPM schedule (default)
    - 'cosine': Improved DDPM schedule, better for images
    - 'sigmoid': Smooth transition, good for high-res
    - 'quadratic': Slower start, faster end
    """
    
    def __init__(
        self, 
        beta_min: float = 0.1, 
        beta_max: float = 20.0, 
        T: float = 1.0,
        schedule: Union[str, NoiseSchedule] = 'linear',
        **schedule_kwargs
    ):
        """
        Args:
            beta_min: Minimum noise level (for linear/quadratic schedules)
            beta_max: Maximum noise level (for linear/quadratic schedules)
            T: Total diffusion time
            schedule: Either a schedule name ('linear', 'cosine', 'sigmoid', 'quadratic')
                     or a NoiseSchedule instance
            **schedule_kwargs: Additional arguments for the schedule
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        
        # Set up the noise schedule
        if isinstance(schedule, NoiseSchedule):
            self.schedule = schedule
        elif schedule == 'linear':
            self.schedule = LinearSchedule(beta_min=beta_min, beta_max=beta_max, T=T)
        else:
            # For other schedules, pass T and any additional kwargs
            self.schedule = get_schedule(schedule, T=T, **schedule_kwargs)
        
        self.schedule_name = schedule if isinstance(schedule, str) else type(schedule).__name__
    
    def beta(self, t):
        """Noise schedule β(t)."""
        return self.schedule.beta(t)
    
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
        
        where alpha_bar_t is computed from the noise schedule.
        
        Args:
            x0: Original data, shape (batch_size, data_dim) or (data_dim,)
            t: Time, scalar or shape (batch_size,)
            
        Returns:
            mean: Same shape as x0
            std: Scalar if t is scalar, else (batch_size, 1) for broadcasting
        """
        # Use the schedule's alpha_bar method
        alpha_bar = self.schedule.alpha_bar(t)
        
        # Ensure alpha_bar broadcasts correctly with x0 for any dimensionality
        # This works for 1D, 2D, 3D (images), 4D (images with channels), etc.
        alpha_bar = np.asarray(alpha_bar)
        while alpha_bar.ndim < x0.ndim:
            alpha_bar = alpha_bar[..., None]
        
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
