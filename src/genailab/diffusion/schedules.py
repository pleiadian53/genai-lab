"""Noise schedules for diffusion models.

Different schedules affect:
- Training dynamics (how quickly the model learns different noise levels)
- Sample quality (some schedules preserve more signal at intermediate times)
- Generation speed (some schedules allow fewer sampling steps)

References:
- Linear: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
- Cosine: Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models"
- Sigmoid: Jabri et al. (2022) "Scalable Adaptive Computation for Iterative Generation"
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union

ArrayLike = Union[float, np.ndarray]


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""
    
    def __init__(self, T: float = 1.0):
        self.T = T
    
    @abstractmethod
    def beta(self, t: ArrayLike) -> ArrayLike:
        """Instantaneous noise rate β(t)."""
        pass
    
    @abstractmethod
    def alpha_bar(self, t: ArrayLike) -> ArrayLike:
        """Cumulative signal retention: ᾱ(t) = exp(-∫₀ᵗ β(s) ds)."""
        pass
    
    def snr(self, t: ArrayLike) -> ArrayLike:
        """Signal-to-noise ratio: SNR(t) = ᾱ(t) / (1 - ᾱ(t))."""
        ab = self.alpha_bar(t)
        return ab / (1 - ab + 1e-8)
    
    def log_snr(self, t: ArrayLike) -> ArrayLike:
        """Log signal-to-noise ratio."""
        return np.log(self.snr(t) + 1e-8)


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule (DDPM default).
    
    β(t) = β_min + (β_max - β_min) * t / T
    
    Simple and effective, but may spend too much capacity on high-noise regions.
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        super().__init__(T)
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: ArrayLike) -> ArrayLike:
        return self.beta_min + (self.beta_max - self.beta_min) * t / self.T
    
    def alpha_bar(self, t: ArrayLike) -> ArrayLike:
        # ∫₀ᵗ β(s) ds = β_min * t + 0.5 * (β_max - β_min) * t² / T
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2 / self.T
        return np.exp(-0.5 * integral)


class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule (Improved DDPM).
    
    ᾱ(t) = cos²(π/2 * (t/T + s) / (1 + s))
    
    Benefits:
    - More gradual noise increase at start
    - Better preservation of signal at intermediate times
    - Often produces higher quality samples
    
    The offset s prevents ᾱ(0) from being exactly 1.
    """
    
    def __init__(self, s: float = 0.008, T: float = 1.0):
        super().__init__(T)
        self.s = s
    
    def _f(self, t: ArrayLike) -> ArrayLike:
        """Helper function for cosine schedule."""
        return np.cos(np.pi / 2 * (t / self.T + self.s) / (1 + self.s)) ** 2
    
    def alpha_bar(self, t: ArrayLike) -> ArrayLike:
        return self._f(t) / self._f(0)
    
    def beta(self, t: ArrayLike) -> ArrayLike:
        # β(t) = -d/dt log(ᾱ(t))
        # Numerical approximation
        eps = 1e-4
        t_arr = np.asarray(t)
        log_ab = np.log(self.alpha_bar(t_arr) + 1e-8)
        log_ab_next = np.log(self.alpha_bar(t_arr + eps) + 1e-8)
        beta_val = -(log_ab_next - log_ab) / eps
        # Clip to prevent numerical issues
        return np.clip(beta_val, 0.0001, 20.0)


class SigmoidSchedule(NoiseSchedule):
    """Sigmoid noise schedule.
    
    ᾱ(t) = σ(-k * (t/T - 0.5)) / σ(k * 0.5)
    
    where σ is the sigmoid function.
    
    Benefits:
    - Smooth transition
    - Controllable steepness via k parameter
    - Good for high-resolution generation
    """
    
    def __init__(self, k: float = 10.0, T: float = 1.0):
        super().__init__(T)
        self.k = k
        # Normalization constant
        self._norm = self._sigmoid(self.k * 0.5)
    
    def _sigmoid(self, x: ArrayLike) -> ArrayLike:
        return 1 / (1 + np.exp(-x))
    
    def alpha_bar(self, t: ArrayLike) -> ArrayLike:
        return self._sigmoid(-self.k * (t / self.T - 0.5)) / self._norm
    
    def beta(self, t: ArrayLike) -> ArrayLike:
        # Numerical derivative
        eps = 1e-4
        t_arr = np.asarray(t)
        log_ab = np.log(self.alpha_bar(t_arr) + 1e-8)
        log_ab_next = np.log(self.alpha_bar(t_arr + eps) + 1e-8)
        beta_val = -(log_ab_next - log_ab) / eps
        return np.clip(beta_val, 0.0001, 20.0)


class QuadraticSchedule(NoiseSchedule):
    """Quadratic noise schedule.
    
    β(t) = β_min + (β_max - β_min) * (t/T)²
    
    Slower noise increase at the start, faster at the end.
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        super().__init__(T)
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: ArrayLike) -> ArrayLike:
        return self.beta_min + (self.beta_max - self.beta_min) * (t / self.T) ** 2
    
    def alpha_bar(self, t: ArrayLike) -> ArrayLike:
        # ∫₀ᵗ β(s) ds = β_min * t + (β_max - β_min) * t³ / (3T²)
        integral = self.beta_min * t + (self.beta_max - self.beta_min) * t**3 / (3 * self.T**2)
        return np.exp(-0.5 * integral)


# Convenience function to get schedule by name
def get_schedule(name: str, **kwargs) -> NoiseSchedule:
    """Get a noise schedule by name.
    
    Args:
        name: One of 'linear', 'cosine', 'sigmoid', 'quadratic'
        **kwargs: Schedule-specific parameters
        
    Returns:
        NoiseSchedule instance
    """
    schedules = {
        'linear': LinearSchedule,
        'cosine': CosineSchedule,
        'sigmoid': SigmoidSchedule,
        'quadratic': QuadraticSchedule,
    }
    
    if name not in schedules:
        raise ValueError(f"Unknown schedule: {name}. Choose from {list(schedules.keys())}")
    
    return schedules[name](**kwargs)
