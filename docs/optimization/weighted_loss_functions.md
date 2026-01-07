# Implementing Weighted Loss Functions

## Overview

Weighted loss functions of the form:

$$
\mathcal{L}(\theta) = \mathbb{E}\left[\lambda(x) \cdot \ell(\theta; x)\right] = \sum_{i} \lambda_i \cdot \ell(\theta; x_i)
$$

appear throughout machine learning. This document explains how to implement them effectively.

---

## Back References

This document was motivated by studying the diffusion model training loss:

- **Source**: [`notebooks/diffusion/02_sde_formulation/supplements/03_training_loss_and_denoising.md`](../notebooks/diffusion/sde_formulation/training_loss_and_denoising.md)

The weighted loss in diffusion models is:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \varepsilon} \left[\lambda(t) \left\| s_\theta(x_t, t) - \nabla_x \log p_t(x_t \mid x_0) \right\|^2\right]
$$

---

## Common Examples of Weighted Losses

| Algorithm | Loss Function | Weight $\lambda$ |
|-----------|---------------|------------------|
| **Diffusion Models** | $\lambda(t) \|s_\theta - \text{score}\|^2$ | Time-dependent, often $\lambda(t) = \sigma_t^2$ |
| **Weighted Least Squares** | $\sum_i w_i (y_i - \hat{y}_i)^2$ | Inverse variance or importance |
| **GloVe** | $\sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j - \log X_{ij})^2$ | Co-occurrence frequency cap |
| **Importance Sampling** | $\frac{1}{n}\sum_i \frac{p(x_i)}{q(x_i)} \ell(x_i)$ | Likelihood ratio |
| **Focal Loss** | $(1-p_t)^\gamma \cdot \text{CE}$ | Down-weight easy examples |
| **Class-Weighted CE** | $\sum_c w_c \cdot \text{CE}_c$ | Inverse class frequency |

---

## Implementation Strategies

### Strategy 1: Direct Weighting (Most Common)

Multiply the per-sample loss by the weight before reduction.

```python
import torch
import torch.nn.functional as F

def weighted_mse_loss(pred, target, weights):
    """
    Weighted mean squared error loss.
    
    Args:
        pred: Predictions, shape (batch_size, ...)
        target: Targets, shape (batch_size, ...)
        weights: Per-sample weights, shape (batch_size,) or (batch_size, 1, ...)
    
    Returns:
        Scalar loss
    """
    # Compute per-sample squared error
    squared_error = (pred - target) ** 2  # (batch_size, ...)
    
    # Reduce over non-batch dimensions
    per_sample_loss = squared_error.mean(dim=tuple(range(1, squared_error.dim())))  # (batch_size,)
    
    # Apply weights
    weighted_loss = weights * per_sample_loss  # (batch_size,)
    
    # Final reduction
    return weighted_loss.mean()
```

### Strategy 2: Weighting Before Reduction

For more control, weight before any reduction:

```python
def weighted_mse_loss_v2(pred, target, weights):
    """
    Alternative: weight each element before any reduction.
    Useful when weights vary across dimensions (e.g., per-pixel weights).
    """
    squared_error = (pred - target) ** 2
    
    # Expand weights to match error shape if needed
    if weights.dim() < squared_error.dim():
        weights = weights.view(-1, *([1] * (squared_error.dim() - 1)))
    
    weighted_error = weights * squared_error
    return weighted_error.mean()
```

### Strategy 3: Using `reduction='none'`

PyTorch's built-in losses with `reduction='none'` make this easy:

```python
def weighted_cross_entropy(logits, labels, weights):
    """
    Weighted cross-entropy using built-in loss with reduction='none'.
    """
    # Get per-sample loss
    per_sample_loss = F.cross_entropy(logits, labels, reduction='none')  # (batch_size,)
    
    # Apply weights and reduce
    return (weights * per_sample_loss).mean()
```

---

## Diffusion Model Implementation

Here's how the weighted diffusion loss is typically implemented:

```python
import torch
import torch.nn as nn

class DiffusionLoss(nn.Module):
    def __init__(self, weighting='sigma_squared'):
        super().__init__()
        self.weighting = weighting
    
    def get_weights(self, t, sigma_t):
        """
        Compute time-dependent weights.
        
        Args:
            t: Timesteps, shape (batch_size,)
            sigma_t: Noise std at each timestep, shape (batch_size,)
        """
        if self.weighting == 'sigma_squared':
            # Standard weighting: λ(t) = σ_t²
            return sigma_t ** 2
        elif self.weighting == 'uniform':
            # Uniform weighting: λ(t) = 1
            return torch.ones_like(t)
        elif self.weighting == 'snr':
            # Signal-to-noise ratio weighting
            alpha_t = torch.sqrt(1 - sigma_t ** 2)
            snr = (alpha_t / sigma_t) ** 2
            return snr
        elif self.weighting == 'min_snr':
            # Min-SNR weighting (Hang et al., 2023)
            alpha_t = torch.sqrt(1 - sigma_t ** 2)
            snr = (alpha_t / sigma_t) ** 2
            return torch.clamp(snr, max=5.0)  # Clamp to max SNR of 5
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
    
    def forward(self, score_pred, score_target, t, sigma_t):
        """
        Compute weighted diffusion loss.
        
        Args:
            score_pred: Network prediction, shape (batch_size, d)
            score_target: True score, shape (batch_size, d)
            t: Timesteps, shape (batch_size,)
            sigma_t: Noise std, shape (batch_size,)
        
        Returns:
            Scalar loss
        """
        # Compute per-sample MSE
        squared_error = (score_pred - score_target) ** 2  # (batch_size, d)
        per_sample_mse = squared_error.mean(dim=-1)  # (batch_size,)
        
        # Get weights
        weights = self.get_weights(t, sigma_t)  # (batch_size,)
        
        # Weighted mean
        loss = (weights * per_sample_mse).mean()
        
        return loss
```

### Full Training Loop Example

```python
def train_step(model, optimizer, x_0, noise_schedule):
    """
    Single training step for diffusion model.
    """
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # 1. Sample random timesteps
    t = torch.rand(batch_size, device=device)  # Uniform(0, 1)
    
    # 2. Get noise schedule values
    alpha_t = noise_schedule.alpha(t)  # (batch_size,)
    sigma_t = noise_schedule.sigma(t)  # (batch_size,)
    
    # Reshape for broadcasting
    alpha_t = alpha_t.view(-1, 1)  # (batch_size, 1)
    sigma_t_expanded = sigma_t.view(-1, 1)  # (batch_size, 1)
    
    # 3. Sample noise
    epsilon = torch.randn_like(x_0)  # (batch_size, d)
    
    # 4. Create noisy data
    x_t = alpha_t * x_0 + sigma_t_expanded * epsilon  # (batch_size, d)
    
    # 5. Predict score
    score_pred = model(x_t, t)  # (batch_size, d)
    
    # 6. Compute target score
    score_target = -epsilon / sigma_t_expanded  # (batch_size, d)
    
    # 7. Compute weighted loss
    weights = sigma_t ** 2  # (batch_size,)
    squared_error = (score_pred - score_target) ** 2
    per_sample_loss = squared_error.mean(dim=-1)  # (batch_size,)
    loss = (weights * per_sample_loss).mean()
    
    # 8. Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## GloVe Implementation

GloVe uses a capped weighting function:

$$
f(X_{ij}) = \begin{cases} 
(X_{ij}/x_{\max})^\alpha & \text{if } X_{ij} < x_{\max} \\
1 & \text{otherwise}
\end{cases}
$$

```python
def glove_weight(cooccurrence, x_max=100, alpha=0.75):
    """
    GloVe weighting function.
    
    Args:
        cooccurrence: Co-occurrence counts X_ij
        x_max: Maximum co-occurrence for full weight
        alpha: Exponent (typically 0.75)
    """
    return torch.where(
        cooccurrence < x_max,
        (cooccurrence / x_max) ** alpha,
        torch.ones_like(cooccurrence)
    )

def glove_loss(word_vectors, context_vectors, biases_w, biases_c, cooccurrence):
    """
    GloVe loss function.
    
    Args:
        word_vectors: (vocab_size, embedding_dim)
        context_vectors: (vocab_size, embedding_dim)
        biases_w, biases_c: (vocab_size,)
        cooccurrence: Sparse matrix of co-occurrence counts
    """
    # Get non-zero entries
    i, j = cooccurrence.nonzero(as_tuple=True)
    X_ij = cooccurrence[i, j]
    
    # Compute predictions
    dot_products = (word_vectors[i] * context_vectors[j]).sum(dim=-1)
    predictions = dot_products + biases_w[i] + biases_c[j]
    
    # Compute targets
    targets = torch.log(X_ij)
    
    # Compute weights
    weights = glove_weight(X_ij)
    
    # Weighted squared error
    squared_error = (predictions - targets) ** 2
    weighted_loss = (weights * squared_error).mean()
    
    return weighted_loss
```

---

## Weighted Least Squares Implementation

```python
import numpy as np
from scipy import linalg

def weighted_least_squares(X, y, weights):
    """
    Solve weighted least squares: min_β Σ w_i (y_i - X_i β)²
    
    Closed-form solution: β = (X^T W X)^{-1} X^T W y
    
    Args:
        X: Design matrix (n, p)
        y: Target (n,)
        weights: Weights (n,)
    
    Returns:
        beta: Coefficients (p,)
    """
    W = np.diag(weights)
    
    # Normal equations with weights
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    
    # Solve
    beta = linalg.solve(XtWX, XtWy)
    
    return beta

# PyTorch version for gradient-based optimization
def weighted_mse_regression(X, y, weights, beta):
    """
    Compute weighted MSE for linear regression.
    """
    predictions = X @ beta
    residuals = y - predictions
    weighted_squared_residuals = weights * (residuals ** 2)
    return weighted_squared_residuals.mean()
```

---

## Key Implementation Considerations

### 1. Weight Normalization

Should weights sum to 1, or be unnormalized?

```python
# Normalized weights (sum to 1)
weights_normalized = weights / weights.sum()
loss = (weights_normalized * per_sample_loss).sum()  # Note: .sum() not .mean()

# Unnormalized weights (more common in practice)
loss = (weights * per_sample_loss).mean()
```

**When to normalize**:
- Importance sampling (to get unbiased estimates)
- When comparing losses across different batches

**When not to normalize**:
- Most neural network training (weights are relative, not absolute)
- When using `.mean()` for final reduction

### 2. Numerical Stability

```python
def stable_weighted_loss(pred, target, weights, eps=1e-8):
    """
    Numerically stable weighted loss.
    """
    # Clamp weights to avoid extreme values
    weights = torch.clamp(weights, min=eps, max=1e6)
    
    per_sample_loss = ((pred - target) ** 2).mean(dim=-1)
    
    # Use log-sum-exp for very large weight ranges
    # (usually not needed, but good for importance sampling)
    
    return (weights * per_sample_loss).mean()
```

### 3. Gradient Flow

Weights can be:
- **Constants**: No gradient flows through weights
- **Learned**: Gradients flow, weights are updated
- **Computed from data**: May or may not want gradients

```python
# Weights as constants (most common)
with torch.no_grad():
    weights = compute_weights(t, sigma_t)
loss = (weights * per_sample_loss).mean()

# Weights with gradients (learned weighting)
weights = weight_network(t)  # Learnable
loss = (weights * per_sample_loss).mean()

# Detach if weights depend on model but shouldn't affect gradients
weights = compute_weights(model_output.detach())
```

### 4. Batch Statistics

Be careful when weights affect batch statistics:

```python
# Weighted mean (correct)
weighted_mean = (weights * values).sum() / weights.sum()

# Weighted variance
weighted_var = (weights * (values - weighted_mean) ** 2).sum() / weights.sum()
```

---

## Why Different Weightings?

### Diffusion Models: $\lambda(t) = \sigma_t^2$

**Problem**: At high noise levels, the score magnitude is smaller ($\propto 1/\sigma_t$), so the MSE is naturally smaller.

**Solution**: Weight by $\sigma_t^2$ to balance the loss across all noise levels. This ensures the network learns equally well at all timesteps.

### GloVe: Capped Frequency Weighting

**Problem**: Common word pairs (e.g., "the, of") have huge co-occurrence counts and would dominate the loss.

**Solution**: Cap the weight so frequent pairs don't overwhelm rare but informative pairs.

### Weighted Least Squares: Inverse Variance

**Problem**: Some observations are more reliable than others.

**Solution**: Weight by inverse variance $w_i = 1/\sigma_i^2$ so unreliable observations contribute less.

---

## Summary

| Aspect | Implementation |
|--------|----------------|
| **Basic pattern** | `loss = (weights * per_sample_loss).mean()` |
| **PyTorch built-ins** | Use `reduction='none'`, then weight manually |
| **Gradient flow** | Usually `torch.no_grad()` for weight computation |
| **Normalization** | Usually not needed with `.mean()` |
| **Numerical stability** | Clamp weights to reasonable range |

---

## References

- **Nichol & Dhariwal (2021)**: "Improved Denoising Diffusion Probabilistic Models" — Analysis of loss weighting
- **Pennington et al. (2014)**: "GloVe: Global Vectors for Word Representation"
- **Carroll & Ruppert (1988)**: "Transformation and Weighting in Regression"
- **Hang et al. (2023)**: "Efficient Diffusion Training via Min-SNR Weighting"

