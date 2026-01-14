# Latent Diffusion Foundations: Architecture and Components

This document covers the detailed architecture of latent diffusion models for computational biology, including VAE encoders with NB/ZINB decoders, latent diffusion models, and conditioning mechanisms.

**Prerequisites**: Understanding of [latent diffusion overview](00_latent_diffusion_overview.md), VAE basics, and diffusion models.

---

## Architecture Overview

### Two-Stage Pipeline

**Stage 1: Autoencoder** (Compress to latent space)
```
Gene Expression (20K dims) → Encoder → Latent (256 dims) → Decoder → Gene Expression
```

**Stage 2: Diffusion** (Generate in latent space)
```
Noise → Denoising Process → Latent → Decoder → Gene Expression
```

**Key insight**: Diffuse in compressed latent space (256 dims) instead of raw gene space (20K dims) → 78× fewer dimensions.

---

## 1. Autoencoder Stage

### 1.1 VAE for Gene Expression

**Why VAE for biology**:
- Probabilistic (captures uncertainty)
- Continuous latent space (smooth interpolation)
- Well-understood training (ELBO objective)
- Works well with count data (via appropriate decoder)

### 1.2 Encoder Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneExpressionEncoder(nn.Module):
    """
    Encoder for gene expression data.
    
    Maps high-dimensional gene expression to low-dimensional latent space.
    
    Args:
        num_genes: Number of genes (input dimension)
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(
        self,
        num_genes=20000,
        latent_dim=256,
        hidden_dims=[2048, 1024, 512],
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        
        # Encoder layers
        layers = []
        in_dim = num_genes
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x):
        """
        Encode gene expression to latent distribution.
        
        Args:
            x: Gene expression (B, num_genes)
        
        Returns:
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
        """
        # Encode
        h = self.encoder(x)
        
        # Latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        
        Args:
            mu: Mean (B, latent_dim)
            logvar: Log variance (B, latent_dim)
        
        Returns:
            z: Sampled latent (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
```

### 1.3 Decoder with NB/ZINB Distribution

**Why NB/ZINB for gene expression**:
- Gene expression counts are overdispersed (variance > mean)
- Negative Binomial (NB) models overdispersion
- Zero-Inflated NB (ZINB) models excess zeros (dropout)

**Negative Binomial Decoder**:

```python
class NegativeBinomialDecoder(nn.Module):
    """
    Decoder with Negative Binomial output distribution.
    
    Appropriate for count data with overdispersion.
    
    Args:
        latent_dim: Latent space dimension
        num_genes: Number of genes (output dimension)
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(
        self,
        latent_dim=256,
        num_genes=20000,
        hidden_dims=[512, 1024, 2048],
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_genes = num_genes
        
        # Decoder layers
        layers = []
        in_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # NB parameters
        self.fc_mean = nn.Linear(hidden_dims[-1], num_genes)
        self.fc_dispersion = nn.Linear(hidden_dims[-1], num_genes)
    
    def forward(self, z, library_size=None):
        """
        Decode latent to NB parameters.
        
        Args:
            z: Latent (B, latent_dim)
            library_size: Optional library size (B,) for normalization
        
        Returns:
            mean: NB mean (B, num_genes)
            dispersion: NB dispersion (B, num_genes)
        """
        # Decode
        h = self.decoder(z)
        
        # NB mean (must be positive)
        mean = torch.exp(self.fc_mean(h))
        
        # Scale by library size if provided
        if library_size is not None:
            mean = mean * library_size.unsqueeze(-1)
        
        # NB dispersion (must be positive)
        dispersion = torch.exp(self.fc_dispersion(h))
        
        return mean, dispersion
    
    def log_prob(self, x, mean, dispersion, eps=1e-8):
        """
        Negative Binomial log probability.
        
        Args:
            x: Observed counts (B, num_genes)
            mean: NB mean (B, num_genes)
            dispersion: NB dispersion (B, num_genes)
        
        Returns:
            log_prob: Log probability (B,)
        """
        # NB log probability
        # p(x | mean, dispersion) = Gamma(x + 1/dispersion) / (Gamma(x+1) * Gamma(1/dispersion))
        #                            * (dispersion * mean)^x / (1 + dispersion * mean)^(x + 1/dispersion)
        
        theta = 1.0 / (dispersion + eps)  # inverse dispersion
        
        # Log probability
        log_theta_mu = torch.log(theta + mean + eps)
        
        log_prob = (
            torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * torch.log(theta + eps) - theta * log_theta_mu
            + x * torch.log(mean + eps) - x * log_theta_mu
        )
        
        return log_prob.sum(dim=-1)  # Sum over genes
```

**Zero-Inflated Negative Binomial Decoder**:

```python
class ZINBDecoder(nn.Module):
    """
    Decoder with Zero-Inflated Negative Binomial distribution.
    
    Models excess zeros (dropout) in addition to overdispersion.
    
    Args:
        latent_dim: Latent space dimension
        num_genes: Number of genes
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(
        self,
        latent_dim=256,
        num_genes=20000,
        hidden_dims=[512, 1024, 2048],
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_genes = num_genes
        
        # Decoder layers
        layers = []
        in_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # ZINB parameters
        self.fc_mean = nn.Linear(hidden_dims[-1], num_genes)
        self.fc_dispersion = nn.Linear(hidden_dims[-1], num_genes)
        self.fc_dropout = nn.Linear(hidden_dims[-1], num_genes)  # Zero-inflation
    
    def forward(self, z, library_size=None):
        """
        Decode latent to ZINB parameters.
        
        Args:
            z: Latent (B, latent_dim)
            library_size: Optional library size (B,)
        
        Returns:
            mean: NB mean (B, num_genes)
            dispersion: NB dispersion (B, num_genes)
            dropout: Dropout probability (B, num_genes)
        """
        # Decode
        h = self.decoder(z)
        
        # NB mean
        mean = torch.exp(self.fc_mean(h))
        if library_size is not None:
            mean = mean * library_size.unsqueeze(-1)
        
        # NB dispersion
        dispersion = torch.exp(self.fc_dispersion(h))
        
        # Dropout probability (zero-inflation)
        dropout = torch.sigmoid(self.fc_dropout(h))
        
        return mean, dispersion, dropout
    
    def log_prob(self, x, mean, dispersion, dropout, eps=1e-8):
        """
        ZINB log probability.
        
        Args:
            x: Observed counts (B, num_genes)
            mean: NB mean (B, num_genes)
            dispersion: NB dispersion (B, num_genes)
            dropout: Dropout probability (B, num_genes)
        
        Returns:
            log_prob: Log probability (B,)
        """
        # ZINB = mixture of point mass at 0 and NB
        # p(x) = dropout * I(x=0) + (1-dropout) * NB(x | mean, dispersion)
        
        theta = 1.0 / (dispersion + eps)
        log_theta_mu = torch.log(theta + mean + eps)
        
        # NB log prob
        nb_log_prob = (
            torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * torch.log(theta + eps) - theta * log_theta_mu
            + x * torch.log(mean + eps) - x * log_theta_mu
        )
        
        # ZINB log prob
        zero_mask = (x < eps).float()
        
        # For zeros: log(dropout + (1-dropout) * NB(0))
        nb_zero = theta * (torch.log(theta + eps) - log_theta_mu)
        log_prob_zero = torch.log(dropout + (1 - dropout) * torch.exp(nb_zero) + eps)
        
        # For non-zeros: log((1-dropout) * NB(x))
        log_prob_nonzero = torch.log(1 - dropout + eps) + nb_log_prob
        
        # Combine
        log_prob = zero_mask * log_prob_zero + (1 - zero_mask) * log_prob_nonzero
        
        return log_prob.sum(dim=-1)  # Sum over genes
```

### 1.4 Complete VAE with NB/ZINB

```python
class GeneExpressionVAE(nn.Module):
    """
    Complete VAE for gene expression with NB or ZINB decoder.
    
    Args:
        num_genes: Number of genes
        latent_dim: Latent dimension
        decoder_type: 'nb' or 'zinb'
    """
    def __init__(
        self,
        num_genes=20000,
        latent_dim=256,
        decoder_type='zinb',
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.decoder_type = decoder_type
        
        # Encoder
        self.encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            latent_dim=latent_dim,
        )
        
        # Decoder
        if decoder_type == 'nb':
            self.decoder = NegativeBinomialDecoder(
                latent_dim=latent_dim,
                num_genes=num_genes,
            )
        elif decoder_type == 'zinb':
            self.decoder = ZINBDecoder(
                latent_dim=latent_dim,
                num_genes=num_genes,
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def forward(self, x, library_size=None):
        """
        Forward pass.
        
        Args:
            x: Gene expression (B, num_genes)
            library_size: Optional library size (B,)
        
        Returns:
            recon_params: Reconstruction parameters
            mu: Latent mean
            logvar: Latent log variance
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)
        
        # Decode
        recon_params = self.decoder(z, library_size)
        
        return recon_params, mu, logvar
    
    def loss(self, x, recon_params, mu, logvar, library_size=None, beta=1.0):
        """
        VAE loss (ELBO).
        
        Args:
            x: Gene expression (B, num_genes)
            recon_params: Reconstruction parameters from decoder
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
            library_size: Optional library size (B,)
            beta: KL weight (beta-VAE)
        
        Returns:
            loss: Total loss
            loss_dict: Dictionary with loss components
        """
        # Reconstruction loss (negative log likelihood)
        if self.decoder_type == 'nb':
            mean, dispersion = recon_params
            recon_loss = -self.decoder.log_prob(x, mean, dispersion).mean()
        elif self.decoder_type == 'zinb':
            mean, dispersion, dropout = recon_params
            recon_loss = -self.decoder.log_prob(x, mean, dispersion, dropout).mean()
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        loss_dict = {
            'loss': loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
        }
        
        return loss, loss_dict
    
    @torch.no_grad()
    def encode(self, x):
        """Encode to latent space (deterministic)."""
        mu, logvar = self.encoder(x)
        return mu
    
    @torch.no_grad()
    def decode(self, z, library_size=None):
        """Decode from latent space."""
        recon_params = self.decoder(z, library_size)
        
        if self.decoder_type == 'nb':
            mean, dispersion = recon_params
            return mean
        elif self.decoder_type == 'zinb':
            mean, dispersion, dropout = recon_params
            # Expected value: (1 - dropout) * mean
            return (1 - dropout) * mean
```

---

## 2. Latent Diffusion Model

### 2.1 Diffusion in Latent Space

**Key idea**: Run diffusion on latent codes instead of raw data.

```python
class LatentDiffusionModel(nn.Module):
    """
    Diffusion model operating in latent space.
    
    Args:
        latent_dim: Latent space dimension
        model_type: 'dit' or 'unet'
        num_steps: Number of diffusion steps
    """
    def __init__(
        self,
        latent_dim=256,
        model_type='dit',
        num_steps=1000,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        
        # Denoising model
        if model_type == 'dit':
            self.model = DiTLatent(latent_dim=latent_dim)
        elif model_type == 'unet':
            self.model = UNetLatent(latent_dim=latent_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Noise schedule (linear for simplicity)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, num_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def forward(self, z, t, condition=None):
        """
        Predict noise at timestep t.
        
        Args:
            z: Noisy latent (B, latent_dim)
            t: Timestep (B,)
            condition: Optional conditioning (B, cond_dim)
        
        Returns:
            noise_pred: Predicted noise (B, latent_dim)
        """
        return self.model(z, t, condition)
    
    def add_noise(self, z0, t):
        """
        Add noise to clean latent.
        
        Args:
            z0: Clean latent (B, latent_dim)
            t: Timestep (B,)
        
        Returns:
            zt: Noisy latent (B, latent_dim)
            noise: Added noise (B, latent_dim)
        """
        noise = torch.randn_like(z0)
        
        # Get alpha_t
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        
        # zt = sqrt(alpha_t) * z0 + sqrt(1 - alpha_t) * noise
        zt = torch.sqrt(alpha_t) * z0 + torch.sqrt(1 - alpha_t) * noise
        
        return zt, noise
    
    @torch.no_grad()
    def sample(self, batch_size, condition=None, num_steps=50):
        """
        Sample from diffusion model (DDIM sampling).
        
        Args:
            batch_size: Number of samples
            condition: Optional conditioning
            num_steps: Number of sampling steps
        
        Returns:
            z0: Sampled latent (batch_size, latent_dim)
        """
        device = next(self.parameters()).device
        
        # Start from noise
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Sampling timesteps
        timesteps = torch.linspace(self.num_steps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(z, t_batch, condition)
            
            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            # Predicted x0
            pred_z0 = (z - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction pointing to zt
            dir_zt = torch.sqrt(1 - alpha_t_prev) * noise_pred
            
            # Update
            z = torch.sqrt(alpha_t_prev) * pred_z0 + dir_zt
        
        return z
```

### 2.2 DiT for Latent Space

```python
class DiTLatent(nn.Module):
    """
    DiT (Diffusion Transformer) for latent space.
    
    Args:
        latent_dim: Latent dimension
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize to zero (residual connection)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, z, t, condition=None):
        """
        Args:
            z: Noisy latent (B, latent_dim)
            t: Timestep (B,)
            condition: Optional conditioning (B, cond_dim)
        
        Returns:
            noise_pred: Predicted noise (B, latent_dim)
        """
        # Input projection
        h = self.input_proj(z)  # (B, hidden_dim)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        
        # Add condition if provided
        if condition is not None:
            # Project condition to hidden_dim
            cond_emb = self.condition_proj(condition)
            t_emb = t_emb + cond_emb
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output projection
        noise_pred = self.output_proj(h)
        
        return noise_pred


class DiTBlock(nn.Module):
    """DiT block with AdaLN."""
    def __init__(self, dim, num_heads):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
    
    def forward(self, x, t_emb):
        """
        Args:
            x: Input (B, dim)
            t_emb: Time embedding (B, dim)
        
        Returns:
            out: Output (B, dim)
        """
        # AdaLN parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        
        # Self-attention with AdaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, x_norm, x_norm)[0]
        
        # MLP with AdaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings
```

---

## 3. Conditioning Mechanisms

### 3.1 Concatenation Conditioning

**Simplest approach**: Concatenate condition to latent.

```python
class ConcatenationConditioning(nn.Module):
    """
    Conditioning via concatenation.
    
    Args:
        latent_dim: Latent dimension
        condition_dim: Condition dimension
    """
    def __init__(self, latent_dim=256, condition_dim=128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Project concatenated input
        self.proj = nn.Linear(latent_dim + condition_dim, latent_dim)
    
    def forward(self, z, condition):
        """
        Args:
            z: Latent (B, latent_dim)
            condition: Condition (B, condition_dim)
        
        Returns:
            z_cond: Conditioned latent (B, latent_dim)
        """
        # Concatenate
        z_cat = torch.cat([z, condition], dim=-1)
        
        # Project back to latent_dim
        z_cond = self.proj(z_cat)
        
        return z_cond
```

### 3.2 Cross-Attention Conditioning

**More flexible**: Condition attends to latent.

```python
class CrossAttentionConditioning(nn.Module):
    """
    Conditioning via cross-attention.
    
    Args:
        latent_dim: Latent dimension
        condition_dim: Condition dimension
        num_heads: Number of attention heads
    """
    def __init__(self, latent_dim=256, condition_dim=128, num_heads=8):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Project condition to latent_dim
        self.condition_proj = nn.Linear(condition_dim, latent_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, z, condition):
        """
        Args:
            z: Latent (B, latent_dim) or (B, num_tokens, latent_dim)
            condition: Condition (B, condition_dim) or (B, num_cond, condition_dim)
        
        Returns:
            z_cond: Conditioned latent
        """
        # Ensure z has sequence dimension
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, latent_dim)
        
        # Project condition
        cond = self.condition_proj(condition)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, latent_dim)
        
        # Cross-attention: z attends to condition
        z_norm = self.norm(z)
        z_attn = self.cross_attn(z_norm, cond, cond)[0]
        
        # Residual
        z_cond = z + z_attn
        
        return z_cond.squeeze(1) if z_cond.shape[1] == 1 else z_cond
```

### 3.3 FiLM Conditioning

**Affine transformation**: Scale and shift based on condition.

```python
class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) conditioning.
    
    Args:
        latent_dim: Latent dimension
        condition_dim: Condition dimension
    """
    def __init__(self, latent_dim=256, condition_dim=128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Predict scale and shift from condition
        self.film = nn.Sequential(
            nn.Linear(condition_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )
    
    def forward(self, z, condition):
        """
        Args:
            z: Latent (B, latent_dim)
            condition: Condition (B, condition_dim)
        
        Returns:
            z_cond: Conditioned latent (B, latent_dim)
        """
        # Predict scale and shift
        film_params = self.film(condition)
        scale, shift = film_params.chunk(2, dim=-1)
        
        # Apply FiLM
        z_cond = scale * z + shift
        
        return z_cond
```

### 3.4 Classifier-Free Guidance

**Training**: Randomly drop condition (unconditional training).

```python
def classifier_free_guidance_training(model, z, t, condition, dropout_prob=0.1):
    """
    Training with classifier-free guidance.
    
    Args:
        model: Diffusion model
        z: Noisy latent (B, latent_dim)
        t: Timestep (B,)
        condition: Condition (B, condition_dim)
        dropout_prob: Probability of dropping condition
    
    Returns:
        noise_pred: Predicted noise
    """
    # Randomly drop condition
    mask = torch.rand(condition.shape[0], device=condition.device) > dropout_prob
    condition_masked = condition * mask.unsqueeze(-1)
    
    # Predict noise
    noise_pred = model(z, t, condition_masked)
    
    return noise_pred


@torch.no_grad()
def classifier_free_guidance_sampling(model, batch_size, condition, guidance_scale=7.5):
    """
    Sampling with classifier-free guidance.
    
    Args:
        model: Diffusion model
        batch_size: Number of samples
        condition: Condition (batch_size, condition_dim)
        guidance_scale: Guidance strength
    
    Returns:
        z0: Sampled latent
    """
    device = next(model.parameters()).device
    
    # Start from noise
    z = torch.randn(batch_size, model.latent_dim, device=device)
    
    # Unconditional (null) condition
    condition_null = torch.zeros_like(condition)
    
    for t in reversed(range(model.num_steps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Conditional prediction
        noise_cond = model(z, t_batch, condition)
        
        # Unconditional prediction
        noise_uncond = model(z, t_batch, condition_null)
        
        # Classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        # Update (simplified DDPM step)
        alpha_t = model.alphas_cumprod[t]
        alpha_t_prev = model.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        
        pred_z0 = (z - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        z = torch.sqrt(alpha_t_prev) * pred_z0 + torch.sqrt(1 - alpha_t_prev) * noise_pred
    
    return z
```

---

## 4. Complete Latent Diffusion System

### 4.1 Full Pipeline

```python
class CompleteLaten​tDiffusion(nn.Module):
    """
    Complete latent diffusion system for gene expression.
    
    Combines VAE and latent diffusion model.
    
    Args:
        num_genes: Number of genes
        latent_dim: Latent dimension
        decoder_type: 'nb' or 'zinb'
    """
    def __init__(
        self,
        num_genes=20000,
        latent_dim=256,
        decoder_type='zinb',
    ):
        super().__init__()
        
        # VAE
        self.vae = GeneExpressionVAE(
            num_genes=num_genes,
            latent_dim=latent_dim,
            decoder_type=decoder_type,
        )
        
        # Latent diffusion
        self.diffusion = LatentDiffusionModel(
            latent_dim=latent_dim,
            model_type='dit',
        )
    
    def train_vae(self, x, library_size=None, beta=1.0):
        """Train VAE."""
        recon_params, mu, logvar = self.vae(x, library_size)
        loss, loss_dict = self.vae.loss(x, recon_params, mu, logvar, library_size, beta)
        return loss, loss_dict
    
    def train_diffusion(self, x, condition=None):
        """Train diffusion on latent codes."""
        # Encode to latent (frozen VAE)
        with torch.no_grad():
            z0 = self.vae.encode(x)
        
        # Sample timestep
        t = torch.randint(0, self.diffusion.num_steps, (z0.shape[0],), device=z0.device)
        
        # Add noise
        zt, noise = self.diffusion.add_noise(z0, t)
        
        # Predict noise
        noise_pred = self.diffusion(zt, t, condition)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def generate(self, batch_size, condition=None, library_size=None):
        """
        Generate gene expression samples.
        
        Args:
            batch_size: Number of samples
            condition: Optional conditioning
            library_size: Optional library size
        
        Returns:
            x_gen: Generated gene expression (batch_size, num_genes)
        """
        # Sample latent from diffusion
        z0 = self.diffusion.sample(batch_size, condition)
        
        # Decode to gene expression
        x_gen = self.vae.decode(z0, library_size)
        
        return x_gen
```

---

## Key Takeaways

### Architecture

1. **VAE with NB/ZINB decoder** — Appropriate for count data
2. **Latent diffusion** — Diffuse in compressed space (256 dims)
3. **DiT backbone** — Transformer-based denoising
4. **Flexible conditioning** — Concatenation, cross-attention, FiLM, CFG

### Design Choices

1. **ZINB decoder** — Models overdispersion + excess zeros
2. **DiT over U-Net** — Better for latent space (no spatial structure)
3. **Classifier-free guidance** — Better controllability
4. **Two-stage training** — VAE first, then diffusion

### For Biology

1. **78× compression** — 20K genes → 256 latent dims
2. **10-100× speedup** — Faster training and sampling
3. **Better quality** — Sharper than VAE, stable than GAN
4. **Interpretable latent** — Can analyze latent space

---

## Related Documents

- [00_latent_diffusion_overview.md](00_latent_diffusion_overview.md) — High-level concepts
- [02_latent_diffusion_training.md](02_latent_diffusion_training.md) — Training strategies
- [03_latent_diffusion_applications.md](03_latent_diffusion_applications.md) — Applications
- [04_latent_diffusion_combio.md](04_latent_diffusion_combio.md) — Complete implementation

---

## References

**Latent Diffusion**:
- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
- Vahdat et al. (2021): "Score-based Generative Modeling in Latent Space"

**VAE for Biology**:
- Lopez et al. (2018): "Deep generative modeling for single-cell transcriptomics" (scVI)
- Eraslan et al. (2019): "Single-cell RNA-seq denoising using a deep count autoencoder"

**NB/ZINB Distributions**:
- Hilbe (2011): "Negative Binomial Regression"
- Risso et al. (2018): "A general and flexible method for signal extraction from single-cell RNA-seq data" (ZINB-WaVE)

**DiT**:
- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers"
