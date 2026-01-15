# Latent Diffusion Applications: Computational Biology

This document covers applications of latent diffusion models in computational biology, including single-cell generation, perturbation prediction, multi-omics translation, trajectory modeling, and spatial transcriptomics.

**Prerequisites**: Understanding of [latent diffusion foundations](01_latent_diffusion_foundations.md) and [training](02_latent_diffusion_training.md).

---

## Overview

Latent diffusion models are particularly well-suited for computational biology because:

1. **Efficiency** — 10-100× faster than pixel-space diffusion
2. **Quality** — Better than VAE, stable than GAN
3. **Flexibility** — Multi-modal, multi-task, controllable
4. **Interpretability** — Latent space has biological meaning

---

## 1. Single-Cell Generation

### 1.1 Task Definition

**Goal**: Generate realistic single-cell gene expression profiles.

**Applications**:

- Data augmentation for rare cell types
- Synthetic controls for experiments
- Batch effect removal
- Counterfactual cell states

### 1.2 Architecture

```python
class SingleCellLatentDiffusion(nn.Module):
    """
    Latent diffusion for single-cell generation.
    
    Args:
        num_genes: Number of genes
        latent_dim: Latent dimension
        num_cell_types: Number of cell types (for conditioning)
    """
    def __init__(
        self,
        num_genes=5000,
        latent_dim=256,
        num_cell_types=20,
    ):
        super().__init__()
        
        # VAE with ZINB decoder
        self.vae = GeneExpressionVAE(
            num_genes=num_genes,
            latent_dim=latent_dim,
            decoder_type='zinb',
        )
        
        # Latent diffusion with cell type conditioning
        self.diffusion = LatentDiffusionModel(
            latent_dim=latent_dim,
            model_type='dit',
        )
        
        # Cell type embedding
        self.cell_type_embed = nn.Embedding(num_cell_types, 128)
    
    def forward(self, x, cell_type, library_size=None):
        """
        Training forward pass.
        
        Args:
            x: Gene expression (B, num_genes)
            cell_type: Cell type indices (B,)
            library_size: Library size (B,)
        
        Returns:
            vae_loss, diffusion_loss
        """
        # Train VAE
        recon_params, mu, logvar = self.vae(x, library_size)
        vae_loss, _ = self.vae.loss(x, recon_params, mu, logvar, library_size)
        
        # Train diffusion on latent
        with torch.no_grad():
            z0 = mu
        
        # Cell type conditioning
        cell_type_emb = self.cell_type_embed(cell_type)
        
        # Diffusion loss
        t = torch.randint(0, self.diffusion.num_steps, (z0.shape[0],), device=z0.device)
        zt, noise = self.diffusion.add_noise(z0, t)
        noise_pred = self.diffusion(zt, t, cell_type_emb)
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        return vae_loss, diffusion_loss
    
    @torch.no_grad()
    def generate(self, cell_type, num_samples=100, library_size=None):
        """
        Generate single-cell profiles.
        
        Args:
            cell_type: Cell type index (scalar or (num_samples,))
            num_samples: Number of samples
            library_size: Optional library size
        
        Returns:
            x_gen: Generated expression (num_samples, num_genes)
        """
        device = next(self.parameters()).device
        
        # Cell type conditioning
        if isinstance(cell_type, int):
            cell_type = torch.full((num_samples,), cell_type, device=device)
        cell_type_emb = self.cell_type_embed(cell_type)
        
        # Sample latent from diffusion
        z0 = self.diffusion.sample(num_samples, cell_type_emb)
        
        # Decode to gene expression
        x_gen = self.vae.decode(z0, library_size)
        
        return x_gen
```

### 1.3 Training Strategy

```python
def train_single_cell_generation(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
):
    """
    Train single-cell generation model.
    
    Args:
        model: SingleCellLatentDiffusion
        train_loader: Training data (x, cell_type, library_size)
        val_loader: Validation data
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device
    """
    model.to(device)
    
    # Separate optimizers
    optimizer_vae = torch.optim.AdamW(model.vae.parameters(), lr=lr)
    optimizer_diffusion = torch.optim.AdamW(
        list(model.diffusion.parameters()) + list(model.cell_type_embed.parameters()),
        lr=lr
    )
    
    for epoch in range(num_epochs):
        model.train()
        
        for x, cell_type, library_size in train_loader:
            x = x.to(device)
            cell_type = cell_type.to(device)
            library_size = library_size.to(device)
            
            # Forward
            vae_loss, diffusion_loss = model(x, cell_type, library_size)
            
            # Update VAE
            optimizer_vae.zero_grad()
            vae_loss.backward(retain_graph=True)
            optimizer_vae.step()
            
            # Update diffusion
            optimizer_diffusion.zero_grad()
            diffusion_loss.backward()
            optimizer_diffusion.step()
        
        print(f"Epoch {epoch+1}: VAE={vae_loss:.4f}, Diff={diffusion_loss:.4f}")
```

### 1.4 Evaluation

```python
@torch.no_grad()
def evaluate_single_cell_generation(model, test_data, cell_types, device='cuda'):
    """
    Evaluate single-cell generation quality.
    
    Args:
        model: Trained model
        test_data: Real test data (N, num_genes)
        cell_types: Cell type labels (N,)
        device: Device
    
    Returns:
        metrics: Evaluation metrics
    """
    import scanpy as sc
    from scipy.stats import wasserstein_distance
    
    model.eval()
    model.to(device)
    
    # Generate samples for each cell type
    unique_cell_types = torch.unique(cell_types)
    
    metrics = {}
    
    for ct in unique_cell_types:
        # Real data for this cell type
        mask = (cell_types == ct)
        real_data = test_data[mask]
        
        # Generate synthetic data
        num_samples = len(real_data)
        synthetic_data = model.generate(ct.item(), num_samples)
        
        # 1. Mean expression correlation
        real_mean = real_data.mean(dim=0)
        synth_mean = synthetic_data.mean(dim=0)
        corr = torch.corrcoef(torch.stack([real_mean, synth_mean]))[0, 1]
        
        # 2. Wasserstein distance (per gene, average)
        w_dists = []
        for gene_idx in range(real_data.shape[1]):
            w_dist = wasserstein_distance(
                real_data[:, gene_idx].cpu().numpy(),
                synthetic_data[:, gene_idx].cpu().numpy()
            )
            w_dists.append(w_dist)
        
        # 3. Sparsity similarity
        real_sparsity = (real_data < 0.1).float().mean()
        synth_sparsity = (synthetic_data < 0.1).float().mean()
        
        metrics[f'cell_type_{ct.item()}'] = {
            'mean_correlation': corr.item(),
            'wasserstein_distance': np.mean(w_dists),
            'real_sparsity': real_sparsity.item(),
            'synth_sparsity': synth_sparsity.item(),
        }
    
    return metrics
```

---

## 2. Perturbation Prediction

### 2.1 Task Definition

**Goal**: Predict cellular response to genetic/chemical perturbations.

**Applications**:

- Virtual screening (predict without experiment)
- Combination prediction (multiple perturbations)
- Mechanism discovery (analyze latent changes)

### 2.2 Architecture

```python
class PerturbationLatentDiffusion(nn.Module):
    """
    Latent diffusion for perturbation prediction.
    
    Predicts perturbed state from baseline + perturbation.
    
    Args:
        num_genes: Number of genes
        latent_dim: Latent dimension
        perturbation_dim: Perturbation embedding dimension
    """
    def __init__(
        self,
        num_genes=5000,
        latent_dim=256,
        perturbation_dim=128,
    ):
        super().__init__()
        
        # VAE
        self.vae = GeneExpressionVAE(
            num_genes=num_genes,
            latent_dim=latent_dim,
            decoder_type='zinb',
        )
        
        # Perturbation encoder
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, perturbation_dim),
        )
        
        # Latent diffusion conditioned on baseline + perturbation
        self.diffusion = LatentDiffusionModel(
            latent_dim=latent_dim,
            model_type='dit',
        )
        
        # Conditioning projection
        self.condition_proj = nn.Linear(latent_dim + perturbation_dim, latent_dim)
    
    def forward(self, x_baseline, x_perturbed, perturbation_indicator):
        """
        Training forward pass.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            x_perturbed: Perturbed expression (B, num_genes)
            perturbation_indicator: One-hot perturbation (B, num_genes)
        
        Returns:
            vae_loss, diffusion_loss
        """
        # Encode baseline and perturbed
        z_baseline = self.vae.encode(x_baseline)
        z_perturbed = self.vae.encode(x_perturbed)
        
        # Encode perturbation
        pert_emb = self.perturbation_encoder(perturbation_indicator)
        
        # Condition: baseline latent + perturbation
        condition = torch.cat([z_baseline, pert_emb], dim=-1)
        condition = self.condition_proj(condition)
        
        # Diffusion: predict perturbed latent from baseline + perturbation
        t = torch.randint(0, self.diffusion.num_steps, (z_perturbed.shape[0],), device=z_perturbed.device)
        zt, noise = self.diffusion.add_noise(z_perturbed, t)
        noise_pred = self.diffusion(zt, t, condition)
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        return diffusion_loss
    
    @torch.no_grad()
    def predict_perturbation(self, x_baseline, perturbation_indicator, library_size=None):
        """
        Predict perturbed state.
        
        Args:
            x_baseline: Baseline expression (B, num_genes)
            perturbation_indicator: Perturbation (B, num_genes)
            library_size: Library size (B,)
        
        Returns:
            x_perturbed_pred: Predicted perturbed expression (B, num_genes)
        """
        # Encode baseline
        z_baseline = self.vae.encode(x_baseline)
        
        # Encode perturbation
        pert_emb = self.perturbation_encoder(perturbation_indicator)
        
        # Condition
        condition = torch.cat([z_baseline, pert_emb], dim=-1)
        condition = self.condition_proj(condition)
        
        # Sample perturbed latent
        z_perturbed = self.diffusion.sample(len(x_baseline), condition)
        
        # Decode
        x_perturbed_pred = self.vae.decode(z_perturbed, library_size)
        
        return x_perturbed_pred
```

### 2.3 Delta-in-Latent Formulation

**Alternative approach**: Predict the change in latent space.

```python
class DeltaLatentPerturbation(nn.Module):
    """
    Predict delta in latent space.
    
    More stable than predicting absolute perturbed state.
    """
    def __init__(
        self,
        num_genes=5000,
        latent_dim=256,
        perturbation_dim=128,
    ):
        super().__init__()
        
        self.vae = GeneExpressionVAE(num_genes, latent_dim, 'zinb')
        self.perturbation_encoder = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.GELU(),
            nn.Linear(512, perturbation_dim),
        )
        
        # Predict delta
        self.delta_predictor = nn.Sequential(
            nn.Linear(latent_dim + perturbation_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )
    
    def forward(self, x_baseline, x_perturbed, perturbation_indicator):
        """
        Training: predict delta in latent space.
        """
        # Encode
        z_baseline = self.vae.encode(x_baseline)
        z_perturbed = self.vae.encode(x_perturbed)
        
        # True delta
        delta_true = z_perturbed - z_baseline
        
        # Encode perturbation
        pert_emb = self.perturbation_encoder(perturbation_indicator)
        
        # Predict delta
        delta_pred = self.delta_predictor(torch.cat([z_baseline, pert_emb], dim=-1))
        
        # MSE loss on delta
        loss = F.mse_loss(delta_pred, delta_true)
        
        return loss
    
    @torch.no_grad()
    def predict_perturbation(self, x_baseline, perturbation_indicator, library_size=None):
        """
        Predict perturbed state via delta.
        """
        # Encode baseline
        z_baseline = self.vae.encode(x_baseline)
        
        # Encode perturbation
        pert_emb = self.perturbation_encoder(perturbation_indicator)
        
        # Predict delta
        delta_pred = self.delta_predictor(torch.cat([z_baseline, pert_emb], dim=-1))
        
        # Add delta to baseline
        z_perturbed = z_baseline + delta_pred
        
        # Decode
        x_perturbed_pred = self.vae.decode(z_perturbed, library_size)
        
        return x_perturbed_pred
```

### 2.4 Evaluation

```python
@torch.no_grad()
def evaluate_perturbation_prediction(
    model,
    test_data,
    perturbations,
    device='cuda',
):
    """
    Evaluate perturbation prediction.
    
    Args:
        model: Trained model
        test_data: (baseline, perturbed, perturbation_indicator) tuples
        perturbations: List of perturbation names
        device: Device
    
    Returns:
        metrics: Evaluation metrics
    """
    from scipy.stats import pearsonr
    
    model.eval()
    model.to(device)
    
    all_correlations = []
    all_mse = []
    
    for x_baseline, x_perturbed_true, pert_indicator in test_data:
        x_baseline = x_baseline.to(device)
        x_perturbed_true = x_perturbed_true.to(device)
        pert_indicator = pert_indicator.to(device)
        
        # Predict
        x_perturbed_pred = model.predict_perturbation(x_baseline, pert_indicator)
        
        # Correlation (per sample)
        for i in range(len(x_baseline)):
            corr, _ = pearsonr(
                x_perturbed_true[i].cpu().numpy(),
                x_perturbed_pred[i].cpu().numpy()
            )
            all_correlations.append(corr)
        
        # MSE
        mse = F.mse_loss(x_perturbed_pred, x_perturbed_true)
        all_mse.append(mse.item())
    
    metrics = {
        'mean_correlation': np.mean(all_correlations),
        'median_correlation': np.median(all_correlations),
        'mean_mse': np.mean(all_mse),
    }
    
    return metrics
```

---

## 3. Multi-Omics Translation

### 3.1 Task Definition

**Goal**: Predict one modality from another (e.g., protein from RNA).

**Applications**:

- Fill missing modalities
- Cross-modality validation
- Integrated multi-omics analysis

### 3.2 Architecture

```python
class MultiOmicsLatentDiffusion(nn.Module):
    """
    Latent diffusion for multi-omics translation.
    
    Shared latent space for RNA and Protein.
    
    Args:
        num_genes_rna: Number of RNA genes
        num_genes_protein: Number of protein genes
        latent_dim: Shared latent dimension
    """
    def __init__(
        self,
        num_genes_rna=5000,
        num_genes_protein=100,
        latent_dim=256,
    ):
        super().__init__()
        
        # RNA encoder
        self.rna_encoder = GeneExpressionEncoder(
            num_genes=num_genes_rna,
            latent_dim=latent_dim,
        )
        
        # Protein encoder
        self.protein_encoder = GeneExpressionEncoder(
            num_genes=num_genes_protein,
            latent_dim=latent_dim,
        )
        
        # RNA decoder (ZINB)
        self.rna_decoder = ZINBDecoder(
            latent_dim=latent_dim,
            num_genes=num_genes_rna,
        )
        
        # Protein decoder (NB, less sparse)
        self.protein_decoder = NegativeBinomialDecoder(
            latent_dim=latent_dim,
            num_genes=num_genes_protein,
        )
        
        # Latent diffusion
        self.diffusion = LatentDiffusionModel(
            latent_dim=latent_dim,
            model_type='dit',
        )
    
    def forward(self, x_rna, x_protein):
        """
        Training: align RNA and Protein in latent space.
        
        Args:
            x_rna: RNA expression (B, num_genes_rna)
            x_protein: Protein expression (B, num_genes_protein)
        
        Returns:
            loss_dict: Dictionary of losses
        """
        # Encode both modalities
        mu_rna, logvar_rna = self.rna_encoder(x_rna)
        mu_protein, logvar_protein = self.protein_encoder(x_protein)
        
        z_rna = self.rna_encoder.reparameterize(mu_rna, logvar_rna)
        z_protein = self.protein_encoder.reparameterize(mu_protein, logvar_protein)
        
        # Reconstruction losses
        rna_recon = self.rna_decoder(z_rna)
        protein_recon = self.protein_decoder(z_protein)
        
        loss_rna_recon = -self.rna_decoder.log_prob(x_rna, *rna_recon).mean()
        loss_protein_recon = -self.protein_decoder.log_prob(x_protein, *protein_recon).mean()
        
        # KL losses
        kl_rna = -0.5 * torch.sum(1 + logvar_rna - mu_rna.pow(2) - logvar_rna.exp(), dim=-1).mean()
        kl_protein = -0.5 * torch.sum(1 + logvar_protein - mu_protein.pow(2) - logvar_protein.exp(), dim=-1).mean()
        
        # Alignment loss (latents should be similar)
        loss_align = F.mse_loss(z_rna, z_protein)
        
        # Total loss
        loss = loss_rna_recon + loss_protein_recon + kl_rna + kl_protein + 10.0 * loss_align
        
        return {
            'loss': loss,
            'rna_recon': loss_rna_recon.item(),
            'protein_recon': loss_protein_recon.item(),
            'kl_rna': kl_rna.item(),
            'kl_protein': kl_protein.item(),
            'align': loss_align.item(),
        }
    
    @torch.no_grad()
    def translate_rna_to_protein(self, x_rna):
        """
        Translate RNA to Protein.
        
        Args:
            x_rna: RNA expression (B, num_genes_rna)
        
        Returns:
            x_protein_pred: Predicted protein (B, num_genes_protein)
        """
        # Encode RNA to latent
        mu_rna, _ = self.rna_encoder(x_rna)
        
        # Decode to protein
        protein_params = self.protein_decoder(mu_rna)
        x_protein_pred = protein_params[0]  # Mean
        
        return x_protein_pred
    
    @torch.no_grad()
    def translate_protein_to_rna(self, x_protein):
        """
        Translate Protein to RNA.
        
        Args:
            x_protein: Protein expression (B, num_genes_protein)
        
        Returns:
            x_rna_pred: Predicted RNA (B, num_genes_rna)
        """
        # Encode protein to latent
        mu_protein, _ = self.protein_encoder(x_protein)
        
        # Decode to RNA
        rna_params = self.rna_decoder(mu_protein)
        mean, _, dropout = rna_params
        x_rna_pred = (1 - dropout) * mean
        
        return x_rna_pred
```

---

## 4. Trajectory Modeling

### 4.1 Task Definition

**Goal**: Model developmental or disease trajectories over time.

**Applications**:

- Predict future cell states
- Identify branch points
- Model differentiation

### 4.2 Architecture

```python
class TrajectoryLatentDiffusion(nn.Module):
    """
    Latent diffusion for trajectory modeling.
    
    Predicts future states conditioned on time.
    
    Args:
        num_genes: Number of genes
        latent_dim: Latent dimension
    """
    def __init__(
        self,
        num_genes=5000,
        latent_dim=256,
    ):
        super().__init__()
        
        self.vae = GeneExpressionVAE(num_genes, latent_dim, 'zinb')
        
        # Time encoder
        self.time_encoder = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        
        # Latent diffusion conditioned on current state + time
        self.diffusion = LatentDiffusionModel(latent_dim, 'dit')
    
    @torch.no_grad()
    def predict_trajectory(self, x_start, time_points, library_size=None):
        """
        Predict trajectory from starting state.
        
        Args:
            x_start: Starting expression (B, num_genes)
            time_points: Time points to predict (T,)
            library_size: Library size (B,)
        
        Returns:
            trajectory: Predicted trajectory (B, T, num_genes)
        """
        device = x_start.device
        batch_size = len(x_start)
        
        # Encode starting state
        z_current = self.vae.encode(x_start)
        
        trajectory = []
        
        for t in time_points:
            # Time embedding
            t_tensor = torch.full((batch_size,), t, device=device)
            t_emb = self.time_encoder(t_tensor)
            
            # Condition on current state + time
            condition = torch.cat([z_current, t_emb], dim=-1)
            
            # Sample next state
            z_next = self.diffusion.sample(batch_size, condition)
            
            # Decode
            x_next = self.vae.decode(z_next, library_size)
            trajectory.append(x_next)
            
            # Update current state
            z_current = z_next
        
        trajectory = torch.stack(trajectory, dim=1)  # (B, T, num_genes)
        
        return trajectory
```

---

## 5. Spatial Transcriptomics

### 5.1 Task Definition

**Goal**: Generate spatial gene expression patterns.

**Applications**:

- Super-resolution (increase spatial resolution)
- Missing region imputation
- 3D reconstruction

### 5.2 Architecture

```python
class SpatialLatentDiffusion(nn.Module):
    """
    Latent diffusion for spatial transcriptomics.
    
    Conditioned on spatial coordinates.
    
    Args:
        num_genes: Number of genes
        latent_dim: Latent dimension
        spatial_dim: Spatial coordinate dimension (2 or 3)
    """
    def __init__(
        self,
        num_genes=5000,
        latent_dim=256,
        spatial_dim=2,
    ):
        super().__init__()
        
        self.vae = GeneExpressionVAE(num_genes, latent_dim, 'zinb')
        
        # Spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        
        self.diffusion = LatentDiffusionModel(latent_dim, 'dit')
    
    @torch.no_grad()
    def generate_at_location(self, coordinates, library_size=None):
        """
        Generate expression at spatial locations.
        
        Args:
            coordinates: Spatial coordinates (B, spatial_dim)
            library_size: Library size (B,)
        
        Returns:
            x_gen: Generated expression (B, num_genes)
        """
        # Encode spatial coordinates
        spatial_emb = self.spatial_encoder(coordinates)
        
        # Sample latent conditioned on location
        z = self.diffusion.sample(len(coordinates), spatial_emb)
        
        # Decode
        x_gen = self.vae.decode(z, library_size)
        
        return x_gen
    
    @torch.no_grad()
    def super_resolution(self, x_low_res, coords_low_res, coords_high_res):
        """
        Super-resolution: predict high-res from low-res.
        
        Args:
            x_low_res: Low-resolution expression (N_low, num_genes)
            coords_low_res: Low-res coordinates (N_low, spatial_dim)
            coords_high_res: High-res coordinates (N_high, spatial_dim)
        
        Returns:
            x_high_res: High-resolution expression (N_high, num_genes)
        """
        # Encode low-res to latent
        z_low_res = self.vae.encode(x_low_res)
        
        # Interpolate latent to high-res locations
        # (Simple approach: nearest neighbor or Gaussian kernel)
        from scipy.spatial import cKDTree
        
        tree = cKDTree(coords_low_res.cpu().numpy())
        distances, indices = tree.query(coords_high_res.cpu().numpy(), k=3)
        
        # Weighted average based on distance
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        z_high_res = torch.zeros(len(coords_high_res), z_low_res.shape[1], device=z_low_res.device)
        for i in range(len(coords_high_res)):
            for j, idx in enumerate(indices[i]):
                z_high_res[i] += weights[i, j] * z_low_res[idx]
        
        # Decode
        x_high_res = self.vae.decode(z_high_res)
        
        return x_high_res
```

---

## Comparison with Existing Methods

### Single-Cell Generation

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **scGAN** | GAN | Fast sampling | Mode collapse, unstable |
| **scVI** | VAE | Fast, interpretable | Blurry samples |
| **Latent Diffusion** | VAE + Diffusion | Sharp, diverse | Slower sampling |

### Perturbation Prediction

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **scGen** | VAE + arithmetic | Simple, fast | Linear assumption |
| **CPA** | Autoencoder + perturbation | Flexible | No uncertainty |
| **Latent Diffusion** | VAE + Diffusion | Uncertainty, flexible | More complex |

### Multi-Omics

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **MOFA** | Factor analysis | Interpretable | Linear |
| **totalVI** | Joint VAE | Unified framework | Fixed architecture |
| **Latent Diffusion** | Joint VAE + Diffusion | Flexible, high-quality | Training complexity |

---

## Key Takeaways

### Applications

1. **Single-cell generation** — Data augmentation, rare cell types
2. **Perturbation prediction** — Virtual screening, combinations
3. **Multi-omics translation** — Fill missing modalities
4. **Trajectory modeling** — Predict future states
5. **Spatial transcriptomics** — Super-resolution, imputation

### Advantages

1. **Efficiency** — 10-100× faster than pixel-space
2. **Quality** — Better than VAE, stable than GAN
3. **Flexibility** — Multi-modal, multi-task
4. **Uncertainty** — Probabilistic predictions

### Best Practices

1. **Start with VAE** — Get good latent space first
2. **Condition carefully** — Use appropriate conditioning mechanism
3. **Validate biologically** — Not just loss metrics
4. **Compare baselines** — scGen, CPA, totalVI

---

## Related Documents

- [00_latent_diffusion_overview.md](00_latent_diffusion_overview.md) — High-level concepts
- [01_latent_diffusion_foundations.md](01_latent_diffusion_foundations.md) — Architecture details
- [02_latent_diffusion_training.md](02_latent_diffusion_training.md) — Training strategies
- [04_latent_diffusion_combio.md](04_latent_diffusion_combio.md) — Complete implementation

---

## References

**Single-Cell Generation**:

- Marouf et al. (2020): "Realistic in silico generation and augmentation of single-cell RNA-seq data using generative adversarial networks" (scGAN)
- Lopez et al. (2018): "Deep generative modeling for single-cell transcriptomics" (scVI)

**Perturbation Prediction**:

- Lotfollahi et al. (2019): "scGen predicts single-cell perturbation responses"
- Lotfollahi et al. (2023): "Predicting cellular responses to novel drug combinations with a deep generative model" (CPA)

**Multi-Omics**:

- Argelaguet et al. (2018): "Multi-Omics Factor Analysis" (MOFA)
- Gayoso et al. (2021): "Joint probabilistic modeling of single-cell multi-omic data with totalVI"

**Spatial Transcriptomics**:

- Cable et al. (2022): "Robust decomposition of cell type mixtures in spatial transcriptomics"
- Biancalani et al. (2021): "Deep learning and alignment of spatially resolved single-cell transcriptomes"
