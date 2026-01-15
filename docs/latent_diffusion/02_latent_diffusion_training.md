# Latent Diffusion Training: Strategies and Best Practices

This document covers training strategies for latent diffusion models in computational biology, including two-stage training, hyperparameters, optimization, and debugging.

**Prerequisites**: Understanding of [latent diffusion foundations](01_latent_diffusion_foundations.md) and [overview](00_latent_diffusion_overview.md).

---

## Training Overview

### Two-Stage Training Pipeline

**Stage 1: Train VAE** (Learn latent space)
```python
# Train VAE on gene expression
vae = GeneExpressionVAE(num_genes=20000, latent_dim=256)
train_vae(vae, gene_expression_data, num_epochs=100)

# Freeze VAE
vae.eval()
for param in vae.parameters():
    param.requires_grad = False
```

**Stage 2: Train Diffusion** (Learn generation in latent space)
```python
# Encode data to latent
latents = encode_dataset(vae, gene_expression_data)

# Train diffusion on latent codes
diffusion = LatentDiffusionModel(latent_dim=256)
train_diffusion(diffusion, latents, num_epochs=100)
```

**Optional Stage 3: Joint Fine-Tuning**
```python
# Unfreeze VAE
for param in vae.parameters():
    param.requires_grad = True

# Fine-tune end-to-end
train_joint(vae, diffusion, gene_expression_data, num_epochs=20)
```

---

## 1. Stage 1: VAE Training

### 1.1 Data Preparation

```python
import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GeneExpressionDataset(Dataset):
    """
    Dataset for gene expression.
    
    Args:
        adata: AnnData object with gene expression
        transform: Optional transform (e.g., log1p)
    """
    def __init__(self, adata, transform=True):
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            self.X = adata.X.toarray()
        else:
            self.X = adata.X
        
        # Log transform
        if transform:
            self.X = np.log1p(self.X)
        
        # Library size (for NB/ZINB)
        self.library_size = self.X.sum(axis=1)
        
        # Convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.library_size = torch.tensor(self.library_size, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.library_size[idx]


def prepare_data(adata_path, batch_size=64):
    """
    Prepare data loaders.
    
    Args:
        adata_path: Path to AnnData file
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data
    adata = sc.read_h5ad(adata_path)
    
    # Basic preprocessing
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    adata = adata[:, adata.var['highly_variable']]
    
    # Split train/val/test
    n_cells = adata.n_obs
    n_train = int(0.7 * n_cells)
    n_val = int(0.15 * n_cells)
    
    indices = np.random.permutation(n_cells)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    # Create datasets
    train_dataset = GeneExpressionDataset(adata[train_idx])
    val_dataset = GeneExpressionDataset(adata[val_idx])
    test_dataset = GeneExpressionDataset(adata[test_idx])
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
```

### 1.2 VAE Training Loop

```python
def train_vae(
    vae,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-3,
    beta=1.0,
    device='cuda',
    save_dir='checkpoints/vae',
):
    """
    Train VAE on gene expression.
    
    Args:
        vae: VAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        beta: KL weight (beta-VAE)
        device: Device
        save_dir: Checkpoint directory
    """
    import os
    from torch.utils.tensorboard import SummaryWriter
    
    vae.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.01)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Logging
    writer = SummaryWriter(f'{save_dir}/logs')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        vae.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        
        for x, library_size in train_loader:
            x = x.to(device)
            library_size = library_size.to(device)
            
            # Forward
            recon_params, mu, logvar = vae(x, library_size)
            loss, loss_dict = vae.loss(x, recon_params, mu, logvar, library_size, beta)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate
            train_loss += loss_dict['loss']
            train_recon += loss_dict['recon']
            train_kl += loss_dict['kl']
        
        # Average
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)
        
        # Validation
        vae.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for x, library_size in val_loader:
                x = x.to(device)
                library_size = library_size.to(device)
                
                recon_params, mu, logvar = vae(x, library_size)
                loss, loss_dict = vae.loss(x, recon_params, mu, logvar, library_size, beta)
                
                val_loss += loss_dict['loss']
                val_recon += loss_dict['recon']
                val_kl += loss_dict['kl']
        
        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Recon={train_recon:.4f}, KL={train_kl:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Recon={val_recon:.4f}, KL={val_kl:.4f}")
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/recon', train_recon, epoch)
        writer.add_scalar('train/kl', train_kl, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/recon', val_recon, epoch)
        writer.add_scalar('val/kl', val_kl, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{save_dir}/best_model.pt')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        # Step scheduler
        scheduler.step()
    
    writer.close()
    print("\nVAE training complete!")
```

### 1.3 VAE Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Learning rate** | 1e-3 | Standard for Adam |
| **Batch size** | 64-256 | Larger is better (up to memory) |
| **Latent dim** | 256-512 | Balance compression and quality |
| **Hidden dims** | [2048, 1024, 512] | Gradual compression |
| **Beta (KL weight)** | 0.1-1.0 | Lower = better reconstruction |
| **Weight decay** | 0.01 | Regularization |
| **Epochs** | 100-200 | Until convergence |

**Beta-VAE tuning**:

- `beta < 1.0`: Better reconstruction, less disentanglement
- `beta = 1.0`: Standard VAE
- `beta > 1.0`: More disentanglement, worse reconstruction

For biology, typically use `beta = 0.5-1.0` to prioritize reconstruction quality.

---

## 2. Stage 2: Latent Diffusion Training

### 2.1 Encode Dataset to Latent Space

```python
@torch.no_grad()
def encode_dataset(vae, data_loader, device='cuda'):
    """
    Encode entire dataset to latent space.
    
    Args:
        vae: Trained VAE model
        data_loader: Data loader
        device: Device
    
    Returns:
        latents: Encoded latent codes (N, latent_dim)
        library_sizes: Library sizes (N,)
    """
    vae.eval()
    vae.to(device)
    
    all_latents = []
    all_library_sizes = []
    
    for x, library_size in data_loader:
        x = x.to(device)
        
        # Encode (deterministic)
        mu, logvar = vae.encoder(x)
        latents = mu  # Use mean, not sample
        
        all_latents.append(latents.cpu())
        all_library_sizes.append(library_size)
    
    latents = torch.cat(all_latents, dim=0)
    library_sizes = torch.cat(all_library_sizes, dim=0)
    
    return latents, library_sizes


class LatentDataset(Dataset):
    """Dataset of latent codes."""
    def __init__(self, latents, library_sizes=None, conditions=None):
        self.latents = latents
        self.library_sizes = library_sizes
        self.conditions = conditions
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        if self.conditions is not None:
            return self.latents[idx], self.conditions[idx]
        else:
            return self.latents[idx]
```

### 2.2 Diffusion Training Loop

```python
def train_diffusion(
    diffusion,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-4,
    device='cuda',
    save_dir='checkpoints/diffusion',
):
    """
    Train latent diffusion model.
    
    Args:
        diffusion: Latent diffusion model
        train_loader: Training latent loader
        val_loader: Validation latent loader
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device
        save_dir: Checkpoint directory
    """
    import os
    from torch.utils.tensorboard import SummaryWriter
    
    diffusion.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=0.01)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Logging
    writer = SummaryWriter(f'{save_dir}/logs')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        diffusion.train()
        train_loss = 0.0
        
        for batch in train_loader:
            if isinstance(batch, tuple):
                z0, condition = batch
                z0 = z0.to(device)
                condition = condition.to(device)
            else:
                z0 = batch.to(device)
                condition = None
            
            # Sample timestep
            t = torch.randint(0, diffusion.num_steps, (z0.shape[0],), device=device)
            
            # Add noise
            zt, noise = diffusion.add_noise(z0, t)
            
            # Predict noise
            noise_pred = diffusion(zt, t, condition)
            
            # MSE loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        diffusion.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, tuple):
                    z0, condition = batch
                    z0 = z0.to(device)
                    condition = condition.to(device)
                else:
                    z0 = batch.to(device)
                    condition = None
                
                t = torch.randint(0, diffusion.num_steps, (z0.shape[0],), device=device)
                zt, noise = diffusion.add_noise(z0, t)
                noise_pred = diffusion(zt, t, condition)
                loss = F.mse_loss(noise_pred, noise)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'{save_dir}/best_model.pt')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        scheduler.step()
    
    writer.close()
    print("\nDiffusion training complete!")
```

### 2.3 Diffusion Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Learning rate** | 1e-4 | Lower than VAE |
| **Batch size** | 128-512 | Can be larger (latent is small) |
| **Num steps** | 1000 | Diffusion timesteps |
| **Model depth** | 12 | DiT layers |
| **Hidden dim** | 512 | DiT hidden dimension |
| **Num heads** | 8 | Attention heads |
| **Weight decay** | 0.01 | Regularization |
| **Epochs** | 100-200 | Until convergence |

---

## 3. Stage 3: Joint Fine-Tuning (Optional)

### 3.1 When to Use Joint Fine-Tuning

**Use when**:

- VAE reconstruction is suboptimal
- Want end-to-end optimization
- Have sufficient data

**Don't use when**:

- VAE is already good
- Limited data (risk overfitting)
- Want modular components

### 3.2 Joint Training Loop

```python
def train_joint(
    vae,
    diffusion,
    train_loader,
    val_loader,
    num_epochs=20,
    lr_vae=1e-5,
    lr_diffusion=1e-4,
    device='cuda',
    save_dir='checkpoints/joint',
):
    """
    Joint fine-tuning of VAE and diffusion.
    
    Args:
        vae: VAE model
        diffusion: Diffusion model
        train_loader: Training data loader (gene expression)
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr_vae: Learning rate for VAE
        lr_diffusion: Learning rate for diffusion
        device: Device
        save_dir: Checkpoint directory
    """
    import os
    from torch.utils.tensorboard import SummaryWriter
    
    vae.to(device)
    diffusion.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    # Separate optimizers for VAE and diffusion
    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=lr_vae, weight_decay=0.01)
    optimizer_diffusion = torch.optim.AdamW(diffusion.parameters(), lr=lr_diffusion, weight_decay=0.01)
    
    # Schedulers
    scheduler_vae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vae, num_epochs)
    scheduler_diffusion = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_diffusion, num_epochs)
    
    # Logging
    writer = SummaryWriter(f'{save_dir}/logs')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        vae.train()
        diffusion.train()
        
        train_vae_loss = 0.0
        train_diff_loss = 0.0
        
        for x, library_size in train_loader:
            x = x.to(device)
            library_size = library_size.to(device)
            
            # 1. VAE forward
            recon_params, mu, logvar = vae(x, library_size)
            vae_loss, vae_loss_dict = vae.loss(x, recon_params, mu, logvar, library_size)
            
            # 2. Diffusion forward (on latent)
            z0 = vae.encoder.reparameterize(mu, logvar)
            t = torch.randint(0, diffusion.num_steps, (z0.shape[0],), device=device)
            zt, noise = diffusion.add_noise(z0, t)
            noise_pred = diffusion(zt, t)
            diff_loss = F.mse_loss(noise_pred, noise)
            
            # 3. Backward (alternate updates)
            # Update VAE
            optimizer_vae.zero_grad()
            vae_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer_vae.step()
            
            # Update diffusion
            optimizer_diffusion.zero_grad()
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer_diffusion.step()
            
            train_vae_loss += vae_loss.item()
            train_diff_loss += diff_loss.item()
        
        train_vae_loss /= len(train_loader)
        train_diff_loss /= len(train_loader)
        
        # Validation
        vae.eval()
        diffusion.eval()
        
        val_vae_loss = 0.0
        val_diff_loss = 0.0
        
        with torch.no_grad():
            for x, library_size in val_loader:
                x = x.to(device)
                library_size = library_size.to(device)
                
                recon_params, mu, logvar = vae(x, library_size)
                vae_loss, _ = vae.loss(x, recon_params, mu, logvar, library_size)
                
                z0 = mu
                t = torch.randint(0, diffusion.num_steps, (z0.shape[0],), device=device)
                zt, noise = diffusion.add_noise(z0, t)
                noise_pred = diffusion(zt, t)
                diff_loss = F.mse_loss(noise_pred, noise)
                
                val_vae_loss += vae_loss.item()
                val_diff_loss += diff_loss.item()
        
        val_vae_loss /= len(val_loader)
        val_diff_loss /= len(val_loader)
        val_total_loss = val_vae_loss + val_diff_loss
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: VAE={train_vae_loss:.4f}, Diff={train_diff_loss:.4f}")
        print(f"  Val:   VAE={val_vae_loss:.4f}, Diff={val_diff_loss:.4f}")
        
        writer.add_scalar('train/vae_loss', train_vae_loss, epoch)
        writer.add_scalar('train/diff_loss', train_diff_loss, epoch)
        writer.add_scalar('val/vae_loss', val_vae_loss, epoch)
        writer.add_scalar('val/diff_loss', val_diff_loss, epoch)
        
        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save({
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'diffusion_state_dict': diffusion.state_dict(),
                'val_loss': val_total_loss,
            }, f'{save_dir}/best_model.pt')
            print(f"  Saved best model (val_loss: {val_total_loss:.4f})")
        
        scheduler_vae.step()
        scheduler_diffusion.step()
    
    writer.close()
    print("\nJoint fine-tuning complete!")
```

---

## 4. Conditioning Training

### 4.1 Conditional Dataset

```python
class ConditionalGeneExpressionDataset(Dataset):
    """
    Dataset with conditioning information.
    
    Args:
        adata: AnnData with expression
        condition_key: Key in adata.obs for condition (e.g., 'cell_type')
        condition_encoder: Function to encode condition to embedding
    """
    def __init__(self, adata, condition_key='cell_type', condition_encoder=None):
        # Expression
        if hasattr(adata.X, 'toarray'):
            self.X = adata.X.toarray()
        else:
            self.X = adata.X
        
        self.X = torch.tensor(np.log1p(self.X), dtype=torch.float32)
        
        # Library size
        self.library_size = torch.tensor(self.X.sum(axis=1), dtype=torch.float32)
        
        # Condition
        self.conditions = adata.obs[condition_key].values
        
        # Encode conditions
        if condition_encoder is None:
            # Default: one-hot encoding
            unique_conditions = np.unique(self.conditions)
            self.condition_to_idx = {c: i for i, c in enumerate(unique_conditions)}
            self.num_conditions = len(unique_conditions)
            
            condition_indices = [self.condition_to_idx[c] for c in self.conditions]
            self.condition_embeddings = F.one_hot(
                torch.tensor(condition_indices),
                num_classes=self.num_conditions
            ).float()
        else:
            self.condition_embeddings = condition_encoder(self.conditions)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.library_size[idx], self.condition_embeddings[idx]
```

### 4.2 Classifier-Free Guidance Training

```python
def train_diffusion_with_cfg(
    diffusion,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-4,
    dropout_prob=0.1,
    device='cuda',
    save_dir='checkpoints/diffusion_cfg',
):
    """
    Train diffusion with classifier-free guidance.
    
    Args:
        diffusion: Latent diffusion model
        train_loader: Training loader (with conditions)
        val_loader: Validation loader
        num_epochs: Number of epochs
        lr: Learning rate
        dropout_prob: Probability of dropping condition
        device: Device
        save_dir: Checkpoint directory
    """
    import os
    from torch.utils.tensorboard import SummaryWriter
    
    diffusion.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    writer = SummaryWriter(f'{save_dir}/logs')
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        diffusion.train()
        train_loss = 0.0
        
        for z0, condition in train_loader:
            z0 = z0.to(device)
            condition = condition.to(device)
            
            # Randomly drop condition (for classifier-free guidance)
            mask = torch.rand(condition.shape[0], device=device) > dropout_prob
            condition_masked = condition * mask.unsqueeze(-1)
            
            # Sample timestep
            t = torch.randint(0, diffusion.num_steps, (z0.shape[0],), device=device)
            
            # Add noise
            zt, noise = diffusion.add_noise(z0, t)
            
            # Predict noise
            noise_pred = diffusion(zt, t, condition_masked)
            
            # MSE loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation (without dropout)
        diffusion.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for z0, condition in val_loader:
                z0 = z0.to(device)
                condition = condition.to(device)
                
                t = torch.randint(0, diffusion.num_steps, (z0.shape[0],), device=device)
                zt, noise = diffusion.add_noise(z0, t)
                noise_pred = diffusion(zt, t, condition)
                loss = F.mse_loss(noise_pred, noise)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'val_loss': val_loss,
            }, f'{save_dir}/best_model.pt')
        
        scheduler.step()
    
    writer.close()
```

---

## 5. Monitoring and Debugging

### 5.1 Key Metrics

**VAE metrics**:

- Reconstruction loss (NLL)
- KL divergence
- Total ELBO
- Reconstruction quality (correlation with original)

**Diffusion metrics**:

- MSE loss (noise prediction)
- Sample quality (FID, IS for images; biological metrics for gene expression)
- Sampling diversity

### 5.2 Monitoring Script

```python
@torch.no_grad()
def evaluate_vae(vae, test_loader, device='cuda'):
    """
    Evaluate VAE reconstruction quality.
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    vae.eval()
    vae.to(device)
    
    all_x = []
    all_x_recon = []
    total_loss = 0.0
    
    for x, library_size in test_loader:
        x = x.to(device)
        library_size = library_size.to(device)
        
        # Reconstruct
        recon_params, mu, logvar = vae(x, library_size)
        loss, _ = vae.loss(x, recon_params, mu, logvar, library_size)
        
        # Get reconstruction
        if vae.decoder_type == 'nb':
            mean, _ = recon_params
            x_recon = mean
        elif vae.decoder_type == 'zinb':
            mean, _, dropout = recon_params
            x_recon = (1 - dropout) * mean
        
        all_x.append(x.cpu())
        all_x_recon.append(x_recon.cpu())
        total_loss += loss.item()
    
    all_x = torch.cat(all_x, dim=0)
    all_x_recon = torch.cat(all_x_recon, dim=0)
    
    # Compute correlation
    from scipy.stats import pearsonr
    
    correlations = []
    for i in range(len(all_x)):
        corr, _ = pearsonr(all_x[i].numpy(), all_x_recon[i].numpy())
        correlations.append(corr)
    
    metrics = {
        'loss': total_loss / len(test_loader),
        'mean_correlation': np.mean(correlations),
        'median_correlation': np.median(correlations),
    }
    
    return metrics


@torch.no_grad()
def evaluate_diffusion_samples(vae, diffusion, num_samples=1000, device='cuda'):
    """
    Evaluate diffusion sample quality.
    
    Returns:
        samples: Generated samples
        metrics: Quality metrics
    """
    vae.eval()
    diffusion.eval()
    vae.to(device)
    diffusion.to(device)
    
    # Sample from diffusion
    z_samples = diffusion.sample(num_samples)
    
    # Decode to gene expression
    x_samples = vae.decode(z_samples)
    
    # Compute metrics
    metrics = {
        'mean_expression': x_samples.mean().item(),
        'std_expression': x_samples.std().item(),
        'sparsity': (x_samples < 0.1).float().mean().item(),
    }
    
    return x_samples.cpu(), metrics
```

### 5.3 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **VAE posterior collapse** | KL → 0, poor reconstruction | Decrease beta, add KL warmup |
| **VAE blurry reconstruction** | Low correlation | Increase model capacity, decrease beta |
| **Diffusion mode collapse** | Low diversity | Check data diversity, increase model capacity |
| **Diffusion poor quality** | Noisy samples | Train longer, increase num_steps |
| **NaN loss** | Loss becomes NaN | Reduce LR, add gradient clipping |
| **Slow convergence** | Loss plateaus early | Increase LR, check data preprocessing |

---

## Key Takeaways

### Training Strategy

1. **Two-stage training** — VAE first, then diffusion
2. **Freeze VAE** — During diffusion training
3. **Optional joint fine-tuning** — End-to-end optimization
4. **Classifier-free guidance** — For better controllability

### Hyperparameters

1. **VAE**: lr=1e-3, beta=0.5-1.0, latent_dim=256-512
2. **Diffusion**: lr=1e-4, num_steps=1000, hidden_dim=512
3. **Joint**: lr_vae=1e-5 (lower), lr_diffusion=1e-4
4. **CFG**: dropout_prob=0.1, guidance_scale=7.5

### Monitoring

1. **VAE**: Reconstruction loss, KL, correlation
2. **Diffusion**: MSE loss, sample quality, diversity
3. **Biological**: Expression distribution, sparsity, pathway activity

### Best Practices

1. **Start simple** — Train VAE well before diffusion
2. **Monitor closely** — Check reconstruction quality
3. **Use CFG** — Better conditional generation
4. **Validate biologically** — Not just loss metrics

---

## Related Documents

- [00_latent_diffusion_overview.md](00_latent_diffusion_overview.md) — High-level concepts
- [01_latent_diffusion_foundations.md](01_latent_diffusion_foundations.md) — Architecture details
- [03_latent_diffusion_applications.md](03_latent_diffusion_applications.md) — Applications
- [04_latent_diffusion_combio.md](04_latent_diffusion_combio.md) — Complete implementation

---

## References

**Latent Diffusion**:

- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models"
- Ho & Salimans (2022): "Classifier-Free Diffusion Guidance"

**VAE Training**:

- Higgins et al. (2017): "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- Bowman et al. (2016): "Generating Sentences from a Continuous Space" (KL annealing)

**Biology Applications**:

- Lopez et al. (2018): "Deep generative modeling for single-cell transcriptomics" (scVI)
- Lotfollahi et al. (2020): "scGen predicts single-cell perturbation responses"
