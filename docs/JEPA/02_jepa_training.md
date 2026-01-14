# JEPA Training: Strategies and Best Practices

This document covers training strategies for JEPA models, including optimization, hyperparameters, monitoring, debugging, and advanced techniques.

**Prerequisites**: Understanding of [JEPA architecture](01_jepa_foundations.md) and [overview](00_jepa_overview.md).

---

## Training Overview

### Basic Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_jepa(
    model,
    train_loader,
    optimizer,
    num_epochs=100,
    device='cuda',
):
    """
    Basic JEPA training loop.
    
    Args:
        model: JEPA model
        train_loader: Training data loader
        optimizer: Optimizer
        num_epochs: Number of epochs
        device: Device to train on
    """
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_inv = 0.0
        epoch_var = 0.0
        epoch_cov = 0.0
        
        for batch_idx, (x_context, x_target) in enumerate(train_loader):
            x_context = x_context.to(device)
            x_target = x_target.to(device)
            
            # Forward pass
            loss, loss_dict = model(x_context, x_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss_dict['loss']
            epoch_inv += loss_dict['inv']
            epoch_var += loss_dict['var']
            epoch_cov += loss_dict['cov']
        
        # Average losses
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        epoch_inv /= num_batches
        epoch_var /= num_batches
        epoch_cov /= num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Inv: {epoch_inv:.4f}, Var: {epoch_var:.4f}, Cov: {epoch_cov:.4f}")
```

---

## 1. Data Preparation

### 1.1 Image Data (I-JEPA)

```python
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class IJEPADataset(Dataset):
    """
    Dataset for I-JEPA training.
    
    Returns context and target views of the same image.
    """
    def __init__(
        self,
        images,
        img_size=224,
        patch_size=16,
        num_blocks=4,
    ):
        self.images = images
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and transform image
        img = self.images[idx]
        img = self.transform(img)
        
        # Context: full image
        x_context = img
        
        # Target: same image (masking handled in model)
        x_target = img
        
        return x_context, x_target
```

### 1.2 Gene Expression Data (Bio-JEPA)

```python
class PerturbSeqDataset(Dataset):
    """
    Dataset for Perturb-seq JEPA training.
    
    Returns baseline and perturbed expression pairs.
    """
    def __init__(
        self,
        baseline_expr,
        perturbed_expr,
        perturbation_info,
        perturbation_encoder,
    ):
        """
        Args:
            baseline_expr: Baseline expression (N, num_genes)
            perturbed_expr: Perturbed expression (N, num_genes)
            perturbation_info: Perturbation metadata (N, ...)
            perturbation_encoder: Function to encode perturbation info
        """
        self.baseline_expr = baseline_expr
        self.perturbed_expr = perturbed_expr
        self.perturbation_info = perturbation_info
        self.perturbation_encoder = perturbation_encoder
    
    def __len__(self):
        return len(self.baseline_expr)
    
    def __getitem__(self, idx):
        # Baseline (context)
        x_baseline = torch.tensor(self.baseline_expr[idx], dtype=torch.float32)
        
        # Perturbed (target)
        x_perturbed = torch.tensor(self.perturbed_expr[idx], dtype=torch.float32)
        
        # Perturbation embedding
        pert_info = self.perturbation_info[idx]
        pert_emb = self.perturbation_encoder(pert_info)
        
        return x_baseline, x_perturbed, pert_emb


def encode_perturbation(pert_info, gene_vocab_size=20000):
    """
    Encode perturbation information.
    
    Args:
        pert_info: Dictionary with perturbation details
            - 'gene_id': ID of perturbed gene
            - 'type': 'knockout', 'overexpression', etc.
            - 'dose': Perturbation strength
    
    Returns:
        pert_emb: Perturbation embedding
    """
    # One-hot encode gene
    gene_onehot = torch.zeros(gene_vocab_size)
    gene_onehot[pert_info['gene_id']] = 1.0
    
    # Encode type
    type_map = {'knockout': 0, 'overexpression': 1, 'knockdown': 2}
    type_id = type_map.get(pert_info['type'], 0)
    type_onehot = torch.zeros(len(type_map))
    type_onehot[type_id] = 1.0
    
    # Dose
    dose = torch.tensor([pert_info.get('dose', 1.0)])
    
    # Concatenate
    pert_emb = torch.cat([gene_onehot, type_onehot, dose])
    
    return pert_emb
```

### 1.3 Time-Series Data (V-JEPA)

```python
class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series JEPA training.
    
    Returns past frames (context) and future frames (target).
    """
    def __init__(
        self,
        time_series,
        context_length=4,
        target_length=4,
    ):
        """
        Args:
            time_series: Time-series data (N, T, ...)
            context_length: Number of past frames
            target_length: Number of future frames
        """
        self.time_series = time_series
        self.context_length = context_length
        self.target_length = target_length
        
        # Valid starting indices
        self.valid_indices = list(range(
            0,
            len(time_series) - context_length - target_length + 1
        ))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Context: past frames
        x_context = self.time_series[start_idx:start_idx + self.context_length]
        
        # Target: future frames
        x_target = self.time_series[
            start_idx + self.context_length:
            start_idx + self.context_length + self.target_length
        ]
        
        return torch.tensor(x_context), torch.tensor(x_target)
```

---

## 2. Optimization

### 2.1 Optimizer Choice

**AdamW** (recommended):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.05,
)
```

**Why AdamW**:
- Decoupled weight decay (better than Adam)
- Stable for transformers
- Good default choice

**Alternatives**:
- **Adam**: If weight decay not needed
- **SGD + momentum**: For smaller models
- **LAMB**: For very large batch sizes

### 2.2 Learning Rate Schedule

**Warmup + Cosine Decay** (recommended):

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr=1e-6,
):
    """
    Cosine learning rate schedule with warmup.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr: Minimum learning rate
    
    Returns:
        scheduler: Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Usage
num_epochs = 100
steps_per_epoch = len(train_loader)
num_training_steps = num_epochs * steps_per_epoch
num_warmup_steps = 10 * steps_per_epoch  # 10 epochs warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
```

**Schedule visualization**:
```
LR
 │
 │    ╱─────╲
 │   ╱       ╲
 │  ╱         ╲___
 │ ╱               ╲___
 │╱                    ╲___
 └────────────────────────── Steps
   Warmup    Cosine Decay
```

### 2.3 Gradient Clipping

**Prevent exploding gradients**:

```python
# In training loop
loss.backward()

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

**When to use**:
- Always for transformers
- Especially with large learning rates
- If seeing NaN losses

---

## 3. Hyperparameters

### 3.1 Core Hyperparameters

| Parameter | Image (I-JEPA) | Gene Expression (Bio-JEPA) | Notes |
|-----------|----------------|----------------------------|-------|
| **Learning rate** | 1e-4 | 1e-3 to 1e-4 | Higher for smaller models |
| **Batch size** | 256-1024 | 64-256 | Larger is better (up to memory) |
| **Warmup epochs** | 10 | 5-10 | ~10% of total epochs |
| **Weight decay** | 0.05 | 0.01-0.05 | Regularization |
| **Embed dim** | 768 | 256-512 | Model capacity |
| **Encoder depth** | 12 | 4-8 | Deeper = more capacity |
| **Predictor depth** | 6 | 2-4 | 0.5× encoder depth |

### 3.2 VICReg Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| **λ_inv** | 25.0 | 10-50 | Invariance weight |
| **λ_var** | 25.0 | 10-50 | Variance weight |
| **λ_cov** | 1.0 | 0.5-2.0 | Covariance weight |
| **γ (target std)** | 1.0 | 0.5-2.0 | Target variance |

**Tuning guidelines**:
- If embeddings collapse → increase λ_var
- If dimensions correlated → increase λ_cov
- If predictions too rigid → decrease λ_inv

### 3.3 Masking Hyperparameters (I-JEPA)

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| **Num blocks** | 4 | 2-8 | Number of masked blocks |
| **Block scale** | 0.15-0.2 | 0.1-0.3 | Fraction of image |
| **Aspect ratio** | 0.75-1.5 | 0.5-2.0 | Block shape |

---

## 4. Training Loop with All Features

### 4.1 Complete Training Script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import math

def train_jepa_complete(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=1e-4,
    weight_decay=0.05,
    warmup_epochs=10,
    device='cuda',
    save_dir='checkpoints',
    log_dir='logs',
):
    """
    Complete JEPA training with all features.
    
    Args:
        model: JEPA model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        device: Device
        save_dir: Checkpoint directory
        log_dir: Log directory
    """
    # Setup
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    
    # Scheduler
    steps_per_epoch = len(train_loader)
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Logging
    writer = SummaryWriter(log_dir)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, writer, global_step
        )
        global_step += steps_per_epoch
        
        # Validation
        model.eval()
        val_metrics = validate(model, val_loader, device)
        
        # Logging
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"    Inv: {train_metrics['inv']:.4f}, "
              f"Var: {train_metrics['var']:.4f}, "
              f"Cov: {train_metrics['cov']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"    Inv: {val_metrics['inv']:.4f}, "
              f"Var: {val_metrics['var']:.4f}, "
              f"Cov: {val_metrics['cov']:.4f}")
        
        # TensorBoard
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        writer.add_scalar('val/inv', val_metrics['inv'], epoch)
        writer.add_scalar('val/var', val_metrics['var'], epoch)
        writer.add_scalar('val/cov', val_metrics['cov'], epoch)
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    writer.close()
    print("\nTraining complete!")


def train_one_epoch(model, train_loader, optimizer, scheduler, device, writer, global_step):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_inv = 0.0
    total_var = 0.0
    total_cov = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch
        if len(batch) == 2:
            x_context, x_target = batch
            x_context = x_context.to(device)
            x_target = x_target.to(device)
            loss, loss_dict = model(x_context, x_target)
        elif len(batch) == 3:
            x_context, x_target, condition = batch
            x_context = x_context.to(device)
            x_target = x_target.to(device)
            condition = condition.to(device)
            loss, loss_dict = model(x_context, x_target, condition)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Accumulate
        total_loss += loss_dict['loss']
        total_inv += loss_dict['inv']
        total_var += loss_dict['var']
        total_cov += loss_dict['cov']
        
        # Log to TensorBoard
        if batch_idx % 100 == 0:
            step = global_step + batch_idx
            writer.add_scalar('train/loss', loss_dict['loss'], step)
            writer.add_scalar('train/inv', loss_dict['inv'], step)
            writer.add_scalar('train/var', loss_dict['var'], step)
            writer.add_scalar('train/cov', loss_dict['cov'], step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], step)
    
    # Average
    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'inv': total_inv / num_batches,
        'var': total_var / num_batches,
        'cov': total_cov / num_batches,
    }


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_inv = 0.0
    total_var = 0.0
    total_cov = 0.0
    
    for batch in val_loader:
        # Unpack batch
        if len(batch) == 2:
            x_context, x_target = batch
            x_context = x_context.to(device)
            x_target = x_target.to(device)
            loss, loss_dict = model(x_context, x_target)
        elif len(batch) == 3:
            x_context, x_target, condition = batch
            x_context = x_context.to(device)
            x_target = x_target.to(device)
            condition = condition.to(device)
            loss, loss_dict = model(x_context, x_target, condition)
        
        total_loss += loss_dict['loss']
        total_inv += loss_dict['inv']
        total_var += loss_dict['var']
        total_cov += loss_dict['cov']
    
    num_batches = len(val_loader)
    return {
        'loss': total_loss / num_batches,
        'inv': total_inv / num_batches,
        'var': total_var / num_batches,
        'cov': total_cov / num_batches,
    }
```

---

## 5. Monitoring and Debugging

### 5.1 Key Metrics to Monitor

**1. Loss components**:
- Total loss
- Invariance loss (MSE between prediction and target)
- Variance loss (embedding spread)
- Covariance loss (dimension decorrelation)

**2. Embedding statistics**:
- Mean embedding norm
- Embedding variance per dimension
- Correlation between dimensions

**3. Training dynamics**:
- Learning rate
- Gradient norms
- Weight norms

### 5.2 Monitoring Script

```python
@torch.no_grad()
def compute_embedding_stats(model, data_loader, device):
    """
    Compute embedding statistics for monitoring.
    
    Args:
        model: JEPA model
        data_loader: Data loader
        device: Device
    
    Returns:
        stats: Dictionary of statistics
    """
    model.eval()
    
    all_embeddings = []
    
    for batch in data_loader:
        x = batch[0].to(device)
        z = model.encode(x)
        
        # Flatten tokens
        z = z.reshape(-1, z.shape[-1])  # (B*num_tokens, embed_dim)
        all_embeddings.append(z.cpu())
    
    # Concatenate
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, embed_dim)
    
    # Compute statistics
    stats = {
        'mean_norm': all_embeddings.norm(dim=1).mean().item(),
        'std_norm': all_embeddings.norm(dim=1).std().item(),
        'mean_per_dim': all_embeddings.mean(dim=0),
        'std_per_dim': all_embeddings.std(dim=0),
        'correlation': torch.corrcoef(all_embeddings.T),
    }
    
    return stats


# Usage in training loop
if epoch % 5 == 0:
    stats = compute_embedding_stats(model, val_loader, device)
    print(f"  Embedding norm: {stats['mean_norm']:.4f} ± {stats['std_norm']:.4f}")
    print(f"  Avg std per dim: {stats['std_per_dim'].mean():.4f}")
    
    # Check for collapse
    if stats['std_per_dim'].min() < 0.1:
        print("  WARNING: Some dimensions have collapsed!")
```

### 5.3 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Embedding collapse** | All embeddings → same vector | Increase λ_var, check batch size |
| **Dimension collapse** | Some dims have zero variance | Increase λ_var, λ_cov |
| **High correlation** | Dimensions are correlated | Increase λ_cov |
| **NaN loss** | Loss becomes NaN | Reduce LR, add gradient clipping |
| **No learning** | Loss doesn't decrease | Check LR, check data |
| **Overfitting** | Train loss << val loss | Increase weight decay, add dropout |

---

## 6. Advanced Techniques

### 6.1 Exponential Moving Average (EMA)

**Maintain EMA of model weights**:

```python
class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# Usage
ema = EMA(model, decay=0.999)

# In training loop
for batch in train_loader:
    loss, _ = model(x_context, x_target)
    loss.backward()
    optimizer.step()
    
    # Update EMA
    ema.update()

# For validation/inference, use EMA weights
ema.apply_shadow()
val_loss = validate(model, val_loader, device)
ema.restore()
```

### 6.2 Mixed Precision Training

**Use FP16 for faster training**:

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop
for batch in train_loader:
    x_context, x_target = batch
    x_context = x_context.to(device)
    x_target = x_target.to(device)
    
    # Forward with autocast
    with autocast():
        loss, loss_dict = model(x_context, x_target)
    
    # Backward with scaler
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2-3× faster training
- 2× less memory
- Minimal accuracy loss

### 6.3 Distributed Training

**Train on multiple GPUs**:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    """Setup distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_distributed(model, train_dataset, val_dataset, ...):
    """Train with DDP."""
    # Setup
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Wrap model
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Distributed sampler
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        for batch in train_loader:
            # Training step
            ...
    
    # Cleanup
    dist.destroy_process_group()
```

**Launch**:
```bash
torchrun --nproc_per_node=4 train_jepa.py
```

### 6.4 Curriculum Learning

**Start with easier tasks, gradually increase difficulty**:

```python
def get_curriculum_schedule(epoch, total_epochs):
    """
    Curriculum schedule for masking difficulty.
    
    Start with small masks, gradually increase.
    """
    progress = epoch / total_epochs
    
    # Increase mask size over time
    min_scale = 0.05
    max_scale = 0.25
    scale = min_scale + progress * (max_scale - min_scale)
    
    # Increase number of blocks
    min_blocks = 1
    max_blocks = 8
    num_blocks = int(min_blocks + progress * (max_blocks - min_blocks))
    
    return {
        'block_scale': (scale, scale + 0.05),
        'num_blocks': num_blocks,
    }


# Usage
for epoch in range(num_epochs):
    curriculum = get_curriculum_schedule(epoch, num_epochs)
    
    # Update dataset masking parameters
    train_dataset.set_masking_params(
        num_blocks=curriculum['num_blocks'],
        block_scale=curriculum['block_scale'],
    )
    
    # Train
    train_one_epoch(...)
```

---

## 7. Evaluation

### 7.1 Downstream Task Evaluation

**Evaluate learned representations on downstream tasks**:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@torch.no_grad()
def evaluate_downstream(model, train_data, train_labels, test_data, test_labels, device):
    """
    Evaluate on downstream classification task.
    
    Args:
        model: JEPA model
        train_data: Training data
        train_labels: Training labels
        test_data: Test data
        test_labels: Test labels
        device: Device
    
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    
    # Extract embeddings
    train_embeddings = []
    for x in train_data:
        x = x.to(device)
        z = model.encode(x)
        z = z.mean(dim=1)  # Average over tokens
        train_embeddings.append(z.cpu())
    train_embeddings = torch.cat(train_embeddings, dim=0).numpy()
    
    test_embeddings = []
    for x in test_data:
        x = x.to(device)
        z = model.encode(x)
        z = z.mean(dim=1)
        test_embeddings.append(z.cpu())
    test_embeddings = torch.cat(test_embeddings, dim=0).numpy()
    
    # Train linear classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embeddings, train_labels)
    
    # Evaluate
    pred_labels = clf.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, pred_labels)
    
    return accuracy


# Usage
accuracy = evaluate_downstream(
    model,
    train_data, train_labels,
    test_data, test_labels,
    device
)
print(f"Downstream accuracy: {accuracy:.4f}")
```

### 7.2 Embedding Quality Metrics

```python
@torch.no_grad()
def compute_embedding_quality(model, data_loader, device):
    """
    Compute embedding quality metrics.
    
    Returns:
        metrics: Dictionary of quality metrics
    """
    model.eval()
    
    all_embeddings = []
    for batch in data_loader:
        x = batch[0].to(device)
        z = model.encode(x)
        z = z.reshape(-1, z.shape[-1])
        all_embeddings.append(z.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 1. Effective rank (measure of dimensionality usage)
    U, S, V = torch.svd(all_embeddings)
    S_normalized = S / S.sum()
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()
    
    # 2. Uniformity (how uniformly distributed on hypersphere)
    normalized = F.normalize(all_embeddings, dim=1)
    similarity = normalized @ normalized.T
    uniformity = torch.log(torch.exp(similarity).mean()).item()
    
    # 3. Alignment (for paired data)
    # ... (depends on task)
    
    metrics = {
        'effective_rank': effective_rank,
        'uniformity': uniformity,
        'max_dim': all_embeddings.shape[1],
    }
    
    return metrics
```

---

## Key Takeaways

### Training Strategy

1. **Optimizer**: AdamW with weight decay
2. **Schedule**: Warmup + cosine decay
3. **Batch size**: As large as possible (256-1024)
4. **Gradient clipping**: Always use (max_norm=1.0)

### Hyperparameters

1. **Learning rate**: 1e-4 for images, 1e-3 for gene expression
2. **VICReg weights**: λ_inv=25, λ_var=25, λ_cov=1
3. **Warmup**: ~10% of total epochs
4. **Weight decay**: 0.05 for images, 0.01-0.05 for gene expression

### Monitoring

1. **Watch for collapse**: Check embedding variance
2. **Monitor all loss components**: inv, var, cov
3. **Evaluate downstream**: Linear probing on tasks
4. **Use EMA**: For more stable evaluation

### Advanced

1. **EMA**: Decay=0.999 for smoother weights
2. **Mixed precision**: 2-3× speedup
3. **Distributed**: Scale to multiple GPUs
4. **Curriculum**: Start easy, increase difficulty

---

## Related Documents

- [01_jepa_foundations.md](01_jepa_foundations.md) — Architecture details
- [03_jepa_applications.md](03_jepa_applications.md) — Applications
- [04_jepa_perturbseq.md](04_jepa_perturbseq.md) — Perturb-seq implementation

---

## References

- Assran et al. (2023): "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
- Bardes et al. (2022): "VICReg: Variance-Invariance-Covariance Regularization"
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization" (AdamW)
