# Open Research: Joint Latent Spaces for Biology

This document explores the concept of joint latent spaces for computational biology, drawing insights from the Goku model and discussing how static and dynamic biological data can share the same latent manifold.

**Key insight**: If two data types differ only by dimensionality or observation density, they probably want the same latent space.

---

## The Goku Insight

### The Problem with Separate Models

**Historical accident in computer vision**:
- Images and videos treated as fundamentally different
- Separate models, separate latent spaces
- No knowledge transfer

**Mathematical reality**:
$$
\text{Video} \in \mathbb{R}^{T \times H \times W \times C}
$$

An image is just $T = 1$.

**The split was artificial** — driven by engineering constraints, not fundamental differences.

### The Solution: One Latent Space

**Goku's approach** (ByteDance & HKU, 2024):
- **Single encoder-decoder** for both images and videos
- **Same latent manifold** for static and dynamic data
- **Mutual training** — images teach spatial priors, videos teach dynamics

**Why this works**:
1. **Shared structure** — Both have spatial organization
2. **Complementary information** — Static (appearance) + dynamic (motion)
3. **Prior sharing** — Learn better representations together

---

## Biology Parallel

### Static vs Dynamic Data

**Static data** (like images):
- **Bulk RNA-seq** — Population-level expression
- **Baseline scRNA-seq** — Single-cell snapshots
- **Spatial transcriptomics** — Tissue organization
- **Proteomics** — Protein abundance

**Dynamic data** (like videos):
- **Time-series** — Developmental trajectories
- **Perturb-seq** — Perturbation responses
- **Lineage tracing** — Cell fate decisions
- **Drug response** — Temporal treatment effects

**Key observation**: These differ in observation density and temporal resolution, not fundamental biology.

### The Biological Manifold

**Hypothesis**: All cellular states live on the same biological manifold.

```
Bulk RNA-seq ──┐
               ├──> Shared Encoder ──> Joint Latent Space ──> Shared Decoder
Time-series ───┤
               │
Perturb-seq ───┘
```

**Benefits**:
1. **Static data teaches spatial priors** — Cell types, pathways, gene modules
2. **Dynamic data teaches temporal dynamics** — Transitions, trajectories, causality
3. **Both inform the same representation** — Better than either alone

---

## Joint VAE Architecture

### Concept

**Traditional approach**:
```python
# Separate models
bulk_vae = VAE(input_dim=20000, latent_dim=256)
timeseries_vae = VAE(input_dim=20000, latent_dim=256)

# Different latent spaces
z_bulk = bulk_vae.encode(x_bulk)
z_time = timeseries_vae.encode(x_time)
# z_bulk and z_time are incompatible
```

**Joint approach**:
```python
# Single model
joint_vae = JointVAE(input_dim=20000, latent_dim=256)

# Same latent space
z_bulk = joint_vae.encode(x_bulk)
z_time = joint_vae.encode(x_time)
# z_bulk and z_time are comparable
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointBioVAE(nn.Module):
    """
    Joint VAE for static and dynamic biological data.
    
    Encodes both bulk RNA-seq and time-series into same latent space.
    """
    def __init__(
        self,
        num_genes=20000,
        latent_dim=256,
        hidden_dims=[2048, 1024, 512],
    ):
        super().__init__()
        
        # Shared encoder
        encoder_layers = []
        in_dim = num_genes
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        
        # Latent projection
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Shared decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(in_dim, num_genes))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode to latent space.
        
        Works for both static and dynamic data.
        
        Args:
            x: Expression (B, num_genes) or (B, T, num_genes)
        
        Returns:
            mu: Mean (B, latent_dim) or (B, T, latent_dim)
            logvar: Log variance
        """
        original_shape = x.shape
        
        # Flatten time dimension if present
        if len(x.shape) == 3:
            B, T, G = x.shape
            x = x.view(B * T, G)
        
        # Encode
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reshape if time dimension
        if len(original_shape) == 3:
            mu = mu.view(B, T, -1)
            logvar = logvar.view(B, T, -1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode from latent space.
        
        Args:
            z: Latent (B, latent_dim) or (B, T, latent_dim)
        
        Returns:
            x_recon: Reconstructed expression
        """
        original_shape = z.shape
        
        # Flatten time dimension if present
        if len(z.shape) == 3:
            B, T, L = z.shape
            z = z.view(B * T, L)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Reshape if time dimension
        if len(original_shape) == 3:
            x_recon = x_recon.view(B, T, -1)
        
        return x_recon
    
    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
```

### Training Strategy

**Joint training on mixed batches**:

```python
def train_joint_vae(
    model,
    bulk_data,
    timeseries_data,
    num_epochs=100,
    batch_size=64,
    device='cuda',
):
    """
    Train joint VAE on mixed batches.
    
    Args:
        model: JointBioVAE
        bulk_data: Static bulk RNA-seq (N, num_genes)
        timeseries_data: Time-series data (M, T, num_genes)
        num_epochs: Number of epochs
        batch_size: Batch size
        device: Device
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        # Sample mixed batch
        # Half bulk, half time-series
        bulk_idx = np.random.choice(len(bulk_data), batch_size // 2)
        time_idx = np.random.choice(len(timeseries_data), batch_size // 2)
        
        x_bulk = torch.tensor(bulk_data[bulk_idx], dtype=torch.float32).to(device)
        x_time = torch.tensor(timeseries_data[time_idx], dtype=torch.float32).to(device)
        
        # Forward pass on bulk
        x_bulk_recon, mu_bulk, logvar_bulk = model(x_bulk)
        loss_bulk = vae_loss(x_bulk_recon, x_bulk, mu_bulk, logvar_bulk)
        
        # Forward pass on time-series
        x_time_recon, mu_time, logvar_time = model(x_time)
        loss_time = vae_loss(x_time_recon, x_time, mu_time, logvar_time)
        
        # Total loss
        loss = loss_bulk + loss_time
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Bulk loss = {loss_bulk:.4f}, Time loss = {loss_time:.4f}")


def vae_loss(x_recon, x, mu, logvar):
    """VAE loss (ELBO)."""
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss
```

---

## Joint JEPA Architecture

### Concept

**Extend JEPA to joint latent spaces**:

```python
class JointBioJEPA(nn.Module):
    """
    Joint JEPA for static and dynamic biological data.
    
    Predicts in shared embedding space.
    """
    def __init__(
        self,
        num_genes=20000,
        embed_dim=256,
        num_tokens=64,
    ):
        super().__init__()
        
        # Shared encoder (works for both static and dynamic)
        self.encoder = GeneExpressionEncoder(
            num_genes=num_genes,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
        )
        
        # Predictor for temporal dynamics
        self.temporal_predictor = TemporalPredictor(
            embed_dim=embed_dim,
            depth=6,
        )
        
        # Predictor for perturbations
        self.perturbation_predictor = ConditionalPredictor(
            embed_dim=embed_dim,
            condition_dim=128,
            depth=4,
        )
        
        # VICReg loss
        self.vicreg = VICRegLoss()
    
    def forward_temporal(self, x_past, x_future):
        """
        Temporal prediction task.
        
        Args:
            x_past: Past frames (B, T_past, num_genes)
            x_future: Future frames (B, T_future, num_genes)
        
        Returns:
            loss: Temporal prediction loss
        """
        # Encode past and future
        B, T_past, G = x_past.shape
        x_past_flat = x_past.view(B * T_past, G)
        z_past = self.encoder(x_past_flat)
        z_past = z_past.view(B, T_past, -1, self.encoder.embed_dim)
        
        _, T_future, _ = x_future.shape
        x_future_flat = x_future.view(B * T_future, G)
        with torch.no_grad():
            z_future = self.encoder(x_future_flat)
        z_future = z_future.view(B, T_future, -1, self.encoder.embed_dim)
        
        # Predict future from past
        z_future_pred = self.temporal_predictor(z_past)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_future_pred, z_future)
        
        return loss, loss_dict
    
    def forward_perturbation(self, x_baseline, x_perturbed, pert_emb):
        """
        Perturbation prediction task.
        
        Args:
            x_baseline: Baseline (B, num_genes)
            x_perturbed: Perturbed (B, num_genes)
            pert_emb: Perturbation embedding (B, condition_dim)
        
        Returns:
            loss: Perturbation prediction loss
        """
        # Encode
        z_baseline = self.encoder(x_baseline)
        with torch.no_grad():
            z_perturbed = self.encoder(x_perturbed)
        
        # Predict
        z_perturbed_pred = self.perturbation_predictor(z_baseline, pert_emb)
        
        # VICReg loss
        loss, loss_dict = self.vicreg(z_perturbed_pred, z_perturbed)
        
        return loss, loss_dict
```

### Multi-Task Training

**Train on both tasks simultaneously**:

```python
def train_joint_jepa(
    model,
    timeseries_loader,
    perturbseq_loader,
    num_epochs=100,
    device='cuda',
):
    """
    Train joint JEPA on multiple tasks.
    
    Args:
        model: JointBioJEPA
        timeseries_loader: Time-series data loader
        perturbseq_loader: Perturb-seq data loader
        num_epochs: Number of epochs
        device: Device
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        # Iterate over both datasets
        for (x_past, x_future), (x_baseline, x_perturbed, pert_emb) in zip(
            timeseries_loader, perturbseq_loader
        ):
            # Temporal task
            x_past = x_past.to(device)
            x_future = x_future.to(device)
            loss_temporal, _ = model.forward_temporal(x_past, x_future)
            
            # Perturbation task
            x_baseline = x_baseline.to(device)
            x_perturbed = x_perturbed.to(device)
            pert_emb = pert_emb.to(device)
            loss_pert, _ = model.forward_perturbation(x_baseline, x_perturbed, pert_emb)
            
            # Total loss
            loss = loss_temporal + loss_pert
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Temporal = {loss_temporal:.4f}, Pert = {loss_pert:.4f}")
```

---

## Patch n' Pack: Variable-Length Sequences

### The Problem

**Traditional approach**:
- Pad all sequences to same length
- Waste computation on padding tokens
- Inefficient for variable-length data

**Biological example**:
- Time-series with different lengths (3 time points vs 10 time points)
- Single-cell samples with different cell counts
- Spatial data with different tissue sizes

### The Solution: Patch n' Pack

**Key idea**: Concatenate all tokens into one long sequence, use block attention masks.

```python
class PatchAndPack:
    """
    Patch n' Pack for variable-length biological sequences.
    
    Allows mixing different data types in same batch.
    """
    def __init__(self, max_tokens=2048):
        self.max_tokens = max_tokens
    
    def pack_batch(self, samples):
        """
        Pack variable-length samples into single batch.
        
        Args:
            samples: List of samples, each (num_tokens, embed_dim)
        
        Returns:
            packed: Packed tokens (1, total_tokens, embed_dim)
            attention_mask: Block diagonal mask (total_tokens, total_tokens)
            sample_boundaries: List of (start, end) indices
        """
        # Concatenate all tokens
        all_tokens = []
        sample_boundaries = []
        current_pos = 0
        
        for sample in samples:
            num_tokens = sample.shape[0]
            all_tokens.append(sample)
            sample_boundaries.append((current_pos, current_pos + num_tokens))
            current_pos += num_tokens
            
            if current_pos > self.max_tokens:
                break
        
        # Stack
        packed = torch.cat(all_tokens, dim=0).unsqueeze(0)  # (1, total_tokens, embed_dim)
        
        # Create block diagonal attention mask
        total_tokens = packed.shape[1]
        attention_mask = torch.zeros(total_tokens, total_tokens)
        
        for start, end in sample_boundaries:
            attention_mask[start:end, start:end] = 1
        
        return packed, attention_mask, sample_boundaries
    
    def unpack_batch(self, packed_output, sample_boundaries):
        """
        Unpack batch back to individual samples.
        
        Args:
            packed_output: Packed output (1, total_tokens, embed_dim)
            sample_boundaries: List of (start, end) indices
        
        Returns:
            samples: List of unpacked samples
        """
        samples = []
        for start, end in sample_boundaries:
            sample = packed_output[0, start:end, :]
            samples.append(sample)
        
        return samples
```

### Usage

```python
# Create samples of different lengths
sample1 = torch.randn(32, 256)  # 32 tokens
sample2 = torch.randn(64, 256)  # 64 tokens
sample3 = torch.randn(48, 256)  # 48 tokens

# Pack
packer = PatchAndPack(max_tokens=2048)
packed, mask, boundaries = packer.pack_batch([sample1, sample2, sample3])

print(f"Packed shape: {packed.shape}")  # (1, 144, 256)
print(f"Mask shape: {mask.shape}")      # (144, 144)

# Process with transformer
output = transformer(packed, attention_mask=mask)

# Unpack
samples_out = packer.unpack_batch(output, boundaries)
print(f"Sample 1 out: {samples_out[0].shape}")  # (32, 256)
print(f"Sample 2 out: {samples_out[1].shape}")  # (64, 256)
print(f"Sample 3 out: {samples_out[2].shape}")  # (48, 256)
```

---

## Open Research Questions

### 1. Optimal Latent Dimensionality

**Question**: What is the right dimensionality for joint biological latent space?

**Current practice**:
- Images: 256-1024 dim
- Gene expression: 128-512 dim

**Open issues**:
- Does biology need more or less than vision?
- How does dimensionality affect transfer?
- Can we learn optimal dimensionality?

**Proposed experiments**:
- Train joint models with varying latent dims
- Measure downstream task performance
- Analyze information content (effective rank)

### 2. Cross-Modality Transfer

**Question**: How much does static data help dynamic modeling, and vice versa?

**Hypothesis**: Joint training improves both tasks.

**Proposed experiments**:
1. Train separate models (static only, dynamic only)
2. Train joint model
3. Compare performance on both tasks
4. Measure transfer via ablation

**Metrics**:
- Downstream task accuracy
- Sample efficiency (performance with less data)
- Generalization to held-out conditions

### 3. Biological Priors

**Question**: Should we encode biological structure (pathways, GRNs) into latent space?

**Options**:
1. **Fully learned** — Let model discover structure
2. **Structured latent** — Enforce pathway/module structure
3. **Hybrid** — Soft biological priors

**Trade-offs**:
- Learned: Flexible but data-hungry
- Structured: Sample-efficient but rigid
- Hybrid: Best of both?

**Proposed approach**:
```python
class StructuredLatentJEPA(nn.Module):
    """
    JEPA with structured latent space.
    
    Latent dimensions correspond to biological pathways.
    """
    def __init__(self, num_genes, pathways):
        super().__init__()
        
        # Encoder projects to pathway space
        self.encoder = PathwayEncoder(num_genes, pathways)
        
        # Predictor operates on pathway embeddings
        self.predictor = PathwayPredictor(len(pathways))
```

### 4. Temporal Consistency

**Question**: How to enforce temporal consistency in joint models?

**Challenge**: Time-series should have smooth trajectories, but static data has no temporal structure.

**Proposed solutions**:
1. **Separate losses** — Temporal smoothness only for time-series
2. **Pseudo-time** — Infer temporal ordering for static data
3. **Consistency regularization** — Encourage smooth latent paths

**Implementation**:
```python
def temporal_consistency_loss(z_sequence):
    """
    Penalize large jumps in latent space.
    
    Args:
        z_sequence: Latent trajectory (B, T, latent_dim)
    
    Returns:
        loss: Temporal consistency loss
    """
    # Compute differences between consecutive time points
    dz = z_sequence[:, 1:, :] - z_sequence[:, :-1, :]
    
    # Penalize large jumps
    loss = torch.norm(dz, dim=-1).mean()
    
    return loss
```

### 5. Multi-Species Transfer

**Question**: Can joint latent spaces transfer across species?

**Motivation**: Conserved biology should have conserved latent structure.

**Proposed experiments**:
1. Train joint model on human + mouse data
2. Test cross-species prediction
3. Identify conserved vs species-specific dimensions

**Applications**:
- Translate mouse perturbations to human
- Validate drug effects across species
- Identify evolutionary conservation

### 6. Scalability

**Question**: How to scale joint models to millions of cells and thousands of conditions?

**Challenges**:
- Memory constraints
- Training time
- Data heterogeneity

**Proposed solutions**:
1. **Hierarchical latents** — Coarse-to-fine structure
2. **Distributed training** — Multi-GPU/node
3. **Efficient attention** — Sparse, linear, or kernel-based
4. **Patch n' Pack** — Variable-length batching

---

## Practical Recommendations

### When to Use Joint Latent Spaces

**Use joint latent spaces when**:
1. You have both static and dynamic data
2. Data types are related (same genes, different conditions)
3. You want to transfer knowledge across modalities
4. Sample efficiency matters (limited data per modality)

**Don't use joint latent spaces when**:
1. Data types are fundamentally different (e.g., images + text)
2. You have abundant data for each modality separately
3. Modalities have different downstream tasks with no overlap

### Implementation Checklist

**1. Data preparation**:
- [ ] Normalize static and dynamic data consistently
- [ ] Align feature spaces (same genes)
- [ ] Create mixed batches

**2. Model architecture**:
- [ ] Shared encoder for both modalities
- [ ] Modality-specific predictors (optional)
- [ ] VICReg or similar regularization

**3. Training**:
- [ ] Mixed batch sampling
- [ ] Balanced loss weighting
- [ ] Monitor both tasks

**4. Evaluation**:
- [ ] Test on both modalities
- [ ] Measure cross-modality transfer
- [ ] Validate on downstream tasks

---

## Key Takeaways

### Conceptual

1. **Joint latent spaces** — Static and dynamic data can share representations
2. **Mutual benefit** — Each modality improves the other
3. **Biological manifold** — All cellular states on same manifold
4. **Patch n' Pack** — Efficient variable-length batching

### Practical

1. **Shared encoder** — Same architecture for all modalities
2. **Mixed training** — Alternate between tasks
3. **VICReg regularization** — Prevent collapse
4. **Evaluate transfer** — Measure cross-modality benefits

### Open Questions

1. **Optimal dimensionality** — How large should latent space be?
2. **Biological priors** — Should we encode known structure?
3. **Temporal consistency** — How to enforce smooth trajectories?
4. **Multi-species** — Can latents transfer across species?
5. **Scalability** — How to handle millions of cells?

---

## Related Documents

- [00_jepa_overview.md](00_jepa_overview.md) — JEPA concepts
- [01_jepa_foundations.md](01_jepa_foundations.md) — Architecture details
- [03_jepa_applications.md](03_jepa_applications.md) — Applications
- [04_jepa_perturbseq.md](04_jepa_perturbseq.md) — Perturb-seq implementation

---

## References

**Joint latent spaces**:
- ByteDance & HKU (2024): "Goku: Native Joint Image-Video Generation"
- Meta AI (2025): "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction, and Planning"

**JEPA**:
- LeCun (2022): "A Path Towards Autonomous Machine Intelligence"
- Assran et al. (2023): "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"

**Biological applications**:
- Lotfollahi et al. (2023): "Predicting cellular responses to novel drug combinations"
- Bunne et al. (2023): "Learning Single-Cell Perturbation Responses using Neural Optimal Transport"

**Variable-length sequences**:
- Vaswani et al. (2017): "Attention is All You Need"
- Press et al. (2021): "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
