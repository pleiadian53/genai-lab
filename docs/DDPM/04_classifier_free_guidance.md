# Classifier-Free Guidance for DDPM: Implementation Guide

**Related**: For theoretical foundations and general diffusion context, see [classifier_free_guidance.md](../diffusion/classifier_free_guidance.md). This document focuses on **practical implementation** in DDPM.

---

## Overview

Classifier-free guidance enables high-quality conditional generation in DDPM without needing a separate classifier network. This document covers:

1. How to modify DDPM training for guidance
2. How to sample with guidance
3. Practical implementation details
4. Hyperparameter tuning
5. Common pitfalls and solutions

**Key idea**: Train one model that handles both conditional and unconditional generation, then blend their predictions at sampling time.

---

## Quick Reference

### Training

```python
# Randomly drop condition with probability p_uncond (typically 0.1)
if random.random() < p_uncond:
    c = null_token  # Unconditional
else:
    c = condition   # Conditional

# Train as normal
epsilon_pred = model(x_t, t, c)
loss = MSE(epsilon, epsilon_pred)
```

### Sampling

```python
# Two forward passes per step
epsilon_uncond = model(x_t, t, null_token)
epsilon_cond = model(x_t, t, condition)

# Blend with guidance scale w
epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)

# Use guided prediction for denoising
x_{t-1} = denoise_step(x_t, epsilon_guided, t)
```

---

## Part 1: Modifying DDPM Training

### Standard DDPM Training (Unconditional)

Recall the standard DDPM training algorithm:

```python
# Standard unconditional DDPM
for epoch in range(num_epochs):
    for batch in dataloader:
        x_0 = batch['image']
        
        # Sample timestep and noise
        t = random.randint(1, T)
        epsilon = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        # Predict noise (unconditional)
        epsilon_pred = model(x_t, t)
        
        # Loss
        loss = F.mse_loss(epsilon_pred, epsilon)
        loss.backward()
        optimizer.step()
```

### Modified Training for Classifier-Free Guidance

**Key changes**:
1. Dataset includes conditions: `(x, c)` pairs
2. Randomly replace condition with null token
3. Model takes condition as input

```python
# Classifier-free guidance training
for epoch in range(num_epochs):
    for batch in dataloader:
        x_0 = batch['image']      # Data
        c = batch['condition']    # Condition (class, text, etc.)
        
        # Sample timestep and noise
        t = random.randint(1, T)
        epsilon = torch.randn_like(x_0)
        
        # Forward diffusion (same as before)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        # ──────────────────────────────────────────────
        # NEW: Randomly drop condition
        # ──────────────────────────────────────────────
        if random.random() < p_uncond:  # Typically p_uncond = 0.1
            c = null_token  # Use null/empty condition
        
        # Predict noise (now conditional)
        epsilon_pred = model(x_t, t, c)
        
        # Loss (same as before)
        loss = F.mse_loss(epsilon_pred, epsilon)
        loss.backward()
        optimizer.step()
```

### Implementation Details

#### 1. Null Token Representation

Different ways to represent "no condition":

```python
# Option 1: Zero vector (simplest)
null_token = torch.zeros(condition_dim)

# Option 2: Learnable embedding
class Model(nn.Module):
    def __init__(self):
        self.null_embedding = nn.Parameter(torch.randn(condition_dim))
    
    def get_null_token(self):
        return self.null_embedding

# Option 3: Special token index (for discrete conditions)
null_token = -1  # or num_classes + 1

# Option 4: Mask flag (most explicit)
use_condition = False  # Boolean flag to model
```

**Recommendation**: 

- **Discrete conditions** (class labels): Use special index
- **Continuous conditions** (embeddings): Use zero vector or learnable embedding
- **Text conditions**: Use empty string "" or padding tokens

#### 2. Condition Dropout Probability

```python
p_uncond = 0.1  # Typical value

# Higher values (0.2): Better unconditional generation
# Lower values (0.05): Better conditional generation
# Trade-off: unconditional quality vs conditional quality
```

**Guidelines**:

- `p_uncond = 0.1`: Standard choice for most applications
- `p_uncond = 0.2`: If unconditional quality matters (text-to-image)
- `p_uncond = 0.05`: If you always want conditioned output

#### 3. Model Architecture Changes

Your model must accept the condition:

```python
class DDPMWithGuidance(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.time_embed = TimeEmbedding(256)
        self.class_embed = nn.Embedding(num_classes + 1, 256)  # +1 for null
        
        # Combine time and class embeddings
        self.combine = nn.Linear(512, 512)
        
        # U-Net or other architecture
        self.unet = UNet(...)
    
    def forward(self, x_t, t, c):
        """
        Args:
            x_t: Noisy input (batch, channels, height, width)
            t: Timesteps (batch,)
            c: Conditions (batch,) - class indices
        
        Returns:
            epsilon_pred: Predicted noise (batch, channels, height, width)
        """
        # Embed time
        t_emb = self.time_embed(t)  # (batch, 256)
        
        # Embed condition
        c_emb = self.class_embed(c)  # (batch, 256)
        
        # Combine
        conditioning = self.combine(torch.cat([t_emb, c_emb], dim=1))
        
        # U-Net with conditioning
        epsilon_pred = self.unet(x_t, conditioning)
        
        return epsilon_pred
```

**For Transformers (DiT)**:

```python
class DiTWithGuidance(nn.Module):
    def forward(self, x_t, t, c):
        # Create embeddings
        time_embed = self.time_embed(t)
        class_embed = self.class_embed(c)
        
        # Combine for AdaLN
        combined = time_embed + class_embed
        gamma, beta = self.adaln_mlp(combined)
        
        # Standard DiT processing
        h = self.patchify(x_t)
        for block in self.blocks:
            h = block(h, gamma, beta)
        
        return self.unpatchify(h)
```

---

## Part 2: Sampling with Guidance

### Standard DDPM Sampling (Review)

```python
def ddpm_sample(model, shape, T=1000):
    """Standard unconditional DDPM sampling"""
    x = torch.randn(shape)  # Start from noise
    
    for t in reversed(range(1, T+1)):
        # Predict noise
        epsilon_pred = model(x, t)
        
        # Compute mean
        alpha_t = get_alpha(t)
        alpha_bar_t = get_alpha_bar(t)
        mean = (1 / sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / sqrt(1 - alpha_bar_t)) * epsilon_pred
        )
        
        # Add noise (except last step)
        if t > 1:
            sigma = get_sigma(t)
            z = torch.randn_like(x)
            x = mean + sigma * z
        else:
            x = mean
    
    return x
```

### Guided Sampling (Two Forward Passes)

```python
def guided_sample(model, shape, condition, guidance_scale=7.5, T=1000):
    """
    Classifier-free guided DDPM sampling
    
    Args:
        model: Trained model with condition input
        shape: Output shape
        condition: Condition to use (class, text embedding, etc.)
        guidance_scale: w, typically 1.0-10.0
        T: Number of diffusion steps
    
    Returns:
        Generated sample
    """
    x = torch.randn(shape)  # Start from noise
    null_token = get_null_token()  # Unconditional token
    
    for t in reversed(range(1, T+1)):
        # ──────────────────────────────────────────────
        # KEY: Two forward passes
        # ──────────────────────────────────────────────
        
        # 1. Unconditional prediction
        epsilon_uncond = model(x, t, null_token)
        
        # 2. Conditional prediction
        epsilon_cond = model(x, t, condition)
        
        # 3. Blend with guidance scale
        epsilon_guided = epsilon_uncond + guidance_scale * (
            epsilon_cond - epsilon_uncond
        )
        
        # ──────────────────────────────────────────────
        # Standard DDPM update (same as before)
        # ──────────────────────────────────────────────
        
        alpha_t = get_alpha(t)
        alpha_bar_t = get_alpha_bar(t)
        
        # Compute mean using GUIDED noise prediction
        mean = (1 / sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / sqrt(1 - alpha_bar_t)) * epsilon_guided
        )
        
        # Add noise (except last step)
        if t > 1:
            sigma = get_sigma(t)
            z = torch.randn_like(x)
            x = mean + sigma * z
        else:
            x = mean
    
    return x
```

### Alternative: Single Forward Pass (Training with Guidance Embedding)

Some implementations embed the guidance scale during training:

```python
# Training with guidance scale as input
epsilon_pred = model(x_t, t, c, guidance_scale)

# Sampling (single forward pass)
epsilon_guided = model(x_t, t, c, w)
```

**Trade-off**: 

- ✅ Faster sampling (1 pass instead of 2)
- ❌ Less flexible (can't change w without retraining)
- ❌ More complex training

**Recommendation**: Use two-pass approach for flexibility.

---

## Part 3: Guidance Scale Selection

### Effect of Guidance Scale

```python
w = 0.0   # Pure unconditional (ignores condition)
w = 1.0   # Standard conditional (no guidance)
w = 3.0   # Mild guidance
w = 7.5   # Strong guidance (common for text-to-image)
w = 15.0  # Very strong guidance (may cause artifacts)
```

### Empirical Guidelines

| Application | Typical w | Notes |
|-------------|-----------|-------|
| **Class-conditional images** | 3-5 | Lower values work well |
| **Text-to-image** | 7-10 | Higher values needed |
| **Image inpainting** | 5-7 | Balance coherence and diversity |
| **Super-resolution** | 1-3 | Lower to preserve details |

### Trade-offs

```
Guidance Scale     Fidelity to Condition     Diversity     Quality
─────────────────────────────────────────────────────────────────
w = 1              ○ Weak                    ✓ High        ○ Moderate
w = 3-5            ✓ Good                    ✓ Good        ✓ Good
w = 7-10           ✓✓ Strong                 ○ Lower       ✓ Good
w > 15             ✓✓✓ Very Strong           ✗ Very Low    ✗ Artifacts
```

### Adaptive Guidance

You can vary guidance scale during sampling:

```python
def adaptive_guided_sample(model, shape, condition, T=1000):
    x = torch.randn(shape)
    null_token = get_null_token()
    
    for t in reversed(range(1, T+1)):
        # Adaptive guidance scale
        if t > 800:  # Early steps: high noise
            w = 10.0  # Strong guidance for structure
        elif t > 200:  # Middle steps
            w = 7.5   # Moderate guidance
        else:  # Final steps: low noise
            w = 3.0   # Weak guidance for details
        
        epsilon_uncond = model(x, t, null_token)
        epsilon_cond = model(x, t, condition)
        epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
        
        # Standard update
        x = denoise_step(x, epsilon_guided, t)
    
    return x
```

---

## Part 4: Implementation Examples

### Example 1: Class-Conditional CIFAR-10

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassConditionalDDPM(nn.Module):
    def __init__(self, num_classes=10, img_channels=3, base_channels=128):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(
            num_classes + 1,  # +1 for null token
            base_channels * 4
        )
        
        # U-Net architecture (simplified)
        self.encoder = UNetEncoder(img_channels, base_channels)
        self.bottleneck = UNetBottleneck(base_channels * 8)
        self.decoder = UNetDecoder(base_channels, img_channels)
        
        self.null_class_idx = num_classes  # Index for unconditional
    
    def forward(self, x_t, t, c):
        """
        Args:
            x_t: (batch, 3, 32, 32) - noisy images
            t: (batch,) - timesteps
            c: (batch,) - class indices (or null_class_idx)
        """
        # Embeddings
        t_emb = self.time_embed(t)  # (batch, 512)
        c_emb = self.class_embed(c)  # (batch, 512)
        
        # Combine (simple addition)
        conditioning = t_emb + c_emb  # (batch, 512)
        
        # U-Net forward
        features = self.encoder(x_t)
        bottleneck = self.bottleneck(features[-1], conditioning)
        epsilon_pred = self.decoder(bottleneck, features)
        
        return epsilon_pred
    
    def get_null_token(self, batch_size, device):
        """Return null token for unconditional generation"""
        return torch.full((batch_size,), self.null_class_idx, 
                         dtype=torch.long, device=device)


# Training
def train_with_guidance(model, dataloader, p_uncond=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x_0 = batch['image']  # (batch, 3, 32, 32)
            c = batch['label']    # (batch,) class indices 0-9
            
            # Sample timestep and noise
            t = torch.randint(1, T + 1, (x_0.shape[0],))
            epsilon = torch.randn_like(x_0)
            
            # Forward diffusion
            alpha_bar_t = get_alpha_bar(t)
            x_t = torch.sqrt(alpha_bar_t)[:, None, None, None] * x_0 + \
                  torch.sqrt(1 - alpha_bar_t)[:, None, None, None] * epsilon
            
            # Randomly drop condition
            mask = torch.rand(x_0.shape[0]) < p_uncond
            c = torch.where(mask, model.null_class_idx, c)
            
            # Predict noise
            epsilon_pred = model(x_t, t, c)
            
            # Loss
            loss = F.mse_loss(epsilon_pred, epsilon)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# Sampling
@torch.no_grad()
def sample_with_guidance(model, class_idx, guidance_scale=5.0, 
                        num_samples=4, device='cuda'):
    """Generate samples for a specific class"""
    model.eval()
    
    # Initialize from noise
    x = torch.randn(num_samples, 3, 32, 32, device=device)
    
    # Prepare conditions
    condition = torch.full((num_samples,), class_idx, 
                          dtype=torch.long, device=device)
    null_token = model.get_null_token(num_samples, device)
    
    # Reverse diffusion
    for t in reversed(range(1, T + 1)):
        t_batch = torch.full((num_samples,), t, device=device)
        
        # Two forward passes
        epsilon_uncond = model(x, t_batch, null_token)
        epsilon_cond = model(x, t_batch, condition)
        
        # Guided prediction
        epsilon_guided = epsilon_uncond + guidance_scale * (
            epsilon_cond - epsilon_uncond
        )
        
        # DDPM update
        alpha_t = get_alpha(t)
        alpha_bar_t = get_alpha_bar(t)
        beta_t = 1 - alpha_t
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_guided
        )
        
        if t > 1:
            sigma = torch.sqrt(beta_t)
            z = torch.randn_like(x)
            x = mean + sigma * z
        else:
            x = mean
    
    return x
```

### Example 2: Text-Conditional Generation

```python
class TextConditionalDDPM(nn.Module):
    def __init__(self, text_embed_dim=512):
        super().__init__()
        
        self.time_embed = TimeEmbedding(256)
        
        # Text encoder (e.g., pre-trained CLIP or T5)
        self.text_encoder = CLIPTextEncoder()
        
        # Cross-attention U-Net
        self.unet = CrossAttentionUNet(
            text_dim=text_embed_dim,
            time_dim=256
        )
        
        # Null text embedding (learnable)
        self.null_text_embed = nn.Parameter(
            torch.randn(text_embed_dim)
        )
    
    def forward(self, x_t, t, text_embedding):
        """
        Args:
            x_t: Noisy images
            t: Timesteps
            text_embedding: Text embeddings (batch, seq_len, 512)
                           Or null_text_embed for unconditional
        """
        t_emb = self.time_embed(t)
        epsilon_pred = self.unet(x_t, t_emb, text_embedding)
        return epsilon_pred
    
    def encode_text(self, text_prompts):
        """Encode text prompts to embeddings"""
        return self.text_encoder(text_prompts)
    
    def get_null_token(self, batch_size):
        """Return null token for unconditional generation"""
        # Expand to match text sequence shape if needed
        return self.null_text_embed.unsqueeze(0).expand(
            batch_size, -1, -1
        )


# Training
def train_text_conditional(model, dataloader, p_uncond=0.1):
    for batch in dataloader:
        x_0 = batch['image']
        text = batch['caption']
        
        # Encode text
        text_embed = model.encode_text(text)
        
        # Standard diffusion forward
        t = torch.randint(1, T + 1, (x_0.shape[0],))
        epsilon = torch.randn_like(x_0)
        x_t = add_noise(x_0, t, epsilon)
        
        # Randomly use null token
        mask = torch.rand(x_0.shape[0]) < p_uncond
        null_embed = model.get_null_token(x_0.shape[0])
        text_embed = torch.where(
            mask[:, None, None], null_embed, text_embed
        )
        
        # Predict and train
        epsilon_pred = model(x_t, t, text_embed)
        loss = F.mse_loss(epsilon_pred, epsilon)
        # ... backprop
```

---

## Part 5: Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Train on Unconditional

**Problem**: Only training with conditions, forgetting to drop them

```python
# ❌ WRONG: Never drops condition
epsilon_pred = model(x_t, t, c)  # Always has c
loss = F.mse_loss(epsilon_pred, epsilon)
```

**Solution**: Always implement condition dropout

```python
# ✅ CORRECT: Randomly drop condition
if random.random() < p_uncond:
    c = null_token
epsilon_pred = model(x_t, t, c)
```

### Pitfall 2: Using Wrong Null Token

**Problem**: Inconsistent null token between training and sampling

```python
# Training: uses zeros
if random.random() < p_uncond:
    c = torch.zeros_like(c)

# Sampling: uses -1
null_token = torch.full_like(c, -1)  # ❌ Mismatch!
```

**Solution**: Use the same null representation

```python
# Define once, use everywhere
NULL_CLASS_IDX = num_classes  # e.g., 10 for CIFAR-10

# Training
if random.random() < p_uncond:
    c = torch.full_like(c, NULL_CLASS_IDX)

# Sampling
null_token = torch.full((batch_size,), NULL_CLASS_IDX)
```

### Pitfall 3: Guidance Scale Too High

**Problem**: Over-guidance causes artifacts

```python
# ❌ Too high: w=20
epsilon_guided = epsilon_uncond + 20 * (epsilon_cond - epsilon_uncond)
# Result: Oversaturated, unrealistic images
```

**Solution**: Start low and increase gradually

```python
# ✅ Start with moderate values
for w in [1.0, 3.0, 5.0, 7.5, 10.0]:
    samples = sample_with_guidance(model, condition, w)
    evaluate(samples)  # Find best w
```

### Pitfall 4: Not Caching Unconditional Predictions

**Problem**: Computing unconditional prediction every step is wasteful

```python
# Inefficient: compute unconditional every time
for t in range(T, 0, -1):
    epsilon_uncond = model(x, t, null_token)  # Same for all conditions!
    epsilon_cond = model(x, t, condition)
```

**Solution**: Batch multiple conditions

```python
# Better: batch process if generating multiple samples with same x_t
# (Not always applicable, but useful for parallel generation)
batch_conditions = [cond1, cond2, ..., null_token]
epsilon_all = model(x.repeat(len(batch_conditions), 1, 1, 1), 
                   t, batch_conditions)
```

### Pitfall 5: Numerical Instability with High Guidance

**Problem**: Extremely large guidance scales cause NaNs

```python
# Can cause numerical issues
epsilon_guided = epsilon_uncond + 100 * (epsilon_cond - epsilon_uncond)
```

**Solution**: Clip guided predictions

```python
# Clip to reasonable range
epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)
epsilon_guided = torch.clamp(epsilon_guided, -10, 10)  # Prevent extremes
```

---

## Part 6: Advanced Techniques

### 1. Dynamic Guidance Schedules

Vary guidance strength throughout sampling:

```python
def get_dynamic_guidance(t, T):
    """
    Higher guidance early (structure)
    Lower guidance late (details)
    """
    progress = t / T  # 1.0 → 0.0
    
    if progress > 0.8:  # Very noisy
        return 10.0
    elif progress > 0.5:  # Moderately noisy
        return 7.5
    elif progress > 0.2:  # Less noisy
        return 5.0
    else:  # Almost clean
        return 3.0
```

### 2. Guidance with Multiple Conditions

```python
# Multiple conditions: class + style + color
epsilon_guided = (
    epsilon_uncond + 
    w_class * (epsilon_class - epsilon_uncond) +
    w_style * (epsilon_style - epsilon_uncond) +
    w_color * (epsilon_color - epsilon_uncond)
)
```

### 3. Negative Guidance

Push away from unwanted conditions:

```python
# Generate "not a dog"
epsilon_guided = epsilon_uncond - w_negative * (
    epsilon_dog - epsilon_uncond
)
```

### 4. Self-Guidance (Unconditional Only)

Use guidance even without conditions:

```python
# Split noise prediction into components
epsilon_mean = epsilon_pred.mean()
epsilon_guided = epsilon_mean + w * (epsilon_pred - epsilon_mean)
```

---

## Part 7: Evaluation and Debugging

### Metrics to Track

#### 1. Condition Fidelity

How well do samples match the condition?

```python
# For class-conditional
classifier_accuracy = pretrained_classifier(samples)

# For text-to-image
clip_score = compute_clip_score(samples, text_prompts)
```

#### 2. Sample Quality

```python
# FID (Fréchet Inception Distance)
fid_score = compute_fid(generated_samples, real_samples)

# Inception Score
is_score = compute_inception_score(generated_samples)
```

#### 3. Diversity

```python
# Intra-class diversity
diversity = compute_pairwise_distance(samples_same_class)

# Should decrease with higher guidance
```

### Debugging Checklist

```python
# 1. Check unconditional generation works
samples_uncond = sample_with_guidance(model, condition, w=0.0)
# Should produce diverse, realistic (but generic) samples

# 2. Check conditional generation works
samples_cond = sample_with_guidance(model, condition, w=1.0)
# Should produce samples matching condition

# 3. Check guidance improves fidelity
for w in [1.0, 3.0, 5.0, 7.5]:
    samples = sample_with_guidance(model, condition, w)
    fidelity = measure_condition_fidelity(samples, condition)
    print(f"w={w}: fidelity={fidelity}")
# Fidelity should increase with w

# 4. Check for mode collapse
samples = [sample_with_guidance(model, condition, w=7.5) 
           for _ in range(100)]
diversity = compute_diversity(samples)
# Diversity should be reasonable (not all identical)

# 5. Visualize guidance scale sweep
visualize_guidance_sweep(model, condition, w_values=[0, 1, 3, 5, 7, 10, 15])
```

### Common Issues and Fixes

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Unconditional (w=0) is poor | Not enough uncond training | Increase p_uncond |
| High w causes artifacts | Guidance too strong | Lower w or clip predictions |
| All samples look identical | Mode collapse | Lower w, check diversity loss |
| Condition ignored even at w=10 | Model didn't learn conditioning | Check condition embedding, increase training |
| NaN during sampling | Numerical instability | Clip predictions, lower w |

---

## Part 8: Practical Tips

### Training

1. **Start with unconditional**: Train a good unconditional model first, then add guidance
2. **Use moderate p_uncond**: 0.1 is safe, 0.2 if unconditional quality matters
3. **Monitor both**: Track unconditional and conditional losses separately
4. **Longer training**: Guidance requires more iterations to converge

### Sampling

1. **Start low**: Begin with w=1-3 and increase if needed
2. **Application-specific**: Text-to-image needs higher w than class-conditional
3. **Quality > adherence**: Don't sacrifice quality for perfect condition matching
4. **Use dynamic schedules**: High w early, low w late often works well

### Hyperparameters

```python
# Recommended starting points
CONFIG = {
    'p_uncond': 0.1,           # Unconditional probability
    'guidance_scale': 7.5,      # Default w (tune for your task)
    'min_guidance': 1.0,        # Minimum w
    'max_guidance': 15.0,       # Maximum w
    'clip_epsilon': 10.0,       # Clip range for predictions
}
```

---

## Part 9: Complete Working Example

Here's a minimal, complete implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDDPM(nn.Module):
    """Minimal DDPM with classifier-free guidance"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Simple architecture for demonstration
        self.class_embed = nn.Embedding(num_classes + 1, 128)
        self.time_embed = SinusoidalEmbedding(128)
        self.net = SimpleUNet(in_channels=3, time_dim=128, class_dim=128)
        self.null_class = num_classes
    
    def forward(self, x, t, c):
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c)
        return self.net(x, t_emb + c_emb)

# Training
def train(model, dataloader, T=1000, p_uncond=0.1, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # Sample t and noise
            t = torch.randint(1, T+1, (images.shape[0],))
            noise = torch.randn_like(images)
            
            # Add noise
            alpha_bar = get_alpha_bar(t)
            x_t = torch.sqrt(alpha_bar[:, None, None, None]) * images + \
                  torch.sqrt(1 - alpha_bar[:, None, None, None]) * noise
            
            # Drop condition randomly
            mask = torch.rand(images.shape[0]) < p_uncond
            labels = torch.where(mask, model.null_class, labels)
            
            # Predict and loss
            noise_pred = model(x_t, t, labels)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Sampling
@torch.no_grad()
def sample(model, class_idx, w=7.5, T=1000):
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    
    for t in reversed(range(1, T+1)):
        # Two forward passes
        t_tensor = torch.tensor([t])
        null = torch.tensor([model.null_class])
        cond = torch.tensor([class_idx])
        
        noise_uncond = model(x, t_tensor, null)
        noise_cond = model(x, t_tensor, cond)
        
        # Guided prediction
        noise_guided = noise_uncond + w * (noise_cond - noise_uncond)
        
        # DDPM step
        alpha = get_alpha(t)
        alpha_bar = get_alpha_bar(t)
        beta = 1 - alpha
        
        mean = (x - (beta / torch.sqrt(1 - alpha_bar)) * noise_guided) / torch.sqrt(alpha)
        
        if t > 1:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    
    return x

# Usage
model = SimpleDDPM(num_classes=10)
train(model, train_loader)
sample_image = sample(model, class_idx=3, w=7.5)  # Generate class 3
```

---

## Summary

**Classifier-free guidance** enables high-quality conditional generation in DDPM:

### Key Points

1. **Training**: Randomly drop conditions (10-20% of time) with null token
2. **Sampling**: Two forward passes per step (unconditional + conditional)
3. **Guidance**: Blend predictions with scale w (typically 3-10)
4. **Trade-off**: Higher w → better condition matching, lower diversity

### Implementation Checklist

- [ ] Model accepts condition input
- [ ] Define null token consistently
- [ ] Implement condition dropout in training (p_uncond ≈ 0.1)
- [ ] Two forward passes during sampling
- [ ] Start with moderate guidance scale (w=5-7)
- [ ] Monitor both conditional and unconditional quality
- [ ] Clip predictions if numerical issues arise

### When to Use

✅ **Use classifier-free guidance when**:
- Need high-quality conditional generation
- Want control over condition strength
- Don't want to train a separate classifier
- Text-to-image, class-conditional, or any conditional task

❌ **Don't use when**:
- Unconditional generation only
- Inference speed is critical (2x slower)
- Limited training data (needs both conditional and unconditional)

---

**Next**: [Advanced Sampling Techniques](05_advanced_sampling.md) | [Back to DDPM Overview](README.md)

**Related**: [Theoretical Foundations](../diffusion/classifier_free_guidance.md) | [DiT with Guidance](../diffusion/DiT/diffusion_transformer.md)
