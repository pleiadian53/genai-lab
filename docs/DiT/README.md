# Diffusion Transformers (DiT) + Rectified Flow

This directory contains comprehensive documentation on **Diffusion Transformers (DiT)** combined with **rectified flow** — the modern architecture for scalable, flexible generative modeling.

DiT represents the shift from convolutional U-Nets to Transformers, enabling better scaling, flexible conditioning, and modality-agnostic generation.

---

## Core Documentation Series

This series follows the same structure as DDPM, SDE, and flow matching documentation.

| Document | Description |
|----------|-------------|
| [00_dit_overview.md](00_dit_overview.md) | **Overview**: Why DiT matters, key concepts, modern stack |
| [01_dit_foundations.md](01_dit_foundations.md) | **Foundations**: Architecture details, components, design choices |
| [02_dit_training.md](02_dit_training.md) | **Training**: How to train DiT + rectified flow models |
| [03_dit_sampling.md](03_dit_sampling.md) | **Sampling**: How to generate samples efficiently |

### Supplementary Documents

Deep dives on specific topics (located in `docs/diffusion/DiT/`):

| Document | Description |
|----------|-------------|
| [diffusion_transformer.md](../diffusion/DiT/diffusion_transformer.md) | Comprehensive tutorial with biology applications |
| [time_embeddings_explained.md](../diffusion/DiT/time_embeddings_explained.md) | Deep dive on time conditioning mechanisms |

---

## Quick Navigation

### For Beginners
Start with the overview to understand the big picture, then move through foundations and training.

**Path**: [Overview](00_dit_overview.md) → [Foundations](01_dit_foundations.md) → [Training](02_dit_training.md)

### For Implementation
Focus on the practical training and sampling guides with code examples.

**Path**: [Training](02_dit_training.md) → [Sampling](03_dit_sampling.md) → Supplementary docs

### For Theory Deep Dive
Understand the architectural choices and mathematical foundations.

**Path**: [Foundations](01_dit_foundations.md) → [Supplementary docs](../diffusion/DiT/) → [Flow matching theory](../flow_matching/)

---

## Key Concepts

### The Modern Generative Stack

```
Rectified Flow (objective) + DiT (architecture) + AdaLN (conditioning)
```

**Rectified Flow**: Simple regression target
$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

**DiT**: Transformer-based architecture
- Tokenization: Input → patches → tokens
- Self-attention: Global dependencies
- AdaLN: Time/condition modulation

**Result**: Fast, scalable, flexible generation

### DiT vs U-Net

| Aspect | U-Net | DiT |
|--------|-------|-----|
| **Architecture** | Convolutional | Transformer |
| **Receptive field** | Local → Global | Global from start |
| **Input format** | Fixed grids | Flexible tokens |
| **Conditioning** | Architectural changes | Built-in (AdaLN) |
| **Scaling** | Limited | Excellent |
| **Best for** | Images, fixed size | Any modality |

### Core Components

**1. Tokenization**
```
Image/Data → Patches → Flatten → Embed → Tokens
```

**2. Time Conditioning (AdaLN)**
```
t → TimeEmbed(t) → MLP → (γ, β) → Modulate features
```

**3. Transformer Blocks**
```
Tokens → Self-Attention → MLP → Updated Tokens
```

**4. Output Projection**
```
Tokens → Linear → Velocity Field
```

---

## Training Overview

### Rectified Flow Loss

**Simple regression**:

$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

where:

- $x_0 \sim p_{\text{data}}$ (real data)
- $x_1 \sim \mathcal{N}(0, I)$ (noise)
- $x_t = t x_1 + (1-t) x_0$ (linear interpolation)

**Key advantages**:

- No noise schedules
- No variance parameterization
- Direct regression target
- Stable training

### Training Algorithm

```python
for batch in dataloader:
    x_0 = batch  # Real data
    x_1 = torch.randn_like(x_0)  # Noise
    t = torch.rand(batch_size)  # Random time
    
    # Linear interpolation
    x_t = t * x_1 + (1 - t) * x_0
    
    # Predict velocity
    v_pred = model(x_t, t)
    
    # Compute loss
    target = x_1 - x_0
    loss = F.mse_loss(v_pred, target)
    
    # Update
    loss.backward()
    optimizer.step()
```

---

## Sampling Overview

### ODE Integration

**Forward ODE** (noise → data):

$$
\frac{dx}{dt} = v_\theta(x, t)
$$

**Euler discretization**:
```python
x = torch.randn(shape)  # Start from noise
dt = 1.0 / num_steps

for k in range(num_steps):
    t = k * dt
    v = model(x, t)
    x = x + v * dt

return x  # Generated sample
```

**Properties**:

- Deterministic (same noise → same output)
- Fast (20-50 steps)
- Straight paths (rectified flow)

---

## Applications

### Vision
- **Images**: Stable Diffusion 3, DALL-E 3
- **Videos**: Sora, Goku
- **3D**: Point clouds, meshes

### Audio
- **Music**: MusicGen
- **Speech**: AudioLDM
- **Sound effects**: Foley generation

### Biology
- **Gene expression**: Cell state generation
- **Perturbations**: Predict intervention effects
- **Trajectories**: Developmental paths
- **Molecules**: Protein structure

### Other
- **Robotics**: Trajectory planning
- **Physics**: Simulation
- **Design**: CAD, architecture

---

## Why DiT for Biology?

### Challenges with Traditional Approaches

**Gene expression data**:

- High-dimensional (10K-30K genes)
- Unordered (no natural sequence)
- Sparse (many zeros)
- Compositional (relative values matter)

**U-Net limitations**:

- Assumes spatial structure
- Fixed input sizes
- Hard to condition on perturbations

### DiT Advantages

**Flexibility**:

- Genes/cells/regions as tokens
- Variable-length sequences
- Natural conditioning on perturbations

**Global interactions**:

- Self-attention captures gene-gene dependencies
- No locality bias
- Learn regulatory networks

**Scalability**:

- Handle large gene panels
- Batch different experiments
- Scale to billions of parameters

### Open Questions

1. **Tokenization**: How to represent genes as tokens?
   - Rank by expression? (Geneformer approach)
   - Gene embeddings? (learned representations)
   - Set-based? (permutation invariant)

2. **Latent space**: Better to work in latent space?
   - Encode expression → latent → diffusion
   - Avoids sparsity issues
   - More stable training

3. **Architecture**: DiT vs alternatives?
   - State-space models (Mamba, S4)
   - Hyena (long convolutions)
   - Hybrid approaches

**See**: Supplementary documents for deeper exploration.

---

## Learning Path

### Conceptual Understanding

1. **[DiT Overview](00_dit_overview.md)** — Why DiT matters
   - Architectural shift from U-Net
   - Modern generative stack
   - Key advantages

2. **[Flow Matching Basics](../flow_matching/)** — Rectified flow theory
   - Velocity fields
   - Linear interpolation
   - ODE sampling

3. **[DiT Foundations](01_dit_foundations.md)** — Architecture details
   - Tokenization strategies
   - Transformer blocks
   - Time conditioning

### Practical Implementation

4. **[DiT Training](02_dit_training.md)** — Training pipeline
   - Data preparation
   - Model architecture
   - Training loop
   - Hyperparameters

5. **[DiT Sampling](03_dit_sampling.md)** — Generation strategies
   - ODE solvers
   - Conditional generation
   - Quality vs speed

### Advanced Topics

6. **[Comprehensive Tutorial](../diffusion/DiT/diffusion_transformer.md)** — Deep dive
   - Alternative backbones
   - Biology applications
   - State-space models

7. **[Time Embeddings](../diffusion/DiT/time_embeddings_explained.md)** — Conditioning mechanisms
   - Sinusoidal embeddings
   - AdaLN details
   - FiLM modulation

---

## Comparison with Other Methods

### DiT vs DDPM

| Aspect | DDPM | DiT + Rectified Flow |
|--------|------|----------------------|
| **Architecture** | U-Net | Transformer |
| **Objective** | Noise prediction | Velocity prediction |
| **Training** | Noise schedule needed | Simple regression |
| **Sampling** | 1000 steps (SDE) | 20-50 steps (ODE) |
| **Conditioning** | Concatenation/FiLM | AdaLN/Cross-attention |
| **Flexibility** | Images mainly | Any modality |

### DiT vs Flow Matching (U-Net)

| Aspect | Flow Matching + U-Net | DiT + Rectified Flow |
|--------|----------------------|----------------------|
| **Objective** | Same (velocity) | Same (velocity) |
| **Architecture** | Convolutional | Transformer |
| **Scaling** | Limited | Excellent |
| **Conditioning** | Moderate | Excellent |
| **Speed** | Fast convolutions | Slower attention |

**Key insight**: DiT is an architectural choice, orthogonal to the training objective.

---

## Key Takeaways

### Conceptual

1. **DiT = Transformer architecture** for diffusion/flow models
2. **Rectified flow = simple objective** (velocity regression)
3. **Together = modern stack** for state-of-the-art generation
4. **Tokenization enables** modality-agnostic modeling

### Practical

1. **Training is simple**: Regression on $v = x_1 - x_0$
2. **Sampling is fast**: 20-50 ODE steps
3. **Conditioning is easy**: Tokens or AdaLN
4. **Scales well**: Proven to billions of parameters

### For Biology

1. **Flexible representation**: Genes, cells, perturbations
2. **Global interactions**: Attention captures dependencies
3. **Conditional generation**: Model interventions
4. **Active research**: Best practices still emerging

---

## Related Documentation

### Prerequisites
- [Flow Matching](../flow_matching/) — Rectified flow theory
- [DDPM](../DDPM/) — Discrete diffusion models
- [SDE](../SDE/) — Continuous-time perspective

### Advanced Topics
- [Latent Diffusion](../latent_diffusion/) — Diffusion in latent space
- [JEPA](../JEPA/) — Joint-embedding predictive architectures

### Code Examples
- `notebooks/diffusion/` — Interactive tutorials
- `examples/` — Production scripts

---

## References

### Key Papers

**DiT**:

- Peebles & Xie (2023): "Scalable Diffusion Models with Transformers"

**Rectified Flow**:

- Liu et al. (2022): "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- Liu et al. (2023): "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation"

**Transformers**:

- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words" (ViT)
- Vaswani et al. (2017): "Attention is All You Need"

**Conditioning**:

- Perez et al. (2018): "FiLM: Visual Reasoning with a General Conditioning Layer"

### Modern Implementations

- **Stable Diffusion 3**: DiT-based text-to-image
- **Sora**: DiT for video generation
- **Hugging Face Diffusers**: DiT implementations
- **OpenAI**: DALL-E 3

---

## Summary

**Diffusion Transformers (DiT)** combined with **rectified flow** represent the modern approach to generative modeling:

**Architecture**: Transformers replace U-Nets
- Global attention from the start
- Flexible tokenization
- First-class conditioning

**Objective**: Rectified flow simplifies training
- Direct velocity regression
- No noise schedules
- Fast ODE sampling

**Result**: State-of-the-art generation
- Images, video, audio
- Scalable to billions of parameters
- Emerging applications in biology

**The modern stack**:
```
Rectified Flow + DiT + AdaLN = Powerful, flexible generation
```

This combination has become the foundation for cutting-edge generative models and is particularly promising for computational biology applications.
