# Diffusion Transformers (DiT): Overview

This document provides a high-level introduction to **Diffusion Transformers (DiT)** — the architectural shift from convolutional U-Nets to Transformers for generative modeling, particularly when combined with **rectified flow**.

---

## What is DiT?

**Diffusion Transformer (DiT)** is an **architectural choice**, not a new diffusion theory.

DiT uses a Transformer to parameterize the function learned in diffusion or flow-based models:

$$
v_\theta(x, t, c) \quad \text{(velocity for flow matching)}
$$

$$
\epsilon_\theta(x, t, c) \quad \text{(noise for DDPM)}
$$

$$
s_\theta(x, t) \quad \text{(score for score matching)}
$$

**Key insight**: The **objective** (what to learn) and the **architecture** (how to learn it) are orthogonal design choices.

---

## Why DiT Matters

### The Architectural Evolution

**Historical progression**:
1. **U-Net era** (2020-2022): Convolutional architectures dominated
2. **DiT era** (2023+): Transformers became the standard
3. **Modern stack**: DiT + Rectified Flow

### What Changed

| Aspect | U-Net | DiT |
|--------|-------|-----|
| **Inductive bias** | Spatial locality | Global attention |
| **Input format** | Fixed grids | Flexible tokens |
| **Conditioning** | Architectural changes needed | First-class via modulation |
| **Scaling** | Limited | Excellent |
| **Flexibility** | Images only | Any modality |

---

## The Core Idea: Grids → Tokens

**U-Net thinking**: Process images as spatial grids with local convolutions

**DiT thinking**: Represent inputs as sequences of tokens, process with global attention

**For images**:
1. Split image into patches (e.g., 16×16 pixels)
2. Flatten each patch into a vector
3. Embed into token space
4. Process with Transformer
5. Project back to image space

**For other domains**:
- Genes, cells, regions → tokens
- Time series → temporal tokens
- Latent representations → abstract tokens

---

## DiT + Rectified Flow: The Modern Stack

### Why This Combination Works

**Rectified Flow** provides:
- Simple regression target: $v = x_1 - x_0$
- Straight paths in data space
- Fast ODE sampling
- No density assumptions

**DiT** provides:
- Global context via self-attention
- Flexible conditioning via modulation
- Scalability to large models
- Modality-agnostic architecture

**Together**:
```
Simple objective + Powerful architecture = State-of-the-art generation
```

### Key Components

**1. Tokenization**: Convert input to sequence
```
Image → Patches → Tokens
```

**2. Time Conditioning**: Adaptive LayerNorm (AdaLN)
```
t → TimeEmbed → (γ, β) → Modulate features
```

**3. Self-Attention**: Global dependencies
```
Tokens → Attention → Updated tokens
```

**4. Output Projection**: Map back to target space
```
Tokens → Linear → Velocity field
```

---

## Training Objective

### Rectified Flow Loss

**Standard form**:
$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left[ \left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2 \right]
$$

where:
- $x_0 \sim p_{\text{data}}$ (real data)
- $x_1 \sim \mathcal{N}(0, I)$ (noise)
- $x_t = t x_1 + (1-t) x_0$ (linear interpolation)
- $v_\theta$ is the DiT network

**With conditioning**:
$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t, c} \left[ \left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|^2 \right]
$$

### Why This is Simple

**Compared to DDPM**:
- No noise schedules to tune
- No variance parameterization
- Direct regression target

**Compared to score matching**:
- No score function computation
- No Langevin dynamics
- Deterministic sampling via ODE

---

## Sampling Process

### ODE Integration

**Forward ODE** (noise → data):
$$
\frac{dx}{dt} = v_\theta(x, t, c)
$$

**Discretization** (Euler method):
```python
x = torch.randn(shape)  # Start from noise
dt = 1.0 / num_steps

for k in range(num_steps):
    t = k * dt
    v = model(x, t, condition)
    x = x + v * dt

return x  # Generated sample
```

**Properties**:
- Deterministic (same noise → same output)
- Fast (20-50 steps typical)
- Straight paths (rectified flow)

---

## Why DiT Scales Better

### 1. Global Context is Native

**U-Net**: Needs deep pyramids to propagate information
- Downsample → process → upsample
- Limited receptive field at each layer
- Information bottleneck

**DiT**: Self-attention is global by default
- Every token attends to every other token
- No information bottleneck
- Direct long-range dependencies

### 2. Flexible Input Shapes

**U-Net**: Fixed grid sizes
- Must pad/crop to specific resolutions
- Awkward for variable-size inputs
- Hard to batch different sizes

**DiT**: Variable-length sequences
- Different number of tokens per sample
- Batch with masking/packing
- Natural for heterogeneous data

### 3. First-Class Conditioning

**U-Net**: Conditioning requires architectural changes
- Concatenate channels
- Add FiLM layers
- Modify skip connections

**DiT**: Conditioning is built-in
- Add condition tokens (cross-attention)
- Modulate with AdaLN
- No architectural surgery

---

## Applications

### Images
- **Stable Diffusion 3**: DiT backbone
- **DALL-E 3**: Transformer-based
- **Imagen**: Cascaded DiT

### Videos
- **Sora**: DiT for video generation
- **Goku**: Efficient video DiT

### Beyond Vision
- **Audio**: AudioLDM, MusicGen
- **Molecules**: Protein structure generation
- **Robotics**: Trajectory generation
- **Biology**: Gene expression, cell states

---

## Key Advantages

### Theoretical

1. **Unified framework**: Same architecture for all modalities
2. **Scalability**: Proven to scale to billions of parameters
3. **Interpretability**: Attention maps show what model focuses on

### Practical

1. **Training stability**: Rectified flow is well-behaved
2. **Fast sampling**: 20-50 steps vs 1000 for DDPM
3. **Easy conditioning**: Add tokens or modulation
4. **Transfer learning**: Pretrained transformers can be adapted

### Engineering

1. **Infrastructure**: Leverage existing Transformer tools
2. **Optimization**: Well-understood training dynamics
3. **Debugging**: Attention visualization helps
4. **Deployment**: Standard Transformer serving

---

## Comparison: U-Net vs DiT

| Aspect | U-Net | DiT |
|--------|-------|-----|
| **Architecture** | Convolutional | Transformer |
| **Receptive field** | Local → Global | Global from start |
| **Input format** | Fixed grids | Flexible tokens |
| **Conditioning** | Concatenation/FiLM | AdaLN/Cross-attention |
| **Scaling** | Limited | Excellent |
| **Speed** | Fast convolutions | Slower attention |
| **Memory** | Moderate | Higher |
| **Best for** | Images, fixed size | Any modality, variable size |

**Modern trend**: DiT is becoming the default for new models.

---

## DiT for Computational Biology

### Why DiT is Promising for Biology

**Traditional challenges**:
- Gene expression: High-dimensional, unordered
- Cell states: Continuous, compositional
- Perturbations: Need flexible conditioning
- Time series: Variable-length trajectories

**DiT solutions**:
- **Tokens**: Genes, cells, regions, timepoints
- **Attention**: Capture gene-gene interactions
- **Conditioning**: Perturbations, cell types, experimental conditions
- **Flexibility**: Handle variable numbers of cells/genes

### Potential Applications

1. **Perturb-seq modeling**: Predict perturbation effects
2. **Cell state generation**: Sample from cell type distributions
3. **Trajectory inference**: Model developmental paths
4. **Counterfactual generation**: "What if" scenarios

### Open Questions

1. **Tokenization**: How to represent gene expression as tokens?
2. **Ordering**: Genes have no natural sequence — use set-based attention?
3. **Sparsity**: Many genes have zero expression — special handling?
4. **Latent space**: Better to work in latent space than raw expression?

**See**: Advanced topics in supplementary documents for deeper exploration.

---

## Document Organization

This DiT documentation is organized as follows:

### Core Series (Practical Workflow)

1. **00_dit_overview.md** (this document) — High-level introduction
2. **01_dit_foundations.md** — Architecture details, components
3. **02_dit_training.md** — How to train DiT + rectified flow
4. **03_dit_sampling.md** — How to generate samples

### Supplementary Documents (Deep Dives)

Located in `docs/diffusion/DiT/`:
- **diffusion_transformer.md** — Comprehensive tutorial with biology focus
- **time_embeddings_explained.md** — Deep dive on time conditioning
- Additional topics as needed

---

## Learning Path

### For Beginners

1. **Start here** (00_dit_overview.md) — Understand the big picture
2. **Flow matching basics** ([docs/flow_matching/](../flow_matching/)) — Learn rectified flow
3. **DiT foundations** (01_dit_foundations.md) — Architecture details
4. **Training guide** (02_dit_training.md) — Practical implementation

### For Implementers

1. **Training guide** (02_dit_training.md) — Complete training pipeline
2. **Sampling guide** (03_dit_sampling.md) — Generation strategies
3. **Supplementary docs** — Advanced techniques
4. **Code examples** — See `examples/` and `notebooks/`

### For Theorists

1. **Flow matching theory** ([docs/flow_matching/](../flow_matching/)) — Mathematical foundations
2. **DiT paper** (Peebles & Xie 2023) — Original architecture
3. **Supplementary docs** — Deep dives on specific topics
4. **SDE view** ([docs/SDE/](../SDE/)) — Continuous-time perspective

---

## Key Takeaways

### Conceptual

1. **DiT is an architecture**, not a new diffusion method
2. **Transformers replace U-Nets** for better scaling and flexibility
3. **Rectified flow + DiT** is the modern generative stack
4. **Tokenization enables** modality-agnostic generation

### Practical

1. **Training is simple**: Regression on velocity field
2. **Sampling is fast**: 20-50 ODE steps
3. **Conditioning is easy**: Add tokens or modulation
4. **Scales well**: Proven to billions of parameters

### For Biology

1. **Flexible representation**: Genes, cells, perturbations as tokens
2. **Global interactions**: Attention captures dependencies
3. **Conditional generation**: Model perturbation effects
4. **Open research**: Best tokenization strategies still being explored

---

## Next Steps

**Continue to**:
- [01_dit_foundations.md](01_dit_foundations.md) — Detailed architecture
- [02_dit_training.md](02_dit_training.md) — Training pipeline
- [03_dit_sampling.md](03_dit_sampling.md) — Sampling strategies

**Related documentation**:
- [Flow Matching](../flow_matching/) — Rectified flow theory
- [DDPM](../DDPM/) — Discrete diffusion models
- [SDE](../SDE/) — Continuous-time perspective

**Supplementary deep dives**:
- [diffusion_transformer.md](../diffusion/DiT/diffusion_transformer.md) — Comprehensive tutorial
- [time_embeddings_explained.md](../diffusion/DiT/time_embeddings_explained.md) — Time conditioning

---

## References

### Key Papers

- **Peebles & Xie (2023)**: "Scalable Diffusion Models with Transformers" (DiT)
- **Liu et al. (2022)**: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- **Dosovitskiy et al. (2020)**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
- **Perez et al. (2018)**: "FiLM: Visual Reasoning with a General Conditioning Layer"

### Modern Implementations

- **Stable Diffusion 3**: DiT-based text-to-image
- **Sora**: DiT for video generation
- **Hugging Face Diffusers**: DiT implementations

---

## Summary

**Diffusion Transformers (DiT)** represent the modern approach to generative modeling:

- **Replace U-Nets** with Transformers for better scaling
- **Combine with rectified flow** for simple, fast training
- **Enable flexible conditioning** via tokens and modulation
- **Scale to any modality** through tokenization

**The modern generative stack**:
```
Rectified Flow (objective) + DiT (architecture) + AdaLN (conditioning)
```

This combination has become the foundation for state-of-the-art generative models across images, video, audio, and emerging applications in biology.
