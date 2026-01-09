# Classifier-Free Guidance: Conditional Generation Without Classifiers

## Overview

**Classifier-free guidance** is a technique for **conditional generation** in diffusion models that:
- Enables high-quality conditional sampling (e.g., text-to-image, class-conditional)
- **Does not require a separate classifier network**
- Uses a single model trained for both conditional and unconditional generation
- Provides a guidance scale to control the trade-off between diversity and fidelity

This technique has become the **standard approach** for conditional diffusion models, powering systems like Stable Diffusion, DALL-E 2, and Imagen.

---

## The Problem: Conditional Generation

### Goal

Generate samples from $p(x|c)$ where:
- $x$ is the data (e.g., image)
- $c$ is the condition (e.g., text prompt, class label, image features)

### Naive Approach Doesn't Work

Simply training $s_\theta(x_t, t, c) \approx \nabla_x \log p_t(x|c)$ and sampling doesn't give high-quality results because:
- The model learns to average over all modes of $p(x|c)$
- Generated samples are **generic** and lack detail
- No control over how strongly to follow the condition

**Example**: For prompt "a red car", the model might generate a blurry, generic car that's somewhat reddish.

---

## Prior Art: Classifier Guidance

### The Classifier Guidance Approach (Dhariwal & Nichol, 2021)

**Idea**: Use Bayes' rule to incorporate a classifier into the score:

$$
\nabla_x \log p_t(x|c) = \nabla_x \log p_t(x) + \nabla_x \log p_t(c|x)
$$

**Implementation**:
1. Train an unconditional diffusion model: $s_\theta(x_t, t) \approx \nabla_x \log p_t(x)$
2. Train a separate time-dependent classifier: $p_\phi(c|x_t, t)$
3. At sampling, use the guided score:

$$
\tilde{s}(x_t, t, c) = s_\theta(x_t, t) + w \cdot \nabla_{x_t} \log p_\phi(c|x_t, t)
$$

where $w$ is the **guidance scale**.

**With guidance scale** $w > 1$:

$$
\tilde{s}(x_t, t, c) = s_\theta(x_t, t) + w \cdot \nabla_{x_t} \log p_\phi(c|x_t, t) = \nabla_x \log p_t(x) + w \cdot \nabla_x \log p_t(c|x)
$$

This can be rewritten as sampling from:

$$
p_t(x|c)^w \cdot p_t(x)^{1-w} \propto p_t(c|x)^w \cdot p_t(x)
$$

**Effect**: Amplifies the classifier gradient, making samples more strongly conditioned but less diverse.

### Problems with Classifier Guidance

1. **Need to train a separate classifier** on noisy images $x_t$ at all timesteps
2. **Classifier can be adversarially fooled**: Model may generate images that fool the classifier rather than truly match the condition
3. **Inefficient**: Two networks instead of one
4. **Limited to differentiable conditions**: Can't easily use discrete or structured conditions

---

## Classifier-Free Guidance: The Elegant Solution

### Core Idea (Ho & Salimans, 2021)

**Train a single model** that learns both:
- **Conditional score**: $\nabla_x \log p_t(x|c)$
- **Unconditional score**: $\nabla_x \log p_t(x)$

**Key trick**: Use the same network but randomly **drop the condition** during training.

### Training Procedure

**Dataset**: Pairs $(x, c)$ where $c$ is the condition (text, class, etc.)

**Training**:
1. With probability $p_{\text{uncond}}$ (typically 10-20%), replace $c$ with a **null token** $\varnothing$
2. Train the score network $s_\theta(x_t, t, c)$ on both:
   - Conditional examples: $(x_t, t, c)$
   - Unconditional examples: $(x_t, t, \varnothing)$

**Loss** (standard score matching):

$$
\mathbb{E}_{t, x_0, \epsilon, c}\left[\left\| s_\theta(x_t, t, c) - \nabla_{x_t} \log p_t(x_t|x_0) \right\|^2\right]
$$

where we sample $c \sim p_{\text{data}}$ with probability $(1-p_{\text{uncond}})$ and $c = \varnothing$ with probability $p_{\text{uncond}}$.

### Sampling with Guidance

At generation time, we use **implicit classifier guidance** by combining conditional and unconditional scores:

$$
\boxed{\tilde{s}_\theta(x_t, t, c) = (1-w) \cdot s_\theta(x_t, t, \varnothing) + w \cdot s_\theta(x_t, t, c)}
$$

where $w$ is the **guidance scale**.

**Alternative formulation** (more common in practice):

$$
\boxed{\tilde{s}_\theta(x_t, t, c) = s_\theta(x_t, t, \varnothing) + w \cdot \left[s_\theta(x_t, t, c) - s_\theta(x_t, t, \varnothing)\right]}
$$

This makes it clearer that we're **amplifying the difference** between conditional and unconditional scores.

### Why This Works

**Mathematical justification**:

Recall that the conditional score can be decomposed:

$$
\nabla_x \log p_t(x|c) = \nabla_x \log p_t(x) + \nabla_x \log p_t(c|x)
$$

Rearranging:

$$
\nabla_x \log p_t(c|x) = \nabla_x \log p_t(x|c) - \nabla_x \log p_t(x)
$$

So:

$$
\nabla_x \log p_t(x|c) + w \cdot \nabla_x \log p_t(c|x) = \nabla_x \log p_t(x|c) + w \cdot [\nabla_x \log p_t(x|c) - \nabla_x \log p_t(x)]
$$

$$
= (1+w) \cdot \nabla_x \log p_t(x|c) - w \cdot \nabla_x \log p_t(x)
$$

**In classifier-free guidance**, we directly learn both terms, so:

$$
\tilde{s}_\theta(x_t, t, c) = s_\theta(x_t, t, \varnothing) + w \cdot [s_\theta(x_t, t, c) - s_\theta(x_t, t, \varnothing)]
$$

is equivalent to classifier guidance without needing the classifier!

---

## Intuition: What Does Guidance Do?

### The Guidance Scale $w$

| $w$ | Effect | Samples |
|-----|--------|---------|
| $w = 0$ | Pure unconditional: $\tilde{s} = s_\theta(x_t, t, \varnothing)$ | Diverse, generic |
| $w = 1$ | Pure conditional: $\tilde{s} = s_\theta(x_t, t, c)$ | Balanced |
| $w > 1$ | Amplified conditioning: Over-emphasize $c$ | High fidelity to $c$, less diverse |
| $w < 0$ | Negative guidance: Push away from $c$ | Opposite of $c$ |

### Visual Intuition

Think of the score function as a vector field pointing toward higher probability regions:

```
Unconditional score s(x_t, ∅):
  Points toward "generic realistic images"

Conditional score s(x_t, c):
  Points toward "realistic images matching c"

Guided score (w=2):
  Amplifies the difference, points toward "images that STRONGLY match c"
```

**Effect of $w > 1$**:
- Samples become **more aligned** with the condition
- But also **less diverse** (mode-seeking behavior)
- Trade-off: fidelity vs. diversity

### Example: Text-to-Image

**Prompt**: "A cat wearing a top hat"

- $w = 1$: Might generate a cat with a hat, somewhat detailed
- $w = 5$: Cat clearly wearing a fancy top hat, high detail, but less variety across samples
- $w = 10$: Very specific top-hat-wearing cat, but might lose overall image quality (over-saturated, artifacts)

**Typical values**: $w \in [1, 10]$ for text-to-image, with $w = 7-8$ being common.

---

## Implementation Details

### Training Code Structure

```python
class ConditionalScoreNetwork(nn.Module):
    def __init__(self, data_dim, hidden_dim, cond_dim, time_dim):
        super().__init__()
        # Network that takes data, time, and condition
        self.net = UNet(data_dim, hidden_dim, cond_dim, time_dim)
        # Learnable null token for unconditional generation
        self.null_embedding = nn.Parameter(torch.zeros(cond_dim))
    
    def forward(self, x_t, t, c, is_conditional):
        """
        Args:
            x_t: Noisy data [batch, data_dim]
            t: Time [batch]
            c: Condition [batch, cond_dim]
            is_conditional: Whether to use condition [batch] (bool)
        """
        # Replace condition with null token where is_conditional is False
        c_masked = torch.where(
            is_conditional[:, None], 
            c, 
            self.null_embedding[None, :]
        )
        return self.net(x_t, t, c_masked)


def train_step(model, x_0, c, p_uncond=0.1):
    """Single training step with classifier-free guidance."""
    batch_size = x_0.shape[0]
    
    # Sample timesteps
    t = torch.rand(batch_size) * T_max
    
    # Sample noise
    eps = torch.randn_like(x_0)
    
    # Forward diffusion
    x_t = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * eps
    
    # Randomly drop conditions
    is_conditional = torch.rand(batch_size) > p_uncond
    
    # Predict score
    score_pred = model(x_t, t, c, is_conditional)
    
    # True score (negative scaled noise for VP-SDE)
    score_true = -eps / (1 - alpha_bar_t).sqrt()
    
    # Loss
    loss = ((score_pred - score_true) ** 2).mean()
    
    return loss
```

### Sampling with Guidance

```python
def sample_with_cfg(model, condition, guidance_scale, num_steps=1000):
    """Sample using classifier-free guidance."""
    x_t = torch.randn(batch_size, data_dim)  # Start from noise
    
    dt = T_max / num_steps
    
    for i in reversed(range(num_steps)):
        t = torch.full((batch_size,), i * dt)
        
        # Get conditional and unconditional scores
        with torch.no_grad():
            # Conditional score
            s_cond = model(x_t, t, condition, is_conditional=True)
            
            # Unconditional score
            s_uncond = model(x_t, t, condition, is_conditional=False)
            
            # Guided score
            s_guided = s_uncond + guidance_scale * (s_cond - s_uncond)
        
        # Reverse SDE step (Euler-Maruyama)
        drift = -0.5 * beta(t) * (x_t + s_guided)
        diffusion = beta(t).sqrt()
        
        x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * dt.sqrt()
    
    return x_t
```

### Practical Tips

1. **Null token**: 
   - For text: Use a special `<UNCOND>` token or zero embedding
   - For class labels: Use a special "no class" index
   - For continuous conditions: Use zeros or learnable null embedding

2. **Unconditional probability**:
   - Typical: $p_{\text{uncond}} = 0.1$ (10% of training samples)
   - Too low: Poor unconditional generation, weak guidance
   - Too high: Poor conditional generation

3. **Guidance scale**:
   - Start with $w = 1$ (pure conditional)
   - Increase to $w = 5-10$ for stronger conditioning
   - Monitor diversity: Too high $w$ can collapse to a few modes

4. **Computational cost**:
   - Need **two forward passes** per sampling step (conditional + unconditional)
   - Can be mitigated with caching or distillation

---

## Variants and Extensions

### 1. Dynamic Guidance

Instead of fixed $w$, use time-dependent guidance:

$$
\tilde{s}_\theta(x_t, t, c) = s_\theta(x_t, t, \varnothing) + w(t) \cdot [s_\theta(x_t, t, c) - s_\theta(x_t, t, \varnothing)]
$$

**Intuition**: Early steps (high noise) benefit from less guidance; later steps (low noise) benefit from stronger guidance.

### 2. Multi-Conditional Guidance

For multiple conditions $c_1, c_2, \ldots$:

$$
\tilde{s} = s(x_t, t, \varnothing) + \sum_i w_i \cdot [s(x_t, t, c_i) - s(x_t, t, \varnothing)]
$$

**Example**: Combine text prompt + style reference + color palette.

### 3. Negative Prompting

Use $w < 0$ for certain conditions to **avoid** them:

$$
\tilde{s} = s(x_t, t, c_{\text{positive}}) + w_{\text{neg}} \cdot [s(x_t, t, \varnothing) - s(x_t, t, c_{\text{negative}})]
$$

**Example**: "A beautiful landscape" (positive) + avoid "blurry, low quality" (negative).

### 4. Guidance Distillation

Train a separate model to directly predict the guided score, eliminating the two-forward-pass cost:

$$
\tilde{s}_{\text{student}}(x_t, t, c) \approx s_\theta(x_t, t, \varnothing) + w \cdot [s_\theta(x_t, t, c) - s_\theta(x_t, t, \varnothing)]
$$

---

## Connection to Both DDPM and SDE Views

### DDPM View

In the discrete-time DDPM formulation, classifier-free guidance modifies the denoising step:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot [\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)]
$$

where $\epsilon_\theta$ predicts the noise instead of the score.

**Sampling**:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\tilde{\epsilon}_\theta(x_t, t, c)\right) + \sigma_t z
$$

### SDE View

In the continuous-time SDE formulation, guidance modifies the reverse-time SDE:

**Original reverse SDE**:

$$
dx = \left[-f(x, t) + g(t)^2 \nabla_x \log p_t(x|c)\right]dt + g(t)\,d\bar{w}
$$

**With classifier-free guidance**:

$$
dx = \left[-f(x, t) + g(t)^2 \tilde{s}_\theta(x_t, t, c)\right]dt + g(t)\,d\bar{w}
$$

where $\tilde{s}_\theta$ is the guided score.

**Key insight**: Both views are modifying the drift term by using an amplified score/noise prediction.

---

## Empirical Results

### From Original Paper (Ho & Salimans, 2021)

**ImageNet 128×128**:
- $w = 1$: FID = 12.6, Diversity = high
- $w = 2$: FID = 7.0, Diversity = medium
- $w = 4$: FID = 4.6, Diversity = low

**Trade-off**:
- Higher $w$ → Better FID (sample quality)
- Higher $w$ → Worse diversity (mode collapse)

### Modern Applications

**Stable Diffusion** (text-to-image):
- Default: $w = 7.5$
- Users can adjust from 1 to 20

**DALL-E 2, Imagen**:
- Heavy use of classifier-free guidance
- Enables high-fidelity text-to-image generation

---

## Summary

### Key Takeaways

1. **Classifier-free guidance enables conditional generation without a separate classifier**
   - Train one model for both conditional and unconditional generation
   - Randomly drop conditions during training

2. **Guidance scale $w$ controls fidelity vs. diversity**
   - $w = 1$: Balanced conditional generation
   - $w > 1$: Stronger conditioning, less diversity
   - Typical: $w \in [5, 10]$ for text-to-image

3. **Works in both DDPM and SDE views**
   - DDPM: Modify noise prediction
   - SDE: Modify score/drift term

4. **Has become the standard approach**
   - Simpler than classifier guidance
   - More flexible
   - Better results in practice

5. **Main cost: Two forward passes per step**
   - Can be mitigated with distillation
   - Worth it for generation quality

### When to Use

**Use classifier-free guidance when**:
- You want high-quality conditional generation
- You have paired data $(x, c)$
- You want control over conditioning strength
- You don't want to train a separate classifier

**Consider alternatives when**:
- Two forward passes are too expensive (use distillation)
- You have very limited conditional data (might not learn unconditional well)
- Conditions are complex or structured (might need specialized architectures)

---

## References

### Original Papers

- **Ho & Salimans (2021)**: ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2207.12598) (NeurIPS 2021 Workshop)
- **Dhariwal & Nichol (2021)**: ["Diffusion Models Beat GANs on Image Synthesis"](https://arxiv.org/abs/2105.05233) (introduced classifier guidance)

### Applications

- **Rombach et al. (2022)**: ["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752) (Stable Diffusion)
- **Ramesh et al. (2022)**: ["Hierarchical Text-Conditional Image Generation with CLIP Latents"](https://arxiv.org/abs/2204.06125) (DALL-E 2)
- **Saharia et al. (2022)**: ["Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding"](https://arxiv.org/abs/2205.11487) (Imagen)

---

## Related Documents

- **DDPM View**: See `docs/DDPM/` for discrete-time formulation
- **SDE View**: See `docs/SDE/01_diffusion_sde_view.md` for continuous-time perspective
- **Conditional Generation**: See `docs/DDPM/04_ddpm_extensions.md` (when available)
- **Score Networks**: See `docs/diffusion/score_network/` for architecture details
