# Time Embedding and FiLM: Conditioning Score Networks on Time

## Overview

This document explains two fundamental components used in modern score network architectures:
1. **Time Embedding**: How to represent the continuous time variable $t$ as input to neural networks
2. **FiLM (Feature-wise Linear Modulation)**: How to effectively condition network layers on time

These techniques are essential for building score networks that can accurately estimate $\nabla_x \log p_t(x)$ across different noise levels.

---

## Referenced From

- **Notebook**: [`notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb`](../../../notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb)
- **Implementation Note**: [`dev/notebooks/diffusion/02_sde_formulation/score_network_architecture.md`](../../../dev/notebooks/diffusion/02_sde_formulation/score_network_architecture.md)
- **Module**: `genailab.diffusion` (contains the production implementation)

---

## Time Embedding

### Why is Time Embedding Necessary?

#### The Core Problem

The score function we want to learn is:

$$
s_\theta(x, t) \approx \nabla_x \log p_t(x)
$$

This function depends on **both** $x$ (the noisy data) and $t$ (the noise level/time).

**Key Challenge**: Neural networks struggle to represent high-frequency functions when given raw scalar inputs. If we simply feed $t \in [0, 1]$ directly as a scalar, the network would have difficulty:
- Distinguishing between close time values (e.g., $t=0.5$ vs. $t=0.51$)
- Learning different noise characteristics at different scales
- Capturing the smooth but complex variation of the score function over time

#### The Solution: High-Dimensional Embeddings

Instead of using $t$ directly, we transform it into a **high-dimensional embedding** $\gamma(t) \in \mathbb{R}^d$ using sinusoidal functions. This gives the network:
1. **Multiple frequencies** to work with
2. **Better expressiveness** for representing time-dependent behavior
3. **Smooth interpolation** between time steps

### Sinusoidal Time Embedding

#### Mathematical Form

The sinusoidal embedding transforms a scalar $t$ into a vector of dimension $d$:

$$
\gamma(t) = \begin{bmatrix}
\sin(\omega_1 t) \\
\cos(\omega_1 t) \\
\sin(\omega_2 t) \\
\cos(\omega_2 t) \\
\vdots \\
\sin(\omega_{d/2} t) \\
\cos(\omega_{d/2} t)
\end{bmatrix}
$$

where the frequencies $\omega_i$ are chosen as:

$$
\omega_i = \frac{1}{10000^{2i/d}}
$$

This creates a spectrum of frequencies from low (slow variation) to high (fast variation).

#### Why Sinusoidal Functions?

1. **Bounded**: All values stay in $[-1, 1]$, which helps with training stability
2. **Smooth**: Differentiable everywhere, allowing smooth transitions
3. **Periodic**: Can represent cyclical patterns
4. **Linear Interpolation Property**: For any fixed offset $k$, $\gamma(t+k)$ can be expressed as a linear function of $\gamma(t)$

#### Implementation

Here's the typical PyTorch implementation:

```python
def time_embedding(self, t):
    """Sinusoidal time embedding.
    
    Args:
        t: Time tensor [batch_size]
    
    Returns:
        embedding: [batch_size, time_dim]
    """
    half_dim = self.time_dim // 2
    
    # Create frequencies: 1, 1/10000^(1/(half_dim-1)), ..., 1/10000
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    
    # emb has shape [half_dim], t[:, None] has shape [batch_size, 1]
    # Broadcasting gives [batch_size, half_dim]
    emb = t[:, None] * emb[None, :]
    
    # Concatenate sin and cos: [batch_size, time_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    return emb
```

**Step-by-step breakdown**:
1. `torch.arange(half_dim)` creates $[0, 1, 2, \ldots, \text{half\_dim}-1]$
2. `torch.exp(...  * -emb)` computes the frequencies $\omega_i$
3. `t[:, None] * emb[None, :]` broadcasts to compute $\omega_i t$ for all $i$ and all batch samples
4. Apply sin and cos and concatenate

#### Intuition: Frequency Spectrum

Think of time embedding as representing $t$ in multiple "resolutions":
- **Low frequencies** ($\sin(\omega_1 t)$ with small $\omega_1$): Capture slow changes over time
- **High frequencies** ($\sin(\omega_k t)$ with large $\omega_k$): Capture rapid changes

This is similar to a **Fourier basis**, giving the network multiple "channels" to understand time at different scales.

### Alternative: Learned Embeddings

Some architectures use **learned embeddings** instead:
- Treat $t$ as a discrete index (after discretization)
- Use an embedding table like in NLP: `nn.Embedding(num_timesteps, embedding_dim)`

**Pros**: Fully learnable, no prior assumptions
**Cons**: Doesn't generalize to unseen timesteps, requires discretization

Sinusoidal embeddings are preferred for continuous-time models like SDEs.

---

## FiLM: Feature-wise Linear Modulation

### What is FiLM?

**FiLM (Feature-wise Linear Modulation)** is a conditioning mechanism that modulates the features of a neural network layer using external information (in our case, time).

For each feature map $h$ in a layer, FiLM applies:

$$
\text{FiLM}(h | \gamma) = \gamma_{\text{scale}} \odot h + \gamma_{\text{shift}}
$$

where:
- $\gamma_{\text{scale}}$, $\gamma_{\text{shift}}$ are computed from the time embedding $\gamma(t)$
- $\odot$ denotes element-wise multiplication

This is also called **affine transformation** or **adaptive normalization**.

### Why is FiLM Effective?

#### 1. **Multiplicative and Additive Control**

FiLM provides two types of control:
- **Scale** ($\gamma_{\text{scale}} \odot h$): Controls the magnitude/importance of features
- **Shift** ($\gamma_{\text{shift}}$): Controls the bias/offset

This allows the network to completely change its behavior based on time:
- At $t \approx 1$ (high noise): Might amplify certain features
- At $t \approx 0$ (low noise): Might suppress the same features

#### 2. **Layer-wise Adaptation**

By applying FiLM at multiple layers, each layer can adapt differently to the time condition:
- **Early layers**: Might adjust based on coarse noise levels
- **Deep layers**: Might adjust based on fine-grained details

#### 3. **Better than Concatenation**

Compare to the simple approach of concatenating time embedding with input:

**Concatenation**:
```python
h = torch.cat([x, time_emb], dim=-1)
h = layer(h)
```

**FiLM**:
```python
h = layer(x)
scale, shift = compute_film_params(time_emb)
h = scale * h + shift
```

**Why FiLM is better**:
- Concatenation only affects the input; FiLM can modulate at any layer
- FiLM provides multiplicative control, which is more expressive
- FiLM allows the base features and time conditioning to be processed separately before combining

### FiLM Implementation

#### Basic Implementation

```python
class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, feature_dim, time_dim):
        super().__init__()
        
        # Project time embedding to scale and shift parameters
        self.film = nn.Linear(time_dim, 2 * feature_dim)
    
    def forward(self, h, time_emb):
        """
        Args:
            h: Features [batch_size, feature_dim]
            time_emb: Time embedding [batch_size, time_dim]
        
        Returns:
            modulated: [batch_size, feature_dim]
        """
        # Compute scale and shift
        film_params = self.film(time_emb)
        scale, shift = torch.chunk(film_params, 2, dim=-1)
        
        # Apply FiLM
        return scale * h + shift
```

#### Full Example: MLP with FiLM

```python
class ScoreNetworkWithFiLM(nn.Module):
    """Score network using FiLM conditioning."""
    
    def __init__(self, data_dim=2, hidden_dim=128, time_dim=32):
        super().__init__()
        
        self.time_dim = time_dim
        
        # Main network layers
        self.layer1 = nn.Linear(data_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, data_dim)
        
        # FiLM layers (one per hidden layer)
        self.film1 = nn.Linear(time_dim, 2 * hidden_dim)
        self.film2 = nn.Linear(time_dim, 2 * hidden_dim)
        self.film3 = nn.Linear(time_dim, 2 * hidden_dim)
        
        self.act = nn.SiLU()
    
    def time_embedding(self, t):
        """Sinusoidal time embedding."""
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, t):
        """
        Args:
            x: Data [batch_size, data_dim]
            t: Time [batch_size]
        
        Returns:
            score: [batch_size, data_dim]
        """
        # Get time embedding
        t_emb = self.time_embedding(t)
        
        # Layer 1
        h = self.layer1(x)
        scale, shift = torch.chunk(self.film1(t_emb), 2, dim=-1)
        h = self.act(scale * h + shift)
        
        # Layer 2
        h = self.layer2(h)
        scale, shift = torch.chunk(self.film2(t_emb), 2, dim=-1)
        h = self.act(scale * h + shift)
        
        # Layer 3
        h = self.layer3(h)
        scale, shift = torch.chunk(self.film3(t_emb), 2, dim=-1)
        h = self.act(scale * h + shift)
        
        # Output (no FiLM on final layer)
        return self.output(h)
```

### FiLM in U-Net Architecture

FiLM is particularly powerful in U-Net architectures used for image generation:

```python
class UNetBlock(nn.Module):
    """U-Net block with FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # FiLM parameters for each conv layer
        self.film1 = nn.Linear(time_dim, 2 * out_channels)
        self.film2 = nn.Linear(time_dim, 2 * out_channels)
        
        self.act = nn.SiLU()
    
    def forward(self, x, time_emb):
        """
        Args:
            x: Image features [batch, in_channels, H, W]
            time_emb: Time embedding [batch, time_dim]
        
        Returns:
            features: [batch, out_channels, H, W]
        """
        # First conv
        h = self.conv1(x)
        
        # FiLM conditioning
        scale, shift = torch.chunk(self.film1(time_emb), 2, dim=1)
        # Reshape for broadcasting: [batch, out_channels, 1, 1]
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.act(scale * h + shift)
        
        # Second conv
        h = self.conv2(h)
        scale, shift = torch.chunk(self.film2(time_emb), 2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.act(scale * h + shift)
        
        return h
```

**Note**: For convolutional layers, we reshape the scale/shift to `[batch, channels, 1, 1]` to broadcast across spatial dimensions.

---

## FiLM vs. Other Conditioning Methods

### Comparison Table

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Concatenation** | Concatenate time embedding with input | Simple, always works | Only affects input; weak conditioning |
| **Additive** | Add time embedding to features: $h + \gamma$ | Simple | Limited expressiveness; no scaling |
| **FiLM** | Affine transform: $\gamma_s \odot h + \gamma_b$ | Strong conditioning; layer-wise control | More parameters |
| **Attention** | Cross-attention between features and time | Very expressive | Computationally expensive |
| **Adaptive Group Norm** | Normalize then apply FiLM | Combines normalization benefits | Requires batch statistics |

### When to Use FiLM

FiLM is the **standard choice** for diffusion models because:
1. **Strong conditioning**: Both multiplicative and additive
2. **Efficient**: Linear projection, no complex operations
3. **Proven effectiveness**: Used in DDPM, Stable Diffusion, etc.
4. **Layer-wise control**: Can adapt behavior at every layer

---

## Putting It All Together: Time Conditioning Pipeline

Here's the complete flow of how time information flows through a score network:

```
Input: (x, t)
    ↓
1. Time Embedding: t → γ(t) ∈ ℝ^d
   - Transform scalar to high-dimensional representation
   - Use sinusoidal functions for multiple frequencies
    ↓
2. Feature Processing: x → h
   - Pass data through network layers
   - Extract features
    ↓
3. FiLM Conditioning: h, γ(t) → h'
   - Compute scale and shift from γ(t)
   - Modulate features: h' = scale ⊙ h + shift
   - Apply at multiple layers
    ↓
4. Output: h' → ∇_x log p_t(x)
   - Final linear layer
   - Produces score estimate
```

### Why This Works

**Time Embedding** gives the network a rich representation of when we are in the diffusion process:
- Early times ($t \approx 0$): Low noise, need to model fine details
- Late times ($t \approx 1$): High noise, model only coarse structure

**FiLM** allows the network to adapt its processing based on time:
- Different layers can learn different time-dependent behaviors
- Multiplicative control allows complete feature modulation
- The network learns to "turn on/off" different feature detectors based on noise level

---

## Advanced Topics

### 1. Adaptive Group Normalization (AdaGN)

Combines Group Normalization with FiLM:

$$
\text{AdaGN}(h | \gamma) = \gamma_s \odot \frac{h - \mu}{\sigma} + \gamma_b
$$

where $\mu$, $\sigma$ are computed per group of channels.

**Benefits**: 
- Normalization helps training stability
- FiLM provides conditioning
- Used in Stable Diffusion

### 2. Time-dependent Skip Connections

In U-Net architectures, skip connections can also be modulated:

```python
# Standard skip connection
h = h_down + h_up

# Time-modulated skip connection
scale, shift = compute_film(time_emb)
h = scale * h_down + h_up + shift
```

### 3. Multi-scale Time Embeddings

For hierarchical models, different scales can use different time embeddings:

```python
# Coarse scale: low frequencies
t_emb_coarse = time_embedding(t, max_freq=100)

# Fine scale: high frequencies  
t_emb_fine = time_embedding(t, max_freq=10000)
```

---

## Practical Tips

### 1. Time Embedding Dimension

**Typical choices**: 32, 64, 128, 256

**Rule of thumb**: 
- Simple tasks (2D toy data): 32-64
- Image generation: 128-256
- Higher dimensions give more expressiveness but add parameters

### 2. Where to Apply FiLM

**Common patterns**:
- After each convolution/linear layer
- Before activation functions
- Not on the final output layer (usually)

### 3. Initialization

Initialize FiLM layers to output $(1, 0)$ initially:
```python
# Initialize to identity transformation
self.film.weight.data.zero_()
self.film.bias.data.copy_(torch.tensor([1.0] * hidden_dim + [0.0] * hidden_dim))
```

This makes training more stable initially.

### 4. Debugging Time Conditioning

To verify time conditioning is working:
```python
# Test: Does output change with time?
x = torch.randn(1, 2)
t1 = torch.tensor([0.1])
t2 = torch.tensor([0.9])

out1 = model(x, t1)
out2 = model(x, t2)

print(f"Output difference: {(out1 - out2).abs().mean()}")
# Should be substantial (>0.1 typically)
```

---

## Summary

### Time Embedding
- **Purpose**: Transform scalar time to high-dimensional representation
- **Method**: Sinusoidal functions at multiple frequencies
- **Benefit**: Allows network to represent complex time-dependent behavior

### FiLM
- **Purpose**: Condition network features on time
- **Method**: Affine transformation (scale + shift)
- **Benefit**: Strong, layer-wise adaptation to time

### Together
They form the backbone of modern score network architectures, enabling the network to accurately estimate $\nabla_x \log p_t(x)$ across all noise levels.

---

## Further Reading

- **Original FiLM Paper**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
- **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **Transformer Positional Encoding**: Vaswani et al., "Attention Is All You Need" (2017) - similar sinusoidal embedding idea
- **U-Net with Time Conditioning**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)

---

## Back References

- **Where this is used**: [`notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb`](../../../notebooks/diffusion/02_sde_formulation/02_sde_formulation.ipynb)
- **Related concepts**: 
  - [Score Network Architecture](../../../dev/notebooks/diffusion/02_sde_formulation/score_network_architecture.md)
  - [Forward Process Derivation](../forward_process_derivation.md)
  - [Training Loss and Denoising](../../../dev/notebooks/diffusion/02_sde_formulation/training_loss_and_denoising.md)
- **Related topic**: [Numerical Embeddings and Continuous Values](../../../docs/incubation/numerical_embeddings_and_continuous_values.md) - Explores how similar sinusoidal embedding techniques are used for numerical values in LLMs and their relevance to time embedding