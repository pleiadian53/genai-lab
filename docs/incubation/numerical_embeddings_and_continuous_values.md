# Numerical Embeddings and Continuous Values: From LLMs to Diffusion Models

## Overview

This document explores a fundamental challenge in modern deep learning: **how to represent continuous numerical values** in neural networks, particularly in contexts where tokenization breaks down. We'll examine:

1. The problem of numerical representation in language models
2. Recent research directions (2024-2026) for numerical embeddings
3. Connections to time embedding in diffusion models
4. Whether these techniques are relevant for computational biology applications

---

## The Core Problem

### Why Numbers Are Hard for LLMs

Language models excel at discrete tokens (words, subwords, characters) but struggle with **continuous numerical values**. Here's why:

**The Tokenization Problem:**
- Numbers like `3.14159` get tokenized as `["3", ".", "14", "159"]`
- This destroys numerical relationships: `3.14` and `3.15` are close numerically but may have completely different token sequences
- The model can't learn that `3.14159 ≈ π` or that `1000 > 999` in a meaningful way
- Arithmetic operations become nearly impossible: the model can't reliably compute `2 + 2 = 4`

**The Scale Problem:**
- Small numbers (`0.001`) and large numbers (`1,000,000`) are treated as unrelated tokens
- No inherent understanding of magnitude, order, or relationships
- Scientific notation (`1.23e-4`) is even more fragmented

**The Precision Problem:**
- Floating-point precision is lost in tokenization
- `3.141592653589793` and `3.141592653589794` might tokenize identically
- Fine-grained distinctions disappear

### Why This Matters

Numbers appear everywhere:
- **Scientific literature**: Measurements, statistics, experimental results
- **Code**: Variables, constants, calculations
- **Financial data**: Prices, quantities, percentages
- **Biological data**: Expression levels, concentrations, measurements
- **Time series**: Timestamps, durations, intervals

If LLMs can't handle numbers well, they can't truly understand these domains.

---

## Recent Research Directions (2024-2026)

### 1. Learned Numerical Embeddings

**Approach**: Treat numbers as a special token type with learned embeddings.

**Methods:**
- **Number-aware tokenization**: Split numbers into components (integer part, decimal part, exponent) and learn embeddings for each
- **Magnitude-aware embeddings**: Embed numbers in a way that preserves scale relationships
- **Hybrid approaches**: Combine tokenization with learned numerical representations

**Example Architecture:**
```python
class NumericalEmbedding(nn.Module):
    """Learned embedding for numerical values."""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Separate embeddings for different number components
        self.int_embedding = nn.Embedding(10000, embedding_dim)  # Integer part
        self.dec_embedding = nn.Embedding(1000, embedding_dim)   # Decimal part
        self.exp_embedding = nn.Embedding(100, embedding_dim)    # Exponent
    
    def forward(self, number):
        # Parse number into components
        int_part, dec_part, exp_part = self.parse_number(number)
        # Combine embeddings
        return self.int_embedding(int_part) + self.dec_embedding(dec_part) + self.exp_embedding(exp_part)
```

**Pros:**
- Fully learnable, can adapt to task
- Can capture domain-specific numerical patterns

**Cons:**
- Doesn't generalize to unseen numbers well
- Requires careful design of number decomposition
- Still loses some precision

### 2. Sinusoidal Numerical Embeddings

**Approach**: Use sinusoidal functions (similar to positional encoding) to represent numbers.

**Key Insight**: This is **exactly the same idea** as time embedding in diffusion models!

**Mathematical Form:**
For a number $n$, create an embedding:

$$
\gamma(n) = \begin{bmatrix}
\sin(\omega_1 n) \\
\cos(\omega_1 n) \\
\sin(\omega_2 n) \\
\cos(\omega_2 n) \\
\vdots \\
\sin(\omega_{d/2} n) \\
\cos(\omega_{d/2} n)
\end{bmatrix}
$$

where frequencies $\omega_i$ are chosen to span different scales:

$$
\omega_i = \frac{1}{10000^{2i/d}}
$$

**Why This Works:**
- **Bounded**: Values stay in $[-1, 1]$, training stable
- **Smooth**: Differentiable, allows interpolation
- **Multi-scale**: Different frequencies capture different magnitudes
- **Relative relationships**: Can represent that $n_1$ is close to $n_2$

**Connection to Time Embedding:**
This is **identical** to the time embedding used in diffusion models! Time $t$ and numbers $n$ are both continuous scalars that need high-dimensional representations.

### 3. Direct Numerical Encoding

**Approach**: Feed numbers directly as floating-point values, but with special processing.

**Methods:**
- **Normalization**: Scale numbers to a standard range (e.g., $[0, 1]$ or $[-1, 1]$)
- **Log scaling**: Use $\log(n + \epsilon)$ for wide-ranging values
- **Quantization**: Discretize into bins, then use embeddings
- **Feature engineering**: Extract magnitude, sign, precision as separate features

**Example:**
```python
def encode_number(n):
    """Direct encoding with normalization."""
    # Extract components
    sign = 1 if n >= 0 else -1
    magnitude = abs(n)
    
    # Log scale for wide range
    log_mag = torch.log(magnitude + 1e-8)
    
    # Normalize
    normalized = torch.tanh(log_mag / 10.0)  # Scale to [-1, 1]
    
    # Combine
    return torch.cat([sign * normalized, log_mag])
```

**Pros:**
- Simple, interpretable
- Preserves exact values (within precision)
- Can handle arbitrary ranges with normalization

**Cons:**
- Requires careful normalization
- May not capture complex numerical relationships
- Less expressive than learned embeddings

### 4. RoPE-Style Numerical Embeddings

**Approach**: Extend Rotary Position Embedding (RoPE) to numerical values.

**RoPE for Positions:**
RoPE rotates query/key vectors by an angle proportional to position, preserving relative distances.

**RoPE for Numbers:**
Apply similar rotation based on numerical value:

$$
\text{RoPE}(x, n) = \begin{bmatrix}
x_0 \cos(\omega n) - x_1 \sin(\omega n) \\
x_0 \sin(\omega n) + x_1 \cos(\omega n) \\
x_2 \cos(2\omega n) - x_3 \sin(2\omega n) \\
\vdots
\end{bmatrix}
$$

**Why This Might Work:**
- Preserves relative relationships: numbers close in value have similar rotations
- Can be applied to attention mechanisms
- Naturally handles different scales through frequency selection

**Status (2026)**: This is an active area of research, with some promising results for mathematical reasoning tasks.

### 5. Hybrid Approaches

**Approach**: Combine multiple methods.

**Example Architecture:**
```python
class HybridNumericalEmbedding(nn.Module):
    """Combines multiple numerical representation strategies."""
    
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(embedding_dim // 2)
        self.learned = nn.Linear(1, embedding_dim // 2)
        self.magnitude_embedding = nn.Embedding(100, embedding_dim // 4)
    
    def forward(self, n):
        # Sinusoidal component (multi-scale)
        sin_emb = self.sinusoidal(n)
        
        # Learned component (task-specific)
        learned_emb = self.learned(n.unsqueeze(-1))
        
        # Magnitude bucket (coarse scale)
        mag_bucket = self.get_magnitude_bucket(n)
        mag_emb = self.magnitude_embedding(mag_bucket)
        
        # Combine
        return torch.cat([sin_emb, learned_emb, mag_emb], dim=-1)
```

---

## Connection to Time Embedding in Diffusion Models

### The Parallel

Time embedding in diffusion models and numerical embeddings in LLMs solve **the same fundamental problem**: representing a continuous scalar in a way that neural networks can process effectively.

**Time Embedding (Diffusion):**
- Input: Continuous time $t \in [0, 1]$
- Challenge: Network needs to distinguish $t=0.5$ from $t=0.51$ and learn time-dependent behavior
- Solution: Sinusoidal embedding $\gamma(t)$ with multiple frequencies

**Numerical Embedding (LLMs):**
- Input: Continuous number $n \in \mathbb{R}$
- Challenge: Network needs to understand $n=3.14$ vs. $n=3.15$ and numerical relationships
- Solution: Similar sinusoidal embedding $\gamma(n)$ with multiple frequencies

### Why Sinusoidal Functions Work for Both

1. **Multi-scale representation**: Different frequencies capture different scales
   - Low frequencies: Coarse magnitude (thousands vs. millions)
   - High frequencies: Fine distinctions (3.14 vs. 3.15)

2. **Bounded and stable**: Values in $[-1, 1]$ prevent training instability

3. **Smooth interpolation**: Can represent values between training examples

4. **Relative relationships**: Embeddings for close values are similar

### Key Difference: Normalization

**Time embedding**: Usually $t \in [0, 1]$ (already normalized)

**Numerical embedding**: $n \in \mathbb{R}$ (unbounded, needs normalization)

For numerical embeddings, you typically need:
- **Log scaling**: $\log(|n| + \epsilon)$ for wide ranges
- **Normalization**: $\tanh(n / \text{scale})$ to bound values
- **Sign handling**: Separate representation for positive/negative

---

## Relevance to Computational Biology

### Where Continuous Values Appear

1. **Gene Expression**: Expression levels (TPM, FPKM, counts)
2. **Concentrations**: Protein concentrations, drug doses
3. **Measurements**: Cell counts, viability, size
4. **Time**: Time points in time-series experiments
5. **Coordinates**: Genomic positions, spatial coordinates
6. **Scores**: Prediction scores, p-values, fold changes

### Potential Applications

#### 1. Multi-modal Embeddings

If you're building a joint latent space (as discussed in `joint_latent_space_and_JEPA.md`), you need to embed:
- **Discrete**: Gene IDs, cell types, perturbations
- **Continuous**: Expression values, concentrations, time

Numerical embedding techniques could help create a unified representation.

#### 2. Time-Series Modeling

For time-series data (Perturb-seq, lineage tracing), you need to embed:
- **Time points**: $t \in [0, T]$
- **Expression values**: $x(t) \in \mathbb{R}^d$

Both benefit from sinusoidal embeddings, potentially sharing the same embedding architecture.

#### 3. Score Networks for Biological Data

In diffusion models for gene expression:
- **Time embedding**: For noise level $t$ in the diffusion process
- **Expression embedding**: For continuous expression values $x$

These could use similar sinusoidal embedding strategies, creating architectural consistency.

#### 4. Attention Mechanisms

If using Transformers for biological sequences:
- **Positional encoding**: For sequence position
- **Numerical encoding**: For expression values, scores, measurements

RoPE-style approaches could unify these.

---

## Practical Considerations

### When to Use Sinusoidal Embeddings

**Good for:**
- Continuous values that need smooth interpolation
- Values with known ranges (can normalize)
- When you want multi-scale representation
- When you need to generalize to unseen values

**Not ideal for:**
- Very sparse or discrete values (learned embeddings better)
- Values with complex, non-smooth relationships
- When exact precision is critical (may need direct encoding)

### Implementation Tips

1. **Normalize your inputs**: Scale to a reasonable range before embedding
2. **Choose frequencies carefully**: Too many high frequencies can cause instability
3. **Combine with learned components**: Hybrid approaches often work best
4. **Test interpolation**: Verify that close values have similar embeddings

### Example: Gene Expression Embedding

```python
class ExpressionEmbedding(nn.Module):
    """Embedding for gene expression values."""
    
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, expression):
        """
        Args:
            expression: Expression values [batch_size, num_genes]
        
        Returns:
            embeddings: [batch_size, num_genes, embedding_dim]
        """
        # Log transform (expression is typically log-normal)
        log_expr = torch.log(expression + 1e-8)
        
        # Normalize to reasonable range
        normalized = torch.tanh(log_expr / 10.0)
        
        # Sinusoidal embedding (same as time embedding!)
        return self.sinusoidal_embedding(normalized)
    
    def sinusoidal_embedding(self, x):
        """Same implementation as time embedding."""
        half_dim = self.embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
```

---

## Research Status (January 2026)

### Active Areas

1. **Mathematical reasoning**: Improving LLM performance on math problems
2. **Scientific literature**: Better understanding of numerical data in papers
3. **Code generation**: Handling numerical constants and calculations
4. **Multi-modal learning**: Combining numerical and textual information

### Recent Papers (2024-2025)

While specific 2026 papers are still emerging, the following directions are active:

- **Number-aware tokenization**: Better ways to split numbers into tokens
- **Magnitude-preserving embeddings**: Maintaining scale relationships
- **RoPE extensions**: Applying rotary embeddings to numerical values
- **Hybrid architectures**: Combining multiple representation strategies

### Open Questions

1. **Optimal frequency selection**: How to choose $\omega_i$ for different domains?
2. **Normalization strategies**: Best practices for different value ranges?
3. **Integration with attention**: How to effectively use numerical embeddings in Transformers?
4. **Domain adaptation**: Can embeddings learned on one domain transfer to another?

---

## Summary

### Key Takeaways

1. **The Problem**: Continuous numerical values are hard for token-based models (LLMs) because tokenization destroys numerical relationships.

2. **The Solution**: Sinusoidal embeddings (like time embedding) provide a natural way to represent continuous values with:
   - Multi-scale representation
   - Smooth interpolation
   - Bounded, stable values

3. **The Connection**: Time embedding in diffusion models and numerical embeddings in LLMs solve the same problem—representing continuous scalars effectively.

4. **The Relevance**: For computational biology, numerical embeddings could help:
   - Create unified representations of discrete and continuous biological data
   - Improve time-series modeling
   - Enhance score networks for biological data
   - Enable better attention mechanisms in biological Transformers

5. **The Future**: This is an active area of research, with promising directions including RoPE-style approaches and hybrid architectures.

### Next Steps

1. **Experiment with sinusoidal embeddings** for gene expression values in your diffusion models
2. **Compare** sinusoidal vs. learned vs. direct encoding for your specific biological data
3. **Explore RoPE-style approaches** if using Transformers for biological sequences
4. **Monitor research** in this area—it's rapidly evolving

---

## References

### Time Embedding (Diffusion Models)
- See: [`docs/diffusion/score_network/time_embedding_and_film.md`](../diffusion/score_network/time_embedding_and_film.md)

### Positional Encoding (Transformers)
- Vaswani et al. (2017). "Attention Is All You Need" - Original sinusoidal positional encoding

### RoPE (Rotary Position Embedding)
- Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"

### Numerical Representation in LLMs
- Active research area (2024-2026) - watch for recent papers on:
  - Number-aware tokenization
  - Magnitude-preserving embeddings
  - Mathematical reasoning improvements

---

## Related Documents

- **Time Embedding**: [`docs/diffusion/score_network/time_embedding_and_film.md`](../diffusion/score_network/time_embedding_and_film.md)
- **Joint Latent Spaces**: [`docs/incubation/joint_latent_space_and_JEPA.md`](joint_latent_space_and_JEPA.md)
- **Score Networks**: [`docs/diffusion/score_network/advanced_architectures.md`](../diffusion/score_network/advanced_architectures.md)
