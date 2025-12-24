# Score Matching Documentation

This folder contains documentation on score matching techniques—methods for training models with intractable normalizing constants by matching the gradient of the log-density (the "score") rather than the density itself.

## Reading Order

### Foundations

1. **[Score Matching: The Core Objective](SM-core-objective.md)**  
   Explains the fundamental score matching objective for training energy-based models. Covers why the score function bypasses the partition function $Z_\theta$, the explicit vs. tractable objectives, and the integration-by-parts trick.

2. **[Fisher Score Matching for Likelihood-Free Inference](SM-fisher-score-matching.md)**  
   Tutorial walkthrough of *"Direct Fisher Score Estimation for Likelihood Maximization"* (Khoo et al., 2025). Extends score matching from data-space gradients to parameter-space gradients for simulation-based inference.

### Coming Soon

3. **Denoising Score Matching** — Practical variant using noisy data
4. **Sliced Score Matching** — Scalable approximation for high dimensions
5. **Connection to Diffusion Models** — How score matching underlies modern diffusion models

## Practical Considerations: ESM vs DSM

When implementing score matching for real applications (e.g., modeling gene expression data), you have two main options:

| Method | Objective | When to Use |
|--------|-----------|-------------|
| **Explicit SM (ESM)** | Squared norm + trace of Jacobian | Low-dimensional data; need exact objective |
| **Denoising SM (DSM)** | Squared error to noise gradient | High-dimensional data; practical default |

**Why DSM is often preferred:**

- ESM requires computing $\mathrm{tr}(\nabla_x s_\theta)$, which costs $O(d)$ backprop passes (or Hutchinson estimation)
- DSM only needs forward passes through the score network
- With Gaussian noise $\tilde{x} = x + \sigma\epsilon$, the target is analytic: $\nabla_{\tilde{x}} \log p(\tilde{x}|x) = -(\tilde{x} - x)/\sigma^2$

**Both learn the same thing:** the Stein score $\nabla_x \log p(x)$, just with different computational trade-offs.

See [Roadmap Stage 5](../ROADMAP.md) for implementation milestones.

## Key Concepts

| Concept | Symbol | Description |
|---------|--------|-------------|
| Stein score | $s(x)$ | Gradient of log-density w.r.t. data |
| Fisher score | $g(\theta)$ | Gradient of log-density w.r.t. parameters |
| Energy function | $E_\theta(x)$ | Defines density via $p_\theta(x) \propto \exp(-E_\theta(x))$ |
| Partition function | $Z_\theta$ | Intractable normalizing constant |

## Two Flavors of Score Matching

| Method | Estimates | Use Case |
|--------|-----------|----------|
| **Original Score Matching** | $\nabla_x \log p_\theta(x)$ | Training EBMs, diffusion models |
| **Fisher Score Matching** | Fisher score $\nabla_\theta \log p$ | Simulation-based inference, likelihood-free MLE |

Both use **integration-by-parts** to eliminate intractable terms.

## Connection to Other Topics

- **EBMs**: See [`../EBM/`](../EBM/) — Score matching is the primary training method for EBMs
- **VAEs**: See [`../VAE/`](../VAE/) — VAEs avoid the partition function via tractable encoder/decoder
- **Diffusion Models**: Build on denoising score matching across noise levels

## References

- Hyvärinen (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*
- Vincent (2011). *A Connection Between Score Matching and Denoising Autoencoders*
- Khoo et al. (2025). *Direct Fisher Score Estimation for Likelihood Maximization*
