# Clarification: Dimensionality of Brownian Motion $w(t)$

## Key Point

**$w(t)$ and $x(t)$ live in the same space: $\mathbb{R}^d$**

## Details

In the SDE formulation:

$$
dx(t) = f(x(t), t)\,dt + g(t)\,dw(t)
$$

- **$x(t) \in \mathbb{R}^d$**: The state vector (e.g., flattened image, gene expression vector)
- **$w(t) \in \mathbb{R}^d$**: A **d-dimensional Wiener process** (Brownian motion)

## Why This Matters

The Brownian motion $w(t)$ must be d-dimensional because:

1. The differential $dw(t)$ behaves like:

   $$

   dw(t) \sim \sqrt{dt} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_d)
   $$

   where $I_d$ is the $d \times d$ identity matrix, so $\varepsilon \in \mathbb{R}^d$.

2. The noise term $g(t) \, dw(t)$ is added directly to $x(t)$:
   - Both must have the same dimension for the addition to be valid
   - The actual noise added to the image is $g(t) \, dw(t)$, not just $dw(t)$

3. For an image diffusion model:
   - If the image is $H \times W \times C$ pixels, then $d = H \times W \times C$
   - $w(t)$ is a $d$-dimensional random walk
   - Each component of $w(t)$ is an independent 1D Brownian motion

## Intuition

- **$w(t)$**: The underlying d-dimensional random walk (source of randomness)
- **$g(t)$**: A scalar (or matrix) that scales the noise magnitude
- **$g(t) \, dw(t)$**: The actual noise increment added to the image at time $t$

Think of it as: $w(t)$ provides the "direction and magnitude" of randomness in d-dimensional space, and $g(t)$ controls how much of that randomness gets injected into the image.

## Connection to the Document

This clarification relates to section 2.3 (Brownian Motion) in `sde_formulation.md`. The document could be more explicit that $w(t)$ is d-dimensional to match $x(t)$.

