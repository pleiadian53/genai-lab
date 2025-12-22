# Variational Autoencoders: A Visual Derivation

A 10-slide presentation on the ELBO derivation and VAE intuition.

---

## Slide 1: Latent Variable Model & Notation

**Goal**: Model data $x$ as generated from latent variables $z$.

$$
p_\theta(x, z) = p_\theta(x | z) \cdot p(z)
$$

$$
p_\theta(x) = \int p_\theta(x, z) \, dz
$$

**Notation**:

| Symbol | Meaning |
|--------|---------|
| $x$ | Observed data |
| $z$ | Latent variable |
| $p(z)$ | Prior over latents, e.g., $\mathcal{N}(0, I)$ |
| $p_\theta(x|z)$ | Decoder / likelihood |
| $q_\phi(z|x)$ | Encoder / approximate posterior |

---

## Slide 2: Why Direct Likelihood is Hard

**The problem**: Computing $p_\theta(x)$ requires an intractable integral.

$$
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz
$$

⚠️ **This integral is the bottleneck.**

The true posterior $p_\theta(z|x)$ is complex and unknown.

**Key idea**: Instead of solving exactly, introduce a tractable approximation.

---

## Slide 3: Variational Approximation

**Solution**: Learn an encoder network to approximate the true posterior.

$$
q_\phi(z|x) \approx p_\theta(z|x)
$$

**Question**: How do we use $q_\phi$ to get a tractable learning objective?

---

## Slide 4: Rewriting the Likelihood

**Step 1**: Multiply and divide by $q_\phi(z|x)$ inside the integral.

$$
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz
$$

$$
= \log \int q_\phi(z|x) \cdot \frac{p_\theta(x, z)}{q_\phi(z|x)} \, dz
$$

This doesn't change the value—it just rewrites the integral in a useful form.

---

## Slide 5: Expectation Under the Encoder

**Step 2**: Recognize this as an expectation under $q_\phi(z|x)$.

$$
= \log \mathbb{E}_{q_\phi(z|x)} \left[ \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]
$$

**Why this matters**: We can now estimate this by sampling from $q_\phi(z|x)$.

---

## Slide 6: Lower Bounding the Log-Likelihood

**Step 3**: Apply Jensen's inequality.

$$
\log \mathbb{E}[X] \geq \mathbb{E}[\log X]
$$

Therefore:

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x, z) - \log q_\phi(z|x) \right]
$$

**This is where "lower bound" comes from.**

---

## Slide 7: Decomposing the Bound

**Step 4**: Expand the joint distribution $p_\theta(x, z) = p_\theta(x|z) \cdot p(z)$.

$$
= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \| p(z))
$$

| Term | Interpretation |
|------|----------------|
| $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ | **Reconstruction**: How well does the decoder explain $x$? |
| $\mathrm{KL}(q_\phi(z|x) \| p(z))$ | **Regularization**: Keep encoder close to prior |

---

## Slide 8: Evidence Lower Bound (ELBO)

**The VAE objective**:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \| p(z))
$$

**Maximizing ELBO simultaneously**:

1. Improves reconstruction quality
2. Keeps the latent distribution well-structured

---

## Slide 9: Why the KL Term Matters

**Three reasons to keep $q(z|x)$ close to $p(z)$**:

| Reason | Explanation |
|--------|-------------|
| **Enables generation** | At test time, we sample $z \sim p(z)$. If encoder pushes latents away from prior, decoder never sees those regions. |
| **Prevents memorization** | Without KL, encoder could assign each datapoint a unique, sharp latent code (lookup table). |
| **Smooth latent space** | KL encourages nearby $z$'s to produce similar $x$'s, enabling interpolation. |

**Intuition**: "Force all data to live in the same coordinate system."

---

## Slide 10: What Does ELBO Really Optimize?

**The gap between true likelihood and ELBO**:

$$
\log p_\theta(x) - \mathcal{L} = \mathrm{KL}(q_\phi(z|x) \| p_\theta(z|x))
$$

**Interpretation**:

- The gap is the KL between approximate and true posterior
- Gap $\geq 0$ always (KL is non-negative)
- When $q_\phi = p_\theta$, gap is zero

**Maximizing ELBO simultaneously**:

1. Learns the generative model $p_\theta$
2. Performs approximate Bayesian inference

---

## Summary: The VAE Story

```
1. Goal         →  Maximize log p(x)
2. Obstacle     →  Intractable integral over z
3. Solution     →  Introduce approximate posterior q(z|x)
4. Derivation   →  One inequality (Jensen's), one decomposition
5. Result       →  ELBO = Reconstruction − KL
```

**The ELBO is a principled objective that balances**:

- Explaining the data (reconstruction)
- Maintaining a usable latent space (regularization)

---

## References

- Kingma & Welling (2014) — "Auto-Encoding Variational Bayes"
- [VAE-01-overview.md](VAE-01-overview.md) — Detailed theory
- [VAE-02-elbo.md](VAE-02-elbo.md) — Step-by-step derivation
