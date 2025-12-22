# The ELBO: Derivation and Intuition

This document expands on the Evidence Lower Bound (ELBO), the central objective in Variational Autoencoders.

---

## Notation Reference

| Symbol | Name | Description |
|--------|------|-------------|
| $x$ | Data | Observed data point (e.g., gene expression vector) |
| $z$ | Latent | Unobserved latent variable |
| $\theta$ | Decoder parameters | Weights of the generative model |
| $\phi$ | Encoder parameters | Weights of the inference network |
| $p_\theta(x \mid z)$ | Likelihood | Probability of data given latent (decoder) |
| $p(z)$ | Prior | Prior distribution over latents, typically $\mathcal{N}(0, I)$ |
| $p_\theta(x, z)$ | Joint | Joint distribution $= p_\theta(x \mid z) \cdot p(z)$ |
| $p_\theta(x)$ | Marginal likelihood | Evidence; what we want but can't compute |
| $q_\phi(z \mid x)$ | Approximate posterior | Encoder's guess at $p_\theta(z \mid x)$ |
| $\mathrm{KL}(\cdot \| \cdot)$ | KL divergence | Measures "distance" between distributions |

---

## 1. The ELBO Equation

**In words**: The log-probability of the data is at least as large as the expected reconstruction quality minus the KL penalty.

**In math**:

$$
\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction term}} - \underbrace{\mathrm{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization term}}
$$

**Explanation**:

- **Left side**: $\log p_\theta(x)$ is the log marginal likelihood (or "evidence"). This is what we *want* to maximize—it measures how well our model explains the data.

- **Right side**: The ELBO (Evidence Lower BOund). Since we can't compute the left side directly, we maximize this lower bound instead.

- **Reconstruction term**: $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ asks: "If I sample latents from my encoder $q_\phi(z|x)$, how well does my decoder $p_\theta(x|z)$ reconstruct the original data?"

- **KL term**: $\mathrm{KL}(q_\phi(z|x) \| p(z))$ measures how far the encoder's distribution is from the prior. It penalizes encoders that stray too far from $\mathcal{N}(0, I)$.

---

## 2. Deriving the ELBO (Step by Step)

### Step 1: Start with the marginal likelihood

**In words**: The probability of data $x$ is obtained by integrating over all possible latent values $z$.

**In math**:

$$
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz
$$

**Explanation**: This integral is intractable for deep networks because we'd need to evaluate the decoder for every possible $z$.

---

### Step 2: Introduce the approximate posterior

**In words**: Multiply and divide by $q_\phi(z|x)$ inside the integral—this doesn't change the value.

**In math**:

$$
= \log \int q_\phi(z|x) \cdot \frac{p_\theta(x, z)}{q_\phi(z|x)} \, dz
$$

**Explanation**: This is a mathematical trick. We're rewriting the integral in a form that lets us use importance sampling.

---

### Step 3: Rewrite as an expectation

**In words**: The integral over $q_\phi(z|x)$ is just an expectation under that distribution.

**In math**:

$$
= \log \mathbb{E}_{q_\phi(z|x)} \left[ \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]
$$

**Explanation**: We've converted the integral into an expectation, which we can estimate by sampling.

---

### Step 4: Apply Jensen's inequality

**In words**: The log of an expectation is at least as large as the expectation of the log.

**In math**:

$$
\geq \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p_\theta(x, z)}{q_\phi(z|x)} \right]
$$

**Explanation**: Jensen's inequality states that for a concave function (like $\log$):

$$
\log \mathbb{E}[X] \geq \mathbb{E}[\log X]
$$

This is where the "lower bound" comes from—we're trading equality for tractability.

---

### Step 5: Expand the log ratio

**In words**: Split the log of a ratio into a difference of logs.

**In math**:

$$
= \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x, z) - \log q_\phi(z|x) \right]
$$

**Explanation**: Using $\log(a/b) = \log a - \log b$.

---

### Step 6: Factor the joint distribution

**In words**: The joint $p_\theta(x, z)$ equals the likelihood times the prior.

**In math**:

$$
= \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x) \right]
$$

**Explanation**: We used $p_\theta(x, z) = p_\theta(x|z) \cdot p(z)$.

---

### Step 7: Rearrange into ELBO form

**In words**: Group the terms to reveal reconstruction and KL components.

**In math**:

$$
= \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p(z)}\right]}_{\text{KL divergence}}
$$

$$
= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \| p(z))
$$

**This is the ELBO.**

---

## 3. The Gap: What Are We Losing?

**In words**: The difference between the true log-likelihood and the ELBO is exactly the KL divergence between the approximate and true posteriors.

**In math**:

$$
\log p_\theta(x) - \text{ELBO} = \mathrm{KL}(q_\phi(z|x) \| p_\theta(z|x))
$$

**Explanation**:

- $p_\theta(z|x)$ is the *true* posterior—what Bayes' rule would give us if we could compute it.
- $q_\phi(z|x)$ is our *approximate* posterior—what the encoder outputs.
- The gap is always $\geq 0$ (KL is non-negative).
- When $q_\phi = p_\theta$, the gap is zero and ELBO equals the true log-likelihood.

**Implication**: Maximizing the ELBO simultaneously:
1. Increases the marginal likelihood $p_\theta(x)$
2. Pushes $q_\phi(z|x)$ toward the true posterior $p_\theta(z|x)$

---

## 4. Why Keep $q(z|x)$ Close to $p(z)$?

Three practical reasons: **sampling**, **generalization**, and **geometry**.

### Reason 1: So generation is possible at all

**In words**: At test time, we sample from the prior $p(z)$, not from any encoder.

**In math**:

$$
z \sim p(z) = \mathcal{N}(0, I) \quad \Rightarrow \quad x \sim p_\theta(x|z)
$$

**Explanation**: If the encoder learns to place latents in some weird region far from the prior, then sampling from $p(z)$ lands you in "dead space" the decoder never saw. Result: garbage samples.

**Intuition**: The KL term says: *"Don't hide all your data in a secret corner of latent space."*

---

### Reason 2: It prevents memorization

**In words**: Without KL, the encoder could assign each datapoint its own unique, sharply-peaked latent.

**Explanation**: 

- The encoder could make $q_\phi(z|x)$ extremely sharp (tiny $\sigma$) and well-separated for each datapoint.
- This is like a lookup table: perfect reconstruction, but no generalization.
- The decoder learns to memorize, not to generate.

**What KL penalizes**:

- Large $\mu(x)$: moving the mean far from the prior center
- Tiny $\sigma(x)$: collapsing the variance (over-confidence)

---

### Reason 3: It makes the latent space smooth

**In words**: We want nearby latents to produce similar outputs.

**Explanation**:

- If nearby $z$'s correspond to wildly different $x$'s, interpolation fails.
- Keeping $q$ near a simple, smooth prior encourages a globally consistent latent geometry.
- This is why VAEs can interpolate between samples—the latent space is "well-organized."

---

## 5. The Two Terms: A Balancing Act

| Term | Wants | Risk if too strong |
|------|-------|-------------------|
| **Reconstruction** | Perfect reproduction of input | Memorization, sharp posteriors |
| **KL** | Posteriors match prior | Posterior collapse, blurry outputs |

**Posterior collapse**: When KL dominates, the encoder learns $q_\phi(z|x) \approx p(z)$ for all $x$. The latent carries no information, and the decoder ignores it.

**The β-VAE insight**: Multiply KL by $\beta$ to control this trade-off explicitly.

---

## 6. Summary

1. **ELBO** = Reconstruction − KL
2. **Reconstruction** encourages the decoder to explain the data
3. **KL** keeps the encoder honest and the latent space usable
4. **The gap** between ELBO and true likelihood measures posterior approximation quality
5. **Maximizing ELBO** improves both the generative model and the inference network

---

## References

- [VAE-01-overview.md](VAE-01-overview.md) — Main VAE theory
- [VAE-03-inference.md](VAE-03-inference.md) — Why we introduce q(z|x)
- [beta_vae.md](../beta-VAE/beta_vae.md) — How β controls the reconstruction-KL trade-off
- Kingma & Welling (2014) — "Auto-Encoding Variational Bayes"
