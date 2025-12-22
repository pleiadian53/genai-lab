# The Negative Binomial Log-Likelihood: Derivation and Biological Interpretation

This document derives the NB log-likelihood explicitly, compares it with MSE, and interprets each term biologically.

---

## 1. The Negative Binomial Distribution

### 1.1 Probability Mass Function

For a count $x \in \{0, 1, 2, \ldots\}$, the Negative Binomial PMF is:

$$
\text{NB}(x \mid \mu, \alpha) = \binom{x + r - 1}{x} \left(\frac{r}{r + \mu}\right)^r \left(\frac{\mu}{r + \mu}\right)^x
$$

where:
* $\mu$ = mean
* $r = 1/\alpha$ = "number of failures" parameter (inverse dispersion)
* $\alpha$ = dispersion parameter

### 1.2 Alternative Parameterization (More Common in ML)

Using $\theta = 1/\alpha$ (inverse dispersion):

$$
\text{NB}(x \mid \mu, \theta) = \frac{\Gamma(x + \theta)}{\Gamma(\theta) \cdot x!} \left(\frac{\theta}{\theta + \mu}\right)^\theta \left(\frac{\mu}{\theta + \mu}\right)^x
$$

### 1.3 Mean and Variance

$$
\mathbb{E}[x] = \mu
$$

$$
\text{Var}(x) = \mu + \frac{\mu^2}{\theta} = \mu + \alpha \mu^2
$$

**Key insight**: Variance grows faster than the mean. This is **overdispersion**.

---

## 2. The NB Log-Likelihood

Taking the log of the PMF:

$$
\log \text{NB}(x \mid \mu, \theta) = \log \Gamma(x + \theta) - \log \Gamma(\theta) - \log(x!) + \theta \log\left(\frac{\theta}{\theta + \mu}\right) + x \log\left(\frac{\mu}{\theta + \mu}\right)
$$

### 2.1 Simplified Form

Let $p = \frac{\theta}{\theta + \mu}$ (probability of "failure" in the NB interpretation). Then:

$$
\log \text{NB}(x \mid \mu, \theta) = \log \binom{x + \theta - 1}{x} + \theta \log p + x \log(1 - p)
$$

### 2.2 The Loss Function

For a VAE, the reconstruction loss for one gene $g$ is:

$$
\mathcal{L}_{\text{recon}}^{(g)} = -\log \text{NB}(x_g \mid \mu_g, \theta_g)
$$

Summing over all $G$ genes:

$$
\mathcal{L}_{\text{recon}} = -\sum_{g=1}^{G} \log \text{NB}(x_g \mid \mu_g, \theta_g)
$$

---

## 3. Side-by-Side Comparison: NB vs MSE

### 3.1 MSE (Gaussian Likelihood)

If $p(x \mid z) = \mathcal{N}(\mu, \sigma^2)$, the log-likelihood is:

$$
\log p(x \mid z) = -\frac{1}{2\sigma^2}(x - \mu)^2 - \frac{1}{2}\log(2\pi\sigma^2)
$$

Ignoring constants, the loss is:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{2\sigma^2} \sum_g (x_g - \mu_g)^2
$$

### 3.2 NB Negative Log-Likelihood

$$
\mathcal{L}_{\text{NB}} = -\sum_g \left[\log \Gamma(x_g + \theta_g) - \log \Gamma(\theta_g) - \log(x_g!) + \theta_g \log\left(\frac{\theta_g}{\theta_g + \mu_g}\right) + x_g \log\left(\frac{\mu_g}{\theta_g + \mu_g}\right)\right]
$$

### 3.3 Key Differences

| Aspect | MSE (Gaussian) | NB NLL |
|--------|----------------|--------|
| **Penalizes** | Squared deviation $(x - \mu)^2$ | Log-probability of count |
| **Variance model** | Constant $\sigma^2$ | $\mu + \mu^2/\theta$ (grows with mean) |
| **Zero handling** | No special treatment | Natural probability mass at 0 |
| **Domain** | $x \in \mathbb{R}$ | $x \in \{0, 1, 2, \ldots\}$ |
| **Gradient behavior** | Linear in residual | Depends on $\mu$, $\theta$, and $x$ |

---

## 4. Biological Interpretation of Each Term

Let's read the NB log-likelihood term by term:

### Term 1: $\log \Gamma(x + \theta) - \log \Gamma(\theta) - \log(x!)$

**What it is**: Combinatorial normalization

**Biological meaning**: Accounts for the number of ways to observe $x$ counts given the underlying process. This term ensures the distribution sums to 1.

### Term 2: $\theta \log\left(\frac{\theta}{\theta + \mu}\right)$

**What it is**: Contribution from the dispersion parameter

**Biological meaning**: 
* When $\theta$ is large (low dispersion), this term dominates → distribution is more Poisson-like
* When $\theta$ is small (high dispersion), variance is large → accounts for biological heterogeneity

### Term 3: $x \log\left(\frac{\mu}{\theta + \mu}\right)$

**What it is**: Contribution from the observed count

**Biological meaning**:
* Penalizes mismatch between predicted mean $\mu$ and observed count $x$
* Larger $x$ → stronger pull toward higher $\mu$
* This is where the decoder's prediction $\mu_g(z, y)$ enters

---

## 5. Why NB Handles Zeros Better Than Gaussian

### 5.1 Probability of Zero Under NB

$$
\text{NB}(0 \mid \mu, \theta) = \left(\frac{\theta}{\theta + \mu}\right)^\theta
$$

This is a **proper probability mass** at zero.

* When $\mu$ is small → $P(x=0)$ is large (expected)
* When $\mu$ is large → $P(x=0)$ is small (rare event)

### 5.2 Probability of Zero Under Gaussian

$$
\mathcal{N}(0 \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\mu^2}{2\sigma^2}\right)
$$

This is a **probability density**, not a mass. The Gaussian:
* Assigns positive density to negative values (impossible for counts)
* Treats $x=0$ the same as any other point
* Has no special structure for zeros

---

## 6. The Dispersion Parameter $\theta$ (or $\alpha = 1/\theta$)

### 6.1 What It Controls

$$
\text{Var}(x) = \mu + \frac{\mu^2}{\theta}
$$

* **Large $\theta$** (small $\alpha$): Variance ≈ $\mu$ → Poisson-like
* **Small $\theta$** (large $\alpha$): Variance ≫ $\mu$ → highly overdispersed

### 6.2 How It's Learned

In practice:
* **Gene-specific $\theta_g$**: Each gene has its own dispersion (more flexible)
* **Shared $\theta$**: One dispersion for all genes (simpler)
* **Predicted $\theta_g(z)$**: Dispersion depends on latent state (most flexible)

scVI uses gene-specific dispersion learned during training.

---

## 7. Gradient Behavior: Why NB Trains Differently

### 7.1 MSE Gradient

$$
\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \mu_g} = \frac{\mu_g - x_g}{\sigma^2}
$$

Linear in the residual. Large counts → large gradients.

### 7.2 NB Gradient

$$
\frac{\partial \mathcal{L}_{\text{NB}}}{\partial \mu_g} = \frac{\theta_g}{\theta_g + \mu_g} - \frac{x_g}{\mu_g}
$$

This gradient:
* Is **bounded** as $\mu_g \to \infty$
* Naturally handles the mean-variance relationship
* Doesn't explode for large counts

---

## 8. Implementation Note: Log-Space Stability

In practice, we often predict $\log \mu$ instead of $\mu$ directly:

```python
log_mu = decoder(z)  # Predict log-mean
mu = torch.exp(log_mu)  # Ensures mu > 0
```

This avoids numerical issues with negative means and improves optimization stability.

---

## 9. Summary Table

| Component | Symbol | Biological Meaning |
|-----------|--------|-------------------|
| Mean | $\mu_g$ | Expected expression level of gene $g$ |
| Dispersion | $\theta_g$ (or $\alpha_g = 1/\theta_g$) | Gene-specific noise level |
| Variance | $\mu_g + \mu_g^2/\theta_g$ | Total variability (Poisson + overdispersion) |
| $P(x=0)$ | $(\theta/(\theta+\mu))^\theta$ | Probability of zero counts |

---

## 10. Key Takeaway

> **The NB log-likelihood is not just a "different loss function" — it encodes a specific belief about how count data is generated, with mean-variance coupling and proper handling of zeros.**

When you minimize NB NLL, you are finding parameters that maximize the probability of observing your data under this generative model.

---

## Next Steps

The next document will cover:
* **ZINB log-likelihood** — adding the zero-inflation component
* **Comparing NB vs ZINB diagnostics** — when to switch

---

## References

* [VAE-07-NB-ZINB.md](VAE-07-NB-ZINB.md) — Overview of likelihood choices
* Lopez et al. (2018) — "Deep generative modeling for single-cell transcriptomics"
* Hilbe (2011) — "Negative Binomial Regression"
