# Choosing the Likelihood: NB vs ZINB for Gene Expression

This document explains why the choice of $p(x \mid z)$ matters for VAEs in computational biology, and when to use Negative Binomial (NB) vs Zero-Inflated Negative Binomial (ZINB).

---

## 1. The Core Principle

> **Choosing $p(x \mid z)$ means choosing a statistical noise model for your data.**
>
> Once you choose it, **maximum likelihood training forces a specific loss function**.

This is not an ML trick — it is straight probability theory.

---

## 2. What is $x$ in Gene Expression Modeling?

For RNA-seq, your observed data $x$ is usually one of:

* **Raw counts**: non-negative integers (e.g., gene A has 0, 3, 17 reads)
* **Normalized/transformed values**: log1p(TPM), logCPM, etc. (continuous)

This choice alone already restricts what likelihoods make sense.

---

## 3. Why Gaussian (→ MSE) is Often Wrong for Counts

A Gaussian likelihood assumes:

* Data is continuous
* Symmetric noise
* Variance independent of the mean
* Can take negative values

Raw RNA-seq counts violate *all* of these.

If you write:

$$
p(x \mid z) = \mathcal{N}(\mu(z), \sigma^2)
$$

you are implicitly saying:

> "Gene expression fluctuates symmetrically around a mean, with constant variance."

This is **biologically false** for counts.

MSE is only acceptable after heavy preprocessing (log transforms), and even then it's an approximation.

---

## 4. The Negative Binomial Distribution: Standard Interpretation

### 4.1 Classical Definition

The Negative Binomial distribution has two standard interpretations:

**Interpretation 1: Counting Failures**

> Count the number of failures before observing $r$ successes, where each trial has success probability $p$.

$$
X \sim \text{NB}(r, p) \quad \Rightarrow \quad P(X = k) = \binom{k + r - 1}{k} p^r (1-p)^k
$$

**Interpretation 2: Gamma-Poisson Mixture** (more relevant for ML)

> A Poisson distribution whose rate $\lambda$ is itself random, drawn from a Gamma distribution.

$$
\lambda \sim \text{Gamma}(\theta, \theta/\mu) \quad \Rightarrow \quad X \mid \lambda \sim \text{Poisson}(\lambda)
$$

Marginalizing out $\lambda$ gives $X \sim \text{NB}(\mu, \theta)$.

This second interpretation is why NB is called an **overdispersed Poisson** — it adds extra variance beyond what Poisson allows.

### 4.2 Why NB is Used in ML

NB appears whenever you have:

* **Count data** with more variance than Poisson predicts
* **Heterogeneity** in the underlying rate (different cells, users, events)
* **Clustering** or "burstiness" in arrivals

**Common applications**:

| Domain | Phenomenon Modeled |
|--------|--------------------|
| **Genomics** | Gene expression counts (RNA-seq) |
| **NLP** | Word counts in documents |
| **Epidemiology** | Disease case counts |
| **E-commerce** | Purchase counts per user |
| **Insurance** | Claim counts per policy |
| **Ecology** | Species abundance counts |

In all cases, the key property is:

$$
\text{Var}(X) = \mu + \frac{\mu^2}{\theta} > \mu
$$

This **overdispersion** (variance > mean) is ubiquitous in real count data.

### 4.3 NB in Neural Networks

In deep learning, NB is used as the **output distribution** when:

1. The target is non-negative integer counts
2. A Poisson assumption underestimates variance
3. You want a proper probabilistic model (not just MSE)

The network predicts the parameters $(\mu, \theta)$, and the loss is the NB negative log-likelihood.

---

## 5. Why Negative Binomial (NB) is the Default for RNA-seq

### 5.1 What NB Models

Negative Binomial models:

* **Non-negative integer counts**
* **Variance that grows with the mean** (overdispersion)

This matches RNA-seq extremely well.

Formally, for one gene $g$:

$$
x_g \sim \text{NB}(\mu_g, \alpha_g)
$$

where:
* $\mu_g$ = mean (predicted by decoder)
* $\alpha_g$ = dispersion parameter

**Key property** — the mean-variance relationship:

$$
\text{Var}(x_g) = \mu_g + \alpha_g \cdot \mu_g^2
$$

This captures:

* Biological variability
* Technical noise
* Sequencing depth effects

### 5.2 How NB Becomes the Loss Function

In a VAE/cVAE, the decoder predicts $\mu_g(z, y)$ (and possibly $\alpha_g$).

The training objective includes:

$$
\log p(x \mid z, y) = \sum_{g} \log \text{NB}(x_g \mid \mu_g(z, y), \alpha_g)
$$

So the **reconstruction loss** is the **negative log-likelihood of the NB distribution**.

This replaces MSE. There is no freedom here:

* If you assume NB → maximum likelihood forces this loss

---

## 6. Why ZINB Exists (and When You Need It)

### 6.1 The Zero Problem

In **single-cell RNA-seq**, you see *many* zeros:

* Some are **true biological zeros** (gene not expressed)
* Many are **dropout** (gene expressed but not captured)

NB alone often underestimates zeros.

### 6.2 Zero-Inflated NB (ZINB)

ZINB says:

> "Some zeros come from the NB process, but others come from an extra 'always zero' process."

Mathematically, for each gene:

$$
p(x_g = 0) = \pi_g + (1 - \pi_g) \cdot \text{NB}(0 \mid \mu_g, \alpha_g)
$$

$$
p(x_g = k) = (1 - \pi_g) \cdot \text{NB}(k \mid \mu_g, \alpha_g), \quad k > 0
$$

The decoder now predicts three quantities:

| Parameter | Meaning |
|-----------|---------|
| $\mu_g(z, y)$ | Mean expression |
| $\alpha_g$ | Dispersion |
| $\pi_g(z, y)$ | Zero-inflation probability |

> The likelihood of the observed expression is computed gene by gene using a zero-inflated Negative Binomial distribution whose parameters are predicted by the decoder network

---

## 7. When to Use NB vs ZINB

### Use **NB** if:

* Bulk RNA-seq
* Pseudo-bulk scRNA
* UMI-based scRNA with good depth
* Zeros are mostly explained by low expression

### Use **ZINB** if:

* Sparse scRNA-seq
* Very shallow sequencing
* Dropout dominates zero counts
* NB underfits zeros badly

**Practical advice**: Many modern pipelines start with NB and only escalate to ZINB if diagnostics demand it.

---

## 8. Where Do NB/ZINB Apply? (Decoder Only)

**Important clarification**: NB and ZINB are used **only in the decoder**, not the encoder.

### Why?

The VAE has two parts:

| Component | Input | Output | Distribution |
|-----------|-------|--------|-------------|
| **Encoder** | $x$ (observed data) | $z$ (latent) | $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x))$ |
| **Decoder** | $z$ (latent) | $x$ (reconstructed) | $p_\theta(x \mid z) = \text{NB}(\mu_\theta(z), \alpha)$ |

* **Encoder**: Maps data to latent space. The latent $z$ is continuous and Gaussian — this is required for the reparameterization trick.
* **Decoder**: Maps latent back to data space. The likelihood $p(x \mid z)$ must match the data type — hence NB/ZINB for counts.

### The Encoder Stays Gaussian

Even when modeling count data:

$$
q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))
$$

This is because:

1. **Reparameterization requires continuous distributions** — you can't easily reparameterize discrete distributions
2. **The latent space is a learned representation** — it doesn't need to match the data distribution
3. **KL divergence is tractable** — Gaussian-to-Gaussian KL has a closed form

### Summary

```text
┌─────────────────────────────────────────────────────────────┐
│  ENCODER: q(z|x)                                            │
│  • Always Gaussian (for reparameterization)                 │
│  • Input: count data x                                      │
│  • Output: continuous latent z                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  DECODER: p(x|z)                                            │
│  • NB or ZINB (matches data type)                           │
│  • Input: continuous latent z                               │
│  • Output: count data x                                     │
└─────────────────────────────────────────────────────────────┘
```

### Why Gaussian Prior Even with NB/ZINB Decoder?

You might wonder: if we're modeling count data with NB/ZINB, why is the KL divergence still computed against a Gaussian prior $p(z) = \mathcal{N}(0, I)$?

**Key insight**: The latent space $z$ is separate from the data space $x$.

The NB/ZINB likelihood only affects $p(x \mid z)$ — how we model the **output**. The latent $z$ is a **learned representation**, not the data itself.

**Why keep the latent Gaussian?**

1. **Reparameterization trick requires continuous, differentiable sampling**
   - You can't easily backpropagate through discrete distributions
   - Gaussian allows $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$

2. **The prior $p(z) = \mathcal{N}(0, I)$ is a regularizer**
   - Prevents the encoder from "cheating" by encoding each sample at a unique, far-away point
   - Forces the latent space to be **smooth and interpolable**
   - Without it, the VAE degenerates into a deterministic autoencoder

3. **Tractable KL divergence**
   - Gaussian-to-Gaussian KL has a closed form:
     $$\text{KL}(q \| p) = -\frac{1}{2} \sum_j \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$
   - No sampling needed for the KL term

**What would happen without the Gaussian prior?**

- The encoder could map each training sample to a unique, isolated point
- The latent space would have "holes" — regions with no training data
- Sampling from $p(z)$ would generate garbage
- No smooth interpolation between samples

> **Summary**: The decoder likelihood (NB/ZINB) determines how we model the *data*. The encoder prior (Gaussian) determines how we regularize the *latent space*. These are independent design choices.

---

## 9. Summary: Likelihood → Loss

| Data Representation | Likelihood $p(x \mid z)$ | Loss Function |
|--------------------|--------------------------|---------------|
| log1p(TPM) | Gaussian | MSE |
| Counts (bulk) | NB | NB NLL |
| Counts (scRNA) | ZINB | ZINB NLL |

You are not choosing a "loss function" — you are choosing a **data-generating assumption**.

> **Note on `log1p(TPM)` notation**:
>
> `log1p(x) = log(1 + x)`
>
> Why use `log1p` instead of `log`?
>
> - **Handles zeros**: `log1p(0) = log(1) = 0` (vs `log(0) = -∞`)
> - **Reduces skewness**: Compresses high values more than low values
> - **Stabilizes variance**: Makes variance less dependent on the mean

---

## 10. Why This Motivates Score Matching (Preview)

You're now seeing the pressure point:

* Biology data is messy
* NB vs ZINB is already a modeling compromise
* Wrong likelihood → biased gradients everywhere

Score matching later says:

> "Let's stop committing to a precise likelihood."

But you **cannot appreciate that move** until you fully understand NB/ZINB — which is what this document covers.

---

## 11. Key Takeaway

> **Choosing $p(x \mid z)$ is choosing how you believe noise enters your biological measurements; maximum likelihood then forces the corresponding loss.**

---

## Next Steps

The next document ([VAE-08-NB-likelihood.md](VAE-08-NB-likelihood.md)) covers:

1. Write out the NB log-likelihood term explicitly
2. Compare it side-by-side with MSE
3. Interpret each term biologically (mean, dispersion, zeros)

---

## References

* [VAE-02-elbo.md](VAE-02-elbo.md) — ELBO derivation
* [VAE-08-NB-likelihood.md](VAE-08-NB-likelihood.md) — NB log-likelihood derivation
* Lopez et al. (2018) — "Deep generative modeling for single-cell transcriptomics" (scVI)
* Eraslan et al. (2019) — "Single-cell RNA-seq denoising using a deep count autoencoder" (DCA)
* Cameron & Trivedi (2013) — "Regression Analysis of Count Data"
