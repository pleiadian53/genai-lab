# VAE Optimization: The ELBO and Gradient Flow

> **Core question**: How does a VAE actually learn? What objective is being optimized, and how do gradients flow through both encoder and decoder?

This document unpacks the optimization mechanics of VAEs — the single equation that governs training, how both networks are updated jointly, and why the reparameterization trick is essential. Understanding this is prerequisite for diagnosing issues like posterior collapse and for extending VAEs to conditional or β-VAE variants.

---

## 1. The Optimization Objective (The One Equation That Rules Them All)

A VAE is trained by **maximizing the Evidence Lower Bound (ELBO)** on the data likelihood.

We start from the quantity we *wish* we could maximize directly:

$$
\log p_\theta(x)
$$

This is the log probability that the generative model assigns to a datapoint $x$ (gene expression vector, cell profile, etc.). Unfortunately, it involves an intractable integral over latent variables $z$.

So we introduce an approximate posterior $q_\phi(z \mid x)$ and derive a lower bound:

$$
\log p_\theta(x) \;\geq\; \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \mathrm{KL}\left(q_\phi(z \mid x) \,\|\, p(z)\right)
$$

This right-hand side is the **ELBO**.

Spelled out in words:

- **First term (Reconstruction)**: How well can the decoder reconstruct the data from the latent code?
- **Second term (KL Regularization)**: How close is the encoder's posterior to the prior we want the latent space to follow?

Training a VAE means **maximizing ELBO**, or equivalently **minimizing the negative ELBO** as a loss.

---

## 2. Is ELBO Used in Both Encoder and Decoder?

Yes — and this point is subtle but crucial.

There is **one single scalar objective (ELBO)**, but **both networks appear inside it**, so both are optimized jointly.

### Parameter Breakdown

- **Decoder parameters** $\theta$ appear in $\log p_\theta(x \mid z)$
- **Encoder parameters** $\phi$ appear in $q_\phi(z \mid x)$, which affects:
  - The expectation (where we sample $z$ from)
  - The KL divergence term

### Gradient Flow During Backpropagation

- Gradients flow into **decoder weights** via reconstruction quality
- Gradients flow into **encoder weights** via:
  - Reconstruction quality (through sampled $z$)
  - KL regularization

**No separate objectives, no alternating optimization by default.** One ELBO, one joint training loop.

---

## 3. Walking Through Optimization Step-by-Step

Let's narrate a single forward + backward pass as if we were the optimizer.

### Step 1: Encoder Outputs $q_\phi(z \mid x)$

You feed in a data point $x$ (say a gene expression vector).

The encoder outputs **parameters of a distribution**:

$$
q_\phi(z \mid x) = \mathcal{N}\left(\mu_\phi(x), \sigma_\phi^2(x)\right)
$$

**Important**: The encoder does not output $z$. It outputs a *distribution over possible latent codes*. This is what makes the model Bayesian.

---

### Step 2: Reparameterization Trick (So Gradients Don't Die)

Sampling is not differentiable. So we rewrite:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon \quad\text{where}\quad \varepsilon \sim \mathcal{N}(0, I)
$$

Now:

- Randomness is isolated in $\varepsilon$
- $z$ is a deterministic function of $\phi$

This lets gradients flow **through the sample** back into the encoder.

---

### Step 3: Decoder Outputs $p_\theta(x \mid z)$

The sampled $z$ is fed into the decoder.

The decoder outputs **parameters of a likelihood distribution**:

| Likelihood | Use Case |
|------------|----------|
| Gaussian | Real-valued expression |
| Poisson / Negative Binomial | RNA-seq counts |
| ZINB | scRNA-seq with dropouts |

So the decoder is not saying "here is $x$" — it is saying:

> "Given $z$, this is the probability distribution over possible $x$."

---

### Step 4: Compute the ELBO

Now we compute two quantities:

#### (a) Reconstruction Term

$$
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
$$

In practice:

- One or a few Monte Carlo samples of $z$
- Log-likelihood under the decoder distribution

This term:

- Trains the **decoder** to reconstruct data
- Trains the **encoder** to produce useful latent codes

#### (b) KL Divergence

$$
\mathrm{KL}\left(q_\phi(z \mid x) \,\|\, p(z)\right)
$$

This:

- Pulls latent representations toward the prior (usually $\mathcal{N}(0,I)$)
- Prevents pathological memorization
- Gives latent space semantic structure

---

### Step 5: Backpropagation (The Unifying Moment)

The total loss is:

$$
\mathcal{L} = -\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] + \mathrm{KL}\left(q_\phi(z \mid x) \,\|\, p(z)\right)
$$

Gradients flow:

- From reconstruction → decoder parameters $\theta$
- Through $z$ → encoder parameters $\phi$
- From KL → encoder parameters $\phi$

Everything is learned **jointly** via standard SGD/Adam.

---

## 4. Conceptual Summary

A VAE is **not**:

- "Encoder learns posterior, decoder learns reconstruction" (too shallow)

A VAE **is**:

> A single probabilistic model trained by maximizing a variational bound, where the encoder proposes latent explanations and the decoder judges how well those explanations generate the data, under a global pressure to keep the latent space simple and continuous.

This framing scales beautifully to:

- scRNA-seq (scVI, scGen)
- Conditional VAEs (tissue, disease, batch)
- β-VAEs (disentanglement)
- Semi-supervised VAEs
- Diffusion–VAE hybrids

And it sets you up perfectly for score matching and diffusion, where the "posterior" idea gets replaced by something even sneakier.

---

## Next Steps

**Next natural topic**: Why the KL term causes **posterior collapse**, and why scRNA-seq VAEs often weaken or reshape it.
