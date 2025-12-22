# The Reparameterization Trick

How to make sampling differentiable for end-to-end training.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $z$ | Latent variable (what we sample) |
| $\epsilon$ | Random noise from a fixed distribution (e.g., $\mathcal{N}(0, I)$) |
| $\mu_\phi(x)$ | Mean of the approximate posterior, output by encoder with parameters $\phi$ |
| $\sigma_\phi(x)$ | Standard deviation of the approximate posterior |
| $q_\phi(z \mid x)$ | Approximate posterior distribution |
| $f(z)$ | Any function of $z$ (e.g., $\log p_\theta(x \mid z)$) |
| $g_\phi(\epsilon)$ | Deterministic transformation: $g_\phi(\epsilon) = \mu_\phi + \sigma_\phi \cdot \epsilon$ |

---

## 1. The Core Problem

In a VAE, we want to optimize an objective like:

$$
\mathbb{E}_{q_\phi(z \mid x)}[f(z)]
$$

**In words**: The expected value of some function $f(z)$ where $z$ is sampled from a distribution $q_\phi(z \mid x)$ that depends on learnable parameters $\phi$.

**The problem**: How do we differentiate this expectation with respect to $\phi$?

If you sample directly:

$$
z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))
$$

then the sampling operation **breaks the computational graph**. Autograd sees a random number that "appeared from nowhere"—no gradient path exists.

---

## 2. Why Naïve Sampling Is Not Differentiable

Consider this code:

```python
z = Normal(mu, sigma).sample()
loss = f(z)
```

The mapping $(\mu, \sigma) \to z$ is *stochastic*, not deterministic.

There is no path for:

$$
\frac{\partial z}{\partial \mu}, \quad \frac{\partial z}{\partial \sigma}
$$

because the randomness is *inside* the operation. Gradients stop at the sampling step.

---

## 3. The Reparameterization Trick (The Exact Move)

Instead of sampling from the parameterized distribution, we rewrite the random variable as a **deterministic transformation of parameter-free noise**.

**Before** (not differentiable):

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

**After** (differentiable):

$$
\epsilon \sim \mathcal{N}(0, I), \quad z = \mu + \sigma \odot \epsilon
$$

Now:

- $\epsilon$ carries *all the randomness* and has no learnable parameters
- $\mu$ and $\sigma$ are deterministic outputs of the encoder
- $z$ is a deterministic function of $(\mu, \sigma, \epsilon)$

This restores the gradient path:

$$
\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon
$$

Gradients flow *through* the sample.

---

## 4. Why This Is Not a Hack

This is a **change of variables**, not a trick in the pejorative sense.

### What "moving randomness" means

Consider the VAE encoder-decoder pipeline:

```text
┌─────────────────────────────────────────────────────────────────┐
│  BEFORE (randomness inside the model):                          │
│                                                                  │
│  x → Encoder → (μ, σ) → [SAMPLE z ~ N(μ,σ²)] → Decoder → x̂     │
│                              ↑                                   │
│                         randomness here                          │
│                         (breaks gradients)                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  AFTER (randomness at the input):                               │
│                                                                  │
│  ε ~ N(0,I) ─────────────────────────┐                          │
│                                      ↓                          │
│  x → Encoder → (μ, σ) → [z = μ + σ·ε] → Decoder → x̂            │
│                              ↑                                   │
│                    deterministic function                        │
│                    (gradients flow through)                      │
└─────────────────────────────────────────────────────────────────┘
```

**"Inside the model"** = the sampling step $z \sim \mathcal{N}(\mu, \sigma^2)$ occurs between encoder and decoder, blocking gradients.

**"Input of the model"** = the noise $\epsilon$ is sampled *before* any computation, then passed through as a regular input.

The key insight: sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is *mathematically equivalent* to sampling $\epsilon \sim \mathcal{N}(0, 1)$ and computing $z = \mu + \sigma \cdot \epsilon$. We've just rewritten the same random variable in a differentiable form.

---

## 5. What the Reparameterization Trick Enables

### The Key Identity

Define the **reparameterization function**:

$$
g_\phi(\epsilon) = \mu_\phi + \sigma_\phi \cdot \epsilon
$$

This is a deterministic function that transforms noise $\epsilon$ into a sample $z$ using the encoder parameters $\phi$.

### Derivation

We want to compute:

$$
\nabla_\phi \, \mathbb{E}_{q_\phi(z \mid x)}[f(z)]
$$

**Step 1**: Rewrite $z$ using the reparameterization:

$$
z = g_\phi(\epsilon) = \mu_\phi + \sigma_\phi \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Step 2**: Substitute into the expectation. Since $z$ is now a deterministic function of $\epsilon$:

$$
\mathbb{E}_{q_\phi(z \mid x)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[f(g_\phi(\epsilon))]
$$

**Step 3**: Move the gradient inside the expectation (valid because $\epsilon$ doesn't depend on $\phi$):

$$
\nabla_\phi \, \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[f(g_\phi(\epsilon))] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[\nabla_\phi f(g_\phi(\epsilon))]
$$

### The Result

$$
\nabla_\phi \, \mathbb{E}_{q_\phi(z \mid x)}[f(z)] = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[\nabla_\phi f(g_\phi(\epsilon))\right]
$$

**In words**: The gradient of an expectation over a parameterized distribution equals the expectation of the gradient, computed by:

1. Sampling noise $\epsilon$ from a fixed distribution
2. Transforming it to $z = g_\phi(\epsilon)$
3. Computing $\nabla_\phi f(z)$ via standard backpropagation

This gives us:

- **Low-variance gradients** (compared to REINFORCE)
- **End-to-end backpropagation** through the sampling step
- **Scalable variational inference**

Without this, VAEs would not work in practice.

---

## 6. Other Reparameterizable Distributions

The trick generalizes beyond Gaussians.

### Location-Scale Families

Any distribution of the form:

$$
z = \mu + \sigma \cdot \epsilon
$$

Examples: Gaussian, Logistic, Laplace—all reparameterizable.

### Log-Normal

$$
z \sim \text{LogNormal}(\mu, \sigma) \quad \Rightarrow \quad z = \exp(\mu + \sigma \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, 1)
$$

### Gumbel Distribution

$$
g = -\log(-\log u), \quad u \sim \text{Uniform}(0, 1)
$$

This underlies **Gumbel-Softmax** for differentiable categorical sampling.

---

## 7. Discrete Variables: When Reparameterization Fails

Discrete sampling (e.g., categorical variables) is **not reparameterizable** in the same clean way.

### Gumbel-Softmax (Concrete Distribution)

An approximation that:

- Samples continuous noise
- Applies softmax with temperature

This gives approximate differentiability with biased but low-variance gradients. It's *inspired* by reparameterization but not exact.

---

## 8. Contrast with REINFORCE (Score-Function Estimator)

Before reparameterization, people used the score-function estimator:

$$
\nabla_\phi \, \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{q_\phi(z)}\left[f(z) \cdot \nabla_\phi \log q_\phi(z)\right]
$$

This works for *any* distribution, but has:

- Extremely high variance
- Unstable training
- Slow convergence

Reparameterization trades generality for smooth, low-variance gradients. This is why VAEs beat earlier variational methods in practice.

---

## 9. Where This Idea Shows Up Elsewhere

### Normalizing Flows

Flows are **pure reparameterization**:

$$
z = f_\theta(\epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

All expressivity comes from the transformation $f_\theta$.

### Diffusion Models

Reverse diffusion is effectively sampling via deterministic transforms driven by injected noise. The conceptual lineage is direct.

### Policy Gradients in RL

In continuous-action RL (SAC, some PPO variants):

$$
a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon
$$

This gives lower-variance gradients than REINFORCE.

### Bayesian Neural Networks

Weight uncertainty:

$$
w = \mu + \sigma \cdot \epsilon
$$

Makes uncertainty differentiable.

---

## 10. Summary

> **The reparameterization trick works by moving randomness from inside the model to the input, turning stochastic nodes into deterministic functions of noise.**

---

## 11. Connection to Other Generative Models

| Model | How It Uses Reparameterization |
|-------|-------------------------------|
| **VAE** | Reparameterize latent sampling |
| **Flows** | Reparameterize entire distributions |
| **Diffusion** | Reparameterize generation as noise removal |
| **World Models** | Reparameterize uncertainty over dynamics |

They all share the same theme: *make uncertainty differentiable*.

---

## References

- [VAE-01-overview.md](VAE-01-overview.md) — Main VAE theory
- [VAE-02-elbo.md](VAE-02-elbo.md) — ELBO derivation
- Kingma & Welling (2014) — "Auto-Encoding Variational Bayes"
