# VAE: Why We Introduce q(z|x)

Addressing the fundamental question: "Why are we allowed to just introduce a distribution over latent variables?"

## 1. The Natural Objection

We *introduced* the posterior $q(z|x)$ in order to derive the ELBO.

And that immediately raises the natural objection:

> "If our goal is to learn latent variables $z$, why are we allowed to just introduce a distribution over them?"

**This is the right question.**

## 2. The Key Clarification

### We do NOT introduce $q(z|x)$ because we want it

We introduce it because **we cannot compute the true posterior**.

**In words**: The true object of interest is the posterior $p_\theta(z|x)$.

**In math**:

$$
p_\theta(z|x) = \frac{p_\theta(x|z) \cdot p(z)}{p_\theta(x)}
$$

**The problem**: The denominator requires:

$$
p_\theta(x) = \int p_\theta(x|z) \, p(z) \, dz
$$

which is **intractable**.

So the logic is not:

> "Let's introduce $q$ because it's convenient."

It is:

> "Exact inference is impossible, so we must approximate it."

This is classical **variational inference**, not a neural-network trick.

## 3. What Is Actually Being Optimized

Let's separate **variables**, **distributions**, and **parameters**.

### Latent variable $z$

- $z$ is a **random variable**
- It is *not* a parameter we optimize directly

### True posterior (unavailable)

- $p_\theta(z|x)$ — what Bayes' rule would give us
- Depends on unknown normalization $p_\theta(x)$
- Cannot be evaluated or sampled from

### Variational posterior (introduced)

- $q_\phi(z|x)$ — our approximation
- Tractable, learnable
- Parameterized by a neural network (the encoder)

**The crucial distinction**: We introduce $q(z|x)$ **not to replace $z$**, but to replace **inference about $z$**.

## 4. What "Learning z" Really Means

This sentence usually causes confusion:

> "VAEs learn latent variables."

What they actually do is:

> **VAEs learn a conditional distribution over latent variables given data.**

**In math**:

$$
z \sim q_\phi(z|x)
$$

So instead of learning:

- A single latent code $z_i$ per datapoint

We learn:

- A *function* that maps $x \to$ distribution over plausible $z$'s

This is **Bayesian inference**, amortized across the dataset.

## 5. Why Introducing $q(z|x)$ Is Mathematically Legitimate

Here's the clean justification:

> "We introduce $q(z|x)$ as an auxiliary distribution. This does not change the likelihood. It only allows us to rewrite it in a form we can optimize."

Nothing is assumed *about the data* at this step — only about tractability.

**In math**: The ELBO derivation starts with an exact identity:

$$
\log p(x) = \log \mathbb{E}_{q(z|x)} \left[ \frac{p(x, z)}{q(z|x)} \right]
$$

This identity is **exact**, before Jensen's inequality.

The approximation only enters when we **lower-bound** this expression, not when we introduce $q$.

## 6. What Assumptions Are Actually Being Made

### Assumptions about the model

- Data is generated from latent variables:

$$
z \sim p(z), \quad x \sim p_\theta(x|z)
$$

- The prior $p(z)$ is simple (e.g., Gaussian)

### Assumptions about inference

- Exact posterior inference is intractable
- A parametric family $q_\phi(z|x)$ is expressive enough to approximate it

### What is NOT assumed

- That there is a single "true" latent code
- That the posterior is Gaussian in reality
- That $q(z|x)$ is correct — only that it is optimizable

## 7. Known vs Unknown (Final Summary)

### Known / Fixed

| What | Value |
|------|-------|
| Observed data | $x$ |
| Prior | $p(z) = \mathcal{N}(0, I)$ |
| Network architectures | Encoder and decoder structure |

### Unknown / Learned

| What | Learned by |
|------|------------|
| Decoder parameters $\theta$ | Maximizing ELBO |
| Encoder parameters $\phi$ | Maximizing ELBO |
| Latent geometry | Emerges from training |

### Random (Sampled, Not Learned)

| What | Role |
|------|------|
| $z$ | Latent variable, sampled from $q_\phi(z|x)$ |
| $\epsilon$ | Noise for reparameterization, sampled from $\mathcal{N}(0, I)$ |

## 8. The One Sentence That Resolves Everything

If you remember only one sentence, make it this:

> **We are not learning latent variables directly — we are learning how to perform inference over latent variables.**

That sentence dissolves the apparent contradiction.

## 9. Why This Matters for What Comes Next

Once this clicks, the evolution of generative models becomes obvious:

| Model | Approach to Inference |
|-------|----------------------|
| **VAE** | Explicit approximate posterior $q_\phi(z|x)$ |
| **Diffusion** | Implicit posterior via score matching |
| **EBMs / JEPA** | No normalized posterior at all |
| **World models** | Latent inference + dynamics |

But they all inherit this core move:

> *Replace intractable inference with learnable inference.*

## References

- [VAE-01-overview.md](VAE-01-overview.md) — Main VAE theory
- [VAE-02-elbo.md](VAE-02-elbo.md) — ELBO derivation
- [ROADMAP.md](../ROADMAP.md) — Learning path to diffusion and beyond
