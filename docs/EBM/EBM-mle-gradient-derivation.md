# Why MLE for EBMs is Hard: The Villain Origin Story

This document derives the gradient of the log-likelihood for energy-based models, revealing why maximum likelihood estimation (MLE) is computationally challenging. The derivation shows that the gradient contains an intractable expectation under the model distribution—the fundamental obstacle that motivates alternative training methods like score matching and contrastive divergence.

---

## Goal

For an energy-based model

$$
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}, \qquad Z_\theta = \int e^{-E_\theta(x)} \, dx
$$

show that the gradient of the log-likelihood has the form

$$
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{x' \sim p_\theta}\big[\nabla_\theta E_\theta(x')\big]
$$

That second term is the painful part: it's an expectation under the model distribution $p_\theta$, which is usually intractable and needs sampling (often MCMC).

---

## Step 0: Notation

- $x$: a data point
- $\theta$: model parameters
- $E_\theta(x)$: energy function
- $Z_\theta$: partition function (normalizer)
- $\nabla_\theta$: gradient w.r.t. parameters $\theta$
- $\mathbb{E}_{p_\theta}[\cdot]$: expectation where $x' \sim p_\theta(x')$

**Assumption (standard regularity):** we can swap gradient and integral:

$$
\nabla_\theta \int f_\theta(x) \, dx = \int \nabla_\theta f_\theta(x) \, dx
$$

(You can justify this with dominated convergence / Leibniz rule; most ML papers assume it.)

---

## Step 1: Start with the log density

$$
\log p_\theta(x) = \log\left(\frac{e^{-E_\theta(x)}}{Z_\theta}\right)
$$

**Explanation:** Just take logs of the EBM definition.

---

## Step 2: Split numerator and denominator

$$
\log p_\theta(x) = \log(e^{-E_\theta(x)}) - \log Z_\theta
$$

**Explanation:** $\log(a/b) = \log a - \log b$.

---

## Step 3: Simplify the first term

$$
\log p_\theta(x) = -E_\theta(x) - \log Z_\theta
$$

**Explanation:** $\log(e^{u}) = u$. Here $u = -E_\theta(x)$.

---

## Step 4: Differentiate w.r.t. $\theta$

$$
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta
$$

**Explanation:** Gradient is linear; derivative of $-E_\theta(x)$ is $-\nabla_\theta E_\theta(x)$.

So the *only* remaining job is to compute $\nabla_\theta \log Z_\theta$.

---

## Step 5: Differentiate $\log Z_\theta$ using the chain rule

$$
\nabla_\theta \log Z_\theta = \frac{1}{Z_\theta} \nabla_\theta Z_\theta
$$

**Explanation:** $\nabla_\theta \log u = (\nabla_\theta u)/u$.

---

## Step 6: Expand $Z_\theta$ and move gradient inside the integral

$$
\nabla_\theta Z_\theta = \nabla_\theta \int e^{-E_\theta(x')} \, dx' = \int \nabla_\theta \left(e^{-E_\theta(x')}\right) \, dx'
$$

**Explanation:** This is the "swap gradient and integral" step.

---

## Step 7: Differentiate the exponential

$$
\nabla_\theta \left(e^{-E_\theta(x')}\right) = e^{-E_\theta(x')} \cdot \nabla_\theta(-E_\theta(x')) = -e^{-E_\theta(x')} \cdot \nabla_\theta E_\theta(x')
$$

**Explanation:** Chain rule: derivative of $e^{u}$ is $e^{u} \nabla u$. Here $u = -E_\theta(x')$.

---

## Step 8: Substitute back into $\nabla_\theta Z_\theta$

$$
\nabla_\theta Z_\theta = \int \left(-e^{-E_\theta(x')} \cdot \nabla_\theta E_\theta(x')\right) \, dx' = -\int e^{-E_\theta(x')} \cdot \nabla_\theta E_\theta(x') \, dx'
$$

**Explanation:** Just plug in the expression from Step 7.

---

## Step 9: Plug into $\nabla_\theta \log Z_\theta$

$$
\nabla_\theta \log Z_\theta = \frac{1}{Z_\theta}\left(-\int e^{-E_\theta(x')} \cdot \nabla_\theta E_\theta(x') \, dx'\right) = -\int \frac{e^{-E_\theta(x')}}{Z_\theta} \cdot \nabla_\theta E_\theta(x') \, dx'
$$

**Explanation:** Divide the integral by $Z_\theta$; that's exactly how $p_\theta$ is defined.

---

## Step 10: Recognize $p_\theta(x')$ and rewrite as an expectation

Since

$$
p_\theta(x') = \frac{e^{-E_\theta(x')}}{Z_\theta}
$$

we have

$$
\nabla_\theta \log Z_\theta = -\int p_\theta(x') \cdot \nabla_\theta E_\theta(x') \, dx' = -\mathbb{E}_{x' \sim p_\theta}\left[\nabla_\theta E_\theta(x')\right]
$$

**Explanation:** An expectation is just an integral weighted by the density.

---

## Step 11: Put it all together (the classic EBM gradient)

Recall Step 4:

$$
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta
$$

Substitute Step 10:

$$
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \left(-\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x')]\right)
$$

$$
\boxed{\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{x' \sim p_\theta}\big[\nabla_\theta E_\theta(x')\big]}
$$

**Explanation (the intuition):**

- **First term (data term):** push down energy on observed data $x$ → "make data likely."
- **Second term (model term):** push up energy on typical samples $x' \sim p_\theta$ → "make non-data less likely."

---

## Why This Makes MLE Hard (The Punchline)

That expectation

$$
\mathbb{E}_{x' \sim p_\theta}[\nabla_\theta E_\theta(x')]
$$

requires sampling from $p_\theta$.

But sampling from $p_\theta$ is hard because:

- $p_\theta$ is only defined via an energy (unnormalized form)
- You usually need **MCMC** (Langevin dynamics, HMC, Gibbs, etc.)
- MCMC can be slow and biased if it doesn't mix well
- Doing this inside every gradient step is brutal

This is why people use:

- **Contrastive divergence / persistent CD** (approximate MCMC)
- **Score matching / denoising score matching** (avoid $Z_\theta$)
- **Noise-contrastive estimation** (reframing as classification)
- **Diffusion/score-based models** (learn $\nabla_x \log p$ directly)

---

## What's Next

If you want to go one level deeper, the natural continuation is: **derive the score matching objective's "trace(Jacobian)" form via integration by parts**, and show exactly where the $p_D$ terms drop out. That's the other half of the magic.
