# Denoising Diffusion Probabilistic Models (DDPM): Foundations

This document provides a comprehensive mathematical introduction to DDPM, the foundational discrete-time diffusion model introduced by Ho et al. (2020).

---

## Overview

**Denoising Diffusion Probabilistic Models (DDPM)** are a class of generative models that learn to generate data by reversing a gradual noising process. They achieve state-of-the-art results in image generation and have been successfully applied to various domains including gene expression, protein design, and molecular generation.

### Key Idea

1. **Forward process**: Gradually add Gaussian noise to data over $T$ steps until it becomes pure noise
2. **Reverse process**: Learn to denoise, step by step, starting from pure noise
3. **Training**: Predict the noise added at each step (equivalently, predict the score)

### Why DDPM Matters

- **Theoretical foundation**: Connects variational inference, score matching, and SDEs
- **Training stability**: Simple MSE loss, no adversarial training
- **Sample quality**: State-of-the-art FID scores on image generation
- **Flexibility**: Works for continuous, discrete, and structured data
- **Interpretability**: Clear probabilistic interpretation via ELBO

---

## The Forward Process (Data → Noise)

### Definition

The forward process is a **fixed Markov chain** that gradually adds Gaussian noise to data $x_0 \sim q(x_0)$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

where:
- $t = 1, 2, \ldots, T$ (typically $T = 1000$)
- $\beta_t \in (0, 1)$ is the **variance schedule** (how much noise to add)
- $\beta_1 < \beta_2 < \cdots < \beta_T$ (increasing noise)

### Intuition

At each step:
- Keep $\sqrt{1 - \beta_t}$ of the previous signal
- Add $\sqrt{\beta_t}$ of fresh noise

As $t \to T$, the signal becomes pure Gaussian noise.

### Reparameterization

Using the reparameterization trick:

$$
x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)
$$

---

## Closed-Form Forward Process

### Key Insight

Because each step is Gaussian and the process is Markovian, we can jump directly from $x_0$ to any $x_t$ without computing intermediate steps.

### Notation

Define:
- $\alpha_t := 1 - \beta_t$ (signal retention coefficient)
- $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$ (cumulative product)

### Closed-Form Marginal

$$
\boxed{q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)}
$$

**Reparameterization form**:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### Derivation

By induction:

**Base case** ($t=1$):
$$
x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1 - \alpha_1} \epsilon_0 = \sqrt{\bar{\alpha}_1} x_0 + \sqrt{1 - \bar{\alpha}_1} \epsilon_0
$$

**Inductive step**: Assume $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}_t$.

Then:
$$
\begin{align}
x_{t+1} &= \sqrt{\alpha_{t+1}} x_t + \sqrt{1 - \alpha_{t+1}} \epsilon_t \\
&= \sqrt{\alpha_{t+1}} \left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \bar{\epsilon}_t\right) + \sqrt{1 - \alpha_{t+1}} \epsilon_t \\
&= \sqrt{\alpha_{t+1} \bar{\alpha}_t} x_0 + \sqrt{\alpha_{t+1}(1 - \bar{\alpha}_t)} \bar{\epsilon}_t + \sqrt{1 - \alpha_{t+1}} \epsilon_t
\end{align}
$$

The two noise terms combine (sum of independent Gaussians):
$$
\sqrt{\alpha_{t+1}(1 - \bar{\alpha}_t)} \bar{\epsilon}_t + \sqrt{1 - \alpha_{t+1}} \epsilon_t \sim \mathcal{N}(0, [\alpha_{t+1}(1 - \bar{\alpha}_t) + (1 - \alpha_{t+1})] I)
$$

Simplify the variance:
$$
\alpha_{t+1}(1 - \bar{\alpha}_t) + (1 - \alpha_{t+1}) = \alpha_{t+1} - \alpha_{t+1}\bar{\alpha}_t + 1 - \alpha_{t+1} = 1 - \alpha_{t+1}\bar{\alpha}_t = 1 - \bar{\alpha}_{t+1}
$$

Therefore:
$$
x_{t+1} = \sqrt{\bar{\alpha}_{t+1}} x_0 + \sqrt{1 - \bar{\alpha}_{t+1}} \epsilon
$$

---

## The Reverse Process (Noise → Data)

### Goal

Learn to reverse the forward process: $p_\theta(x_{t-1} \mid x_t)$.

If we can sample $x_T \sim \mathcal{N}(0, I)$ and iteratively apply the reverse process, we can generate new data.

### Parameterization

Model the reverse process as a Markov chain:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
$$

where:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

### Key Question

What should $\mu_\theta$ and $\Sigma_\theta$ be?

**Answer**: We'll derive them from the **posterior** $q(x_{t-1} \mid x_t, x_0)$.

---

## Posterior Distribution

### Bayes' Rule

Using Bayes' rule:

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

Since the forward process is Markovian, $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$.

### Gaussian Posterior

All three terms are Gaussian, so the posterior is also Gaussian. We can compute it in closed form.

**Result**:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

where:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

### Derivation

We have:
- $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I)$
- $q(x_{t-1} \mid x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1 - \bar{\alpha}_{t-1}) I)$
- $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$

Using the formula for the product of Gaussians (completing the square), we get the result above.

**Key insight**: The posterior mean is a **weighted average** of predictions from $x_t$ and $x_0$.

---

## Training Objective: The ELBO

### Variational Lower Bound

DDPM is trained by maximizing the evidence lower bound (ELBO):

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T} \mid x_0)} \left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]
$$

### Decomposition

The ELBO can be decomposed into three terms:

$$
\mathcal{L} = \mathbb{E}_q \left[\underbrace{D_{KL}(q(x_T \mid x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0}\right]
$$

**Interpretation**:
- $L_T$: How close is $q(x_T \mid x_0)$ to $p(x_T) = \mathcal{N}(0, I)$? (Usually negligible)
- $L_{t-1}$: How well does $p_\theta$ match the true posterior $q$?
- $L_0$: Reconstruction term (discrete decoder or continuous likelihood)

### Simplification

Since both $q(x_{t-1} \mid x_t, x_0)$ and $p_\theta(x_{t-1} \mid x_t)$ are Gaussian, the KL divergence has a closed form:

$$
L_{t-1} = \mathbb{E}_{q(x_t, x_0)} \left[\frac{1}{2\sigma_t^2} \|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\right] + \text{const}
$$

---

## Noise Prediction Parameterization

### Key Insight

Instead of directly predicting $\mu_\theta(x_t, t)$, we can predict the **noise** $\epsilon$ that was added.

### Reparameterization

Recall:
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

Solving for $x_0$:
$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)
$$

Substituting into $\tilde{\mu}_t$:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon\right)
$$

### Noise Prediction Network

Define:
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)
$$

where $\epsilon_\theta(x_t, t)$ is a neural network that predicts the noise.

### Simplified Loss

The ELBO simplifies to:

$$
\boxed{L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]}
$$

where:
- $t \sim \text{Uniform}(\{1, \ldots, T\})$
- $x_0 \sim q(x_0)$
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

**This is just MSE on noise prediction!**

---

## Algorithm Summary

### Training

```
1. Sample x_0 ~ q(x_0)
2. Sample t ~ Uniform({1, ..., T})
3. Sample ε ~ N(0, I)
4. Compute x_t = sqrt(α_bar_t) * x_0 + sqrt(1 - α_bar_t) * ε
5. Compute loss = ||ε - ε_θ(x_t, t)||²
6. Update θ via gradient descent
```

### Sampling

```
1. Sample x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. Predict noise: ε_θ(x_t, t)
   b. Compute mean: μ_θ = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-α_bar_t)) * ε_θ)
   c. Sample: x_{t-1} = μ_θ + σ_t * z  (z ~ N(0,I) if t>1, else z=0)
3. Return x_0
```

---

## Variance Schedule

### Linear Schedule (Original DDPM)

$$
\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)
$$

Typical values: $\beta_1 = 10^{-4}$, $\beta_T = 0.02$

### Cosine Schedule (Improved DDPM)

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

where $s = 0.008$ is a small offset.

**Advantages**:
- More uniform signal-to-noise ratio across timesteps
- Better sample quality
- Fewer steps needed

---

## Connection to Score Matching

### Score Function

The **score** is the gradient of the log density:

$$
\nabla_{x_t} \log q(x_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon
$$

### Equivalence

Predicting noise $\epsilon$ is equivalent to predicting the score (up to a constant):

$$
\epsilon_\theta(x_t, t) \approx \epsilon \quad \Leftrightarrow \quad -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t)
$$

**Therefore**: DDPM training is **score matching** with a specific noise schedule.

---

## Summary

We derived DDPM from first principles:

1. **Forward process**: Fixed Markov chain adding Gaussian noise
2. **Closed-form marginal**: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
3. **Reverse process**: Learned Gaussian transitions
4. **Training objective**: Variational lower bound (ELBO)
5. **Simplified loss**: MSE on noise prediction
6. **Connection to score matching**: Noise prediction ≈ score prediction

**Key insights**:
- Training is simple: predict the noise that was added
- Sampling is iterative denoising
- The model learns the score function implicitly
- DDPM is a discrete-time approximation of continuous SDEs

---

## Related Documents

- [DDPM Training Details](02_ddpm_training.md) — Loss functions, architectures, conditioning
- [DDPM Sampling Methods](03_ddpm_sampling.md) — DDPM, DDIM, ancestral sampling
- [From DDPM to VP-SDE](../SDE/02c_ddpm_to_vpsde.md) — Continuous-time limit
- [VP-SDE to DDPM](../SDE/02_sde_and_ddpm.md) — Discretization perspective

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
2. Sohl-Dickstein, J., et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. ICML.
3. Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. NeurIPS.
4. Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. ICML.
