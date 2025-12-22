# Energy Functions in Generative Modeling

> **The energy function is the unifying language of generative models—from Boltzmann machines to diffusion models to JEPA.**

---

## 0. Motivation: Why Energy?

The concept of "energy" comes from statistical physics, where systems naturally settle into low-energy states. This intuition transfers beautifully to machine learning:

- **Low energy** = high probability (likely configurations)
- **High energy** = low probability (unlikely configurations)

Energy-based thinking provides a **unified framework** for understanding:

| Model Family | How Energy Appears |
|--------------|-------------------|
| **Boltzmann Machines** | Explicit energy function over binary states |
| **Score Matching** | Score = negative gradient of energy |
| **Diffusion Models** | Denoising = following energy gradients |
| **Contrastive Learning** | Energy difference between positive/negative pairs |
| **JEPA** | Energy measures prediction error in latent space |

---

## 1. Notation Clarification: $\mathbb{E}[\cdot]$ vs $E_\theta(x)$

A common source of confusion: both use the letter "E" but mean completely different things.

### Expectation: $\mathbb{E}[\cdot]$

$$
\mathbb{E}_{x \sim p(x)}[f(x)] = \int f(x) \, p(x) \, dx
$$

> *"The average value of $f(x)$ when $x$ is drawn from distribution $p(x)$."*

### Energy Function: $E_\theta(x)$

$$
E_\theta(x) : \mathbb{R}^d \to \mathbb{R}
$$

> *"A scalar-valued function that assigns an 'energy' (unnormalized negative log-probability) to each configuration $x$."*

**Key rule**:

| Symbol | Meaning | Type |
|--------|---------|------|
| $\mathbb{E}[\cdot]$ | Expectation (average) | Operator |
| $E_\theta(x)$ | Energy function | Scalar function |

They are **completely unrelated** despite sharing a letter.

---

## 2. The Boltzmann Distribution

In an energy-based model, probability is defined via the **Boltzmann distribution**:

$$
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}
$$

where:

- $E_\theta(x)$: energy of configuration $x$
- $Z_\theta$: partition function (normalization constant)

### Intuition

| Energy | Probability | Interpretation |
|--------|-------------|----------------|
| Low $E_\theta(x)$ | High $p_\theta(x)$ | Likely, stable configuration |
| High $E_\theta(x)$ | Low $p_\theta(x)$ | Unlikely, unstable configuration |

The exponential ensures:

- Probabilities are always positive
- Small energy differences create large probability ratios

---

## 3. The Partition Function Problem

### Definition

$$
Z_\theta = \int e^{-E_\theta(x)} \, dx
$$

This integral sums the unnormalized probability mass over **all possible configurations**.

### Why It's Intractable

For high-dimensional $x$ (images, gene expression vectors):

- The integral is over $\mathbb{R}^d$ where $d$ can be thousands or millions
- No closed-form solution exists for neural network energies
- Monte Carlo estimation requires samples from $p_\theta(x)$—a chicken-and-egg problem

This is **the fundamental computational bottleneck** of energy-based models.

---

## 4. From Energy to Log-Probability

Starting from the Boltzmann distribution:

$$
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}
$$

Take the logarithm:

$$
\log p_\theta(x) = \log e^{-E_\theta(x)} - \log Z_\theta
$$

$$
\log p_\theta(x) = -E_\theta(x) - \log Z_\theta
$$

This is the key equation connecting energy to log-probability.

---

## 5. The Score Cancels the Partition Function

Here's the crucial insight that makes score matching work.

The **score** is defined as:

$$
s(x) = \nabla_x \log p_\theta(x)
$$

Substituting our expression for log-probability:

$$
\nabla_x \log p_\theta(x) = \nabla_x \left( -E_\theta(x) - \log Z_\theta \right)
$$

$$
= -\nabla_x E_\theta(x) - \nabla_x \log Z_\theta
$$

But $Z_\theta$ is a constant with respect to $x$ (it integrates over all $x$), so:

$$
\nabla_x \log Z_\theta = 0
$$

Therefore:

$$
\boxed{s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)}
$$

### The Miracle

> *"The score is just the negative gradient of the energy. The intractable partition function disappears completely."*

This is **why score matching works**—we never need to compute $Z_\theta$.

---

## 6. Energy Landscapes: A Visual Intuition

Think of $E_\theta(x)$ as defining a landscape over data space:

```text
Energy
  ^
  |     /\
  |    /  \      /\
  |   /    \    /  \
  |  /      \  /    \
  | /        \/      \
  +-------------------> x
       valleys = data modes (low energy, high probability)
       peaks = unlikely regions (high energy, low probability)
```

- **Score vectors** point downhill (toward lower energy / higher probability)
- **Sampling** = rolling a ball downhill with some noise (Langevin dynamics)
- **Training** = shaping the landscape so valleys align with data

---

## 7. Historical Context: From Physics to ML

The energy-based view has deep roots:

| Era | Development |
|-----|-------------|
| **1900s** | Boltzmann: Statistical mechanics, partition functions |
| **1980s** | Hopfield networks, Boltzmann machines |
| **2000s** | Contrastive divergence, RBMs, deep belief networks |
| **2010s** | Score matching (Hyvärinen), noise-contrastive estimation |
| **2020s** | Diffusion models, energy-based priors, JEPA |

The energy perspective keeps returning because it's **mathematically natural** for describing probability without normalization.

---

## 8. Foreshadowing: Energy in Modern Generative Models

### 8.1 Diffusion Models

Diffusion models are **implicitly energy-based**. The denoising network learns:

$$
\epsilon_\theta(x_t, t) \approx -\sigma_t \nabla_{x_t} \log p_t(x_t)
$$

This is exactly the score! The connection:

$$
\text{Denoising direction} = \text{Score} = -\nabla E
$$

Diffusion sampling is **gradient descent on a time-varying energy landscape**.

### 8.2 Energy-Based Models (Modern)

Recent work trains EBMs directly using:

- **Contrastive divergence**: Compare real vs. model samples
- **Score matching**: Avoid partition function entirely
- **Noise-contrastive estimation**: Classify real vs. noise

EBMs are making a comeback for:

- Composable generation (add energies = combine concepts)
- Out-of-distribution detection (high energy = anomaly)
- Hybrid models (EBM prior + VAE decoder)

### 8.3 JEPA (Joint Embedding Predictive Architecture)

Yann LeCun's JEPA uses energy in a fundamentally different way:

$$
E(x, y) = \| s_y - \text{Predictor}(s_x) \|^2
$$

where $s_x, s_y$ are learned representations.

Key differences from generative EBMs:

| Aspect | Generative EBM | JEPA |
|--------|---------------|------|
| **Space** | Data space $x$ | Latent space $s$ |
| **Goal** | Model $p(x)$ | Learn representations |
| **Energy meaning** | Unnormalized log-prob | Prediction error |
| **Sampling** | Generate new $x$ | Not the goal |

JEPA's insight: **You don't need to model pixel-level probability to learn useful representations.**

### 8.4 Contrastive Learning

InfoNCE and similar objectives are energy-based:

$$
\mathcal{L} = -\log \frac{e^{-E(x, x^+)}}{e^{-E(x, x^+)} + \sum_j e^{-E(x, x^-_j)}}
$$

- Low energy for positive pairs (similar)
- High energy for negative pairs (dissimilar)

---

## 9. Summary: The Energy Perspective

| Concept | Definition | Role |
|---------|------------|------|
| $E_\theta(x)$ | Energy function | Unnormalized negative log-probability |
| $Z_\theta$ | Partition function | Normalization constant (intractable) |
| $p_\theta(x)$ | Probability | $\propto e^{-E_\theta(x)}$ |
| $s_\theta(x)$ | Score | $-\nabla_x E_\theta(x)$ (no $Z$!) |

### The Core Insight

> **Energy-based thinking lets us work with unnormalized probabilities. Score matching lets us learn without ever computing the normalization.**

---

## 10. Next Steps on the Roadmap

1. **Langevin Dynamics**: How to sample by following score/energy gradients
2. **Contrastive Divergence**: Training EBMs with MCMC
3. **Diffusion as Energy**: The SDE/ODE perspective
4. **JEPA Deep Dive**: Energy in representation learning

---

## References

- Hopfield, J. (1982). *Neural Networks and Physical Systems with Emergent Collective Computational Abilities*. PNAS.
- Hinton, G. (2002). *Training Products of Experts by Minimizing Contrastive Divergence*. Neural Computation.
- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR.
- LeCun, Y. et al. (2006). *A Tutorial on Energy-Based Learning*. In Predicting Structured Data.
- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
