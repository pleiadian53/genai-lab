# Score Matching: Overview

![Inference and Learning](https://blog.ml.cmu.edu/wp-content/uploads/2021/11/coverInferenceLearning-1.png)

![Noise to Data](https://jmtomczak.github.io/blog/16/noise_to_data.svg)

![Score Function Estimator](https://blog.shakirm.com/wp-content/uploads/2015/11/scoreFnEstimator.png)

![Score Function Visualization](https://ericmjl.github.io/score-models/notebooks/01-score-function_files/figure-html/fig-likelihood-output-1.png)

> **Score matching learns the *shape* of a data distribution by learning how probability mass locally flows, without ever computing or normalizing the probability itself.**

---

## 0. Why Score Matching is the Conceptual Hinge

Score matching sits at the **conceptual crossroads** between VAEs and diffusion/energy-based models. Understanding why requires seeing what each approach optimizes:

### The Generative Modeling Landscape

| Approach | What it learns | Core object | Key limitation |
|----------|---------------|-------------|----------------|
| **VAEs** | $p_\theta(x \mid z)$ and $q_\phi(z \mid x)$ | Latent representation $z$ | KL-to-prior tension, blurry samples |
| **Score Matching** | $\nabla_x \log p(x)$ | Gradient field in data space | No explicit latent, slow sampling |
| **Diffusion Models** | Multi-scale score functions | Denoising trajectory | Expensive iteration |
| **EBMs** | $E_\theta(x)$ (energy) | Unnormalized density | Intractable partition function |

### The Key Insight

**VAEs** ask: *"What latent code $z$ explains this data point $x$?"*

**Score matching** asks: *"Which direction does probability increase from here?"*

This shift—from **latent inference** to **gradient estimation**—is profound:

1. **No encoder needed**: Score models work directly in data space
2. **No KL divergence**: No tension between reconstruction and prior matching
3. **No normalization constant**: The score $\nabla_x \log p(x)$ cancels $Z$

### Why "Hinge"?

Score matching is the hinge because:

- It **inherits** the probabilistic foundation from likelihood-based models (like VAEs)
- It **enables** diffusion models by providing the gradient field for sampling
- It **connects to** energy-based models since $s(x) = -\nabla_x E(x)$

In other words:

$$
\text{VAE} \xrightarrow{\text{drop encoder, learn gradient}} \text{Score Matching} \xrightarrow{\text{multi-scale + Langevin}} \text{Diffusion}
$$

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $x \in \mathbb{R}^d$ | Observed data (image, gene expression, etc.) |
| $p_{\text{data}}(x)$ | Unknown true data distribution |
| $p_\theta(x)$ | Model distribution we want to learn |
| $\log p_\theta(x)$ | Log-density (unknown up to normalization) |
| $s_\theta(x)$ | **Score function**: $\nabla_x \log p_\theta(x)$ |

### Critical Distinction

- **VAEs** differentiate with respect to **parameters** $\theta$
- **Score matching** differentiates with respect to **data** $x$

This is a deep conceptual shift.

---

## 2. What is the "Score"?

If you stand at a point $x$ in data space:

- The **score** tells you **which direction probability increases fastest**
- It points "uphill" toward regions of higher density
- Its magnitude tells you *how steep* that increase is

> *"The score is a local vector field that tells us, at each point in space, which way the data distribution wants to pull samples."*

No probabilities yet. Just directions.

---

## 3. The Core Problem Score Matching Solves

Maximum likelihood wants to minimize:

$$
\mathbb{E}_{x \sim p_{\text{data}}}\left[-\log p_\theta(x)\right]
$$

But for energy-based models:

$$
\log p_\theta(x) = -E_\theta(x) - \log Z_\theta
$$

The partition function $Z_\theta = \int e^{-E_\theta(x)} dx$ is usually **intractable**.

Score matching says:

> *"If we can't compute the height of the landscape, let's learn its slope."*

Taking the gradient with respect to $x$:

$$
\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)
$$

The $\log Z_\theta$ term vanishes because it doesn't depend on $x$.

---

## 4. The Original Score Matching Objective (Hyvärinen 2005)

### Objective

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \frac{1}{2} \left\| \nabla_x \log p_\theta(x) - \nabla_x \log p_{\text{data}}(x) \right\|^2 \right]
$$

### In Words

> *"We want the gradient of the model's log-density to match the gradient of the true data log-density, on average over real data."*

But there's a problem: we don't know $\nabla_x \log p_{\text{data}}(x)$.

---

## 5. The Clever Trick: Integration by Parts

Hyvärinen showed that after integration by parts, the objective becomes:

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \left[ \frac{1}{2} \|s_\theta(x)\|^2 + \text{tr}\left(\nabla_x s_\theta(x)\right) \right]
$$

where:

- $s_\theta(x) = \nabla_x \log p_\theta(x)$ is the model's score
- $\text{tr}(\nabla_x s_\theta(x))$ is the divergence (sum of diagonal Hessian elements)

### The Miracle

> *"We can train a model to produce a vector field without ever knowing the true data density."*

The unknown $\nabla_x \log p_{\text{data}}(x)$ disappears from the objective.

---

## 6. Why Classical Score Matching is Fragile

The divergence term:

- Requires **second derivatives** (Hessian diagonal)
- Is **numerically unstable** in high dimensions
- Has **cubic complexity** in naive implementations

This is why classical score matching was beautiful but impractical for deep learning.

---

## 7. Denoising Score Matching (DSM) — The Modern Approach

### Key Idea

> *"Instead of learning the score of the data distribution directly, learn the score of noisy data distributions, which are smoother and easier."*

### Setup

Add Gaussian noise:

$$
\tilde{x} = x + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### DSM Objective

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x, \epsilon} \left[ \left\| s_\theta(\tilde{x}) + \frac{\tilde{x} - x}{\sigma^2} \right\|^2 \right]
$$

Or equivalently (in terms of $\epsilon$):

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x, \epsilon} \left[ \left\| s_\theta(\tilde{x}) + \frac{\epsilon}{\sigma} \right\|^2 \right]
$$

### In Words

> *"We corrupt a real data point with Gaussian noise. We then train a network to predict the direction that points back toward the original clean data. That direction is exactly the score of the noisy data distribution."*

---

## 8. Why DSM Works

For Gaussian corruption, the conditional score is known analytically:

$$
\nabla_{\tilde{x}} \log p(\tilde{x} \mid x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

So DSM trains:

$$
s_\theta(\tilde{x}) \approx \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})
$$

> *"The model learns how noise should be removed, infinitesimally."*

---

## 9. From DSM to Diffusion Models

If you train **many noise levels** $\sigma_1 > \sigma_2 > \cdots > \sigma_T$:

| Noise Level | What the Score Captures |
|-------------|------------------------|
| High $\sigma$ | Coarse global structure |
| Low $\sigma$ | Fine local details |

Sampling then becomes:

1. Start from pure noise $x_T \sim \mathcal{N}(0, I)$
2. Repeatedly move along the score field (gradient ascent on log-density)
3. Add small noise (Langevin dynamics)

This is exactly **diffusion / score-based generative modeling**.

> *Diffusion is not a new idea—it is multi-scale score matching + numerical integration.*

---

## 10. Comparison: VAEs vs Score Matching

### What VAEs Do

- Learn a **latent-variable model** $p_\theta(x, z) = p(z) p_\theta(x \mid z)$
- Optimize likelihood lower bound (ELBO)
- Require a prior $p(z)$ and inference model $q_\phi(z \mid x)$
- Enforce global latent geometry

### What Score Matching Does

- Learn a **vector field in data space**
- Never compute likelihood directly
- No latent variables required
- No global coordinate system imposed

### Trade-offs

| Aspect | VAE | Score Matching |
|--------|-----|----------------|
| **Sample quality** | Often blurry | High fidelity |
| **Latent space** | Explicit, interpretable | None |
| **Sampling speed** | Fast (single forward pass) | Slow (iterative) |
| **Training stability** | KL collapse issues | More stable |
| **Controllability** | Easy via latent manipulation | Harder (guidance needed) |

---

## 11. When to Use Each Approach

### Use Score Matching / Diffusion When

- Sample **fidelity** matters more than speed
- Data is high-dimensional and complex (images, audio)
- You don't need fast inference or explicit latents

### Use VAEs When

- You need **explicit latent representations** (world models, planning)
- Fast sampling is required
- Interpretability and controllability matter
- Downstream tasks need embeddings

### The Complementary View

> *VAEs and score models are complementary, not rivals.*

Modern architectures often combine them (e.g., latent diffusion models use a VAE encoder + diffusion in latent space).

---

## 12. Key Takeaway

> **VAEs learn *where* probability mass is.**
> **Score matching learns *how* probability mass flows.**

---

## 13. Next Steps on the Roadmap

1. **Langevin Dynamics**: How scores generate samples via gradient-based MCMC
2. **Diffusion Forward–Reverse Processes**: The full SDE/ODE framework
3. **Guidance and Conditioning**: Classifier-free guidance, conditional generation
4. **Latent Diffusion**: Combining VAE compression with diffusion sampling

---

## References

- Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR.
- Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation.
- Song, Y. & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.
- Song, Y. et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR.
