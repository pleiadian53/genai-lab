# Pathwise Derivatives, Stochastic Calculus, and Autodiff

This document clarifies:

1. **Determinism vs. randomness** — what backprop actually requires
2. **Pathwise derivatives vs. distributional derivatives** — two different mathematical operations
3. **Why SDEs are not a counterexample** — stochastic calculus and autodiff serve different purposes
4. **What diffusion models actually do** — and how they relate to reparameterization

---

## 1. The Precise Requirement for Backpropagation

### The Question

> "For $\partial z / \partial \mu$ to be computable, does $\mu$ need to be deterministic?"

### The Correct Answer

**Not quite.** The precise statement is:

> For $\partial z / \partial \mu$ to be computable by backprop, $z$ must be a **deterministic function** of $\mu$ **given the source of randomness**.

This is called **pathwise determinism**.

### What This Means

* Randomness is allowed
* But it must be **externalized** — treated as an input, not generated inside the computation
* Given a fixed noise sample $\epsilon$, the mapping $\mu \to z$ must be deterministic

---

## 2. What Actually Breaks Gradients in Naïve Sampling

### The Problematic Code

```python
z = Normal(mu, sigma).sample()
loss = f(z)
```

Mathematically, this means:

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

### Why Autodiff Fails

The sampling operator is **opaque** to the computational graph:

* Autograd treats `sample()` as a black box
* There is no explicit functional relationship $z = g(\mu, \sigma, \text{noise})$
* The graph sees no edge connecting $\mu$ to $z$

```text
Computational graph (naïve sampling):

    μ ──────?──────> z ──────> f(z) ──────> loss
              ↑
         no defined path
         (sampling is opaque)
```

Therefore:

$$
\frac{\partial z}{\partial \mu} \quad \text{is undefined in autodiff}
$$

**Not because math forbids it** — but because the path is not represented in the graph.

---

## 3. What the Reparameterization Trick Actually Does

Rewrite the same random variable as:

$$
\epsilon \sim \mathcal{N}(0, I) \quad \text{(sampled once, treated as input)}
$$

$$
z = \mu + \sigma \cdot \epsilon \quad \text{(deterministic given } \epsilon \text{)}
$$

### Key Properties

* Randomness is now **explicit** and **external**
* $z$ is a deterministic function of $(\mu, \sigma, \epsilon)$
* $\epsilon$ is treated as an input, not a parameter to differentiate through

```text
Computational graph (reparameterized):

    ε (external input)
         ↓
    μ ──────> z = μ + σ·ε ──────> f(z) ──────> loss
         ↑
    σ ───┘

    Clear paths: ∂z/∂μ = 1, ∂z/∂σ = ε
```

Now the gradient is well-defined:

$$
\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon
$$

### The Critical Distinction

> We are **not differentiating through randomness**.
> We are differentiating through a **deterministic function that uses randomness as input**.

This distinction is everything.

---

## 4. Two Different Notions of "Stochastic Derivative"

This is where confusion often arises. There are **two fundamentally different operations** that both involve "derivatives" and "randomness":

### (A) Pathwise Derivatives — What ML Uses

**Goal**: Compute $\nabla_\theta \mathbb{E}_{z \sim p_\theta}[f(z)]$

**Method**:

1. Reparameterize: $z = g_\theta(\epsilon)$ where $\epsilon \sim p(\epsilon)$ is fixed
2. Differentiate the deterministic function $g_\theta$
3. Average over noise samples

$$
\nabla_\theta \mathbb{E}_{z \sim p_\theta}[f(z)] = \mathbb{E}_{\epsilon}\left[\nabla_\theta f(g_\theta(\epsilon))\right]
$$

**Key property**: Given $\epsilon$, everything is deterministic. Standard chain rule applies.

### (B) Stochastic Calculus — What SDEs Use

**Goal**: Define dynamics driven by continuous noise (Brownian motion)

**Method**: Itô or Stratonovich calculus — special rules for integrating against nowhere-differentiable processes

$$
dX_t = b(X_t)\,dt + \sigma(X_t)\,dW_t
$$

**Key property**: $W_t$ is nowhere differentiable. "Derivatives" are defined in an integral/distributional sense.

### Comparison Table

| Aspect | Pathwise (ML) | Stochastic Calculus (SDEs) |
|--------|---------------|---------------------------|
| **Noise** | Fixed sample $\epsilon$ | Continuous process $W_t$ |
| **Derivative** | Standard chain rule | Itô's lemma |
| **Result** | Deterministic gradient | Distribution over paths |
| **Use case** | Backprop, VAEs, RL | Physics, finance, diffusion |

---

## 5. How SDEs Actually Handle Randomness

Consider a stochastic differential equation:

$$
dX_t = b(X_t)\,dt + \sigma(X_t)\,dW_t
$$

### Key Facts About SDEs

1. **$W_t$ (Brownian motion) is nowhere differentiable** — you cannot write $dW_t/dt$
2. **Individual sample paths are not classically differentiable**
3. **SDEs are defined in an integral sense**, not as pointwise derivatives

### What the Notation Actually Means

When we write $dX_t$, we are **not** taking a derivative. We are defining:

$$
X_t = X_0 + \int_0^t b(X_s)\,ds + \int_0^t \sigma(X_s)\,dW_s
$$

The second integral is a **stochastic integral** (Itô integral), which requires special rules because $W_t$ has infinite variation.

### Itô's Lemma — The Chain Rule for SDEs

For a function $f(X_t)$ where $X_t$ follows an SDE:

$$
df = \frac{\partial f}{\partial x}\,dX + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}\sigma^2\,dt
$$

The extra $\frac{1}{2}\sigma^2 f''$ term is the **Itô correction** — it arises because $dW_t \cdot dW_t = dt$ (quadratic variation).

This is fundamentally different from the standard chain rule.

---

## 6. Why SDE Theory Doesn't Help Naïve Backprop

### What Autodiff Requires

* Explicit functional dependencies in a computational graph
* Pathwise gradients via chain rule
* Deterministic mappings given all inputs

### What SDE Theory Provides

* Weak/distributional derivatives
* Distributions over paths
* Expectation-level results (e.g., Fokker-Planck equations)

**These live in different mathematical worlds.**

SDE machinery does not automatically give you gradients for:

```python
z = sample(mu)  # Still opaque to autodiff
```

You still need to reparameterize to get pathwise gradients.

---

## 7. How Diffusion Models Actually Work

Diffusion models use SDE language but train with **pathwise derivatives**.

### The Forward Process (Adding Noise)

$$
dx_t = -\frac{1}{2}\beta(t)x_t\,dt + \sqrt{\beta(t)}\,dW_t
$$

This is an SDE — but we don't backprop through it. We just use it to generate noisy training data.

### The Reverse Process (Denoising)

$$
dx_t = \left[-\frac{1}{2}\beta(t)x_t - \beta(t)\nabla_x \log p_t(x_t)\right]dt + \sqrt{\beta(t)}\,d\bar{W}_t
$$

The score $\nabla_x \log p_t(x)$ is approximated by a neural network $s_\theta(x, t)$.

### How Training Works

**Key insight**: Training does NOT backprop through the SDE.

Instead, it uses a **denoising score matching** objective:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t | x_0)\|^2\right]
$$

where $x_t = \alpha_t x_0 + \sigma_t \epsilon$ is a **reparameterized** noisy sample.

### The Connection to Reparameterization

```python
# Diffusion training (simplified)
epsilon = torch.randn_like(x0)           # External noise
x_t = alpha_t * x0 + sigma_t * epsilon   # Reparameterized!
predicted_noise = model(x_t, t)
loss = F.mse_loss(predicted_noise, epsilon)
loss.backward()                          # Pathwise gradient!
```

This is the reparameterization trick at scale:

* Noise $\epsilon$ is sampled externally
* $x_t$ is a deterministic function of $(x_0, \epsilon, t)$
* Gradients flow through the deterministic path

### Sampling (Inference)

At inference, we do solve an SDE/ODE:

$$
dx_t = \left[-\frac{1}{2}\beta(t)x_t - \beta(t)s_\theta(x_t, t)\right]dt
$$

But we don't need gradients here — just forward simulation.

---

## 8. Reparameterization in Other Domains

### Reinforcement Learning (SAC, TD3)

$$
a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Exactly the same trick — externalize noise, differentiate through the deterministic path.

### Bayesian Neural Networks

$$
w = \mu + \sigma \cdot \epsilon
$$

Weight uncertainty becomes differentiable.

### Normalizing Flows

$$
z = f_\theta(\epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

Pure reparameterization — all expressivity in the transformation.

---

## 9. Summary: Two Worlds, One Bridge

### The Core Insight

> **Backpropagation requires deterministic paths in the computational graph.**
> **The reparameterization trick works by making randomness an explicit input, so that the mapping from parameters to samples is deterministic conditioned on that noise.**

### The Companion Insight

> **Stochastic calculus defines derivatives in distribution or expectation, not in the pathwise sense required by autodiff.**

No contradiction — just different tools for different jobs.

### Visual Summary

```text
┌─────────────────────────────────────────────────────────────────┐
│  PATHWISE DERIVATIVES (what ML uses)                            │
│                                                                  │
│  1. Sample noise ε ~ N(0,I) once                                │
│  2. Compute z = g_θ(ε) deterministically                        │
│  3. Backprop through g_θ using standard chain rule              │
│  4. Average gradients over many ε samples                       │
│                                                                  │
│  Result: ∇_θ E[f(z)] ≈ (1/N) Σ ∇_θ f(g_θ(εᵢ))                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STOCHASTIC CALCULUS (what SDEs use)                            │
│                                                                  │
│  1. Define dynamics: dX = b(X)dt + σ(X)dW                       │
│  2. W_t is nowhere differentiable                               │
│  3. Use Itô's lemma (modified chain rule)                       │
│  4. Results are distributions over paths                        │
│                                                                  │
│  Result: Fokker-Planck equations, path distributions            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Intuition Check

Think of randomness like an input image:

* You don't differentiate **with respect to the pixels**
* You differentiate **with respect to the parameters that process the image**

Reparameterization treats noise the same way:

* $\epsilon$ is like input data — fixed during the forward/backward pass
* $\theta$ is what we optimize — gradients flow through the deterministic transformation

---

## References

* [VAE-04-reparameterization.md](VAE-04-reparameterization.md) — The reparameterization trick
* [VAE-QA.md](VAE-QA.md) — Why the prior matters
* Kingma & Welling (2014) — "Auto-Encoding Variational Bayes"
* Song et al. (2021) — "Score-Based Generative Modeling through SDEs"
