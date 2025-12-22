# Deriving the Pathwise Gradient Estimator

This document builds the **pathwise (reparameterization) gradient estimator** step by step, then explains why the simple derivative $\partial z / \partial \mu = 1$ is the key insight.

---

## Step 0: The Object We Want to Differentiate

We want gradients of an expectation where the distribution depends on parameters:

$$
J(\phi) = \mathbb{E}_{z \sim q_\phi(z)}\left[f(z)\right]
$$

For VAEs: $f(z) = \log p_\theta(x \mid z)$, and $q_\phi$ is $q_\phi(z \mid x)$.

---

## Step 1: Reparameterize the Random Variable

Assume we can write samples as a deterministic transform of parameter-free noise:

$$
\epsilon \sim p(\epsilon) \quad \text{(does NOT depend on } \phi \text{)}
$$

$$
z = g_\phi(\epsilon)
$$

**Example (Gaussian)**: $g_\phi(\epsilon) = \mu_\phi + \sigma_\phi \odot \epsilon$, with $\epsilon \sim \mathcal{N}(0, I)$.

Then:

$$
J(\phi) = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[f(g_\phi(\epsilon))\right]
$$

---

## Step 2: Differentiate Under the Expectation (The "Pathwise" Move)

Because $p(\epsilon)$ doesn't depend on $\phi$, we can move the gradient inside:

$$
\nabla_\phi J(\phi) = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[\nabla_\phi f(g_\phi(\epsilon))\right]
$$

Now apply the chain rule:

$$
\nabla_\phi f(g_\phi(\epsilon)) = \underbrace{\nabla_z f(z)}_{\text{gradient of downstream loss}} \bigg|_{z=g_\phi(\epsilon)} \cdot \underbrace{\nabla_\phi g_\phi(\epsilon)}_{\text{how sample moves w.r.t. } \phi}
$$

So the **pathwise gradient estimator** is:

$$
\nabla_\phi J(\phi) = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[\nabla_z f(g_\phi(\epsilon)) \cdot \nabla_\phi g_\phi(\epsilon)\right]
$$

And the Monte Carlo estimator with $K$ samples is:

$$
\nabla_\phi J(\phi) \approx \frac{1}{K} \sum_{k=1}^{K} \nabla_z f(g_\phi(\epsilon_k)) \cdot \nabla_\phi g_\phi(\epsilon_k), \quad \epsilon_k \sim p(\epsilon)
$$

That's the whole method: **differentiate through the sampling path**.

---

## Exercise: The Gaussian Case

Take the Gaussian case:

$$
z = \mu + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

Assume $\phi$ parameterizes $\mu$ and $\sigma$.

**Question**: What is $\frac{\partial z}{\partial \mu}$?

**Answer**: It's **1**.

And that simple answer is actually the *entire reason* the reparameterization trick works.

---

## 1. Why $\partial z / \partial \mu = 1$ Matters

You have:

$$
z = \mu + \sigma \epsilon
$$

Treat $\epsilon$ as a fixed input (a sampled number) during backprop.

Then:

$$
\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \epsilon
$$

These derivatives are:

* **Well-defined**
* **Finite**
* **Independent of probability theory**

They are just ordinary calculus.

This is the key:

> **Once the randomness is made explicit, the gradient is completely classical.**

---

## 2. Contrast with Naïve Sampling

If instead you wrote:

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

there is *no expression* for:

$$
\frac{\partial z}{\partial \mu}
$$

because:

* "Sampling" is not a mathematical function
* It hides randomness inside an opaque operation

Autodiff has nothing to differentiate.

So it's not that the derivative "should" be 1 mathematically — it's that **you never gave the system a function to differentiate**.

---

## 3. What Gradient Actually Flows in a VAE

Putting it together, the gradient w.r.t. $\mu$ becomes:

$$
\nabla_\mu \mathbb{E}[f(z)] \approx \frac{1}{K} \sum_{k=1}^{K} \underbrace{\nabla_z f(z_k)}_{\text{decoder gradient}} \cdot \underbrace{\frac{\partial z_k}{\partial \mu}}_{= 1}
$$

So gradients flow **straight through the sample**.

* No likelihood ratios
* No REINFORCE
* No variance explosion

---

## 4. Why It's Called a "Pathwise" Derivative

Each sampled $\epsilon_k$ defines a **path**:

$$
\mu \longrightarrow z_k \longrightarrow f(z_k)
$$

You differentiate **along that path**.

That's why the name "pathwise gradient estimator" is more descriptive than "reparameterization trick".

---

## 5. A Subtle but Crucial Point

During backprop:

* $\epsilon$ is treated as a **constant**
* We are *not* differentiating randomness
* We are differentiating a deterministic computation *conditioned on noise*

This is why the intuition about "determinism" is right, but needs refinement: **pathwise determinism**, not absolute determinism.

---

## 6. Where This Fails

This only works when:

* You can write $z = g_\phi(\epsilon)$
* With $\epsilon$ independent of $\phi$

That's why:

* **Discrete sampling breaks it** — no continuous path to differentiate
* **Gumbel-Softmax is an approximation** — continuous relaxation of discrete
* **Score-function estimators exist as a fallback** — REINFORCE for non-reparameterizable cases

---

## 7. Differentiating Sample Paths in SDEs (Clarification)

There's potential confusion between "pathwise derivatives" in ML and "sample paths" in SDEs. Let's clarify.

### In ML: "Pathwise" = Along a Fixed Noise Realization

Given a fixed $\epsilon$, we have a deterministic path:

$$
\theta \xrightarrow{g_\theta} z \xrightarrow{f} \text{loss}
$$

We differentiate this path using the standard chain rule. The word "path" refers to **the computational graph path**.

### In SDEs: "Sample Path" = A Realization of a Stochastic Process

A sample path $\{X_t(\omega)\}_{t \geq 0}$ is one realization of the process, indexed by outcome $\omega$.

Key differences:

| Aspect | ML Pathwise | SDE Sample Path |
|--------|-------------|-----------------|
| **What varies** | Parameters $\theta$ | Time $t$ |
| **Noise** | Fixed $\epsilon$ | Continuous $W_t(\omega)$ |
| **Differentiability** | Standard calculus | Nowhere differentiable in $t$ |
| **Goal** | $\nabla_\theta \mathbb{E}[f(z)]$ | Describe evolution $X_t$ |

### Why SDEs Need Special Calculus

For an SDE sample path $X_t(\omega)$:

$$
X_t = X_0 + \int_0^t b(X_s)\,ds + \int_0^t \sigma(X_s)\,dW_s
$$

* The path $t \mapsto X_t(\omega)$ is **continuous but nowhere differentiable**
* You cannot write $dX_t/dt$ in the classical sense
* Itô calculus provides rules for manipulating these objects

### The Confusion Resolved

When ML papers say "pathwise gradient," they mean:

> Differentiate w.r.t. parameters along a fixed noise realization

When SDE papers say "sample path," they mean:

> One realization of a continuous-time stochastic process

These are **different uses of the word "path"**:

* ML: path through the computational graph
* SDEs: path through state space over time

### Can We Differentiate SDE Sample Paths w.r.t. Parameters?

Yes! This is called **sensitivity analysis** or **pathwise sensitivity**:

$$
\frac{\partial X_t}{\partial \theta}
$$

where $\theta$ parameterizes the drift $b_\theta$ or diffusion $\sigma_\theta$.

This requires solving an auxiliary SDE (the **sensitivity equation**):

$$
d\left(\frac{\partial X_t}{\partial \theta}\right) = \frac{\partial b}{\partial x}\frac{\partial X_t}{\partial \theta}\,dt + \frac{\partial b}{\partial \theta}\,dt + \text{(diffusion terms)}
$$

This is computationally expensive and rarely used in ML. Instead, diffusion models use:

* **Denoising score matching** — avoids differentiating through the SDE
* **Reparameterization at each timestep** — $x_t = \alpha_t x_0 + \sigma_t \epsilon$

---

## 8. Summary

> **The reparameterization trick works because it turns sampling into a differentiable computation graph with respect to model parameters.**

The key insight: $\partial z / \partial \mu = 1$ is just ordinary calculus once you externalize the noise.

---

## References

* [VAE-05-followup-1.md](VAE-05-followup-1.md) — Pathwise derivatives vs. stochastic calculus
* [VAE-04-reparameterization.md](VAE-04-reparameterization.md) — The reparameterization trick
* Kingma & Welling (2014) — "Auto-Encoding Variational Bayes"
* Mohamed et al. (2020) — "Monte Carlo Gradient Estimation in Machine Learning"
