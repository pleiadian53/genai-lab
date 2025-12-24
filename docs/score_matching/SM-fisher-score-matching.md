# Fisher Score Matching for Likelihood-Free Inference

This document is a tutorial-style walkthrough of the key ideas from the paper *"Direct Fisher Score Estimation for Likelihood Maximization"* (Khoo et al., 2025), explaining how score matching ideas extend to **parameter-space gradients** for simulation-based inference.

---

## The Problem: Implicit Simulators

Many scientific models (biology, physics, cosmology, neuroscience, etc.) are **implicit simulators**:

- You can **simulate data** $x \sim p(x|\theta)$
- But you **cannot evaluate the likelihood** $p(x|\theta)$ or its gradient

This is called **Simulation-Based Inference (SBI)**.

If you want to do **maximum likelihood estimation (MLE)**, you need the **Fisher score**:

$$
\nabla_\theta \log p(x|\theta)
$$

But this derivative is unavailable because the likelihood is unknown.

---

## Main Idea: Local Fisher Score Matching

The authors propose **Direct Fisher Score Estimation** via a new method called **Local Fisher Score Matching (FSM)**.

FSM **directly estimates the Fisher score** using only:

- Samples from a local region around the current parameter $\theta_t$
- A simple linear regression model

No likelihoods, no densities, no gradients of the simulator are needed.

This enables a **gradient-based MLE method** in fully likelihood-free settings.

The method works *sequentially*:

1. At parameter iterate $\theta_t$, draw nearby parameters from a Gaussian
2. Simulate data at those parameters
3. Fit a local surrogate model $S_W(x) \approx \nabla_\theta \log p(x|\theta_t)$
4. Use this surrogate to take a gradient step in $\theta$

---

## Why Score Matching?

Score matching is a classical method for training energy-based models when the normalizing constant is intractable.

The *ordinary* score matching objective is:

$$
\mathbb{E}_{x \sim p_D} \left[ \frac{1}{2} |s_\theta(x) - \nabla_x \log p_D(x)|^2 \right]
$$

But FSM **adapts** the idea in a novel way:

| Approach | Differentiate w.r.t. | Estimates |
|----------|---------------------|----------|
| **Original score matching** | Data $x$ | Stein score $\nabla_x \log p_\theta(x)$ |
| **Fisher score matching** | Parameters $\theta$ | Fisher score $\nabla_\theta \log p_\theta(x)$ |

This is non-standard and the main conceptual innovation of the paper.

---

## Background: Score Matching (Section 2.1)

Score matching solves density estimation without computing the normalizing constant $Z_\theta$.

### Energy-Based Model (EBM)

$$
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}
$$

The score is:

$$
s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)
$$

### The explicit score matching loss

$$
L_{\text{ESM}} = \mathbb{E}_{x \sim p_D} \left[ \frac{1}{2}|s_\theta(x)|^2 + \mathrm{tr}(\nabla_x s_\theta(x)) \right]
$$

No need to compute $Z_\theta$! But it still requires computing Jacobians/Hessians, which may be expensive.

This background is crucial because **FSM uses a *parameter-space analogue* of this trick**.

---

## How FSM Modifies Score Matching

FSM wants to estimate:

$$
g(x; \theta_t) := \nabla_\theta \log p(x|\theta)\big|_{\theta=\theta_t}
$$

But this is unknown.

So the authors define a **joint distribution over data and parameters**:

- Sample parameters locally: $\theta \sim q(\theta|\theta_t) = \mathcal{N}(\theta_t, \sigma^2 I)$
- For each sampled $\theta$, simulate data: $x \sim p(x|\theta)$

This gives a simple joint distribution:

$$
p(x, \theta | \theta_t) = p(x|\theta) \cdot q(\theta|\theta_t)
$$

Then define the **local score-matching objective**:

$$
J(W; \theta_t) = \mathbb{E}_{x,\theta} \left[ |S_W(x) - \nabla_\theta \log p(x|\theta)|^2 \right] \tag{1}
$$

Here $S_W(x)$ is a surrogate model for the Fisher score.

But $\nabla_\theta \log p(x|\theta)$ is unknown! So how do we optimize (1)?

---

## Trick #1: Integration by Parts Removes the Likelihood Term

*(Section 3.1, Theorem 3.1)*

Through an integration-by-parts identity very similar to original score matching, the intractable term disappears:

$$
J(W; \theta_t) = \mathbb{E} \left[ |S_W(x)|^2 + 2 S_W(x)^\top \nabla_\theta \log q(\theta|\theta_t) \right] \tag{2}
$$

**This is remarkable:**

- The likelihood gradient **vanishes entirely**
- Only the **proposal distribution's gradient** remains:

$$
\nabla_\theta \log q(\theta|\theta_t) = -\frac{1}{\sigma^2}(\theta - \theta_t)
$$

Thus the entire objective is computable by simulation.

---

## What Is the Optimal Solution of FSM?

*(Theorem 3.2)*

$$
S^*(x; \theta_t) = \mathbb{E}_{\theta \sim p(\theta|x,\theta_t)} \left[ \nabla_\theta \log p(x|\theta) \right]
$$

This is the **Bayes estimator** of the score under the local posterior:

$$
p(\theta|x,\theta_t) \propto p(x|\theta) \cdot q(\theta|\theta_t)
$$

**Intuition:**

- You can't estimate the true score at a single point $\theta_t$ because you never see data exactly at that point
- So FSM estimates a **locally smoothed version** of the Fisher score

This becomes important in Section 5.

---

## Trick #2: FSM = Gradient of a Gaussian-Smoothed Likelihood

*(Section 5.1, Theorem 5.1)*

FSM is *exactly* the gradient of the smoothed likelihood:

$$
\tilde{\ell}(\theta_t; x) = \log \int p(x|\theta) \cdot q(\theta|\theta_t) \, d\theta
$$

And:

$$
S^*(x; \theta_t) = \nabla_{\theta_t} \tilde{\ell}(\theta_t; x)
$$

Hence the algorithm is performing:

> **Gradient descent on a locally smoothed likelihood**, not on the raw likelihood.

This explains:

- Robustness to non-smooth likelihoods
- Ability to escape flat regions
- Improved stability vs. finite-difference estimators

---

## Practical Parameterization: Linear Score Model

The authors choose:

$$
S_W(x) = W^\top x
$$

leading to a **closed-form linear regression solution**:

$$
\hat{W} = -\left(\sum_j G_j\right)^{-1} \sum_j \sum_k x_{j,k} \cdot \nabla_\theta \log q(\theta_j|\theta_t)^\top
$$

This converts FSM into an extremely efficient method.

---

## Full FSM-MLE Algorithm

At iteration $t$:

1. **Sample parameters:** $\theta_j \sim \mathcal{N}(\theta_t, \sigma^2 I)$

2. **Simulate data:** $x_{j,k} \sim p(x|\theta_j)$

3. **Fit linear model** $S_{\hat{W}}(x)$ by solving the FSM least-squares problem

4. **Estimate gradient** of smoothed log-likelihood:

$$
\widehat{\nabla_\theta \ell(\theta_t)} = \sum_{i=1}^N S_{\hat{W}}(x_i)
$$

5. **Update parameters** using SGD or Adam

---

## Why This Works (Intuition)

FSM pulls off something surprising:

- It **never** evaluates $p(x|\theta)$
- It **never** computes $\nabla_\theta \log p(x|\theta)$
- It **never** estimates likelihoods like KDE-based methods do

Yet it performs a **gradient-based maximum likelihood optimization**.

The key is the **joint sampling over $(x, \theta)$** and score-matching identity that replaces the intractable terms with derivatives of a simple Gaussian proposal.

---

## Understanding the Bias / Smoothing Effect (Section 5.2)

If $\sigma$ (local proposal width) is too small:

- Variance explodes (like finite differences)
- Estimator becomes unstable

If $\sigma$ is too large:

- Bias grows (you oversmooth the likelihood)

Theorem 5.2 shows:

$$
\text{Bias} \le L \sqrt{d} \cdot \sigma \cdot \mathbb{E}[R(x)]
$$

This formalizes the **bias–variance tradeoff**.

---

## Summary of Core Contributions

1. **Novel local Fisher score matching objective**
2. **Likelihood-free derivation using integration-by-parts**
3. **Closed-form linear surrogate model**
4. **Equivalence to Gaussian smoothing gradient estimator**
5. **Strong theoretical properties**
   - Bias bounds
   - Convergence via averaged SGD
   - Asymptotic normality of estimator
6. **Superior empirical performance**
   - Over KDE + SPSA
   - Over Neural Likelihood Estimators
   - In high-dimensional SBI tasks

---

## Connection to EBMs

Fisher score matching connects directly to the EBM training problem:

- **EBM challenge:** The MLE gradient requires $\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x')]$ (see [EBM MLE Gradient Derivation](../EBM/EBM-mle-gradient-derivation.md))
- **FSM solution:** Bypass the likelihood entirely by estimating the Fisher score directly from simulations

Both approaches share the core insight: **use integration-by-parts to eliminate intractable terms**.

---

## References

- Khoo et al. (2025). *Direct Fisher Score Estimation for Likelihood Maximization*
- Hyvärinen (2005). *Estimation of Non-Normalized Statistical Models by Score Matching*
