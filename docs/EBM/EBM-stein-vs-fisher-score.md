# Stein Score vs Fisher Score: Two Flavors of "Score"

This document clarifies the distinction between the **Stein score** (gradient w.r.t. data) and the **Fisher score** (gradient w.r.t. parameters)—two different objects that both go by "score" in the literature.

---

## The Two Scores at a Glance

| Name | Symbol | Gradient w.r.t. | What it measures |
|------|--------|-----------------|------------------|
| **Stein score** | $s(x) = \nabla_x \log p(x)$ | Data $x$ | Direction of steepest increase in log-density |
| **Fisher score** | $g(\theta) = \nabla_\theta \log p(x \| \theta)$ | Parameters $\theta$ | Sensitivity of log-likelihood to parameters |

Both are gradients of a log-probability, but they differentiate with respect to *different variables*.

---

## Stein Score: $\nabla_x \log p(x)$

### Stein score definition

For a density $p(x)$ over data $x \in \mathbb{R}^d$:

$$
s(x) := \nabla_x \log p(x)
$$

This is a **vector field over data space**—at each point $x$, it points in the direction where the density increases most rapidly.

### Stein score intuition

- **High-density regions:** The score points "inward" toward the mode
- **Low-density regions:** The score points toward higher-density areas
- **At the mode:** The score is zero (gradient of log-density vanishes at maximum)

### Why the Stein score is useful

The Stein score is central to:

1. **Training EBMs:** For $p_\theta(x) = \exp(-E_\theta(x))/Z_\theta$, the score is $s_\theta(x) = -\nabla_x E_\theta(x)$, which **doesn't depend on $Z_\theta$**
2. **Diffusion models:** Learn $\nabla_x \log p_t(x)$ at multiple noise levels
3. **Langevin dynamics:** Sample from $p(x)$ using $x_{t+1} = x_t + \epsilon \nabla_x \log p(x_t) + \sqrt{2\epsilon} z$

### The score matching objective

We want to learn $s_\theta(x) \approx \nabla_x \log p_D(x)$, but $p_D$ is unknown. Score matching solves this via integration by parts:

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{p_D}\left[ \frac{1}{2}|s_\theta(x)|^2 + \mathrm{tr}(\nabla_x s_\theta(x)) \right]
$$

See [Score Matching Objective Derivation](EBM-score-matching-objective.md) for the full proof. For implementation guidance, see [Roadmap Stage 5](../ROADMAP.md) and the [ESM vs DSM comparison](../score_matching/README.md#practical-considerations-esm-vs-dsm).

---

## Fisher Score: $\nabla_\theta \log p(x|\theta)$

### Fisher score definition

For a parametric model $p(x|\theta)$ with parameters $\theta \in \mathbb{R}^d$:

$$
g(x; \theta) := \nabla_\theta \log p(x|\theta)
$$

This is a **vector in parameter space**—it tells you how the log-likelihood of observation $x$ changes as you vary $\theta$.

### Fisher score intuition

- **Positive component $g_i > 0$:** Increasing $\theta_i$ would increase the likelihood of $x$
- **Negative component $g_i < 0$:** Increasing $\theta_i$ would decrease the likelihood of $x$
- **At the MLE:** $\mathbb{E}_{p_D}[g(x; \hat{\theta})] = 0$ (score equations)

### Why the Fisher score is useful

The Fisher score is central to:

1. **Maximum likelihood estimation:** The MLE gradient is $\nabla_\theta \ell(\theta) = \sum_i \nabla_\theta \log p(x_i|\theta)$
2. **Fisher information:** $I(\theta) = \mathbb{E}[g(x;\theta) g(x;\theta)^\top]$ measures parameter uncertainty
3. **Simulation-based inference:** When $p(x|\theta)$ is intractable but simulable

### The Fisher score matching objective

We want to learn $S_W(x) \approx \nabla_\theta \log p(x|\theta)$, but the likelihood is intractable. Fisher score matching solves this via integration by parts in parameter space:

$$
J(W; \theta_t) = \mathbb{E}\left[ |S_W(x)|^2 + 2 S_W(x)^\top \nabla_\theta \log q(\theta|\theta_t) \right]
$$

See [Fisher Score Matching Derivation](EBM-score-matching-FSM-analogue.md) for the full proof.

---

## Side-by-Side Comparison

| Aspect | Stein Score | Fisher Score |
|--------|-------------|--------------|
| **Symbol** | $\nabla_x \log p(x)$ | $\nabla_\theta \log p(x\|\theta)$ |
| **Lives in** | Data space $\mathbb{R}^d$ | Parameter space $\mathbb{R}^p$ |
| **Input** | Data point $x$ | Data $x$ and parameters $\theta$ |
| **Output** | Vector in $\mathbb{R}^d$ | Vector in $\mathbb{R}^p$ |
| **Measures** | Where density increases in data space | How likelihood changes with parameters |
| **Zero at** | Mode of $p(x)$ | MLE (in expectation) |

---

## When to Use Which

### Use Stein score when

- **Training generative models** (EBMs, diffusion models)
- **Sampling** via Langevin dynamics or score-based MCMC
- **Density estimation** where you want to model $p(x)$ directly
- The **partition function $Z_\theta$** is intractable

### Use Fisher score when

- **Parameter estimation** via MLE
- **Simulation-based inference** where $p(x|\theta)$ is implicit
- **Sensitivity analysis** of model parameters
- The **likelihood $p(x|\theta)$** is intractable but simulable

---

## The Same Trick, Different Spaces

Both score matching methods use the **same mathematical trick**:

1. Start with a squared error loss against an unknown score
2. The cross term contains the unknown score
3. Use $p \nabla \log p = \nabla p$ to eliminate the log
4. Integration by parts moves the derivative onto something computable

| Method | IBP in | Eliminates | Replaces with |
|--------|--------|------------|---------------|
| **Stein score matching** | Data space | Unknown data score | Trace of Jacobian |
| **Fisher score matching** | Parameter space | Intractable likelihood score | Proposal score |

---

## Historical Note

The terminology can be confusing because:

- **"Score function"** in classical statistics usually means the Fisher score
- **"Score"** in the diffusion/EBM literature usually means the Stein score
- Both communities use "score matching" but for different objects

This document uses explicit names (Stein vs Fisher) to avoid ambiguity.

---

## References

- Hyvärinen (2005). *Estimation of Non-Normalized Statistical Models by Score Matching* — Original Stein score matching
- Khoo et al. (2025). *Direct Fisher Score Estimation for Likelihood Maximization* — Fisher score matching for SBI
- Song & Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution* — Score-based generative models
