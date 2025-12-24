# Fisher Score Matching: The Parameter-Space Integration-by-Parts Trick

This document proves the **FSM analogue** of the score-matching integration-by-parts trick (Theorem 3.1 from Khoo et al., 2025). It's the same magic as classic score matching, but with a crucial twist:

| Method | Gradient w.r.t. | Eliminates |
|--------|-----------------|------------|
| **Classic score matching** | Data $x$ | Unknown $p_D(x)$ |
| **Fisher score matching** | Parameters $\theta$ | Intractable likelihood score |

This derivation shows exactly how the intractable likelihood score disappears. 

---

## Step 0: Notation

### Spaces and variables

- $x \in \mathbb{R}^k$ — data (observation or summary statistic)
- $\theta \in \mathbb{R}^d$ — model parameter
- $\theta_t$ — current parameter iterate (a fixed point around which we localize)

### Model and proposal

- $p(x|\theta)$ — simulator-defined likelihood (intractable density, but we can sample $x \sim p(\cdot|\theta)$)
- $q(\theta|\theta_t)$ — a **local proposal** density around $\theta_t$ (often Gaussian)

### Joint distribution (the key construction)

Define a joint density over $(x, \theta)$ by:

$$
p(x, \theta | \theta_t) := p(x|\theta) \cdot q(\theta|\theta_t)
$$

**Interpretation:** sample $\theta \sim q(\cdot|\theta_t)$, then sample $x \sim p(\cdot|\theta)$.

### Surrogate score model

- $S_W(x) \in \mathbb{R}^d$ — a function of **data only**, meant to approximate the Fisher score
- $W$ — its parameters (can be linear weights, NN weights, etc.)

### Fisher score (the unknown target)

$$
\nabla_\theta \log p(x|\theta)
$$

This is what we *cannot compute* in SBI.

### Boundary condition

We need a standard "boundary term vanishes" condition:

$$
\forall x, \quad \lim_{|\theta| \to \infty} p(x|\theta) \cdot q(\theta|\theta_t) = 0
$$

This is the regularity assumption stated in Appendix A.1 of *"Direct Fisher Score Estimation for Likelihood Maximization"* (Khoo et al., 2025). 

---

## Step 1: The "Intractable" Fisher Score Matching Objective

Start from Eq. (1):

$$
J(W; \theta_t) = \mathbb{E}_{\theta \sim q(\cdot|\theta_t), x \sim p(\cdot|\theta)} \left[ |\nabla_\theta \log p(x|\theta) - S_W(x)|^2 \right] \tag{1}
$$

**Explanation:** We want $S_W(x)$ to predict the Fisher score, but we'll never evaluate the Fisher score directly. So we'll rewrite this objective into something computable.

---

## Step 2: Expand the Square

Expand $|a-b|^2 = |a|^2 + |b|^2 - 2a^\top b$:

$$
J(W; \theta_t) = \mathbb{E}\left[ |\nabla_\theta \log p(x|\theta)|^2 + |S_W(x)|^2 - 2 S_W(x)^\top \nabla_\theta \log p(x|\theta) \right]
$$

**Explanation:** Standard algebra. When optimizing over $W$, the first term does **not** involve $W$, so it's a constant.

Up to an additive constant w.r.t. $W$:

$$
J(W; \theta_t) = \mathbb{E}\left[ |S_W(x)|^2 - 2 S_W(x)^\top \nabla_\theta \log p(x|\theta) \right] + \text{const}
$$

**Explanation:** We keep only terms that influence the optimal $S_W$.

---

## Step 3: Rewrite as an Integral

Using the joint density $p(x, \theta | \theta_t) = p(x|\theta) q(\theta|\theta_t)$:

$$
\mathbb{E}[f(x, \theta)] = \int_{\mathbb{R}^d} \int_{\mathbb{R}^k} f(x, \theta) \cdot p(x|\theta) \cdot q(\theta|\theta_t) \, dx \, d\theta
$$

where the inner integral is over $x \in \mathbb{R}^k$ (data space) and the outer integral is over $\theta \in \mathbb{R}^d$ (parameter space).

So the problematic cross term becomes:

$$
\mathbb{E}\left[ S_W(x)^\top \nabla_\theta \log p(x|\theta) \right] = \iint S_W(x)^\top \nabla_\theta \log p(x|\theta) \cdot p(x|\theta) \cdot q(\theta|\theta_t) \, dx \, d\theta
$$

**Explanation:** We're turning expectation into integrals so we can integrate by parts.

---

## Step 4: Eliminate the Log

Use the identity:

$$
\nabla_\theta \log p(x|\theta) \cdot p(x|\theta) = \nabla_\theta p(x|\theta)
$$

So:

$$
\iint S_W(x)^\top \nabla_\theta \log p(x|\theta) \cdot p(x|\theta) \cdot q(\theta|\theta_t) \, dx \, d\theta = \iint S_W(x)^\top \nabla_\theta p(x|\theta) \cdot q(\theta|\theta_t) \, dx \, d\theta
$$

**Explanation:** This is the same maneuver as classic score matching where $p \nabla \log p = \nabla p$. We're pushing "log" out of the way.

---

## Step 5: Integration by Parts (The Core Trick)

Write component-wise. Let $S_W(x) = (S_{W,1}(x), \dots, S_{W,d}(x))^\top$. Then:

$$
S_W(x)^\top \nabla_\theta p(x|\theta) = \sum_{i=1}^d S_{W,i}(x) \cdot \frac{\partial}{\partial \theta_i} p(x|\theta)
$$

So the cross term is:

$$
\sum_{i=1}^d \iint S_{W,i}(x) \cdot \frac{\partial}{\partial \theta_i} p(x|\theta) \cdot q(\theta|\theta_t) \, dx \, d\theta
$$

Now for each $i$, apply integration by parts in $\theta_i$:

$$
\int_{\mathbb{R}^d} q(\theta|\theta_t) \cdot \frac{\partial}{\partial \theta_i} p(x|\theta) \, d\theta = \Big[q(\theta|\theta_t) p(x|\theta)\Big]_{\text{boundary}} - \int_{\mathbb{R}^d} p(x|\theta) \cdot \frac{\partial}{\partial \theta_i} q(\theta|\theta_t) \, d\theta
$$

**Explanation:** This is $\int u \, dv = uv - \int v \, du$ with:

- $u = q(\theta|\theta_t)$
- $dv = \partial_{\theta_i} p(x|\theta) \, d\theta_i$
- so $v = p(x|\theta)$, $du = \partial_{\theta_i} q(\theta|\theta_t) \, d\theta_i$

Under the boundary condition, the boundary term is zero:

$$
\Big[q(\theta|\theta_t) p(x|\theta)\Big]_{\text{boundary}} = 0
$$

So:

$$
\int q(\theta|\theta_t) \cdot \frac{\partial}{\partial \theta_i} p(x|\theta) \, d\theta = -\int p(x|\theta) \cdot \frac{\partial}{\partial \theta_i} q(\theta|\theta_t) \, d\theta
$$

**Explanation:** The derivative moved from the intractable $p(x|\theta)$ to the proposal $q$, which we chose and can differentiate.

Substitute back into the cross term:

$$
\iint S_W(x)^\top \nabla_\theta p(x|\theta) \cdot q(\theta|\theta_t) \, dx \, d\theta = -\iint S_W(x)^\top p(x|\theta) \cdot \nabla_\theta q(\theta|\theta_t) \, dx \, d\theta
$$

---

## Step 6: Convert to Log Form

Use:

$$
\nabla_\theta q(\theta|\theta_t) = q(\theta|\theta_t) \cdot \nabla_\theta \log q(\theta|\theta_t)
$$

So:

$$
-\iint S_W(x)^\top p(x|\theta) \cdot \nabla_\theta q(\theta|\theta_t) \, dx \, d\theta = -\iint S_W(x)^\top p(x|\theta) \cdot q(\theta|\theta_t) \cdot \nabla_\theta \log q(\theta|\theta_t) \, dx \, d\theta
$$

This is:

$$
= -\mathbb{E}_{x \sim p(\cdot|\theta), \theta \sim q(\cdot|\theta_t)} \left[ S_W(x)^\top \nabla_\theta \log q(\theta|\theta_t) \right]
$$

**Explanation:** Now the cross term involves only $\nabla_\theta \log q$, which is analytic.

**The key identity:**

$$
\mathbb{E}\left[ S_W(x)^\top \nabla_\theta \log p(x|\theta) \right] = -\mathbb{E}\left[ S_W(x)^\top \nabla_\theta \log q(\theta|\theta_t) \right]
$$

---

## Step 7: Substitute Back into the Objective

Recall (up to constants):

$$
J(W; \theta_t) = \mathbb{E}\left[ |S_W(x)|^2 - 2 S_W(x)^\top \nabla_\theta \log p(x|\theta) \right] + \text{const}
$$

Replace the cross term using our identity:

$$
\mathbb{E}\left[ S_W^\top \nabla_\theta \log p \right] = -\mathbb{E}\left[ S_W^\top \nabla_\theta \log q \right]
$$

Therefore:

$$
-2 \mathbb{E}\left[ S_W^\top \nabla_\theta \log p \right] = -2 \left( -\mathbb{E}[S_W^\top \nabla_\theta \log q] \right) = +2 \mathbb{E}\left[ S_W^\top \nabla_\theta \log q \right]
$$

So the rewritten FSM objective is:

$$
\boxed{J(W; \theta_t) = \mathbb{E}_{x \sim p(\cdot|\theta), \theta \sim q(\cdot|\theta_t)} \left[ |S_W(x)|^2 + 2 S_W(x)^\top \nabla_\theta \log q(\theta|\theta_t) \right] + \text{const}}
$$

This is exactly **Theorem 3.1 / Eq. (2)**.

**Explanation:** The entire dependence on the intractable likelihood score is gone. We only need:

- Simulated pairs $(\theta, x)$
- $\nabla_\theta \log q(\theta|\theta_t)$, which is easy to compute

---

## The Same Trick as Classic Score Matching

Both methods follow the same schema:

1. Start with a squared error loss against an unknown score
2. Expand the square
3. The cross term has "unknown score" inside
4. Use $p \nabla \log p = \nabla p$
5. Integration by parts moves the derivative onto something we *control*
   - **Classic SM:** onto the model score's divergence
   - **FSM:** onto the proposal score $\nabla_\theta \log q$

It's score matching… but in **parameter space** instead of **data space**.

---

## Concrete Example: Gaussian Proposal

If:

$$
q(\theta|\theta_t) = \mathcal{N}(\theta_t, \sigma^2 I)
$$

then:

$$
\nabla_\theta \log q(\theta|\theta_t) = -\frac{1}{\sigma^2}(\theta - \theta_t)
$$

**Explanation:** This is why the rewritten loss is practical: you can compute this exactly and it behaves like a "pull back toward $\theta_t$" vector.

---

## What's Next

The minimizer of this objective is a **smoothed Fisher score**:

$$
S^*(x; \theta_t) = \mathbb{E}_{\theta \sim p(\theta|x, \theta_t)} \left[ \nabla_\theta \log p(x|\theta) \right]
$$

This connects to **Gaussian smoothing** (Theorem 5.1), which explains the method's robustness properties.

---

## Connection to Data-Space Score Matching

See [Score Matching Objective Derivation](EBM-score-matching-objective.md) for the data-space analogue that eliminates $\nabla_x \log p_D(x)$ instead of $\nabla_\theta \log p(x|\theta)$.
