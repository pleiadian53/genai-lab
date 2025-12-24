# Score Matching Objective: The Integration-by-Parts Derivation

This document proves the classic identity behind **(explicit) score matching**—the "integration-by-parts trick" that makes score matching usable without knowing $p_D$. This derivation shows exactly where the unknown data distribution terms drop out.

**The key identity:**

$$
\mathbb{E}_{p_D}\left[\frac{1}{2}|s_\theta(x) - \nabla_x \log p_D(x)|^2\right] = \mathbb{E}_{p_D}\left[\frac{1}{2}|s_\theta(x)|^2 + \mathrm{tr}(\nabla_x s_\theta(x))\right] + \text{const}
$$

The left side requires the unknown $\nabla_x \log p_D(x)$; the right side doesn't. 

---

## Step 0: Notation and Assumptions

### Variables and distributions

- $x = (x_1, \dots, x_d) \in \mathbb{R}^d$ — data vector
- $p_D(x)$ — true (unknown) data density; we can sample from it
- $p_\theta(x)$ — model density (e.g., an EBM); $\theta$ are model parameters

### Score functions

- **Model score:**

$$
s_\theta(x) := \nabla_x \log p_\theta(x) \in \mathbb{R}^d
$$

Component form: $s_{\theta,i}(x) = \frac{\partial}{\partial x_i} \log p_\theta(x)$

- **Data score:**

$$
s_D(x) := \nabla_x \log p_D(x) \in \mathbb{R}^d
$$

which we cannot compute directly because $p_D$ is unknown.

### Differential operators

- **Gradient** w.r.t. $x$: $\nabla_x$
- **Jacobian** of a vector field $s_\theta(x)$:

$$
J_x s_\theta(x) \in \mathbb{R}^{d \times d}, \quad (J_x s_\theta)_{ij} = \frac{\partial s_{\theta,i}}{\partial x_j}
$$

- **Divergence** (a scalar):

$$
\nabla_x \cdot s_\theta(x) := \sum_{i=1}^d \frac{\partial s_{\theta,i}(x)}{\partial x_i} = \mathrm{tr}(J_x s_\theta(x))
$$

### Boundary condition

We assume the boundary term vanishes. For $\mathcal{X} = \mathbb{R}^d$:

$$
\lim_{|x| \to \infty} p_D(x) \cdot s_{\theta,i}(x) = 0 \quad \text{for each } i
$$

This is the standard regularity condition used in score matching proofs.

**Why we need it:** Integration by parts produces a boundary term; we want it to be zero.

---

## Step 1: Start from the Explicit Score Matching Objective

Define:

$$
\mathcal{L}_{\text{ESM}}(\theta) := \mathbb{E}_{x \sim p_D} \left[ \frac{1}{2} |s_\theta(x) - s_D(x)|^2 \right]
$$

**Explanation:** We want the model score $s_\theta$ to match the true score $s_D$, because matching scores identifies the density (up to a constant).

---

## Step 2: Expand the Squared Norm

$$
\frac{1}{2}|s_\theta - s_D|^2 = \frac{1}{2}|s_\theta|^2 - \langle s_\theta, s_D \rangle + \frac{1}{2}|s_D|^2
$$

**Explanation:** This is just $|a-b|^2 = |a|^2 - 2a^\top b + |b|^2$, with the factor $\frac{1}{2}$ removing the 2.

Taking expectation under $p_D$:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_D}\left[\frac{1}{2}|s_\theta(x)|^2\right] - \mathbb{E}_{p_D}\left[\langle s_\theta(x), s_D(x) \rangle\right] + \mathbb{E}_{p_D}\left[\frac{1}{2}|s_D(x)|^2\right]
$$

**Explanation:** Linear property of expectation.

Now note: the last term does **not depend on $\theta$**, because $s_D$ is fixed by the data distribution.

So:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_D}\left[\frac{1}{2}|s_\theta(x)|^2\right] - \mathbb{E}_{p_D}\left[\langle s_\theta(x), s_D(x) \rangle\right] + \text{const}
$$

**Explanation:** When optimizing over $\theta$, constants can be ignored.

**The real enemy** is the cross term $\mathbb{E}_{p_D}[\langle s_\theta, s_D \rangle]$, because it contains $s_D = \nabla \log p_D$, which we can't compute.

---

## Step 3: Rewrite the Cross Term

Write expectation as an integral:

$$
\mathbb{E}_{p_D}\left[\langle s_\theta(x), s_D(x) \rangle\right] = \int p_D(x) \cdot s_\theta(x)^\top \nabla_x \log p_D(x) \, dx
$$

**Explanation:** By definition, $\mathbb{E}_{p_D}[f(x)] = \int f(x) p_D(x) \, dx$.

Now use the identity:

$$
\nabla_x \log p_D(x) = \frac{\nabla_x p_D(x)}{p_D(x)} \quad \text{(where } p_D(x) > 0 \text{)}
$$

Substitute:

$$
\int p_D(x) \cdot s_\theta(x)^\top \nabla_x \log p_D(x) \, dx = \int p_D(x) \cdot s_\theta(x)^\top \frac{\nabla_x p_D(x)}{p_D(x)} \, dx = \int s_\theta(x)^\top \nabla_x p_D(x) \, dx
$$

**Explanation:** The $p_D(x)$ cancels. Now we no longer have $\log p_D$, only $\nabla p_D$. Still not computable directly, but now integration by parts can help.

---

## Step 4: Apply Integration by Parts (Component-wise)

Write the dot product as a sum over components:

$$
\int s_\theta(x)^\top \nabla_x p_D(x) \, dx = \sum_{i=1}^d \int s_{\theta,i}(x) \cdot \frac{\partial}{\partial x_i} p_D(x) \, dx
$$

**Explanation:** $a^\top b = \sum_i a_i b_i$.

Now apply 1D integration by parts in the $x_i$ direction while holding other coordinates fixed:

$$
\int s_{\theta,i}(x) \cdot \frac{\partial}{\partial x_i} p_D(x) \, dx = \Big[s_{\theta,i}(x) \cdot p_D(x)\Big]_{\text{boundary}} - \int p_D(x) \cdot \frac{\partial}{\partial x_i} s_{\theta,i}(x) \, dx
$$

**Explanation:** This is the multivariate version of $\int u \, dv = uv - \int v \, du$, where $u = s_{\theta,i}$, $dv = \partial_i p_D \, dx_i$, so $v = p_D$, $du = \partial_i s_{\theta,i} \, dx_i$.

Under the boundary condition, the boundary term is zero:

$$
\Big[s_{\theta,i}(x) \cdot p_D(x)\Big]_{\text{boundary}} = 0
$$

So:

$$
\int s_{\theta,i}(x) \cdot \frac{\partial}{\partial x_i} p_D(x) \, dx = -\int p_D(x) \cdot \frac{\partial}{\partial x_i} s_{\theta,i}(x) \, dx
$$

**Explanation:** This is where the "magic" happens: the derivative moves from $p_D$ to $s_\theta$.

Summing over $i$:

$$
\int s_\theta(x)^\top \nabla_x p_D(x) \, dx = -\sum_{i=1}^d \int p_D(x) \cdot \frac{\partial}{\partial x_i} s_{\theta,i}(x) \, dx
$$

Recognize divergence:

$$
-\sum_{i=1}^d \int p_D(x) \cdot \frac{\partial}{\partial x_i} s_{\theta,i}(x) \, dx = -\int p_D(x) \cdot (\nabla_x \cdot s_\theta(x)) \, dx
$$

So the cross term becomes:

$$
\mathbb{E}_{p_D}\left[\langle s_\theta(x), s_D(x) \rangle\right] = -\mathbb{E}_{p_D}\left[\nabla_x \cdot s_\theta(x)\right]
$$

**Explanation:** We have removed the unknown $s_D$. Everything left involves $s_\theta$ and its derivatives.

---

## Step 5: Substitute Back into the Objective

Recall:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_D}\left[\frac{1}{2}|s_\theta(x)|^2\right] - \mathbb{E}_{p_D}\left[\langle s_\theta(x), s_D(x) \rangle\right] + \text{const}
$$

Substitute the identity we just proved:

$$
-\mathbb{E}_{p_D}\left[\langle s_\theta, s_D \rangle\right] = -\left(-\mathbb{E}_{p_D}[\nabla_x \cdot s_\theta]\right) = \mathbb{E}_{p_D}[\nabla_x \cdot s_\theta]
$$

So:

$$
\boxed{\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_D}\left[ \frac{1}{2}|s_\theta(x)|^2 + \nabla_x \cdot s_\theta(x) \right] + \text{const}}
$$

Finally, use $\nabla_x \cdot s_\theta(x) = \mathrm{tr}(J_x s_\theta(x))$:

$$
\boxed{\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{p_D}\left[ \frac{1}{2}|s_\theta(x)|^2 + \mathrm{tr}(J_x s_\theta(x)) \right] + \text{const}}
$$

**Explanation:** This is exactly the tractable form stated in the paper's score-matching section. 

---

## Intuition: What This Proof Achieves

- The explicit objective *wanted* to match $s_\theta$ to $s_D$, but $s_D$ is unknown
- The trick converts the unknown cross term into the **divergence of the model score**, which *is* computable (if you can differentiate your model w.r.t. $x$)
- The remaining constant is $\frac{1}{2}\mathbb{E}_{p_D}|s_D|^2$, which doesn't matter for optimization over $\theta$

---

## Why This Still Gets Expensive

That $\mathrm{tr}(J_x s_\theta(x))$ term requires computing the trace of a Jacobian (often involving second derivatives of the energy). In high dimensions, this is costly—hence:

- **Denoising Score Matching (DSM)** — avoids the trace term by adding noise
- **Sliced Score Matching (SSM)** — uses random projections to approximate the trace

---

## Connection to Fisher Score Matching

The same integration-by-parts trick applies in **parameter space**:

| Aspect | Data-Space Score Matching | Parameter-Space (Fisher) Score Matching |
|--------|---------------------------|----------------------------------------|
| Eliminates | $\nabla_x \log p_D(x)$ | $\nabla_\theta \log p(x\|\theta)$ |
| Replaces with | $\nabla_x \cdot s_\theta(x)$ | $\nabla_\theta \log q(\theta\|\theta_t)$ |

See [Fisher Score Matching](../score_matching/SM-fisher-score-matching.md) for the parameter-space analogue used in simulation-based inference.
