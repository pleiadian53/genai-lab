# Score Matching: The Core Objective

This document explains the score matching objective—a technique for training energy-based models without computing the intractable partition function. Score matching is foundational to modern generative models including diffusion models.

---

## 1. What Problem Score Matching Solves

We want to **learn a probability density** over data $p_D(x)$, but we **only have samples** $x \sim p_D$.

This is the classic *density estimation* problem.

**The difficulty:** Many flexible models define densities only **up to a normalizing constant**, which makes maximum likelihood hard or impossible.

**Score matching offers a workaround:** Instead of matching the density itself, we match its **score** (the gradient of the log-density).

---

## 2. The Modeling Setup: Energy-Based Models (EBMs)

### 2.1 Data space and variables

- $x \in \mathbb{R}^d$ — A data vector (e.g., an image flattened into pixels, a feature vector, etc.)
- $p_D(x)$ — The *true but unknown* data-generating distribution

---

### 2.2 Model density via an energy function

We model the data using an **energy-based model**:

$$
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}
$$

Where:

- $E_\theta(x) : \mathbb{R}^d \to \mathbb{R}$ — A scalar-valued **energy function**, typically a neural network
- $\theta$ — Model parameters
- $Z_\theta = \int \exp(-E_\theta(x)) \, dx$ — The **partition function** (normalizing constant)

**Key issue:** $Z_\theta$ depends on $\theta$ and is usually **intractable**.

---

## 3. The Score Function: The Central Object

### 3.1 Definition

The **score function** of a density is:

$$
s_\theta(x) := \nabla_x \log p_\theta(x)
$$

This is a vector in $\mathbb{R}^d$.

### 3.2 Why the score is special

Let's expand it:

$$
\log p_\theta(x) = -E_\theta(x) - \log Z_\theta
$$

Taking gradient w.r.t. $x$:

$$
\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)
$$

**Important observation:**

- The normalizing constant $Z_\theta$ disappears
- The score depends **only on the energy gradient**

This is the loophole score matching exploits.

---

## 4. What Does It Mean to "Match Scores"?

If two densities have the same score everywhere (under mild regularity conditions), then they are the **same density up to a constant**, which is exactly what we want.

So instead of minimizing:

$$
\mathrm{KL}(p_D \| p_\theta)
$$

we try to make:

$$
\nabla_x \log p_\theta(x) \approx \nabla_x \log p_D(x)
$$

---

## 5. The Explicit Score Matching Objective

### 5.1 The ideal (but infeasible) objective

We start with the **explicit score matching (ESM)** loss:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{x \sim p_D(x)} \left[ \frac{1}{2} \left| s_\theta(x) - \nabla_x \log p_D(x) \right|^2 \right]
$$

Let's unpack every symbol.

---

### 5.2 Notation breakdown

- $\mathbb{E}_{x \sim p_D(x)}[\cdot]$ — Expectation over true data samples
- $s_\theta(x)$ — Model score $= \nabla_x \log p_\theta(x)$
- $\nabla_x \log p_D(x)$ — **True data score** (unknown!)
- $|\cdot|$ — Euclidean norm
- Factor $\frac{1}{2}$ — For mathematical convenience

---

### 5.3 Why this objective is impossible to compute

We do **not know** $p_D(x)$, so:

- We cannot compute $\log p_D(x)$
- We cannot compute $\nabla_x \log p_D(x)$

So this loss is conceptually useful but computationally useless.

---

## 6. The Key Mathematical Trick: Integration by Parts

Score matching transforms the explicit objective into one that **does not involve $p_D$**.

To do this, we introduce differential operators.

---

## 7. Differential Operators and Notation

### 7.1 Gradient operator

For a scalar function $f : \mathbb{R}^d \to \mathbb{R}$:

$$
\nabla_x f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_d} \end{pmatrix}
$$

---

### 7.2 Jacobian operator

For a vector-valued function $f(x) = (f_1(x), \dots, f_d(x))^\top$, the **Jacobian matrix** is:

$$
J_x f(x) = \left[ \frac{\partial f_i}{\partial x_j} \right]_{i,j} \in \mathbb{R}^{d \times d}
$$

---

### 7.3 Trace operator

For a square matrix $A$:

$$
\mathrm{tr}(A) = \sum_i A_{ii}
$$

---

## 8. The Tractable Score Matching Objective

Using integration by parts (details omitted in the main text but standard), the explicit objective becomes:

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{x \sim p_D(x)} \left[ \frac{1}{2}|s_\theta(x)|^2 + \mathrm{tr}(J_x s_\theta(x)) \right] + \text{const}
$$



---

## 9. Why This Works

### 9.1 What disappeared?

- $\nabla_x \log p_D(x)$ is gone
- Only $s_\theta(x)$ and its derivatives remain

### 9.2 What we can compute

Both terms in the tractable objective are computable:

- $|s_\theta(x)|^2 = |\nabla_x \log p_\theta(x)|^2$ — squared norm of the model score
- $\mathrm{tr}(J_x s_\theta(x))$ — trace of the Jacobian (sum of second derivatives)

The expectation is approximated by sampling from the data distribution.

---

## 10. Connection to EBMs and Beyond

Score matching is the foundation for:

- **Training EBMs** without computing $Z_\theta$
- **Denoising score matching** — a practical variant using noisy data
- **Diffusion models** — learn scores at multiple noise levels
- **Fisher score matching** — parameter-space analogue for simulation-based inference

---

## What's Next

See [Fisher Score Matching](SM-fisher-score-matching.md) for how these ideas extend to estimating gradients w.r.t. *parameters* (not data) in likelihood-free settings.
