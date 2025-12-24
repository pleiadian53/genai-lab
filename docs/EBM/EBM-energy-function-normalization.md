# Why the Energy Function Formulation Normalizes

In energy-based models (EBMs), we define probability distributions through an **energy function** $E_\theta(x)$ rather than directly specifying probabilities. This approach is powerful because it allows us to model complex distributions without worrying about normalization during model design—we simply assign lower energy to more probable configurations. But why does this work? Why does dividing by the partition function $Z_\theta$ actually produce a valid probability distribution?

This document provides a rigorous, step-by-step proof that the energy-based formulation indeed yields a properly normalized probability density. Understanding this foundation is essential before diving into the computational challenges (like intractable partition functions) that motivate techniques such as score matching and contrastive divergence.

---

## Goal

We'll prove—step by step—that defining

$$
Z_\theta := \int_{\mathcal{X}} \exp\big(-E_\theta(x)\big) \, dx
$$

and then defining

$$
p_\theta(x) := \frac{\exp\big(-E_\theta(x)\big)}{Z_\theta}
$$

indeed makes $p_\theta$ a **normalized probability density** on $\mathcal{X} \subseteq \mathbb{R}^d$. Each step includes a brief explanation of what it means.

---

## Setup and Assumptions

Let:

- $\mathcal{X} \subseteq \mathbb{R}^d$ be the data domain (often all of $\mathbb{R}^d$).
- $E_\theta: \mathcal{X} \to \mathbb{R}$ be the energy function.
- Define the unnormalized "density-like" function:

$$
\tilde{p}_\theta(x) := \exp(-E_\theta(x))
$$

**Assumption (integrability):**

$$
0 < Z_\theta := \int_{\mathcal{X}} \exp(-E_\theta(x)) \, dx < \infty
$$

This says the integral exists and is finite (otherwise normalization is impossible). In practice, modelers choose $E_\theta$ so this holds.

---

## Claim

With $p_\theta(x) = \tilde{p}_\theta(x) / Z_\theta$, we have:

1. $p_\theta(x) \ge 0$ for all $x \in \mathcal{X}$
2. $\int_{\mathcal{X}} p_\theta(x) \, dx = 1$

These are exactly the two requirements for a (Lebesgue) probability density.

---

## Proof, Step-by-Step

### 1) Non-negativity

$$
\tilde{p}_\theta(x) = \exp(-E_\theta(x)) \ge 0 \quad \text{for all } x
$$

**Explanation:** The exponential of any real number is strictly positive, hence nonnegative.

$$
Z_\theta = \int_{\mathcal{X}} \tilde{p}_\theta(x) \, dx > 0
$$

**Explanation:** An integral of a nonnegative function is nonnegative; and under the assumption $Z_\theta > 0$, it's strictly positive (i.e., $\tilde{p}_\theta$ isn't zero almost everywhere).

$$
p_\theta(x) = \frac{\tilde{p}_\theta(x)}{Z_\theta} \ge 0
$$

**Explanation:** A nonnegative numerator divided by a positive constant stays nonnegative. So $p_\theta$ can't assign negative probability "density" anywhere.

---

### 2) Normalization (the key result)

Start from the definition:

$$
\int_{\mathcal{X}} p_\theta(x) \, dx = \int_{\mathcal{X}} \frac{\exp(-E_\theta(x))}{Z_\theta} \, dx
$$

**Explanation:** We're checking whether the total probability mass under $p_\theta$ equals 1.

Pull the constant $1/Z_\theta$ outside the integral:

$$
= \frac{1}{Z_\theta} \int_{\mathcal{X}} \exp(-E_\theta(x)) \, dx
$$

**Explanation:** $Z_\theta$ depends on $\theta$, not on $x$, so with respect to the $x$-integral it's a constant. Constants factor out of integrals.

Now substitute the definition of $Z_\theta$:

$$
= \frac{1}{Z_\theta} \cdot Z_\theta
$$

**Explanation:** By definition, $\int_{\mathcal{X}} \exp(-E_\theta(x)) \, dx = Z_\theta$. So the integral is literally the partition function.

Finally simplify:

$$
= 1
$$

**Explanation:** The whole point of $Z_\theta$ is to be exactly the constant that makes this equal to 1.

This proves $p_\theta$ is normalized. $\square$

---

## What "Integrating Out $x$" Means Here

When people say "integrate out $x$" in this context, they mean:

- We start with an **unnormalized** nonnegative function $\tilde{p}_\theta(x) = \exp(-E_\theta(x))$.
- We compute its total mass over $x$:

$$
Z_\theta = \int \tilde{p}_\theta(x) \, dx
$$

- Then we divide by that mass so the new function has total mass 1:

$$
p_\theta(x) = \tilde{p}_\theta(x) / Z_\theta
$$

So "integrating out $x$" is just "compute the total mass over the data space."

---

## Two Important Practical Notes

### (A) Why we need $Z_\theta < \infty$

If $Z_\theta = \infty$, then $\tilde{p}_\theta$ has infinite mass and no finite constant can normalize it.

A typical sufficient condition (not necessary) is that for large $|x|$,

$$
E_\theta(x) \to +\infty
$$

fast enough that $\exp(-E_\theta(x))$ decays and is integrable.

### (B) Discrete vs. Continuous

If $x$ is discrete (e.g., categorical), replace integrals with sums:

$$
Z_\theta = \sum_x \exp(-E_\theta(x)), \quad p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \quad \sum_x p_\theta(x) = 1
$$

Same proof; sums instead of integrals.

---

## What's Next

If you want to connect this directly to **why MLE is hard for EBMs**, the next proof is: show that $\nabla_\theta \log Z_\theta$ becomes an expectation under $p_\theta(x)$, which usually requires MCMC. That's the "villain origin story" that score matching is responding to.
