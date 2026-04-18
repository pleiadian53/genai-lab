# Supplementary: Bayes' Theorem with Three Variables

*This note supplements [Flow Matching Foundations](01_flow_matching_foundations.md), specifically the Step 4 identity used in the derivation of the conditional expectation property:*

$$
\frac{p_t(x \mid x_0, x_1)\; p(x_0)\, p(x_1)}{p_t(x)} = p(x_0, x_1 \mid x_t = x)
$$

---

## Bayes' Theorem in Two Variables (Reminder)

In the familiar two-variable form, Bayes' theorem relates the conditional probability of a "cause" $A$ given "evidence" $B$:

$$
p(A \mid B) = \frac{p(B \mid A)\; p(A)}{p(B)}
$$

The four terms have standard names:

| Term | Name | Meaning |
|------|------|---------|
| $p(A \mid B)$ | Posterior | Probability of the cause after seeing the evidence |
| $p(B \mid A)$ | Likelihood | Probability of seeing this evidence if the cause is $A$ |
| $p(A)$ | Prior | Probability of the cause before seeing anything |
| $p(B)$ | Marginal (evidence) | Total probability of the observed evidence |

---

## The Three-Variable Setup in Flow Matching

We have three random variables:

- $X_0 \sim p(x_0)$ — a data sample, drawn from the data distribution
- $X_1 \sim p(x_1)$ — a noise sample, drawn independently from $\mathcal{N}(0, I)$
- $X_t = \psi_t(X_0, X_1)$ — the interpolated point, a deterministic function of the pair

The question asked by Step 4 is: *given that the interpolated point landed at $x_t = x$, what is the probability that the generating pair was $(x_0, x_1)$?*

That is, we want the **posterior** $p(x_0, x_1 \mid x_t = x)$.

---

## The Key Move: Treat the Pair as One Variable

Three variables look messy, but there is no new machinery needed. Simply **treat the pair $(x_0, x_1)$ as a single compound variable** — call it the "cause" — and treat $x_t$ as the "evidence." Bayes' theorem applies directly:

$$
\underbrace{p(x_0, x_1 \mid x_t = x)}_{\text{posterior over pairs}}
= \frac{
    \overbrace{p_t(x \mid x_0, x_1)}^{\text{likelihood}} \;
    \overbrace{p(x_0, x_1)}^{\text{prior over pairs}}
}{
    \underbrace{p_t(x)}_{\text{marginal}}
}
$$

This is just $p(A \mid B) = p(B \mid A)\,p(A)\,/\,p(B)$ with $A = (x_0, x_1)$ and $B = x_t$.

---

## Unpacking Each Term

**Prior over pairs: $p(x_0, x_1)$**

$X_0$ and $X_1$ are drawn *independently* (a data sample paired with a noise sample, no coupling between them). For independent variables, the joint factorises:

$$
p(x_0, x_1) = p(x_0)\; p(x_1)
$$

This is why the numerator in the document's expression has $p(x_0)\,p(x_1)$ rather than $p(x_0, x_1)$ — they are the same thing under the independence assumption.

**Likelihood: $p_t(x \mid x_0, x_1)$**

Given the pair $(x_0, x_1)$, the interpolated point $x_t = \psi_t(x_0, x_1)$ is *deterministic* — there is no randomness left. So this is a delta mass:

$$
p_t(x \mid x_0, x_1) = \delta\!\bigl(x - \psi_t(x_0, x_1)\bigr)
$$

It is zero unless $x$ is exactly the interpolated point for this pair, and a unit spike there. The likelihood asks: *"if this pair generated the path, how probable is it that the path passed through $x$ at time $t$?"* — and the answer is either 0 or 1 (exactly).

**Marginal: $p_t(x)$**

The marginal distribution of $X_t$ averages the delta mass over all possible pairs:

$$
p_t(x) = \int p_t(x \mid x_0, x_1)\; p(x_0)\; p(x_1)\; dx_0\, dx_1
$$

This is the probability that *any* data-noise pair produces a trajectory passing through $x$ at time $t$, regardless of which specific pair it was.

**Posterior: $p(x_0, x_1 \mid x_t = x)$**

The posterior weights each pair by how "responsible" it is for the observed $x_t = x$. Because $p_t(x \mid x_0, x_1)$ is a delta function, the posterior is supported only on pairs whose interpolated point at time $t$ exactly equals $x$:

$$
p(x_0, x_1 \mid x_t = x) \propto \delta\!\bigl(x - \psi_t(x_0, x_1)\bigr)\; p(x_0)\; p(x_1)
$$

In words: the pairs that could have generated $x_t = x$ are exactly those lying on the *preimage* of $x$ under the interpolant map — all $(x_0, x_1)$ such that $(1-t)x_0 + t x_1 = x$ (for linear interpolation).

---

## Putting It Together

Substituting into Bayes' theorem:

$$
p(x_0, x_1 \mid x_t = x)
= \frac{p_t(x \mid x_0, x_1)\; p(x_0)\; p(x_1)}{p_t(x)}
$$

and the marginal velocity identity follows immediately:

$$
v_t(x)
= \int u_t(x_0, x_1)\; p(x_0, x_1 \mid x_t = x)\; dx_0\, dx_1
= \mathbb{E}_{x_0, x_1 \mid x_t = x}\bigl[u_t(x_0, x_1)\bigr]
$$

The posterior $p(x_0, x_1 \mid x_t = x)$ gives each pair exactly the weight it deserves: pairs whose trajectory passes through $x$ at time $t$ receive positive weight; all other pairs receive zero weight (via the delta function). The marginal velocity is then the posterior-weighted average of the conditional velocities.

---

## Why Independence Matters

The factorisation $p(x_0, x_1) = p(x_0)\,p(x_1)$ is load-bearing. If $x_0$ and $x_1$ were coupled — for instance, if we always paired a data point with its nearest noise sample — the joint prior would not factorise and the Bayes step would still hold formally, but the posterior and the marginal velocity would change.

This is precisely what **minibatch optimal transport** exploits: choose couplings $p(x_0, x_1)$ that are *not* the independent product, but instead pair data and noise more strategically (e.g., match nearby points). The theoretical machinery is the same; only the prior changes.

---

## Summary

| Step | What happens |
|------|-------------|
| Identify variables | Cause = $(x_0, x_1)$; evidence = $x_t$ |
| Apply Bayes | $p(\text{cause} \mid \text{evidence}) = p(\text{evidence} \mid \text{cause})\, p(\text{cause})\,/\,p(\text{evidence})$ |
| Use independence | $p(x_0, x_1) = p(x_0)\,p(x_1)$ (data and noise drawn independently) |
| Interpret likelihood | $p_t(x \mid x_0, x_1)$ is a delta function — deterministic interpolation |
| Interpret marginal | $p_t(x)$ averages the delta function over all pairs |
| Interpret posterior | Selects only pairs whose path passes through $x$ at time $t$ |

---

## Related Documents

- [Flow Matching Foundations](01_flow_matching_foundations.md) — full derivation context (§ Flow Matching Objective)
- [Flow Map and Pushforward](01b_flowmap.md) — how distributions evolve under deterministic maps
