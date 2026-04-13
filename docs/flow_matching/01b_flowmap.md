# Supplementary: The Flow Map and the Pushforward

*This note supplements [Flow Matching Foundations](01_flow_matching_foundations.md). It unpacks the flow map $\phi_t$, the pushforward of a distribution, and how these two objects relate to the velocity field and the continuity equation.*

---

## The Flow Map

Given a velocity field $v(x, t)$, each starting point $x_0 \in \mathbb{R}^d$ traces a unique trajectory determined by the ODE:

$$
\dot{x}(t) = v(x(t),\, t), \qquad x(0) = x_0
$$

The **flow map** $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$ collects all of these trajectories into a single function:

$$
\phi_t(x_0) \;=\; x_0 + \int_0^t v\!\left(\phi_s(x_0),\, s\right) ds
$$

In words: $\phi_t(x_0)$ is the position at time $t$ of a particle that started at $x_0$ at time $0$.

Rather than tracking one trajectory, $\phi_t$ simultaneously deforms the *entire* space $\mathbb{R}^d$ — it is a smooth map that warps every point forward in time.

---

## Structural Properties

When $v(x, t)$ is sufficiently smooth (Lipschitz in $x$), the Picard–Lindelöf theorem guarantees uniqueness of solutions, which implies:

| Property | Statement | Intuition |
|----------|-----------|-----------|
| Identity at $t=0$ | $\phi_0 = \mathrm{id}$ | No time has passed; nothing has moved |
| Composition | $\phi_{0 \to t} = \phi_{s \to t} \circ \phi_{0 \to s}$ | Travel from $0$ to $t$ by going $0 \to s$, then $s \to t$ |
| Invertibility | $\phi_t^{-1}$ exists | Run the ODE backward: $\dot{x} = -v(x, t)$ |
| Diffeomorphism | $\phi_t$ is smooth and bijective | Two distinct starting points can never collide |

The **no-crossing** property deserves attention. Suppose two particles are at the same position $x$ at time $t$. From that moment on, both obey $\dot{x} = v(x, t)$ with the same initial condition — so they must follow the *same* trajectory forever. Trajectories can only share a point if they are identical. This is why crossing trajectories in learned flows are costly: crossing indicates the marginal velocity field $v_t(x)$ is averaging over conflicting directions at $x$, which forces an ODE solver to take small, careful steps to resolve the ambiguity.

---

## The Pushforward

The flow map acts on *points*. The **pushforward** extends this action to *probability distributions*.

If $X_0 \sim p_0$, define $X_t = \phi_t(X_0)$. The distribution of $X_t$ is the pushforward of $p_0$ under $\phi_t$, written ${\phi_t}_{\,\#}\, p_0$:

$$
p_t \;=\; {\phi_t}_{\,\#}\, p_0
\qquad \Longleftrightarrow \qquad
X_0 \sim p_0 \;\Rightarrow\; \phi_t(X_0) \sim p_t
$$

The $\#$ subscript reads "push forward through." The definition says: to sample from $p_t$, draw from $p_0$ and apply the flow map.

### Concrete Formula via Change of Variables

When $\phi_t$ is differentiable with an invertible Jacobian, the pushed-forward density can be written explicitly:

$$
p_t(y) \;=\; p_0\!\left(\phi_t^{-1}(y)\right) \cdot \left|\det \nabla_y \,\phi_t^{-1}(y)\right|
$$

The two factors have clear roles:

- $p_0\!\left(\phi_t^{-1}(y)\right)$ — look up the probability of the *source* point that maps to $y$
- $\left|\det \nabla_y \,\phi_t^{-1}(y)\right|$ — correct for local volume change: if the flow compresses a region, density increases; if it stretches, density decreases

This is simply the standard change-of-variables formula from probability, applied to the flow map.

---

## The Continuity Equation as the Infinitesimal Pushforward

The change-of-variables formula above is a *finite-time* statement: it tells you the density at time $t$ given the density at time $0$. The **continuity equation** is the infinitesimal version — it tells you how density changes over an infinitesimal step $dt$:

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t\, v) = 0
$$

To see the connection, differentiate the pushforward relation $p_t = {\phi_t}_{\,\#}\, p_0$ with respect to $t$. The result is the continuity equation. They encode exactly the same physics: probability is conserved as particles flow.

---

## The Relationship Between All Three Objects

The velocity field, the flow map, and the continuity equation are three views of the same underlying structure:

```
Velocity field  v(x, t)
       │
       │  Integrate the ODE:
       │  ẋ = v(x, t),  x(0) = x₀
       ▼
Flow map  φ_t : ℝᵈ → ℝᵈ           ← acts on points
       │
       │  Apply to every sample from p₀
       │  (change-of-variables formula)
       ▼
Pushforward  (φ_t)# p₀ = p_t       ← acts on distributions
       │
       │  Differentiate with respect to t
       ▼
Continuity equation  ∂p_t/∂t + ∇·(p_t v) = 0
```

Each arrow is a precise mathematical operation:

- **ODE integration** is the most computational step: given $v$, numerically solve to get $\phi_t$. An ODE solver does this at sampling time.
- **Pushforward** is a distributional operation: no integration needed if you have $\phi_t$; just transform samples.
- **Differentiation** connects the finite-time picture ($\phi_t$) to the instantaneous picture ($\partial_t p_t$).

Flow matching operates at the top: learn $v_\theta$. Everything else — the flow map, the transported distributions, the conserved probability — follows automatically by construction. The model never explicitly computes $\phi_t$ during training; the ODE solver reconstructs it at inference.

---

## Why This Matters for Flow Matching

The flow map perspective clarifies several design choices in flow matching:

**Why straight paths are attractive.** If the velocity field is constant, $v(x, t) = c$, then $\phi_t(x_0) = x_0 + tc$ — a straight line. Straight flow maps have identity-like Jacobians and are trivial for ODE solvers. Rectified flow's goal of "straightening paths" is precisely the goal of making $\phi_t$ as close to a translation as possible.

**Why reflow reduces sampling steps.** Each reflow iteration makes trajectories more parallel and less curved. A less curved $\phi_t$ requires fewer ODE solver steps to integrate accurately, because the velocity field changes less along each trajectory.

**Why trajectory crossings are expensive.** When two trajectories cross, the marginal velocity field $v_t(x)$ at the crossing point is a compromise between two different directions. The flow map is still well-defined (trajectories only *appear* to cross in the marginal $x$-space, but come from different $x_0$ values), but the network must learn a velocity that serves both — resulting in a curved effective trajectory that needs more solver steps.

---

## Related Documents

- [Flow Matching Foundations](01_flow_matching_foundations.md) — The continuity equation and its derivation
- [Rectifying Flow Tutorial](rectifying_flow.md) — Straightening trajectories via reflow
- [Flow Matching Sampling](03_flow_matching_sampling.md) — ODE solvers and step-count tradeoffs
