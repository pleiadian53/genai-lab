# Flow Matching

This directory contains documentation on **flow matching** methods for generative modeling — an alternative to score-based diffusion that learns velocity fields via simple regression.

## Contents

| Document | Description |
|----------|-------------|
| [rectifying_flow.md](rectifying_flow.md) | Tutorial on rectified flow from first principles |

## Key Concepts

**Flow matching** learns a velocity field $v_\theta(x, t)$ that transports samples from a noise distribution to a data distribution:

$$
\frac{dx}{dt} = v_\theta(x, t)
$$

**Rectified flow** is the simplest instantiation:

- Linear interpolation path: $x_t = (1-t) x_0 + t x_1$
- Constant velocity target: $v = x_1 - x_0$
- Simple MSE loss: $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$

## Comparison with Score Matching

| Aspect | Score Matching | Flow Matching |
|--------|---------------|---------------|
| What's learned | Score: $\nabla_x \log p_t(x)$ | Velocity: $v_\theta(x, t)$ |
| Forward process | Stochastic (add noise) | Deterministic (interpolate) |
| Reverse process | Stochastic SDE | Deterministic ODE |
| Sampling steps | 100-1000 | 10-50 |

## Related Documentation

- [Diffusion Transformers](../diffusion/DiT/diffusion_transformer.md) — Architecture for flow/diffusion models
- [Score Matching](../score_matching/README.md) — Alternative approach via score functions
- [ROADMAP](../ROADMAP.md) — Stage 7 covers flow matching

## References

- Liu et al. (2022) - "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
- Lipman et al. (2023) - "Flow Matching for Generative Modeling"
