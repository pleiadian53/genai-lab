# Forward SDE Design Choices: Understanding $f(x,t)$ and $g(t)$

## Core Principle

**$f(x,t)$ and $g(t)$ are design choices, not learned parameters.**

In the forward SDE:

$$
dx = f(x,t)\,dt + g(t)\,dw(t)
$$

- **$f(x,t)$**: Drift function — You choose this
- **$g(t)$**: Diffusion coefficient — You choose this
- **$s_\theta(x,t)$**: Score function — The **only thing** the neural network learns

This document explains the common choices for $f$ and $g$, why they work, and how to select them for your application.

---

## Why Do We Need to Choose $f$ and $g$?

The forward SDE defines **how clean data is corrupted into noise**. Different choices of $f$ and $g$ lead to:

1. **Different noise schedules**: How fast data becomes noise
2. **Different variance behaviors**: Whether variance grows, shrinks, or stays constant
3. **Different mathematical properties**: Closed-form marginals, ease of sampling, etc.

**Key insight**: The choice of forward SDE determines:
- What the training objective looks like
- How the reverse SDE behaves during sampling
- Whether we can compute things analytically

---

## The Three Standard SDEs

### Overview Table

| SDE Type | $f(x,t)$ | $g(t)$ | Variance Behavior | Used In |
|----------|----------|--------|-------------------|---------|
| **VP-SDE** | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)}$ | Constant (preserved) | DDPM, most models |
| **VE-SDE** | $0$ | $\sqrt{\frac{d\sigma^2(t)}{dt}}$ | Exploding | Score-based models |
| **sub-VP-SDE** | $-\frac{1}{2}\beta(t) x$ | $\sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}$ | Decreasing | Compromise |

---

## 1. VP-SDE (Variance-Preserving)

### Definition

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw(t)
$$

### Why These Specific Functions?

**Drift: $f(x,t) = -\frac{1}{2}\beta(t) x$**

- **Linear in $x$**: Pulls state toward origin (shrinks signal)
- **Time-dependent strength**: $\beta(t)$ controls how fast signal decays
- **Factor of $\frac{1}{2}$**: Carefully chosen to balance the diffusion term

**Diffusion: $g(t) = \sqrt{\beta(t)}$**

- **Adds noise** proportional to $\sqrt{\beta(t)}$
- **Balances drift**: The $\frac{1}{2}$ in drift exactly compensates diffusion growth

### Key Property: Variance Preservation

The drift and diffusion are **perfectly balanced** so that:

$$
\mathbb{E}[\|x_t\|^2] = \mathbb{E}[\|x_0\|^2] \quad \text{for all } t
$$

**Why this matters**:

- Data doesn't "blow up" or "collapse"
- Stable numerical behavior
- Clean mathematical analysis

### Closed-Form Marginal

VP-SDE has a tractable marginal distribution:

$$
p_t(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t) I\right)
$$

where:

$$
\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)\,ds\right)
$$

**Sampling formula**:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)
$$

**This is exactly the DDPM forward process!**

### Common $\beta(t)$ Schedules

**Linear schedule** (DDPM):

$$
\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \cdot \frac{t}{T}
$$

Typical values: $\beta_{\min} = 0.1$, $\beta_{\max} = 20$, $T = 1$

**Cosine schedule** (Improved DDPM):

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)
$$

where $s = 0.008$ is a small offset.

### When to Use VP-SDE

✅ **Use VP-SDE when**:
- You want stable, well-understood behavior
- You're implementing DDPM-style models
- You need closed-form marginals for efficient training
- You want variance to stay bounded

❌ **Consider alternatives when**:
- You need very high noise levels (variance exploding might be better)
- You're working with score-based models from scratch

---

## 2. VE-SDE (Variance-Exploding)

### Definition

$$
dx = \sqrt{\frac{d\sigma^2(t)}{dt}}\,dw(t)
$$

**Key difference**: $f(x,t) = 0$ — **No drift!**

### Why These Specific Functions?

**Drift: $f(x,t) = 0$**

- **Pure diffusion**: No deterministic flow
- **Simpler**: Noise is just added, signal isn't shrunk

**Diffusion: $g(t) = \sqrt{\frac{d\sigma^2(t)}{dt}}$**

- Chosen so that $\text{Var}(x_t) = \sigma^2(t)$
- Variance grows (explodes) over time

### Key Property: Variance Explosion

$$
\mathbb{E}[\|x_t\|^2] = \mathbb{E}[\|x_0\|^2] + \sigma^2(t)
$$

Variance **grows without bound** as $t \to \infty$.

### Closed-Form Marginal

$$
p_t(x_t \mid x_0) = \mathcal{N}(x_t; x_0, \sigma^2(t) I)
$$

**Sampling formula**:

$$
x_t = x_0 + \sigma(t)\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)
$$

**Simpler than VP-SDE**: Signal isn't scaled, just noise is added.

### Common $\sigma(t)$ Schedules

**Geometric schedule**:

$$
\sigma(t) = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{t/T}
$$

Typical values: $\sigma_{\min} = 0.01$, $\sigma_{\max} = 50$, $T = 1$

### When to Use VE-SDE

✅ **Use VE-SDE when**:
- You want very high noise levels at the end
- You're implementing score-based models (NCSN)
- You prefer simpler forward process (no signal scaling)
- You don't mind unbounded variance

❌ **Consider alternatives when**:
- You want bounded variance (use VP-SDE)
- You need to match DDPM exactly
- Numerical stability is a concern at high noise

---

## 3. sub-VP-SDE (Sub-Variance-Preserving)

### Definition

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}\,dw(t)
$$

### Why This Specific Form?

**Drift**: Same as VP-SDE ($-\frac{1}{2}\beta(t) x$)

**Diffusion**: Modified to make variance **decrease** over time

- At $t=0$: $g(0) = 0$ (no noise initially)
- As $t \to \infty$: $g(t) \to \sqrt{\beta(t)}$ (approaches VP-SDE)

### Key Property: Variance Decreasing

$$
\mathbb{E}[\|x_t\|^2] < \mathbb{E}[\|x_0\|^2] \quad \text{for } t > 0
$$

Variance **shrinks** over time, unlike VP (constant) or VE (exploding).

### When to Use sub-VP-SDE

✅ **Use sub-VP-SDE when**:
- You want a compromise between VP and VE
- You need variance to decrease
- You're experimenting with novel schedules

❌ **Consider alternatives when**:
- You want standard, well-tested behavior (use VP-SDE)
- You need maximum simplicity (use VE-SDE)

**Note**: sub-VP-SDE is less commonly used in practice. Most models stick with VP or VE.

---

## Design Considerations: How to Choose

### 1. Mathematical Tractability

**Question**: Do you need closed-form marginals?

- **Yes** → VP-SDE or VE-SDE (both have closed forms)
- **No** → Can use custom SDEs, but training is harder

**Why it matters**: Closed-form marginals let you sample $x_t$ directly from $x_0$ during training, avoiding expensive SDE simulation.

### 2. Variance Behavior

**Question**: How should variance evolve?

- **Constant** → VP-SDE (most stable)
- **Growing** → VE-SDE (higher noise levels)
- **Shrinking** → sub-VP-SDE (less common)

**Why it matters**: Affects numerical stability and the range of noise levels explored.

### 3. Connection to Existing Methods

**Question**: Do you want to match a specific algorithm?

- **DDPM** → VP-SDE with linear $\beta(t)$
- **Score-based models (NCSN)** → VE-SDE
- **DDIM** → Probability flow ODE from VP-SDE

**Why it matters**: Easier to compare results and leverage existing hyperparameters.

### 4. Signal-to-Noise Ratio (SNR)

The **SNR** at time $t$ measures how much signal remains:

**VP-SDE**:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
$$

**VE-SDE**:

$$
\text{SNR}(t) = \frac{\mathbb{E}[\|x_0\|^2]}{\sigma^2(t)}
$$

**Design goal**: SNR should decay smoothly from high (clean) to low (noisy).

---

## Practical Recommendations

### For Most Applications: Use VP-SDE

**Why**:

- ✅ Well-understood and widely used
- ✅ Stable variance behavior
- ✅ Matches DDPM (easy to find good hyperparameters)
- ✅ Closed-form marginals (efficient training)

**Start with**:

- Linear $\beta(t)$: $\beta_{\min} = 0.1$, $\beta_{\max} = 20$
- Or cosine schedule for better performance

### For Score-Based Models: Use VE-SDE

**Why**:

- ✅ Simpler forward process (no signal scaling)
- ✅ Very high noise levels (good for score matching)
- ✅ Matches NCSN framework

**Start with**:

- Geometric $\sigma(t)$: $\sigma_{\min} = 0.01$, $\sigma_{\max} = 50$

### For Custom Applications: Experiment Carefully

**Guidelines**:
1. **Start with VP-SDE** as a baseline
2. **Ensure closed-form marginals** if possible
3. **Check SNR decay**: Should be smooth and monotonic
4. **Verify numerical stability**: Test with different step sizes
5. **Compare to baselines**: Make sure your custom SDE actually helps

---

## Connection to the Reverse SDE

Once you choose $f(x,t)$ and $g(t)$, the **reverse SDE** is determined by Anderson's theorem:

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}(t)
$$

**Key point**: The reverse SDE inherits $f$ and $g$ from your forward choice. You only need to learn the score $\nabla_x \log p_t(x)$.

### Example: VP-SDE Reverse Process

**Forward**:

$$
dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dw(t)
$$

**Reverse**:

$$
dx = \left[-\frac{1}{2}\beta(t) x - \beta(t) s_\theta(x,t)\right]dt + \sqrt{\beta(t)}\,d\bar{w}(t)
$$

Notice:
- Original drift $-\frac{1}{2}\beta(t) x$ appears
- Score correction $-\beta(t) s_\theta(x,t)$ added (note: $g(t)^2 = \beta(t)$)
- Same diffusion coefficient $\sqrt{\beta(t)}$

---

## Summary

### Key Takeaways

1. **$f(x,t)$ and $g(t)$ are design choices**, not learned
2. **Three standard SDEs**:
   - **VP-SDE**: Variance-preserving, most common, matches DDPM
   - **VE-SDE**: Variance-exploding, simpler, used in score-based models
   - **sub-VP-SDE**: Variance-decreasing, less common
3. **Choose based on**:
   - Mathematical tractability (closed-form marginals?)
   - Variance behavior (constant, growing, shrinking?)
   - Connection to existing methods (DDPM, NCSN?)
   - Signal-to-noise ratio decay
4. **Default recommendation**: Start with VP-SDE (linear or cosine schedule)

### What's Fixed vs What's Learned

**Fixed (your design choices)**:

- Forward SDE: $f(x,t)$, $g(t)$
- Noise schedule: $\beta(t)$ or $\sigma(t)$
- Time range: $[0, T]$

**Learned (neural network)**:

- Score function: $s_\theta(x,t) \approx \nabla_x \log p_t(x)$

**Everything else follows**: Once you choose the forward SDE and train the score, the reverse SDE is fully determined.

---

## Further Reading

- **Song et al. (2021)**: [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — Introduces VP-SDE, VE-SDE, and sub-VP-SDE
- **Ho et al. (2020)**: [DDPM](https://arxiv.org/abs/2006.11239) — Original discrete-time formulation (corresponds to VP-SDE)
- **Song & Ermon (2019)**: [NCSN](https://arxiv.org/abs/1907.05600) — Score-based models (corresponds to VE-SDE)
- **Nichol & Dhariwal (2021)**: [Improved DDPM](https://arxiv.org/abs/2102.09672) — Cosine noise schedule

---

## Next Steps

Now that you understand the forward SDE design choices, you can:

1. **See them in action**: [`../02_sde_formulation.ipynb`](../02_sde_formulation.ipynb) implements VP-SDE
2. **Understand training**: [`04_training_loss_and_denoising.md`](./04_training_loss_and_denoising.md) shows how the loss depends on $g(t)$
3. **Understand sampling**: [`05_reverse_sde_and_probability_flow_ode.md`](./05_reverse_sde_and_probability_flow_ode.md) shows how $f$ and $g$ appear in the reverse SDE
